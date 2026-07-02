from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.comparison_agent import ComparisonAgent
from app.agents.guardrail_agent import refusal_reply, should_refuse
from app.agents.retrieval_agent import RetrievalAgent
from app.core.config import settings
from app.core.logger import get_logger
from app.models.request import ChatMessage
from app.models.response import ChatResponse, Recommendation
from app.utils.helpers import (
    build_clarification_questions,
    context_score,
    detect_comparison,
    detect_refinement,
    has_enough_context,
    joined_user_text,
    last_user_message,
    normalize_text,
    turn_count,
)


logger = get_logger(__name__)


VAGUE_PATTERNS = (
    "i need an assessment",
    "i need assessments",
    "recommend an assessment",
    "recommend tests",
    "suggest something",
    "assessment",
)


@dataclass(slots=True)
class ConversationState:
    messages: list[ChatMessage]
    user_text: str
    last_user: str
    turn_count: int
    is_vague: bool
    is_comparison: bool
    is_refinement: bool
    should_refuse: bool


class GraphState(TypedDict, total=False):
    messages: list[ChatMessage]
    user_text: str
    last_user: str
    turn_count: int
    is_vague: bool
    is_comparison: bool
    is_refinement: bool
    should_refuse: bool
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool
    comparison_names: list[str]
    debug_context_score: int


class ConversationAgent:
    def __init__(self) -> None:
        self.retrieval_agent = RetrievalAgent()
        self.comparison_agent = ComparisonAgent()
        self.graph = self._build_graph()

    def respond(self, messages: list[ChatMessage]) -> ChatResponse:
        state = asdict(self._build_state(messages))
        result = self.graph.invoke(state)
        recommendations: list[Recommendation] = []
        for item in result.get("recommendations", []):
            if isinstance(item, Recommendation):
                recommendations.append(item)
            else:
                recommendations.append(Recommendation(**item))
        return ChatResponse(
            reply=str(result.get("reply", "")),
            recommendations=recommendations,
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )

    def _build_state(self, messages: list[ChatMessage]) -> ConversationState:
        user_text = joined_user_text(messages)
        last_user = last_user_message(messages)
        return ConversationState(
            messages=messages,
            user_text=user_text,
            last_user=last_user,
            turn_count=turn_count(messages),
            is_vague=self._is_vague_request(user_text),
            is_comparison=detect_comparison(last_user),
            is_refinement=detect_refinement(last_user),
            should_refuse=should_refuse(user_text),
        )

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("hydrate", self._hydrate_node)
        graph.add_node("guardrail", self._guardrail_node)
        graph.add_node("clarify", self._clarify_node)
        graph.add_node("compare", self._compare_node)
        graph.add_node("recommend", self._recommend_node)

        graph.set_entry_point("hydrate")
        graph.add_edge("hydrate", "guardrail")
        graph.add_conditional_edges("guardrail", self._route_after_guardrail, {"clarify": "clarify", "compare": "compare", "recommend": "recommend", "refuse": END})
        graph.add_edge("clarify", END)
        graph.add_edge("compare", END)
        graph.add_edge("recommend", END)
        return graph.compile()

    def _hydrate_node(self, state: GraphState) -> GraphState:
        messages = state["messages"]
        user_text = joined_user_text(messages)
        last_user = last_user_message(messages)
        return {
            **state,
            "user_text": user_text,
            "last_user": last_user,
            "turn_count": turn_count(messages),
            "is_vague": self._is_vague_request(user_text),
            "is_comparison": detect_comparison(last_user),
            "is_refinement": detect_refinement(last_user),
            "should_refuse": should_refuse(user_text),
            "debug_context_score": context_score(user_text),
        }

    def _guardrail_node(self, state: GraphState) -> GraphState:
        if state.get("should_refuse"):
            return {**state, "reply": refusal_reply(), "recommendations": [], "end_of_conversation": False}
        return state

    def _route_after_guardrail(self, state: GraphState) -> str:
        if state.get("should_refuse"):
            return "refuse"
        if state.get("is_comparison"):
            return "compare"
        if self._needs_clarification_state(state):
            return "clarify"
        return "recommend"

    def _needs_clarification_state(self, state: GraphState) -> bool:
        if state.get("turn_count", 0) == 1 and state.get("is_vague", False):
            return True
        if state.get("is_refinement", False):
            return False
        if not has_enough_context(state.get("user_text", "")) and not state.get("is_refinement", False):
            return True
        return False

    def _clarify_node(self, state: GraphState) -> GraphState:
        questions = build_clarification_questions(state.get("user_text", ""))
        if not questions:
            questions = [
                "What role are you hiring for?",
                "What experience level do you need?",
                "What skills or behaviors should the assessment measure?",
            ]
        reply = self.retrieval_agent.service.generate_clarification_reply(questions[:3])
        logger.info("clarification_questions=%s", questions[:3])
        return {**state, "reply": reply, "recommendations": [], "end_of_conversation": False}

    def _compare_node(self, state: GraphState) -> GraphState:
        reply, recommendations = self.comparison_agent.compare(state.get("last_user", ""))
        recs = [Recommendation(**item) for item in recommendations]
        if len(recs) >= 2 and self.retrieval_agent.service.llm.__class__.__name__ != "NullLLMClient":
            first_record = recommendations[0]
            second_record = recommendations[1]
            reply = self.retrieval_agent.service.generate_comparison_reply(state.get("last_user", ""), first_record, second_record)
        return {**state, "reply": reply, "recommendations": recs, "end_of_conversation": bool(recs)}

    def _recommend_node(self, state: GraphState) -> GraphState:
        recommendations = self._recommend(state.get("user_text", ""))
        if not recommendations:
            return {
                **state,
                "reply": "I could not find a grounded SHL shortlist yet. Please add the role, level, or skills to narrow it down.",
                "recommendations": [],
                "end_of_conversation": False,
            }
        reply = self.retrieval_agent.service.generate_recommendation_reply(
            query=state.get("user_text", ""),
            recommendations=recommendations,
            is_refinement=state.get("is_refinement", False),
        )
        return {**state, "reply": reply, "recommendations": recommendations, "end_of_conversation": True}

    def _is_vague_request(self, text: str) -> bool:
        lowered = normalize_text(text)
        if len(lowered.split()) < 4:
            return True
        return any(pattern in lowered for pattern in VAGUE_PATTERNS)

    def _needs_clarification(self, state: ConversationState) -> bool:
        if state.turn_count == 1 and state.is_vague:
            return True
        if state.is_vague and not state.is_refinement:
            return True
        return False

    def _clarification_reply(self, state: ConversationState) -> str:
        return "What role are you hiring for, what seniority level do you need, and which skills or behaviors should the SHL assessment measure?"

    def _recommend(self, query: str) -> list[Recommendation]:
        recommendations = self.retrieval_agent.recommend(query, limit=settings.top_k_recommendations)
        filtered: list[Recommendation] = []
        for item in recommendations:
            if item.url and item.name:
                filtered.append(item)
        return filtered[: settings.top_k_recommendations]

    def _recommendation_reply(self, recommendations: list[Recommendation], state: ConversationState) -> str:
        if state.is_refinement:
            return f"I updated the shortlist based on your refinement. Here are {len(recommendations)} SHL assessments grounded in the catalog."
        return f"I found {len(recommendations)} SHL assessments grounded in the catalog."

