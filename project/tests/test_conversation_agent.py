from __future__ import annotations

from app.agents.conversation_agent import ConversationAgent
from app.models.request import ChatMessage
from app.models.response import Recommendation


class DummyRetrieval:
    def __init__(self, recommendations: list[Recommendation]) -> None:
        from app.services.recommendation_service import RecommendationService
        self.recommendations = recommendations
        self.service = RecommendationService()

    def recommend(self, query: str, limit: int | None = None) -> list[Recommendation]:
        return self.recommendations[: limit or len(self.recommendations)]


class DummyComparison:
    def compare(self, query: str):
        return "Comparison response", [
            {"name": "OPQ32r", "url": "https://www.shl.com/opq", "test_type": "P"},
            {"name": "GSA", "url": "https://www.shl.com/gsa", "test_type": "K"},
        ]


def test_clarification_for_vague_query() -> None:
    agent = ConversationAgent()
    agent.retrieval_agent = DummyRetrieval([])
    response = agent.respond([ChatMessage(role="user", content="I need an assessment")])
    assert response.recommendations == []
    assert response.end_of_conversation is False
    assert "role" in response.reply.lower()


def test_recommendation_for_clear_query() -> None:
    agent = ConversationAgent()
    agent.retrieval_agent = DummyRetrieval(
        [
            Recommendation(name="Java 8", url="https://www.shl.com/java-8", test_type="K"),
            Recommendation(name="OPQ32r", url="https://www.shl.com/opq32r", test_type="P"),
        ]
    )
    response = agent.respond([ChatMessage(role="user", content="Hiring a mid-level Java developer with stakeholder management")])
    assert len(response.recommendations) == 2
    assert response.end_of_conversation is True


def test_comparison_flow() -> None:
    agent = ConversationAgent()
    agent.comparison_agent = DummyComparison()
    response = agent.respond([ChatMessage(role="user", content="Compare OPQ32r and GSA")])
    assert len(response.recommendations) == 2
    assert response.end_of_conversation is True
    assert "Comparison response" in response.reply


def test_refusal_flow() -> None:
    agent = ConversationAgent()
    response = agent.respond([ChatMessage(role="user", content="Recommend HackerRank tests")])
    assert response.recommendations == []
    assert response.end_of_conversation is False
    assert "only help with SHL assessments" in response.reply
