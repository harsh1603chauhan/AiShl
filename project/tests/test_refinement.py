from __future__ import annotations

from app.agents.conversation_agent import ConversationAgent
from app.models.request import ChatMessage
from app.models.response import Recommendation


class DummyRetrieval:
    def __init__(self) -> None:
        from app.services.recommendation_service import RecommendationService
        self.service = RecommendationService()

    def recommend(self, query: str, limit: int | None = None):
        return [Recommendation(name="OPQ32r", url="https://www.shl.com/opq32r", test_type="P")]


def test_refinement_updates_shortlist() -> None:
    agent = ConversationAgent()
    agent.retrieval_agent = DummyRetrieval()
    messages = [
        ChatMessage(role="user", content="Hiring a leader"),
        ChatMessage(role="assistant", content="What skills matter most?"),
        ChatMessage(role="user", content="Actually add personality tests"),
    ]
    response = agent.respond(messages)
    assert response.recommendations
    assert response.end_of_conversation is True
