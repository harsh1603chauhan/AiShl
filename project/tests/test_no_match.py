from __future__ import annotations

from app.agents.conversation_agent import ConversationAgent
from app.models.request import ChatMessage
from app.models.response import Recommendation


class EmptyRetrieval:
    def __init__(self) -> None:
        from app.services.recommendation_service import RecommendationService
        self.service = RecommendationService()

    def recommend(self, query: str, limit: int | None = None) -> list[Recommendation]:
        return []


def test_no_match_returns_clarification_style_response() -> None:
    agent = ConversationAgent()
    agent.retrieval_agent = EmptyRetrieval()
    response = agent.respond([ChatMessage(role="user", content="Hiring a rare role with unknown skills")])
    assert response.recommendations == []
    assert response.end_of_conversation is False
    assert "narrow" in response.reply.lower() or "add" in response.reply.lower()
