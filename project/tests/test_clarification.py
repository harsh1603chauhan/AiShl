from __future__ import annotations

from app.agents.conversation_agent import ConversationAgent
from app.models.request import ChatMessage


def test_clarification_asks_followup_questions() -> None:
    agent = ConversationAgent()
    response = agent.respond([ChatMessage(role="user", content="I need an assessment")])
    assert response.recommendations == []
    assert "role" in response.reply.lower()
    assert "seniority" in response.reply.lower()
