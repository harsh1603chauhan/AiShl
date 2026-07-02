from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.conversation_agent import ConversationAgent
from app.api import chat as chat_module
from app.main import app
from app.models.response import ChatResponse, Recommendation


client = TestClient(app)


def test_chat_endpoint_schema(monkeypatch) -> None:
    def fake_respond(self, messages):
        return ChatResponse(
            reply="Here are the best SHL assessments.",
            recommendations=[Recommendation(name="Java 8", url="https://www.shl.com/java-8", test_type="K")],
            end_of_conversation=True,
        )

    monkeypatch.setattr(ConversationAgent, "respond", fake_respond)
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hiring a Java developer"}]})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"][0]["url"].startswith("https://www.shl.com/")
