from __future__ import annotations

from app.models.response import ChatResponse, Recommendation


def test_response_schema() -> None:
    response = ChatResponse(reply="Hi", recommendations=[Recommendation(name="A", url="https://x", test_type="K")], end_of_conversation=False)
    payload = response.model_dump()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"][0]["name"] == "A"
