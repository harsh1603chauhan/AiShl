from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_messages() -> None:
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request payload"


def test_chat_rejects_invalid_json() -> None:
    response = client.post("/chat", data="not-json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid request payload"
