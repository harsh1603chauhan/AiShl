from __future__ import annotations

from app.agents.comparison_agent import ComparisonAgent


class FakeRetriever:
    def retrieve(self, query: str, top_k: int | None = None):
        class Hit:
            def __init__(self, record):
                self.record = record

        if "opq" in query.lower():
            return [Hit({"name": "OPQ32r", "url": "https://www.shl.com/opq", "test_type": "P", "skills_measured": ["personality"], "job_level": "mid"})]
        return [Hit({"name": "GSA", "url": "https://www.shl.com/gsa", "test_type": "K", "skills_measured": ["ability"], "job_level": "mid"})]


def test_comparison_agent_returns_grounded_reply(monkeypatch) -> None:
    agent = ComparisonAgent()
    agent.retriever = FakeRetriever()
    reply, recommendations = agent.compare("Compare OPQ32r and GSA")
    assert "OPQ32r" in reply or "GSA" in reply
    assert len(recommendations) <= 2
