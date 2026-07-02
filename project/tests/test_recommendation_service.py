from __future__ import annotations

from app.models.response import Recommendation
from app.services.recommendation_service import RecommendationService


class FakeRetriever:
    def retrieve(self, query: str, top_k: int | None = None):
        return []


def test_recommendation_schema() -> None:
    service = RecommendationService()
    service.retriever = FakeRetriever()
    recommendations = service.recommend("Java developer")
    assert isinstance(recommendations, list)
    assert recommendations == []
