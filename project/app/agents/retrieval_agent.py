from __future__ import annotations

from app.services.recommendation_service import RecommendationService


class RetrievalAgent:
    def __init__(self) -> None:
        self.service = RecommendationService()

    def recommend(self, query: str, limit: int | None = None):
        return self.service.recommend(query=query, limit=limit)
