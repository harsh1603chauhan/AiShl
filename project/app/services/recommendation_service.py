from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.llm import NullLLMClient, get_llm_client
from app.core.prompts import COMPARISON_PROMPT, RECOMMENDATION_PROMPT, SYSTEM_PROMPT
from app.core.logger import get_logger
from app.models.response import Recommendation
from app.retrieval.retriever import get_retriever
from app.services.ranking_service import RankingService
from app.utils.helpers import dedupe_recommendations


logger = get_logger(__name__)


class RecommendationService:
    def __init__(self) -> None:
        self.retriever = get_retriever()
        self.ranker = RankingService()
        self.llm = get_llm_client()

    def recommend(self, query: str, limit: int | None = None) -> list[Recommendation]:
        max_items = limit or settings.top_k_recommendations
        hits = self.retriever.retrieve(query, top_k=settings.top_k_retrieval)
        ranked = self.ranker.rank(hits, query=query, limit=max_items)
        recommendations = [self._to_recommendation(record) for record in ranked[:max_items]]
        recommendations = dedupe_recommendations(recommendations)
        logger.info("recommendations_built count=%s query=%s", len(recommendations), query)
        return recommendations

    def generate_recommendation_reply(self, query: str, recommendations: list[Recommendation], is_refinement: bool = False) -> str:
        if not recommendations:
            return "I could not find a grounded SHL shortlist yet. Please add the role, level, or skills to narrow it down."
        if isinstance(self.llm, NullLLMClient):
            if is_refinement:
                return f"I updated the shortlist based on your refinement. Here are {len(recommendations)} SHL assessments grounded in the catalog."
            return f"I found {len(recommendations)} SHL assessments grounded in the catalog."

        catalog_block = self._format_recommendations(recommendations)
        prompt = (
            f"User request: {query}\n\n"
            f"Catalog-backed shortlist:\n{catalog_block}\n\n"
            "Write a short recruiter-friendly reply that explains the shortlist without inventing new assessments."
        )
        reply = self.llm.generate(prompt=prompt, system_prompt=f"{SYSTEM_PROMPT}\n{RECOMMENDATION_PROMPT}")
        return reply.strip() or f"I found {len(recommendations)} SHL assessments grounded in the catalog."

    def generate_clarification_reply(self, questions: list[str]) -> str:
        if not questions:
            return "What role are you hiring for, what seniority level do you need, and which skills or behaviors should the SHL assessment measure?"
        return " ".join(questions[:3])

    def compare(self, first: dict[str, Any], second: dict[str, Any]) -> str:
        first_name = first.get("name", "Unknown assessment")
        second_name = second.get("name", "Unknown assessment")
        first_type = first.get("test_type", "") or "not specified"
        second_type = second.get("test_type", "") or "not specified"
        first_level = first.get("job_level", "") or "not specified"
        second_level = second.get("job_level", "") or "not specified"
        first_skills = ", ".join(first.get("skills_measured", []) or []) or "not specified"
        second_skills = ", ".join(second.get("skills_measured", []) or []) or "not specified"
        return (
            f"{first_name} is a {first_type} assessment targeting {first_skills} at {first_level} level, "
            f"while {second_name} is a {second_type} assessment targeting {second_skills} at {second_level} level."
        )

    def generate_comparison_reply(self, query: str, first: dict[str, Any], second: dict[str, Any]) -> str:
        if isinstance(self.llm, NullLLMClient):
            return self.compare(first, second)
        prompt = (
            f"User request: {query}\n\n"
            f"Assessment A: {first}\n\n"
            f"Assessment B: {second}\n\n"
            "Compare only the provided catalog fields. Do not introduce outside facts."
        )
        reply = self.llm.generate(prompt=prompt, system_prompt=f"{SYSTEM_PROMPT}\n{COMPARISON_PROMPT}")
        return reply.strip() or self.compare(first, second)

    def _format_recommendations(self, recommendations: list[Recommendation]) -> str:
        lines: list[str] = []
        for index, item in enumerate(recommendations, start=1):
            lines.append(f"{index}. {item.name} | {item.url} | {item.test_type}")
        return "\n".join(lines)

    def _to_recommendation(self, record: dict[str, Any]) -> Recommendation:
        return Recommendation(
            name=str(record.get("name", "")).strip(),
            url=str(record.get("url", "")).strip(),
            test_type=str(record.get("test_type", "")).strip() or "Unknown",
        )
