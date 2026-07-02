from __future__ import annotations

from typing import Any

from app.retrieval.retriever import get_retriever
from app.services.recommendation_service import RecommendationService
from app.utils.helpers import extract_candidate_names, normalize_text


class ComparisonAgent:
    def __init__(self) -> None:
        self.retriever = get_retriever()
        self.recommendation_service = RecommendationService()

    def compare(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        names = extract_candidate_names(query)
        if len(names) < 2:
            hits = self.retriever.retrieve(query, top_k=2)
            names = [hit.record.get("name", "") for hit in hits[:2]]
        matched_records = self._retrieve_records(names[:2])
        if len(matched_records) < 2:
            return ("I could not find enough SHL catalog items to compare yet.", [])
        reply = self.recommendation_service.compare(matched_records[0], matched_records[1])
        recommendations = [
            {
                "name": str(record.get("name", "")).strip(),
                "url": str(record.get("url", "")).strip(),
                "test_type": str(record.get("test_type", "")).strip() or "Unknown",
            }
            for record in matched_records
        ]
        return reply, recommendations

    def _retrieve_records(self, names: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for name in names:
            hits = self.retriever.retrieve(name, top_k=3)
            for hit in hits:
                record_name = normalize_text(str(hit.record.get("name", "")))
                if normalize_text(name) in record_name or record_name in normalize_text(name):
                    results.append(hit.record)
                    break
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in results:
            key = normalize_text(f"{record.get('name', '')}|{record.get('url', '')}")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped[:2]
