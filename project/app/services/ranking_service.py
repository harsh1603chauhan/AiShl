from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.retrieval.retriever import RetrievalHit
from app.utils.helpers import normalize_text


class RankingService:
    def rank(self, hits: Iterable[RetrievalHit], query: str, limit: int) -> list[dict[str, Any]]:
        query_tokens = set(normalize_text(query).split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for hit in hits:
            record = hit.record
            text = normalize_text(
                " ".join(
                    [
                        str(record.get("name", "")),
                        str(record.get("description", "")),
                        str(record.get("purpose", "")),
                        str(record.get("test_type", "")),
                        str(record.get("job_level", "")),
                        " ".join(record.get("skills_measured", []) or []),
                        " ".join(record.get("keywords", []) or []),
                    ]
                )
            )
            record_tokens = set(text.split())
            overlap = len(query_tokens & record_tokens)
            name_boost = 2.0 if any(token in normalize_text(str(record.get("name", ""))) for token in query_tokens) else 0.0
            score = hit.score + overlap + name_boost
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, record in scored:
            key = normalize_text(f"{record.get('name', '')}|{record.get('url', '')}")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
            if len(deduped) >= limit:
                break
        return deduped
