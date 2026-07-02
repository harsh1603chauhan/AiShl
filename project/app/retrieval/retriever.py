from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.retrieval.embeddings import get_embedder
from app.retrieval.faiss_store import FaissIndexBundle, build_document_text, load_faiss_index
from app.utils.helpers import normalize_text


logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalHit:
    record: dict[str, Any]
    score: float


class CatalogRetriever:
    def __init__(self, bundle: FaissIndexBundle | None = None) -> None:
        self.bundle = bundle or load_faiss_index()
        self.embedder = get_embedder()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        limit = top_k or settings.top_k_retrieval
        if not self.bundle.metadata:
            return []

        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        hits: list[RetrievalHit] = []

        if self.bundle.index is not None:
            try:
                search_vector = query_vector.reshape(1, -1)
                scores, indices = self.bundle.index.search(search_vector, min(limit, len(self.bundle.metadata)))
                for score, index in zip(scores[0], indices[0], strict=False):
                    if index < 0 or index >= len(self.bundle.metadata):
                        continue
                    hits.append(RetrievalHit(record=self.bundle.metadata[index], score=float(score)))
            except Exception as exc:
                logger.exception("faiss_retrieval_failed: %s", exc)

        if not hits:
            hits = self._fallback_search(query, limit)

        deduped: list[RetrievalHit] = []
        seen: set[str] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            key = normalize_text(f"{hit.record.get('name', '')}|{hit.record.get('url', '')}")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped[:limit]

    def _fallback_search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_tokens = set(normalize_text(query).split())
        scored: list[RetrievalHit] = []
        for record in self.bundle.metadata:
            document = normalize_text(build_document_text(record))
            record_tokens = set(document.split())
            overlap = len(query_tokens & record_tokens)
            lexical_bonus = 0.0
            if record.get("name") and normalize_text(record["name"]) in normalize_text(query):
                lexical_bonus = 2.0
            score = float(overlap) + lexical_bonus
            if score > 0:
                scored.append(RetrievalHit(record=record, score=score))
        if not scored:
            scored = [RetrievalHit(record=record, score=0.0) for record in self.bundle.metadata[:top_k]]
        return scored[:top_k]


_retriever: CatalogRetriever | None = None


def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever()
    return _retriever
