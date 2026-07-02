from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.retrieval.embeddings import get_embedder
from app.catalog.preprocess import record_to_text


logger = get_logger(__name__)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    faiss = None


@dataclass(slots=True)
class FaissIndexBundle:
    index: Any
    metadata: list[dict[str, Any]]


def index_path() -> Path:
    return settings.faiss_index_path


def meta_path() -> Path:
    return settings.faiss_meta_path


def build_faiss_index(records: list[dict[str, Any]], index_file: Path | None = None, meta_file: Path | None = None) -> FaissIndexBundle:
    target_index_path = index_file or index_path()
    target_meta_path = meta_file or meta_path()
    target_index_path.parent.mkdir(parents=True, exist_ok=True)
    target_meta_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        target_meta_path.write_text(json.dumps([], indent=2, ensure_ascii=True), encoding="utf-8")
        if faiss is not None:
            dim = 384
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, str(target_index_path))
        logger.warning("faiss_index_built_with_no_records")
        return FaissIndexBundle(index=None, metadata=[])

    embedder = get_embedder()
    documents = [record_to_text(record) for record in records]
    vectors = np.asarray(embedder.embed_documents(documents), dtype=np.float32)
    if vectors.ndim != 2:
        vectors = vectors.reshape(len(records), -1)

    if faiss is None:
        target_meta_path.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.warning("faiss_unavailable_using_metadata_only_index")
        return FaissIndexBundle(index=None, metadata=records)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(target_index_path))
    target_meta_path.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("faiss_index_built count=%s dim=%s", len(records), dim)
    return FaissIndexBundle(index=index, metadata=records)


def load_faiss_index(index_file: Path | None = None, meta_file: Path | None = None) -> FaissIndexBundle:
    target_index_path = index_file or index_path()
    target_meta_path = meta_file or meta_path()

    metadata: list[dict[str, Any]] = []
    if target_meta_path.exists():
        metadata = json.loads(target_meta_path.read_text(encoding="utf-8"))

    if faiss is None or not target_index_path.exists():
        return FaissIndexBundle(index=None, metadata=metadata)

    index = faiss.read_index(str(target_index_path))
    return FaissIndexBundle(index=index, metadata=metadata)


def build_document_text(record: dict[str, Any]) -> str:
    return record_to_text(record)
