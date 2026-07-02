from __future__ import annotations

from app.catalog.scraper import build_and_save_catalog
from app.core.config import settings
from app.core.logger import configure_logging, get_logger
from app.retrieval.faiss_store import build_faiss_index


logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    records = build_and_save_catalog()
    build_faiss_index([record.to_dict() for record in records], settings.faiss_index_path, settings.faiss_meta_path)
    logger.info("catalog_build_complete records=%s", len(records))


if __name__ == "__main__":
    main()
