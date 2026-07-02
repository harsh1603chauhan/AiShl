from __future__ import annotations

from pathlib import Path

import requests

from app.catalog.parser import parse_catalog_html
from app.catalog.preprocess import NormalizedAssessment, normalize_assessment, save_processed_catalog
from app.core.config import settings
from app.core.logger import get_logger


logger = get_logger(__name__)


def fetch_catalog_html(url: str | None = None, timeout_seconds: int | None = None, user_agent: str | None = None) -> str:
    target_url = url or settings.shl_catalog_url
    timeout = timeout_seconds or settings.http_timeout_seconds
    headers = {"User-Agent": user_agent or settings.user_agent}
    response = requests.get(target_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def scrape_catalog(url: str | None = None) -> list[NormalizedAssessment]:
    html = fetch_catalog_html(url=url)
    settings.raw_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    settings.raw_catalog_path.write_text(html, encoding="utf-8")
    parsed = parse_catalog_html(html, source_url=url or settings.shl_catalog_url)
    normalized = [normalize_assessment(record) for record in parsed]
    logger.info("scraped_catalog_records=%s", len(normalized))
    return normalized


def build_and_save_catalog(output_path: Path | None = None) -> list[NormalizedAssessment]:
    normalized = scrape_catalog()
    destination = output_path or settings.processed_catalog_path
    save_processed_catalog(normalized, destination)
    return normalized
