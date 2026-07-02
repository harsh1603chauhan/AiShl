from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from collections.abc import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.utils.helpers import normalize_text


@dataclass(slots=True)
class ParsedAssessment:
    name: str
    description: str
    purpose: str
    skills_measured: list[str]
    test_type: str
    duration: str
    job_level: str
    remote_testing_support: str
    languages: list[str]
    url: str
    category: str
    keywords: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def parse_catalog_html(html: str, source_url: str) -> list[ParsedAssessment]:
    soup = BeautifulSoup(html, "lxml")
    records: list[ParsedAssessment] = []

    for url in extract_assessment_urls(soup, source_url):
        title, description, body_text = extract_page_signals(soup, url)
        candidate = build_record_from_signals(url, title, description, body_text)
        if candidate is not None:
            records.append(candidate)

    if not records:
        card_selectors = [
            "article",
            "li",
            "tr",
            "div[class*='card']",
            "div[class*='item']",
            "div[class*='product']",
            "section",
        ]

        for container in soup.select(",".join(card_selectors)):
            candidate = _parse_container(container, source_url)
            if candidate is not None:
                records.append(candidate)

    if not records:
        for link in soup.select("a[href]"):
            candidate = _parse_link(link, source_url)
            if candidate is not None:
                records.append(candidate)

    return deduplicate_parsed_records(records)


def extract_assessment_urls(soup: BeautifulSoup, source_url: str) -> list[str]:
    urls: set[str] = set()
    for link in soup.select("a[href*='/products/assessments/']"):
        href = link.get("href", "").strip()
        if not href:
            continue
        absolute = urljoin(source_url, href)
        if _looks_like_job_solution(link.get_text(" ", strip=True), href):
            continue
        if _looks_like_catalog_item(link.get_text(" ", strip=True), href):
            urls.add(absolute)
    return sorted(urls)


def extract_page_signals(soup: BeautifulSoup, source_url: str) -> tuple[str, str, str]:
    title = _pick_title(soup)
    if not title:
        title = soup.title.get_text(strip=True) if soup.title else ""
    description = _meta_content(soup, "description") or _meta_content(soup, "og:description")
    body_text = soup.get_text(" ", strip=True)
    return title, description, body_text


def build_record_from_signals(url: str, title: str, description: str, body_text: str) -> ParsedAssessment | None:
    if not title:
        return None
    if _looks_like_job_solution(title, url):
        return None
    if not _looks_like_catalog_item(title, url):
        return None

    combined_text = f"{title} {description} {body_text}"
    return ParsedAssessment(
        name=title.strip(),
        description=description.strip() or _extract_description_text(body_text),
        purpose=_extract_field(combined_text, r"purpose[:\-]\s*([^|•\n]+)"),
        skills_measured=_extract_list_field(combined_text, [r"skills? measured[:\-]\s*([^|•\n]+)", r"skills?[:\-]\s*([^|•\n]+)"]),
        test_type=_extract_field(combined_text, r"test type[:\-]\s*([^|•\n]+)"),
        duration=_extract_field(combined_text, r"duration[:\-]\s*([^|•\n]+)"),
        job_level=_extract_field(combined_text, r"job level[:\-]\s*([^|•\n]+)"),
        remote_testing_support=_extract_field(combined_text, r"remote testing(?: support)?[:\-]\s*([^|•\n]+)"),
        languages=_extract_list_field(combined_text, [r"languages?[:\-]\s*([^|•\n]+)" ]),
        url=url,
        category=_infer_category_from_url(url),
        keywords=_build_keywords_from_text(combined_text),
    )


def _parse_container(container, source_url: str) -> ParsedAssessment | None:
    link = container.select_one("a[href]")
    title = _pick_title(container)
    if not link and not title:
        return None
    href = link.get("href", "").strip() if link else ""
    text = normalize_text(f"{title} {container.get_text(' ', strip=True)}")
    if not title:
        title = (link.get_text(" ", strip=True) if link else "").strip()
    if not title or not href:
        return None
    if _looks_like_job_solution(text, href):
        return None
    if not _looks_like_catalog_item(text, href):
        return None
    return ParsedAssessment(
        name=title.strip(),
        description=_extract_description(container),
        purpose=_extract_field(container.get_text(" ", strip=True), r"purpose[:\-]\s*([^|•\n]+)"),
        skills_measured=_extract_list_field(container.get_text(" ", strip=True), [r"skills? measured[:\-]\s*([^|•\n]+)"] ),
        test_type=_extract_field(container.get_text(" ", strip=True), r"test type[:\-]\s*([^|•\n]+)"),
        duration=_extract_field(container.get_text(" ", strip=True), r"duration[:\-]\s*([^|•\n]+)"),
        job_level=_extract_field(container.get_text(" ", strip=True), r"job level[:\-]\s*([^|•\n]+)"),
        remote_testing_support=_extract_field(container.get_text(" ", strip=True), r"remote testing[:\-]\s*([^|•\n]+)"),
        languages=_extract_list_field(container.get_text(" ", strip=True), [r"languages?[:\-]\s*([^|•\n]+)"]),
        url=urljoin(source_url, href),
        category="Individual Test Solutions",
        keywords=_build_keywords_from_text(container.get_text(" ", strip=True)),
    )


def _parse_link(link, source_url: str) -> ParsedAssessment | None:
    href = link.get("href", "").strip()
    title = link.get_text(" ", strip=True).strip()
    if not href or not title:
        return None
    if _looks_like_job_solution(title, href):
        return None
    if not _looks_like_catalog_item(title, href):
        return None
    return ParsedAssessment(
        name=title,
        description=link.get("title", "").strip(),
        purpose="",
        skills_measured=[],
        test_type="",
        duration="",
        job_level="",
        remote_testing_support="",
        languages=[],
        url=urljoin(source_url, href),
        category="Individual Test Solutions",
        keywords=_build_keywords_from_text(title),
    )


def _pick_title(container) -> str:
    for selector in ("h1", "h2", "h3", "h4", "h5", "h6", ".title", ".name", ".product-title", "strong", "b"):
        node = container.select_one(selector)
        if node:
            value = node.get_text(" ", strip=True).strip()
            if value:
                return value
    return ""


def _extract_description(container) -> str:
    text = container.get_text(" ", strip=True)
    for pattern in (r"description[:\-]\s*([^|•\n]+)", r"summary[:\-]\s*([^|•\n]+)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text[:500].strip()


def _extract_description_text(text: str) -> str:
    lowered = normalize_text(text)
    for marker in ("our assessment", "this assessment", "assessment measures", "what does it measure"):
        if marker in lowered:
            start = lowered.find(marker)
            return text[start : start + 500].strip()
    return text[:500].strip()


def _extract_field(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_list_field(text: str, patterns: list[str]) -> list[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1)
            values = [item.strip() for item in re.split(r",|/|;|\|", raw) if item.strip()]
            return values
    return []


def _build_keywords_from_text(text: str) -> list[str]:
    tokens = {token for token in normalize_text(text).split() if len(token) > 2}
    tokens.update(token for token in normalize_text(text).replace("/", " ").replace("-", " ").split() if len(token) > 2)
    return sorted(tokens)


def _looks_like_job_solution(text: str, href: str) -> bool:
    lowered = normalize_text(text)
    return "job solution" in lowered or "job-solution" in href.lower()


def _looks_like_catalog_item(text: str, href: str) -> bool:
    lowered_text = normalize_text(text)
    lowered_href = href.lower()
    if not lowered_text:
        return False
    if any(token in lowered_text for token in ("assessment", "test", "opq", "ability", "personality", "verbal", "numerical", "logical", "verify", "inductive", "deductive", "simulat", "cognitive", "skill", "questionnaire", "framework")):
        return True
    if "/solutions/products/" in lowered_href or "/product/" in lowered_href:
        return True
    return len(lowered_text.split()) <= 6 and any(char.isalpha() for char in lowered_text)


def _meta_content(soup: BeautifulSoup, key: str) -> str:
    node = soup.select_one(f'meta[name="{key}"], meta[property="{key}"]')
    if node and node.get("content"):
        return node.get("content", "").strip()
    return ""


def _infer_category_from_url(url: str) -> str:
    lowered = url.lower()
    parts = [segment.replace("-", " ") for segment in lowered.split("/") if segment]
    for part in parts[::-1]:
        if part in {"personality assessment", "behavioral assessments", "cognitive assessments", "job focused assessments", "skills and simulations", "assessment and development centers"}:
            return part.title()
    return "Individual Test Solutions"


def deduplicate_parsed_records(records: list[ParsedAssessment]) -> list[ParsedAssessment]:
    seen: set[str] = set()
    deduped: list[ParsedAssessment] = []
    for record in records:
        key = normalize_text(f"{record.name}|{record.url}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped
