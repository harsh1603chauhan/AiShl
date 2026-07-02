from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Iterable

from app.catalog.parser import ParsedAssessment
from app.utils.helpers import normalize_text


@dataclass(slots=True)
class NormalizedAssessment:
    name: str
    description: str
    purpose: str
    skills_measured: list[str] = field(default_factory=list)
    test_type: str = ""
    duration: str = ""
    job_level: str = ""
    remote_testing_support: str = ""
    languages: list[str] = field(default_factory=list)
    url: str = ""
    category: str = "Individual Test Solutions"
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "skills_measured": self.skills_measured,
            "test_type": self.test_type,
            "duration": self.duration,
            "job_level": self.job_level,
            "remote_testing_support": self.remote_testing_support,
            "languages": self.languages,
            "url": self.url,
            "category": self.category,
            "keywords": self.keywords,
        }


DEFAULT_KEYWORDS = ("assessment", "shl", "test")


def record_to_text(source: dict) -> str:
    parts = [
        str(source.get("name", "")),
        str(source.get("description", "")),
        str(source.get("purpose", "")),
        " ".join(source.get("skills_measured", []) or []),
        str(source.get("test_type", "")),
        str(source.get("duration", "")),
        str(source.get("job_level", "")),
        str(source.get("remote_testing_support", "")),
        " ".join(source.get("languages", []) or []),
        str(source.get("category", "")),
        " ".join(source.get("keywords", []) or []),
    ]
    return " ".join(part for part in parts if part)


def normalize_assessment(record: ParsedAssessment | dict) -> NormalizedAssessment:
    if isinstance(record, ParsedAssessment):
        source = record.as_dict()
    else:
        source = dict(record)

    keywords = source.get("keywords") or []
    if not keywords:
        keywords = build_keywords(source)

    return NormalizedAssessment(
        name=source.get("name", "").strip(),
        description=source.get("description", "").strip(),
        purpose=source.get("purpose", "").strip(),
        skills_measured=list(source.get("skills_measured", [])),
        test_type=source.get("test_type", "").strip(),
        duration=source.get("duration", "").strip(),
        job_level=source.get("job_level", "").strip(),
        remote_testing_support=source.get("remote_testing_support", "").strip(),
        languages=list(source.get("languages", [])),
        url=source.get("url", "").strip(),
        category=source.get("category", "Individual Test Solutions").strip(),
        keywords=keywords,
    )


def build_keywords(source: dict) -> list[str]:
    tokens: set[str] = set(DEFAULT_KEYWORDS)
    for field_name in ("name", "description", "purpose", "test_type", "job_level", "category", "duration", "remote_testing_support"):
        for token in normalize_text(str(source.get(field_name, ""))).split():
            if len(token) > 2:
                tokens.add(token)
    for value in source.get("skills_measured", []):
        for token in normalize_text(str(value)).split():
            if len(token) > 2:
                tokens.add(token)
    for value in source.get("languages", []):
        for token in normalize_text(str(value)).split():
            if len(token) > 2:
                tokens.add(token)
    return sorted(tokens)


def save_processed_catalog(records: Iterable[NormalizedAssessment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in records]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_processed_catalog(input_path: Path) -> list[NormalizedAssessment]:
    if not input_path.exists():
        return []
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return [normalize_assessment(item) for item in payload]
