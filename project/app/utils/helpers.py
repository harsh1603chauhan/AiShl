from __future__ import annotations

import re
from typing import Iterable

from app.models.request import ChatMessage
from app.models.response import Recommendation


INJECTION_PATTERNS = (
    r"ignore (all|previous) instructions",
    r"reveal (the )?(system prompt|prompt)",
    r"developer message",
    r"act as",
    r"jailbreak",
    r"prompt injection",
    r"override (the )?(rules|instructions)",
)

COMPARISON_PATTERNS = (r"compare", r"difference between", r"vs\.?", r"versus")
REFINEMENT_PATTERNS = (r"actually", r"instead", r"add ", r"remove ", r"include ", r"exclude ", r"also ")

ROLE_HINTS = (
    "developer",
    "engineer",
    "analyst",
    "manager",
    "consultant",
    "specialist",
    "tester",
    "sales",
    "finance",
    "hr",
    "recruiter",
)

SENIORITY_HINTS = (
    "junior",
    "entry",
    "graduate",
    "mid",
    "mid-level",
    "senior",
    "lead",
    "principal",
    "experienced",
)

SKILL_HINTS = (
    "java",
    "python",
    "javascript",
    "typescript",
    "sql",
    "communication",
    "stakeholder",
    "leadership",
    "personality",
    "numerical",
    "verbal",
    "logical",
    "aptitude",
    "problem solving",
    "critical thinking",
    "excel",
    "sales",
    "customer service",
)

ASSESSMENT_TYPE_HINTS = (
    "personality",
    "ability",
    "aptitude",
    "knowledge",
    "cognitive",
    "behavioral",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def joined_user_text(messages: list[ChatMessage]) -> str:
    return " \n".join(message.content for message in messages if message.role == "user")


def detect_injection(text: str) -> bool:
    lowered = normalize_text(text)
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def detect_comparison(text: str) -> bool:
    lowered = normalize_text(text)
    return any(re.search(pattern, lowered) for pattern in COMPARISON_PATTERNS)


def detect_refinement(text: str) -> bool:
    lowered = normalize_text(text)
    return any(re.search(pattern, lowered) for pattern in REFINEMENT_PATTERNS)


def extract_candidate_names(text: str) -> list[str]:
    lowered = normalize_text(text)
    if not lowered:
        return []

    lowered = re.sub(r"^(compare|compare the|difference between|what is the difference between|difference in|difference for)\s+", "", lowered)

    separators = (" versus ", " vs ", " compared to ", " and ")
    for separator in separators:
        if separator in lowered:
            left, right = lowered.split(separator, maxsplit=1)
            left = re.sub(r"^(between|among|the)\s+", "", left).strip(" ?,.")
            right = re.sub(r"^(between|among|the)\s+", "", right).strip(" ?,.")
            return [left, right]

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    names = [first or second for first, second in quoted if (first or second)]
    if names:
        return [normalize_text(name) for name in names]

    return []


def extract_facets(text: str) -> dict[str, list[str]]:
    lowered = normalize_text(text)
    facets = {
        "roles": [],
        "seniority": [],
        "skills": [],
        "assessment_types": [],
    }

    for hint in ROLE_HINTS:
        if hint in lowered:
            facets["roles"].append(hint)

    for hint in SENIORITY_HINTS:
        if hint in lowered:
            facets["seniority"].append(hint)

    for hint in SKILL_HINTS:
        if hint in lowered:
            facets["skills"].append(hint)

    for hint in ASSESSMENT_TYPE_HINTS:
        if hint in lowered:
            facets["assessment_types"].append(hint)

    return facets


def context_score(text: str) -> int:
    facets = extract_facets(text)
    return sum(1 for values in facets.values() if values)


def has_enough_context(text: str) -> bool:
    lowered = normalize_text(text)
    if len(lowered.split()) >= 10:
        return context_score(text) >= 2
    return context_score(text) >= 3


def build_clarification_questions(text: str) -> list[str]:
    facets = extract_facets(text)
    questions: list[str] = []
    if not facets["roles"]:
        questions.append("What role are you hiring for?")
    if not facets["seniority"]:
        questions.append("What seniority or experience level do you need?")
    if not facets["skills"]:
        questions.append("What skills or behaviors should the assessment measure? Please add them.")
    if not facets["assessment_types"]:
        questions.append("Do you want a knowledge, ability, or personality assessment?")
    return questions[:3]


def extract_comparison_query(text: str) -> tuple[str, str]:
    names = extract_candidate_names(text)
    if len(names) >= 2:
        return names[0], names[1]
    return "", ""


def dedupe_recommendations(items: Iterable[Recommendation]) -> list[Recommendation]:
    seen: set[str] = set()
    deduped: list[Recommendation] = []
    for item in items:
        key = normalize_text(f"{item.name}|{item.url}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def turn_count(messages: list[ChatMessage]) -> int:
    return sum(1 for message in messages if message.role == "user")
