from __future__ import annotations

from app.utils.helpers import detect_injection


OUT_OF_SCOPE_KEYWORDS = (
    "hackerrank",
    "leetcode",
    "coderbyte",
    "legal advice",
    "lawyer",
    "employment law",
    "salary negotiation",
    "competitor",
)


def is_prompt_injection(text: str) -> bool:
    return detect_injection(text)


def is_out_of_scope(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in OUT_OF_SCOPE_KEYWORDS)


def should_refuse(text: str) -> bool:
    return is_prompt_injection(text) or is_out_of_scope(text)


def refusal_reply() -> str:
    return "I can only help with SHL assessments from the Individual Test Solutions catalog. Please ask about SHL assessments, comparison, or recommendation needs."
