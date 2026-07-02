from __future__ import annotations

from app.utils.helpers import build_clarification_questions, extract_candidate_names, has_enough_context


def test_extract_candidate_names_from_comparison_query() -> None:
    assert extract_candidate_names("Compare OPQ32r and GSA") == ["opq32r", "gsa"]


def test_clarification_questions_are_targeted() -> None:
    questions = build_clarification_questions("I need an assessment")
    assert any("role" in question.lower() for question in questions)
    assert any("experience" in question.lower() for question in questions)


def test_context_sufficiency_tracks_multiple_facets() -> None:
    assert has_enough_context("Hiring a mid-level Java developer with stakeholder management")
    assert not has_enough_context("assessment")
