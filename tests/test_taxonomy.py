"""Tests for the taxonomy port.

The value of these rules is fidelity to AgentRx's scorer, so the cases that
matter most are the ones drawn from the benchmark's own inconsistent naming.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe.rca.taxonomy import (
    CATEGORY_TO_FAILURE_CASE,
    FAILURE_CASE_TO_CATEGORY,
    category_to_case,
    extract_failure_case,
    normalize_category,
    taxonomy_checklist,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_taxonomy_has_ten_categories():
    """Ten, not nine -- the tenth is the Inconclusive fallback."""
    assert len(FAILURE_CASE_TO_CATEGORY) == 10
    assert FAILURE_CASE_TO_CATEGORY[10] == "Inconclusive"
    assert set(FAILURE_CASE_TO_CATEGORY) == set(range(1, 11))
    assert len(CATEGORY_TO_FAILURE_CASE) == 10


@pytest.mark.parametrize(
    ("raw", "expected_case"),
    [
        # The cross-domain split that makes this port necessary: tau and
        # Magentic-One spell category 1 differently.
        ("Instruction Adherence Failure", 1),
        ("Instruction/Plan Adherence Failure", 1),
        # Magentic-One's casing differs from tau's for these too.
        ("Invention of new information", 2),
        ("Invention of New Information", 2),
        ("Intent not supported", 7),
        ("Intent Not Supported", 7),
        ("Misinterpretation of Tool Output", 4),
        ("Intent Plan Misalignment", 5),
        ("Underspecified User Intent", 6),
        ("Invalid Invocation", 3),
        ("Guardrails Triggered", 8),
        ("System Failure", 9),
        ("Inconclusive", 10),
    ],
)
def test_real_category_names_from_both_domains(raw, expected_case):
    assert category_to_case(raw) == expected_case


def test_every_ground_truth_category_resolves():
    """No category in the published ground truth may fall through to Inconclusive."""
    seen = set()
    for name in ("tau_ground_truth_slice.json", "magentic_ground_truth_slice.json"):
        for entry in json.loads((FIXTURES / name).read_text()):
            for failure in entry["failures"]:
                seen.add(failure["failure_category"])
    assert seen
    for category in seen:
        assert category_to_case(category) != 10, f"{category!r} fell through to Inconclusive"


def test_handoff_maps_to_tool_output_misinterpretation():
    """An upstream quirk, preserved deliberately."""
    assert normalize_category("Handoff error") == "Misinterpretation of Tool Output"


def test_rule_order_lets_intent_not_supported_through():
    """It only resolves because the plan/misalignment and underspecified rules miss it."""
    assert normalize_category("intent not supported") == "Intent Not Supported"


def test_unknown_category_is_returned_unchanged_then_scored_inconclusive():
    assert normalize_category("Cosmic Ray") == "Cosmic Ray"
    assert category_to_case("Cosmic Ray") == 10


def test_empty_category():
    assert normalize_category("") == "Unknown"
    assert normalize_category(None) == "Unknown"
    assert category_to_case(None) == 10


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, 4),
        ("4", 4),
        ("FailureCase.INVENTION_OF_NEW_INFORMATION", 2),
        ("INSTRUCTION_PLAN_ADHERENCE_FAILURE", 1),
        ("instruction-plan-adherence-failure", 1),
        (0, 10),
        (11, 10),
        ("99", 10),
        ("nonsense", 10),
        (True, 10),
    ],
)
def test_extract_failure_case(value, expected):
    assert extract_failure_case(value) == expected


def test_checklist_lists_all_ten_with_descriptions():
    text = taxonomy_checklist()
    for case, name in FAILURE_CASE_TO_CATEGORY.items():
        assert f"{case}. {name}" in text
    assert len(text.splitlines()) == 10
