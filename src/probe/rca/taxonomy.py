"""The failure taxonomy, and the alias rules needed to score against AgentRx.

There are **ten** categories, not nine. The tenth, `Inconclusive`, is the bucket
for anything that does not match a known category -- both in AgentRx's own
scoring code and here.

The normalization rules below are a deliberate, faithful port of
`normalize_category()` and `extract_failure_case_number()` in AgentRx's
`agentrx/reports/analyze_metrics.py`. They exist because category names are not
consistent across the benchmark's own domains: tau-retail annotations say
"Instruction Adherence Failure" while Magentic-One says "Instruction/Plan
Adherence Failure", and both must resolve to case 1. If we normalized
differently from them, our attribution numbers would not be comparable to
theirs -- which is the entire point of the benchmark -- so the quirks are
reproduced exactly, including rule ordering, rather than cleaned up.
"""

from __future__ import annotations

from enum import IntEnum

INCONCLUSIVE_CASE = 10


class FailureCase(IntEnum):
    """Canonical category ids. The numbering is AgentRx's and is load-bearing."""

    INSTRUCTION_ADHERENCE_FAILURE = 1
    INVENTION_OF_NEW_INFORMATION = 2
    INVALID_INVOCATION = 3
    MISINTERPRETATION_OF_TOOL_OUTPUT = 4
    INTENT_PLAN_MISALIGNMENT = 5
    UNDERSPECIFIED_USER_INTENT = 6
    INTENT_NOT_SUPPORTED = 7
    GUARDRAILS_TRIGGERED = 8
    SYSTEM_FAILURE = 9
    INCONCLUSIVE = 10


FAILURE_CASE_TO_CATEGORY: dict[int, str] = {
    1: "Instruction Adherence Failure",
    2: "Invention of New Information",
    3: "Invalid Invocation",
    4: "Misinterpretation of Tool Output",
    5: "Intent Plan Misalignment",
    6: "Underspecified User Intent",
    7: "Intent Not Supported",
    8: "Guardrails Triggered",
    9: "System Failure",
    10: "Inconclusive",
}

CATEGORY_TO_FAILURE_CASE: dict[str, int] = {v: k for k, v in FAILURE_CASE_TO_CATEGORY.items()}

# Guidance shown to the judge. Kept next to the taxonomy so the prompt and the
# scorer can never drift apart.
CATEGORY_DESCRIPTIONS: dict[int, str] = {
    1: "The agent did not follow an explicit instruction, domain policy, or its own stated plan.",
    2: "The agent asserted a fact, value, or identifier that was never provided or returned.",
    3: "The agent called a tool that does not exist, or with missing/malformed/wrong arguments.",
    4: "The agent misread a tool result -- miscounted, misparsed, or drew a wrong conclusion.",
    5: "The agent's plan does not serve the user's actual intent, even if executed faithfully.",
    6: "The user's request was ambiguous and the agent proceeded without resolving it.",
    7: "The request is outside what the available tools or policy can accomplish.",
    8: "A safety, permission, or policy guardrail blocked progress.",
    9: "Infrastructure-level breakage: crash, timeout, unavailable service, truncated context.",
    10: "No category is adequately supported by the evidence.",
}

_ENUM_NAME_TO_CASE: dict[str, int] = {
    "INSTRUCTION_ADHERENCE_FAILURE": 1,
    "INSTRUCTION_PLAN_ADHERENCE_FAILURE": 1,
    "INVENTION_OF_NEW_INFORMATION": 2,
    "INVALID_INVOCATION": 3,
    "MISINTERPRETATION_OF_TOOL_OUTPUT": 4,
    "INTENT_PLAN_MISALIGNMENT": 5,
    "UNDERSPECIFIED_USER_INTENT": 6,
    "INTENT_NOT_SUPPORTED": 7,
    "GUARDRAILS_TRIGGERED": 8,
    "SYSTEM_FAILURE": 9,
    "INCONCLUSIVE": 10,
}


def normalize_category(category: str | None) -> str:
    """Resolve a free-form category string to a canonical category name.

    Port of AgentRx's `normalize_category`. Rule order matters and is preserved:
    "Intent Not Supported" only resolves correctly because the plan/misalignment
    and underspecified rules are tested first and both fall through for it.

    An unrecognized non-empty string is returned stripped but unchanged, exactly
    as upstream does -- it will then fail an equality check and, via
    `category_to_case`, land in `Inconclusive`.
    """
    if not category:
        return "Unknown"

    cat_lower = category.strip().lower()

    if "instruction" in cat_lower and "adherence" in cat_lower:
        return "Instruction Adherence Failure"
    elif "invention" in cat_lower or ("new" in cat_lower and "information" in cat_lower):
        return "Invention of New Information"
    elif "invalid" in cat_lower and "invocation" in cat_lower:
        return "Invalid Invocation"
    elif "misinterpretation" in cat_lower or "handoff" in cat_lower:
        return "Misinterpretation of Tool Output"
    elif "intent" in cat_lower and ("plan" in cat_lower or "misalignment" in cat_lower):
        return "Intent Plan Misalignment"
    elif "underspecified" in cat_lower or (
        "user" in cat_lower and "intent" in cat_lower and "not" not in cat_lower
    ):
        return "Underspecified User Intent"
    elif "not supported" in cat_lower or (
        "intent" in cat_lower and "not" in cat_lower and "supported" in cat_lower
    ):
        return "Intent Not Supported"
    elif "guardrail" in cat_lower:
        return "Guardrails Triggered"
    elif "system" in cat_lower and "failure" in cat_lower:
        return "System Failure"
    elif "inconclusive" in cat_lower:
        return "Inconclusive"
    else:
        return category.strip()


def extract_failure_case(value: object) -> int:
    """Coerce ints, numeric strings, and enum-style names to a case number 1-10.

    Port of AgentRx's `extract_failure_case_number`. Anything unrecognized or
    out of range becomes 10 (`Inconclusive`) rather than raising, so a judge that
    returns garbage is scored as inconclusive instead of crashing the run.
    """
    if isinstance(value, bool):
        return INCONCLUSIVE_CASE
    if isinstance(value, int):
        return value if 1 <= value <= 10 else INCONCLUSIVE_CASE

    text = str(value).strip()
    if text.isdigit():
        num = int(text)
        return num if 1 <= num <= 10 else INCONCLUSIVE_CASE

    enum_name = text.split(".")[-1] if "." in text else text
    upper = enum_name.upper().replace(" ", "_").replace("-", "_")
    return _ENUM_NAME_TO_CASE.get(upper, INCONCLUSIVE_CASE)


def category_to_case(category: str | None) -> int:
    """Normalize a category name, then map it to its case number.

    This is the function scoring should use: it composes the two upstream steps
    and routes anything unrecognized to `Inconclusive`.
    """
    normalized = normalize_category(category)
    if normalized in CATEGORY_TO_FAILURE_CASE:
        return CATEGORY_TO_FAILURE_CASE[normalized]
    return extract_failure_case(normalized)


def case_to_category(case: int) -> str:
    """Inverse of `category_to_case`, for rendering."""
    return FAILURE_CASE_TO_CATEGORY.get(case, "Inconclusive")


def taxonomy_checklist() -> str:
    """The numbered taxonomy as prompt text for the judge."""
    return "\n".join(
        f"{case}. {FAILURE_CASE_TO_CATEGORY[case]} — {CATEGORY_DESCRIPTIONS[case]}"
        for case in sorted(FAILURE_CASE_TO_CATEGORY)
    )
