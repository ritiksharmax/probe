"""Regression tests for the benchmark table renderer.

A drift between two parallel `headers`/`keys` lists once crashed a completed
25-minute benchmark at the final `print`, discarding every diagnosis. These pin
both the structural fix and the ordering that makes such a crash survivable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "agentrx"))

from protocol import AnnotatedFailure, GroundTruthEntry, Prediction, score  # noqa: E402
from run import COLUMNS, format_table  # noqa: E402


def _row() -> dict:
    gt = {
        "t": GroundTruthEntry(
            trajectory_id="t",
            domain="d",
            failures=(AnnotatedFailure(failure_id="1", step_number=5, category="System Failure"),),
            root_cause_failure_id="1",
        )
    }
    return score(
        [Prediction("t", step=5, category=9, windows=((1, 9),), prompt_tokens=100)],
        gt,
        domain="tau_retail",
        system="probe/small",
    ).as_row()


def test_every_column_key_is_produced_by_the_scorer():
    """The exact drift that crashed a finished run."""
    produced = set(_row())
    for header, key in COLUMNS:
        assert key in produced, f"column {header!r} reads missing key {key!r}"


def test_renders_without_raising():
    text = format_table([_row()])
    assert "tau_retail" in text
    assert "probe/small" in text


def test_header_separator_then_body():
    lines = format_table([_row()]).splitlines()
    assert lines[0].startswith("domain")
    assert set(lines[1].strip()) == {"-", " "}
    assert "tau_retail" in lines[2]


def test_missing_values_render_as_a_dash_not_a_crash():
    text = format_table([{"domain": "d", "system": "s"}])
    assert "-" in text


def test_all_columns_are_present_in_the_header():
    header = format_table([_row()]).splitlines()[0]
    for name, _ in COLUMNS:
        assert name in header
