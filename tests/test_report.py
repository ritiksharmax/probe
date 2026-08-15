"""Tests for the results reporter.

The reporter exists to stop three specific reporting mistakes that were all made
by hand here: averaging a contaminated row in as if it were a result, pooling a
system across domains it did not survive, and quoting a single domain's slice as
though it were the whole benchmark. Each is pinned below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "agentrx"))

from report import _pool, _pooled_std, compare, report  # noqa: E402


def _row(domain, system, n, exact, *, err=0, std=0.0, cat=0.1, pm5=0.4):
    return {
        "domain": domain,
        "system": system,
        "n": n,
        "err": err,
        "exact": exact,
        "±5": pm5,
        "root_cause_cat": cat,
        "exact_std": std,
    }


def _results(rows):
    return {"code": {"commit": "a" * 40, "dirty": False}, "config": {}, "rows": rows}


class TestPooling:
    def test_pool_weights_by_trajectory_count_not_by_domain(self):
        """A 44-trajectory domain must not count the same as a 29-trajectory one."""
        rows = [_row("tau", "s", 29, 1.0), _row("mag", "s", 44, 0.0)]
        mean, total = _pool(rows, "exact")
        assert total == 73
        assert mean == 29 / 73

    def test_pooled_std_shrinks_relative_to_domain_spread(self):
        rows = [_row("tau", "s", 29, 0.2, std=0.06), _row("mag", "s", 44, 0.1, std=0.04)]
        pooled = _pooled_std(rows)
        assert pooled < max(r["exact_std"] for r in rows)
        assert pooled > 0

    def test_pool_ignores_missing_metrics(self):
        rows = [_row("tau", "s", 29, 0.2), {"domain": "mag", "system": "s", "n": 44}]
        mean, total = _pool(rows, "exact")
        assert total == 29
        assert mean == 0.2


class TestContaminatedRows:
    def test_a_row_with_errors_is_reported_invalid(self):
        text = report(_results([_row("mag", "naive", 44, 0.1, err=24)]))
        assert "INVALID" in text
        assert "24 failed" in text

    def test_a_contaminated_row_is_not_averaged_into_the_pool(self):
        """The whole point: a depressed row must not quietly move a pooled number."""
        rows = [_row("tau", "s", 29, 0.2), _row("mag", "s", 44, 0.0, err=24)]
        text = report(_results(rows))
        assert "not pooled" in text
        assert "mag" in text

    def test_per_repeat_errors_are_surfaced(self):
        row = _row("mag", "naive", 44, 0.1, err=24)
        row["err_per_repeat"] = [24, 0, 0]
        assert "[24, 0, 0]" in report(_results([row]))


class TestPartialCoverage:
    def test_system_missing_a_domain_is_not_pooled(self):
        rows = [
            _row("tau", "full", 29, 0.2),
            _row("mag", "full", 44, 0.1),
            _row("tau", "half", 29, 0.9),
        ]
        text = report(_results(rows))
        pooled = text.split("Pooled across domains")[1]
        assert "half" in pooled and "not pooled" in pooled
        # The complete system still pools, and to the weighted value.
        assert f"{(0.2 * 29 + 0.1 * 44) / 73:.3f}" in pooled

    def test_spread_is_restated_in_trajectories(self):
        """'±0.06' reads as precision until it is also shown as ±1.7 of 29."""
        text = report(_results([_row("tau", "s", 29, 0.2, std=0.06)]))
        assert "±0.060" in text
        assert "±1.7 traj" in text


class TestTokenCost:
    def test_total_tokens_are_shown_when_present(self):
        """`$` reads 0.0000 on a self-hosted model, so `tok` is the cost signal."""
        row = _row("tau", "s", 29, 0.2)
        row["tok"] = 6489.333  # averaged across repeats, so a float
        assert "tok=6,489" in report(_results([row]))
        assert "6,489.3" not in report(_results([row]))

    def test_absent_tokens_are_omitted_rather_than_printed_as_zero(self):
        text = report(_results([_row("tau", "signals-only", 29, 0.2)]))
        assert "tok=" not in text


def test_provenance_is_shown():
    text = report(_results([_row("tau", "s", 29, 0.2)]))
    assert "aaaaaaaaaaaa" in text


def test_dirty_tree_is_called_out():
    results = _results([_row("tau", "s", 29, 0.2)])
    results["code"]["dirty"] = True
    assert "dirty tree" in report(results)


def test_partial_runs_are_labelled():
    """A checkpointed file must not read as a finished benchmark."""
    results = _results([_row("tau", "s", 29, 0.2)])
    results["complete"] = False
    assert "PARTIAL" in report(results)


def test_complete_runs_are_not_labelled_partial():
    results = _results([_row("tau", "s", 29, 0.2)])
    results["complete"] = True
    assert "PARTIAL" not in report(results)


class TestCrossRunComparison:
    """Two runs of one configuration are the sharpest test of an ordering.

    Repeats inside a run capture serving non-determinism only. An ordering that
    reshuffles between separate runs is noise however clean either run looked --
    which is what happened to "the violation log helps attribution" here.
    """

    def test_reshuffled_ordering_is_visible(self):
        a = _results([_row("tau", "x", 29, 0.3), _row("tau", "y", 29, 0.1)])
        b = _results([_row("tau", "x", 29, 0.1), _row("tau", "y", 29, 0.3)])
        text = compare(a, b)
        assert "run A: x > y" in text
        assert "run B: y > x" in text

    def test_contaminated_rows_are_not_ranked(self):
        a = _results([_row("tau", "x", 29, 0.3), _row("tau", "bad", 29, 0.9, err=5)])
        text = compare(a, a)
        assert "bad" not in text.split("Systems valid")[0]

    def test_deltas_reported_only_for_rows_valid_in_both(self):
        a = _results([_row("tau", "x", 29, 0.20), _row("tau", "only_a", 29, 0.5)])
        b = _results([_row("tau", "x", 29, 0.25)])
        tail = compare(a, b).split("Systems valid in both runs")[1]
        assert "0.200→0.250 (+0.050)" in tail
        assert "only_a" not in tail
