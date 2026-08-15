"""Tests for the benchmark scoring protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "agentrx"))

from protocol import (  # noqa: E402
    AnnotatedFailure,
    GroundTruthEntry,
    Prediction,
    aggregate_seeds,
    load_ground_truth,
    score,
    score_detection,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tau_gt() -> dict[str, GroundTruthEntry]:
    return load_ground_truth(FIXTURES / "tau_ground_truth_slice.json", "tau_retail")


class TestGroundTruthLoading:
    def test_critical_step_comes_from_the_root_cause_not_the_first_failure(self, tau_gt):
        entry = tau_gt["2"]
        assert len(entry.failures) > 1
        # The earliest annotated failure is step 3, but the root cause is the
        # designated one -- scoring must follow root_cause.failure_id.
        assert entry.failures[0].step_number == 3
        assert entry.root_cause is not None
        assert entry.critical_step == entry.root_cause.step_number

    def test_derived_scoring_targets(self, tau_gt):
        entry = tau_gt["2"]
        steps = sorted(f.step_number for f in entry.failures)
        assert entry.earliest_case == next(
            f.case for f in entry.failures if f.step_number == steps[0]
        )
        assert entry.terminal_case == next(
            f.case for f in entry.failures if f.step_number == steps[-1]
        )
        assert entry.root_cause_case in entry.all_cases

    def test_string_failure_id_still_resolves(self):
        """The one record storing root_cause.failure_id as "1" against int ids.

        Upstream's strict `==` drops this trajectory's root cause; coercing both
        sides to str recovers it, so our denominator is complete.
        """
        gt = load_ground_truth(FIXTURES / "magentic_gt_stringid.json", "magentic_one")
        entry = gt["08cae58d-4084-4616-b6dd-dd6534e4825b"]
        assert entry.root_cause_failure_id == "1"
        assert entry.root_cause is not None
        assert entry.critical_step is not None


class TestLocalizationScoring:
    def _gt(self, critical_step: int, case_name: str = "Invalid Invocation"):
        return {
            "t": GroundTruthEntry(
                trajectory_id="t",
                domain="d",
                failures=(
                    AnnotatedFailure(failure_id="1", step_number=critical_step, category=case_name),
                ),
                root_cause_failure_id="1",
            )
        }

    def test_exact_and_tolerance_bands(self):
        gt = self._gt(10)
        result = score([Prediction("t", step=12, category=None)], gt, domain="d", system="s")
        assert result.localization[0] == 0.0
        assert result.localization[1] == 0.0
        assert result.localization[2] == 1.0
        assert result.localization[5] == 1.0
        assert result.mean_step_distance == 2

    def test_exact_hit_counts_in_every_band(self):
        gt = self._gt(10)
        result = score([Prediction("t", step=10, category=None)], gt, domain="d", system="s")
        assert all(v == 1.0 for v in result.localization.values())

    def test_tolerance_is_symmetric(self):
        gt = self._gt(10)
        early = score([Prediction("t", step=8, category=None)], gt, domain="d", system="s")
        late = score([Prediction("t", step=12, category=None)], gt, domain="d", system="s")
        assert early.localization[2] == late.localization[2] == 1.0

    def test_unknown_trajectories_are_ignored(self):
        gt = self._gt(10)
        result = score([Prediction("nope", step=10, category=None)], gt, domain="d", system="s")
        assert result.n_scored == 0

    def test_no_step_prediction_scores_zero_but_still_counts(self):
        gt = self._gt(10)
        result = score([Prediction("t", step=None, category=None)], gt, domain="d", system="s")
        assert result.n_scored == 1
        assert result.localization[0] == 0.0
        assert result.mean_step_distance is None


class TestAttributionScoring:
    def test_all_four_attribution_variants(self, tau_gt):
        entry = tau_gt["2"]
        correct = score(
            [Prediction("2", step=entry.critical_step, category=entry.root_cause_case)],
            tau_gt,
            domain="tau_retail",
            system="s",
        )
        assert correct.attribution["root_cause"] == 1.0
        assert correct.attribution["any"] == 1.0

    def test_category_accepted_as_name_or_case_number(self, tau_gt):
        entry = tau_gt["2"]
        by_case = score(
            [Prediction("2", step=None, category=entry.root_cause_case)],
            tau_gt,
            domain="d",
            system="s",
        )
        by_name = score(
            [Prediction("2", step=None, category=entry.root_cause.category)],
            tau_gt,
            domain="d",
            system="s",
        )
        assert by_case.attribution["root_cause"] == by_name.attribution["root_cause"] == 1.0

    def test_wrong_category_scores_zero(self, tau_gt):
        wrong = 10 if tau_gt["2"].root_cause_case != 10 else 1
        result = score([Prediction("2", step=None, category=wrong)], tau_gt, domain="d", system="s")
        assert result.attribution["root_cause"] == 0.0


class TestDetectionScoring:
    def test_recall_and_false_positive_rate(self):
        preds = [
            Prediction("f1", None, None, predicted_failed=True),
            Prediction("f2", None, None, predicted_failed=False),
            Prediction("s1", None, None, predicted_failed=False),
            Prediction("s2", None, None, predicted_failed=True),
            Prediction("s3", None, None, predicted_failed=False),
        ]
        result = score_detection(preds, failed_ids={"f1", "f2"}, succeeded_ids={"s1", "s2", "s3"})
        assert result["recall"] == 0.5
        assert result["false_positive_rate"] == pytest.approx(1 / 3)
        assert result["n_positive"] == 2
        assert result["n_negative"] == 3

    def test_predictions_without_a_verdict_are_skipped(self):
        result = score_detection(
            [Prediction("f1", None, None, predicted_failed=None)],
            failed_ids={"f1"},
            succeeded_ids=set(),
        )
        assert result["n_positive"] == 0


def test_aggregate_seeds_reports_mean_and_spread(tau_gt):
    runs = [
        score([Prediction("2", step=s, category=None)], tau_gt, domain="d", system="s")
        for s in (tau_gt["2"].critical_step, tau_gt["2"].critical_step + 9)
    ]
    agg = aggregate_seeds(runs)
    assert agg["localization"][0]["mean"] == 0.5
    assert agg["localization"][0]["std"] > 0
    assert agg["localization"][0]["n_seeds"] == 2


def test_aggregate_seeds_empty():
    assert aggregate_seeds([]) == {}
