"""Tests for the benchmark scoring protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "agentrx"))

from protocol import (  # noqa: E402
    AnnotatedFailure,
    GroundTruthEntry,
    Prediction,
    aggregate_repeats,
    as_records,
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


def test_aggregate_repeats_reports_mean_and_spread(tau_gt):
    runs = [
        score([Prediction("2", step=s, category=None)], tau_gt, domain="d", system="s")
        for s in (tau_gt["2"].critical_step, tau_gt["2"].critical_step + 9)
    ]
    row = aggregate_repeats(runs)
    assert row["exact"] == 0.5
    assert row["exact_std"] > 0
    assert row["repeats"] == 2
    assert row["exact_per_repeat"] == [1.0, 0.0]


def test_aggregate_repeats_empty():
    assert aggregate_repeats([]) == {}


class TestRepeatsCannotHideAFailure:
    """A failure in any repeat must reach the row, not just a failure in the first.

    The old aggregator took `err` off repeat 1 and overwrote only the accuracies,
    so a dead endpoint during repeat 2 or 3 depressed the mean while the row still
    reported zero errors -- numbers that look clean and are not.
    """

    def _runs(self, tau_gt, error_on):
        runs = []
        for i in range(3):
            pred = (
                Prediction("2", step=None, category=None, error="connection refused")
                if i == error_on
                else Prediction("2", step=tau_gt["2"].critical_step, category=None)
            )
            runs.append(score([pred], tau_gt, domain="d", system="s"))
        return runs

    @pytest.mark.parametrize("error_on", [0, 1, 2])
    def test_error_in_any_repeat_is_counted(self, tau_gt, error_on):
        row = aggregate_repeats(self._runs(tau_gt, error_on))
        assert row["err"] == 1, f"failure in repeat {error_on + 1} vanished from the row"
        assert row["err_per_repeat"][error_on] == 1

    def test_surviving_repeats_are_recoverable(self, tau_gt):
        """A contaminated repeat must not force re-running the clean ones."""
        row = aggregate_repeats(self._runs(tau_gt, error_on=0))
        assert row["exact_per_repeat"] == [0.0, 1.0, 1.0]


class TestErrorAccounting:
    """A failed diagnosis must never be indistinguishable from a wrong one.

    A dead endpoint once turned 251 failed calls into zeros that read as results;
    these pin the accounting that makes that visible.
    """

    def _gt(self):
        return {
            "t": GroundTruthEntry(
                trajectory_id="t",
                domain="d",
                failures=(
                    AnnotatedFailure(failure_id="1", step_number=5, category="System Failure"),
                ),
                root_cause_failure_id="1",
            )
        }

    def test_errors_are_counted(self):
        result = score(
            [Prediction("t", step=None, category=None, error="connection refused")],
            self._gt(),
            domain="d",
            system="s",
        )
        assert result.n_errors == 1
        assert result.as_row()["err"] == 1

    def test_clean_run_reports_zero_errors(self):
        result = score([Prediction("t", step=5, category=9)], self._gt(), domain="d", system="s")
        assert result.n_errors == 0
        assert result.localization[0] == 1.0

    def test_an_errored_prediction_still_counts_against_accuracy(self):
        """It is a failure to diagnose, so it must not be quietly excluded."""
        result = score(
            [Prediction("t", step=None, category=None, error="boom")],
            self._gt(),
            domain="d",
            system="s",
        )
        assert result.n_scored == 1
        assert result.localization[0] == 0.0


class TestWindowRecall:
    def _gt(self, step=20):
        return {
            "t": GroundTruthEntry(
                trajectory_id="t",
                domain="d",
                failures=(
                    AnnotatedFailure(failure_id="1", step_number=step, category="System Failure"),
                ),
                root_cause_failure_id="1",
            )
        }

    def test_true_step_inside_a_window(self):
        result = score(
            [Prediction("t", step=1, category=None, windows=((15, 25),))],
            self._gt(),
            domain="d",
            system="s",
        )
        assert result.window_recall == 1.0

    def test_true_step_outside_every_window_caps_the_ceiling(self):
        result = score(
            [Prediction("t", step=1, category=None, windows=((1, 5), (30, 35)))],
            self._gt(),
            domain="d",
            system="s",
        )
        assert result.window_recall == 0.0

    def test_unfiltered_systems_report_no_window_recall(self):
        result = score(
            [Prediction("t", step=20, category=None)], self._gt(), domain="d", system="s"
        )
        assert result.window_recall is None

    def test_exact_accuracy_cannot_exceed_window_recall(self):
        """The invariant that makes win_rec a ceiling worth reporting."""
        gt = self._gt(20)
        preds = [
            Prediction(
                "t", step=20, category=None, windows=((1, 5),)
            ),  # right answer, unseen window
        ]
        result = score(preds, gt, domain="d", system="s")
        assert result.window_recall == 0.0
        # The judge happened to be right, but it could not have seen the step --
        # which is exactly why the ceiling is reported alongside the accuracy.
        assert result.localization[0] == 1.0


class TestPredictionRecords:
    """Per-trajectory records, so a post-hoc question does not cost another run.

    Aggregates cannot say *why* a system scored as it did -- whether a violation
    log pushes attribution toward the categories its own signals name, for
    instance. That question was asked here and could not be answered without
    re-running the whole benchmark.
    """

    def test_pairs_each_prediction_with_its_target(self, tau_gt):
        gt = tau_gt["2"]
        records = as_records([Prediction("2", step=99, category=4)], tau_gt)
        assert len(records) == 1
        assert records[0]["pred_step"] == 99
        assert records[0]["pred_case"] == 4
        assert records[0]["gt_step"] == gt.critical_step
        assert records[0]["gt_case"] == gt.root_cause_case

    def test_failed_diagnoses_keep_their_error(self, tau_gt):
        records = as_records([Prediction("2", None, None, error="refused")], tau_gt)
        assert records[0]["error"] == "refused"
        assert records[0]["pred_step"] is None

    def test_unannotated_trajectories_are_dropped(self, tau_gt):
        assert as_records([Prediction("not-in-gt", step=1, category=1)], tau_gt) == []

    def test_windows_survive_json_round_trip(self, tau_gt):
        records = as_records([Prediction("2", step=1, category=1, windows=((3, 9),))], tau_gt)
        assert json.loads(json.dumps(records))[0]["windows"] == [[3, 9]]
