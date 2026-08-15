"""Tests for failure detection and weight calibration."""

from __future__ import annotations

import pytest

from probe.detect import CalibratedDetector, Detector, cross_val_report, fit
from probe.signals.base import SignalEvent
from probe.trace.model import Step, ToolCall, ToolResult, Trajectory


def traj(tid: str, *steps: Step) -> Trajectory:
    return Trajectory(trajectory_id=tid, steps=list(steps))


def healthy(tid: str = "ok") -> Trajectory:
    return traj(
        tid,
        Step(index=1, role="user", content="place an order"),
        Step(
            index=2,
            role="assistant",
            tool_calls=[ToolCall(id="c", name="order", arguments={"id": 1})],
        ),
        Step(
            index=3,
            role="tool",
            content='{"status":"placed"}',
            tool_result=ToolResult(call_id="c", name="order", content='{"status":"placed"}'),
        ),
        Step(index=4, role="assistant", content="Done, your order is placed."),
    )


def broken(tid: str = "bad") -> Trajectory:
    return traj(
        tid,
        Step(index=1, role="user", content="place an order"),
        Step(
            index=2,
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="order", arguments={"id": 1})],
        ),
        Step(
            index=3,
            role="tool",
            content="Error: item not found",
            tool_result=ToolResult(
                call_id="c1", name="order", content="Error: item not found", is_error=True
            ),
        ),
        Step(
            index=4,
            role="assistant",
            tool_calls=[ToolCall(id="c2", name="order", arguments={"id": 1})],
        ),
        Step(
            index=5,
            role="tool",
            content="Error: item not found",
            tool_result=ToolResult(
                call_id="c2", name="order", content="Error: item not found", is_error=True
            ),
        ),
        Step(index=6, role="assistant", content="I am unable to complete this request."),
    )


class TestDetector:
    def test_flags_a_broken_run(self):
        verdict = Detector()(broken())
        assert verdict.failed
        assert verdict.confidence >= 0.5

    def test_clears_a_healthy_run(self):
        verdict = Detector()(healthy())
        assert not verdict.failed

    def test_broken_scores_above_healthy(self):
        assert Detector()(broken()).confidence > Detector()(healthy()).confidence

    def test_repeats_of_one_kind_do_not_saturate_confidence(self):
        """The point of per-kind max: one problem repeated is still one problem."""
        det = Detector()
        one = [SignalEvent(step_index=1, kind="tool_error", severity=0.6, evidence="e")]
        many = [
            SignalEvent(step_index=i, kind="tool_error", severity=0.6, evidence="e")
            for i in range(1, 21)
        ]
        assert det.confidence(one) == pytest.approx(det.confidence(many))

    def test_distinct_kinds_corroborate(self):
        det = Detector()
        one_kind = [SignalEvent(step_index=1, kind="tool_error", severity=0.6, evidence="e")]
        two_kinds = one_kind + [
            SignalEvent(step_index=2, kind="refusal", severity=0.6, evidence="e")
        ]
        assert det.confidence(two_kinds) > det.confidence(one_kind)

    def test_no_signals_means_no_failure(self):
        assert Detector().confidence([]) == 0.0

    def test_threshold_is_respected(self):
        assert Detector(threshold=0.99)(broken()).failed is False
        assert Detector(threshold=0.01)(broken()).failed is True

    def test_a_run_with_no_signals_is_never_flagged(self):
        """Zero evidence means zero confidence, at any threshold above zero."""
        verdict = Detector(threshold=0.01)(healthy())
        assert verdict.confidence == 0.0
        assert not verdict.failed

    def test_explain_is_readable(self):
        text = Detector()(broken()).explain()
        assert "FAILED" in text
        assert "step" in text

    def test_explain_with_no_events(self):
        assert Detector()(Trajectory(trajectory_id="empty")).explain() == "no signals fired"

    def test_kinds_reports_strongest_per_kind(self):
        verdict = Detector()(broken())
        assert set(verdict.kinds) <= {e.kind for e in verdict.events}
        for kind, severity in verdict.kinds.items():
            assert severity == max(e.severity for e in verdict.events if e.kind == kind)


class TestCalibration:
    def _corpus(self):
        trajectories = [broken(f"b{i}") for i in range(6)] + [healthy(f"g{i}") for i in range(6)]
        labels = [True] * 6 + [False] * 6
        return trajectories, labels

    def test_fit_learns_positive_weights_for_failure_signals(self):
        model = fit(*self._corpus())
        assert model.log_ratios
        assert model.log_ratios.get("tool_error", 0) > 0
        assert model.log_ratios.get("refusal", 0) > 0

    def test_fitted_model_separates_the_classes(self):
        model = fit(*self._corpus())
        assert model.score(broken("x")) > model.score(healthy("y"))

    def test_unseen_kind_is_ignored_rather_than_fatal(self):
        model = fit(*self._corpus())
        assert 0.0 <= model._score_kinds({"never_seen_before"}) <= 1.0

    def test_smoothing_keeps_weights_finite(self):
        """A kind firing on only one class must not produce an infinite weight."""
        model = fit(*self._corpus())
        assert all(abs(w) < 100 for w in model.log_ratios.values())

    def test_fit_rejects_single_class(self):
        with pytest.raises(ValueError, match="at least one failed and one successful"):
            fit([broken("a"), broken("b")], [True, True])

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            fit([broken("a")], [True, False])

    def test_fit_rejects_empty_corpus(self):
        with pytest.raises(ValueError, match="empty corpus"):
            fit([], [])

    def test_cross_val_holds_out(self):
        report = cross_val_report(*self._corpus(), folds=3)
        assert report.n_positive == 6
        assert report.n_negative == 6
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.false_positive_rate <= 1.0
        assert report.folds == 3

    def test_cross_val_folds_are_clamped_to_class_size(self):
        """Asking for more folds than positives must not produce empty training sets."""
        trajectories = [broken(f"b{i}") for i in range(2)] + [healthy(f"g{i}") for i in range(5)]
        labels = [True] * 2 + [False] * 5
        report = cross_val_report(trajectories, labels, folds=10)
        assert report.folds == 2
        assert report.n_positive == 2

    def test_report_is_printable(self):
        assert "recall=" in str(cross_val_report(*self._corpus(), folds=3))

    def test_calibrated_detector_returns_a_verdict(self):
        model = fit(*self._corpus())
        verdict = model(broken("z"))
        assert verdict.trajectory_id == "z"
        assert verdict.events
        assert isinstance(verdict.failed, bool)

    def test_top_weights_ordered_by_magnitude(self):
        weights = fit(*self._corpus()).top_weights()
        assert [abs(w) for _, w in weights] == sorted((abs(w) for _, w in weights), reverse=True)

    def test_empty_model_scores_the_bias(self):
        model = CalibratedDetector(log_ratios={}, bias=0.0)
        assert model._score_kinds(set()) == pytest.approx(0.5)
