"""Tests for evidence filtering and signal-prior localization."""

from __future__ import annotations

import pytest

from probe.localize import EvidenceFilter, SignalPriorLocalizer
from probe.signals.base import SignalEvent
from probe.trace.model import Step, ToolCall, ToolResult, Trajectory


def plain(n: int, **kw) -> Trajectory:
    return Trajectory(
        trajectory_id="t",
        steps=[Step(index=i, role="assistant", content=f"step {i}", **kw) for i in range(1, n + 1)],
    )


def event(step: int, severity: float = 0.6, kind: str = "tool_error") -> SignalEvent:
    return SignalEvent(step_index=step, kind=kind, severity=severity, evidence="e")


class TestStepScoring:
    def test_signal_mass_lands_on_its_step(self):
        scores = EvidenceFilter().score_steps(plain(10), [event(5)])
        by_index = {s.index: s for s in scores}
        assert by_index[5].own_mass == pytest.approx(0.6)
        assert by_index[5].score > by_index[1].score

    def test_neighbours_receive_decayed_spill(self):
        scores = {
            s.index: s.score
            for s in EvidenceFilter(radius=2, decay=0.5).score_steps(plain(10), [event(5)])
        }
        assert scores[5] > scores[4] > scores[3]
        assert scores[3] > 0
        assert scores[2] == 0  # beyond the radius

    def test_earliness_prior_breaks_ties_toward_the_earlier_step(self):
        """Ground truth defines the root cause as the first unrecoverable failure."""
        traj = plain(21)
        scores = {
            s.index: s.score for s in EvidenceFilter().score_steps(traj, [event(3), event(19)])
        }
        assert scores[3] > scores[19]

    def test_thought_steps_are_discounted(self):
        traj = Trajectory(
            trajectory_id="t",
            steps=[
                Step(index=1, role="agent", agent="Orchestrator (thought)", content="thinking"),
                Step(index=2, role="agent", agent="WebSurfer", content="acting"),
            ],
        )
        scores = {
            s.index: s.score
            for s in EvidenceFilter(radius=0).score_steps(traj, [event(1), event(2)])
        }
        assert scores[1] < scores[2]

    def test_empty_trajectory(self):
        assert EvidenceFilter().score_steps(Trajectory(trajectory_id="t"), []) == []

    def test_out_of_range_events_are_ignored(self):
        scores = EvidenceFilter().score_steps(plain(3), [event(99)])
        assert all(s.own_mass == 0 for s in scores)


class TestWindows:
    def test_window_centres_on_the_peak(self):
        windows = EvidenceFilter(radius=2, max_windows=1).windows(plain(20), [event(10)])
        assert len(windows) == 1
        assert windows[0].peak == 10
        assert windows[0].start == 8
        assert windows[0].end == 12

    def test_windows_do_not_overlap(self):
        windows = EvidenceFilter(radius=2, max_windows=3).windows(
            plain(30), [event(5), event(9), event(20)]
        )
        claimed = [i for w in windows for i in range(w.start, w.end + 1)]
        assert len(claimed) == len(set(claimed))

    def test_respects_max_windows(self):
        events = [event(i) for i in (3, 10, 17, 24, 31)]
        assert len(EvidenceFilter(max_windows=2).windows(plain(40), events)) <= 2

    def test_windows_are_clipped_to_the_trajectory(self):
        windows = EvidenceFilter(radius=5, max_windows=1).windows(plain(4), [event(1)])
        assert windows[0].start == 1
        assert windows[0].end == 4

    def test_windows_returned_in_step_order(self):
        windows = EvidenceFilter(max_windows=3).windows(plain(40), [event(30), event(5), event(18)])
        assert [w.start for w in windows] == sorted(w.start for w in windows)

    def test_no_signals_falls_back_to_the_tail(self):
        """An unexplained failure still has to be diagnosed somewhere."""
        windows = EvidenceFilter(radius=2).windows(plain(20), [])
        assert len(windows) == 1
        assert windows[0].end == 20

    def test_membership_and_size(self):
        window = EvidenceFilter(radius=2, max_windows=1).windows(plain(20), [event(10)])[0]
        assert 10 in window
        assert 99 not in window
        assert window.n_steps == 5


class TestRendering:
    def test_render_marks_elisions_and_keeps_step_numbers(self):
        traj = plain(30)
        filt = EvidenceFilter(radius=1, max_windows=2)
        windows = filt.windows(traj, [event(5), event(25)])
        text = filt.render(traj, windows)
        assert "omitted" in text
        assert "[5]" in text and "[25]" in text
        assert "[15]" not in text

    def test_render_includes_signal_evidence(self):
        traj = plain(10)
        filt = EvidenceFilter(radius=1, max_windows=1)
        text = filt.render(traj, filt.windows(traj, [event(5)]))
        assert "tool_error" in text

    def test_render_with_no_windows(self):
        assert EvidenceFilter().render(plain(5), []) == "(no evidence windows)"


class TestSignalPriorLocalizer:
    def test_localizes_to_the_failing_step(self):
        traj = Trajectory(
            trajectory_id="t",
            steps=[
                Step(index=1, role="user", content="do it"),
                Step(
                    index=2, role="assistant", tool_calls=[ToolCall(id="c", name="f", arguments={})]
                ),
                Step(
                    index=3,
                    role="tool",
                    content="Error: not found",
                    tool_result=ToolResult(
                        call_id="c", name="f", content="Error: not found", is_error=True
                    ),
                ),
                Step(index=4, role="assistant", content="All set!"),
            ],
        )
        result = SignalPriorLocalizer()(traj)
        assert result.step in (2, 3)
        assert result.confidence > 0
        assert result.source == "signal_prior"

    def test_falls_back_to_the_last_step_when_nothing_fires(self):
        """Returning nothing would forfeit the trajectory entirely."""
        result = SignalPriorLocalizer()(plain(6))
        assert result.step == 6
        assert result.confidence == 0.0

    def test_empty_trajectory_yields_no_step(self):
        result = SignalPriorLocalizer()(Trajectory(trajectory_id="t"))
        assert result.step is None

    def test_ranked_steps_are_ordered(self):
        traj = plain(20)
        localizer = SignalPriorLocalizer(signals=[lambda t: [event(5), event(12, 0.3)]])
        ranked = localizer(traj).ranked_steps(limit=3)
        assert [s for s, _ in ranked][0] == 5
        assert [score for _, score in ranked] == sorted((s for _, s in ranked), reverse=True)

    def test_windows_are_attached(self):
        traj = plain(20)
        localizer = SignalPriorLocalizer(signals=[lambda t: [event(5)]])
        assert localizer(traj).windows
