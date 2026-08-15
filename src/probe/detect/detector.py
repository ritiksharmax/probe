"""Failure detection (H1): decide whether a run failed, using signals alone.

This is the capability AgentRx does not have -- it is handed trajectories already
known to be failures. In production nobody labels the failures for you, so
detection is what makes the rest of the pipeline usable at all, and it has to be
cheap enough to run on every trace. No LLM calls happen here.

Aggregation is a **per-kind noisy-OR**: within a signal kind we keep the single
strongest event, then combine across kinds. Taking the max within a kind first is
the load-bearing part -- an agent that repeats one call twenty times produces
twenty `repeated_call` events describing one problem, and a naive noisy-OR over
all events would drive any such run to certainty. Independent *kinds* of evidence
corroborate each other; repetitions of one kind do not.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from probe.signals.base import Signal, SignalEvent, default_signals, run_signals
from probe.trace.model import Trajectory

# Kinds whose presence says little about whether the run *failed*, even though
# they help locate a failure once one is known. Down-weighted for detection so
# they cannot convict a healthy run on their own.
_WEAK_FOR_DETECTION = {
    "no_progress": 0.35,
    "budget_anomaly": 0.35,
    "stall": 0.4,
    "repeated_call": 0.5,
    "truncated_run": 0.6,
    "signal_error": 0.0,
}


@dataclass
class FailureVerdict:
    """Whether a trajectory failed, and the evidence behind the call."""

    trajectory_id: str
    failed: bool
    confidence: float
    events: list[SignalEvent] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def kinds(self) -> dict[str, float]:
        """Strongest severity observed per signal kind."""
        strongest: dict[str, float] = {}
        for event in self.events:
            strongest[event.kind] = max(strongest.get(event.kind, 0.0), event.severity)
        return strongest

    def explain(self, limit: int = 5) -> str:
        """Human-readable rationale, strongest evidence first."""
        if not self.events:
            return "no signals fired"
        ranked = sorted(self.events, key=lambda e: -e.severity)[:limit]
        lines = [f"  step {e.step_index}: [{e.kind}] {e.evidence}" for e in ranked]
        verdict = "FAILED" if self.failed else "ok"
        return f"{verdict} (confidence {self.confidence:.2f})\n" + "\n".join(lines)


class Detector:
    """Aggregates signal events into a failure verdict."""

    def __init__(
        self,
        signals: list[Signal] | None = None,
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.signals = signals if signals is not None else default_signals()
        self.threshold = threshold
        self.weights = {**_WEAK_FOR_DETECTION, **(weights or {})}

    def confidence(self, events: list[SignalEvent]) -> float:
        """Per-kind noisy-OR over weighted severities."""
        strongest: dict[str, float] = defaultdict(float)
        for event in events:
            weighted = event.severity * self.weights.get(event.kind, 1.0)
            strongest[event.kind] = max(strongest[event.kind], weighted)

        product = 1.0
        for severity in strongest.values():
            product *= 1.0 - min(max(severity, 0.0), 1.0)
        return 1.0 - product

    def __call__(self, trajectory: Trajectory) -> FailureVerdict:
        events = run_signals(trajectory, self.signals)
        confidence = self.confidence(events)
        return FailureVerdict(
            trajectory_id=trajectory.trajectory_id,
            failed=confidence >= self.threshold,
            confidence=confidence,
            events=events,
            threshold=self.threshold,
        )
