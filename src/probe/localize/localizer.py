"""Critical-step localization.

Two localizers live here. `SignalPriorLocalizer` is LLM-free: it returns the
highest-scoring step from the evidence filter. It is both a usable cheap tier and
the signals-only ablation row of the benchmark -- the floor any judge has to beat
to justify its cost.

The ensemble localizer that combines this prior with a judge's opinion is the
other half, and it is what H2 predicts should win: the judge supplies semantic
understanding the signals cannot, while the prior keeps it anchored to steps
where something demonstrably happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from probe.localize.evidence import EvidenceFilter, EvidenceWindow, StepScore
from probe.signals.base import Signal, SignalEvent, default_signals, run_signals
from probe.trace.model import Trajectory


@dataclass
class Localization:
    """A predicted critical step, with the evidence that produced it."""

    trajectory_id: str
    step: int | None
    confidence: float
    windows: list[EvidenceWindow] = field(default_factory=list)
    scores: list[StepScore] = field(default_factory=list)
    events: list[SignalEvent] = field(default_factory=list)
    source: str = "signal_prior"

    def ranked_steps(self, limit: int = 5) -> list[tuple[int, float]]:
        """Top candidate steps, best first."""
        return [
            (s.index, s.score)
            for s in sorted(self.scores, key=lambda s: (-s.score, s.index))[:limit]
        ]


class SignalPriorLocalizer:
    """Picks the step with the most signal mass. No LLM calls."""

    def __init__(
        self,
        signals: list[Signal] | None = None,
        evidence_filter: EvidenceFilter | None = None,
    ) -> None:
        self.signals = signals if signals is not None else default_signals()
        self.filter = evidence_filter or EvidenceFilter()

    def __call__(self, trajectory: Trajectory) -> Localization:
        events = run_signals(trajectory, self.signals)
        scores = self.filter.score_steps(trajectory, events)
        windows = self.filter.windows(trajectory, events)

        best = max(scores, key=lambda s: (s.score, -s.index), default=None)
        if best is None or best.score <= 0:
            # Nothing fired. Fall back to the last step: with no evidence to the
            # contrary, the end of a failed run is the least-wrong guess, and
            # returning nothing would forfeit the trajectory entirely.
            step = len(trajectory) or None
            confidence = 0.0
        else:
            step = best.index
            total = sum(s.score for s in scores) or 1.0
            confidence = best.score / total

        return Localization(
            trajectory_id=trajectory.trajectory_id,
            step=step,
            confidence=confidence,
            windows=windows,
            scores=scores,
            events=events,
            source="signal_prior",
        )
