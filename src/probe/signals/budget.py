"""Signals for runs that consumed an unusual amount of budget.

Length is weak evidence on its own -- some tasks are legitimately long -- so
these fire softly and matter mainly in aggregate. Where corpus statistics are
available they are used in preference to absolute thresholds, because "long" is
only meaningful relative to the traffic a deployment actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from probe.signals.base import SignalEvent
from probe.trace.model import Trajectory


@dataclass(frozen=True)
class CorpusStats:
    """Reference distribution of trajectory lengths for a population of runs."""

    mean_steps: float
    std_steps: float

    @classmethod
    def from_trajectories(cls, trajectories: list[Trajectory]) -> CorpusStats:
        lengths = [len(t) for t in trajectories if len(t) > 0]
        if not lengths:
            return cls(mean_steps=0.0, std_steps=0.0)
        return cls(
            mean_steps=fmean(lengths),
            std_steps=pstdev(lengths) if len(lengths) > 1 else 0.0,
        )


class BudgetAnomaly:
    """The run is much longer than comparable runs.

    With `stats`, fires on a z-score; without, falls back to an absolute step
    ceiling. The fallback is deliberately generous -- a false positive here costs
    a wasted judge call, and the detector requires corroboration anyway.
    """

    name = "budget_anomaly"

    def __init__(
        self,
        stats: CorpusStats | None = None,
        z_threshold: float = 2.0,
        absolute_steps: int = 80,
    ) -> None:
        self.stats = stats
        self.z_threshold = z_threshold
        self.absolute_steps = absolute_steps

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        n = len(trajectory)
        if n == 0:
            return []

        if self.stats and self.stats.std_steps > 0:
            z = (n - self.stats.mean_steps) / self.stats.std_steps
            if z >= self.z_threshold:
                return [
                    SignalEvent(
                        step_index=n,
                        kind=self.name,
                        severity=min(0.2 + 0.1 * (z - self.z_threshold), 0.5),
                        evidence=f"{n} steps is {z:.1f}σ above the corpus mean of "
                        f"{self.stats.mean_steps:.0f}",
                        detail={"steps": n, "z": round(z, 2)},
                    )
                ]
            return []

        if n >= self.absolute_steps:
            return [
                SignalEvent(
                    step_index=n,
                    kind=self.name,
                    severity=0.25,
                    evidence=f"{n} steps exceeds the default ceiling of {self.absolute_steps}",
                    detail={"steps": n},
                )
            ]
        return []


class StallSignal:
    """A long uninterrupted run of tool calls with no reply to the user.

    Distinct from `no_progress`, which covers deliberation without action. This
    covers frantic action without communication, the shape of an agent thrashing
    against an obstacle it has not recognized.
    """

    name = "stall"

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        run_start: int | None = None
        run_len = 0

        for step in trajectory.steps:
            is_tool_activity = bool(step.tool_calls) or step.tool_result is not None
            answers_user = (
                step.role == "assistant"
                and not step.tool_calls
                and step.tool_result is None
                and step.content.strip()
            )

            if is_tool_activity:
                run_start = step.index if run_start is None else run_start
                run_len += 1
            elif answers_user:
                if run_len >= self.window and run_start is not None:
                    events.append(self._event(run_start, run_len))
                run_start, run_len = None, 0

        if run_len >= self.window and run_start is not None:
            events.append(self._event(run_start, run_len))
        return events

    def _event(self, start: int, length: int) -> SignalEvent:
        return SignalEvent(
            step_index=start,
            kind=self.name,
            severity=min(0.2 + 0.03 * (length - self.window), 0.45),
            evidence=f"{length} consecutive tool interactions from step {start} "
            "without responding to the user",
            detail={"start": start, "length": length},
        )
