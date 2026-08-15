"""The signal interface.

A signal is a cheap, deterministic, LLM-free predicate over a trajectory that
emits evidence of trouble at specific steps. Signals are the foundation of both
hypotheses PROBE rests on: they are what makes **detection** (H1) possible
without paying for a judge, and their density over the trajectory is the prior
that drives **evidence filtering** (H2).

Signals are intentionally allowed to be noisy. A signal firing does not mean the
run failed; the detector aggregates them, and precision is recovered there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from probe.trace.model import Trajectory


@dataclass(frozen=True)
class SignalEvent:
    """Evidence that something went wrong at a step."""

    step_index: int
    kind: str
    # 0..1. Calibrated so that ~1.0 means "this alone strongly implies failure"
    # and ~0.2 means "mildly suspicious in isolation".
    severity: float
    evidence: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if self.step_index < 1:
            raise ValueError(f"step_index must be 1-based, got {self.step_index}")


@runtime_checkable
class Signal(Protocol):
    """Anything that can read a trajectory and emit `SignalEvent`s."""

    name: str

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]: ...


def run_signals(trajectory: Trajectory, signals: list[Signal]) -> list[SignalEvent]:
    """Run every signal and return all events, ordered by step.

    A signal that raises is skipped rather than allowed to abort the analysis:
    production traces are malformed in ways no fixture anticipates, and one bad
    signal should not cost the whole diagnosis.
    """
    events: list[SignalEvent] = []
    for signal in signals:
        try:
            events.extend(signal(trajectory))
        except Exception as exc:  # noqa: BLE001 - deliberate isolation
            events.append(
                SignalEvent(
                    step_index=1,
                    kind="signal_error",
                    severity=0.0,
                    evidence=f"signal {getattr(signal, 'name', signal)!r} raised: {exc}",
                )
            )
    return sorted(events, key=lambda e: (e.step_index, e.kind))


def default_signals() -> list[Signal]:
    """The standard signal battery."""
    from probe.signals.budget import BudgetAnomaly, StallSignal
    from probe.signals.loop import NoProgressSignal, OscillationSignal, RepeatedCallSignal
    from probe.signals.outcome import IncompleteOutcomeSignal, RefusalSignal
    from probe.signals.tool import (
        MalformedArgumentsSignal,
        ToolErrorSignal,
        UnknownToolSignal,
    )

    return [
        ToolErrorSignal(),
        MalformedArgumentsSignal(),
        UnknownToolSignal(),
        RepeatedCallSignal(),
        OscillationSignal(),
        NoProgressSignal(),
        RefusalSignal(),
        IncompleteOutcomeSignal(),
        BudgetAnomaly(),
        StallSignal(),
    ]
