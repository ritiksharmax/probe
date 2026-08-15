"""Signals for agents that are going in circles.

An agent that repeats itself has stopped making progress, and the step where the
repetition *starts* is usually nearer the true root cause than the step where it
finally gives up. These signals therefore attribute to the onset of the pattern,
not its tail.
"""

from __future__ import annotations

import json
from collections import defaultdict

from probe.signals.base import SignalEvent
from probe.trace.model import Step, Trajectory


def _call_key(step: Step) -> tuple[str, ...] | None:
    """A hashable identity for a step's tool calls, or None if it made none."""
    if not step.tool_calls:
        return None
    return tuple(
        f"{c.name}({json.dumps(c.arguments, sort_keys=True, default=str)})" for c in step.tool_calls
    )


class RepeatedCallSignal:
    """The same tool call, with identical arguments, issued more than once.

    Repeating an identical call cannot produce new information in a deterministic
    environment, so it is nearly always either a retry after a failure or a loop.
    """

    name = "repeated_call"

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        occurrences: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for step in trajectory.steps:
            key = _call_key(step)
            if key is not None:
                occurrences[key].append(step.index)

        events = []
        for key, indices in occurrences.items():
            if len(indices) < self.threshold:
                continue
            # Severity grows with repetition but saturates -- eight repeats is not
            # meaningfully worse evidence than five.
            severity = min(0.3 + 0.15 * (len(indices) - 1), 0.85)
            for index in indices[1:]:
                events.append(
                    SignalEvent(
                        step_index=index,
                        kind=self.name,
                        severity=severity,
                        evidence=f"repeats the identical call {key[0]} "
                        f"first made at step {indices[0]} ({len(indices)}x total)",
                        detail={"first_step": indices[0], "count": len(indices)},
                    )
                )
        return events


class OscillationSignal:
    """The agent alternates between two actions (A, B, A, B) without resolving.

    Distinct from a plain repeat: oscillation usually means the agent is toggling
    between two readings of the same ambiguous situation.
    """

    name = "oscillation"

    def __init__(self, min_cycles: int = 2) -> None:
        self.min_cycles = min_cycles

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        acting = [(s.index, _call_key(s)) for s in trajectory.steps if _call_key(s) is not None]
        events = []
        i = 0
        while i + 3 < len(acting) + 1:
            window = acting[i : i + 4]
            if len(window) < 4:
                break
            (i1, a), (_, b), (_, a2), (i4, b2) = window
            if a == a2 and b == b2 and a != b:
                events.append(
                    SignalEvent(
                        step_index=i1,
                        kind=self.name,
                        severity=0.65,
                        evidence=f"alternates between {a[0]} and {b[0]} "
                        f"across steps {i1}-{i4} without resolving",
                        detail={"start": i1, "end": i4},
                    )
                )
                i += 4  # do not re-report the same cycle from its midpoint
            else:
                i += 1
        return events


class NoProgressSignal:
    """A long run of internal deliberation that never acts.

    Weighted low on its own. In Magentic-One, `Orchestrator (thought)` steps are
    the majority of the trajectory, so thinking is normal and only an unusually
    long unbroken stretch of it is notable.
    """

    name = "no_progress"

    def __init__(self, window: int = 6) -> None:
        self.window = window

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        run_start: int | None = None
        run_len = 0

        for step in trajectory.steps:
            acted = bool(step.tool_calls) or step.tool_result is not None
            speaks_to_user = step.role in {"user", "assistant"} and not step.is_thought

            if acted or speaks_to_user:
                if run_len >= self.window and run_start is not None:
                    events.append(self._event(run_start, run_len))
                run_start, run_len = None, 0
            else:
                run_start = step.index if run_start is None else run_start
                run_len += 1

        if run_len >= self.window and run_start is not None:
            events.append(self._event(run_start, run_len))
        return events

    def _event(self, start: int, length: int) -> SignalEvent:
        return SignalEvent(
            step_index=start,
            kind=self.name,
            severity=min(0.2 + 0.05 * (length - self.window), 0.5),
            evidence=f"{length} consecutive steps of deliberation without acting, "
            f"starting at step {start}",
            detail={"start": start, "length": length},
        )
