"""Signals read off how the run ended.

These are the highest-precision detection signals available without a judge: an
agent that says it cannot do something is reporting its own failure. They are
also the most position-sensitive, so they attribute to the closing steps and let
localization move the blame earlier.
"""

from __future__ import annotations

import re

from probe.signals.base import SignalEvent
from probe.trace.model import Trajectory

# The agent declining, giving up, or reporting it could not finish.
_REFUSAL = re.compile(
    r"\b(i (?:can(?:no|')t|am unable to|was unable to|cannot)"
    r"|unable to (?:complete|find|locate|access|determine|proceed|help)"
    r"|i (?:do not|don't) have (?:access|permission|the ability)"
    r"|i(?:'m| am) (?:sorry|afraid)[^.]{0,60}\b(?:can(?:no|')t|unable|not able)"
    r"|not (?:permitted|allowed|authorized)"
    r"|i (?:give up|could not complete))",
    re.IGNORECASE,
)

# The agent signalling the task is unfinished or blocked.
_INCOMPLETE = re.compile(
    r"\b(unfortunately|regrettably)\b"
    r"|\b(?:could|was) not (?:be )?(?:complete[d]?|finish(?:ed)?|resolve[d]?|found)\b"
    r"|\bno (?:results?|matches?|information) (?:were |was )?(?:found|available)\b"
    r"|\bfailed to\b"
    r"|\bgiving up\b",
    re.IGNORECASE,
)

# An error surfaced verbatim to the user rather than handled.
# Each alternative carries its own boundaries: a trailing \b cannot follow the
# ":" in "ValueError:", since neither side of that position is a word character.
_LEAKED_ERROR = re.compile(
    r"\btraceback \(most recent call last\)"
    r"|\bstack ?trace\b"
    r"|\b[A-Za-z]+(?:Error|Exception):"
    r"|\bstatus code [45]\d\d\b"
    r"|\bhttp [45]\d\d\b",
    re.IGNORECASE,
)


def _tail(trajectory, fraction: float = 0.25, minimum: int = 3):
    """The closing steps, where outcome evidence concentrates."""
    n = len(trajectory)
    cutoff = max(1, n - max(minimum, int(n * fraction)) + 1)
    return [s for s in trajectory.steps if s.index >= cutoff]


class RefusalSignal:
    """The agent said it could not do the task.

    Restricted to the tail: an agent noting mid-run that it cannot use one
    approach is normal, whereas ending on that note is not. Tool *observations*
    are excluded -- a tool reporting "not found" is `tool_error`'s business, and
    counting it here would double-count the same evidence.
    """

    name = "refusal"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        for step in _tail(trajectory):
            if step.tool_result is not None or step.role in {"system", "user"}:
                continue
            match = _REFUSAL.search(step.content)
            if match:
                events.append(
                    SignalEvent(
                        step_index=step.index,
                        kind=self.name,
                        severity=0.7,
                        evidence=f"agent declines or gives up: …{_around(step.content, match)}…",
                        detail={"phrase": match.group(0)},
                    )
                )
        return events


class IncompleteOutcomeSignal:
    """The run ends without a delivered answer, or ends on an error.

    Three distinct endings are treated as evidence: the agent reporting
    incompleteness, a raw error leaking into user-visible text, and a trajectory
    that simply stops mid-action with no closing response at all.
    """

    name = "incomplete_outcome"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        for step in _tail(trajectory):
            if step.role in {"system", "user"}:
                continue
            text = step.content
            if step.tool_result is None and (match := _INCOMPLETE.search(text)):
                events.append(
                    SignalEvent(
                        step_index=step.index,
                        kind=self.name,
                        severity=0.5,
                        evidence=f"reports the task is incomplete: …{_around(text, match)}…",
                        detail={"phrase": match.group(0)},
                    )
                )
            if match := _LEAKED_ERROR.search(text):
                events.append(
                    SignalEvent(
                        step_index=step.index,
                        kind="leaked_error",
                        severity=0.6,
                        evidence=f"raw error surfaced in output: …{_around(text, match)}…",
                        detail={"phrase": match.group(0)},
                    )
                )

        if trajectory.steps:
            last = trajectory.steps[-1]
            # Ending on a pending tool call means the run was cut off before the
            # agent could act on the result, or never answered the user.
            if last.tool_calls or last.tool_result is not None:
                events.append(
                    SignalEvent(
                        step_index=last.index,
                        kind="truncated_run",
                        severity=0.45,
                        evidence="trajectory ends on tool activity with no closing "
                        "response to the user",
                    )
                )
        return events


def _around(text: str, match: re.Match, width: int = 70) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return " ".join(text[start:end].split())
