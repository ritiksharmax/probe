"""Evidence filtering (H2): decide what the judge is allowed to read.

This is PROBE's central bet. AgentRx hands its judge the whole trajectory; these
run to 130 steps, most of them irrelevant, and the interesting evidence is
diluted. PROBE instead scores every step by signal density, expands the strongest
into windows, and shows the judge only those. Two things should follow: better
signal-to-noise for any judge, and a small enough prompt that a 4B model can do
the job a frontier model was doing.

Scoring has three parts, each earning its place:

* **Signal mass** at the step -- what actually fired there.
* **Neighbourhood spill**, decayed by distance. A tool error at step 12 makes
  step 11 (the call that caused it) and step 13 (the reaction to it) interesting
  too, and the true root cause is frequently the neighbour rather than the step
  that tripped the signal.
* **An earliness prior.** Ground truth defines the root cause as the *first
  unrecoverable* failure, so when evidence is otherwise comparable the earlier
  step is the better answer. Without this the ranking drifts toward the end of
  the trajectory, where consequences pile up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from probe.signals.base import SignalEvent
from probe.trace.model import Trajectory


@dataclass
class StepScore:
    """Why a step is or is not suspicious."""

    index: int
    score: float
    own_mass: float
    events: list[SignalEvent] = field(default_factory=list)


@dataclass
class EvidenceWindow:
    """A contiguous span of steps offered to the judge as evidence."""

    start: int
    end: int
    score: float
    peak: int
    events: list[SignalEvent] = field(default_factory=list)

    def __contains__(self, index: int) -> bool:
        return self.start <= index <= self.end

    @property
    def n_steps(self) -> int:
        return self.end - self.start + 1

    def render(self, trajectory: Trajectory, width: int = 200) -> str:
        """The window as prompt text, with its signals attached."""
        header = f"--- steps {self.start}-{self.end} (suspicion {self.score:.2f}) ---"
        body = trajectory.render(self.start, self.end, width=width)
        if not self.events:
            return f"{header}\n{body}"
        findings = "\n".join(
            f"  ! step {e.step_index} [{e.kind}] {e.evidence}"
            for e in sorted(self.events, key=lambda e: (e.step_index, e.kind))
        )
        return f"{header}\n{body}\nsignals:\n{findings}"


class EvidenceFilter:
    """Ranks steps by suspicion and extracts the top windows."""

    def __init__(
        self,
        radius: int = 2,
        max_windows: int = 3,
        decay: float = 0.5,
        earliness_weight: float = 0.15,
        thought_weight: float = 0.5,
    ) -> None:
        self.radius = radius
        self.max_windows = max_windows
        self.decay = decay
        self.earliness_weight = earliness_weight
        self.thought_weight = thought_weight

    def score_steps(self, trajectory: Trajectory, events: list[SignalEvent]) -> list[StepScore]:
        """Score every step by local and neighbouring signal mass."""
        n = len(trajectory)
        if n == 0:
            return []

        own: dict[int, float] = dict.fromkeys(range(1, n + 1), 0.0)
        at_step: dict[int, list[SignalEvent]] = {i: [] for i in range(1, n + 1)}
        for event in events:
            if 1 <= event.step_index <= n:
                own[event.step_index] += event.severity
                at_step[event.step_index].append(event)

        scores: list[StepScore] = []
        for index in range(1, n + 1):
            total = own[index]
            # Spill from neighbours, decayed by distance.
            for offset in range(1, self.radius + 1):
                weight = self.decay**offset
                for neighbour in (index - offset, index + offset):
                    if 1 <= neighbour <= n:
                        total += own[neighbour] * weight

            step = trajectory.step(index)
            # Deliberation rarely *is* the failure, even when it surrounds one.
            if step.is_thought:
                total *= self.thought_weight

            # Earliness prior: the root cause is the first unrecoverable failure.
            if total > 0 and n > 1:
                position = (index - 1) / (n - 1)
                total *= 1.0 + self.earliness_weight * (1.0 - position)

            scores.append(
                StepScore(index=index, score=total, own_mass=own[index], events=at_step[index])
            )
        return scores

    def windows(self, trajectory: Trajectory, events: list[SignalEvent]) -> list[EvidenceWindow]:
        """Extract up to `max_windows` non-overlapping evidence windows.

        Windows are grown greedily from the highest-scoring steps. When nothing
        fired at all, the tail of the trajectory is returned rather than nothing:
        an unexplained failure still has to be diagnosed, and the ending is the
        best available guess at where to look.
        """
        scores = self.score_steps(trajectory, events)
        if not scores:
            return []

        n = len(trajectory)
        ranked = sorted(scores, key=lambda s: (-s.score, s.index))
        if ranked[0].score <= 0:
            start = max(1, n - 2 * self.radius)
            return [EvidenceWindow(start=start, end=n, score=0.0, peak=n, events=[])]

        claimed: set[int] = set()
        windows: list[EvidenceWindow] = []
        for candidate in ranked:
            if len(windows) >= self.max_windows:
                break
            if candidate.score <= 0 or candidate.index in claimed:
                continue

            start = max(1, candidate.index - self.radius)
            end = min(n, candidate.index + self.radius)
            span = set(range(start, end + 1))
            if span & claimed:
                # Trim back to the unclaimed portion rather than overlapping.
                while start <= end and start in claimed:
                    start += 1
                while end >= start and end in claimed:
                    end -= 1
                if start > end:
                    continue
                span = set(range(start, end + 1))

            claimed |= span
            windows.append(
                EvidenceWindow(
                    start=start,
                    end=end,
                    score=candidate.score,
                    peak=candidate.index,
                    events=[e for e in events if start <= e.step_index <= end],
                )
            )

        return sorted(windows, key=lambda w: w.start)

    def render(
        self, trajectory: Trajectory, windows: list[EvidenceWindow], width: int = 200
    ) -> str:
        """All windows as a single prompt block, with elisions marked.

        The gaps are labelled rather than silently dropped so the judge knows it
        is seeing an excerpt and that step numbers are not contiguous.
        """
        if not windows:
            return "(no evidence windows)"

        parts: list[str] = []
        previous_end = 0
        for window in windows:
            skipped = window.start - previous_end - 1
            if skipped > 0:
                parts.append(f"… {skipped} step(s) omitted …")
            parts.append(window.render(trajectory, width=width))
            previous_end = window.end

        trailing = len(trajectory) - previous_end
        if trailing > 0:
            parts.append(f"… {trailing} step(s) omitted …")
        return "\n".join(parts)
