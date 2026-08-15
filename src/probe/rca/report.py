"""The diagnosis PROBE produces for a trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from probe.rca.taxonomy import case_to_category
from probe.signals.base import SignalEvent


@dataclass
class RCAReport:
    """A root-cause diagnosis, with the evidence and accounting behind it."""

    trajectory_id: str
    critical_step: int | None
    category_case: int
    cause: str = ""
    # What would have had to change at the critical step for the run to succeed.
    # Included because a diagnosis you cannot act on is not worth much, and
    # because it is a good check on whether the judge understood the failure.
    counterfactual: str = ""
    confidence: float = 0.0
    evidence: list[SignalEvent] = field(default_factory=list)
    # Which steps the judge was actually shown, for the H2 ablation.
    windows: list[tuple[int, int]] = field(default_factory=list)
    steps_shown: int = 0
    steps_total: int = 0

    model: str = ""
    tier: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    repairs: int = 0
    # Set when the judge could not be parsed and the signal prior was used alone.
    degraded: bool = False

    @property
    def category(self) -> str:
        return case_to_category(self.category_case)

    @property
    def compression(self) -> float:
        """Fraction of the trajectory the judge actually read."""
        return self.steps_shown / self.steps_total if self.steps_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "critical_step": self.critical_step,
            "category": self.category,
            "category_case": self.category_case,
            "cause": self.cause,
            "counterfactual": self.counterfactual,
            "confidence": self.confidence,
            "windows": self.windows,
            "steps_shown": self.steps_shown,
            "steps_total": self.steps_total,
            "compression": round(self.compression, 3),
            "model": self.model,
            "tier": self.tier,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "repairs": self.repairs,
            "degraded": self.degraded,
        }

    def render(self) -> str:
        """Human-readable report, for `probe analyze`."""
        lines = [
            f"trajectory {self.trajectory_id}",
            f"  critical step : {self.critical_step}",
            f"  category      : {self.category} (case {self.category_case})",
            f"  confidence    : {self.confidence:.2f}",
        ]
        if self.cause:
            lines.append(f"  cause         : {self.cause}")
        if self.counterfactual:
            lines.append(f"  counterfactual: {self.counterfactual}")
        if self.steps_total:
            lines.append(
                f"  evidence      : {self.steps_shown}/{self.steps_total} steps "
                f"({self.compression:.0%}) across {len(self.windows)} window(s)"
            )
        if self.evidence:
            lines.append("  signals:")
            for event in sorted(self.evidence, key=lambda e: -e.severity)[:5]:
                lines.append(f"    step {event.step_index} [{event.kind}] {event.evidence}")
        if self.degraded:
            lines.append("  NOTE: judge output was unusable; fell back to the signal prior")
        lines.append(
            f"  cost          : ${self.cost_usd:.4f} "
            f"({self.prompt_tokens}+{self.completion_tokens} tok, {self.latency_s:.1f}s)"
        )
        return "\n".join(lines)
