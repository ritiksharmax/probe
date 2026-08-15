"""The systems compared in the benchmark table.

Because the paper publishes only relative gains (+23.6% localization, +22.9%
attribution over a Who&When baseline with a GPT-5 judge) and no absolute
accuracies, there is no published number to score against. So every comparison
point here is **run by us, on the same trajectories, with the same judge model**.
That is the only way to make the claim honest.

The systems form a 2x2 over the two things PROBE adds to a plain judge, plus an
LLM-free floor:

|                     | full trajectory     | filtered evidence |
|---------------------|---------------------|-------------------|
| no violation log    | `naive-full`        | `filtered-only`   |
| with violation log  | `agentrx-style`     | `probe`           |

`naive-full` is the prompting baseline. `agentrx-style` is a faithful
reproduction of AgentRx's shape — a violation log plus the whole trajectory
handed to a judge — which is the point of comparison that matters. `probe` is
both contributions together, and the two off-diagonal cells are what attribute
the difference between them.

`signals-only` is the LLM-free floor: whatever a judge buys has to be measured
against this, not against zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe.llm.client import LLMClient
from probe.localize.localizer import SignalPriorLocalizer
from probe.rca.judge import RCAJudge
from probe.rca.report import RCAReport
from probe.rca.taxonomy import INCONCLUSIVE_CASE
from probe.trace.model import Trajectory


@dataclass(frozen=True)
class SystemSpec:
    """One row of the comparison table."""

    name: str
    mode: str = "filtered"
    include_violations: bool = True
    uses_llm: bool = True
    description: str = ""


SYSTEMS: dict[str, SystemSpec] = {
    "probe": SystemSpec(
        name="probe",
        mode="filtered",
        include_violations=True,
        description="Filtered evidence windows + violation log (both contributions)",
    ),
    "agentrx-style": SystemSpec(
        name="agentrx-style",
        mode="full",
        include_violations=True,
        description="Violation log + full trajectory (AgentRx's shape, our judge)",
    ),
    "filtered-only": SystemSpec(
        name="filtered-only",
        mode="filtered",
        include_violations=False,
        description="Filtered evidence, no violation log (isolates the filter)",
    ),
    "naive-full": SystemSpec(
        name="naive-full",
        mode="full",
        include_violations=False,
        description="Whole trajectory, no violation log (prompting baseline)",
    ),
    "signals-only": SystemSpec(
        name="signals-only",
        uses_llm=False,
        description="Signal-density prior, no LLM call (the floor)",
    ),
}

DEFAULT_SYSTEMS = ("probe", "agentrx-style", "naive-full", "signals-only")


class SignalsOnlySystem:
    """The LLM-free floor, wrapped to return an `RCAReport` like the judges do.

    It never attributes a category — there is no cheap signal that maps reliably
    onto the taxonomy — so it scores `Inconclusive` on attribution by
    construction. That is the honest result, not a placeholder: it shows exactly
    how much of the attribution number the judge is responsible for.
    """

    def __init__(self) -> None:
        self._localizer = SignalPriorLocalizer()

    def __call__(self, trajectory: Trajectory) -> RCAReport:
        result = self._localizer(trajectory)
        return RCAReport(
            trajectory_id=trajectory.trajectory_id,
            critical_step=result.step,
            category_case=INCONCLUSIVE_CASE,
            cause="signal-density prior; no root-cause attribution attempted",
            confidence=result.confidence,
            evidence=result.events,
            windows=[(w.start, w.end) for w in result.windows],
            steps_shown=0,
            steps_total=len(trajectory),
            model="none",
            tier="signals",
        )


def build_system(
    spec: SystemSpec,
    client: LLMClient | None,
    tier: str = "",
    max_tokens: int = 8192,
    signal_caveat: bool = True,
):
    """Instantiate a system from its spec."""
    if not spec.uses_llm:
        return SignalsOnlySystem()
    if client is None:
        raise ValueError(f"system {spec.name!r} needs an LLM client")
    return RCAJudge(
        client,
        mode=spec.mode,  # type: ignore[arg-type]
        include_violations=spec.include_violations,
        include_signal_caveat=signal_caveat,
        tier=tier,
        max_tokens=max_tokens,
    )
