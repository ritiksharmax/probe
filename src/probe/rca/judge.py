"""The RCA judge: localize the critical step and attribute a root cause.

This is where the H2 bet is cashed. The judge runs in one of two modes:

* `filtered` — it sees the violation log plus a few evidence windows, typically
  under a fifth of the trajectory. This is PROBE.
* `full` — it sees the entire trajectory. This is the prompting baseline, and the
  control for the ablation.

Both modes share every other input — same taxonomy, same instructions, same
schema, same model — so a difference between them is attributable to the
evidence filter and nothing else. That is the whole point of building the
baseline into the same class rather than as a separate script.

The judge never sees ground truth, and it never sees the reference action
sequence some domains ship. PROBE's premise is diagnosis without ground-truth
trajectories; leaking it here would make every benchmark number meaningless.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from probe.llm.client import LLMClient, parse_json
from probe.localize.evidence import EvidenceFilter
from probe.rca.report import RCAReport
from probe.rca.taxonomy import INCONCLUSIVE_CASE, category_to_case, taxonomy_checklist
from probe.signals.base import Signal, SignalEvent, default_signals, run_signals
from probe.trace.model import Trajectory

Mode = Literal["filtered", "full"]

# Measured on the AgentRx data: the nearest signal sits a median of 6 (tau-retail)
# to 8 (Magentic-One) steps *after* the annotated root cause. Handing the judge a
# bare list of signal locations therefore anchors it on symptoms. This caveat
# tells it what the list actually represents.
_SIGNAL_CAVEAT = (
    "These are automatically detected *symptoms*, not causes. A signal marks "
    "where a problem became visible; the step that caused it is usually EARLIER "
    "-- typically several steps before the first signal. Use these to locate the "
    "region of trouble, then look back from there for the decision that made the "
    "failure inevitable."
)

SYSTEM_PROMPT = """You diagnose failures in AI agent execution traces.

You are given a failed agent trajectory. Your job is to identify:
1. The CRITICAL STEP — the first unrecoverable failure, the step where the run \
became doomed. This is usually not the step where the failure became visible: \
prefer the earliest step that made the bad outcome inevitable, not the step that \
reported it.
2. The ROOT CAUSE CATEGORY, chosen from the fixed taxonomy below.

Steps are numbered starting at 1. Answer with the step number as shown.

Be decisive. If several steps look plausible, pick the earliest one that is \
genuinely unrecoverable. Only answer with category 10 (Inconclusive) when the \
evidence genuinely does not support any other category."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critical_step": {
            "type": "integer",
            "description": "1-based step number of the first unrecoverable failure",
        },
        "category": {
            "type": "integer",
            "description": "Root cause category number, 1-10",
        },
        "cause": {
            "type": "string",
            "description": "One or two sentences explaining what went wrong at that step",
        },
        "counterfactual": {
            "type": "string",
            "description": "What the agent should have done instead at that step",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this diagnosis, 0 to 1",
        },
    },
    "required": ["critical_step", "category", "cause", "counterfactual", "confidence"],
    "additionalProperties": False,
}


class RCAJudge:
    """Assembles the prompt, calls the model, and parses a diagnosis."""

    def __init__(
        self,
        client: LLMClient,
        mode: Mode = "filtered",
        signals: list[Signal] | None = None,
        evidence_filter: EvidenceFilter | None = None,
        tier: str = "",
        max_repairs: int = 2,
        max_tokens: int = 8192,
        include_violations: bool = True,
        include_signal_caveat: bool = True,
    ) -> None:
        self.client = client
        self.mode = mode
        self.signals = signals if signals is not None else default_signals()
        self.filter = evidence_filter or EvidenceFilter()
        self.tier = tier
        self.max_repairs = max_repairs
        self.max_tokens = max_tokens
        # Separable from `mode` on purpose. PROBE makes two claims — that
        # filtering the trajectory helps, and that a cheap violation log helps —
        # and crossing the two flags is what tells them apart. Without this,
        # "filtered + violations" beating "full, no violations" would be a result
        # with two possible explanations.
        self.include_violations = include_violations
        # A flag, not a source edit, because the caveat's effect is itself a
        # result worth measuring and re-measuring. The first A/B of it was run by
        # hand-editing this string, which left the "without" arm unreproducible
        # from any commit.
        self.include_signal_caveat = include_signal_caveat

    # ------------------------------------------------------------------ prompt

    def build_prompt(
        self, trajectory: Trajectory, events: list[SignalEvent]
    ) -> tuple[str, list, int]:
        """Return the prompt, the evidence windows, and how many steps are shown."""
        parts: list[str] = []

        task = trajectory.task
        if task.instruction:
            parts.append(f"## Task\n{task.instruction.strip()}")
        if task.policy:
            parts.append(f"## Domain policy (excerpt)\n{_clip(task.policy, 2000)}")

        if events and self.include_violations:
            preamble = _SIGNAL_CAVEAT + "\n" if self.include_signal_caveat else ""
            parts.append("## Violation log\n" + preamble + _render_violations(events))

        if self.mode == "filtered":
            windows = self.filter.windows(trajectory, events)
            shown = sum(w.n_steps for w in windows)
            parts.append(
                "## Evidence (excerpt)\n"
                f"These are the most suspicious spans of a {len(trajectory)}-step "
                "trajectory. Step numbers are absolute; omitted spans are marked.\n\n"
                + self.filter.render(trajectory, windows)
            )
        else:
            windows = []
            shown = len(trajectory)
            parts.append(f"## Trajectory ({len(trajectory)} steps)\n" + trajectory.render())

        parts.append("## Failure taxonomy\n" + taxonomy_checklist())
        parts.append(
            "## Your answer\n"
            "Respond with a JSON object containing: critical_step (integer), "
            "category (integer 1-10), cause (string), counterfactual (string), "
            "confidence (number 0-1). Respond with JSON only."
        )
        return "\n\n".join(parts), windows, shown

    # ------------------------------------------------------------------- call

    def __call__(self, trajectory: Trajectory) -> RCAReport:
        started = time.perf_counter()
        events = run_signals(trajectory, self.signals)
        prompt, windows, shown = self.build_prompt(trajectory, events)

        prompt_tokens = completion_tokens = 0
        cost = 0.0
        repairs = 0
        parsed: dict[str, Any] | None = None
        model = getattr(self.client, "model", "")

        attempt_prompt = prompt
        for attempt in range(self.max_repairs + 1):
            response = self.client.complete(
                attempt_prompt,
                system=SYSTEM_PROMPT,
                schema=RESPONSE_SCHEMA,
                max_tokens=self.max_tokens,
            )
            model = response.model or model
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            cost += response.cost_usd

            try:
                parsed = parse_json(response.text)
                break
            except ValueError:
                repairs = attempt + 1
                if attempt == self.max_repairs:
                    break
                # Small models routinely wrap JSON in prose. Re-ask rather than
                # discarding the trajectory, and count the retry against the tier.
                attempt_prompt = (
                    prompt + "\n\n## Correction\nYour previous reply was not valid JSON. "
                    "Reply with the JSON object only — no prose, no code fences."
                )

        latency = time.perf_counter() - started

        report = RCAReport(
            trajectory_id=trajectory.trajectory_id,
            critical_step=None,
            category_case=INCONCLUSIVE_CASE,
            evidence=events,
            windows=[(w.start, w.end) for w in windows],
            steps_shown=shown,
            steps_total=len(trajectory),
            model=model,
            tier=self.tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_s=latency,
            repairs=repairs,
        )

        if parsed is None:
            # Judge unusable. Fall back to the signal prior rather than forfeiting
            # the trajectory — a degraded answer still scores, and the flag keeps
            # the failure visible in the results.
            report.degraded = True
            report.critical_step = self._signal_prior_step(trajectory, events)
            return report

        report.critical_step = _coerce_step(parsed.get("critical_step"), len(trajectory))
        report.category_case = _coerce_category(parsed.get("category"))
        report.cause = str(parsed.get("cause") or "")
        report.counterfactual = str(parsed.get("counterfactual") or "")
        report.confidence = _coerce_confidence(parsed.get("confidence"))
        if report.critical_step is None:
            report.critical_step = self._signal_prior_step(trajectory, events)
        return report

    def _signal_prior_step(self, trajectory: Trajectory, events: list[SignalEvent]) -> int | None:
        """Highest signal mass, used when the judge produces nothing usable.

        Uses *this judge's* evidence filter rather than a fresh default one: a
        caller who tuned the filter would otherwise get a fallback scored against
        different settings than the evidence the judge actually saw.
        """
        scores = self.filter.score_steps(trajectory, events)
        best = max(scores, key=lambda s: (s.score, -s.index), default=None)
        if best is None:
            return len(trajectory) or None
        return best.index if best.score > 0 else (len(trajectory) or None)


# --------------------------------------------------------------------- helpers


def _render_violations(events: list[SignalEvent], limit: int = 40) -> str:
    """The violation log, strongest first, capped so it cannot crowd out evidence."""
    ranked = sorted(events, key=lambda e: (-e.severity, e.step_index))[:limit]
    lines = [
        f"- step {e.step_index} [{e.kind}, severity {e.severity:.2f}] {e.evidence}"
        for e in sorted(ranked, key=lambda e: e.step_index)
    ]
    if len(events) > limit:
        lines.append(f"- … {len(events) - limit} weaker signal(s) omitted")
    return "\n".join(lines)


def _coerce_step(value: Any, n_steps: int) -> int | None:
    """Clamp a claimed step into range instead of discarding the answer.

    A judge that says 'step 0' or overshoots the end has still expressed an
    opinion about where the failure is; clamping keeps that signal, and the
    tolerance bands in the metric make near-misses legible anyway.
    """
    try:
        step = int(value)
    except (TypeError, ValueError):
        return None
    if n_steps <= 0:
        return None
    return max(1, min(step, n_steps))


def _coerce_category(value: Any) -> int:
    """Accept a case number or a category name; anything else is Inconclusive."""
    if isinstance(value, bool):
        return INCONCLUSIVE_CASE
    if isinstance(value, int):
        return value if 1 <= value <= 10 else INCONCLUSIVE_CASE
    if isinstance(value, str):
        return category_to_case(value)
    return INCONCLUSIVE_CASE


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"
