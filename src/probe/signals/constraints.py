"""Declarative constraints over a trajectory.

The rest of the signal battery detects failures that *look* broken — errors,
loops, refusals, runs that stop early. It is structurally blind to the failure
that looks perfectly healthy: every tool call succeeds, nothing errors, and the
agent quietly does the wrong thing. Refunding $200 against a $20 order trips no
error signal at all, and "Instruction Adherence Failure" is one of the most
common annotated root causes in the benchmark.

Constraints close that gap. A constraint is a named predicate over the
trajectory that emits a violation with an explanation. Two kinds are supported:

* **Declarative** — written by hand against a domain's rules. Cheap, exact, and
  auditable.
* **Synthesized** — generated once per tool-schema set by an LLM, then cached by
  schema hash. This is AgentRx's constraint-synthesis idea with the cost
  amortized: the same schemas recur across every trace from a deployment, so the
  synthesis is paid once rather than per trajectory.

The cache is the whole economic argument, so `synthesize_constraints` is built
around it rather than treating it as an optimization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from probe.signals.base import SignalEvent
from probe.trace.model import Step, ToolCall, Trajectory

# A check returns an explanation when violated, or None when satisfied.
StepCheck = Callable[[Step, Trajectory], str | None]
CallCheck = Callable[[ToolCall, Step, Trajectory], str | None]


@dataclass
class Constraint:
    """A named rule over a trajectory."""

    name: str
    description: str
    check: Callable[[Trajectory], Iterable[tuple[int, str]]]
    severity: float = 0.75

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        return [
            SignalEvent(
                step_index=step_index,
                kind="constraint_violation",
                severity=self.severity,
                evidence=f"{self.name}: {explanation}",
                detail={"constraint": self.name, "description": self.description},
            )
            for step_index, explanation in self.check(trajectory)
        ]


def step_constraint(
    name: str, description: str, check: StepCheck, severity: float = 0.75
) -> Constraint:
    """Build a constraint that inspects each step independently."""

    def run(trajectory: Trajectory) -> Iterable[tuple[int, str]]:
        for step in trajectory.steps:
            explanation = check(step, trajectory)
            if explanation:
                yield step.index, explanation

    return Constraint(name=name, description=description, check=run, severity=severity)


def tool_call_constraint(
    name: str,
    description: str,
    tool: str,
    check: CallCheck,
    severity: float = 0.75,
) -> Constraint:
    """Build a constraint that inspects every call to one tool.

    The check receives the whole trajectory as well as the call, so a rule can
    refer to what earlier steps established — which is what most real policies
    need ("refund no more than the order total" is meaningless without the
    order).
    """

    def run(trajectory: Trajectory) -> Iterable[tuple[int, str]]:
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.name != tool:
                    continue
                explanation = check(call, step, trajectory)
                if explanation:
                    yield step.index, explanation

    return Constraint(name=name, description=description, check=run, severity=severity)


def require_before(
    name: str,
    description: str,
    prerequisite: str,
    dependent: str,
    severity: float = 0.8,
) -> Constraint:
    """Require one tool to have been called before another is.

    This covers a large share of real policy text — authenticate before acting,
    look up before modifying, confirm before committing — and it is the pattern
    the tau-retail policy violation in the benchmark's own ground truth turns on.
    """

    def run(trajectory: Trajectory) -> Iterable[tuple[int, str]]:
        seen = False
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.name == prerequisite:
                    seen = True
                elif call.name == dependent and not seen:
                    yield (
                        step.index,
                        f"called {dependent!r} before any call to {prerequisite!r}",
                    )

    return Constraint(name=name, description=description, check=run, severity=severity)


def numeric_bound(
    name: str,
    description: str,
    tool: str,
    argument: str,
    maximum: float | None = None,
    minimum: float | None = None,
    severity: float = 0.85,
) -> Constraint:
    """Bound a numeric tool argument by a constant."""

    def check(call: ToolCall, step: Step, trajectory: Trajectory) -> str | None:
        raw = call.arguments.get(argument)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if maximum is not None and value > maximum:
            return f"{tool}.{argument} = {value:g} exceeds the maximum of {maximum:g}"
        if minimum is not None and value < minimum:
            return f"{tool}.{argument} = {value:g} is below the minimum of {minimum:g}"
        return None

    return tool_call_constraint(name, description, tool, check, severity=severity)


class ConstraintSignal:
    """Runs a set of constraints as a single signal."""

    name = "constraint_violation"

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = constraints or []

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events: list[SignalEvent] = []
        for constraint in self.constraints:
            # One malformed constraint must not take down the analysis; a domain
            # rule that raises on an unexpected trace shape is a bug in the rule.
            try:
                events.extend(constraint(trajectory))
            except Exception as exc:  # noqa: BLE001 - deliberate isolation
                events.append(
                    SignalEvent(
                        step_index=1,
                        kind="signal_error",
                        severity=0.0,
                        evidence=f"constraint {constraint.name!r} raised: {exc}",
                    )
                )
        return events


# --------------------------------------------------------------- LLM synthesis


def schema_fingerprint(tool_schemas: list[dict[str, Any]], policy: str | None = None) -> str:
    """Stable hash of a tool-schema set plus policy text.

    This is the cache key that makes synthesis affordable. Schemas are sorted and
    serialized deterministically so that dict ordering never produces a spurious
    miss — the same failure mode that silently breaks prompt caching.
    """
    payload = json.dumps(
        {
            "schemas": sorted(
                (s for s in tool_schemas if isinstance(s, dict)),
                key=lambda s: str(s.get("name", "")),
            ),
            "policy": (policy or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


SYNTHESIS_PROMPT = """You are given an agent's tool schemas and its domain policy.

Write the checkable invariants an agent in this domain must not violate. Prefer \
rules that can be decided from the trajectory alone: ordering requirements \
("authenticate before acting"), argument bounds, and forbidden call sequences.

Respond with a JSON object: {"constraints": [{"name": ..., "description": ..., \
"kind": "require_before"|"numeric_bound"|"other", "params": {...}}]}"""


@dataclass
class ConstraintSynthesizer:
    """Synthesizes constraints from tool schemas, cached by fingerprint.

    The cache is the point. Synthesis costs one LLM call per *deployment* rather
    than per trajectory, so its cost amortizes to nothing across real traffic
    while the resulting checks stay free to evaluate.
    """

    client: Any
    cache: dict[str, list[Constraint]] = field(default_factory=dict)
    calls: int = field(default=0, init=False)

    def for_trajectory(self, trajectory: Trajectory) -> list[Constraint]:
        schemas = trajectory.task.tool_schemas
        if not schemas:
            return []

        key = schema_fingerprint(schemas, trajectory.task.policy)
        if key in self.cache:
            return self.cache[key]

        prompt = (
            f"## Tool schemas\n{json.dumps(schemas, indent=2, default=str)}\n\n"
            f"## Domain policy\n{(trajectory.task.policy or '(none provided)').strip()}"
        )
        self.calls += 1
        response = self.client.complete(prompt, system=SYNTHESIS_PROMPT)
        constraints = _constraints_from_spec(response.text)
        self.cache[key] = constraints
        return constraints


def _constraints_from_spec(text: str) -> list[Constraint]:
    """Turn a synthesized spec into executable constraints.

    Only the kinds with a safe executable form are materialized. Anything else is
    described but not enforced — synthesizing arbitrary predicates and running
    them would mean executing model-authored code, which is not a trade this
    library should make silently.
    """
    from probe.llm.client import parse_json

    try:
        spec = parse_json(text)
    except ValueError:
        return []

    built: list[Constraint] = []
    for entry in spec.get("constraints", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "synthesized")
        description = str(entry.get("description") or "")
        params = entry.get("params") or {}
        kind = entry.get("kind")

        if kind == "require_before" and {"prerequisite", "dependent"} <= set(params):
            built.append(
                require_before(
                    name, description, str(params["prerequisite"]), str(params["dependent"])
                )
            )
        elif kind == "numeric_bound" and {"tool", "argument"} <= set(params):
            maximum = params.get("maximum")
            minimum = params.get("minimum")
            built.append(
                numeric_bound(
                    name,
                    description,
                    str(params["tool"]),
                    str(params["argument"]),
                    maximum=float(maximum) if maximum is not None else None,
                    minimum=float(minimum) if minimum is not None else None,
                )
            )
    return built
