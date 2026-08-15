"""Signals over tool use.

Tool-level breakage is the cheapest and most reliable evidence in a trajectory:
an error observation or a malformed call is a fact, not an inference.
"""

from __future__ import annotations

import re
from typing import Any

from probe.signals.base import SignalEvent
from probe.trace.model import Trajectory

# Phrases that mark a tool observation as a failure. tau-bench returns errors as
# ordinary string payloads with no status field, so there is nothing structural
# to key on and matching text is the only option available.
_ERROR_PATTERNS = re.compile(
    r"^\s*(error|exception|traceback)\b"
    r"|\berror:\s"
    r"|\b(not found|does not exist|no such|invalid|unauthorized|forbidden|denied)\b"
    r"|\b(timed? ?out|timeout)\b"
    r"|\bcannot\b.*\b(find|locate|access)\b",
    re.IGNORECASE,
)


class ToolErrorSignal:
    """A tool returned an error observation."""

    name = "tool_error"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        for step in trajectory.steps:
            result = step.tool_result
            if result is None:
                continue
            text = result.content or ""
            if result.is_error or _ERROR_PATTERNS.search(text[:400]):
                events.append(
                    SignalEvent(
                        step_index=step.index,
                        kind=self.name,
                        severity=0.6,
                        evidence=f"tool {result.name or '?'} returned an error: {_clip(text)}",
                        detail={"tool": result.name},
                    )
                )
        return events


class MalformedArgumentsSignal:
    """A tool call's arguments could not be parsed as JSON.

    Strong evidence of `Invalid Invocation`, and unambiguous -- the adapter
    already tried to parse and failed.
    """

    name = "malformed_arguments"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        events = []
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.parse_error:
                    events.append(
                        SignalEvent(
                            step_index=step.index,
                            kind=self.name,
                            severity=0.8,
                            evidence=f"arguments to {call.name} are not valid JSON: "
                            f"{call.parse_error}",
                            detail={"tool": call.name, "raw": _clip(call.raw_arguments or "")},
                        )
                    )
        return events


class UnknownToolSignal:
    """The agent called a tool that is not in the declared schema set.

    Only fires when tool schemas are available; silence here means "unknown",
    not "clean".
    """

    name = "unknown_tool"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        known = _declared_tool_names(trajectory)
        if not known:
            return []

        events = []
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.name not in known:
                    events.append(
                        SignalEvent(
                            step_index=step.index,
                            kind=self.name,
                            severity=0.9,
                            evidence=f"called undeclared tool {call.name!r}",
                            detail={"tool": call.name},
                        )
                    )
        return events


class ArgumentSchemaSignal:
    """A tool call omits required arguments or passes undeclared ones.

    Requires JSON-Schema-shaped tool definitions. Kept out of the default battery
    because schemas are frequently absent or approximate in production traces,
    and a wrong schema turns this into a noise generator.
    """

    name = "argument_schema_violation"

    def __call__(self, trajectory: Trajectory) -> list[SignalEvent]:
        schemas = _schemas_by_name(trajectory)
        if not schemas:
            return []

        events = []
        for step in trajectory.steps:
            for call in step.tool_calls:
                schema = schemas.get(call.name)
                if not schema:
                    continue
                params = schema.get("parameters") or schema.get("input_schema") or {}
                properties = params.get("properties") or {}
                required = set(params.get("required") or [])
                provided = set(call.arguments)

                missing = required - provided
                if missing:
                    events.append(
                        SignalEvent(
                            step_index=step.index,
                            kind=self.name,
                            severity=0.75,
                            evidence=f"{call.name} missing required argument(s): "
                            f"{', '.join(sorted(missing))}",
                            detail={"tool": call.name, "missing": sorted(missing)},
                        )
                    )
                if properties:
                    unexpected = provided - set(properties)
                    if unexpected:
                        events.append(
                            SignalEvent(
                                step_index=step.index,
                                kind=self.name,
                                severity=0.4,
                                evidence=f"{call.name} passed undeclared argument(s): "
                                f"{', '.join(sorted(unexpected))}",
                                detail={"tool": call.name, "unexpected": sorted(unexpected)},
                            )
                        )
        return events


def _declared_tool_names(trajectory: Trajectory) -> set[str]:
    return set(_schemas_by_name(trajectory))


def _schemas_by_name(trajectory: Trajectory) -> dict[str, dict[str, Any]]:
    """Index declared tool schemas by name, tolerating the common wrapper shapes."""
    schemas: dict[str, dict[str, Any]] = {}
    for entry in trajectory.task.tool_schemas:
        if not isinstance(entry, dict):
            continue
        # OpenAI wraps the definition under "function"; Anthropic does not.
        body = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = body.get("name")
        if isinstance(name, str):
            schemas[name] = body
    return schemas


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
