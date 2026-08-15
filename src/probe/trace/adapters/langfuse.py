"""Langfuse adapter.

Langfuse models a run as a **trace** with nested **observations**, each typed
`GENERATION` (a model call), `SPAN` (anything else, commonly a tool), or `EVENT`.
The conversation is reconstructed by walking observations in start-time order and
turning each into steps.

Accepts an export from `GET /api/public/traces` (a trace object with an
`observations` array), a bare list of observations, or JSONL of either. Field
names are accepted in both camelCase and snake_case, because the REST API and the
Python SDK disagree about which they emit.

As with the OTel adapter, the invariant that matters downstream is ordering:
steps come out chronological and 1-based.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory

_GENERATION = {"GENERATION", "generation"}
_TOOL_HINTS = ("tool", "function", "retriever", "api")


def _get(obj: dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present key among `names`, tolerating camel/snake disagreement."""
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def _as_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except (TypeError, ValueError):
                return value
    return value


def _text(value: Any) -> str:
    """Flatten an input/output payload to text."""
    value = _as_json(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "result", "value"):
            if key in value:
                return _text(value[key])
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value if v is not None)
    return str(value)


def _messages(value: Any) -> list[dict[str, Any]]:
    """Pull a chat message list out of a Langfuse input payload."""
    value = _as_json(value)
    if isinstance(value, dict):
        for key in ("messages", "input", "prompt"):
            if key in value:
                inner = _as_json(value[key])
                if isinstance(inner, list):
                    return [m for m in inner if isinstance(m, dict)]
        return [value] if "role" in value else []
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict) and "role" in m]
    return []


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    raw = _as_json(_get(message, "tool_calls", "toolCalls", default=[])) or []
    if isinstance(raw, dict):
        raw = [raw]
    calls = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        args = _as_json(fn.get("arguments", entry.get("args")))
        if isinstance(args, dict):
            calls.append(
                ToolCall(id=entry.get("id"), name=str(fn.get("name", "<unknown>")), arguments=args)
            )
        else:
            calls.append(
                ToolCall(
                    id=entry.get("id"),
                    name=str(fn.get("name", "<unknown>")),
                    arguments={},
                    raw_arguments=None if args is None else str(args),
                    parse_error=None if args is None else "arguments are not a JSON object",
                )
            )
    return calls


def _is_tool(observation: dict[str, Any]) -> bool:
    kind = str(_get(observation, "type", default="")).upper()
    if kind in _GENERATION:
        return False
    name = str(_get(observation, "name", default="")).lower()
    return any(hint in name for hint in _TOOL_HINTS) or kind == "TOOL"


def observations_to_trajectory(
    observations: list[dict[str, Any]],
    trajectory_id: str,
    trace: dict[str, Any] | None = None,
) -> Trajectory:
    """Build a trajectory from one trace's observations."""
    ordered = sorted(
        observations, key=lambda o: str(_get(o, "startTime", "start_time", default=""))
    )

    steps: list[Step] = []
    instruction = ""
    policy: str | None = None
    seen_system = False

    def add(role: str, **kwargs: Any) -> None:
        steps.append(Step(index=len(steps) + 1, role=role, **kwargs))

    for observation in ordered:
        kind = str(_get(observation, "type", default="")).upper()
        name = str(_get(observation, "name", default="") or "")

        if _is_tool(observation):
            output = _get(observation, "output", default="")
            text = _text(output)
            level = str(_get(observation, "level", default="")).upper()
            status = _text(_get(observation, "statusMessage", "status_message", default=""))
            add(
                "tool",
                content=text,
                agent=name or "tool",
                tool_result=ToolResult(
                    name=name or None,
                    content=text,
                    is_error=level in {"ERROR", "WARNING"} or bool(status),
                ),
            )
            continue

        if kind not in _GENERATION and not _messages(_get(observation, "input")):
            continue

        for message in _messages(_get(observation, "input")):
            role = str(message.get("role", "user")).lower()
            if role == "system":
                if seen_system:
                    continue
                seen_system = True
                policy = policy or _text(message.get("content"))
            if role == "user" and not instruction:
                instruction = _text(message.get("content"))
            if role not in {"system", "user", "assistant", "tool"}:
                role = "agent"
            add(role, content=_text(message.get("content")), tool_calls=_tool_calls(message))

        output = _as_json(_get(observation, "output"))
        out_messages = _messages(output)
        if out_messages:
            for message in out_messages:
                add(
                    "assistant",
                    content=_text(message.get("content")),
                    tool_calls=_tool_calls(message),
                    agent=name or None,
                )
        elif output is not None and _text(output):
            calls = _tool_calls(output) if isinstance(output, dict) else []
            add("assistant", content=_text(output), tool_calls=calls, agent=name or None)

    if trace and not instruction:
        instruction = _text(_get(trace, "input", default=""))

    return Trajectory(
        trajectory_id=trajectory_id,
        domain="langfuse",
        task=TaskContext(instruction=instruction, policy=policy),
        steps=steps,
        metadata={"name": _get(trace or {}, "name", default=None)},
    )


def load_langfuse(path: str | Path) -> list[Trajectory]:
    """Load a Langfuse export into trajectories, one per trace."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    payloads: list[Any] = []
    try:
        payloads = [json.loads(text)]
    except (TypeError, ValueError):
        for line in text.splitlines():
            line = line.strip()
            if line:
                payloads.append(json.loads(line))

    traces: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}

    def absorb(payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                absorb(item)
            return
        if not isinstance(payload, dict):
            return
        if "data" in payload and isinstance(payload["data"], list):
            for item in payload["data"]:
                absorb(item)
            return

        observations = _get(payload, "observations", default=None)
        if isinstance(observations, list):
            trace_id = str(_get(payload, "id", "traceId", "trace_id", default="langfuse-trace"))
            traces[trace_id] = payload
            grouped.setdefault(trace_id, []).extend(o for o in observations if isinstance(o, dict))
            return

        # A bare observation.
        trace_id = str(_get(payload, "traceId", "trace_id", default="langfuse-trace"))
        grouped.setdefault(trace_id, []).append(payload)

    for payload in payloads:
        absorb(payload)

    return [
        observations_to_trajectory(obs, trace_id, traces.get(trace_id))
        for trace_id, obs in grouped.items()
    ]
