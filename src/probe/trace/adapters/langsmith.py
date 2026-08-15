"""LangSmith adapter.

LangSmith models a run as a tree of **runs**, each with a `run_type` — `llm`,
`tool`, `chain`, `retriever`, `agent`. Only `llm` and `tool` runs carry
conversation content; chains are structure. Runs are walked in start-time order
and turned into steps.

Accepts a list of runs, a `{"runs": [...]}` envelope, a single run with nested
`child_runs`, or JSONL of any of those.

LangSmith message dicts come in two flavours: the plain OpenAI shape
(`{"role": ..., "content": ...}`) and LangChain's serialized form
(`{"type": "ai"|"human"|"system", "data": {"content": ...}}`, or the `lc`/`id`
serialization where the class name carries the role). All three are handled,
since which one you get depends on the LangChain version that wrote the trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory

# LangChain message class / type names -> canonical chat roles.
_ROLE_ALIASES = {
    "human": "user",
    "humanmessage": "user",
    "ai": "assistant",
    "aimessage": "assistant",
    "aimessagechunk": "assistant",
    "system": "system",
    "systemmessage": "system",
    "tool": "tool",
    "toolmessage": "tool",
    "function": "tool",
    "functionmessage": "tool",
    "chat": "assistant",
}


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
    value = _as_json(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "result", "answer"):
            if key in value:
                return _text(value[key])
        return json.dumps(value, default=str)
    if isinstance(value, list):
        return "\n".join(t for t in (_text(v) for v in value) if t)
    return str(value)


def _normalize_message(message: Any) -> dict[str, Any] | None:
    """Coerce the several LangChain message serializations into one shape."""
    message = _as_json(message)
    if not isinstance(message, dict):
        return None

    # Plain OpenAI shape.
    if "role" in message:
        role = str(message["role"]).lower()
        return {
            "role": _ROLE_ALIASES.get(role, role),
            "content": _text(message.get("content")),
            "tool_calls": message.get("tool_calls") or message.get("toolCalls"),
        }

    # LangChain serialized: {"type": "ai", "data": {...}}
    if "data" in message and isinstance(message["data"], dict):
        data = message["data"]
        role = str(message.get("type") or data.get("type") or "assistant").lower()
        additional = data.get("additional_kwargs") or {}
        return {
            "role": _ROLE_ALIASES.get(role, "assistant"),
            "content": _text(data.get("content")),
            "tool_calls": data.get("tool_calls") or additional.get("tool_calls"),
        }

    # LangChain `lc` serialization: role comes from the class name in `id`.
    if "id" in message and isinstance(message["id"], list):
        cls = str(message["id"][-1]).lower()
        kwargs = message.get("kwargs") or {}
        return {
            "role": _ROLE_ALIASES.get(cls, "assistant"),
            "content": _text(kwargs.get("content")),
            "tool_calls": kwargs.get("tool_calls"),
        }
    return None


def _messages(payload: Any) -> list[dict[str, Any]]:
    """Extract messages from a run's `inputs` or `outputs`."""
    payload = _as_json(payload)
    found: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        # LLM runs: inputs.messages is often a list-of-lists (one per generation).
        for key in ("messages", "input", "prompts"):
            if key in payload:
                value = _as_json(payload[key])
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, list):
                            found.extend(m for m in map(_normalize_message, item) if m)
                        elif (m := _normalize_message(item)) is not None:
                            found.append(m)
                    if found:
                        return found
        # LLM outputs: generations -> [{message|text}]
        generations = _as_json(payload.get("generations"))
        if isinstance(generations, list):
            for group in generations:
                items = group if isinstance(group, list) else [group]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if (m := _normalize_message(item.get("message"))) is not None:
                        found.append(m)
                    elif item.get("text"):
                        found.append({"role": "assistant", "content": _text(item["text"])})
            if found:
                return found
        if (m := _normalize_message(payload)) is not None:
            return [m]
    elif isinstance(payload, list):
        found.extend(m for m in map(_normalize_message, payload) if m)
    return found


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    raw = _as_json(message.get("tool_calls")) or []
    if isinstance(raw, dict):
        raw = [raw]
    calls = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name") or entry.get("name") or "<unknown>"
        args = _as_json(fn.get("arguments", entry.get("args")))
        if isinstance(args, dict):
            calls.append(ToolCall(id=entry.get("id"), name=str(name), arguments=args))
        else:
            calls.append(
                ToolCall(
                    id=entry.get("id"),
                    name=str(name),
                    arguments={},
                    raw_arguments=None if args is None else str(args),
                    parse_error=None if args is None else "arguments are not a JSON object",
                )
            )
    return calls


def _flatten(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten `child_runs` trees into a single list."""
    out: list[dict[str, Any]] = []
    stack = list(runs)
    while stack:
        run = stack.pop()
        if not isinstance(run, dict):
            continue
        out.append(run)
        children = run.get("child_runs") or run.get("childRuns") or []
        if isinstance(children, list):
            stack.extend(c for c in children if isinstance(c, dict))
    return out


def runs_to_trajectory(runs: list[dict[str, Any]], trajectory_id: str) -> Trajectory:
    """Build a trajectory from one trace's runs."""
    ordered = sorted(runs, key=lambda r: str(r.get("start_time") or r.get("startTime") or ""))

    steps: list[Step] = []
    instruction = ""
    policy: str | None = None
    seen_system = False

    def add(role: str, **kwargs: Any) -> None:
        steps.append(Step(index=len(steps) + 1, role=role, **kwargs))

    for run in ordered:
        run_type = str(run.get("run_type") or run.get("runType") or "").lower()
        name = str(run.get("name") or "")

        if run_type == "tool" or run_type == "retriever":
            text = _text(run.get("outputs"))
            error = run.get("error")
            add(
                "tool",
                content=text,
                agent=name or "tool",
                tool_result=ToolResult(
                    name=name or None,
                    content=text or _text(error),
                    is_error=bool(error),
                ),
            )
            continue

        if run_type != "llm":
            continue

        for message in _messages(run.get("inputs")):
            role = message["role"]
            if role == "system":
                if seen_system:
                    continue
                seen_system = True
                policy = policy or message["content"]
            if role == "user" and not instruction:
                instruction = message["content"]
            add(role, content=message["content"], tool_calls=_tool_calls(message))

        for message in _messages(run.get("outputs")):
            add(
                "assistant",
                content=message["content"],
                tool_calls=_tool_calls(message),
                agent=name or None,
            )

        if run.get("error"):
            add(
                "tool",
                content=_text(run["error"]),
                agent=name or None,
                tool_result=ToolResult(
                    name=name or None, content=_text(run["error"]), is_error=True
                ),
            )

    return Trajectory(
        trajectory_id=trajectory_id,
        domain="langsmith",
        task=TaskContext(instruction=instruction, policy=policy),
        steps=steps,
    )


def load_langsmith(path: str | Path) -> list[Trajectory]:
    """Load a LangSmith export into trajectories, one per trace id."""
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

    runs: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, list):
            runs.extend(_flatten([r for r in payload if isinstance(r, dict)]))
        elif isinstance(payload, dict):
            if isinstance(payload.get("runs"), list):
                runs.extend(_flatten([r for r in payload["runs"] if isinstance(r, dict)]))
            else:
                runs.extend(_flatten([payload]))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = str(run.get("trace_id") or run.get("traceId") or run.get("id") or "langsmith-trace")
        grouped.setdefault(key, []).append(run)

    return [runs_to_trajectory(group, key) for key, group in grouped.items()]
