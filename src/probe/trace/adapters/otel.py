"""OpenTelemetry adapter: GenAI semantic conventions and OpenInference.

Production traces do not arrive as tidy message arrays; they arrive as a bag of
spans with parent pointers, and the conversation has to be reconstructed from
them. This adapter handles the two conventions that actually appear in the wild:

* **OTel GenAI semconv** — `gen_ai.*` attributes, with messages carried either as
  span events (`gen_ai.user.message`, `gen_ai.choice`, …) or, in the newer
  convention, as `gen_ai.input.messages` / `gen_ai.output.messages` attributes.
* **OpenInference** (Arize/Phoenix) — `openinference.span.kind` plus indexed
  `llm.input_messages.N.message.role` style attributes.

Both are supported by the same walker because instrumentation libraries mix them
freely, and a trace that half-matches one convention is the normal case rather
than the exception.

Input may be OTLP/JSON (`{"resourceSpans": [...]}`), a flat list of spans, or
JSONL with one span per line. Attribute values may be OTLP-wrapped
(`{"stringValue": "x"}`) or plain — exporters disagree, so both are unwrapped.

The ordering rule is the load-bearing part: steps come out in **start-time
order**, so `Step.index` stays 1-based and chronological and everything
downstream — signals, evidence windows, the judge — keeps working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory

# Span kinds that carry a model call, across both conventions.
_LLM_KINDS = {"llm", "chat", "text_completion", "generate_content"}
_TOOL_KINDS = {"tool", "execute_tool"}


def _unwrap(value: Any) -> Any:
    """Unwrap an OTLP AnyValue, or pass a plain value through."""
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "arrayValue" in value:
        return [_unwrap(v) for v in (value["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in value:
        return {
            kv.get("key"): _unwrap(kv.get("value"))
            for kv in (value["kvlistValue"] or {}).get("values", [])
        }
    return value


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    """Normalize a span's attributes to a flat dict, whatever the shape."""
    raw = span.get("attributes")
    if isinstance(raw, list):
        return {kv.get("key"): _unwrap(kv.get("value")) for kv in raw if isinstance(kv, dict)}
    if isinstance(raw, dict):
        return {k: _unwrap(v) for k, v in raw.items()}
    return {}


def _start_ns(span: dict[str, Any]) -> int:
    """Start timestamp in nanoseconds, tolerating the several field names in use."""
    for key in ("startTimeUnixNano", "start_time_unix_nano", "startTime", "start_time"):
        if key in span:
            try:
                return int(span[key])
            except (TypeError, ValueError):
                continue
    return 0


def _iter_spans(payload: Any) -> list[dict[str, Any]]:
    """Pull spans out of OTLP/JSON, a flat list, or a nested export."""
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]

    if not isinstance(payload, dict):
        return []

    if "resourceSpans" in payload or "resource_spans" in payload:
        spans: list[dict[str, Any]] = []
        for resource in payload.get("resourceSpans") or payload.get("resource_spans") or []:
            scopes = resource.get("scopeSpans") or resource.get("scope_spans") or []
            for scope in scopes:
                spans.extend(s for s in (scope.get("spans") or []) if isinstance(s, dict))
        return spans

    for key in ("spans", "data", "batches"):
        if key in payload:
            return _iter_spans(payload[key])

    # A bare span, which is what JSONL exports emit one per line.
    if any(k in payload for k in ("traceId", "trace_id", "spanId", "span_id")) or (
        "name" in payload and any(k in payload for k in ("attributes", "startTimeUnixNano"))
    ):
        return [payload]
    return []


def _span_kind(span: dict[str, Any], attrs: dict[str, Any]) -> str:
    """Classify a span as an llm call, a tool call, or neither."""
    kind = str(attrs.get("openinference.span.kind") or "").lower()
    if kind in _LLM_KINDS:
        return "llm"
    if kind in _TOOL_KINDS:
        return "tool"

    operation = str(attrs.get("gen_ai.operation.name") or "").lower()
    if operation in _TOOL_KINDS or "gen_ai.tool.name" in attrs:
        return "tool"
    if operation in _LLM_KINDS or "gen_ai.request.model" in attrs:
        return "llm"

    if "tool.name" in attrs:
        return "tool"
    name = str(span.get("name") or "").lower()
    if name.startswith("execute_tool") or name.startswith("tool."):
        return "tool"
    # A span carrying messages is a model call, whether or not the
    # instrumentation also bothered to set an operation name or model.
    if any(
        k.startswith(("llm.input_messages", "llm.output_messages"))
        or k in ("gen_ai.input.messages", "gen_ai.output.messages")
        for k in attrs
    ):
        return "llm"
    return "other"


def _parse_maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _indexed_messages(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Collect OpenInference `prefix.N.message.*` attributes into messages."""
    collected: dict[int, dict[str, Any]] = {}
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :].lstrip(".")
        index_text, _, field = rest.partition(".")
        if not index_text.isdigit():
            continue
        message = collected.setdefault(int(index_text), {})
        field = field.replace("message.", "", 1)
        message[field] = value
    return [collected[i] for i in sorted(collected)]


def _messages_from_span(span: dict[str, Any], attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract chat messages from a span, whichever convention it follows."""
    messages: list[dict[str, Any]] = []

    # Newer OTel GenAI: whole message arrays as attributes.
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        value = _parse_maybe_json(attrs.get(key))
        if isinstance(value, list):
            messages.extend(m for m in value if isinstance(m, dict))

    # OpenInference indexed attributes.
    for prefix in ("llm.input_messages", "llm.output_messages"):
        messages.extend(_indexed_messages(attrs, prefix))

    # Older OTel GenAI: messages as span events.
    for event in span.get("events") or []:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "")
        if not name.startswith("gen_ai."):
            continue
        body = _attributes(event)
        payload = _parse_maybe_json(body.get("gen_ai.event.content") or body.get("content"))
        if isinstance(payload, dict):
            message = dict(payload)
        elif isinstance(payload, str):
            message = {"content": payload}
        else:
            message = {}
        if "role" not in message:
            # gen_ai.user.message -> user, gen_ai.choice -> assistant
            part = name.split(".")[1] if len(name.split(".")) > 2 else "assistant"
            message["role"] = "assistant" if name.endswith("choice") else part
        messages.append(message)

    return messages


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Build tool calls from a message, across the shapes instrumentations emit."""
    calls: list[ToolCall] = []
    raw = message.get("tool_calls") or message.get("toolCalls") or []
    if isinstance(raw, str):
        raw = _parse_maybe_json(raw) or []
    if isinstance(raw, dict):
        raw = [raw]

    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = function.get("name") or entry.get("name") or "<unknown>"
        arguments = function.get("arguments", entry.get("arguments"))
        parsed = _parse_maybe_json(arguments)
        if isinstance(parsed, dict):
            calls.append(ToolCall(id=entry.get("id"), name=str(name), arguments=parsed))
        else:
            calls.append(
                ToolCall(
                    id=entry.get("id"),
                    name=str(name),
                    arguments={},
                    raw_arguments=None if arguments is None else str(arguments),
                    parse_error=None if arguments is None else "arguments are not a JSON object",
                )
            )
    return calls


def _content(message: dict[str, Any]) -> str:
    """Flatten message content, which may be a string or a content-block list."""
    value = message.get("content", message.get("message.content", ""))
    value = _parse_maybe_json(value)
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return "" if value is None else str(value)


def spans_to_trajectory(
    spans: list[dict[str, Any]], trajectory_id: str | None = None
) -> Trajectory:
    """Reconstruct a trajectory from a list of spans."""
    ordered = sorted(spans, key=_start_ns)

    steps: list[Step] = []
    instruction = ""
    policy: str | None = None
    tool_schemas: list[dict[str, Any]] = []
    seen_roles: set[str] = set()

    def add(role: str, **kwargs: Any) -> None:
        steps.append(Step(index=len(steps) + 1, role=role, **kwargs))

    for span in ordered:
        attrs = _attributes(span)
        kind = _span_kind(span, attrs)

        if not tool_schemas:
            declared = _parse_maybe_json(
                attrs.get("llm.tools") or attrs.get("gen_ai.request.tools")
            )
            if isinstance(declared, list):
                tool_schemas = [t for t in declared if isinstance(t, dict)]

        if kind == "tool":
            name = str(
                attrs.get("gen_ai.tool.name")
                or attrs.get("tool.name")
                or span.get("name")
                or "tool"
            )
            output = _parse_maybe_json(
                attrs.get("output.value")
                or attrs.get("gen_ai.tool.message")
                or attrs.get("tool.result")
                or ""
            )
            text = output if isinstance(output, str) else json.dumps(output, default=str)
            status = span.get("status") or {}
            is_error = str(status.get("code", "")).upper().endswith("ERROR")
            add(
                "tool",
                content=text,
                agent=name,
                tool_result=ToolResult(
                    call_id=attrs.get("gen_ai.tool.call.id") or attrs.get("tool.call.id"),
                    name=name,
                    content=text,
                    is_error=is_error,
                ),
            )
            continue

        if kind != "llm":
            continue

        for message in _messages_from_span(span, attrs):
            role = str(message.get("role") or "assistant").lower()
            content = _content(message)
            calls = _tool_calls(message)

            if role == "system":
                # The system prompt is policy, and repeating it on every model
                # call is normal instrumentation noise — keep the first only.
                if "system" in seen_roles:
                    continue
                policy = policy or content
            if role == "user" and not instruction:
                instruction = content
            seen_roles.add(role)

            if role not in {"system", "user", "assistant", "tool"}:
                role = "agent"
            if not content and not calls and role != "system":
                continue
            add(role, content=content, tool_calls=calls, agent=attrs.get("gen_ai.agent.name"))

    trace_id = trajectory_id or next(
        (
            str(s.get("traceId") or s.get("trace_id"))
            for s in ordered
            if s.get("traceId") or s.get("trace_id")
        ),
        "otel-trace",
    )
    return Trajectory(
        trajectory_id=trace_id,
        domain="otel",
        task=TaskContext(instruction=instruction, policy=policy, tool_schemas=tool_schemas),
        steps=steps,
    )


def load_otel(path: str | Path) -> list[Trajectory]:
    """Load OTLP JSON/JSONL and group spans into one trajectory per trace id."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    spans: list[dict[str, Any]] = []
    try:
        spans = _iter_spans(json.loads(text))
    except (TypeError, ValueError):
        # JSONL: one span, or one OTLP envelope, per line.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                spans.extend(_iter_spans(json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: could not parse OTLP payload: {exc}") from exc

    grouped: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        key = str(span.get("traceId") or span.get("trace_id") or "otel-trace")
        grouped.setdefault(key, []).append(span)

    return [spans_to_trajectory(group, trajectory_id=key) for key, group in grouped.items()]
