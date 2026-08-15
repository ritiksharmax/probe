"""Adapter for the public AgentRx benchmark data.

Two structurally different domains land on the same canonical `Trajectory`:

* **tau-retail** -- OpenAI-style chat messages under `traj`, with `system`,
  `user`, `assistant` (optionally carrying `tool_calls`) and `tool` roles.
* **Magentic-One** -- a flat `[{role, content}]` list where `role` is the acting
  agent's name (`Orchestrator (thought)`, `WebSurfer`, `FileSurfer`, `human`).

The one thing that must not drift is step numbering. Ground-truth `step_number`
is **1-based with one step per message**, verified against the annotations
themselves: tau task 2's annotated steps 3 and 7 land on the assistant turn
calling `list_all_product_types` and the assistant turn claiming "11 available
T-shirt options", matching their annotated reasons; Magentic
`5f982798-...`'s steps 13 and 17 land on the two WebSurfer turns its annotations
describe. Note this means tau's leading `system` policy message is step 1.

tau messages happen to carry their own `index` field, which is 1-based position
on all 129 published runs. We cross-check against it and refuse to build a
trajectory if it ever disagrees, so a silent off-by-one becomes a loud error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory

# Magentic-One roles that denote the human's request rather than an agent turn.
_HUMAN_ROLES = {"human", "user"}


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None, str | None]:
    """Parse tool-call arguments, preserving the raw text when it will not parse.

    Malformed arguments are a failure signal in their own right (`Invalid
    Invocation`), so they are recorded rather than discarded.
    """
    if isinstance(raw, dict):
        return raw, None, None
    if raw is None or raw == "":
        return {}, None, None
    text = raw if isinstance(raw, str) else str(raw)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        return {}, text, str(exc)
    if isinstance(parsed, dict):
        return parsed, None, None
    return {"value": parsed}, text, None


def _tau_step(index: int, msg: dict[str, Any]) -> Step:
    role = msg.get("role", "assistant")
    content = msg.get("content") or ""  # assistant tool-call turns carry null content

    tool_calls: list[ToolCall] = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        args, raw_args, err = _parse_arguments(fn.get("arguments"))
        tool_calls.append(
            ToolCall(
                id=call.get("id"),
                name=fn.get("name", "<unknown>"),
                arguments=args,
                raw_arguments=raw_args,
                parse_error=err,
            )
        )

    tool_result = None
    if role == "tool":
        tool_result = ToolResult(
            call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
            content=content,
            is_error=_looks_like_error(content),
        )

    agent = {"assistant": "Assistant", "tool": msg.get("name") or "Tool"}.get(role)
    return Step(
        index=index,
        role=role,
        agent=agent,
        content=content,
        tool_calls=tool_calls,
        tool_result=tool_result,
    )


def _looks_like_error(content: str) -> bool:
    """Heuristic for a tool observation that reports failure.

    tau returns errors as ordinary string payloads rather than a status field,
    so there is nothing structural to key on.
    """
    head = content.strip()[:200].lower()
    return head.startswith("error") or "error:" in head or "not found" in head


def tau_trajectory(record: dict[str, Any]) -> Trajectory:
    """Build a `Trajectory` from one tau-retail run."""
    messages = record.get("traj", [])
    steps: list[Step] = []
    for position, msg in enumerate(messages, start=1):
        declared = msg.get("index")
        if declared is not None and int(declared) != position:
            raise ValueError(
                f"tau trajectory {record.get('task_id')!r}: message index {declared} "
                f"disagrees with 1-based position {position}; step numbering is not "
                "safe to derive from this record"
            )
        steps.append(_tau_step(position, msg))

    task = record.get("info", {}).get("task", {})
    policy = (
        messages[0].get("content", "") if messages and messages[0].get("role") == "system" else None
    )

    return Trajectory(
        trajectory_id=str(record.get("task_id")),
        domain="tau_retail",
        task=TaskContext(
            instruction=task.get("instruction", ""),
            user_id=task.get("user_id"),
            policy=policy,
            expected_actions=task.get("actions", []),
        ),
        steps=steps,
        reward=float(record["reward"]) if record.get("reward") is not None else None,
        metadata={"trial": record.get("trial"), "source": record.get("info", {}).get("source")},
    )


def magentic_trajectory(trajectory_id: str, messages: list[dict[str, Any]]) -> Trajectory:
    """Build a `Trajectory` from one Magentic-One run.

    `role` here is the acting agent, not a chat role, so it is preserved on
    `Step.agent` and mapped to a coarse canonical role. The distinction matters:
    `Orchestrator (thought)` turns are internal deliberation and dominate these
    trajectories, so signals must be able to tell them from turns that acted.
    """
    steps: list[Step] = []
    for position, msg in enumerate(messages, start=1):
        raw_role = (msg.get("role") or "").strip()
        role = "user" if raw_role.lower() in _HUMAN_ROLES else "agent"
        steps.append(
            Step(
                index=position,
                role=role,
                agent=raw_role or None,
                content=msg.get("content") or "",
            )
        )

    instruction = next((s.content for s in steps if s.role == "user"), "")
    return Trajectory(
        trajectory_id=trajectory_id,
        domain="magentic_one",
        task=TaskContext(instruction=instruction),
        steps=steps,
    )


def load_tau(path: str | Path) -> list[Trajectory]:
    """Load a tau-retail dataset file (`tau_dataset_failed` or `tau_dataset_full`)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [tau_trajectory(record) for record in raw]


def load_magentic(directory: str | Path) -> list[Trajectory]:
    """Load every Magentic-One trajectory in a directory, one file per trajectory."""
    directory = Path(directory)
    trajectories = []
    for path in sorted(directory.glob("*.json")):
        messages = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(messages, list):
            continue
        trajectories.append(magentic_trajectory(path.stem, messages))
    return trajectories
