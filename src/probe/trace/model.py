"""Canonical trace types.

The whole library speaks `Trajectory`. Every adapter's job is to produce one, and
every signal, the detector, the evidence filter and the judge consume one.

Step indexing is the load-bearing invariant here: `Step.index` is **1-based**, one
step per agent/tool message, and it must line up exactly with the step numbering
used by whatever ground truth we are scored against. For the AgentRx benchmark
that means every message counts as a step -- including the leading `system`
policy message in tau-bench, which is step 1. Getting this wrong shifts every
localization number without breaking anything loudly, so `Trajectory` validates
the invariant on construction.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Role = Literal["system", "user", "assistant", "tool", "agent"]


class ToolCall(BaseModel):
    """A single tool invocation requested by the agent."""

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the raw arguments could not be parsed as JSON. Kept rather than
    # dropped because a malformed argument blob is itself a strong failure signal.
    raw_arguments: str | None = None
    parse_error: str | None = None


class ToolResult(BaseModel):
    """The observation returned to the agent for a tool call."""

    call_id: str | None = None
    name: str | None = None
    content: str = ""
    is_error: bool = False


class TaskContext(BaseModel):
    """What the agent was asked to do, and the rules it was meant to follow.

    Populated as far as each source allows; the judge degrades gracefully when
    fields are missing.
    """

    instruction: str = ""
    user_id: str | None = None
    policy: str | None = None
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    # Reference action sequence, when the source provides one. Never fed to the
    # judge -- PROBE's premise is diagnosis without ground-truth trajectories.
    # Retained only so the benchmark harness can report on it.
    expected_actions: list[dict[str, Any]] = Field(default_factory=list)


class Step(BaseModel):
    """One message in the trajectory."""

    index: int = Field(ge=1, description="1-based position in the trajectory")
    role: Role
    # The concrete actor, preserved verbatim from the source: "Assistant",
    # "WebSurfer", "Orchestrator (thought)". Multi-agent domains need this to
    # attribute a failure to an agent, and role alone flattens it away.
    agent: str | None = None
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_thought(self) -> bool:
        """Internal reasoning that took no action on the world.

        Magentic-One trajectories are dominated by `Orchestrator (thought)`
        entries; signals weight these down so localization is not dragged toward
        deliberation and away from the acting step.
        """
        return bool(self.agent and "thought" in self.agent.lower())

    def summary(self, width: int = 160) -> str:
        """One-line rendering used in evidence windows and prompts."""
        who = self.agent or self.role
        if self.tool_calls:
            calls = ", ".join(f"{c.name}({_compact_json(c.arguments)})" for c in self.tool_calls)
            body = f"-> {calls}"
        elif self.tool_result is not None:
            tag = "ERROR " if self.tool_result.is_error else ""
            body = f"<- {tag}{self.tool_result.content}"
        else:
            body = self.content
        body = " ".join(body.split())
        if len(body) > width:
            body = body[: width - 1] + "…"
        return f"[{self.index}] {who}: {body}"


class Trajectory(BaseModel):
    """A complete agent execution."""

    trajectory_id: str
    domain: str = "unknown"
    task: TaskContext = Field(default_factory=TaskContext)
    steps: list[Step] = Field(default_factory=list)
    # Ground-truth outcome when the source records one (tau-bench's `reward`).
    # Used only to score detection; never visible to signals or the judge.
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_contiguous_1based(self) -> Trajectory:
        for expected, step in enumerate(self.steps, start=1):
            if step.index != expected:
                raise ValueError(
                    f"trajectory {self.trajectory_id!r}: step indices must be contiguous "
                    f"and 1-based; expected {expected} at position {expected - 1}, "
                    f"got {step.index}"
                )
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def step(self, index: int) -> Step:
        """Fetch by 1-based index, the same numbering ground truth uses."""
        if not 1 <= index <= len(self.steps):
            raise IndexError(
                f"step {index} out of range for trajectory {self.trajectory_id!r} "
                f"with {len(self.steps)} steps"
            )
        return self.steps[index - 1]

    def render(self, first: int = 1, last: int | None = None, width: int = 160) -> str:
        """Render an inclusive 1-based step range as text for a prompt."""
        last = len(self.steps) if last is None else last
        first = max(1, first)
        last = min(len(self.steps), last)
        return "\n".join(self.step(i).summary(width) for i in range(first, last + 1))


def _compact_json(obj: Any, limit: int = 80) -> str:
    import json

    try:
        text = json.dumps(obj, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(obj)
    return text if len(text) <= limit else text[: limit - 1] + "…"
