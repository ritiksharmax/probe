"""A tiny agent with deliberately injected failures.

Run it to produce a trace, then diagnose the trace:

    uv run python examples/demo_agent.py
    uv run probe detect examples/out/traces.jsonl          # no LLM needed
    uv run probe analyze examples/out/traces.jsonl         # needs model access

No LLM is involved in *generating* these traces — they are scripted, so the
injected failure step is known exactly. That makes this both a demo and the
end-to-end fixture the test suite asserts against: if the pipeline stops finding
a failure we planted ourselves, something is broken.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe.detect import Detector
from probe.signals import (
    Constraint,
    ConstraintSignal,
    default_signals,
    require_before,
    tool_call_constraint,
)
from probe.trace.io import write_jsonl
from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory

TOOL_SCHEMAS = [
    {
        "name": "search_orders",
        "parameters": {"properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
    },
    {
        "name": "get_order",
        "parameters": {"properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "refund_order",
        "parameters": {
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    },
]

POLICY = (
    "You are a support agent. Look up the user's orders before acting on one. "
    "Never issue a refund larger than the order total."
)


def _system(index: int = 1) -> Step:
    return Step(index=index, role="system", content=POLICY)


def _call(index: int, name: str, args: dict, call_id: str) -> Step:
    return Step(
        index=index,
        role="assistant",
        agent="Assistant",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
    )


def _result(index: int, name: str, call_id: str, content: str, error: bool = False) -> Step:
    return Step(
        index=index,
        role="tool",
        content=content,
        tool_result=ToolResult(call_id=call_id, name=name, content=content, is_error=error),
    )


def healthy() -> Trajectory:
    """A run that succeeds. The detector should leave it alone."""
    return Trajectory(
        trajectory_id="demo-healthy",
        domain="demo",
        reward=1.0,
        task=TaskContext(
            instruction="Refund my last order.", policy=POLICY, tool_schemas=TOOL_SCHEMAS
        ),
        steps=[
            _system(),
            Step(index=2, role="user", content="I'd like a refund for my last order."),
            _call(3, "search_orders", {"user_id": "u_42"}, "c1"),
            _result(4, "search_orders", "c1", '{"orders": ["o_9"]}'),
            _call(5, "get_order", {"order_id": "o_9"}, "c2"),
            _result(6, "get_order", "c2", '{"order_id": "o_9", "total": 25.0}'),
            _call(7, "refund_order", {"order_id": "o_9", "amount": 25.0}, "c3"),
            _result(8, "refund_order", "c3", '{"status": "refunded", "amount": 25.0}'),
            Step(
                index=9,
                role="assistant",
                agent="Assistant",
                content="Refunded $25.00. Anything else?",
            ),
        ],
        metadata={"injected_failure_step": None},
    )


def retry_loop() -> Trajectory:
    """Injected failure at step 3: the agent loops on an identical failing call."""
    steps = [
        _system(),
        Step(index=2, role="user", content="Show me my orders."),
    ]
    index = 3
    for i in range(4):
        steps.append(_call(index, "search_orders", {"user_id": "u_unknown"}, f"c{i}"))
        steps.append(
            _result(index + 1, "search_orders", f"c{i}", "Error: user not found", error=True)
        )
        index += 2
    steps.append(
        Step(
            index=index,
            role="assistant",
            agent="Assistant",
            content="I'm sorry, I am unable to complete this request.",
        )
    )
    return Trajectory(
        trajectory_id="demo-retry-loop",
        domain="demo",
        reward=0.0,
        task=TaskContext(
            instruction="Show me my orders.", policy=POLICY, tool_schemas=TOOL_SCHEMAS
        ),
        steps=steps,
        metadata={"injected_failure_step": 3},
    )


def policy_violation() -> Trajectory:
    """Injected failure at step 5: refunds more than the order total."""
    return Trajectory(
        trajectory_id="demo-policy-violation",
        domain="demo",
        reward=0.0,
        task=TaskContext(instruction="Refund order o_7.", policy=POLICY, tool_schemas=TOOL_SCHEMAS),
        steps=[
            _system(),
            Step(index=2, role="user", content="Please refund order o_7."),
            _call(3, "get_order", {"order_id": "o_7"}, "c1"),
            _result(4, "get_order", "c1", '{"order_id": "o_7", "total": 20.0}'),
            # The failure: 200.0 against a 20.0 order, in breach of the policy.
            _call(5, "refund_order", {"order_id": "o_7", "amount": 200.0}, "c2"),
            _result(6, "refund_order", "c2", '{"status": "refunded", "amount": 200.0}'),
            Step(index=7, role="assistant", agent="Assistant", content="Refunded $200.00."),
        ],
        metadata={"injected_failure_step": 5},
    )


def unknown_tool() -> Trajectory:
    """Injected failure at step 3: calls a tool that was never declared."""
    return Trajectory(
        trajectory_id="demo-unknown-tool",
        domain="demo",
        reward=0.0,
        task=TaskContext(
            instruction="Cancel my subscription.", policy=POLICY, tool_schemas=TOOL_SCHEMAS
        ),
        steps=[
            _system(),
            Step(index=2, role="user", content="Cancel my subscription please."),
            _call(3, "cancel_subscription", {"user_id": "u_42"}, "c1"),
            _result(4, "cancel_subscription", "c1", "Error: no such tool", error=True),
            Step(
                index=5,
                role="assistant",
                agent="Assistant",
                content="Unfortunately I cannot cancel subscriptions.",
            ),
        ],
        metadata={"injected_failure_step": 3},
    )


def all_trajectories() -> list[Trajectory]:
    return [healthy(), retry_loop(), policy_violation(), unknown_tool()]


def demo_constraints() -> list[Constraint]:
    """The domain rules from POLICY, made checkable.

    Without these, `demo-policy-violation` is invisible: every tool call
    succeeds, nothing errors, and the agent simply refunds ten times the order
    total. That class of failure is exactly what the error- and loop-based
    signals cannot see, and it is why constraints exist.
    """

    def refund_within_total(call, step, trajectory) -> str | None:
        """Refund must not exceed the total this order was reported to have."""
        order_id = call.arguments.get("order_id")
        amount = call.arguments.get("amount")
        if order_id is None or amount is None:
            return None

        # Find the most recent get_order result for this order, before this step.
        total = None
        for earlier in trajectory.steps:
            if earlier.index >= step.index or earlier.tool_result is None:
                continue
            if earlier.tool_result.name != "get_order":
                continue
            try:
                payload = json.loads(earlier.tool_result.content)
            except (TypeError, ValueError):
                continue
            if payload.get("order_id") == order_id and "total" in payload:
                total = float(payload["total"])

        if total is not None and float(amount) > total:
            return (
                f"refunded {float(amount):.2f} against order {order_id} "
                f"whose reported total is {total:.2f}"
            )
        return None

    return [
        tool_call_constraint(
            name="refund_within_order_total",
            description="Never issue a refund larger than the order total.",
            tool="refund_order",
            check=refund_within_total,
            severity=0.9,
        ),
        require_before(
            name="look_up_before_acting",
            description="Look up the user's orders before acting on one.",
            prerequisite="get_order",
            dependent="refund_order",
        ),
    ]


def main() -> None:
    out = Path(__file__).parent / "out" / "traces.jsonl"
    trajectories = all_trajectories()
    count = write_jsonl(out, trajectories)
    print(f"wrote {count} trajectories to {out}\n")

    # Show the difference constraints make, since it is the point of the example.
    plain = Detector()
    with_constraints = Detector(signals=[*default_signals(), ConstraintSignal(demo_constraints())])

    print(f"{'trajectory':<24} {'injected':>8}  {'default':>8}  {'+constraints':>12}")
    for trajectory in trajectories:
        injected = trajectory.metadata.get("injected_failure_step")
        a, b = plain(trajectory), with_constraints(trajectory)
        print(
            f"{trajectory.trajectory_id:<24} {str(injected):>8}  "
            f"{a.confidence:>8.2f}  {b.confidence:>12.2f}"
        )

    print("\nnext:")
    print(f"  uv run probe detect {out}      # no LLM needed")
    print(f"  uv run probe analyze {out}     # needs model access")


if __name__ == "__main__":
    main()
