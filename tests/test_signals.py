"""Unit tests for the signal battery.

Each signal is tested on a synthetic trajectory built to trigger exactly it, plus
a negative case, because a signal that fires on everything is worse than no
signal at all.
"""

from __future__ import annotations

import pytest

from probe.signals import (
    ArgumentSchemaSignal,
    BudgetAnomaly,
    CorpusStats,
    IncompleteOutcomeSignal,
    MalformedArgumentsSignal,
    NoProgressSignal,
    OscillationSignal,
    RefusalSignal,
    RepeatedCallSignal,
    StallSignal,
    ToolErrorSignal,
    UnknownToolSignal,
    default_signals,
    run_signals,
)
from probe.signals.base import SignalEvent
from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory


def build(*steps: Step, task: TaskContext | None = None) -> Trajectory:
    return Trajectory(trajectory_id="t", steps=list(steps), task=task or TaskContext())


def assistant(index: int, content: str = "", calls: list[tuple[str, dict]] | None = None) -> Step:
    return Step(
        index=index,
        role="assistant",
        agent="Assistant",
        content=content,
        tool_calls=[ToolCall(id=f"c{index}", name=n, arguments=a) for n, a in (calls or [])],
    )


def tool(index: int, name: str, content: str, is_error: bool = False) -> Step:
    return Step(
        index=index,
        role="tool",
        content=content,
        tool_result=ToolResult(
            call_id=f"c{index - 1}", name=name, content=content, is_error=is_error
        ),
    )


class TestSignalEvent:
    def test_rejects_out_of_range_severity(self):
        with pytest.raises(ValueError, match="severity"):
            SignalEvent(step_index=1, kind="k", severity=1.5, evidence="e")

    def test_rejects_zero_based_step(self):
        with pytest.raises(ValueError, match="1-based"):
            SignalEvent(step_index=0, kind="k", severity=0.5, evidence="e")


class TestToolError:
    def test_flags_explicit_error_flag(self):
        traj = build(assistant(1, calls=[("f", {})]), tool(2, "f", "boom", is_error=True))
        assert [e.step_index for e in ToolErrorSignal()(traj)] == [2]

    @pytest.mark.parametrize(
        "payload",
        ["Error: user not found", "not found", "Timeout while calling", "Unauthorized"],
    )
    def test_flags_error_text(self, payload):
        traj = build(assistant(1, calls=[("f", {})]), tool(2, "f", payload))
        assert ToolErrorSignal()(traj)

    def test_ignores_healthy_result(self):
        traj = build(assistant(1, calls=[("f", {})]), tool(2, "f", '{"ok": true, "count": 3}'))
        assert ToolErrorSignal()(traj) == []


class TestMalformedArguments:
    def test_flags_unparseable_arguments(self):
        step = Step(
            index=1,
            role="assistant",
            tool_calls=[ToolCall(name="f", raw_arguments="{bad", parse_error="boom")],
        )
        events = MalformedArgumentsSignal()(build(step))
        assert len(events) == 1
        assert events[0].severity >= 0.7

    def test_ignores_valid_arguments(self):
        assert MalformedArgumentsSignal()(build(assistant(1, calls=[("f", {"a": 1})]))) == []


class TestUnknownTool:
    def test_flags_undeclared_tool(self):
        task = TaskContext(tool_schemas=[{"name": "known", "parameters": {}}])
        traj = build(assistant(1, calls=[("mystery", {})]), task=task)
        events = UnknownToolSignal()(traj)
        assert len(events) == 1
        assert "mystery" in events[0].evidence

    def test_accepts_openai_function_wrapper(self):
        task = TaskContext(tool_schemas=[{"type": "function", "function": {"name": "known"}}])
        traj = build(assistant(1, calls=[("known", {})]), task=task)
        assert UnknownToolSignal()(traj) == []

    def test_silent_without_schemas(self):
        """No schemas means unknown, not clean -- the signal must abstain."""
        assert UnknownToolSignal()(build(assistant(1, calls=[("anything", {})]))) == []


class TestArgumentSchema:
    def test_flags_missing_required_argument(self):
        task = TaskContext(
            tool_schemas=[
                {"name": "f", "parameters": {"properties": {"a": {}, "b": {}}, "required": ["a"]}}
            ]
        )
        events = ArgumentSchemaSignal()(build(assistant(1, calls=[("f", {"b": 1})]), task=task))
        assert any("missing required" in e.evidence for e in events)

    def test_flags_undeclared_argument(self):
        task = TaskContext(
            tool_schemas=[{"name": "f", "parameters": {"properties": {"a": {}}, "required": []}}]
        )
        events = ArgumentSchemaSignal()(build(assistant(1, calls=[("f", {"z": 1})]), task=task))
        assert any("undeclared argument" in e.evidence for e in events)

    def test_clean_call_is_silent(self):
        task = TaskContext(
            tool_schemas=[{"name": "f", "parameters": {"properties": {"a": {}}, "required": ["a"]}}]
        )
        assert ArgumentSchemaSignal()(build(assistant(1, calls=[("f", {"a": 1})]), task=task)) == []


class TestRepeatedCall:
    def test_flags_identical_repeat_and_blames_the_repeat(self):
        traj = build(
            assistant(1, calls=[("f", {"x": 1})]),
            assistant(2, calls=[("f", {"x": 1})]),
        )
        events = RepeatedCallSignal()(traj)
        assert [e.step_index for e in events] == [2]
        assert events[0].detail["first_step"] == 1

    def test_argument_order_does_not_matter(self):
        traj = build(
            assistant(1, calls=[("f", {"a": 1, "b": 2})]),
            assistant(2, calls=[("f", {"b": 2, "a": 1})]),
        )
        assert RepeatedCallSignal()(traj)

    def test_different_arguments_are_not_a_repeat(self):
        traj = build(
            assistant(1, calls=[("f", {"x": 1})]),
            assistant(2, calls=[("f", {"x": 2})]),
        )
        assert RepeatedCallSignal()(traj) == []

    def test_severity_saturates(self):
        steps = [assistant(i, calls=[("f", {})]) for i in range(1, 12)]
        assert all(e.severity <= 0.85 for e in RepeatedCallSignal()(build(*steps)))


class TestOscillation:
    def test_flags_abab(self):
        traj = build(
            assistant(1, calls=[("a", {})]),
            assistant(2, calls=[("b", {})]),
            assistant(3, calls=[("a", {})]),
            assistant(4, calls=[("b", {})]),
        )
        events = OscillationSignal()(traj)
        assert len(events) == 1
        assert events[0].step_index == 1

    def test_does_not_double_report_one_cycle(self):
        traj = build(*[assistant(i, calls=[("a" if i % 2 else "b", {})]) for i in range(1, 9)])
        assert len(OscillationSignal()(traj)) <= 2

    def test_ignores_steady_progress(self):
        traj = build(*[assistant(i, calls=[(f"tool{i}", {})]) for i in range(1, 6)])
        assert OscillationSignal()(traj) == []


class TestNoProgress:
    def test_flags_long_deliberation(self):
        steps = [
            Step(index=i, role="agent", agent="Orchestrator (thought)", content="thinking")
            for i in range(1, 9)
        ]
        events = NoProgressSignal(window=6)(build(*steps))
        assert len(events) == 1
        assert events[0].step_index == 1

    def test_action_resets_the_run(self):
        steps = [
            Step(index=i, role="agent", agent="Orchestrator (thought)", content="t")
            for i in range(1, 5)
        ]
        steps.append(assistant(5, calls=[("f", {})]))
        steps += [
            Step(index=i, role="agent", agent="Orchestrator (thought)", content="t")
            for i in range(6, 10)
        ]
        assert NoProgressSignal(window=6)(build(*steps)) == []


class TestOutcome:
    @pytest.mark.parametrize(
        "text",
        [
            "I'm sorry, but I can't help with that.",
            "I am unable to complete this request.",
            "I do not have access to that system.",
        ],
    )
    def test_refusal_in_tail(self, text):
        traj = build(assistant(1, "working"), assistant(2, "still"), assistant(3, text))
        assert RefusalSignal()(traj)

    def test_tool_output_is_not_a_refusal(self):
        """A tool saying 'not found' is tool_error's business, not a refusal."""
        traj = build(assistant(1, "x"), assistant(2, "y"), tool(3, "f", "not found"))
        assert RefusalSignal()(traj) == []

    def test_incomplete_outcome(self):
        traj = build(
            assistant(1, "a"),
            assistant(2, "b"),
            assistant(3, "Unfortunately no results were found."),
        )
        kinds = {e.kind for e in IncompleteOutcomeSignal()(traj)}
        assert "incomplete_outcome" in kinds

    def test_leaked_error(self):
        traj = build(assistant(1, "a"), assistant(2, "b"), assistant(3, "ValueError: bad input"))
        assert any(e.kind == "leaked_error" for e in IncompleteOutcomeSignal()(traj))

    def test_truncated_run(self):
        traj = build(assistant(1, "a"), assistant(2, calls=[("f", {})]))
        assert any(e.kind == "truncated_run" for e in IncompleteOutcomeSignal()(traj))

    def test_clean_ending_is_silent(self):
        traj = build(
            assistant(1, "a"), assistant(2, "b"), assistant(3, "Done! Your order is placed.")
        )
        assert IncompleteOutcomeSignal()(traj) == []


class TestBudget:
    def test_absolute_ceiling(self):
        traj = build(*[assistant(i, "x") for i in range(1, 91)])
        assert BudgetAnomaly(absolute_steps=80)(traj)

    def test_uses_corpus_stats_when_available(self):
        stats = CorpusStats(mean_steps=10.0, std_steps=2.0)
        long_traj = build(*[assistant(i, "x") for i in range(1, 21)])
        short_traj = build(*[assistant(i, "x") for i in range(1, 12)])
        assert BudgetAnomaly(stats=stats)(long_traj)
        assert BudgetAnomaly(stats=stats)(short_traj) == []

    def test_stall(self):
        steps = []
        for i in range(1, 22, 2):
            steps.append(assistant(i, calls=[(f"f{i}", {})]))
            steps.append(tool(i + 1, f"f{i}", "ok"))
        assert StallSignal(window=10)(build(*steps))

    def test_empty_trajectory(self):
        assert BudgetAnomaly()(Trajectory(trajectory_id="t")) == []


class TestRunSignals:
    def test_events_sorted_by_step(self):
        traj = build(
            assistant(1, calls=[("f", {})]),
            tool(2, "f", "Error: nope"),
            assistant(3, calls=[("f", {})]),
            tool(4, "f", "Error: nope"),
            assistant(5, "I am unable to complete this request."),
        )
        events = run_signals(traj, default_signals())
        assert events
        assert [e.step_index for e in events] == sorted(e.step_index for e in events)

    def test_a_raising_signal_does_not_abort_the_run(self):
        class Exploding:
            name = "boom"

            def __call__(self, trajectory):
                raise RuntimeError("kaboom")

        events = run_signals(build(assistant(1, "x")), [Exploding(), ToolErrorSignal()])
        assert any(e.kind == "signal_error" and "kaboom" in e.evidence for e in events)
