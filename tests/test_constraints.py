"""Tests for declarative constraints and constraint synthesis."""

from __future__ import annotations

import pytest

from probe.detect import Detector
from probe.llm import FakeClient
from probe.signals import (
    ConstraintSignal,
    ConstraintSynthesizer,
    default_signals,
    numeric_bound,
    require_before,
    schema_fingerprint,
    step_constraint,
    tool_call_constraint,
)
from probe.trace.model import Step, TaskContext, ToolCall, Trajectory


def traj(*steps: Step, **kw) -> Trajectory:
    return Trajectory(trajectory_id="t", steps=list(steps), **kw)


def call(index: int, name: str, args: dict) -> Step:
    return Step(
        index=index,
        role="assistant",
        tool_calls=[ToolCall(id=f"c{index}", name=name, arguments=args)],
    )


class TestRequireBefore:
    def test_flags_dependent_without_prerequisite(self):
        constraint = require_before("auth", "authenticate first", "authenticate", "refund")
        events = constraint(traj(call(1, "refund", {})))
        assert len(events) == 1
        assert events[0].step_index == 1
        assert "before any call to" in events[0].evidence

    def test_silent_when_ordered_correctly(self):
        constraint = require_before("auth", "authenticate first", "authenticate", "refund")
        assert constraint(traj(call(1, "authenticate", {}), call(2, "refund", {}))) == []

    def test_order_matters_not_mere_presence(self):
        """Calling the prerequisite afterwards does not retroactively fix it."""
        constraint = require_before("auth", "authenticate first", "authenticate", "refund")
        events = constraint(traj(call(1, "refund", {}), call(2, "authenticate", {})))
        assert [e.step_index for e in events] == [1]

    def test_flags_every_violation(self):
        constraint = require_before("auth", "d", "authenticate", "refund")
        events = constraint(traj(call(1, "refund", {}), call(2, "refund", {})))
        assert [e.step_index for e in events] == [1, 2]


class TestNumericBound:
    def test_flags_over_maximum(self):
        constraint = numeric_bound("cap", "max refund", "refund", "amount", maximum=100)
        events = constraint(traj(call(1, "refund", {"amount": 250})))
        assert len(events) == 1
        assert "exceeds the maximum" in events[0].evidence

    def test_flags_under_minimum(self):
        constraint = numeric_bound("floor", "min", "refund", "amount", minimum=1)
        assert constraint(traj(call(1, "refund", {"amount": 0})))

    def test_silent_within_bounds(self):
        constraint = numeric_bound("cap", "max refund", "refund", "amount", maximum=100)
        assert constraint(traj(call(1, "refund", {"amount": 50}))) == []

    def test_ignores_other_tools(self):
        constraint = numeric_bound("cap", "max refund", "refund", "amount", maximum=100)
        assert constraint(traj(call(1, "charge", {"amount": 9999}))) == []

    def test_non_numeric_argument_is_skipped_not_fatal(self):
        constraint = numeric_bound("cap", "max refund", "refund", "amount", maximum=100)
        assert constraint(traj(call(1, "refund", {"amount": "lots"}))) == []

    def test_string_number_is_still_bounded(self):
        constraint = numeric_bound("cap", "max refund", "refund", "amount", maximum=100)
        assert constraint(traj(call(1, "refund", {"amount": "250"})))


class TestConstraintHelpers:
    def test_step_constraint(self):
        constraint = step_constraint(
            "no_shouting",
            "d",
            lambda s, t: "all caps" if s.content.isupper() and s.content else None,
        )
        events = constraint(traj(Step(index=1, role="assistant", content="HELLO")))
        assert len(events) == 1

    def test_tool_call_constraint_sees_the_trajectory(self):
        """Cross-step rules are the common case; the check gets the whole trace."""
        constraint = tool_call_constraint(
            "later_than_first",
            "d",
            "act",
            lambda c, s, t: "not the first step" if s.index != 1 else None,
        )
        events = constraint(traj(call(1, "act", {}), call(2, "act", {})))
        assert [e.step_index for e in events] == [2]


class TestConstraintSignal:
    def test_runs_all_constraints(self):
        signal = ConstraintSignal(
            [
                require_before("auth", "d", "authenticate", "refund"),
                numeric_bound("cap", "d", "refund", "amount", maximum=10),
            ]
        )
        events = signal(traj(call(1, "refund", {"amount": 99})))
        assert len(events) == 2

    def test_empty_by_default(self):
        assert ConstraintSignal()(traj(call(1, "refund", {}))) == []

    def test_a_raising_constraint_is_isolated(self):
        def explode(trajectory):
            raise RuntimeError("kaboom")

        from probe.signals.constraints import Constraint

        signal = ConstraintSignal(
            [
                Constraint(name="bad", description="d", check=explode),
                numeric_bound("cap", "d", "refund", "amount", maximum=10),
            ]
        )
        events = signal(traj(call(1, "refund", {"amount": 99})))
        assert any(e.kind == "signal_error" for e in events)
        assert any(e.kind == "constraint_violation" for e in events)


class TestFingerprint:
    def test_stable_across_key_order(self):
        a = [{"name": "f", "parameters": {"a": 1, "b": 2}}]
        b = [{"name": "f", "parameters": {"b": 2, "a": 1}}]
        assert schema_fingerprint(a) == schema_fingerprint(b)

    def test_stable_across_schema_order(self):
        a = [{"name": "f"}, {"name": "g"}]
        b = [{"name": "g"}, {"name": "f"}]
        assert schema_fingerprint(a) == schema_fingerprint(b)

    def test_policy_changes_the_fingerprint(self):
        schemas = [{"name": "f"}]
        assert schema_fingerprint(schemas, "policy A") != schema_fingerprint(schemas, "policy B")

    def test_different_schemas_differ(self):
        assert schema_fingerprint([{"name": "f"}]) != schema_fingerprint([{"name": "g"}])


class TestSynthesizer:
    SPEC = """{"constraints": [
        {"name": "auth_first", "description": "authenticate before refunding",
         "kind": "require_before",
         "params": {"prerequisite": "authenticate", "dependent": "refund"}},
        {"name": "refund_cap", "description": "cap refunds",
         "kind": "numeric_bound",
         "params": {"tool": "refund", "argument": "amount", "maximum": 100}}
    ]}"""

    def _traj(self, policy="be careful"):
        return traj(
            call(1, "refund", {"amount": 250}),
            task=TaskContext(tool_schemas=[{"name": "refund"}], policy=policy),
        )

    def test_synthesizes_executable_constraints(self):
        synth = ConstraintSynthesizer(client=FakeClient(responses=[self.SPEC]))
        constraints = synth.for_trajectory(self._traj())
        assert len(constraints) == 2
        events = ConstraintSignal(constraints)(self._traj())
        assert len(events) == 2

    def test_cache_amortizes_across_trajectories(self):
        """The economic claim: synthesis costs one call per deployment, not per trace."""
        synth = ConstraintSynthesizer(
            client=FakeClient(responses=[self.SPEC, self.SPEC, self.SPEC])
        )
        for _ in range(5):
            synth.for_trajectory(self._traj())
        assert synth.calls == 1

    def test_different_policy_misses_the_cache(self):
        synth = ConstraintSynthesizer(client=FakeClient(responses=[self.SPEC, self.SPEC]))
        synth.for_trajectory(self._traj(policy="A"))
        synth.for_trajectory(self._traj(policy="B"))
        assert synth.calls == 2

    def test_no_schemas_means_no_call(self):
        synth = ConstraintSynthesizer(client=FakeClient(responses=[self.SPEC]))
        assert synth.for_trajectory(traj(call(1, "refund", {}))) == []
        assert synth.calls == 0

    def test_unparseable_spec_yields_nothing(self):
        synth = ConstraintSynthesizer(client=FakeClient(responses=["sorry, no idea"]))
        assert synth.for_trajectory(self._traj()) == []

    def test_unknown_kinds_are_described_but_not_executed(self):
        """Materializing arbitrary model-authored predicates is not a trade we make."""
        spec = (
            '{"constraints": [{"name": "x", "kind": "other", "params": {"code": "os.system(1)"}}]}'
        )
        synth = ConstraintSynthesizer(client=FakeClient(responses=[spec]))
        assert synth.for_trajectory(self._traj()) == []


@pytest.fixture(scope="module")
def demo():
    """The example agent, imported as a test fixture."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import demo_agent

    return demo_agent


class TestDemoEndToEnd:
    """The injected-failure fixture, asserted through the real detector."""

    def test_healthy_run_is_not_flagged(self, demo):
        assert not Detector()(demo.healthy()).failed

    @pytest.mark.parametrize("name", ["retry_loop", "unknown_tool"])
    def test_error_shaped_failures_are_detected_without_constraints(self, demo, name):
        assert Detector()(getattr(demo, name)()).failed

    def test_policy_violation_needs_constraints(self, demo):
        """The gap constraints exist to close, pinned so it cannot silently reopen."""
        trajectory = demo.policy_violation()
        assert not Detector()(trajectory).failed

        detector = Detector(signals=[*default_signals(), ConstraintSignal(demo.demo_constraints())])
        verdict = detector(trajectory)
        assert verdict.failed
        assert "constraint_violation" in verdict.kinds

    def test_constraints_do_not_fire_on_the_healthy_run(self, demo):
        detector = Detector(signals=[*default_signals(), ConstraintSignal(demo.demo_constraints())])
        assert not detector(demo.healthy()).failed

    def test_violation_lands_on_the_injected_step(self, demo):
        trajectory = demo.policy_violation()
        events = ConstraintSignal(demo.demo_constraints())(trajectory)
        assert events
        assert any(e.step_index == trajectory.metadata["injected_failure_step"] for e in events)
