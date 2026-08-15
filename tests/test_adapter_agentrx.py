"""Golden tests for the AgentRx adapter.

These are the regression anchors for the entire benchmark. Step numbering is the
one assumption that, if wrong, shifts every localization score without breaking
anything visibly -- so it is pinned here against real annotated data rather than
asserted in the abstract.

Each anchor checks the same thing from both ends: that the ground truth's
`step_number` resolves to the step our adapter built, *and* that the content of
that step actually matches what the annotation says went wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe.trace.adapters.agentrx import (
    load_magentic,
    load_tau,
    magentic_trajectory,
    tau_trajectory,
)

FIXTURES = Path(__file__).parent / "fixtures"
MAGENTIC_ID = "5f982798-16b9-4051-ab57-cfc7ebdb2a91"


@pytest.fixture(scope="module")
def tau_task2():
    return load_tau(FIXTURES / "tau_task2.json")[0]


@pytest.fixture(scope="module")
def magentic():
    return load_magentic(FIXTURES / "magentic")[0]


@pytest.fixture(scope="module")
def tau_gt():
    return json.loads((FIXTURES / "tau_ground_truth_slice.json").read_text())[0]


@pytest.fixture(scope="module")
def magentic_gt():
    return json.loads((FIXTURES / "magentic_ground_truth_slice.json").read_text())[0]


class TestTauAnchor:
    """tau-retail task 2, annotated with failures at steps 3 and 7."""

    def test_basic_shape(self, tau_task2):
        assert tau_task2.trajectory_id == "2"
        assert tau_task2.domain == "tau_retail"
        assert tau_task2.reward == 0.0
        assert len(tau_task2) == 32

    def test_system_message_is_step_one(self, tau_task2):
        """The leading policy message counts as a step; ground truth numbers assume it."""
        first = tau_task2.step(1)
        assert first.index == 1
        assert first.role == "system"
        assert "retail agent" in first.content.lower()
        assert tau_task2.task.policy == first.content

    def test_step_3_is_the_unauthenticated_tool_call(self, tau_task2, tau_gt):
        """GT failure 1: acted before authenticating the user."""
        failure = next(f for f in tau_gt["failures"] if f["failure_id"] == 1)
        assert failure["step_number"] == 3

        step = tau_task2.step(3)
        assert step.role == "assistant"
        assert [c.name for c in step.tool_calls] == ["list_all_product_types"]
        # No authentication call precedes it -- which is precisely the annotation.
        earlier = [c.name for s in tau_task2.steps[:2] for c in s.tool_calls]
        assert "find_user_id_by_name_zip" not in earlier

    def test_step_7_is_the_miscount(self, tau_task2, tau_gt):
        """GT failure 2: miscounted t-shirt options from the tool result."""
        failure = next(f for f in tau_gt["failures"] if f["failure_id"] == 2)
        assert failure["step_number"] == 7

        step = tau_task2.step(7)
        assert step.role == "assistant"
        assert step.tool_calls == []
        assert "11 available" in step.content.replace("  ", " ")

    def test_tool_calls_and_results_are_paired(self, tau_task2):
        call_ids = {c.id for s in tau_task2.steps for c in s.tool_calls}
        result_ids = {s.tool_result.call_id for s in tau_task2.steps if s.tool_result is not None}
        assert result_ids
        assert result_ids <= call_ids

    def test_null_content_tool_call_turn_survives(self, tau_task2):
        """Assistant turns that only call a tool carry null content upstream."""
        step = tau_task2.step(3)
        assert step.content == ""
        assert step.tool_calls

    def test_index_disagreement_is_fatal(self):
        """A record whose own index contradicts position must not be silently accepted."""
        record = {
            "task_id": 999,
            "reward": 0.0,
            "traj": [
                {"role": "system", "content": "policy", "index": 1},
                {"role": "user", "content": "hi", "index": 5},
            ],
        }
        with pytest.raises(ValueError, match="disagrees with 1-based position"):
            tau_trajectory(record)


class TestMagenticAnchor:
    """Magentic-One 5f982798, annotated with failures at steps 13 and 17."""

    def test_basic_shape(self, magentic):
        assert magentic.trajectory_id == MAGENTIC_ID
        assert magentic.domain == "magentic_one"
        assert len(magentic) == 67

    def test_human_request_is_step_one(self, magentic):
        first = magentic.step(1)
        assert first.role == "user"
        assert first.agent == "human"
        assert "fast radio bursts" in first.content
        assert magentic.task.instruction == first.content

    def test_step_13_is_the_pdf_attempt(self, magentic, magentic_gt):
        """GT failure 1: WebSurfer could not download and search the PDF."""
        failure = next(f for f in magentic_gt["failures"] if f["failure_id"] == 1)
        assert failure["step_number"] == 13

        step = magentic.step(13)
        assert step.agent == "WebSurfer"
        assert "PDF" in step.content

    def test_step_17_is_the_empty_summary(self, magentic, magentic_gt):
        failure = next(f for f in magentic_gt["failures"] if f["failure_id"] == 2)
        assert failure["step_number"] == 17

        step = magentic.step(17)
        assert step.agent == "WebSurfer"
        assert "Nothing to summarize" in step.content

    def test_agent_identity_is_preserved(self, magentic):
        """Multi-agent attribution needs the acting agent, not a flattened role."""
        agents = {s.agent for s in magentic.steps}
        assert {"Orchestrator (thought)", "WebSurfer", "FileSurfer", "human"} <= agents

    def test_thought_steps_are_flagged(self, magentic):
        thoughts = [s for s in magentic.steps if s.is_thought]
        assert len(thoughts) == 35
        assert all(s.agent == "Orchestrator (thought)" for s in thoughts)
        # Deliberation dominates these traces, which is why signals must weight it down.
        assert len(thoughts) > len(magentic) / 2

    def test_every_annotated_step_is_in_range(self, magentic, magentic_gt):
        for failure in magentic_gt["failures"]:
            assert 1 <= failure["step_number"] <= len(magentic)

    def test_empty_role_is_tolerated(self):
        traj = magentic_trajectory("x", [{"role": "", "content": "orphan"}])
        assert traj.step(1).agent is None
        assert traj.step(1).role == "agent"
