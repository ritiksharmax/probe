"""Tests for the LLM client layer and the RCA judge.

The judge is exercised entirely through `FakeClient`, so the whole pipeline is
verifiable without a network or an API key. The assertions that matter most are
the ones about *what reaches the prompt*: that filtered mode really does show the
judge less, and that ground truth never leaks into it.
"""

from __future__ import annotations

import json

import pytest

from probe.llm import FakeClient, LLMResponse, parse_json, price
from probe.rca import RCAJudge, RCAReport
from probe.rca.judge import _coerce_category, _coerce_step
from probe.trace.model import Step, TaskContext, ToolCall, ToolResult, Trajectory


def answer(step: int = 3, category: int = 4, **kw) -> str:
    payload = {
        "critical_step": step,
        "category": category,
        "cause": "miscounted the tool result",
        "counterfactual": "should have recounted",
        "confidence": 0.8,
    }
    payload.update(kw)
    return json.dumps(payload)


def failing_trajectory(n: int = 40) -> Trajectory:
    """A long trajectory with one obvious error, so filtering has something to do."""
    steps = [
        Step(index=1, role="system", content="You are a retail agent. Authenticate first."),
        Step(index=2, role="user", content="How many t-shirts do you have?"),
        Step(
            index=3,
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="list_products", arguments={})],
        ),
        Step(
            index=4,
            role="tool",
            content="Error: not found",
            tool_result=ToolResult(
                call_id="c1", name="list_products", content="Error: not found", is_error=True
            ),
        ),
    ]
    for i in range(5, n + 1):
        steps.append(Step(index=i, role="assistant", content=f"filler turn {i}"))
    return Trajectory(
        trajectory_id="t",
        steps=steps,
        task=TaskContext(
            instruction="Find out how many t-shirts are available.",
            policy="Authenticate the user before answering.",
            expected_actions=[{"name": "find_user_id_by_name_zip"}],
        ),
    )


class TestPricing:
    def test_known_model_priced(self):
        # 1M in + 1M out on Opus 5 list rates.
        assert price("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)

    def test_prefix_match_handles_suffixed_ids(self):
        assert price("claude-sonnet-5-something", 1_000_000, 0) == pytest.approx(3.0)

    def test_unknown_model_is_free(self):
        assert price("qwen3:4b", 10_000_000, 10_000_000) == 0.0


class TestParseJson:
    def test_plain(self):
        assert parse_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_surrounded_by_prose(self):
        """Small models narrate around their JSON no matter what you tell them."""
        assert parse_json('Sure! Here is my answer:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert parse_json('x {"a": "} not the end {", "b": 2} y') == {
            "a": "} not the end {",
            "b": 2,
        }

    def test_escaped_quote_inside_string(self):
        assert parse_json(r'{"a": "say \"hi\"", "b": 1}') == {"a": 'say "hi"', "b": 1}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            parse_json("there is no json here")


class TestFakeClient:
    def test_records_prompts(self):
        client = FakeClient(responses=[answer()])
        client.complete("hello", system="sys")
        assert client.prompts == ["hello"]
        assert client.systems == ["sys"]

    def test_raises_without_script(self):
        with pytest.raises(RuntimeError, match="no scripted responses"):
            FakeClient().complete("x")

    def test_repeats_last_response_when_exhausted(self):
        client = FakeClient(responses=["a"])
        assert client.complete("1").text == "a"
        assert client.complete("2").text == "a"


class TestJudgePrompt:
    def test_filtered_mode_shows_far_less_than_full(self):
        """The H2 mechanism, asserted directly."""
        traj = failing_trajectory(60)
        filtered = RCAJudge(FakeClient(responses=[answer()]), mode="filtered")
        full = RCAJudge(FakeClient(responses=[answer()]), mode="full")

        _, _, shown_filtered = filtered.build_prompt(traj, [])
        _, _, shown_full = full.build_prompt(traj, [])
        assert shown_full == len(traj)
        assert shown_filtered < shown_full

    def test_filtered_prompt_is_shorter(self):
        traj = failing_trajectory(60)
        p_filtered, _, _ = RCAJudge(FakeClient(responses=[answer()]), mode="filtered").build_prompt(
            traj, []
        )
        p_full, _, _ = RCAJudge(FakeClient(responses=[answer()]), mode="full").build_prompt(
            traj, []
        )
        assert len(p_filtered) < len(p_full)

    def test_filtered_prompt_marks_elisions(self):
        traj = failing_trajectory(60)
        prompt, _, _ = RCAJudge(FakeClient(responses=[answer()])).build_prompt(traj, [])
        assert "omitted" in prompt

    def test_prompt_includes_task_policy_and_taxonomy(self):
        traj = failing_trajectory()
        prompt, _, _ = RCAJudge(FakeClient(responses=[answer()])).build_prompt(traj, [])
        assert "t-shirts" in prompt
        assert "Authenticate the user" in prompt
        assert "Misinterpretation of Tool Output" in prompt
        assert "Inconclusive" in prompt

    def test_ground_truth_never_reaches_the_prompt(self):
        """PROBE's premise is diagnosis without ground-truth trajectories."""
        traj = failing_trajectory()
        for mode in ("filtered", "full"):
            prompt, _, _ = RCAJudge(FakeClient(responses=[answer()]), mode=mode).build_prompt(
                traj, []
            )
            assert "find_user_id_by_name_zip" not in prompt
            assert "expected_actions" not in prompt

    def test_violation_log_appears_when_signals_fire(self):
        traj = failing_trajectory()
        judge = RCAJudge(FakeClient(responses=[answer()]))
        from probe.signals.base import default_signals, run_signals

        events = run_signals(traj, default_signals())
        prompt, _, _ = judge.build_prompt(traj, events)
        assert "Violation log" in prompt
        assert "tool_error" in prompt


class TestJudgeParsing:
    def test_happy_path(self):
        traj = failing_trajectory()
        report = RCAJudge(FakeClient(responses=[answer(step=4, category=4)]))(traj)
        assert isinstance(report, RCAReport)
        assert report.critical_step == 4
        assert report.category_case == 4
        assert report.category == "Misinterpretation of Tool Output"
        assert report.confidence == pytest.approx(0.8)
        assert not report.degraded

    def test_repairs_bad_json_then_succeeds(self):
        client = FakeClient(responses=["I think step 4 is wrong.", answer(step=4)])
        report = RCAJudge(client, max_repairs=2)(failing_trajectory())
        assert report.critical_step == 4
        assert report.repairs == 1
        assert not report.degraded
        assert "not valid JSON" in client.prompts[1]

    def test_gives_up_and_degrades_to_the_signal_prior(self):
        client = FakeClient(responses=["nope"])
        report = RCAJudge(client, max_repairs=1)(failing_trajectory())
        assert report.degraded
        assert report.critical_step is not None  # never forfeits the trajectory
        assert report.category_case == 10
        assert report.repairs == 2

    def test_accepts_a_category_name_instead_of_a_number(self):
        client = FakeClient(responses=[answer(category="Invalid Invocation")])
        assert RCAJudge(client)(failing_trajectory()).category_case == 3

    def test_out_of_range_step_is_clamped_not_discarded(self):
        traj = failing_trajectory(10)
        assert RCAJudge(FakeClient(responses=[answer(step=999)]))(traj).critical_step == 10
        assert RCAJudge(FakeClient(responses=[answer(step=0)]))(traj).critical_step == 1

    def test_accounting_is_recorded(self):
        report = RCAJudge(FakeClient(responses=[answer()]), tier="small")(failing_trajectory())
        assert report.prompt_tokens > 0
        assert report.completion_tokens > 0
        assert report.tier == "small"
        assert report.steps_total == 40

    def test_compression_reported(self):
        report = RCAJudge(FakeClient(responses=[answer()]))(failing_trajectory(60))
        assert 0 < report.compression < 1
        assert report.windows

    def test_full_mode_reports_no_compression(self):
        report = RCAJudge(FakeClient(responses=[answer()]), mode="full")(failing_trajectory(60))
        assert report.compression == 1.0
        assert report.windows == []


class TestCoercion:
    @pytest.mark.parametrize(
        ("value", "expected"), [(4, 4), ("4", 4), (0, 1), (99, 40), (None, None), ("x", None)]
    )
    def test_step(self, value, expected):
        assert _coerce_step(value, 40) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, 1), (10, 10), (0, 10), (11, 10), ("Guardrails Triggered", 8), (True, 10), (None, 10)],
    )
    def test_category(self, value, expected):
        assert _coerce_category(value) == expected


def test_report_render_is_readable():
    report = RCAJudge(FakeClient(responses=[answer()]))(failing_trajectory())
    text = report.render()
    assert "critical step" in text
    assert "category" in text
    assert "cost" in text


def test_llm_response_json_helper():
    assert LLMResponse(text='{"a": 1}', model="m").json() == {"a": 1}


class TestDegradedFallback:
    """Regression tests for the judge's fallback path."""

    def test_fallback_uses_the_judges_own_filter_not_a_fresh_default(self):
        """A caller who tuned the filter must get a fallback scored against it."""
        from probe.localize.evidence import EvidenceFilter

        traj = failing_trajectory(40)
        wide = RCAJudge(
            FakeClient(responses=["not json"]),
            evidence_filter=EvidenceFilter(look_back=0, look_forward=0, thought_weight=1.0),
            max_repairs=0,
        )
        report = wide(traj)
        assert report.degraded
        # With no neighbourhood spill the peak is the step that actually fired,
        # which is the tool error at step 4 rather than a smeared neighbour.
        assert report.critical_step == 4

    def test_fallback_still_returns_a_step_when_nothing_fires(self):
        traj = Trajectory(
            trajectory_id="quiet",
            steps=[Step(index=i, role="assistant", content=f"x{i}") for i in range(1, 6)],
        )
        report = RCAJudge(FakeClient(responses=["nope"]), max_repairs=0)(traj)
        assert report.degraded
        assert report.critical_step == 5


class TestFakeClientRealism:
    def test_fake_client_splits_think_blocks_like_the_http_clients(self):
        """Otherwise tests exercise a convenient fiction instead of the real path."""
        client = FakeClient(responses=['reasoning here</think>{"critical_step": 3, "category": 1}'])
        response = client.complete("x")
        assert response.text == '{"critical_step": 3, "category": 1}'
        assert "reasoning here" in response.reasoning

    def test_judge_parses_a_scripted_thinking_response(self):
        raw = 'Maybe {"critical_step": 99}? no.\n</think>\n' + answer(step=4, category=4)
        report = RCAJudge(FakeClient(responses=[raw]))(failing_trajectory())
        assert report.critical_step == 4  # not the 99 from the discarded draft
        assert not report.degraded


class TestSignalCaveat:
    """The violation log must frame signals as symptoms, not causes.

    Measured on the benchmark data, the nearest signal sits a median of 6-8 steps
    *after* the annotated root cause, so a bare list of signal locations anchors
    the judge on the wrong steps -- which is the rationale for the framing.

    Its measured effect is *not* settled. A single-run A/B showed tau-retail exact
    match going 0.103 -> 0.241 and was reported as "more than doubled"; that was
    withdrawn. Repeated runs put the run-to-run spread at +/-1.6-1.9 trajectories
    per domain, and the observed swing was 3 trajectories split across two domains
    in opposite directions. Treat the caveat as unquantified until the ablation is
    re-run via `--no-signal-caveat` with repeats.
    """

    def test_violation_log_explains_that_signals_are_symptoms(self):
        traj = failing_trajectory()
        judge = RCAJudge(FakeClient(responses=[answer()]), mode="full")
        from probe.signals.base import default_signals, run_signals

        prompt, _, _ = judge.build_prompt(traj, run_signals(traj, default_signals()))
        assert "symptoms" in prompt
        assert "EARLIER" in prompt

    def test_no_caveat_when_the_violation_log_is_suppressed(self):
        traj = failing_trajectory()
        judge = RCAJudge(FakeClient(responses=[answer()]), mode="full", include_violations=False)
        from probe.signals.base import default_signals, run_signals

        prompt, _, _ = judge.build_prompt(traj, run_signals(traj, default_signals()))
        assert "Violation log" not in prompt
        assert "symptoms" not in prompt

    def test_caveat_can_be_ablated_without_editing_the_source(self):
        """The 'without' arm has to come from a flag, or it pins to no commit."""
        traj = failing_trajectory()
        from probe.signals.base import default_signals, run_signals

        events = run_signals(traj, default_signals())
        judge = RCAJudge(FakeClient(responses=[answer()]), mode="full", include_signal_caveat=False)
        prompt, _, _ = judge.build_prompt(traj, events)
        assert "Violation log" in prompt, "the log itself must survive the ablation"
        assert "symptoms" not in prompt
        assert "EARLIER" not in prompt
