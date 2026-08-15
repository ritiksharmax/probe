"""Tests for the production ingestion adapters: OTel, Langfuse, LangSmith.

Fixtures are written to mirror what these systems actually emit, including the
shape disagreements that make the adapters non-trivial: OTLP-wrapped attribute
values, camelCase vs snake_case, and LangChain's three different message
serializations.

The invariant every adapter must hold is the one the rest of the library depends
on: steps come out **chronological and 1-based**, whatever order the source
records arrived in.
"""

from __future__ import annotations

import json

import pytest

from probe.trace.adapters import load_langfuse, load_langsmith, load_otel
from probe.trace.adapters.langfuse import observations_to_trajectory
from probe.trace.adapters.langsmith import runs_to_trajectory
from probe.trace.adapters.otel import spans_to_trajectory


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------- OTel


def otlp_span(name, start, attrs, events=None, status=None, trace_id="t1"):
    """A span in OTLP/JSON form, with attribute values wrapped as AnyValue."""
    wrapped = []
    for key, value in attrs.items():
        if isinstance(value, bool):
            wrapped.append({"key": key, "value": {"boolValue": value}})
        elif isinstance(value, int):
            wrapped.append({"key": key, "value": {"intValue": str(value)}})
        else:
            wrapped.append({"key": key, "value": {"stringValue": str(value)}})
    span = {
        "traceId": trace_id,
        "name": name,
        "startTimeUnixNano": str(start),
        "attributes": wrapped,
    }
    if events:
        span["events"] = events
    if status:
        span["status"] = status
    return span


class TestOtel:
    def test_genai_semconv_with_message_attributes(self):
        spans = [
            otlp_span(
                "chat",
                200,
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "gpt-x",
                    "gen_ai.input.messages": json.dumps(
                        [
                            {"role": "system", "content": "Be careful."},
                            {"role": "user", "content": "refund my order"},
                        ]
                    ),
                    "gen_ai.output.messages": json.dumps(
                        [{"role": "assistant", "content": "Sure, looking now."}]
                    ),
                },
            )
        ]
        traj = spans_to_trajectory(spans)
        assert [s.role for s in traj.steps] == ["system", "user", "assistant"]
        assert traj.task.instruction == "refund my order"
        assert traj.task.policy == "Be careful."

    def test_openinference_indexed_attributes(self):
        spans = [
            otlp_span(
                "llm",
                100,
                {
                    "openinference.span.kind": "LLM",
                    "llm.input_messages.0.message.role": "user",
                    "llm.input_messages.0.message.content": "hello",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "hi there",
                },
            )
        ]
        traj = spans_to_trajectory(spans)
        assert [s.content for s in traj.steps] == ["hello", "hi there"]

    def test_tool_span_becomes_a_tool_result(self):
        spans = [
            otlp_span(
                "execute_tool refund",
                300,
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "refund",
                    "output.value": '{"ok": true}',
                },
            )
        ]
        traj = spans_to_trajectory(spans)
        assert traj.step(1).role == "tool"
        assert traj.step(1).tool_result.name == "refund"
        assert not traj.step(1).tool_result.is_error

    def test_error_status_marks_the_tool_result(self):
        spans = [
            otlp_span(
                "execute_tool refund",
                300,
                {"gen_ai.tool.name": "refund", "output.value": "boom"},
                status={"code": "STATUS_CODE_ERROR"},
            )
        ]
        assert spans_to_trajectory(spans).step(1).tool_result.is_error

    def test_spans_are_ordered_by_start_time_not_input_order(self):
        """The load-bearing invariant: step index must be chronological."""
        late = otlp_span(
            "chat",
            900,
            {"gen_ai.input.messages": json.dumps([{"role": "user", "content": "second"}])},
        )
        early = otlp_span(
            "chat",
            100,
            {"gen_ai.input.messages": json.dumps([{"role": "user", "content": "first"}])},
        )
        traj = spans_to_trajectory([late, early])
        assert [s.content for s in traj.steps] == ["first", "second"]
        assert [s.index for s in traj.steps] == [1, 2]

    def test_tool_calls_are_extracted(self):
        spans = [
            otlp_span(
                "chat",
                100,
                {
                    "gen_ai.request.model": "m",
                    "gen_ai.output.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "refund",
                                            "arguments": '{"amount": 20}',
                                        },
                                    }
                                ],
                            }
                        ]
                    ),
                },
            )
        ]
        call = spans_to_trajectory(spans).step(1).tool_calls[0]
        assert call.name == "refund"
        assert call.arguments == {"amount": 20}

    def test_malformed_tool_arguments_are_preserved_not_dropped(self):
        spans = [
            otlp_span(
                "chat",
                100,
                {
                    "gen_ai.request.model": "m",
                    "gen_ai.output.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}],
                            }
                        ]
                    ),
                },
            )
        ]
        call = spans_to_trajectory(spans).step(1).tool_calls[0]
        assert call.parse_error
        assert call.raw_arguments == "{oops"

    def test_repeated_system_prompt_is_kept_once(self):
        """Instrumentation resends the system prompt on every call; that is noise."""
        spans = [
            otlp_span(
                "chat",
                i * 100,
                {
                    "gen_ai.request.model": "m",
                    "gen_ai.input.messages": json.dumps(
                        [
                            {"role": "system", "content": "policy"},
                            {"role": "user", "content": f"q{i}"},
                        ]
                    ),
                },
            )
            for i in (1, 2, 3)
        ]
        traj = spans_to_trajectory(spans)
        assert sum(1 for s in traj.steps if s.role == "system") == 1

    def test_events_convention(self):
        spans = [
            otlp_span(
                "chat",
                100,
                {"gen_ai.request.model": "m"},
                events=[
                    {
                        "name": "gen_ai.user.message",
                        "attributes": [
                            {
                                "key": "gen_ai.event.content",
                                "value": {"stringValue": '{"content": "hey"}'},
                            }
                        ],
                    },
                    {
                        "name": "gen_ai.choice",
                        "attributes": [
                            {
                                "key": "gen_ai.event.content",
                                "value": {"stringValue": '{"content": "yo"}'},
                            }
                        ],
                    },
                ],
            )
        ]
        traj = spans_to_trajectory(spans)
        assert [s.role for s in traj.steps] == ["user", "assistant"]

    def test_loads_otlp_envelope_and_groups_by_trace(self, tmp_path):
        payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                otlp_span(
                                    "chat",
                                    100,
                                    {
                                        "gen_ai.request.model": "m",
                                        "gen_ai.input.messages": json.dumps(
                                            [{"role": "user", "content": "a"}]
                                        ),
                                    },
                                    trace_id="A",
                                ),
                                otlp_span(
                                    "chat",
                                    100,
                                    {
                                        "gen_ai.request.model": "m",
                                        "gen_ai.input.messages": json.dumps(
                                            [{"role": "user", "content": "b"}]
                                        ),
                                    },
                                    trace_id="B",
                                ),
                            ]
                        }
                    ]
                }
            ]
        }
        trajs = load_otel(write(tmp_path, "spans.json", payload))
        assert {t.trajectory_id for t in trajs} == {"A", "B"}

    def test_loads_jsonl(self, tmp_path):
        lines = "\n".join(
            json.dumps(
                otlp_span(
                    "chat",
                    i * 10,
                    {
                        "gen_ai.request.model": "m",
                        "gen_ai.input.messages": json.dumps([{"role": "user", "content": f"m{i}"}]),
                    },
                )
            )
            for i in (1, 2)
        )
        trajs = load_otel(write(tmp_path, "spans.jsonl", lines))
        assert len(trajs) == 1
        assert len(trajs[0]) == 2

    def test_empty_file(self, tmp_path):
        assert load_otel(write(tmp_path, "empty.json", "")) == []


# ----------------------------------------------------------------- Langfuse


class TestLangfuse:
    def test_generation_and_tool_observations(self):
        observations = [
            {
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:00Z",
                "input": [
                    {"role": "system", "content": "be careful"},
                    {"role": "user", "content": "refund order 7"},
                ],
                "output": {"role": "assistant", "content": "on it"},
            },
            {
                "type": "SPAN",
                "name": "refund_tool",
                "startTime": "2026-01-01T00:00:01Z",
                "output": {"status": "ok"},
            },
        ]
        traj = observations_to_trajectory(observations, "tr1")
        assert [s.role for s in traj.steps] == ["system", "user", "assistant", "tool"]
        assert traj.task.instruction == "refund order 7"
        assert traj.step(4).tool_result.name == "refund_tool"

    def test_error_level_marks_tool_result(self):
        observations = [
            {
                "type": "SPAN",
                "name": "api_call",
                "startTime": "1",
                "level": "ERROR",
                "output": "failed",
            }
        ]
        assert observations_to_trajectory(observations, "t").step(1).tool_result.is_error

    def test_snake_case_fields_accepted(self):
        observations = [
            {
                "type": "GENERATION",
                "start_time": "1",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "output": "hello",
            }
        ]
        traj = observations_to_trajectory(observations, "t")
        assert [s.content for s in traj.steps] == ["hi", "hello"]

    def test_ordered_by_start_time(self):
        observations = [
            {
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:09Z",
                "input": [{"role": "user", "content": "second"}],
            },
            {
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:01Z",
                "input": [{"role": "user", "content": "first"}],
            },
        ]
        traj = observations_to_trajectory(observations, "t")
        assert [s.content for s in traj.steps] == ["first", "second"]

    def test_json_encoded_strings_are_parsed(self):
        observations = [
            {
                "type": "GENERATION",
                "startTime": "1",
                "input": json.dumps([{"role": "user", "content": "hi"}]),
            }
        ]
        assert observations_to_trajectory(observations, "t").step(1).content == "hi"

    def test_loads_trace_export(self, tmp_path):
        payload = {
            "id": "trace-1",
            "name": "support",
            "observations": [
                {
                    "type": "GENERATION",
                    "startTime": "1",
                    "input": [{"role": "user", "content": "hi"}],
                }
            ],
        }
        trajs = load_langfuse(write(tmp_path, "lf.json", payload))
        assert len(trajs) == 1
        assert trajs[0].trajectory_id == "trace-1"
        assert trajs[0].domain == "langfuse"

    def test_loads_paginated_data_envelope(self, tmp_path):
        payload = {
            "data": [
                {
                    "id": "a",
                    "observations": [
                        {
                            "type": "GENERATION",
                            "startTime": "1",
                            "input": [{"role": "user", "content": "x"}],
                        }
                    ],
                },
                {
                    "id": "b",
                    "observations": [
                        {
                            "type": "GENERATION",
                            "startTime": "1",
                            "input": [{"role": "user", "content": "y"}],
                        }
                    ],
                },
            ]
        }
        assert {t.trajectory_id for t in load_langfuse(write(tmp_path, "lf.json", payload))} == {
            "a",
            "b",
        }


# ---------------------------------------------------------------- LangSmith


class TestLangSmith:
    def test_llm_and_tool_runs(self):
        runs = [
            {
                "run_type": "llm",
                "start_time": "2026-01-01T00:00:00",
                "trace_id": "t",
                "inputs": {
                    "messages": [
                        [
                            {"role": "system", "content": "policy"},
                            {"role": "user", "content": "do it"},
                        ]
                    ]
                },
                "outputs": {"generations": [[{"message": {"role": "assistant", "content": "ok"}}]]},
            },
            {
                "run_type": "tool",
                "name": "search",
                "start_time": "2026-01-01T00:00:01",
                "trace_id": "t",
                "outputs": {"result": "found"},
            },
        ]
        traj = runs_to_trajectory(runs, "t")
        assert [s.role for s in traj.steps] == ["system", "user", "assistant", "tool"]
        assert traj.step(4).tool_result.content == "found"

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "human", "content": "hi"},
            {"type": "human", "data": {"content": "hi"}},
            {
                "id": ["langchain", "schema", "messages", "HumanMessage"],
                "kwargs": {"content": "hi"},
            },
        ],
    )
    def test_all_three_langchain_message_serializations(self, message):
        """Which shape you get depends on the LangChain version that wrote it."""
        runs = [{"run_type": "llm", "start_time": "1", "inputs": {"messages": [message]}}]
        traj = runs_to_trajectory(runs, "t")
        assert traj.step(1).role == "user"
        assert traj.step(1).content == "hi"

    def test_tool_run_error_is_flagged(self):
        runs = [
            {"run_type": "tool", "name": "f", "start_time": "1", "error": "kaboom", "outputs": None}
        ]
        step = runs_to_trajectory(runs, "t").step(1)
        assert step.tool_result.is_error
        assert "kaboom" in step.tool_result.content

    def test_chain_runs_are_skipped_as_structure(self):
        runs = [
            {"run_type": "chain", "start_time": "1", "inputs": {"foo": "bar"}},
            {
                "run_type": "llm",
                "start_time": "2",
                "inputs": {"messages": [{"role": "user", "content": "hi"}]},
            },
        ]
        assert len(runs_to_trajectory(runs, "t")) == 1

    def test_child_runs_are_flattened(self, tmp_path):
        payload = {
            "id": "root",
            "trace_id": "t",
            "run_type": "chain",
            "start_time": "0",
            "child_runs": [
                {
                    "run_type": "llm",
                    "trace_id": "t",
                    "start_time": "1",
                    "inputs": {"messages": [{"role": "user", "content": "hi"}]},
                },
                {
                    "run_type": "tool",
                    "trace_id": "t",
                    "name": "f",
                    "start_time": "2",
                    "outputs": "done",
                },
            ],
        }
        trajs = load_langsmith(write(tmp_path, "ls.json", payload))
        assert len(trajs) == 1
        assert [s.role for s in trajs[0].steps] == ["user", "tool"]

    def test_generation_text_without_message(self):
        runs = [
            {
                "run_type": "llm",
                "start_time": "1",
                "inputs": {},
                "outputs": {"generations": [[{"text": "plain completion"}]]},
            }
        ]
        assert runs_to_trajectory(runs, "t").step(1).content == "plain completion"

    def test_tool_calls_extracted(self):
        runs = [
            {
                "run_type": "llm",
                "start_time": "1",
                "inputs": {},
                "outputs": {
                    "generations": [
                        [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "c",
                                            "function": {"name": "f", "arguments": '{"a":1}'},
                                        }
                                    ],
                                }
                            }
                        ]
                    ]
                },
            }
        ]
        call = runs_to_trajectory(runs, "t").step(1).tool_calls[0]
        assert (call.name, call.arguments) == ("f", {"a": 1})

    def test_groups_by_trace_id(self, tmp_path):
        payload = [
            {
                "run_type": "llm",
                "trace_id": "A",
                "start_time": "1",
                "inputs": {"messages": [{"role": "user", "content": "a"}]},
            },
            {
                "run_type": "llm",
                "trace_id": "B",
                "start_time": "1",
                "inputs": {"messages": [{"role": "user", "content": "b"}]},
            },
        ]
        assert {t.trajectory_id for t in load_langsmith(write(tmp_path, "ls.json", payload))} == {
            "A",
            "B",
        }


def test_every_adapter_produces_contiguous_one_based_indices(tmp_path):
    """Pydantic enforces this, so a violation raises rather than scoring wrong."""
    otel = load_otel(
        write(
            tmp_path,
            "o.json",
            [
                otlp_span(
                    "chat",
                    1,
                    {
                        "gen_ai.request.model": "m",
                        "gen_ai.input.messages": json.dumps([{"role": "user", "content": "a"}]),
                    },
                )
            ],
        )
    )
    lf = load_langfuse(
        write(
            tmp_path,
            "l.json",
            {
                "id": "x",
                "observations": [
                    {
                        "type": "GENERATION",
                        "startTime": "1",
                        "input": [{"role": "user", "content": "a"}],
                    }
                ],
            },
        )
    )
    ls = load_langsmith(
        write(
            tmp_path,
            "s.json",
            [
                {
                    "run_type": "llm",
                    "trace_id": "x",
                    "start_time": "1",
                    "inputs": {"messages": [{"role": "user", "content": "a"}]},
                }
            ],
        )
    )

    for trajectories in (otel, lf, ls):
        for traj in trajectories:
            assert [s.index for s in traj.steps] == list(range(1, len(traj) + 1))
