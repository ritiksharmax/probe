"""Tests for the OpenAI-compatible client: thinking models and transport resilience.

Both areas cost real benchmark runs before they were handled, so they are pinned
here: a thinking model's reasoning silently poisoning JSON extraction, and a
dropped tunnel silently turning 251 failed calls into zeros that looked like
results.
"""

from __future__ import annotations

import httpx
import pytest

from probe.llm.client import (
    OpenAICompatibleClient,
    build_client,
    parse_json,
    split_reasoning,
)


class TestSplitReasoning:
    def test_closed_think_block(self):
        answer, reasoning = split_reasoning('<think>hmm</think>{"a": 1}')
        assert answer == '{"a": 1}'
        assert "hmm" in reasoning

    def test_pre_opened_template_leaves_only_a_closing_tag(self):
        """Qwen3-Thinking's chat template opens <think>, so completions lack it."""
        answer, reasoning = split_reasoning('deliberating...\n</think>\n\n{"a": 1}')
        assert answer == '{"a": 1}'
        assert "deliberating" in reasoning

    def test_reasoning_containing_a_draft_object_does_not_win(self):
        """The failure this exists to prevent: parsing the model's rejected draft."""
        raw = (
            'Maybe {"critical_step": 99, "category": 10}? No, that is wrong.\n'
            '</think>\n{"critical_step": 7, "category": 4}'
        )
        answer, _ = split_reasoning(raw)
        assert parse_json(answer) == {"critical_step": 7, "category": 4}
        # Parsing the raw completion would have returned the discarded draft.
        assert parse_json(raw)["critical_step"] == 99

    def test_no_reasoning_passes_through(self):
        answer, reasoning = split_reasoning('{"a": 1}')
        assert answer == '{"a": 1}'
        assert reasoning == ""

    def test_unclosed_think_yields_no_answer(self):
        """Cut off mid-thought: there is no answer, and inventing one is worse."""
        answer, reasoning = split_reasoning("<think>still going and then truncated")
        assert answer == ""
        assert "still going" in reasoning

    def test_empty(self):
        assert split_reasoning("") == ("", "")

    def test_case_insensitive_tags(self):
        answer, _ = split_reasoning("<THINK>x</THINK>done")
        assert answer == "done"


class TestStructuredOutputToggle:
    def test_thinking_models_default_to_free_form(self):
        """Guided decoding would leave a thinking model no room to think."""
        client = build_client("small", model="qwen3-4b-thinking")
        assert client.structured_output is False

    def test_non_thinking_models_keep_structured_output(self):
        assert build_client("small", model="qwen3:4b").structured_output is True

    def test_explicit_override_wins(self):
        client = build_client("small", model="qwen3-4b-thinking", structured_output=True)
        assert client.structured_output is True


class TestTransport:
    def _client(self, handler, **kw):
        client = OpenAICompatibleClient("m", base_url="http://x/v1", **kw)
        return client, handler

    def test_reads_reasoning_content_when_the_server_splits_it(self, monkeypatch):
        body = {
            "model": "m",
            "choices": [
                {
                    "message": {"content": '{"a": 1}', "reasoning_content": "thought"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }
        client = OpenAICompatibleClient("m", base_url="http://x/v1")
        monkeypatch.setattr(client, "_post_with_retry", lambda payload: body)
        response = client.complete("hi")
        assert response.text == '{"a": 1}'
        assert response.reasoning == "thought"

    def test_reads_the_alternate_reasoning_key(self, monkeypatch):
        body = {
            "model": "m",
            "choices": [{"message": {"content": "x", "reasoning": "why"}, "finish_reason": "stop"}],
            "usage": {},
        }
        client = OpenAICompatibleClient("m", base_url="http://x/v1")
        monkeypatch.setattr(client, "_post_with_retry", lambda payload: body)
        assert client.complete("hi").reasoning == "why"

    def test_schema_is_omitted_when_structured_output_is_off(self, monkeypatch):
        seen = {}

        def capture(payload):
            seen.update(payload)
            return {"model": "m", "choices": [{"message": {"content": "{}"}}], "usage": {}}

        client = OpenAICompatibleClient("m", base_url="http://x/v1", structured_output=False)
        monkeypatch.setattr(client, "_post_with_retry", capture)
        client.complete("hi", schema={"type": "object"})
        assert "response_format" not in seen

    def test_schema_is_sent_when_structured_output_is_on(self, monkeypatch):
        seen = {}

        def capture(payload):
            seen.update(payload)
            return {"model": "m", "choices": [{"message": {"content": "{}"}}], "usage": {}}

        client = OpenAICompatibleClient("m", base_url="http://x/v1", structured_output=True)
        monkeypatch.setattr(client, "_post_with_retry", capture)
        client.complete("hi", schema={"type": "object"})
        assert seen["response_format"]["type"] == "json_schema"


class TestRetry:
    def test_retries_transport_errors_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky(url, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(
                200,
                json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx, "post", flaky)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        client = OpenAICompatibleClient("m", base_url="http://x/v1", max_retries=4)
        assert client.complete("hi").text == "ok"
        assert calls["n"] == 3

    def test_gives_up_with_a_clear_error_rather_than_returning_a_zero(self, monkeypatch):
        """A silent zero is indistinguishable from a wrong answer in the results."""
        monkeypatch.setattr(
            httpx, "post", lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("nope"))
        )
        monkeypatch.setattr("time.sleep", lambda _s: None)
        client = OpenAICompatibleClient("m", base_url="http://x/v1", max_retries=2)
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            client.complete("hi")

    def test_retries_transient_5xx(self, monkeypatch):
        calls = {"n": 0}

        def flaky(url, **kwargs):
            calls["n"] += 1
            request = httpx.Request("POST", url)
            if calls["n"] == 1:
                return httpx.Response(503, text="unavailable", request=request)
            return httpx.Response(
                200,
                json={"model": "m", "choices": [{"message": {"content": "ok"}}], "usage": {}},
                request=request,
            )

        monkeypatch.setattr(httpx, "post", flaky)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        client = OpenAICompatibleClient("m", base_url="http://x/v1")
        assert client.complete("hi").text == "ok"
        assert calls["n"] == 2

    def test_does_not_retry_a_client_error(self, monkeypatch):
        """A 400 is a bug in the request; retrying just wastes minutes."""
        calls = {"n": 0}

        def bad(url, **kwargs):
            calls["n"] += 1
            return httpx.Response(400, text="bad request", request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", bad)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        client = OpenAICompatibleClient("m", base_url="http://x/v1")
        with pytest.raises(httpx.HTTPStatusError):
            client.complete("hi")
        assert calls["n"] == 1
