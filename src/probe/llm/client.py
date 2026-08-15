"""Pluggable LLM clients for the tiered judge.

Two backends, because the tiers genuinely differ in protocol:

* `AnthropicClient` — the reference tier, via the official `anthropic` SDK.
* `OpenAICompatibleClient` — the cheap tier, via plain HTTP against any
  OpenAI-compatible endpoint (Ollama, vLLM, Groq, Together). There is no
  Anthropic SDK for those servers, so httpx is the right tool rather than a
  shim.

`FakeClient` completes the set: the judge, the localizer and the whole benchmark
harness are testable end-to-end without a network or an API key, which is what
lets the pipeline be built and verified before credentials exist.

Every call returns token counts, cost and latency. That accounting is not
incidental — the headline claim of this project is that filtered evidence lets a
small model match a frontier one at a fraction of the cost, and a claim like that
is only as good as its measurement.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

# USD per million tokens, (input, output). Anthropic list prices as of 2026-06.
# Local models are free to run, so they price at zero and the efficiency claim
# rests on latency and hardware rather than API spend.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost in USD for a call, or 0.0 for models with no list price."""
    for name, (in_rate, out_rate) in PRICING.items():
        if model.startswith(name):
            return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
    return 0.0


@dataclass
class LLMResponse:
    """One completion, with the accounting the benchmark needs."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    # Number of parse-repair retries spent getting valid JSON. Small models need
    # these often enough that hiding them would flatter the cheap tier.
    repairs: int = 0
    stop_reason: str | None = None
    # Reasoning a thinking model emitted before its answer. Held separately so it
    # never reaches the JSON parser, and so the cheap tier's real token spend
    # stays visible — thinking tokens are billed and they are not free.
    reasoning: str = ""

    def json(self) -> dict[str, Any]:
        """Parse the response as JSON, tolerating prose or fences around it."""
        return parse_json(self.text)


# Qwen3-Thinking and friends wrap reasoning in <think>…</think>. Their chat
# template usually *pre-opens* the tag, so a completion often carries only the
# closing one — hence both shapes below.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_HEAD = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate a thinking model's reasoning from its actual answer.

    This has to happen *before* JSON extraction, not after. A thinking model
    deliberating about what JSON to emit will write brace-laden prose — often
    including a rejected draft of the very object we want — and a parser pointed
    at the raw completion happily returns the draft. That failure is silent and
    scores as a wrong answer rather than an error, which is the worst kind.

    Returns `(answer, reasoning)`; reasoning is empty when there is none.
    """
    if not text:
        return "", ""

    if match := _THINK_BLOCK.search(text):
        reasoning = match.group(0)
        return (text[: match.start()] + text[match.end() :]).strip(), reasoning.strip()

    # Template pre-opened the tag: everything up to the first </think> is thinking.
    if "</think>" in text.lower():
        head = _THINK_UNCLOSED_HEAD.match(text)
        if head:
            return text[head.end() :].strip(), head.group(0).strip()

    # Opened but never closed — the model was cut off mid-thought. There is no
    # answer to salvage, and returning the reasoning as if it were one would
    # invent a diagnosis the model never made.
    if _THINK_OPEN.search(text):
        return "", text.strip()

    return text.strip(), ""


class LLMClient(Protocol):
    """Anything the judge can call."""

    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response.

    Tries the whole string, then a fenced block, then the outermost brace-balanced
    span. Small models wrap JSON in prose and fences even when told not to, and
    discarding those responses would misattribute a formatting quirk to a
    reasoning failure.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass

    if match := _FENCE.search(text):
        try:
            return json.loads(match.group(1).strip())
        except (TypeError, ValueError):
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except (TypeError, ValueError):
                        break

    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


class AnthropicClient:
    """Reference tier, via the official Anthropic SDK.

    Note what is deliberately *not* sent: `temperature`, `top_p` and `top_k` are
    rejected with a 400 on Claude Opus 5, and `thinking.budget_tokens` is gone
    across the current generation. Thinking is adaptive and on by default, and it
    shares the `max_tokens` budget with the response — hence the generous
    default, since a tight cap truncates the answer rather than the reasoning.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        effort: str = "high",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "the anthropic package is required for the reference tier; "
                "install it with `pip install probe-agents[anthropic]`"
            ) from exc

        self.model = model
        self.effort = effort
        # A bare client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile — an unset env var does not mean no credentials.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if schema:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}

        started = time.perf_counter()
        message = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - started

        # A safety decline returns HTTP 200 with an empty content list, so
        # indexing content[0] unconditionally would raise here.
        if message.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in message.content if b.type == "text")

        usage = message.usage
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        return LLMResponse(
            text=text,
            model=message.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cost_usd=price(message.model, prompt_tokens, completion_tokens),
            stop_reason=message.stop_reason,
        )


class OpenAICompatibleClient:
    """Cheap tier, via any OpenAI-compatible `/chat/completions` endpoint.

    Covers Ollama, vLLM, Groq and Together with one implementation. JSON is
    requested through `response_format` where the server honours it; servers that
    ignore it fall back to the repair loop in the judge, and repairs are counted
    so the cheap tier's real cost stays visible.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        temperature: float = 0.0,
        structured_output: bool = True,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:11434/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.timeout = timeout
        self.temperature = temperature
        # Guided decoding against a JSON schema forces the very first token to
        # open the object, which leaves a *thinking* model no room to think --
        # it silently becomes a non-thinking model, which is the opposite of why
        # you would choose one. Turn this off for reasoning models and let the
        # client split <think> from the answer instead.
        self.structured_output = structured_output
        # Self-hosted endpoints flake: a tunnel drops, a server restarts, a load
        # balancer 503s. Without retries a single blip silently voids an entire
        # benchmark run -- which is exactly how one run here lost 251 of 292
        # calls to a dead SSH tunnel and reported zeros as if they were results.
        self.max_retries = max_retries

    _RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on transport errors and transient 5xx.

        A 4xx that is not rate limiting is a bug in the request, so it is raised
        immediately rather than retried into a long, pointless wait.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code in self._RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and (
                    exc.response.status_code not in self._RETRY_STATUS
                ):
                    raise
                last = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2.0**attempt, 30.0))

        raise RuntimeError(f"{url} failed after {self.max_retries + 1} attempts: {last}") from last

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if schema and self.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }

        started = time.perf_counter()
        body = self._post_with_retry(payload)
        latency = time.perf_counter() - started

        choice = body["choices"][0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        content = message.get("content") or ""
        # Servers running a reasoning parser hand back a clean `content` plus the
        # thinking in a separate field — spelled `reasoning_content` by some
        # builds and `reasoning` by others. Servers without a parser inline the
        # whole <think> block in `content`. Handle all three, so correctness does
        # not depend on how somebody configured the endpoint.
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not reasoning:
            content, reasoning = split_reasoning(content)

        return LLMResponse(
            text=content,
            model=body.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cost_usd=price(self.model, prompt_tokens, completion_tokens),
            stop_reason=choice.get("finish_reason"),
            reasoning=reasoning,
        )


@dataclass
class FakeClient:
    """Scripted client for tests and offline development.

    Records every prompt it sees, so tests can assert on what the evidence filter
    actually put in front of the judge — which is the substance of the H2 claim,
    not an implementation detail.
    """

    responses: list[str] = field(default_factory=list)
    model: str = "fake-model"
    prompts: list[str] = field(default_factory=list, init=False)
    systems: list[str | None] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        if not self.responses:
            raise RuntimeError("FakeClient has no scripted responses")
        raw = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        # Split <think> exactly as the HTTP clients do, so a test that scripts a
        # reasoning-model response exercises the real code path rather than a
        # convenient fiction.
        text, reasoning = split_reasoning(raw)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(raw) // 4,
            latency_s=0.0,
            cost_usd=0.0,
            reasoning=reasoning,
        )


def build_client(tier: str, model: str | None = None, **kwargs: Any) -> LLMClient:
    """Construct the client for a named tier.

    `frontier` is the Anthropic reference tier; `small` is the cheap
    OpenAI-compatible tier, defaulting to a local Qwen3 on Ollama.
    """
    if tier == "frontier":
        return AnthropicClient(model=model or "claude-opus-5", **kwargs)
    if tier == "small":
        name = model or "qwen3:4b"
        # Reasoning models must be left free to emit their <think> block.
        kwargs.setdefault("structured_output", "thinking" not in name.lower())
        return OpenAICompatibleClient(model=name, **kwargs)
    raise ValueError(f"unknown tier {tier!r}; expected 'frontier' or 'small'")
