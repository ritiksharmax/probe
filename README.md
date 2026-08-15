# PROBE

**P**roduction **R**oot-cause **O**bservation & **B**ehavioral **E**valuation — detect, localize, and explain AI-agent failures from production execution traces, without ground-truth trajectories.

> **Status: milestone 1, in progress.** The detection and evidence-filtering layers are built and measured on real benchmark data. The localization/attribution comparison table needs model access and is not filled in yet.

## What it does

Point PROBE at an agent trace and it answers three questions:

1. **Did this run fail?** — from cheap, LLM-free signals: tool errors, loops, no-progress windows, refusal phrases, budget anomalies, and declarative policy constraints.
2. **Where did it go wrong?** — the critical step, via a signal-density prior ensembled with an LLM judge.
3. **Why?** — a root cause from a 10-category taxonomy, with supporting evidence and a counterfactual.

```
adapter → Trajectory → signals → detector → evidence filter → localizer + RCA judge → RCAReport
```

## Why it is different

The closest prior work is Microsoft's [AgentRx](https://github.com/microsoft/AgentRx). PROBE targets three gaps in it:

- **Detection.** AgentRx assumes you already know the run failed. PROBE finds failures in the first place, which is what production actually needs.
- **Cost.** AgentRx judges with GPT-5 over the full trajectory. PROBE filters the trajectory down to a few suspect evidence windows first, so a small local model can attempt the diagnosis. *(This is the weakest of the three claims right now — see Results.)*
- **Production ingestion.** PROBE reads OpenTelemetry GenAI spans (and OpenInference), Langfuse, and LangSmith exports, not just benchmark files.

## Ingestion

| Adapter | Source |
|---|---|
| `jsonl` | PROBE's canonical schema (default) |
| `otel` | OTLP/JSON — OTel GenAI semconv **and** OpenInference attributes |
| `langfuse` | Langfuse trace/observation export |
| `langsmith` | LangSmith run export, including nested `child_runs` |
| `agentrx` | The AgentRx benchmark's τ-retail and Magentic-One formats |

Every adapter emits the same `Trajectory`, with steps **chronological and 1-based** regardless of the order records arrived in.

## Install

```bash
pip install probe-agents                 # import name: probe
pip install "probe-agents[anthropic]"    # + the reference judge tier
```

## Use

```bash
probe detect traces.jsonl                        # triage — no LLM calls
probe analyze traces.jsonl                       # full root-cause reports
probe analyze traces.jsonl --tier small --full   # cheap tier, unfiltered (ablation)
probe bench agentrx --detection-only             # reproduce H1 without model access
```

```python
from probe import read_jsonl
from probe.detect import Detector

detector = Detector()
for trajectory in read_jsonl("traces.jsonl"):
    verdict = detector(trajectory)
    if verdict.failed:
        print(verdict.explain())
```

### Domain constraints

The error- and loop-based signals are structurally blind to a failure that *looks* healthy — every tool call succeeds and the agent quietly does the wrong thing. Constraints close that gap:

```python
from probe.signals import ConstraintSignal, default_signals, require_before

constraints = [
    require_before(
        name="authenticate_first",
        description="Authenticate the user before disclosing order details.",
        prerequisite="find_user_id_by_name_zip",
        dependent="get_order_details",
    ),
]
detector = Detector(signals=[*default_signals(), ConstraintSignal(constraints)])
```

`examples/demo_agent.py` shows the difference this makes: a refund of $200 against a $20 order scores **0.00** with the default battery and **0.90** with constraints, while the healthy run stays at 0.00.

## Results so far

Measured on the public AgentRx data — 73 trajectories (τ-retail 29, Magentic-One 44). See [`benchmarks/agentrx/README.md`](benchmarks/agentrx/README.md) for methodology and an important scope caveat: the Flash domain is unpublished, so this is **not** the paper's 115-trajectory aggregate.

**Detection (H1)** — τ-retail, 29 annotated failures vs 73 successful runs:

| | recall | false-positive rate |
|---|---:|---:|
| hand-set weights | 0.586 | 0.301 |
| calibrated, 5-fold CV | 0.552 | **0.192** |

Calibration cuts false positives by a third. Numbers are cross-validated — a naive-Bayes fit reported on its own training data at n=29 would be meaningless.

**Localization floor** — `signals-only`, no LLM:

| | exact | ±1 | ±5 |
|---|---:|---:|---:|
| τ-retail | 0.034 | 0.172 | 0.379 |
| Magentic-One | 0.136 | 0.159 | 0.273 |

On τ-retail this beats the trivial last-step and midpoint baselines at every tolerance. **On Magentic-One it loses to picking the last step** — Magentic trajectories are flat prose with no parsed tool calls, so every tool-level signal is blind there. That is a known gap, not a tuned result.

## Development

```bash
uv venv && uv pip install -e ".[dev,bench]"
uv run pytest          # 216 tests, fully offline — no network, no API key
uv run ruff check .
uv run python examples/demo_agent.py
```

The test suite never touches the network. Benchmark data is fetched separately and pinned to an upstream commit SHA.

## License

MIT
