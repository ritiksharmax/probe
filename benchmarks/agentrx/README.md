# AgentRx benchmark

Reproduces PROBE's head-to-head comparison on the public [AgentRx](https://github.com/microsoft/AgentRx) benchmark data ([paper](https://arxiv.org/abs/2602.02475)).

```bash
uv run python benchmarks/agentrx/run.py --detection-only        # H1 only, no LLM
uv run python benchmarks/agentrx/run.py --system signals-only   # localization floor, no LLM
uv run python benchmarks/agentrx/run.py --tier frontier         # full table (needs model access)
uv run python benchmarks/agentrx/run.py --tier frontier --tier small --out results.json
```

## Serving the small tier

The cheap tier is **`Qwen/Qwen3-4B-Thinking-2507-FP8`** on an H100. FP8 is not a
compromise here: Hopper has native FP8 tensor cores, Qwen publishes the FP8
checkpoint themselves (it is not a community quant), and at 4.85 GiB it is the
only variant that fits — the bf16 checkpoint is 7.51 GiB and the target host had
5.5 GiB free.

```bash
export HF_HOME=/path/with/room
vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 \
  --served-model-name qwen3-4b-thinking \
  --host 127.0.0.1 --port 8010 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32

uv run python benchmarks/agentrx/run.py \
  --tier small --model qwen3-4b-thinking \
  --base-url http://localhost:8010/v1 --max-tokens 16384 --concurrency 12
```

Bind the server to **loopback** and reach it over an SSH tunnel. An
unauthenticated model server on a host with a public IP is an open proxy to
someone else's GPU.

### Three things a *thinking* model changes

1. **Do not send a JSON schema.** Guided decoding forces the first token to open
   the object, so the model never gets to think — a thinking model silently
   becomes a non-thinking one. Measured here: 87 completion tokens with a schema
   versus 3,633 without. `build_client` therefore disables `structured_output`
   automatically for any model whose name contains `thinking`.
2. **Split `<think>` before parsing JSON.** Qwen's chat template pre-opens the
   tag, so a completion carries only the closing `</think>`. The reasoning
   routinely contains a *draft* of the answer object — a parser pointed at the
   raw completion returns the draft, which scores as a wrong answer rather than
   an error. `probe.llm.client.split_reasoning` handles this, and reads
   `reasoning_content`/`reasoning` when the server has a reasoning parser.
3. **Budget output generously.** Thinking shares the `max_tokens` budget with the
   answer; 16384 is a reasonable floor. Too tight and the answer is truncated
   while the reasoning survives.

### Run it concurrently

`--concurrency` matters more than it looks. Serial execution leaves the GPU idle
between calls: the first run here took over an hour at `Running: 1 reqs` while
the server was configured for 32. At `--concurrency 12` the same work is minutes.

## Scope — read this before quoting any number

The paper evaluates on **115** trajectories across three domains. **The Flash domain (42 trajectories, incident management) is not published** — it appears in neither the GitHub repo nor the Hugging Face dataset. What is publicly obtainable is:

| Domain | Trajectories | Status |
|---|---:|---|
| τ-retail | 29 | public |
| Magentic-One | 44 | public |
| Flash | 42 | **unavailable** |
| **Total** | **73** of 115 | |

So every result here is reported **per domain**, and none of it is comparable to the paper's 115-trajectory aggregate. Any single "PROBE beats AgentRx" number computed across these 73 would be a different measurement wearing the paper's clothes.

Note also that the Hugging Face dataset (`microsoft/AgentRx`) is **access-gated** — it returns 401 unauthenticated, and it carries no trajectories the GitHub repo lacks. This harness therefore pulls from the public GitHub repo, pinned to a commit SHA, and needs no credentials.

## What is being compared

The paper publishes only *relative* gains — +23.6% step localization and +22.9% root-cause attribution over a Who&When baseline with a GPT-5 judge — and no absolute accuracies. There is no published number to score against, so **every comparison point here is run by us**, on the same trajectories, with the same judge model.

The systems form a 2×2 over PROBE's two contributions, plus an LLM-free floor:

| | full trajectory | filtered evidence |
|---|---|---|
| **no violation log** | `naive-full` | `filtered-only` |
| **with violation log** | `agentrx-style` | `probe` |

- **`naive-full`** — the prompting baseline: whole trajectory, no violation log.
- **`agentrx-style`** — a faithful reproduction of AgentRx's shape: violation log + whole trajectory + judge. This is the comparison that matters.
- **`probe`** — both contributions together.
- **`filtered-only`** / **`agentrx-style`** — the off-diagonal cells, which attribute the difference between the other two. Without them, "probe beats naive-full" would have two possible explanations.
- **`signals-only`** — the LLM-free floor. Whatever a judge buys is measured against this, not against zero.

## Evaluation protocol

Mirrors `agentrx/reports/analyze_metrics.py` from the upstream repo, so the numbers mean the same thing theirs do.

- **Localization** — exact match plus ±1…±5 step tolerance, scored against the step number of the *root-cause* failure: the one whose `failure_id` equals `root_cause.failure_id`, not merely the first annotated failure.
- **Attribution** — four variants, because a trajectory carries several annotated failures and "correct category" is genuinely ambiguous: against the root cause, against *any* annotated failure, against the earliest, and against the terminal one.
- **Taxonomy** — **10** categories, not 9; the tenth is `Inconclusive`. Both sides of every comparison pass through the alias rules in `probe.rca.taxonomy`, ported from upstream, because category names differ across the benchmark's own domains (τ says `Instruction Adherence Failure`, Magentic says `Instruction/Plan Adherence Failure`; both are case 1).
- **Detection (H1)** — recall on the 29 annotated τ failures, false-positive rate on the 73 successful τ runs in `tau_dataset_full.json`, which are disjoint from every annotated failure. Magentic-One publishes no successful runs, so detection is τ-only and labelled as such. Calibrated numbers are **5-fold cross-validated** — a naive-Bayes fit reported on its own training data would be meaningless at n=29.
- **`win_rec` (window recall)** — the fraction of trajectories whose true critical step fell inside an evidence window. This is the **hard ceiling on `exact` for any filtered system**: a judge cannot name a step it was never shown. It needs no LLM to compute, so it is reported next to the accuracy it bounds.

## The filtering problem, measured

The first honest result out of this harness is a negative one about PROBE's own central bet.

Signals are good at *detection* and poor at *localization*. Measured against ground truth on all 73 trajectories:

| | τ-retail | Magentic-One |
|---|---:|---:|
| a signal fires **exactly on** the true critical step | 6.9% | 15.9% |
| a signal fires within ±2 | 24.1% | 25.0% |
| median distance from true step to nearest signal | **6 steps** | **8 steps** |
| trajectories with no signals at all | 5 | 1 |

Signals fire on *consequences* — the tool error, the refusal, the stall at the end — while the annotated root cause is the earlier decision that made them inevitable. A window centred on a signal therefore misses the step it is meant to capture.

Two consequences, both acted on:

1. **Windows are now asymmetric** (`look_back=8`, `look_forward=2`), because causes precede symptoms. At equal token budget this beats widening symmetrically: `back=8,fwd=2` reaches 58.6% window recall on τ versus 51.7% for a symmetric `radius=6`.
2. **The original default was badly mistuned.** `radius=2, max_windows=3` compressed to 16–21% of steps but had a ceiling of only **27.6% / 36.4%** — it was throwing away the answer roughly 70% of the time. The current default reaches **58.6% / 65.9%**.

Note the honest caveat: those filter parameters were chosen by looking at this benchmark's ground truth, so they carry some selection bias. Window recall is reported on every run precisely so that the ceiling stays visible rather than being folded silently into an accuracy number.

### Step indexing

Ground-truth `step_number` is **1-based, one step per message**, verified in `agentrx/ir/trajectory_ir.py` (`"index": i + 1`) and checked against the annotations themselves. τ's leading `system` policy message is step 1.

Two alignments are pinned as golden tests in `tests/test_adapter_agentrx.py`:

- τ task 2, annotated steps 3 and 7 → the assistant turn calling `list_all_product_types` and the assistant turn claiming "11 available T-shirt options" — matching their annotated reasons.
- Magentic `5f982798-…`, annotated steps 13 and 17 → the two `WebSurfer` turns its annotations describe.

An off-by-one here would shift every localization number without breaking anything loudly, which is why it is asserted rather than assumed.

### One deliberate divergence

Upstream resolves the root cause with a strict `f["failure_id"] == root_cause["failure_id"]`. In `magentic_one_ground_truth.json`, exactly one trajectory (`08cae58d-…`) stores its `root_cause.failure_id` as the string `"1"` while its `failures[].failure_id` are ints, so upstream silently fails to resolve it and scores it with no root-cause step. We coerce both sides to `str`, which resolves all 73. PROBE and every baseline are scored under identical rules, so the comparison stays fair; the effect is that our denominator is complete.

## Reproducing

Data is pinned to commit `f228165bfec60a801fd5fedd9d8ffe0f9de0c69d` and cached under `.cache/`. Bump `AGENTRX_SHA` in `fetch.py` deliberately, and re-run when you do.

Judges are non-deterministic, so treat a single run's accuracy as a point estimate. `--repeats N` re-runs each configuration and `protocol.aggregate_repeats` reports mean ± population std. Requests are identical across repeats (temperature 0), so what this measures is **serving non-determinism** — the run-to-run floor any claimed difference has to clear — not sampling variance.

Every results file records the probe commit, whether the tree was dirty, the argv and the config alongside the data SHA. A results file that cannot be attributed to a commit cannot be defended, and `results/` is tracked for the same reason.

Read a finished run back with the reporter rather than eyeballing the table — it excludes contaminated rows from the pooled numbers instead of averaging them in, refuses to pool a system across a domain it has no valid row for, and restates every spread in trajectories:

```bash
uv run python benchmarks/agentrx/report.py benchmarks/agentrx/results/clean-repeats3.json
```

## Results

### Localization and attribution — the honest read

3 repeats per configuration, `qwen3-4b-thinking`, all 73 public trajectories. **PROBE's filtering does not improve localization. The plain prompting baseline wins.**

Pooled over both domains (n=73):

| system | exact | ±5 | cat(rc) | tokens/diagnosis |
|---|---:|---:|---:|---:|
| `naive-full` | **0.187** | 0.461 | 0.128 | 7,837 |
| `agentrx-style` | 0.128 | **0.466** | 0.132 | 8,226 |
| `filtered-only` | 0.123 | 0.443 | **0.146** | **6,197** |
| `probe` | 0.110 | 0.470 | 0.100 | 7,065 |
| `signals-only` | 0.096 | 0.315 | 0.000 | 0 |

`naive-full` — whole trajectory, no violation log, the thing PROBE was supposed to beat — leads exact localization by 0.06–0.08 over every system that adds machinery. That gap is 4–6 trajectories against a pooled spread of 0.011–0.037, and it is the one ordering in this table that reproduces.

**Almost nothing else here does.** Two independent 3-repeat runs of the identical configuration rank the systems differently:

| ranking | run A | run B |
|---|---|---|
| τ exact | naive-full > agentrx-style > probe | agentrx-style > naive-full > probe |
| τ cat(rc) | **agentrx-style** > probe > naive-full | filtered-only > naive-full > probe > **agentrx-style** |
| Magentic exact | probe > agentrx-style | naive-full > filtered-only > signals-only > probe > agentrx-style |

`agentrx-style` is *first* on τ attribution in run A and *last* in run B. An earlier version of this section claimed "the violation log helps attribution" on the strength of run A, where it led in both domains. **That is withdrawn too** — run B inverts it on τ, and the whole τ attribution field spans 0.103–0.138, a range of exactly one trajectory. At n=29 no attribution ordering is measurable.

Reproduce the run A vs run B comparison above directly:

```bash
uv run python benchmarks/agentrx/report.py \
  benchmarks/agentrx/results/clean-repeats3.json \
  --compare benchmarks/agentrx/results/qwen3-seeds3.json
```

An ordering that differs between two runs of the same configuration is noise, regardless of how clean either looked alone.

What survives:

1. **`naive-full` leads localization** — reproduces, and is the only claim here with a gap larger than its spread.
2. **A judge beats the LLM-free floor on ±5 but barely on exact** — `signals-only` scores 0.096 exact against `probe`'s 0.110. Most of what the judge buys is proximity, not precision.
3. **Filtering is cheaper, not better** — `filtered-only` uses the fewest tokens of any judged system (6,197 vs `agentrx-style`'s 8,226, ~25% less) at statistically indistinguishable accuracy. Read that as "the filter is free", not "the filter helps". The `$` column reads `0.0000` for a self-hosted model, so judge cost by the token column.

A prior single-run A/B of the signal caveat was reported as "more than doubling" τ exact match (0.103 → 0.241). **Withdrawn**: one run per arm against this noise floor, and the control systems — which cannot see the caveat at all — moved just as much between the same two runs. `--no-signal-caveat` now makes that ablation reproducible from a commit; it needs re-running with repeats before any number is quoted.

### Detection and the LLM-free floor

These need no model access:

| metric | τ-retail | Magentic-One |
|---|---|---|
| detection recall (hand-set weights) | 0.586 | n/a — no successful runs published |
| detection FPR (hand-set weights) | 0.301 | n/a |
| detection recall (calibrated, 5-fold CV) | 0.552 | n/a |
| detection FPR (calibrated, 5-fold CV) | **0.192** | n/a |
| localization exact — `signals-only` | 0.034 | 0.136 |
| localization ±1 — `signals-only` | 0.172 | 0.159 |
| localization ±5 — `signals-only` | 0.379 | 0.273 |

Two things worth stating plainly about the floor:

1. On τ-retail, `signals-only` beats the trivial last-step and midpoint baselines at every tolerance.
2. On Magentic-One it **loses to picking the last step** (0.205 / 0.250 / 0.341 exact/±1/±5). The cause is structural: Magentic trajectories are flat prose messages with no parsed tool calls, so every tool-level signal is blind there — `WebSurfer` reporting "I clicked 'PDF'" is text, not a `ToolCall`. Parsing agent actions out of Magentic prose is the fix, and Magentic localization numbers should not be trusted until it lands.
