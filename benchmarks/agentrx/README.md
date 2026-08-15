# AgentRx benchmark

Reproduces PROBE's head-to-head comparison on the public [AgentRx](https://github.com/microsoft/AgentRx) benchmark data ([paper](https://arxiv.org/abs/2602.02475)).

```bash
uv run python benchmarks/agentrx/run.py --detection-only        # H1 only, no LLM
uv run python benchmarks/agentrx/run.py --system signals-only   # localization floor, no LLM
uv run python benchmarks/agentrx/run.py --tier frontier         # full table (needs model access)
uv run python benchmarks/agentrx/run.py --tier frontier --tier small --out results.json
```

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

Judges are non-deterministic, so treat a single run's accuracy as a point estimate; `protocol.aggregate_seeds` reports mean ± population std across repeated runs of the same configuration.

## Results

Not filled in yet — the localization and attribution table needs model access. Detection and the LLM-free floor run today:

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
