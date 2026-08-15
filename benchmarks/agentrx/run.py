"""Run the AgentRx benchmark and emit the comparison table.

    uv run python benchmarks/agentrx/run.py --tier frontier
    uv run python benchmarks/agentrx/run.py --tier frontier --tier small --repeats 3
    uv run python benchmarks/agentrx/run.py --detection-only   # no LLM needed

Scope caveat, restated wherever the results are: the Flash domain (42 of the
paper's 115 trajectories) is not published, so this runs on the 73 that are —
tau-retail (29) and Magentic-One (44) — and reports them per domain. These
numbers are not comparable to the paper's 115-trajectory aggregate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import DEFAULT_SYSTEMS, SYSTEMS, build_system  # noqa: E402
from fetch import AGENTRX_SHA, fetch  # noqa: E402
from protocol import (  # noqa: E402
    GroundTruthEntry,
    Prediction,
    aggregate_repeats,
    as_records,
    load_ground_truth,
    score,
    score_detection,
)

from probe.detect import cross_val_report  # noqa: E402
from probe.llm.client import build_client  # noqa: E402
from probe.trace.adapters.agentrx import load_magentic, load_tau  # noqa: E402
from probe.trace.model import Trajectory  # noqa: E402


@dataclass
class Domain:
    name: str
    trajectories: dict[str, Trajectory]
    ground_truth: dict[str, GroundTruthEntry]


def load_domains(paths) -> list[Domain]:
    """Load both public domains, keyed by trajectory id."""
    tau_gt = load_ground_truth(paths.tau_ground_truth, "tau_retail")
    tau = {t.trajectory_id: t for t in load_tau(paths.tau_failed)}
    # A couple of annotated tau runs live only in the full dump.
    for traj in load_tau(paths.tau_full):
        tau.setdefault(traj.trajectory_id, traj)

    magentic_gt = load_ground_truth(paths.magentic_ground_truth, "magentic_one")
    magentic = {t.trajectory_id: t for t in load_magentic(paths.magentic_dir)}

    return [
        Domain("tau_retail", tau, tau_gt),
        Domain("magentic_one", magentic, magentic_gt),
    ]


def run_detection(paths) -> dict:
    """H1: recall on annotated failures vs false positives on successful runs.

    tau-retail only — Magentic-One publishes no successful runs, so there is no
    negative set for it and reporting a false-positive rate would be fabricated.
    """
    failed = load_tau(paths.tau_failed)
    successes = [t for t in load_tau(paths.tau_full) if t.reward == 1.0]
    failed_ids = {t.trajectory_id for t in failed}
    successes = [t for t in successes if t.trajectory_id not in failed_ids]

    trajectories = failed + successes
    labels = [True] * len(failed) + [False] * len(successes)

    from probe.detect import Detector

    detector = Detector()
    preds = [
        Prediction(t.trajectory_id, None, None, predicted_failed=detector(t).failed)
        for t in trajectories
    ]
    uncalibrated = score_detection(
        preds,
        failed_ids=failed_ids,
        succeeded_ids={t.trajectory_id for t in successes},
    )
    calibrated = cross_val_report(trajectories, labels, folds=5)
    return {
        "domain": "tau_retail",
        "n_failed": len(failed),
        "n_succeeded": len(successes),
        "uncalibrated": uncalibrated,
        "calibrated_cv": {
            "recall": calibrated.recall,
            "false_positive_rate": calibrated.false_positive_rate,
            "precision": calibrated.precision,
            "f1": calibrated.f1,
            "folds": calibrated.folds,
        },
    }


def run_system(
    system, domain: Domain, limit: int | None = None, concurrency: int = 8
) -> list[Prediction]:
    """Diagnose every annotated trajectory in a domain.

    Trajectories are independent, and a served model handles many requests at
    once — running them one at a time leaves the GPU idle between calls and turns
    a minutes-long benchmark into an hours-long one. Threads are the right tool
    because the work is entirely HTTP wait.
    """
    ids = [t for t in domain.ground_truth if t in domain.trajectories]
    if limit:
        ids = ids[:limit]

    interactive = sys.stderr.isatty()
    done = 0

    def diagnose(tid: str) -> Prediction:
        nonlocal done
        try:
            report = system(domain.trajectories[tid])
        except Exception as exc:  # noqa: BLE001 - one bad trace must not sink the run
            print(f"    {tid}: {exc}", file=sys.stderr)
            return Prediction(trajectory_id=tid, step=None, category=None, error=str(exc))
        finally:
            done += 1
            if interactive:
                print(f"    [{done}/{len(ids)}]".ljust(40), end="\r", file=sys.stderr)
        return Prediction(
            trajectory_id=tid,
            step=report.critical_step,
            category=report.category_case,
            prompt_tokens=report.prompt_tokens,
            completion_tokens=report.completion_tokens,
            cost_usd=report.cost_usd,
            latency_s=report.latency_s,
            windows=tuple(report.windows),
        )

    if concurrency <= 1:
        predictions = [diagnose(t) for t in ids]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            predictions = list(pool.map(diagnose, ids))

    if interactive:
        print(" " * 44, end="\r", file=sys.stderr)
    return predictions


# One source of truth for the table: parallel `headers`/`keys` lists drifted
# apart once and crashed a completed 25-minute run at the final print, losing
# every diagnosis. Pairs cannot drift.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("domain", "domain"),
    ("system", "system"),
    ("n", "n"),
    ("err", "err"),
    ("exact", "exact"),
    ("±1", "±1"),
    ("±3", "±3"),
    ("±5", "±5"),
    ("cat(rc)", "root_cause_cat"),
    ("cat(any)", "any_cat"),
    ("win_rec", "win_rec"),
    ("in_tok", "in_tok"),
    ("out_tok", "out_tok"),
    ("tok", "tok"),
    ("$", "cost_usd"),
    ("s", "latency_s"),
)

_RATE_KEYS = frozenset({"exact", "±1", "±3", "±5", "root_cause_cat", "any_cat", "win_rec"})


def format_table(rows: list[dict]) -> str:
    """Fixed-width comparison table, so results are readable without rich."""

    def cell(row: dict, key: str) -> str:
        value = row.get(key)
        if value is None:
            return "-"
        if key in _RATE_KEYS:
            return f"{value:.3f}"
        if key in {"in_tok", "out_tok", "tok"}:
            return f"{value:,.0f}"
        if key == "cost_usd":
            return f"{value:.4f}"
        if key == "latency_s":
            return f"{value:.1f}"
        return str(value)

    headers = [h for h, _ in COLUMNS]
    body = [[cell(r, k) for _, k in COLUMNS] for r in rows]
    widths = [max(len(line[i]) for line in [headers, *body]) for i in range(len(COLUMNS))]

    def render(line: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True))

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([render(headers), rule, *(render(line) for line in body)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AgentRx benchmark.")
    parser.add_argument("--tier", action="append", default=[], choices=["frontier", "small"])
    parser.add_argument("--model", default=None, help="override the model for the tier")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible endpoint for the small tier"
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="output budget per judge call")
    parser.add_argument("--system", action="append", default=[], choices=sorted(SYSTEMS))
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        choices=["tau_retail", "magentic_one"],
        help=(
            "restrict to these domains. An endpoint outage typically ruins one "
            "system-domain row and leaves the rest clean; this repairs that row "
            "without paying for the whole grid again"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="cap trajectories per domain")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="parallel judge calls (1 = serial)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=13,
        help=(
            "HTTP retries per call. 13 spans ~4.5 min of backoff, sized to ride "
            "out an SSH-tunnel reconnect; a 2-min window did not, and cost a row "
            "24 of 44 diagnoses"
        ),
    )
    parser.add_argument(
        "--repeats",
        "--seeds",
        dest="repeats",
        type=int,
        default=1,
        help=(
            "re-run each configuration N times and report mean +/- std. Requests are "
            "identical (temperature 0), so this measures serving non-determinism -- "
            "the run-to-run floor a difference has to clear -- not sampling variance"
        ),
    )
    parser.add_argument(
        "--no-signal-caveat",
        dest="signal_caveat",
        action="store_false",
        help="drop the symptoms-are-not-causes note from the violation log (ablation)",
    )
    parser.add_argument("--detection-only", action="store_true", help="H1 only; no LLM calls")
    parser.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = parser.parse_args()

    paths = fetch()
    print(f"AgentRx data @ {AGENTRX_SHA[:12]}  ({paths.root})\n")

    # Provenance. `commit` pins the *data*; without the code version and flags a
    # results file cannot be attributed to the code that produced it, which is how
    # an ablation ends up unreproducible from the tree it supposedly came from.
    results: dict = {
        "commit": AGENTRX_SHA,
        "code": _code_provenance(),
        "argv": sys.argv[1:],
        "config": {
            "model": args.model,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "signal_caveat": args.signal_caveat,
        },
        "detection": None,
        "rows": [],
    }

    print("== Detection (H1) ==")
    detection = run_detection(paths)
    results["detection"] = detection
    unc, cal = detection["uncalibrated"], detection["calibrated_cv"]
    print(
        f"  tau_retail: {detection['n_failed']} failures vs "
        f"{detection['n_succeeded']} successful runs"
    )
    print(
        f"    hand-set weights : recall={unc['recall']:.3f}  FPR={unc['false_positive_rate']:.3f}"
    )
    print(
        f"    calibrated (5-fold CV): recall={cal['recall']:.3f}  "
        f"FPR={cal['false_positive_rate']:.3f}  F1={cal['f1']:.3f}"
    )
    print("  magentic_one: no successful runs published — detection not evaluable\n")

    if args.detection_only:
        _write(args.out, results)
        return 0

    domains = load_domains(paths)
    if args.domain:
        domains = [d for d in domains if d.name in set(args.domain)]
    system_names = args.system or list(DEFAULT_SYSTEMS)
    tiers = args.tier or ["frontier"]

    rows: list[dict] = []
    predictions: list[dict] = []
    # LLM-free systems produce identical results on every tier, so run them once.
    done_without_llm: set[str] = set()

    for tier in tiers:
        llm_systems = [n for n in system_names if SYSTEMS[n].uses_llm]
        client_kwargs = {}
        if tier == "small" and args.base_url:
            client_kwargs["base_url"] = args.base_url
        if tier == "small":
            client_kwargs["max_retries"] = args.max_retries
        client = build_client(tier, model=args.model, **client_kwargs) if llm_systems else None
        label = getattr(client, "model", "none")
        print(f"== Localization + attribution — tier={tier} model={label} ==")

        for name in system_names:
            spec = SYSTEMS[name]
            if not spec.uses_llm and name in done_without_llm:
                continue
            system = build_system(
                spec,
                client,
                tier=tier,
                max_tokens=args.max_tokens,
                signal_caveat=args.signal_caveat,
            )
            display = name if not spec.uses_llm else f"{name}/{tier}"
            # An LLM-free system is deterministic, so repeating it is pure waste.
            repeats = 1 if not spec.uses_llm else max(1, args.repeats)
            for domain in domains:
                runs = []
                for repeat in range(repeats):
                    label = f"  {display} on {domain.name}"
                    if repeats > 1:
                        label += f" (repeat {repeat + 1}/{repeats})"
                    print(f"{label}…", file=sys.stderr)
                    preds = run_system(
                        system, domain, limit=args.limit, concurrency=args.concurrency
                    )
                    runs.append(
                        score(preds, domain.ground_truth, domain=domain.name, system=display)
                    )
                    predictions.append(
                        {
                            "domain": domain.name,
                            "system": display,
                            "repeat": repeat,
                            "records": as_records(preds, domain.ground_truth),
                        }
                    )
                # Judges are non-deterministic; a single run's accuracy is a point
                # estimate, and at n=29 one trajectory moves `exact` by 0.034.
                # Report the spread rather than implying precision. Always via
                # the aggregator, so every row carries the same keys whether or
                # not it was repeated and nothing downstream has to special-case.
                rows.append(aggregate_repeats(runs))
                # Checkpoint after every row. A run here spans hours against a
                # flaky tunnel; writing only at the end means a kill or a crash
                # at 90% discards every completed row and there is nothing to
                # repair from. Rows are independent, so a partial file is still
                # a usable result -- just an incomplete one.
                results["rows"] = rows
                results["predictions"] = predictions
                results["complete"] = False
                _write(args.out, results, quiet=True)
            if not spec.uses_llm:
                done_without_llm.add(name)

        print()

    results["rows"] = rows
    results["predictions"] = predictions
    results["complete"] = True
    # Write first. The expensive part is done; a formatting bug downstream must
    # not be able to throw away a run's worth of model calls.
    _write(args.out, results)
    print(format_table(rows))

    total_errors = sum(r.get("err") or 0 for r in rows)
    if total_errors:
        bad = [f"{r['domain']}/{r['system']}" for r in rows if r.get("err")]
        print(
            f"\n*** WARNING: {total_errors} diagnoses failed outright and are scored as "
            "wrong. These numbers are NOT valid -- fix the endpoint and re-run. ***"
            f"\n*** Affected rows: {', '.join(bad)} ***"
        )
    print(
        "\nScope: Flash (42 trajectories) is unpublished; these are the 73 public "
        "trajectories, reported per domain and NOT comparable to the paper's "
        "115-trajectory aggregate."
    )
    return 0


def _code_provenance() -> dict[str, Any]:
    """Which probe revision produced these numbers, and was the tree dirty."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _write(path: Path | None, results: dict, quiet: bool = False) -> None:
    """Write results, atomically, so a checkpoint can never truncate the last one."""
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(path)
    if not quiet:
        print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
