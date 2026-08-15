"""Run the AgentRx benchmark and emit the comparison table.

    uv run python benchmarks/agentrx/run.py --tier frontier
    uv run python benchmarks/agentrx/run.py --tier frontier --tier small --seeds 3
    uv run python benchmarks/agentrx/run.py --detection-only   # no LLM needed

Scope caveat, restated wherever the results are: the Flash domain (42 of the
paper's 115 trajectories) is not published, so this runs on the 73 that are —
tau-retail (29) and Magentic-One (44) — and reports them per domain. These
numbers are not comparable to the paper's 115-trajectory aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import DEFAULT_SYSTEMS, SYSTEMS, build_system  # noqa: E402
from fetch import AGENTRX_SHA, fetch  # noqa: E402
from protocol import (  # noqa: E402
    GroundTruthEntry,
    Prediction,
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


def run_system(system, domain: Domain, limit: int | None = None) -> list[Prediction]:
    """Diagnose every annotated trajectory in a domain."""
    ids = [t for t in domain.ground_truth if t in domain.trajectories]
    if limit:
        ids = ids[:limit]

    # Carriage-return progress only makes sense on a terminal; piped to a file it
    # produces one unreadable mega-line.
    interactive = sys.stderr.isatty()

    predictions = []
    for index, tid in enumerate(ids, start=1):
        report = system(domain.trajectories[tid])
        predictions.append(
            Prediction(
                trajectory_id=tid,
                step=report.critical_step,
                category=report.category_case,
                prompt_tokens=report.prompt_tokens,
                completion_tokens=report.completion_tokens,
                cost_usd=report.cost_usd,
                latency_s=report.latency_s,
            )
        )
        if interactive:
            print(f"    [{index}/{len(ids)}] {tid[:40]}".ljust(60), end="\r", file=sys.stderr)
    if interactive:
        print(" " * 60, end="\r", file=sys.stderr)
    return predictions


def format_table(rows: list[dict]) -> str:
    """Fixed-width comparison table, so results are readable without rich."""
    headers = ["domain", "system", "n", "exact", "±1", "±3", "±5", "cat(rc)", "cat(any)", "$", "s"]
    keys = [
        "domain",
        "system",
        "n",
        "exact",
        "±1",
        "±3",
        "±5",
        "root_cause_cat",
        "any_cat",
        "cost_usd",
        "latency_s",
    ]

    def cell(row, key):
        value = row.get(key)
        if value is None:
            return "-"
        if key in {"exact", "±1", "±3", "±5", "root_cause_cat", "any_cat"}:
            return f"{value:.3f}"
        if key == "cost_usd":
            return f"{value:.4f}"
        if key == "latency_s":
            return f"{value:.1f}"
        return str(value)

    table = [headers] + [[cell(r, k) for k in keys] for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    out.append("  ".join("-" * w for w in widths))
    for row in table[1:]:
        out.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AgentRx benchmark.")
    parser.add_argument("--tier", action="append", default=[], choices=["frontier", "small"])
    parser.add_argument("--model", default=None, help="override the model for the tier")
    parser.add_argument("--system", action="append", default=[], choices=sorted(SYSTEMS))
    parser.add_argument("--limit", type=int, default=None, help="cap trajectories per domain")
    parser.add_argument("--detection-only", action="store_true", help="H1 only; no LLM calls")
    parser.add_argument("--out", type=Path, default=None, help="write results JSON here")
    args = parser.parse_args()

    paths = fetch()
    print(f"AgentRx data @ {AGENTRX_SHA[:12]}  ({paths.root})\n")

    results: dict = {"commit": AGENTRX_SHA, "detection": None, "rows": []}

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
    system_names = args.system or list(DEFAULT_SYSTEMS)
    tiers = args.tier or ["frontier"]

    rows: list[dict] = []
    # LLM-free systems produce identical results on every tier, so run them once.
    done_without_llm: set[str] = set()

    for tier in tiers:
        llm_systems = [n for n in system_names if SYSTEMS[n].uses_llm]
        client = build_client(tier, model=args.model) if llm_systems else None
        label = getattr(client, "model", "none")
        print(f"== Localization + attribution — tier={tier} model={label} ==")

        for name in system_names:
            spec = SYSTEMS[name]
            if not spec.uses_llm and name in done_without_llm:
                continue
            system = build_system(spec, client, tier=tier)
            display = name if not spec.uses_llm else f"{name}/{tier}"
            for domain in domains:
                print(f"  {display} on {domain.name}…", file=sys.stderr)
                preds = run_system(system, domain, limit=args.limit)
                result = score(preds, domain.ground_truth, domain=domain.name, system=display)
                rows.append(result.as_row())
            if not spec.uses_llm:
                done_without_llm.add(name)

        print()

    results["rows"] = rows
    print(format_table(rows))
    print(
        "\nScope: Flash (42 trajectories) is unpublished; these are the 73 public "
        "trajectories, reported per domain and NOT comparable to the paper's "
        "115-trajectory aggregate."
    )
    _write(args.out, results)
    return 0


def _write(path: Path | None, results: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
