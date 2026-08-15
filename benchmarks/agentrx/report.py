"""Read a results JSON and print the comparison with error bars.

    uv run python benchmarks/agentrx/report.py benchmarks/agentrx/results/clean-repeats3.json

Separate from `run.py` on purpose: re-reading a finished run costs nothing, so
the analysis can be redone and argued with without spending another hour of GPU
time. It also means the reported numbers come from a script rather than from
arithmetic done by hand in a shell, which is how a single-domain slice once got
reported as if it were the whole benchmark.

Two rules this enforces, because both were broken by hand at least once here:

1. **A row with errors is not a result.** Failed diagnoses are scored as wrong,
   so a contaminated row is silently depressed. Such rows are excluded from the
   pooled numbers and printed as `INVALID`, never quietly averaged in.
2. **A difference smaller than the spread is not a finding.** Every accuracy is
   printed with its standard deviation, and the spread is restated in
   trajectories, because "±0.06" reads as precision until you notice it means
   "±1.7 of the 29 trajectories".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

METRICS = (("exact", "exact"), ("±5", "±5"), ("cat(rc)", "root_cause_cat"))


def _pool(rows: list[dict], key: str) -> tuple[float, int] | None:
    """Trajectory-weighted mean across domains, with the total denominator."""
    usable = [r for r in rows if r.get(key) is not None and r.get("n")]
    if not usable:
        return None
    total = sum(r["n"] for r in usable)
    return sum(r[key] * r["n"] for r in usable) / total, total


def _pooled_std(rows: list[dict]) -> float:
    """Propagate per-domain spread to the pooled mean, treating domains as independent."""
    total = sum(r["n"] for r in rows if r.get("n")) or 1
    return math.sqrt(sum((r["n"] * r.get("exact_std", 0.0)) ** 2 for r in rows)) / total


def report(results: dict[str, Any]) -> str:
    rows = results.get("rows", [])
    out: list[str] = []

    code = results.get("code") or {}
    if code:
        dirty = "  (dirty tree)" if code.get("dirty") else ""
        out.append(f"probe @ {str(code.get('commit'))[:12]}{dirty}")
    cfg = results.get("config") or {}
    if cfg:
        out.append(
            f"model={cfg.get('model')}  repeats={cfg.get('repeats')}  "
            f"signal_caveat={cfg.get('signal_caveat')}"
        )
    if results.get("complete") is False:
        out.append("PARTIAL — this run did not finish; rows below are the ones that completed.")
    out.append("")

    bad = [r for r in rows if r.get("err")]
    if bad:
        out.append("EXCLUDED — failed diagnoses are scored as wrong, so these are not results:")
        for r in bad:
            per = r.get("err_per_repeat")
            detail = f"  per repeat {per}" if per else ""
            out.append(f"  INVALID  {r['domain']}/{r['system']}  {r['err']} failed{detail}")
        out.append("")

    clean = [r for r in rows if not r.get("err")]

    out.append("Per domain (mean ± sd over repeats; sd also in trajectories)")
    out.append("")
    for domain in dict.fromkeys(r["domain"] for r in clean):
        out.append(f"  {domain}")
        here = sorted(
            (r for r in clean if r["domain"] == domain),
            key=lambda r: -(r.get("exact") or 0),
        )
        for r in here:
            sd = r.get("exact_std") or 0.0
            n = r.get("n") or 0
            cells = "  ".join(
                f"{label}={r[key]:.3f}" for label, key in METRICS if r.get(key) is not None
            )
            tok = f"  tok={r['tok']:,.0f}" if r.get("tok") else ""
            out.append(f"    {r['system']:22s} {cells}  ±{sd:.3f} (±{sd * n:.1f} traj){tok}")
        out.append("")

    out.append("Pooled across domains (only systems valid in every domain)")
    out.append("")
    domains = {r["domain"] for r in rows}
    for system in dict.fromkeys(r["system"] for r in clean):
        mine = [r for r in clean if r["system"] == system]
        if {r["domain"] for r in mine} != domains:
            missing = ", ".join(sorted(domains - {r["domain"] for r in mine}))
            out.append(f"  {system:22s} not pooled — no valid row for {missing}")
            continue
        cells = []
        for label, key in METRICS:
            pooled = _pool(mine, key)
            if pooled:
                cells.append(f"{label}={pooled[0]:.3f}")
        n = sum(r["n"] for r in mine)
        out.append(f"  {system:22s} {'  '.join(cells)}  ±{_pooled_std(mine):.3f}  (n={n})")

    out.append("")
    out.append("A gap narrower than the spread is not a finding. At n=29 one trajectory is 0.034.")
    return "\n".join(out)


def _valid(rows: list[dict], domain: str, key: str) -> list[dict]:
    return [
        r for r in rows if r["domain"] == domain and not r.get("err") and r.get(key) is not None
    ]


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Rank systems in two independent runs of the same configuration.

    The sharpest available test of whether an ordering is real. Repeats inside a
    single run measure serving non-determinism; two separate runs also capture
    whatever drifts between them. An ordering that reshuffles here is not a
    finding no matter how clean it looked in either run alone -- which is exactly
    what happened to "the violation log helps attribution", a claim that held in
    one run and inverted in the next.
    """
    rows_a, rows_b = a.get("rows", []), b.get("rows", [])
    out = ["Rankings in two independent runs of the same configuration.", ""]

    for label, key in (("exact", "exact"), ("cat(rc)", "root_cause_cat")):
        out.append(f"  by {label}")
        for domain in dict.fromkeys(r["domain"] for r in rows_a + rows_b):
            for name, rows in (("A", rows_a), ("B", rows_b)):
                ranked = sorted(_valid(rows, domain, key), key=lambda r: -r[key])
                if not ranked:
                    continue
                order = " > ".join(r["system"].split("/")[0] for r in ranked)
                out.append(f"    {domain:13s} run {name}: {order}")
            out.append("")

    out.append("Systems valid in both runs, same domain:")
    index_a = {(r["domain"], r["system"]): r for r in rows_a if not r.get("err")}
    index_b = {(r["domain"], r["system"]): r for r in rows_b if not r.get("err")}
    for combo in sorted(set(index_a) & set(index_b)):
        ra, rb = index_a[combo], index_b[combo]
        deltas = []
        for label, key in (("exact", "exact"), ("cat(rc)", "root_cause_cat")):
            if ra.get(key) is not None and rb.get(key) is not None:
                deltas.append(f"{label} {ra[key]:.3f}→{rb[key]:.3f} ({rb[key] - ra[key]:+.3f})")
        out.append(f"  {combo[0]:13s} {combo[1]:22s} {'  '.join(deltas)}")

    out.append("")
    out.append("An ordering that differs between A and B is noise, not a result.")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="results JSON written by run.py")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="a second results JSON; rank both runs to test whether an ordering is real",
    )
    args = parser.parse_args()
    first = json.loads(args.results.read_text(encoding="utf-8"))
    print(report(first))
    if args.compare:
        print()
        print(compare(json.loads(args.compare.read_text(encoding="utf-8")), first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
