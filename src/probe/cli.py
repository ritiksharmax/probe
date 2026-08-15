"""The `probe` command line.

probe analyze trace.jsonl                    diagnose failures in a trace file
probe analyze spans.json --adapter agentrx   diagnose from a source format
probe detect traces.jsonl                    triage only, no LLM calls
probe bench agentrx --tier frontier          reproduce the benchmark table
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from probe.detect import Detector
from probe.llm.client import build_client
from probe.rca.judge import RCAJudge
from probe.trace.io import read_jsonl
from probe.trace.model import Trajectory

app = typer.Typer(
    name="probe",
    help="Detect, localize, and explain AI-agent failures from execution traces.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


ADAPTERS = ("jsonl", "otel", "langfuse", "langsmith", "agentrx")


def _load(path: Path, adapter: str) -> list[Trajectory]:
    """Load trajectories through the requested adapter."""
    if adapter == "jsonl":
        return read_jsonl(path)
    if adapter == "otel":
        from probe.trace.adapters.otel import load_otel

        return load_otel(path)
    if adapter == "langfuse":
        from probe.trace.adapters.langfuse import load_langfuse

        return load_langfuse(path)
    if adapter == "langsmith":
        from probe.trace.adapters.langsmith import load_langsmith

        return load_langsmith(path)
    if adapter == "agentrx":
        from probe.trace.adapters.agentrx import load_magentic, load_tau

        return load_magentic(path) if path.is_dir() else load_tau(path)
    raise typer.BadParameter(f"unknown adapter {adapter!r}; expected one of {', '.join(ADAPTERS)}")


@app.command()
def analyze(
    trace: Path = typer.Argument(..., exists=True, help="Trace file or directory"),
    adapter: str = typer.Option(
        "jsonl", "--adapter", "-a", help="jsonl | otel | langfuse | langsmith | agentrx"
    ),
    tier: str = typer.Option("frontier", "--tier", "-t", help="frontier | small"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the model"),
    full: bool = typer.Option(
        False, "--full", help="Show the judge the whole trajectory (disables evidence filtering)"
    ),
    detect_first: bool = typer.Option(
        True,
        "--detect-first/--no-detect-first",
        help="Skip trajectories the detector considers healthy",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a report"),
) -> None:
    """Produce a root-cause report for each failed trajectory in a trace."""
    trajectories = _load(trace, adapter)
    if not trajectories:
        err.print("[yellow]no trajectories found[/yellow]")
        raise typer.Exit(1)

    detector = Detector()
    if detect_first:
        verdicts = {t.trajectory_id: detector(t) for t in trajectories}
        selected = [t for t in trajectories if verdicts[t.trajectory_id].failed]
        skipped = len(trajectories) - len(selected)
        if skipped:
            err.print(f"[dim]detector cleared {skipped} of {len(trajectories)} trajectories[/dim]")
        if not selected:
            console.print("No failures detected.")
            raise typer.Exit(0)
    else:
        selected = trajectories

    try:
        client = build_client(tier, model=model)
    except ImportError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    judge = RCAJudge(client, mode="full" if full else "filtered", tier=tier)

    reports = []
    for trajectory in selected:
        try:
            reports.append(judge(trajectory))
        except Exception as exc:  # noqa: BLE001 - one bad trace must not sink the run
            err.print(f"[red]{trajectory.trajectory_id}: {exc}[/red]")

    if json_out:
        import json

        console.print_json(json.dumps([r.to_dict() for r in reports]))
        return

    for report in reports:
        console.print(report.render())
        console.print()

    if reports:
        total = sum(r.cost_usd for r in reports)
        console.print(f"[dim]{len(reports)} report(s), ${total:.4f} total[/dim]")


@app.command()
def detect(
    trace: Path = typer.Argument(..., exists=True, help="Trace file or directory"),
    adapter: str = typer.Option(
        "jsonl", "--adapter", "-a", help="jsonl | otel | langfuse | langsmith | agentrx"
    ),
    threshold: float = typer.Option(0.5, "--threshold", help="Failure confidence threshold"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the signals behind each call"
    ),
) -> None:
    """Triage a trace for likely failures. Cheap: no LLM calls."""
    trajectories = _load(trace, adapter)
    detector = Detector(threshold=threshold)

    table = Table(title=f"Detection ({len(trajectories)} trajectories)")
    table.add_column("trajectory")
    table.add_column("verdict")
    table.add_column("confidence", justify="right")
    table.add_column("signals")

    failed = 0
    for trajectory in trajectories:
        verdict = detector(trajectory)
        failed += verdict.failed
        table.add_row(
            trajectory.trajectory_id,
            "[red]FAILED[/red]" if verdict.failed else "[green]ok[/green]",
            f"{verdict.confidence:.2f}",
            ", ".join(sorted(verdict.kinds)) or "-",
        )
        if verbose and verdict.events:
            for event in sorted(verdict.events, key=lambda e: -e.severity)[:3]:
                table.add_row("", "", "", f"[dim]step {event.step_index}: {event.evidence}[/dim]")

    console.print(table)
    console.print(f"{failed}/{len(trajectories)} flagged as failures")


@app.command()
def bench(
    suite: str = typer.Argument("agentrx", help="Benchmark suite to run"),
    tier: list[str] = typer.Option([], "--tier", "-t", help="frontier | small (repeatable)"),
    system: list[str] = typer.Option([], "--system", "-s", help="Systems to compare"),
    limit: int | None = typer.Option(None, "--limit", help="Cap trajectories per domain"),
    detection_only: bool = typer.Option(False, "--detection-only", help="H1 only; no LLM calls"),
    out: Path | None = typer.Option(None, "--out", help="Write results JSON here"),
) -> None:
    """Run a benchmark suite and print the comparison table."""
    if suite != "agentrx":
        raise typer.BadParameter(f"unknown suite {suite!r}")

    bench_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "agentrx"
    if not bench_dir.exists():
        err.print(
            "[red]benchmark suite not found — it ships with the repository, "
            "not the installed package. Clone the repo to run it.[/red]"
        )
        raise typer.Exit(2)

    sys.path.insert(0, str(bench_dir))
    import run as bench_run  # type: ignore[import-not-found]

    argv = ["run.py"]
    for value in tier:
        argv += ["--tier", value]
    for value in system:
        argv += ["--system", value]
    if limit:
        argv += ["--limit", str(limit)]
    if detection_only:
        argv.append("--detection-only")
    if out:
        argv += ["--out", str(out)]

    original, sys.argv = sys.argv, argv
    try:
        raise typer.Exit(bench_run.main())
    finally:
        sys.argv = original


@app.command()
def version() -> None:
    """Print the installed version."""
    from probe import __version__

    console.print(f"probe {__version__}")


if __name__ == "__main__":
    app()
