"""Fetch the public AgentRx benchmark data.

Data comes from the **public `microsoft/AgentRx` GitHub repo**, not from the
Hugging Face dataset. The HF dataset (`microsoft/AgentRx`) is access-gated and
returns 401 unauthenticated, and it carries no trajectories the GitHub repo
lacks, so depending on it would only add an auth wall.

Scope, stated plainly because it bounds every number we report: the paper
evaluates on 115 trajectories across three domains, but the **Flash** domain
(42 trajectories, incident management) is not published in either location.
What is publicly obtainable is tau-retail (29) and Magentic-One (44) = **73**.
Results are therefore reported per domain and are not directly comparable to the
paper's 115-trajectory aggregate.

Everything is pinned to a commit SHA so a run is reproducible even if upstream
moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

# Pinned upstream revision. Bump deliberately, and re-run the benchmark when you do.
AGENTRX_REPO = "microsoft/AgentRx"
AGENTRX_SHA = "f228165bfec60a801fd5fedd9d8ffe0f9de0c69d"

RAW_BASE = f"https://raw.githubusercontent.com/{AGENTRX_REPO}/{AGENTRX_SHA}"
API_BASE = f"https://api.github.com/repos/{AGENTRX_REPO}"

DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache" / AGENTRX_SHA[:12]

# Single files we always need.
GROUND_TRUTH_FILES = {
    # 29 annotated tau-retail failures. Note: `data/ground_truth/tau_retail.json`
    # also exists but holds only 24 entries; this is the one that matches the
    # paper's stated tau-bench count.
    "tau": "data/ground_truth/tau_ground_truth.json",
    "magentic": "data/ground_truth/magentic_one_ground_truth.json",
}

TRAJECTORY_FILES = {
    # 29 failed tau runs, joined to ground truth by task_id.
    "tau_failed": "data/tau_retail/tau_dataset_failed.json",
    # 100 tau runs: 73 with reward 1.0 and 27 with reward 0.0. The 73 successes
    # are disjoint from every annotated failure, which makes them the clean
    # negative set for scoring detection false positives.
    "tau_full": "data/tau_retail/tau_dataset_full.json",
}

# Per-trajectory Magentic-One files, discovered from the repo tree.
MAGENTIC_DIR = "data/magentic_dataset"


@dataclass(frozen=True)
class BenchmarkPaths:
    """Where the fetched data landed on disk."""

    root: Path
    tau_ground_truth: Path
    magentic_ground_truth: Path
    tau_failed: Path
    tau_full: Path
    magentic_dir: Path

    def exists(self) -> bool:
        return all(
            p.exists()
            for p in (
                self.tau_ground_truth,
                self.magentic_ground_truth,
                self.tau_failed,
                self.tau_full,
                self.magentic_dir,
            )
        )


def _get(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _download(client: httpx.Client, repo_path: str, dest: Path, *, force: bool) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(client, f"{RAW_BASE}/{repo_path}"))
    return dest


def _list_magentic_files(client: httpx.Client) -> list[str]:
    """List `data/magentic_dataset/*.json` at the pinned SHA."""
    tree = json.loads(
        _get(client, f"{API_BASE}/git/trees/{AGENTRX_SHA}?recursive=1").decode("utf-8")
    )
    return sorted(
        node["path"]
        for node in tree.get("tree", [])
        if node["type"] == "blob"
        and node["path"].startswith(f"{MAGENTIC_DIR}/")
        and node["path"].endswith(".json")
    )


def fetch(cache_dir: str | Path | None = None, *, force: bool = False) -> BenchmarkPaths:
    """Download the public benchmark data into a local cache and return its paths.

    Idempotent: already-present files are left alone unless `force` is set.
    """
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    paths = BenchmarkPaths(
        root=root,
        tau_ground_truth=root / GROUND_TRUTH_FILES["tau"],
        magentic_ground_truth=root / GROUND_TRUTH_FILES["magentic"],
        tau_failed=root / TRAJECTORY_FILES["tau_failed"],
        tau_full=root / TRAJECTORY_FILES["tau_full"],
        magentic_dir=root / MAGENTIC_DIR,
    )
    if paths.exists() and not force:
        return paths

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "probe-agents"}
    with httpx.Client(headers=headers) as client:
        for repo_path in (*GROUND_TRUTH_FILES.values(), *TRAJECTORY_FILES.values()):
            _download(client, repo_path, root / repo_path, force=force)

        magentic_files = _list_magentic_files(client)
        if not magentic_files:
            raise RuntimeError(
                f"no trajectory files found under {MAGENTIC_DIR} at {AGENTRX_SHA[:12]}"
            )
        for repo_path in magentic_files:
            _download(client, repo_path, root / repo_path, force=force)

    return paths


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    p = fetch()
    n_mag = len(list(p.magentic_dir.glob("*.json")))
    print(f"AgentRx data cached at {p.root}")
    print(f"  tau ground truth      {p.tau_ground_truth}")
    print(f"  magentic ground truth {p.magentic_ground_truth}")
    print(f"  magentic trajectories {n_mag} files")
