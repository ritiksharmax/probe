"""Read and write the canonical trajectory schema as JSONL.

One JSON object per line, each a serialized `Trajectory`. This is PROBE's native
interchange format: the thing `probe analyze` accepts, the thing adapters emit,
and the format examples ship in.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .model import Trajectory


def read_jsonl(path: str | Path) -> list[Trajectory]:
    """Load every trajectory in a JSONL file."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[Trajectory]:
    """Stream trajectories, so large exports do not have to fit in memory."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Trajectory.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - want the file/line context
                raise ValueError(f"{path}:{lineno}: could not parse trajectory: {exc}") from exc


def write_jsonl(path: str | Path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories to JSONL, creating parent directories. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(traj.model_dump_json(exclude_none=True))
            fh.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Trajectory:
    """Load a single trajectory from a plain `.json` file."""
    path = Path(path)
    return Trajectory.model_validate(json.loads(path.read_text(encoding="utf-8")))
