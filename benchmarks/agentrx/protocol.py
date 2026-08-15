"""Ground-truth loading and scoring, mirroring AgentRx's evaluation protocol.

Deliberately matches `agentrx/reports/analyze_metrics.py` so our numbers mean the
same thing theirs do:

* **Localization** is scored at exact match plus ±1..±5 step tolerance, against
  the step number of the *root-cause* failure -- the failure whose `failure_id`
  equals `root_cause.failure_id`, not merely the first annotated failure.
* **Attribution** is scored four ways, because a trajectory carries several
  annotated failures and "correct category" is genuinely ambiguous: against the
  root cause, against *any* annotated failure, against the earliest, and against
  the terminal one (earliest/terminal by `step_number`).
* Both sides of every category comparison go through `probe.rca.taxonomy`, whose
  alias rules are ported from theirs.

One deliberate divergence, documented because it changes a number: upstream
resolves the root cause with `f["failure_id"] == root_cause["failure_id"]`, a
strict comparison. In `magentic_one_ground_truth.json` exactly one trajectory
(`08cae58d-...`) stores its `root_cause.failure_id` as the string `"1"` while its
`failures[].failure_id` are ints, so upstream silently fails to resolve it and
scores it with no root-cause step. We coerce both sides to `str`, which resolves
all 73 trajectories. PROBE and every baseline are scored under identical rules,
so the comparison stays fair; the effect is that our denominator is complete.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from probe.rca.taxonomy import category_to_case

# Tolerance windows reported for step localization, matching upstream.
TOLERANCES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class AnnotatedFailure:
    """One annotated failure within a trajectory."""

    failure_id: str
    step_number: int
    category: str
    step_reason: str = ""
    category_reason: str = ""
    failed_agent: str = ""

    @property
    def case(self) -> int:
        return category_to_case(self.category)


@dataclass(frozen=True)
class GroundTruthEntry:
    """Ground truth for a single trajectory, with the derived scoring targets."""

    trajectory_id: str
    domain: str
    failures: tuple[AnnotatedFailure, ...]
    root_cause_failure_id: str | None
    root_cause_reason: str = ""
    failure_summary: str = ""

    @property
    def _sorted(self) -> list[AnnotatedFailure]:
        return sorted(self.failures, key=lambda f: f.step_number)

    @property
    def root_cause(self) -> AnnotatedFailure | None:
        """The annotated failure designated as the root cause, if resolvable."""
        if self.root_cause_failure_id is None:
            return None
        for f in self.failures:
            if f.failure_id == self.root_cause_failure_id:
                return f
        return None

    @property
    def critical_step(self) -> int | None:
        """The step localization is scored against."""
        rc = self.root_cause
        return rc.step_number if rc else None

    @property
    def root_cause_case(self) -> int | None:
        rc = self.root_cause
        return rc.case if rc else None

    @property
    def earliest_case(self) -> int | None:
        s = self._sorted
        return s[0].case if s else None

    @property
    def terminal_case(self) -> int | None:
        s = self._sorted
        return s[-1].case if s else None

    @property
    def all_cases(self) -> set[int]:
        return {f.case for f in self.failures}


@dataclass(frozen=True)
class Prediction:
    """A system's diagnosis of one trajectory."""

    trajectory_id: str
    step: int | None
    category: str | int | None
    # Efficiency accounting for the headline cost claim.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    # Set when detection ran; None when the system was handed a known failure.
    predicted_failed: bool | None = None

    @property
    def case(self) -> int | None:
        if self.category is None:
            return None
        return (
            category_to_case(self.category)
            if isinstance(self.category, str)
            else (self.category if 1 <= int(self.category) <= 10 else 10)
        )


def _as_id(value: Any) -> str:
    """Normalize a trajectory or failure id to a string.

    tau uses integer `task_id`, Magentic-One uses uuid strings, and one record
    mixes int and str failure ids. Everything becomes a string exactly once, here.
    """
    return str(value).strip()


def load_ground_truth(path: str | Path, domain: str) -> dict[str, GroundTruthEntry]:
    """Load one domain's ground-truth file, keyed by trajectory id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = list(raw.values())

    entries: dict[str, GroundTruthEntry] = {}
    for record in raw:
        failures = tuple(
            AnnotatedFailure(
                failure_id=_as_id(f.get("failure_id")),
                step_number=int(f.get("step_number", 0)),
                category=f.get("failure_category", ""),
                step_reason=f.get("step_reason", ""),
                category_reason=f.get("category_reason", ""),
                failed_agent=f.get("failed_agent", ""),
            )
            for f in record.get("failures", [])
        )
        rc = record.get("root_cause") or {}
        rc_id = rc.get("failure_id")
        tid = _as_id(record.get("trajectory_id"))
        entries[tid] = GroundTruthEntry(
            trajectory_id=tid,
            domain=domain,
            failures=failures,
            root_cause_failure_id=_as_id(rc_id) if rc_id is not None else None,
            root_cause_reason=rc.get("reason_for_root_cause", ""),
            failure_summary=record.get("failure_summary", ""),
        )
    return entries


@dataclass
class DomainScore:
    """Scores for one system on one domain."""

    domain: str
    system: str
    n_scored: int = 0
    n_missing_root_cause: int = 0
    localization: dict[int, float] = field(default_factory=dict)
    attribution: dict[str, float] = field(default_factory=dict)
    mean_step_distance: float | None = None
    detection: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mean_latency_s: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "system": self.system,
            "n": self.n_scored,
            "exact": self.localization.get(0),
            "±1": self.localization.get(1),
            "±3": self.localization.get(3),
            "±5": self.localization.get(5),
            "root_cause_cat": self.attribution.get("root_cause"),
            "any_cat": self.attribution.get("any"),
            "cost_usd": round(self.cost_usd, 4),
            "latency_s": round(self.mean_latency_s, 2),
        }


def score(
    predictions: Iterable[Prediction],
    ground_truth: dict[str, GroundTruthEntry],
    *,
    domain: str,
    system: str,
) -> DomainScore:
    """Score one system's predictions for one domain.

    Only trajectories present in `ground_truth` are scored. A trajectory whose
    root cause cannot be resolved is excluded from localization and root-cause
    attribution (and counted in `n_missing_root_cause`) rather than silently
    counted as wrong -- that would flatter or punish systems arbitrarily.
    """
    preds = [p for p in predictions if p.trajectory_id in ground_truth]

    hits: dict[int, int] = dict.fromkeys(TOLERANCES, 0)
    distances: list[int] = []
    attr_hits = {"root_cause": 0, "any": 0, "earliest": 0, "terminal": 0}
    n_localizable = 0
    n_missing_rc = 0

    for pred in preds:
        gt = ground_truth[pred.trajectory_id]
        gt_step = gt.critical_step
        if gt_step is None:
            n_missing_rc += 1
            continue
        n_localizable += 1

        if pred.step is not None:
            distance = abs(int(pred.step) - gt_step)
            distances.append(distance)
            for tol in TOLERANCES:
                if distance <= tol:
                    hits[tol] += 1

        case = pred.case
        if case is not None:
            if case == gt.root_cause_case:
                attr_hits["root_cause"] += 1
            if case in gt.all_cases:
                attr_hits["any"] += 1
            if case == gt.earliest_case:
                attr_hits["earliest"] += 1
            if case == gt.terminal_case:
                attr_hits["terminal"] += 1

    denom = n_localizable or 1
    result = DomainScore(
        domain=domain,
        system=system,
        n_scored=n_localizable,
        n_missing_root_cause=n_missing_rc,
        localization={tol: hits[tol] / denom for tol in TOLERANCES},
        attribution={k: v / denom for k, v in attr_hits.items()},
        mean_step_distance=(sum(distances) / len(distances)) if distances else None,
        cost_usd=sum(p.cost_usd for p in preds),
        prompt_tokens=sum(p.prompt_tokens for p in preds),
        completion_tokens=sum(p.completion_tokens for p in preds),
        mean_latency_s=(sum(p.latency_s for p in preds) / len(preds)) if preds else 0.0,
    )
    return result


def score_detection(
    predictions: Iterable[Prediction],
    *,
    failed_ids: set[str],
    succeeded_ids: set[str],
) -> dict[str, float]:
    """Score H1 detection: recall on known failures, false-positive rate on successes.

    For tau-retail the negative set is the 73 successful runs in
    `tau_dataset_full.json`, which are disjoint from all 29 annotated failures.
    Magentic-One publishes no successful runs, so detection is tau-only.
    """
    tp = fn = fp = tn = 0
    for pred in predictions:
        if pred.predicted_failed is None:
            continue
        if pred.trajectory_id in failed_ids:
            tp += pred.predicted_failed
            fn += not pred.predicted_failed
        elif pred.trajectory_id in succeeded_ids:
            fp += pred.predicted_failed
            tn += not pred.predicted_failed

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
        "n_positive": tp + fn,
        "n_negative": fp + tn,
    }


def aggregate_seeds(scores: Sequence[DomainScore]) -> dict[str, Any]:
    """Mean and population std across repeated seeds of the same configuration.

    Judges are non-deterministic; a single run's accuracy is not a stable number,
    so the harness runs several seeds and reports spread, as upstream does.
    """
    if not scores:
        return {}

    def spread(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }

    return {
        "domain": scores[0].domain,
        "system": scores[0].system,
        "localization": {
            tol: spread([s.localization.get(tol, 0.0) for s in scores]) for tol in TOLERANCES
        },
        "attribution": {
            key: spread([s.attribution.get(key, 0.0) for s in scores])
            for key in ("root_cause", "any", "earliest", "terminal")
        },
        "cost_usd": spread([s.cost_usd for s in scores]),
        "latency_s": spread([s.mean_latency_s for s in scores]),
    }
