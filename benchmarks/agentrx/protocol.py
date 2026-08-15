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
    # Evidence windows the judge was shown, as inclusive 1-based (start, end)
    # pairs. Empty for unfiltered systems. Scoring uses these to report the
    # *ceiling* on filtered accuracy: a judge cannot name a step it never saw.
    windows: tuple[tuple[int, int], ...] = ()
    # Set when the diagnosis could not be produced at all (transport failure,
    # crash). A failed call scores identically to a wrong answer, so without
    # this an outage is indistinguishable from a bad model -- which is exactly
    # how one run here reported zeros from a dead SSH tunnel as if they were
    # results.
    error: str | None = None

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
    n_errors: int = 0
    localization: dict[int, float] = field(default_factory=dict)
    attribution: dict[str, float] = field(default_factory=dict)
    mean_step_distance: float | None = None
    # Fraction of scored trajectories whose true critical step fell inside an
    # evidence window. This is the hard ceiling on `exact` for a filtered system
    # and is measurable without any LLM, so it belongs next to the accuracy.
    window_recall: float | None = None
    detection: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mean_latency_s: float = 0.0

    def as_row(self) -> dict[str, Any]:
        # `tok` is in+out per diagnosis. It is redundant with the two token
        # columns but not with `cost_usd`, which reads 0.0000 for a self-hosted
        # model -- so without it the efficiency comparison is invisible in the
        # table on exactly the runs where efficiency is the point.
        per = (
            round((self.prompt_tokens + self.completion_tokens) / self.n_scored)
            if self.n_scored
            else None
        )
        return {
            "domain": self.domain,
            "system": self.system,
            "n": self.n_scored,
            "err": self.n_errors,
            "tok": per,
            "exact": self.localization.get(0),
            "±1": self.localization.get(1),
            "±3": self.localization.get(3),
            "±5": self.localization.get(5),
            "root_cause_cat": self.attribution.get("root_cause"),
            "any_cat": self.attribution.get("any"),
            "win_rec": self.window_recall,
            "in_tok": (round(self.prompt_tokens / self.n_scored) if self.n_scored else None),
            "out_tok": (round(self.completion_tokens / self.n_scored) if self.n_scored else None),
            "cost_usd": round(self.cost_usd, 4),
            "latency_s": round(self.mean_latency_s, 2),
        }


def as_records(
    predictions: Iterable[Prediction],
    ground_truth: dict[str, GroundTruthEntry],
) -> list[dict[str, Any]]:
    """Per-trajectory predictions paired with their targets, for post-hoc analysis.

    Aggregates alone cannot answer *why* a system scored as it did -- whether a
    violation log pushes attribution toward the categories its signals name, say,
    or which trajectories a filter's windows missed. Without these, every such
    question costs another full run of the benchmark.
    """
    records = []
    for pred in predictions:
        gt = ground_truth.get(pred.trajectory_id)
        if gt is None:
            continue
        records.append(
            {
                "trajectory_id": pred.trajectory_id,
                "pred_step": pred.step,
                "pred_case": pred.case,
                "gt_step": gt.critical_step,
                "gt_case": gt.root_cause_case,
                "gt_all_cases": sorted(gt.all_cases),
                "windows": [list(w) for w in pred.windows],
                "error": pred.error,
            }
        )
    return records


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
    n_errors = sum(1 for p in preds if p.error)

    hits: dict[int, int] = dict.fromkeys(TOLERANCES, 0)
    distances: list[int] = []
    in_window = 0
    n_windowed = 0
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

        if pred.windows:
            n_windowed += 1
            in_window += any(start <= gt_step <= end for start, end in pred.windows)

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
        n_errors=n_errors,
        localization={tol: hits[tol] / denom for tol in TOLERANCES},
        attribution={k: v / denom for k, v in attr_hits.items()},
        mean_step_distance=(sum(distances) / len(distances)) if distances else None,
        window_recall=(in_window / n_windowed) if n_windowed else None,
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


# Averaged across repeats. `err` is deliberately absent: it is summed, because
# the question it answers is "did anything fail anywhere in this row".
_MEAN_KEYS: tuple[str, ...] = (
    "exact",
    "±1",
    "±3",
    "±5",
    "root_cause_cat",
    "any_cat",
    "win_rec",
    "in_tok",
    "out_tok",
    "tok",
    "cost_usd",
    "latency_s",
)


def aggregate_repeats(scores: Sequence[DomainScore]) -> dict[str, Any]:
    """Collapse repeated runs of one configuration into a single table row.

    Every field is aggregated over all repeats. Taking the metadata off the first
    run instead -- which this used to do, overwriting only the accuracies -- hides
    a failure in any *later* repeat: the mean quietly absorbs its wrong answers
    while `err` still reports the first run's zero, so a contaminated row looks
    clean. That happened here: a run lost 24 of 44 diagnoses in repeat 1 of one
    row and was only caught because it was repeat 1.

    Per-repeat accuracies are kept so a partially contaminated row can be salvaged
    from the surviving repeats instead of forcing a full re-run.
    """
    if not scores:
        return {}

    rows = [s.as_row() for s in scores]
    row = dict(rows[0])

    for key in _MEAN_KEYS:
        values = [r[key] for r in rows if r.get(key) is not None]
        row[key] = statistics.fmean(values) if values else None

    exact = [r["exact"] for r in rows if r.get("exact") is not None]
    row["err"] = sum(r.get("err") or 0 for r in rows)
    row["repeats"] = len(rows)
    row["exact_std"] = statistics.pstdev(exact) if len(exact) > 1 else 0.0
    row["exact_per_repeat"] = exact
    row["err_per_repeat"] = [r.get("err") or 0 for r in rows]
    return row
