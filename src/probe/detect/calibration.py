"""Fitting detection weights to a labeled corpus.

The default detector ships hand-set weights, which is the only honest option
when nothing is known about a deployment. They are also demonstrably
mediocre: on the public tau-retail split they reach roughly 0.59 recall at 0.30
false-positive rate, because each signal kind's real diagnostic value is an
empirical property of the traffic, not something to guess.

Anyone with labeled runs -- and in production, outcome labels are usually the
*easiest* thing to obtain, from user thumbs, task success, or retries -- can do
better by fitting. This module implements a naive-Bayes fit over signal kinds:
each kind gets a log-likelihood ratio comparing how often it fires on failures
versus successes, and the verdict is the sum of the ratios for the kinds that
fired.

Naive Bayes is chosen over something stronger deliberately. Failure corpora are
small (29 positives here), the independence assumption fails gracefully, and
every weight stays individually interpretable -- you can read off exactly why a
run was flagged, which is the entire point of a diagnostic tool.

**Report cross-validated numbers.** `cross_val_report` exists because a fit
evaluated on its own training data is meaningless, and with corpora this small
the temptation to quote it is real.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probe.detect.detector import Detector, FailureVerdict
from probe.signals.base import Signal
from probe.trace.model import Trajectory


@dataclass
class CalibratedDetector:
    """A detector whose kind weights are log-likelihood ratios fit to a corpus."""

    log_ratios: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    threshold: float = 0.5
    signals: list[Signal] | None = None

    def __post_init__(self) -> None:
        self._base = Detector(signals=self.signals, threshold=self.threshold)

    def score(self, trajectory: Trajectory) -> float:
        """Probability that this run failed."""
        verdict = self._base(trajectory)
        return self._score_kinds(set(verdict.kinds))

    def _score_kinds(self, kinds: set[str]) -> float:
        total = self.bias + sum(self.log_ratios.get(kind, 0.0) for kind in kinds)
        return 1.0 / (1.0 + math.exp(-total))

    def __call__(self, trajectory: Trajectory) -> FailureVerdict:
        verdict = self._base(trajectory)
        confidence = self._score_kinds(set(verdict.kinds))
        return FailureVerdict(
            trajectory_id=trajectory.trajectory_id,
            failed=confidence >= self.threshold,
            confidence=confidence,
            events=verdict.events,
            threshold=self.threshold,
        )

    def top_weights(self, limit: int = 10) -> list[tuple[str, float]]:
        """Most influential signal kinds, strongest evidence of failure first."""
        return sorted(self.log_ratios.items(), key=lambda kv: -abs(kv[1]))[:limit]


def _kinds_for(trajectory: Trajectory, detector: Detector) -> set[str]:
    return set(detector(trajectory).kinds)


def fit(
    trajectories: Sequence[Trajectory],
    labels: Sequence[bool],
    *,
    signals: list[Signal] | None = None,
    threshold: float = 0.5,
    smoothing: float = 1.0,
) -> CalibratedDetector:
    """Fit kind weights from labeled runs. `labels[i]` is True when run i failed.

    Laplace smoothing keeps a kind that never fires on one class from producing
    an infinite weight -- with 29 positives, that happens easily.
    """
    if len(trajectories) != len(labels):
        raise ValueError("trajectories and labels must be the same length")
    if not trajectories:
        raise ValueError("cannot fit on an empty corpus")

    base = Detector(signals=signals, threshold=threshold)
    observed = [(_kinds_for(t, base), bool(y)) for t, y in zip(trajectories, labels, strict=True)]
    return _fit_from_observations(
        observed, threshold=threshold, smoothing=smoothing, signals=signals
    )


def _fit_from_observations(
    observed: Sequence[tuple[set[str], bool]],
    *,
    threshold: float,
    smoothing: float,
    signals: list[Signal] | None,
) -> CalibratedDetector:
    n_pos = sum(1 for _, y in observed if y)
    n_neg = len(observed) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one failed and one successful run to fit")

    all_kinds = {k for kinds, _ in observed for k in kinds}
    log_ratios: dict[str, float] = {}
    for kind in all_kinds:
        fires_pos = sum(1 for kinds, y in observed if y and kind in kinds)
        fires_neg = sum(1 for kinds, y in observed if not y and kind in kinds)
        p_pos = (fires_pos + smoothing) / (n_pos + 2 * smoothing)
        p_neg = (fires_neg + smoothing) / (n_neg + 2 * smoothing)
        log_ratios[kind] = math.log(p_pos / p_neg)

    # Prior log-odds, so a run with no signals at all scores the base rate.
    bias = math.log(n_pos / n_neg)
    return CalibratedDetector(
        log_ratios=log_ratios, bias=bias, threshold=threshold, signals=signals
    )


@dataclass
class DetectionReport:
    """Cross-validated detection performance."""

    recall: float
    false_positive_rate: float
    precision: float
    f1: float
    n_positive: int
    n_negative: int
    folds: int

    def __str__(self) -> str:
        return (
            f"recall={self.recall:.3f} FPR={self.false_positive_rate:.3f} "
            f"precision={self.precision:.3f} F1={self.f1:.3f} "
            f"(n={self.n_positive}+{self.n_negative}, {self.folds}-fold CV)"
        )


def cross_val_report(
    trajectories: Sequence[Trajectory],
    labels: Sequence[bool],
    *,
    folds: int = 5,
    signals: list[Signal] | None = None,
    threshold: float = 0.5,
    smoothing: float = 1.0,
) -> DetectionReport:
    """Stratified k-fold cross-validated detection metrics.

    Signals are computed once per trajectory and reused across folds -- they do
    not depend on the fit, and recomputing them per fold would be pure waste.
    """
    if len(trajectories) != len(labels):
        raise ValueError("trajectories and labels must be the same length")

    base = Detector(signals=signals, threshold=threshold)
    observed = [(_kinds_for(t, base), bool(y)) for t, y in zip(trajectories, labels, strict=True)]

    positives = [i for i, (_, y) in enumerate(observed) if y]
    negatives = [i for i, (_, y) in enumerate(observed) if not y]
    folds = max(2, min(folds, len(positives), len(negatives)))

    # Stratify by dealing each class round-robin, so every fold holds both classes.
    assignment: dict[int, int] = {}
    for group in (positives, negatives):
        for position, index in enumerate(group):
            assignment[index] = position % folds

    tp = fp = tn = fn = 0
    for fold in range(folds):
        train = [obs for i, obs in enumerate(observed) if assignment[i] != fold]
        test = [(i, obs) for i, obs in enumerate(observed) if assignment[i] == fold]
        if not test or not any(y for _, y in train) or not any(not y for _, y in train):
            continue

        model = _fit_from_observations(
            train, threshold=threshold, smoothing=smoothing, signals=signals
        )
        for _, (kinds, truth) in test:
            predicted = model._score_kinds(kinds) >= threshold
            if truth and predicted:
                tp += 1
            elif truth and not predicted:
                fn += 1
            elif not truth and predicted:
                fp += 1
            else:
                tn += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return DetectionReport(
        recall=recall,
        false_positive_rate=fpr,
        precision=precision,
        f1=f1,
        n_positive=tp + fn,
        n_negative=fp + tn,
        folds=folds,
    )
