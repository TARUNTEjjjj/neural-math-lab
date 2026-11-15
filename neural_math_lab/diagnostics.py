"""Binary decisions, ranking, calibration, and input derivatives in NumPy.

The functions inspect supplied scores. They neither fit a model nor calibrate
its output. ``run_experiment`` supplies validation rows for threshold selection
and evaluates the held-out rows only after fixing that decision rule.
"""

import math

import numpy as np

from .core import (Network, _integer, _matmul, _matrix, _targets,
                   _sigmoid_derivative, _tanh_derivative)
from .datasets import _bounded_real, standardize


# A finite, JSON-safe threshold that predicts class 0 even for a score of 1.
ABOVE_ONE = float(np.nextafter(1.0, np.inf))
MAX_COST = 1_000_000.0
TIE_BREAK = "lowest cost, then closest to 0.5, then higher threshold"
# A few float64 rounding steps must not choose between mathematically tied
# costs or symmetric distances. No absolute floor: zero still beats any
# positive cost, including costs much smaller than machine epsilon.
TIE_REL_TOL = 8 * np.finfo(np.float64).eps


def _scores(y, p):
    p = _matrix(p, "p")
    y = _targets(y, len(p))
    if not len(p) or p.shape != y.shape or np.any((p < 0) | (p > 1)):
        raise ValueError("p must be nonempty with shape (N, 1) and values in [0, 1]")
    return y[:, 0].astype(np.int64), p[:, 0]


def _cost(value, name):
    value = _bounded_real(value, name, 0, MAX_COST)
    if value == 0:
        raise ValueError(name + " must be greater than zero")
    return value


def _rank_counts(y, p):
    """Distinct scores descending, with each tie group admitted as a whole."""
    order = np.argsort(-p, kind="stable")
    ranked = p[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(p) - 1]
    true_positive = np.cumsum(y[order], dtype=np.int64)[ends]
    false_positive = ends + 1 - true_positive
    return (np.r_[ABOVE_ONE, ranked[ends]],
            np.r_[0, true_positive], np.r_[0, false_positive])


def _counts(y, p, threshold, fp_cost, fn_cost):
    predicted = p >= threshold
    actual = y == 1
    tp, fp = int(np.sum(predicted & actual)), int(np.sum(predicted & ~actual))
    fn, tn = int(np.sum(~predicted & actual)), int(np.sum(~predicted & ~actual))
    return _count_document(len(y), tp, fp, tn, fn, threshold, fp_cost, fn_cost)


def _count_document(samples, tp, fp, tn, fn, threshold, fp_cost, fn_cost):
    total_cost = float(fp_cost * fp + fn_cost * fn)
    return {"threshold": float(threshold), "fp_cost": fp_cost, "fn_cost": fn_cost,
            "samples": samples, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "total_cost": total_cost, "mean_cost": total_cost / samples}


def select_threshold(y, p, fp_cost=1.0, fn_cost=1.0):
    """Minimize empirical cost on the supplied validation rows in O(N log N).

    Enumerate distinct scores plus 0.5 and an above-one sentinel. With the rule
    ``p >= threshold``, this includes all distinct classification patterns,
    including all-positive and all-negative. Cost ties prefer the candidate
    closest to 0.5, then the higher threshold. Costs and distances within eight
    float64 epsilons of their respective minimum (relative, no absolute floor)
    are tied. This is a deterministic convention, not a claim that the selected
    threshold generalizes to another population.

    Distance is measured at the candidate itself. A score is the highest
    threshold giving its pattern, so a pattern above 0.5 is measured at its far
    end: a tie can pick a lower threshold although a non-candidate threshold
    with the same validation decisions lies closer to 0.5 (for example,
    y=[1,0,1,0] and p=[.9,.52,.3,.1] select 0.3, not a value in (0.52, 0.9]).
    """
    y, p = _scores(y, p)
    fp_cost, fn_cost = _cost(fp_cost, "fp_cost"), _cost(fn_cost, "fn_cost")
    thresholds, tps, fps = _rank_counts(y, p)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    candidates = [(float(t), int(tp), int(fp)) for t, tp, fp in zip(thresholds, tps, fps)]
    if not np.any(thresholds == 0.5):
        middle = p >= 0.5
        candidates.append((0.5, int(y[middle].sum()), int(middle.sum() - y[middle].sum())))
    costs = [fp_cost * fp + fn_cost * (positives - tp) for _, tp, fp in candidates]
    minimum_cost = min(costs)
    cost_ties = [item for item, cost in zip(candidates, costs)
                 if math.isclose(cost, minimum_cost, rel_tol=TIE_REL_TOL, abs_tol=0.0)]
    # Compare with fixed minima, not the previous candidate: approximate
    # equality is not transitive and must not drift with enumeration order.
    minimum_distance = min(abs(item[0] - 0.5) for item in cost_ties)
    threshold, tp, fp = max(
        (item for item in cost_ties
         if math.isclose(abs(item[0] - 0.5), minimum_distance,
                         rel_tol=TIE_REL_TOL, abs_tol=0.0)),
        key=lambda item: item[0])
    result = _count_document(len(y), tp, fp, negatives - fp, positives - tp,
                             threshold, fp_cost, fn_cost)
    result.update({"candidate_count": len(candidates), "tie_break": TIE_BREAK})
    return result


def evaluate_binary(y, p, threshold=0.5, fp_cost=1.0, fn_cost=1.0, bins=10):
    """JSON-native threshold, ranking, and reliability diagnostics.

    ROC integrates trapezoids after grouping ties. Average precision sums
    recall increments times precision at each distinct score (not trapezoids).
    One-class ROC AUC is undefined (None); AP is None with no positives.
    Reliability uses equal-width bins, left-inclusive and right-exclusive except
    the final bin includes 1. ECE weights each occupied bin's absolute gap by its
    sample fraction. Empty-bin statistics are None, not misleading zero values.
    """
    y, p = _scores(y, p)
    threshold = _bounded_real(threshold, "threshold", 0, ABOVE_ONE)
    fp_cost, fn_cost = _cost(fp_cost, "fp_cost"), _cost(fn_cost, "fn_cost")
    bins = _integer(bins, "bins", 1, 100)
    result = _counts(y, p, threshold, fp_cost, fn_cost)
    tp, fp, tn, fn = (result[key] for key in ("tp", "fp", "tn", "fn"))
    with np.errstate(under="ignore"):
        brier = float(np.mean((p - y) ** 2))
    result.update({
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "predicted_positive": tp + fp,
        "brier": brier,
    })
    thresholds, tps, fps = _rank_counts(y, p)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    roc, pr = [], []
    for index, (threshold, tp, fp) in enumerate(zip(thresholds, tps, fps)):
        tpr, fpr = (float(tp / positives) if positives else None,
                    float(fp / negatives) if negatives else None)
        precision = float(tp / (tp + fp)) if index else 1.0
        roc.append({"threshold": float(threshold), "fpr": fpr, "tpr": tpr,
                    "tp": int(tp), "fp": int(fp)})
        pr.append({"threshold": float(threshold), "precision": precision,
                   "recall": tpr, "tp": int(tp), "fp": int(fp)})
    auc, ap = None, None
    if positives and negatives:
        fpr, tpr = fps / negatives, tps / positives
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    if positives:
        precision = tps[1:] / (tps[1:] + fps[1:])
        ap = float(np.sum(np.diff(tps / positives) * precision))
    edges = np.arange(bins + 1, dtype=np.float64) / bins
    # Search over actual edges avoids disagreement at boundaries such as 0.3.
    memberships = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    reliability, ece = [], 0.0
    for index in range(bins):
        mask = memberships == index
        count = int(np.sum(mask))
        with np.errstate(under="ignore"):
            mean_probability = float(np.mean(p[mask])) if count else None
        fraction_positive = float(np.mean(y[mask])) if count else None
        gap = abs(mean_probability - fraction_positive) if count else None
        if count:
            ece += count / len(y) * gap
        reliability.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                            "count": count, "mean_probability": mean_probability,
                            "fraction_positive": fraction_positive, "absolute_gap": gap})
    result.update({"roc_auc": auc, "average_precision": ap,
                   "roc": roc, "precision_recall": pr,
                   "reliability_bins": reliability, "ece": float(ece)})
    return result


def input_sensitivity(network, raw, mean, scale):
    """Return dP/d(raw input), one row per point and one column per coordinate.

    dP/dx_j = P(1-P) * sum_k[W1[j,k](1-H_k^2)W2[k,0]] / scale_j.
    P(1-P) and 1-H_k^2 are evaluated from z2 and z1, before activation, so an
    output rounded to exactly 1 or a tanh rounded to +-1 keeps a representable
    slope.
    These are local derivatives of a fitted model, not causal explanations.
    """
    if not isinstance(network, Network):
        raise ValueError("network must be a Network")
    X = standardize(raw, mean, scale)
    scale = _matrix(scale, "scale")
    values, weights = network.forward(X), network._parameters()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            hidden_derivatives = _tanh_derivative(values["z1"]) * weights["W2"].T
            logit_derivatives = _matmul(hidden_derivatives, weights["W1"].T) / scale
            result = _sigmoid_derivative(values["z2"]) * logit_derivatives
    except FloatingPointError as exc:
        raise ValueError("Input sensitivity overflowed; use smaller inputs or weights") from exc
    if not np.isfinite(result).all():
        raise ValueError("Input sensitivity produced a non-finite value")
    return result
