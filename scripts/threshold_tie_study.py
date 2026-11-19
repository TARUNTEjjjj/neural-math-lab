"""Compare two ways of measuring threshold-tie distance on held-out rows.

After installing the package (python -m pip install -e .), run this from the
repository root; it takes a few minutes on a laptop:

    python scripts/threshold_tie_study.py

For 20 seeds of the default moons and circles experiments, and equal or 3x
false-negative costs, this freezes the network's threshold with the current
rule (distance at each equal-decision interval's threshold nearest 0.5) and
with the previous rule (distance at each candidate score), then counts test
cost for both. Both rules minimize the same validation cost; they differ only
in which tied decision pattern and which threshold inside it are frozen.
Decision costs do not change trained weights, so each seed's network is
trained once per dataset and reused for both cost settings.
docs/threshold-ties.md records the result. This is evidence about a policy on
synthetic splits, not a per-run guarantee.
"""

import math

import numpy as np

from neural_math_lab import diagnostics
from neural_math_lab.datasets import standardize
from neural_math_lab.experiment import run_experiment


def previous_rule(y, p, fp_cost, fn_cost):
    """The pre-#12 selection: candidate scores, 0.5 and the sentinel as is."""
    y, p = diagnostics._scores(y, p)
    upper, tps, fps = diagnostics._rank_counts(y, p)
    positives = int(y.sum())
    candidates = [(float(t), int(tp), int(fp)) for t, tp, fp in zip(upper, tps, fps)]
    if not np.any(upper == 0.5):
        middle = p >= 0.5
        candidates.append((0.5, int(y[middle].sum()), int(middle.sum() - y[middle].sum())))
    costs = [fp_cost * fp + fn_cost * (positives - tp) for _, tp, fp in candidates]
    tolerance = diagnostics.TIE_REL_TOL
    ties = [item for item, cost in zip(candidates, costs)
            if math.isclose(cost, min(costs), rel_tol=tolerance, abs_tol=0.0)]
    nearest = min(abs(item[0] - 0.5) for item in ties)
    return max((item for item in ties
                if math.isclose(abs(item[0] - 0.5), nearest, rel_tol=tolerance, abs_tol=0.0)),
               key=lambda item: item[0])[0]


def test_cost(y, p, threshold, fp_cost, fn_cost):
    predicted = p >= threshold
    return fp_cost * int(np.sum(predicted & (y == 0))) + fn_cost * int(np.sum(~predicted & (y == 1)))


def main():
    differing = []
    runs = 0
    for dataset in ("moons", "circles"):
        for seed in range(1, 21):
            result = run_experiment(dataset=dataset, seed=seed)
            data, network = result["data"], result["network"]
            val_p = network.predict_proba(standardize(data["X"][data["val_idx"]], data["mean"], data["scale"]))
            val_y = data["y"][data["val_idx"]]
            test_p = network.predict_proba(standardize(data["X"][data["test_idx"]], data["mean"], data["scale"]))[:, 0]
            test_y = data["y"][data["test_idx"]][:, 0]
            for fn_cost in (1.0, 3.0):
                runs += 1
                current = diagnostics.select_threshold(val_y, val_p, 1.0, fn_cost)["threshold"]
                previous = previous_rule(val_y, val_p, 1.0, fn_cost)
                if current != previous:
                    row = (dataset, seed, fn_cost, previous, current,
                           test_cost(test_y, test_p, previous, 1.0, fn_cost),
                           test_cost(test_y, test_p, current, 1.0, fn_cost))
                    differing.append(row)
                    print("{} seed {:2} fn_cost {:g}: previous {:.6f} (test cost {:g}), current {:.6f} (test cost {:g})".format(
                        row[0], row[1], row[2], row[3], row[5], row[4], row[6]), flush=True)
    print("{} of {} selections differ".format(len(differing), runs))
    print("test cost over those selections: previous {:g}, current {:g}".format(
        sum(row[5] for row in differing), sum(row[6] for row in differing)))
    print("current lower / higher / equal: {} / {} / {}".format(
        sum(row[6] < row[5] for row in differing), sum(row[6] > row[5] for row in differing),
        sum(row[6] == row[5] for row in differing)))


if __name__ == "__main__":
    main()
