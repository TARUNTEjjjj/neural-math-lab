# Decision record: measuring threshold-tie distance from 0.5

Status: **decided, 19 November 2026** (issue #12). Distance from 0.5 is measured
at the threshold nearest 0.5 within each equal-decision interval. This replaces
the earlier rule, which measured it at each candidate score.

## Context

`select_threshold` freezes a decision rule `p >= t` using validation rows only.
It minimizes `fp_cost × FP + fn_cost × FN`. When several rules have the same
minimum cost (up to the documented roundoff tolerance), it prefers the
threshold closest to 0.5, then the higher threshold.

Validation scores divide the thresholds from 0 to just above 1 into intervals.
Every threshold in one interval makes the same validation decisions:

- `(next lower distinct score, score]` for each distinct score, or `[0, score]`
  for the lowest score;
- `(highest score, ABOVE_ONE]` for the all-negative rule, where `ABOVE_ONE` is
  the smallest float64 above 1.

Validation cost cannot tell thresholds in one interval apart. The tie rule only
decides which interval is frozen, and which threshold represents it. Held-out
rows whose scores fall inside an interval are classified differently by
different thresholds in it.

## Options

**A. Candidate-score proximity (previous rule).** Each interval was represented
by its upper end, a validation score (or the sentinel). For an interval below
0.5 that is the end nearest 0.5. For an interval above 0.5 it is the farthest
end. With labels `[1, 0, 1, 0]` and scores `[0.9, 0.52, 0.3, 0.1]`, the rules
`(0.52, 0.9]` and `(0.1, 0.3]` both make one mistake. Option A compared 0.9
(0.4 from 0.5) with 0.3 (0.2 away) and selected 0.3. Yet every threshold just
above 0.52 makes the same decisions as 0.9 and is only 0.02 from 0.5.

**B. Nearest representable threshold in each interval (adopted).** Each
interval is represented by its float64 threshold closest to 0.5:

| Interval position | Representative |
| --- | --- |
| Contains 0.5 | 0.5 (always a candidate) |
| Entirely below 0.5 | Its upper end, a validation score |
| Entirely above 0.5 | The smallest float above its lower end, `nextafter(lower, +inf)` |

The four-row example now selects `nextafter(0.52) = 0.5200000000000001`. That
rule predicts only the 0.9 row positive, like 0.9 did.

## Decision and rationale

Option B is adopted, for three reasons.

1. **It makes the stated rule true.** The README, the guide and the report all
   say "ties prefer the threshold closest to 0.5". Option A kept that promise
   only for rules below 0.5. Its documented caveat said a tie "can select a
   lower threshold even when an equal-cost threshold above 0.5 lies closer".
   Option B measures both sides the same way, so the convention no longer
   favors predicting more positives in a tie.
2. **It fits the lab's decision theory.** The guide derives the cutoff
   `c_FP / (c_FP + c_FN)` for calibrated probabilities. With equal costs that
   is 0.5. For a calibrated score `p`, moving a threshold from 0.5 to `t`
   changes only decisions with `p` between 0.5 and `t`. Each such decision
   adds expected cost `|2p - 1| ≤ 2|t - 0.5|`. When validation data cannot
   distinguish thresholds, the nearest one to 0.5 bounds that extra cost most
   tightly, for every score distribution. Option A's choice depended on where
   the validation scores happen to lie, not on that distance. With unequal
   costs, 0.5 is not the theoretical cutoff under either option. Moving the
   anchor to `c_FP / (c_FP + c_FN)` would be a separate decision; it is not
   part of this one.
3. **On held-out data it did no worse in the selections where the rules
   differ.** The test labels of the committed examples were not used to make
   this choice: one held-out split is evidence about one model, not about a
   policy. A separate study ([`scripts/threshold_tie_study.py`](../scripts/threshold_tie_study.py))
   trained the default network for 20 seeds each of moons and circles, and
   froze thresholds for equal costs and for `fn_cost = 3`. It ran on 19
   November 2026 (Python 3.9.6, NumPy 2.0.2, macOS arm64). In 18 of the 80
   selections the two rules froze different thresholds. Held-out cost (120
   test rows each) was lower with Option B in 10, higher in 2 and equal in 6.
   The total over those 18 was 92 with Option B and 107 with Option A. The two
   higher-cost cases are both seed 7, the committed moons example, at the two
   cost settings. These are synthetic splits from one model family; they
   support the choice but do not guarantee it for any one run.

Costs of the decision, accepted knowingly:

- A selected threshold above 0.5 is now usually one float64 step above a
  validation score, for example `0.5304050535298335` instead of
  `0.5304050535298334`. Reports print the exact value. Rounded to a few digits
  it looks like the score itself, but the rule `p >= t` excludes that score.
- The all-negative rule is represented the same way. When the highest
  validation score is at least 0.5, the threshold just above it is frozen,
  rather than the `ABOVE_ONE` sentinel. It still predicts no validation
  positives, but a held-out row scored above every validation row is now
  predicted positive. When the highest score is exactly 1, the threshold is
  `ABOVE_ONE`, as before. When it is below 0.5, the threshold is 0.5, as
  before.
- In the committed moons example, the network's frozen threshold moves from
  0.6830711918697014 to 0.5304050535298335. Validation decisions are
  unchanged (FP 0, FN 1). On the held-out rows, false positives rise from 1 to
  3 (mean cost 0.016667 to 0.033333). This result is recorded rather than
  hidden. Choosing the rule because of it would have used test labels for
  model selection.

## What did not change

- Selection still uses validation rows only, and the test rows are evaluated
  once after the rule is frozen.
- Ties stay deterministic: the cost and distance tolerance is
  `8 × float64 epsilon`, relative to the fixed minimum, with no absolute
  floor. Exact 0.5 still beats a threshold one float step away, and equal
  distances still prefer the higher threshold.
- Tied scores always receive the same decision, because every representative
  lies strictly between distinct scores or on one.
- `candidate_count` still counts the distinct scores, the all-negative rule
  and 0.5 (when 0.5 is not a score).

## Examples, all covered by `tests/test_diagnostics.py`

| Labels | Scores | Costs FP / FN | Selected | Why |
| --- | --- | --- | --- | --- |
| 1, 0, 1, 0 | 0.9, 0.52, 0.3, 0.1 | 1 / 1 | 0.5200000000000001 | Rules (0.52, 0.9] and (0.1, 0.3] tie; 0.02 beats 0.2 from 0.5. Previously 0.3. |
| 0, 1, 0, 1 | 0.1, 0.4, 0.4, 0.9 | 1 / 1 | 0.5 | (0.4, 0.9] contains 0.5. The tied 0.4 scores are never split. |
| 0, 1, 0, 1 | 0.1, 0.4, 0.4, 0.9 | 1 / 3 | 0.4 | A lower cost beats any distance. |
| 1, 0 | 0.5, 0.2 | 1 / 1 | 0.5 | Exact 0.5: the 0.5 score is predicted positive. |
| 0, 1 | 0.5, 0.8 | 1 / 1 | 0.5000000000000001 | The best rule excludes the 0.5 score, so it starts one step above it. |
| 1, 0, 1, 0 | 0.9, 0.6, 0.6, 0.1 | 1 / 1 | 0.5 | Tied scores enter together; (0.1, 0.6] contains 0.5. |
| 1, 0, 1, 0 | 0.9, 0.7, 0.3, 0.1 | 1 / 1 | 0.7000000000000001 | 0.3 and just above 0.7 are equally far up to roundoff; the higher one wins. |
| 1, 1 | 0, 0 | 1 / 1 | 0 | Endpoint: all-positive; [0, 0] lies below 0.5. |
| 0, 0 | 1, 1 | 1 / 1 | 1.0000000000000002 | Endpoint: all-negative above a score of 1 is the sentinel. |
| 0, 0 | 0.6, 0.6 | 1 / 1 | 0.6000000000000001 | All-negative rule, nearest 0.5. |
| 0, 0 | 0.3, 0.3 | 1 / 1 | 0.5 | All-negative interval (0.3, ABOVE_ONE] contains 0.5. |

An independent oracle in the same test file tries 0, 0.5, `ABOVE_ONE`, every
score and the float just above every score. It uses exact integer costs and
exact rational distances on 300 seeded fixtures with tied scores, exact 0.5,
neighbors of 0.5 and both endpoints. The committed moons and circles examples
are checked against the rule from their saved validation scores.

This closes issue #12. Revisit only with a different anchor (such as the
cost-derived cutoff) or a different representative (such as interval
midpoints), as a new decision.
