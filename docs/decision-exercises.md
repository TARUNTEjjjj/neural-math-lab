# Decision lab: predict, calculate, then inspect

Use [the decision mathematics guide](decision-mathematics.md) as a reference.
Try the questions before opening the answers. A useful notebook entry contains
your prediction, intermediate calculations, observed result, and explanation
of any disagreement. Treat validation as development data and the test split
as an evaluation of already-frozen choices.

## Questions

### 1. Follow the equality sign

For labels $[0,1,0,1]$ and scores $[0.1,0.4,0.4,0.9]$, write all four
predicted labels at thresholds 0.4, 0.5, and 0.9. Calculate the confusion matrix,
precision, and recall at 0.4 and 0.5. Can any score threshold distinguish the
two examples with score 0.4?

### 2. Choose costs before looking at the result

Use the same four rows. Which threshold does the lab select with equal costs?
What changes with `fp_cost=1` and `fn_cost=3`? Explain both the cost calculation
and the tie convention. Would costs 10 and 30 select different decisions from
costs 1 and 3?

Run a new moons experiment with costs 1 and 3. Before opening its test summary,
explain why a lower selected threshold is plausible but not guaranteed to be
strictly lower than for equal costs. Explain why its test cost can exceed its
validation cost.

### 3. Derive a theoretical cutoff

Assume $p=P(Y=1\mid X=x)$ is the true probability, a false positive costs 2,
and a false negative costs 8. Compare the conditional expected cost of the two
possible decisions and derive a threshold. Why need the empirical threshold
selected from a finite validation sample not equal it?

### 4. Check ranking two different ways

For the four rows in question 1, calculate ROC AUC from the trapezoids and then
from all positive–negative pairs. Calculate AP using recall increments. Why
must tied scores be processed together? If every label is 1, which of AUC and
AP is defined in this lab? What if every label is 0?

### 5. Change the bins without changing the model

Calculate the four-row example's ECE with two bins and with ten bins. Calculate
its Brier score. Does a lower two-bin ECE mean that the model became better?
For ten bins, say where scores exactly 0, 0.5, and 1 belong, and what should be
displayed for a bin containing no examples.

### 6. Keep derivatives in the correct units

Use $W_1=[2,-1]^T$, $b_1=0$, $W_2=3$, $b_2=0$, $\mu=[1,2]$, and
$s=[2,4]$ at $x=[1,2]$. Derive $\nabla_xp$. Predict the effect of displacement
$[0.02,-0.04]$ by a dot product, then calculate the actual perturbed output.
What would the incorrect gradient be if you omitted the standardizer's
derivative? Does duplicating this input into a batch of ten divide each row's
gradient by ten?

### 7. Find a limit of local reasoning

In the report's input inspector, choose a point near a boundary and make a
small displacement. Record the actual and first-order predicted probability
changes. Repeat with a larger displacement. Then choose a saturated point.
Explain why a nearly zero gradient does not establish that every point in a
large surrounding area has the same predicted class. There is no required
numerical outcome; report what your selected model actually does.

### 8. Catch selection leakage

A researcher tries 20 false-negative costs and reports the one yielding the
best test $F_1$. Another chooses a cost before running, selects a threshold on
validation cost, then reports test cost. Which choices depend on the test
labels in each procedure? Can the first researcher fix the problem simply by
renaming the chosen threshold “validation selected”?

---

## Worked answers

### 1. Decisions and counts

Threshold 0.4 predicts $[0,1,1,1]$. Thresholds 0.5 and 0.9 both predict
$[0,0,0,1]$. At 0.4 the confusion matrix is $[[1,1],[0,2]]$, so precision is
$2/3$ and recall is 1. At 0.5 it is $[[2,0],[1,1]]$, so precision is 1 and
recall is $1/2$. Equal scores always receive equal decisions under the rule
$p\geq t$; row order cannot legitimately break the tie.

### 2. Costs and tie handling

For equal costs, thresholds 0.4, 0.5, and 0.9 each make one mistake and have
mean cost $1/4$. The lab prefers the candidate closest to 0.5 and selects 0.5.
If distances also tie, it prefers the higher candidate.

With costs 1 and 3, threshold 0.4 has one false positive and cost $1/4$.
Thresholds 0.5 and 0.9 have one false negative and cost $3/4$. The selected
threshold is 0.4. Scaling both costs by ten leaves this ordering unchanged and
multiplies mean cost by ten.

Increasing the relative false-negative cost rewards recovering missed
positives, but a finite set of threshold rules can retain the same optimum
over a range of costs. A strictly lower threshold is not guaranteed.
The chosen threshold minimizes the observed validation cost among candidates;
test rows need not have the same score–label arrangement or cost.

### 3. Expected cost

Predicting 1 costs $2(1-p)$ in expectation, and predicting 0 costs $8p$.
Choose 1 when $2(1-p)\leq8p$, or $p\geq1/5$. Estimated scores need not equal
true conditional probabilities, and a finite validation set has sampling
variation. The empirical search also represents decision patterns using
specific candidate values. It need not return 0.2.

### 4. AUC and AP

ROC points are $(0,0)$, $(0,1/2)$, $(1/2,1)$, $(1,1)$. Their area is
$3/8+1/2=7/8$. Pairwise scores are $1,1/2,1,1$, again averaging $7/8$.
The two tied rows enter together when the threshold reaches 0.4. Processing
one before the other would let arbitrary row order alter a metric.

Recall increases from 0 to $1/2$ at precision 1, then from $1/2$ to 1 at
precision $2/3$. Thus $AP=(1/2)(1)+(1/2)(2/3)=5/6$.
All-positive labels yield undefined AUC and AP 1. All-negative labels yield
undefined AUC and undefined AP. The exported undefined values are `null`.

### 5. Calibration arithmetic

With two bins, the contributions are $(3/4)|3/10-1/3|=1/40$ and
$(1/4)|9/10-1|=1/40$, giving ECE $1/20$.

With ten bins, the occupied bins contain $[0.1]$, $[0.4,0.4]$, and $[0.9]$.
Their contributions are $(1/4)(0.1)$, $(1/2)(0.1)$, and $(1/4)(0.1)$,
giving ECE 0.1. Brier is $(0.01+0.36+0.16+0.01)/4=0.135$ for either binning.
The model has not changed. Coarser averaging can hide disagreements.

Scores 0, 0.5, and 1 belong to zero-based bins 0, 5, and 9. Empty bins have
count 0 and undefined (`null`) means, not an invented point on the diagonal.

### 6. Sensitivity

At the stated point, standardized inputs, hidden activation, and logit are
zero, so $p=1/2$. Therefore
$\nabla_xp=[(1/4)(2)(3)/2,(1/4)(-1)(3)/4]=[0.75,-0.1875]$.
The first-order output is $0.5+0.75(0.02)-0.1875(-0.04)=0.5225$.

The actual output is $\sigma(3\tanh(0.03))\approx0.5224780909$, a difference
of about $2.19\times10^{-5}$. Omitting the scale gives $[1.5,-0.75]$, which
is a derivative with respect to standardized inputs, incorrectly interpreted
as a raw-input derivative. Duplicating rows does not change an individual
prediction's derivative; there is no batch-mean loss in this calculation.

### 7. Curvature and saturation

The first-order formula keeps the gradient at the original point fixed.
The actual gradient can change along a displacement, so a larger step can
cross a curved boundary even when the original derivative is small. A small
gradient is local evidence about this function and point. It is not a
certificate of region-wide robustness, predictive correctness, or causality.

### 8. Selection leakage

In the first procedure, test labels influence the selected cost ratio and
therefore the reported rule. Each threshold may have been fitted on validation,
but choosing which fitted threshold to report uses the test set. Renaming
the result does not remove that dependency. It is exploratory development
unless followed by evaluation on fresh reserved data.

The second procedure keeps the stated decision rule independent of test
labels. Its test score estimates performance for those frozen choices, subject
to the limits of the small synthetic dataset and its sampling procedure.
