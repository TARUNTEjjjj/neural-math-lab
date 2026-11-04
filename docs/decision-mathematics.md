# From a probability to a decision

A trained network produces a number. Using that number requires more choices:
which mistakes matter, which threshold turns a score into a class, and whether
the score behaves like a reliable probability. This chapter makes those choices
explicit, then uses the chain rule to explain local input sensitivity.

Read [Beyond XOR](advanced-mathematics.md) first. Open a newly generated moons or
circles report beside this guide, and use the [worked exercises](decision-exercises.md)
to check your understanding. The calculations live in
[`diagnostics.py`](../neural_math_lab/diagnostics.py); they use NumPy directly.

## 1. A score and a class answer different questions

For each row $i$, the model produces $p_i\in[0,1]$. A decision threshold $t$
defines

$$
\widehat y_i(t)=\mathbf 1[p_i\geq t].
$$

The equality matters: a score exactly equal to the threshold is class 1.
Changing $t$ changes decisions without changing the model's weights or scores.
For a fixed set of scores, lowering $t$ cannot decrease the number of predicted
positives; it can recover missed positives and introduce false positives.

| Actual label | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| 0 | True negative, TN | False positive, FP |
| 1 | False negative, FN | True positive, TP |

Our confusion matrix is `[[TN, FP], [FN, TP]]`. Precision is
$TP/(TP+FP)$; recall is $TP/(TP+FN)$. The fixed-threshold summaries report 0
for an undefined precision or recall, and expose the counts so you can see why.
For example, no predicted positives makes precision undefined, rather than
evidence that a meaningful estimate of precision is zero.

The original metric cards and boundary replay retain threshold 0.5. The
**decision lab** uses the validation-selected model and lets you explore a
separate threshold on validation rows. Its held-out summary stays fixed at the
threshold chosen by Python before test evaluation.

## 2. Make the cost of mistakes explicit

Let $c_{FP}>0$ be the cost of one false positive and $c_{FN}>0$ the cost of one
false negative. The empirical mean cost on a partition with $N$ rows is

$$
\widehat C(t)=\frac{c_{FP}FP(t)+c_{FN}FN(t)}{N}.
$$

The numerator has units of cost; dividing by $N$ gives cost per example.
Multiplying both costs by the same positive constant changes the displayed
cost, but not the minimizing decision rule. Costs are assumptions chosen for
the exercise, not values the neural network learns.

In this lab both costs default to 1. To make a missed class-1 example three
times as expensive as a false alarm, run:

```sh
python -m neural_math_lab experiment --dataset moons --fp-cost 1 --fn-cost 3 --output runs/moons-cost-3
```

The sequence is:

1. Fit the scaler and model on training rows.
2. Restore the epoch with the smallest validation cross-entropy.
3. Select a threshold minimizing validation cost using that restored model.
4. Apply the frozen model, scaler, and threshold to the test rows.

The baseline gets its own validation-selected epoch and threshold, using the
same costs and data partitions. Neither model uses test labels to select a
threshold. Since validation is used for both epoch and threshold selection,
its best cost can be optimistic; the separate test partition still matters.
Threshold fitting is itself model selection, as discussed in the
[scikit-learn threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html).

### Why finitely many thresholds are enough

Sort the distinct validation scores. Between two consecutive scores, every
predicted label is constant. It is therefore sufficient to evaluate the
distinct scores, plus a threshold that predicts every row as negative. The
smallest score already represents an all-positive rule. Equal scores must
enter together; their row order cannot decide which class they receive.

The implementation also includes 0.5 as a candidate. It minimizes cost, then
breaks exact cost ties by choosing the candidate closest to 0.5, then the higher
threshold if both distance and cost tie. This is a reproducible convention,
not a claim that 0.5 has special optimality. The search is exhaustive over the
validation decision patterns; it cannot promise the best threshold on new data.

For the all-negative rule, the stored threshold is `np.nextafter(1.0, np.inf)`:
the smallest representable float64 number above 1. This finite sentinel is
approximately `1.0000000000000002`. It includes scores that round to exactly 1,
unlike a threshold of 1, and stays valid JSON. Interpret it as **predict none
positive**, not as a probability greater than 1.

### A complete four-row example

Use these scores from a hypothetical already-fitted model:

| Row | True label $y$ | Score $p$ |
| --- | --- | --- |
| A | 0 | 0.1 |
| B | 1 | 0.4 |
| C | 0 | 0.4 |
| D | 1 | 0.9 |

| Threshold | TN | FP | FN | TP | Mean cost, $c_{FP}=c_{FN}=1$ | Mean cost, $c_{FP}=1,c_{FN}=3$ |
| --- | --- | --- | --- | --- | --- | --- |
| Above 1 | 2 | 0 | 2 | 0 | $2/4$ | $6/4$ |
| 0.9 | 2 | 0 | 1 | 1 | $1/4$ | $3/4$ |
| 0.5 | 2 | 0 | 1 | 1 | $1/4$ | $3/4$ |
| 0.4 | 1 | 1 | 0 | 2 | $1/4$ | $1/4$ |
| 0.1 | 0 | 2 | 0 | 2 | $2/4$ | $2/4$ |

With equal costs, 0.4, 0.5, and 0.9 tie; the stated convention selects 0.5.
With missed positives costing three times as much, the selected threshold is
0.4. Rows B and C have identical scores, so a score threshold cannot classify
both correctly. More threshold searching cannot solve that ambiguity.

If $p$ were the true conditional probability, the expected cost of predicting
1 would be $c_{FP}(1-p)$, while predicting 0 would cost $c_{FN}p$. Comparing
them gives the theoretical cutoff $c_{FP}/(c_{FP}+c_{FN})$. Here the network's
outputs are estimates and a validation set is finite. The empirical selected
threshold need not equal that theoretical cutoff: in the example, $1/4$ is
the theoretical value for costs 1 and 3, while 0.4 is our selected candidate.

## 3. ROC: how well are positives ranked above negatives?

At each distinct score threshold, compute

$$
TPR=\frac{TP}{TP+FN},\qquad FPR=\frac{FP}{FP+TN}.
$$

The ROC curve plots $FPR$ horizontally and $TPR$ vertically. Starting with
all-negative predictions and lowering the threshold, the example produces

$$
(0,0),\ (0,1/2),\ (1/2,1),\ (1,1).
$$

The trapezoidal area is

$$
AUC=\sum_k(FPR_k-FPR_{k-1})\frac{TPR_k+TPR_{k-1}}2
=\frac12\frac{1/2+1}2+\frac12(1)=\frac78.
$$

An independent interpretation checks the arithmetic. Compare every positive
score with every negative score, awarding 1 for a larger score, 1/2 for a tie,
and 0 for a smaller score. The positives $[0.4,0.9]$ versus negatives
$[0.1,0.4]$ earn $[1,1/2,1,1]$, averaging $3.5/4=0.875$. Grouping ties before
forming the curve makes its trapezoidal area agree with this pairwise result.

AUC describes ranking across thresholds; it is neither accuracy at the chosen
threshold nor a probability-calibration measurement. If the evaluated labels
contain only one class, there are no positive–negative pairs. This lab reports
ROC AUC as `null` and any rate with a missing denominator as `null`, rather than
inventing a curve value. For the standard metric definition, see the
[official ROC metric documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics).

## 4. Precision–recall: what happens among predicted positives?

For the same descending thresholds, the example's $(recall,precision)$ pairs
are $(0,1)$, $(1/2,1)$, $(1,2/3)$, and $(1,1/2)$. The first point's precision
1 is a plotting convention for predicting no positives; it is not an observed
precision and does not replace the zero-denominator convention in metric cards.

Average precision uses the recall increments:

$$
AP=\sum_k(R_k-R_{k-1})P_k
=\frac12(1)+\frac12\left(\frac23\right)+0\left(\frac12\right)
=\frac56.
$$

This is a step-weighted sum, not trapezoidal area under a precision–recall
curve. Linear interpolation would give a different quantity. The
[official average-precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
specifies the same non-interpolated sum.

Here AP is `null` when there are no positive labels, because recall has no
denominator. With every label positive, AP is 1; that result does not establish
an ability to reject negative examples. Class prevalence affects precision,
so compare datasets and splits with their positive counts in view.

## 5. Reliability: does a score of 0.8 behave like 80%?

The report groups scores into ten equal-width intervals. Bin $b$ is
$[b/B,(b+1)/B)$ for $b=0,\ldots,B-2$; the final bin includes 1. Thus with ten
bins, 0 belongs in the first and 1 in the last. For each nonempty bin $I_b$,
compute the mean score and observed positive fraction:

$$
\overline p_b=\frac1{n_b}\sum_{i\in I_b}p_i,\qquad
\overline y_b=\frac1{n_b}\sum_{i\in I_b}y_i.
$$

Plot $(\overline p_b,\overline y_b)$. Points near the diagonal indicate
agreement within those bins on those rows. An empty bin has count 0 and
`null` averages; drawing it at $(0,0)$ would falsely invent an observation.
This is a diagnostic view, not a procedure that recalibrates the network.
[Calibration-curve documentation](https://scikit-learn.org/stable/modules/calibration.html#calibration-curves)
explains the underlying comparison.

Our **class-1 probability ECE** is the weighted absolute bin gap:

$$
ECE=\sum_{b:n_b>0}\frac{n_b}{N}|\overline p_b-\overline y_b|.
$$

It bins $p$ directly. It does not bin the predicted class's confidence
$\max(p,1-p)$ or average the correctness of a hard label. Those are different
variants of expected calibration error; specify which one you report. ECE
depends on the chosen bins and sample size. Sparse bins and small ECE alone
do not justify claiming calibrated probabilities. For context on binned
calibration diagnostics, see [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).

For a hand calculation, use **two bins** on the four-row example:

| Bin | Scores | Count | Mean score | Positive fraction | Weighted absolute gap |
| --- | --- | --- | --- | --- | --- |
| $[0,0.5)$ | 0.1, 0.4, 0.4 | 3 | $3/10$ | $1/3$ | $(3/4)(1/30)=1/40$ |
| $[0.5,1]$ | 0.9 | 1 | $9/10$ | 1 | $(1/4)(1/10)=1/40$ |

The ECE is $1/20=0.05$. Ten bins on the same four rows give 0.1. The predictions
have not changed; grouping changed the diagnostic. This is why the exported
bin count and per-bin observations matter.

The binary Brier score uses every row directly:

$$
BS=\frac1N\sum_i(p_i-y_i)^2
=\frac{0.01+0.36+0.16+0.01}{4}=0.135.
$$

This score uses the binary $[0,1]$ convention and is independent of the
classification threshold. Lower is better for squared probability error, but
the score reflects more than calibration alone. See the
[Brier score definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
and [calibration discussion](https://scikit-learn.org/stable/modules/calibration.html).

## 6. Input sensitivity: apply the chain rule to raw coordinates

The earlier backpropagation chapter differentiates loss with respect to
weights. Now hold the learned parameters fixed and differentiate one output
$p(x)$ with respect to the two raw input coordinates.

For one row, let $\widetilde x_j=(x_j-\mu_j)/s_j$,
$a_k=\sum_j\widetilde x_jW_{1,jk}+b_{1,k}$,
$h_k=\tanh(a_k)$, $z=\sum_kh_kW_{2,k}+b_2$, and $p=\sigma(z)$.
Each link contributes a derivative:

$$
\frac{\partial p}{\partial z}=p(1-p),\qquad
\frac{\partial z}{\partial h_k}=W_{2,k},\qquad
\frac{\partial h_k}{\partial a_k}=1-h_k^2,\qquad
\frac{\partial a_k}{\partial x_j}=\frac{W_{1,jk}}{s_j}.
$$

Adding the paths through all hidden units gives

$$
\boxed{\frac{\partial p}{\partial x_j}
=\frac{p(1-p)}{s_j}\sum_{k=1}^{m}W_{1,jk}(1-h_k^2)W_{2,k}.}
$$

For a batch, $H$ has shape $(N,m)$, $W_2^T$ has shape $(1,m)$,
and $W_1^T$ has shape $(m,2)$. The vectorized expression is

```python
dp_d_standard = ((1 - H**2) * W2.T) @ W1.T   # (N, 2)
dp_d_raw = (P * (1 - P)) * dp_d_standard / scale
```

`P` has shape `(N, 1)` and `scale` has shape `(1, 2)`, so broadcasting scales
each row and coordinate appropriately. There is **no division by batch size**:
each row contains the gradient of its own probability, not the gradient of a
mean loss. Forgetting $1/s_j$ gives sensitivity per standardized unit instead
of per raw unit. The baseline uses $p(1-p)W_j/s_j$.

A positive component means that an infinitesimal increase in that coordinate,
holding the other one fixed, increases class-1 probability locally. Its units
are probability change per raw coordinate unit. Components measured in
different units should not be compared as universal feature importance.

### A numerical example with every step visible

Use one hidden unit, $W_1=[2,-1]^T$, $b_1=0$, $W_2=3$, $b_2=0$,
$\mu=[1,2]$, and $s=[2,4]$. At the raw point $x=[1,2]$:

$$
\widetilde x=[0,0],\quad a=h=z=0,\quad p=1/2.
$$

The raw gradient is

$$
\nabla_xp=\left[\frac{(1/4)(2)(3)}2,
\frac{(1/4)(-1)(3)}4\right]=[0.75,-0.1875].
$$

For displacement $\delta=[0.02,-0.04]$, the first-order approximation gives

$$
p(x+\delta)\approx p(x)+\nabla_xp\cdot\delta
=0.5+0.75(0.02)+(-0.1875)(-0.04)=0.5225.
$$

The actual perturbed calculation uses $\widetilde{x+\delta}=[0.01,-0.01]$,
$a=0.03$, and
$\sigma(3\tanh(0.03))\approx0.5224780909$. Its difference from the linear
approximation is about $2.19\times10^{-5}$. Larger displacements can make
curvature matter much more; the approximation can even leave $[0,1]$ while the
actual sigmoid stays bounded.

An independent numerical check perturbs the raw coordinate itself:

$$
\frac{\partial p}{\partial x_j}\approx
\frac{p(x+\epsilon e_j)-p(x-\epsilon e_j)}{2\epsilon}.
$$

Here $e_j$ is the unit vector for coordinate $j$. Use a small finite
$\epsilon$, such as $10^{-5}$ for these scales, and vary it to check stability.
Too large a step introduces curvature error; too small a step amplifies
floating-point cancellation. Saturated `tanh` or sigmoid outputs can yield
very small gradients. That describes the local function, not reliability or
robustness over a large region. A derivative of a learned predictor also does
not establish a causal effect in the world.

## 7. Turn exploration into an explanation

Before moving a control, predict what will change and what will stay fixed.
Lowering the threshold changes confusion counts, but not probabilities, Brier,
AUC, or AP. Moving an input changes that point's probability and gradient, but
does not retrain the weights. Selecting a replay checkpoint changes the point
inspector's model; the decision lab evaluates the fixed selected model.

Record costs, threshold, sample counts, and which split supplied each metric.
Do not choose a cost ratio or model by repeatedly optimizing displayed test
results. Those test rows would then become part of your development process;
reserve fresh evaluation data for a later final claim. Small synthetic
experiments teach these distinctions, without establishing deployment quality.
