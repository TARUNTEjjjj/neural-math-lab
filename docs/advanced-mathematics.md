# Beyond XOR: learning a pattern, then testing it

The [first mathematics guide](mathematics.md) explains every derivative in the
network. This chapter keeps that same network and asks a harder question: does
it learn something useful about examples that did not determine its weights?

The new experiment uses hundreds of synthetic two-dimensional points. You can
see the entire input space, compare a neural network with a simpler baseline,
and trace how the result changes during training. This is a controlled learning
exercise, not evidence of performance on a real-world application.

Start with:

```sh
python -m neural_math_lab experiment --dataset moons --output runs/moons
```

The dataset, initialization, and batch ordering are seeded. Record the seed and
all settings when comparing runs. Using the same seed makes a comparison easier
to reproduce; it does not make that seed a representative estimate of every
possible run.

## 1. Why two-dimensional data is useful

**Moons** contains two curved groups. **Circles** contains an inner and an outer
ring. Noise displaces the coordinates, so points from different classes may
overlap. Their labels come from the group used to generate them, not from the
network's predictions.

The geometry makes the models' different assumptions visible:

- Logistic regression predicts $p=\sigma(x^Tw+b)$. Its $p=0.5$ boundary is a
  straight line in the two input coordinates.
- The neural network predicts
  $p=\sigma(\tanh(x^TW_1+b_1)W_2+b_2)$. Its hidden features allow a curved
  boundary.

A curved boundary can be useful for these datasets, but flexibility alone does
not establish that the neural network will perform better on a particular
finite test set. Initialization, optimization, noise, and the sample split all
matter. We measure the comparison instead of assuming its outcome.

The generators in [`datasets.py`](../neural_math_lab/datasets.py) produce the
examples locally. There is no external dataset to download and no user data is
collected. The `noise` setting controls coordinate noise, not label-flipping
probability.

Before adding independent Gaussian noise with standard deviation `noise` to
each coordinate, the generator uses these shapes:

- Moons: class 0 is $(\cos\theta,\sin\theta)$ and class 1 is
  $(1-\cos\theta,0.5-\sin\theta)$, with independently sampled
  $\theta\sim Uniform[0,\pi]$.
- Circles: class 0 has radius 1 and class 1 radius 0.5, with angles sampled
  uniformly around the full circle. A point on a radius-$r$ circle is
  $(r\cos\theta,r\sin\theta)$.

The two classes have equal counts when `samples` is even; for an odd count,
class 1 receives the extra row. This deliberate construction is not an estimate
of real-world class frequencies.

## 2. Give each split one job

The experiment assigns approximately 60% of rows to training, 20% to validation,
and 20% to testing. It makes the assignment separately within each class,
preserving class representation; integer rounding applies when a class count
does not divide exactly. The row identifiers are disjoint and together cover
the generated dataset.

Within each class containing $n$ rows, the counts are
$\lfloor0.6n\rfloor$, $\lfloor0.2n\rfloor$, and the remaining rows. The
program then shuffles the rows within each partition.

| Split | What it is allowed to influence |
| --- | --- |
| Training | Feature scaling, parameter gradients, and optimizer updates |
| Validation | Which epoch is retained and when training stops; your later hyperparameter choices |
| Test | The final reported comparison after model selection is complete |

For the default balanced dataset of 600 rows, these targets correspond to 360
training rows, 120 validation rows, and 120 test rows.

The model sees validation losses while training, so validation performance is
not an untouched evaluation. The test partition does not influence scaling,
gradients, early stopping, or checkpoint selection. After restoring the selected
parameters, the engine computes the final test metrics.

There is a second source of leakage outside the program: **your own choices**.
If you try many settings and choose the one with the best displayed test score,
the test set has become part of model selection. Use validation results to make
choices. Once you start comparing test scores repeatedly, describe the exercise
as exploration and reserve new data for any later final evaluation.

## 3. Standardize using training data only

For each input coordinate $j$, calculate the training mean and population
standard deviation:

$$
\mu_j=\frac{1}{N_{train}}\sum_{i\in train}X_{ij},\qquad
s_j=\sqrt{\frac{1}{N_{train}}\sum_{i\in train}(X_{ij}-\mu_j)^2}.
$$

Transform every split using the **same** numbers:

$$
\widetilde X_{ij}=\frac{X_{ij}-\mu_j}{s_j}.
$$

A coordinate with zero spread needs a scale of 1 rather than a division by zero.
The implementation applies that rule to constant columns. This transformation
changes units, not labels. It often makes optimization easier because one coordinate
does not dominate solely through a larger numerical scale.

For example, training values $[0,2,4]$ have mean 2 and population standard
deviation $\sqrt{8/3}\approx1.632993$. A new value 6 becomes
$(6-2)/\sqrt{8/3}\approx2.449490$. We do not recalculate the mean using 6: that
would change the transformation based on the new example.

The saved experiment must preserve both the scaler and the network. Weights
trained on standardized coordinates cannot be applied directly to raw
coordinates. The report's point inspector accepts raw coordinates and applies
the saved transformation before calculating the layers.

## 4. Mini-batches change what an epoch means

In the XOR command, one epoch means one update using all four truth-table rows.
Here an epoch is a complete pass over the **training** rows, shuffled into
mini-batches. A batch of 32 is the default. The last batch can be shorter.

For a batch $B$, the data loss is

$$
L_B(\theta)=\frac{1}{|B|}\sum_{i\in B}\ell(y_i,p_i(\theta)).
$$

Backpropagation still starts with $(P-Y)/|B|$. The denominator is the actual
batch size, including the last shorter batch. Mean loss keeps the basic
gradient scale from simply growing in proportion to the number of rows.

Different batches generally give different gradient directions. Mini-batch
updates are inexpensive and can make progress before a whole dataset has been
processed, but their noise means that the full training loss need not decrease
at every epoch. Even full-batch loss need not decrease with an excessively large
learning rate.

The report plots full-split losses measured at epoch boundaries. These are not
the instantaneous losses of individual shuffled batches.

Random generators are kept separate: dataset generation uses `seed`, splitting
uses `seed + 1`, network initialization uses `seed + 2`, network batch ordering
uses `seed + 3`, and baseline batch ordering uses `seed + 4`. This prevents one
stage's extra random draw from silently changing all subsequent stages. Changing
the top-level seed intentionally changes the complete experiment.

## 5. L2 regularization: prefer smaller weights

The advanced experiment optimizes

$$
J_B(\theta)=L_B(\theta)+\frac{\lambda}{2}
\left(\lVert W_1\rVert_F^2+\lVert W_2\rVert_F^2\right).
$$

$\lVert W\rVert_F^2$ is the sum of the squares of every entry of $W$. The
coefficient `l2` is $\lambda$; setting it to zero removes the penalty. The
factor $1/2$ makes the derivative particularly simple:

$$
\nabla_{W_k}J_B=\nabla_{W_k}L_B+\lambda W_k.
$$

Biases are not penalized, so their derivatives stay unchanged. There is no extra
division of the regularization gradient by the batch size: the objective above
already specifies its scale.

For a weight $w=2$, a data gradient of $-0.3$, and $\lambda=0.1$, the complete
gradient is $-0.3+0.1(2)=-0.1$. An SGD step of size 0.05 moves the weight to
$2-0.05(-0.1)=2.005$. Regularization pushes toward zero, but the combined update
can still increase a weight when the data gradient pushes strongly enough.

With ordinary SGD, the weight update can be written

$$
W_{new}=(1-\eta\lambda)W-\eta\nabla_W L_B.
$$

With Adam, adding $\lambda W$ to the gradient is **coupled L2 regularization**.
It enters Adam's running averages. It is not the same algorithm as decoupled
weight decay, commonly called AdamW.

The displayed training and validation losses use data cross-entropy alone, so
their units are comparable across splits. The penalty affects optimization but
is not added to validation or test metrics. Increasing `l2` is not guaranteed
to improve validation performance; too much can prevent the model from fitting
useful structure.

The recorded `gradient_norm` is the Euclidean norm of all coordinates of the
full-training regularized gradient at that epoch's weights. If the gradient
arrays are flattened and concatenated into $g$, it is $\sqrt{\sum_jg_j^2}$.
It is not the most recent mini-batch gradient, an Adam step length, or a test-set
diagnostic. A small norm can indicate a stationary region, but does not prove
that the current parameters are globally optimal.

## 6. SGD and Adam, one update at a time

**SGD** uses the current gradient directly:

$$
\theta_t=\theta_{t-1}-\eta g_t.
$$

Here $g_t$ includes the weights-only L2 term. **Adam** keeps a separate pair of
running averages for every parameter coordinate. With initial $m_0=v_0=0$:

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\widehat m_t&=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\\
\theta_t&=\theta_{t-1}-\eta
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\end{aligned}
$$

All products, squares, square roots, and divisions in these optimizer equations
are coordinatewise. The time counter $t$ counts **updates**, not epochs. One
epoch can contain many Adam updates.

The first moment smooths gradient directions; the second moment tracks gradient
magnitudes. The bias corrections compensate for starting the averages at zero.
For example, with a constant gradient $g$, the recurrence gives
$m_t=(1-\beta_1^t)g$, so dividing restores $g$.

The [original Adam paper](https://arxiv.org/abs/1412.6980) introduces these
adaptive moment estimates. This lab uses $\beta_1=0.9$, $\beta_2=0.999$, and
$\epsilon=10^{-8}$. Adam is not a guarantee of convergence or a substitute for
checking the learning rate and validation curve.

### A two-step numerical example

Take one parameter $\theta_0=1$, learning rate 0.1, and the illustrative
gradient sequence $g_1=2$, $g_2=-1$. These two supplied gradients are an optimizer
arithmetic example, not derivatives claimed for a particular network run.

| Step | $m_t$ | $v_t$ | $\widehat m_t$ | $\widehat v_t$ | Parameter after update |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2 | 0.004 | 2 | 4 | 0.9000000005 |
| 2 | 0.08 | 0.004996 | 0.421052632 | 2.499249625 | 0.873366297 |

Although the second current gradient is negative, the parameter still
decreases. The retained first-moment estimate is positive. This explains why
an Adam update need not point opposite to the current gradient alone.

Reproduce the numbers without any ML library:

```python
from math import sqrt

theta, m, v = 1.0, 0.0, 0.0
for t, gradient in enumerate([2.0, -1.0], start=1):
    m = 0.9 * m + 0.1 * gradient
    v = 0.999 * v + 0.001 * gradient**2
    m_hat = m / (1 - 0.9**t)
    v_hat = v / (1 - 0.999**t)
    theta -= 0.1 * m_hat / (sqrt(v_hat) + 1e-8)
    print(t, theta)
```

## 7. Stop using validation, then restore a checkpoint

Training longer is not automatically better. The experiment evaluates
validation cross-entropy after each epoch and remembers parameters from the
lowest validation loss, including the initial model at epoch 0. When training
ends, it restores those parameters before evaluating the test split.

`patience` limits how many consecutive epochs can pass without a sufficiently
large validation improvement. `min_delta` defines what counts as a sufficiently
large improvement for that stopping decision. The best saved parameters are
selected by the actual lowest validation loss; the stopping threshold prevents
tiny fluctuations from continually postponing the stop.

More precisely, the code keeps a separate reference for the last significant
improvement. A new validation loss resets the patience counter only if it is
strictly less than that reference minus `min_delta`. The reference then moves
to the new loss. Small improvements can therefore accumulate until they cross
the threshold. Saving the actual best checkpoint is a separate comparison and
does not require crossing `min_delta`.

Consequently, three quantities can differ:

- `best_epoch`: the epoch whose parameters are returned;
- `stopped_epoch`: the last epoch actually executed;
- `epochs`: the maximum requested training budget.

Read the saved settings and stop reason alongside the curves. A replay frame
from a later epoch is not necessarily the model used for the final metrics.
The report keeps checkpoint replay distinct from the selected final model.

## 8. Compare against a simpler model

The baseline is logistic regression with the same two standardized features:

$$
p=\sigma(\widetilde x^Tw+b).
$$

It uses the same training, validation, and test partitions and the scaler fitted
on training data. Its weights have an L2 penalty; its bias does not. Its epoch is
also selected using validation loss. The final comparison uses the same test
rows for both models.

The baseline starts with zero weights and bias, and uses the same optimizer,
learning rate, batch size, L2 coefficient, maximum epoch budget, patience, and
minimum-improvement threshold. Unlike identical hidden neurons, logistic
regression has no hidden-unit symmetry that zero initialization needs to break.

This controls the data differences, but one shared hyperparameter configuration
does not exhaust the best possible tuning of either model. The baseline is a
useful reference for this experiment, not a universal ranking of model families.
If a more complicated model fails to beat it, inspect the result instead of
hiding the baseline.

## 9. Read the metrics rather than one headline number

The lab predicts class 1 when $p\geq0.5$. Its confusion matrix is arranged as:

| Actual / predicted | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| Actual 0 | True negative (TN) | False positive (FP) |
| Actual 1 | False negative (FN) | True positive (TP) |

From these counts:

$$
\begin{aligned}
accuracy&=\frac{TP+TN}{N},\\
precision&=\frac{TP}{TP+FP},\qquad
recall=\frac{TP}{TP+FN},\\
F_1&=\frac{2TP}{2TP+FP+FN}.
\end{aligned}
$$

Precision asks how often a positive prediction was right. Recall asks how many
actual positives were found. When a denominator is zero, the code returns 0
for that metric. That convention keeps the result finite; it should not be
mistaken for evidence that a missing class was meaningfully evaluated.

Cross-entropy and **Brier score** also use the probability values:

$$
Brier=\frac1N\sum_{i=1}^{N}(p_i-y_i)^2.
$$

For binary targets, the Brier score lies between 0 and 1, and lower is better.
A model that always outputs 0.5 has Brier score 0.25. Two classifiers can have
identical accuracy and different Brier scores because their probability values
differ. Neither a single Brier score nor the sigmoid function alone establishes
that probabilities are calibrated. Calibration asks whether events predicted
near a given probability actually occur at about that frequency, and requires
adequate independent evaluation data.

## 10. Use the visual report as an investigation

Read the training and validation curves together. If training loss continues
falling while validation loss rises, the model may be fitting details that do
not transfer to validation data. A small gap alone does not prove a good model:
both losses may be high because the model has learned little.

Replay shows recorded checkpoints of one run. It is not live retraining, and
the report need not store every epoch's full parameter arrays. Use the displayed
epoch rather than assuming equally spaced frames imply equally spaced changes
in model quality.

The coordinate inspector calculates what the displayed model (the current
replay checkpoint, or the linear baseline) outputs at your chosen raw point;
press "Selected model" to inspect the validation-selected network. It does
not create a new labeled test example. Especially far from the generated data,
a confident-looking sigmoid value can be an unreliable extrapolation. Use the
plotted held-out points to connect the function to actual evaluation examples.

Your next step is the [advanced exercises](advanced-exercises.md). For each run,
write a prediction, change one setting, record the result, and explain what the
evidence supports.

## Further reading

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980),
  Diederik P. Kingma and Jimmy Ba, gives the optimizer and its motivation.
- [Stanford CS231n: learning neural networks](https://cs231n.github.io/neural-networks-3/)
  discusses gradient checks, training diagnostics, and parameter updates.
- [The first mathematics walkthrough](mathematics.md) derives this lab's forward
  and backward passes. The advanced experiment reuses those derivatives.
