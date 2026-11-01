# Exercises: predict, run, explain

Work through these after the [mathematics guide](mathematics.md). For each
experiment, write your prediction before running the code. Then record what
actually happened and explain any difference. Use different output directories
for training runs so you can compare their artifacts.

## Questions

### 1. Count the parameters

How many trainable values does a network with 2 inputs, 3 hidden neurons, and 1
output have? Give the shape of each parameter array. Are the four target labels
parameters?

### 2. Read a matrix multiplication

With

$$
X=[1,0],\quad W_1=\begin{bmatrix}0.5&-0.5\\1&1\end{bmatrix},\quad b_1=[0,0],
$$

calculate $Z_1$. Which row of $W_1$ has no effect on this particular input?
Would that row also have no effect on input $[0,1]$?

### 3. Predict an update's direction

For one example with target 1 and predicted probability 0.25, what is
$\partial L/\partial b_2$? If the learning rate is 0.1, does the output bias
increase or decrease, and by how much?

### 4. Average exactly once

Suppose you duplicate the complete four-row XOR batch, producing eight rows.
Without changing the parameters, what should happen to the mean loss and the
gradients? What bug would be exposed if the gradients doubled?

### 5. Compare learning rates

Keep the seed, width, and epoch count fixed. Compare learning rates 0.05, 0.5,
and 5.0:

```sh
python -m neural_math_lab train --seed 7 --epochs 4000 --learning-rate 0.05 --output runs/rate-005
python -m neural_math_lab train --seed 7 --epochs 4000 --learning-rate 0.5 --output runs/rate-05
python -m neural_math_lab train --seed 7 --epochs 4000 --learning-rate 5.0 --output runs/rate-5
```

Record each final loss and the number of correctly classified XOR rows. Look at
the curves, not just the last number. Does “ten times the learning rate” imply
“ten times faster learning”? Must the loss decrease at every epoch?

### 6. Remove the hidden nonlinearity on paper

Replace $A=\tanh(XW_1+b_1)$ by $A=XW_1+b_1$. Expand the output logit in terms of
$X$. Can increasing the hidden width now solve XOR perfectly with the usual
threshold rule? Explain using the proof in the guide.

### 7. Start every parameter at zero

Set all four parameter arrays to zero and consider a full XOR batch. Calculate
the hidden activations, output probabilities, and all gradients. What changes
after one update? How does a balanced label count matter here?

You can check your answer directly:

```python
from neural_math_lab.core import Network, xor_data

net = Network(hidden=4, seed=7)
for values in net.parameters.values():
    values.fill(0.0)
X, y = xor_data()
print(net.forward(X))
print(net.gradients(X, y))
```

### 8. Study a saturated hidden neuron

Evaluate $1-\tanh^2 z$ at $z=0$ and $z=10$. Explain what those numbers do to
the hidden-layer gradient. Does a saturated output sigmoid necessarily make
the combined sigmoid-plus-cross-entropy logit gradient small for a confidently
wrong prediction?

### 9. Check a gradient independently

Reproduce the hand example in the guide. Perturb `W1[0, 0]` by $+10^{-5}$ and
$-10^{-5}$, restoring it afterward. Estimate the central-difference gradient.
Compare it to `net.gradients(X, y)["W1"][0, 0]` before taking a training step.

### 10. Describe the result precisely

A run classifies all four XOR rows correctly. Write one accurate sentence
describing that result. Then explain why a prediction at $[0.4,0.8]$ does not
constitute another correctly classified test example.

---

## Answers and checks

### 1. Parameter count

`W1` has shape `(2, 3)` and 6 values; `b1` has `(1, 3)` and 3; `W2` has
`(3, 1)` and 3; `b2` has `(1, 1)` and 1. Total: **13**. Targets are fixed data.
For hidden width $m$, the count is $4m+1$.

### 2. Matrix multiplication

$Z_1=[0.5,-0.5]$. The second row of $W_1$ is multiplied by the zero second input.
For $[0,1]$, the situation reverses and $Z_1=[1,1]$.

### 3. Update direction

For a one-row batch, $db_2=p-y=0.25-1=-0.75$. Gradient descent subtracts
$0.1\times(-0.75)$, so the bias **increases by 0.075**. Holding everything else
fixed, a larger output logit raises the probability of label 1.

### 4. Duplicated batch

The mean loss and all gradients are unchanged, up to floating point rounding.
Both the sum of contributions and the number of rows double. Doubled gradients
would suggest a sum loss or a missing division by the batch size. This check is
particularly useful because the implementation is meant to use a mean loss.

### 5. Learning-rate experiment

There is no fixed numeric answer supplied for this experiment: report your
actual run, including settings. A larger step follows only a local linear
approximation and may overshoot, oscillate, or enter a different region of the
loss surface. A small step can make slower progress. Large rates are not
guaranteed to fail, and a loss curve is not guaranteed to be monotonic.

### 6. No hidden nonlinearity

$$
Z_2=(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2).
$$

This is still one affine map of the two inputs. Sigmoid is monotone and its
0.5 threshold corresponds to a zero logit. Therefore the classification
boundary remains a straight line, regardless of hidden width, and cannot
perfectly classify XOR.

### 7. All-zero initialization

$Z_1=A=Z_2=0$ and every probability equals 0.5. The XOR residuals are
$[0.5,-0.5,-0.5,0.5]^T$. Their mean is zero, so `db2` is zero. Because $A=0$,
`dW2` is zero. Because $W_2=0$, the gradients reaching the hidden layer are also
zero. Nothing changes; repeating the update stays at this stationary point.

For an unbalanced batch, `db2` could move to reflect the class proportion, but
this all-zero construction would still not distinguish the input patterns.
More generally, identical hidden units with identical outgoing weights receive
identical gradients and remain identical. Random weight initialization breaks
that symmetry. Zero biases alone do not cause this problem.

### 8. Saturation

At zero, $1-\tanh^2(0)=1$. At 10, it is approximately $8.24\times10^{-9}$.
Multiplication by the latter transmits a very small gradient through that hidden
activation, which can slow learning.

For the combined output loss, $d\ell/dz=p-y$. If $y=1$ and $p$ is near zero,
this derivative is near $-1$, not near zero. The derivative of sigmoid alone
does not tell you the derivative of the full composed loss.

### 9. Gradient check

The derivative is approximately **-0.2234265841** in the guide's hand example.
Both the numerical and analytical calculations should be close to this value.
If you get a different result, first check that you used a one-row batch, the
specified weights, and parameters from before the update. Restore the perturbed
parameter even when an assertion fails.

### 10. Honest reporting

For example: “With the recorded settings, this network correctly classifies all
four XOR training examples.” All four rows participated in optimization, so
this is not accuracy on held-out data. The Boolean XOR truth table defines no
label for $[0.4,0.8]$ in this lab; the plotted prediction there illustrates the
network's extension between the training points.

## A short experiment log

Copy this table into your notes and fill it with actual runs:

| Seed | Hidden width | Learning rate | Epochs | Final loss | Correct XOR rows | What I learned |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | /4 | |

Before calling an experiment finished, explain in your own words what
`(p - y) / len(X)` means, why `W2.T` appears in the backward pass, and why
training accuracy alone does not establish generalization.
