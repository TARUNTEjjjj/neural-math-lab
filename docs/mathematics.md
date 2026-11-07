# The mathematics of a tiny neural network

This lab learns a function from four examples. The purpose is to understand each
calculation well enough to explain it, check it, and change it yourself. You need
matrix multiplication, derivatives, and the chain rule. The Python syntax is
explained alongside the mathematics.

Read this guide with [`core.py`](../neural_math_lab/core.py) open. Work through
the [exercises](exercises.md) after the numerical example.

## 1. The problem: XOR

XOR means “exactly one input is 1.” Its complete Boolean truth table is:

| First input | Second input | Target output |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

We supply these four input rows and four targets. The network must learn
parameters that give the right outputs. It is not given the symbolic XOR rule.

A single affine score cannot separate these labels with a straight line. To see
why, suppose a score is $s(x_1,x_2)=w_1x_1+w_2x_2+b$, and we classify a score
greater than or equal to zero as 1. Correct classification would require:

$$
b<0,\quad w_1+b\geq0,\quad w_2+b\geq0,\quad w_1+w_2+b<0.
$$

Adding the middle two inequalities gives $w_1+w_2+2b\geq0$. But
$w_1+w_2+b=(w_1+w_2+2b)-b>0$, contradicting the last inequality.
We therefore introduce a nonlinear hidden layer.

## 2. What the words mean

- A **weight** multiplies an incoming value. Learning changes that multiplier.
- A **bias** adds an offset, so a neuron does not have to pass through the origin.
- A **neuron** computes a weighted sum plus a bias, then applies a function.
- A **layer** computes several neurons together using a matrix operation.
- An **activation** is that function. Here the hidden layer uses `tanh`.
- A **parameter** is a value we train: every weight and every bias.
- A **hyperparameter** is a choice we make before training, such as the hidden
  width or learning rate.
- A **loss** is one number measuring how badly the predictions match the targets.
- An **epoch** here is one update using all four examples. Other programs may
  process several smaller batches during an epoch.

Our architecture is **2 inputs → 4 hidden neurons → 1 output** by default. You
can change the hidden width. The examples below call this width $m$ so that it
does not get confused with the hidden activations.

## 3. Organize the numbers into arrays

Let $N$ be the number of examples in a batch. An array's **shape** gives its
number of rows and columns. For the full XOR table, $N=4$.

| Mathematical name | Python name | Shape | Meaning |
| --- | --- | --- | --- |
| $X$ | `X` | $(N,2)$ | Input rows |
| $Y$ | `y` | $(N,1)$ | Target labels |
| $W_1$ | `W1` | $(2,m)$ | Input-to-hidden weights |
| $b_1$ | `b1` | $(1,m)$ | Hidden biases |
| $Z_1$ | `z1` | $(N,m)$ | Hidden values before activation |
| $A$ | `h` | $(N,m)$ | Hidden values after `tanh` |
| $W_2$ | `W2` | $(m,1)$ | Hidden-to-output weights |
| $b_2$ | `b2` | $(1,1)$ | Output bias |
| $Z_2$ | `z2` | $(N,1)$ | Output scores, also called logits |
| $P$ | `p` | $(N,1)$ | Predicted probabilities for label 1 |

For $m=4$, there are $2m+m+m+1=17$ trainable numbers. The input rows and target
labels are data, not trainable parameters.

In NumPy, `@` means matrix multiplication; `*` means elementwise multiplication.
`W.T` transposes a matrix. Adding a $(1,m)$ bias to an $(N,m)$ matrix uses
**broadcasting**: NumPy adds the same bias row to every example. It does not
create a different bias for each example.

## 4. Forward propagation: calculate a prediction

The complete forward pass is:

$$
Z_1=XW_1+b_1,\qquad A=\tanh(Z_1),\qquad Z_2=AW_2+b_2,\qquad P=\sigma(Z_2),
$$

where

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

These activation functions act on each element independently. `tanh` maps real
numbers into $(-1,1)$; sigmoid maps them into $(0,1)$ mathematically. Floating
point arithmetic can round an extreme sigmoid result to exactly 0 or 1.

The hidden neurons transform the original two coordinates into nonlinear
features. The output combines those features. Removing `tanh` would make the
two matrix multiplications collapse into one affine function, bringing back
the XOR limitation above.

The prediction rule is label 1 when $p\geq0.5$, and label 0 otherwise. A value
such as $p=0.9$ is the model's probability output. It is not evidence that the
model is correct on 90% of comparable real-world inputs; that would require a
separate calibration evaluation.

`Network.forward(X)` returns the intermediate arrays as well as `p`, allowing
you to inspect each stage instead of treating prediction as a black box.

## 5. Loss: score a prediction smoothly

For a target $y\in\{0,1\}$ and probability $p$, binary cross-entropy is

$$
\ell(y,p)=-y\log(p)-(1-y)\log(1-p).
$$

If the target is 1, this is $-\log p$, so raising $p$ reduces the loss. If the
target is 0, it is $-\log(1-p)$, so lowering $p$ reduces the loss. A prediction
of 0.5 has loss $\log2\approx0.693147$ for either label. The loss increases
without bound for increasingly confident wrong predictions.

The batch loss is the mean:

$$
L=\frac1N\sum_{i=1}^{N}\ell(y_i,p_i).
$$

Accuracy only records whether a prediction crossed the threshold. Loss also
records movement within either side of that threshold, giving training useful
derivatives.

### Why the code uses logits

Directly taking logs of rounded probabilities can produce `log(0)`. Substituting
$p=\sigma(z)$ gives the equivalent expression

$$
\ell(y,z)=\log(1+e^z)-yz.
$$

For binary labels, the implementation uses an equivalent form that also avoids
subtracting two nearly equal large numbers:

```python
signed_logit = np.where(y == 1, -z2, z2)
losses = np.logaddexp(0.0, signed_logit)
loss = np.sum(losses / len(y))
```

[`np.logaddexp(a, b)`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
computes the log of the sum of the exponentials. Using it avoids explicitly
forming a potentially enormous `exp(z2)`. A positive target uses
$\log(1+e^{-z})$, while a negative target uses $\log(1+e^z)$.
At $z=40,y=1$, the true loss is about $4.248\times10^{-18}$:
computing `logaddexp(0, 40) - 40` would round it to zero. Dividing each loss
before adding also avoids overflowing the mean of several large finite losses.
Sigmoid is also evaluated with
separate positive/negative branches to avoid overflowing an exponential.
These choices preserve the mathematical model while improving numerical
behavior; they do not make arbitrarily large intermediate matrix products safe.

The combined loss derivative is still $\sigma(z)-y$. For a positive target,
the code evaluates it as $-\sigma(-z)$, so a displayed probability rounded to
1 does not erase a representable tiny derivative. Ordinary floating-point
underflow can still round sufficiently small quantities to zero.

## 6. Backpropagation: calculate how each parameter affects the loss

Backpropagation is the chain rule applied in reverse order, reusing values from
the forward pass. A gradient has the same shape as its parameter. For example,
`dW1[i, j]` tells us how the loss changes to first order when `W1[i, j]` changes.

Begin with one example. Combining cross-entropy with sigmoid gives a useful
simplification:

$$
\frac{\partial\ell}{\partial p}
=-\frac{y}{p}+\frac{1-y}{1-p},\qquad
\frac{\partial p}{\partial z}=p(1-p),
$$

so

$$
\frac{\partial\ell}{\partial z}
=-y(1-p)+(1-y)p=p-y.
$$

This derivation assumes interior probabilities. The implementation obtains the
same gradient directly as `p - y`, avoiding divisions by probabilities that may
round to zero.

For the mean loss on $N$ rows, put $D_2=\partial L/\partial Z_2=(P-Y)/N$.
Then reverse each operation:

$$
\begin{aligned}
\frac{\partial L}{\partial W_2}&=A^T D_2,\\
\frac{\partial L}{\partial b_2}&=\sum_{i=1}^{N}(D_2)_{i,:},\\
\frac{\partial L}{\partial A}&=D_2W_2^T,\\
D_1=\frac{\partial L}{\partial Z_1}
&=(D_2W_2^T)\odot(1-A\odot A),\\
\frac{\partial L}{\partial W_1}&=X^TD_1,\\
\frac{\partial L}{\partial b_1}&=\sum_{i=1}^{N}(D_1)_{i,:}.
\end{aligned}
$$

Here $\odot$ denotes elementwise multiplication. The factor $1-A^2$ comes from
$d\tanh z/dz=1-\tanh^2z$. Each bias is shared by every row, so its gradient
sums the contributions from all rows.

The corresponding NumPy operations are:

```python
dz2 = (p - y) / len(X)
dW2 = h.T @ dz2
db2 = dz2.sum(axis=0, keepdims=True)
dz1 = (dz2 @ W2.T) * (1.0 - h * h)
dW1 = X.T @ dz1
db1 = dz1.sum(axis=0, keepdims=True)
```

`axis=0` sums down the examples. `keepdims=True` retains the leading dimension
of size 1, preserving the bias shape. Divide by $N$ **once** at `dz2`; the later
sums already carry that factor. Dividing again would shrink the gradient
incorrectly.

All gradients refer to the same current parameter values. Calculate the entire
gradient before updating any parameter: changing `W2` early would make the
hidden-layer calculation use the wrong weights.

## 7. Gradient descent: turn derivatives into learning

For every parameter array $\theta$, apply

$$
\theta_{\text{new}}=\theta-\eta\nabla_\theta L,
$$

where $\eta>0$ is the learning rate. A positive gradient says that increasing
the parameter increases the loss locally, so we subtract it.

For a sufficiently small step on a smooth loss,
$L(\theta-\eta g)\approx L(\theta)-\eta\lVert g\rVert^2$ when $g$ is the true
gradient. This local approximation explains the update direction. It does not
guarantee improvement for an arbitrarily large learning rate. The network's
optimization problem is nonconvex; settings and initialization can affect the
result.

The initialization seed selects reproducible starting weights. A seed is not a
quality score. Biases can start at zero, but independently initialized weights
let the hidden neurons learn different features.

The code uses Glorot/Xavier uniform initialization. For a layer with $n_{in}$
inputs and $n_{out}$ outputs, each initial weight is sampled uniformly from

$$
\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\quad
       \sqrt{\frac{6}{n_{in}+n_{out}}}\right].
$$

The scale accounts for the number of incoming and outgoing connections, helping
avoid starting with activations that are excessively large or small. It is an
initialization heuristic, not a guarantee of convergence. The same seed and
settings support repeatable comparisons in the same numerical environment;
small floating point differences can occur across platforms or library versions.

## 8. A numerical example you can check by hand

Use two hidden neurons for this example, one input row $X=[1,0]$, target $Y=[1]$,
and these deliberately chosen parameters:

$$
W_1=\begin{bmatrix}0.5&-0.5\\1&1\end{bmatrix},\quad
b_1=\begin{bmatrix}0&0\end{bmatrix},\quad
W_2=\begin{bmatrix}1\\-1\end{bmatrix},\quad b_2=0.
$$

Forward propagation gives:

| Quantity | Value, rounded |
| --- | --- |
| $Z_1$ | $[0.5,-0.5]$ |
| $A=\tanh Z_1$ | $[0.462117,-0.462117]$ |
| $Z_2$ | $0.924234$ |
| $P=\sigma Z_2$ | $0.715904$ |
| $L=-\log P$ | $0.334209$ |

Since this batch has one row, $D_2=P-Y=-0.284096$. The output gradients are

$$
dW_2\approx\begin{bmatrix}-0.131286\\0.131286\end{bmatrix},
\qquad db_2\approx-0.284096.
$$

Both hidden derivatives have factor $1-0.462117^2\approx0.786448$, giving

$$
D_1\approx[-0.223427,0.223427],\quad
dW_1\approx\begin{bmatrix}-0.223427&0.223427\\0&0\end{bmatrix},\quad
db_1\approx[-0.223427,0.223427].
$$

The second row of `dW1` is zero because the second input is zero in this example.
That does not prevent those weights from learning from other examples.

One simultaneous update with $\eta=0.1$ gives

$$
W_1'\approx\begin{bmatrix}0.522343&-0.522343\\1&1\end{bmatrix},\quad
b_1'\approx[0.022343,-0.022343],
$$

$$
W_2'\approx\begin{bmatrix}1.013129\\-1.013129\end{bmatrix},\quad
b_2'\approx0.028410.
$$

Recomputing the forward pass gives $P'\approx0.737787$ and
$L'\approx0.304100$. The probability moved toward this example's target, and
its loss decreased. This is a one-example calculation, not the result of
training on the full XOR table.

Reproduce the calculation using the library:

```python
import numpy as np
from neural_math_lab.core import Network

net = Network(hidden=2, seed=7)
net.parameters["W1"][:] = [[0.5, -0.5], [1.0, 1.0]]
net.parameters["b1"][:] = [[0.0, 0.0]]
net.parameters["W2"][:] = [[1.0], [-1.0]]
net.parameters["b2"][:] = [[0.0]]
X = np.array([[1.0, 0.0]])
y = np.array([[1.0]])

print(net.forward(X))
print("Before:", net.loss(X, y))
grads = net.gradients(X, y)
print(grads)
net.step(grads, learning_rate=0.1)
print("After:", net.loss(X, y))
```

## 9. Check the derivative without trusting the derivation

For each individual parameter $\theta_j$, estimate its derivative numerically:

$$
g_j^{\text{numeric}}\approx
\frac{L(\theta+\epsilon e_j)-L(\theta-\epsilon e_j)}{2\epsilon}.
$$

Only that coordinate changes; all other parameters stay fixed. Compare this
central difference with the analytical backpropagation gradient. A good
agreement across every parameter is stronger evidence than merely seeing a
decreasing training loss: a partly incorrect derivative might still reduce loss.

Run:

```sh
python -m neural_math_lab gradient-check
```

The perturbation should be small, but not arbitrarily tiny. Central differences
have approximation error, while subtracting nearly equal floating point losses
has roundoff error. The default $\epsilon=10^{-5}$ is a practical choice for
this small float64 model. Check absolute error too; relative error can be
misleading when both derivatives are near zero. This is a numerical test at
particular points, not a proof of correctness for every possible input.

## 10. Train, inspect, and interpret honestly

```sh
python -m neural_math_lab train --epochs 4000 --learning-rate 0.5 --hidden 4 --seed 7 --output runs/xor
python -m neural_math_lab explain --model runs/xor/model.json --input 1 0
```

Read the training loss curve, then inspect the four truth-table predictions.
The `explain` command shows the saved model's actual intermediate numbers for
one input. Compare its values to the equations above; they differ from the hand
example because training learns different weights.

Every XOR row is used to train this first lab. Consequently:

- **100% XOR training accuracy means all four training rows were classified
  correctly.** It is not held-out test accuracy.
- There is no unseen Boolean XOR input left after fitting all four rows.
- The colored prediction map between Boolean points shows the network's smooth
  extension of the function. Those locations have no supplied ground-truth
  labels in this experiment.
- A lower loss at the same training accuracy can mean predictions became more
  extreme; it is not independent evidence of better generalization.

A later experiment with many distinct samples should use training, validation,
and untouched test data. For now, understanding the computation and verifying
the gradients is the main result.

## Further reading

- [Stanford CS231n: backpropagation and the chain rule](https://cs231n.github.io/optimization-2/)
  develops the computational-graph view used in this guide.
- [NumPy `logaddexp` reference](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
  documents the numerical operation used in our loss.
- [PyTorch automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  is a useful next step after calculating the derivatives yourself. This lab's
  gradients are written manually; it does not require PyTorch.
