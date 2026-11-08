# Watch a derivative become a learning step

The optimization lab connects three objects that are easy to confuse: a
prediction, a loss, and a derivative of that loss. Choose a parameter, check
its derivative numerically, and compare the loss you expect after an update
with the loss the network actually computes.

Open a newly generated moons or circles report beside this guide. Start with
[the network derivation](mathematics.md) and [training beyond XOR](advanced-mathematics.md)
if the array shapes are unfamiliar. Then try the
[optimization exercises](optimization-exercises.md) before reading their answers.

## 1. Separate the recorded experiment from the sandbox

The sandbox starts from a copy of a saved neural-network checkpoint. You can
inspect an early model or the validation-selected model, then change the copy
with explicit learning steps. These temporary changes do not rewrite the
recorded checkpoint, Python training history, baseline, or held-out results.
Resetting restores the chosen starting checkpoint.

Use **Undo** to inspect the preceding sandbox step or **Reset** to begin again.
A trial is limited to 50 applied steps. Changing the starting checkpoint,
selected batch, or L2 coefficient starts a new trial so its history describes
one fixed objective. These temporary trials are learning experiments, not
replacement saved models.

Calculations use a bounded selection from the recorded **training rows** and
the saved training-set standardizer. The same selected rows are used for both
the proposed update and its before/after loss. A small selection makes the
calculation responsive and inspectable; its loss is not the loss on every
training example. It may contain an unrepresentative mix of classes.

Each step uses the complete selected sandbox batch. Repeated clicks do not
recreate Python's shuffled mini-batches or Adam moment state. This is an
experiment with the gradient-descent update rule. A reduction in sandbox loss
does not establish better validation or test performance, and the original
held-out cards cannot evaluate the modified copy.

## 2. State the objective before differentiating it

Let $N$ be the number of selected rows, $x_i$ a raw input, and $y_i\in\{0,1\}$
its label. The standardizer remains fixed:

$$
\widetilde x_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
$$

For hidden width $H$, the forward pass is

$$
Z_1=\widetilde XW_1+b_1,\quad A=\tanh(Z_1),\quad
z=AW_2+b_2,\quad p=\sigma(z).
$$

Here $\widetilde X$ has shape $N\times2$, $W_1$ is $2\times H$, $b_1$ is
$1\times H$, $W_2$ is $H\times1$, and $b_2$ is $1\times1$. Biases broadcast
over the rows. The sandbox changes network parameters, not $\mu$ or $s$.

The objective separates data fit from a weight penalty:

$$
J(\theta)=\underbrace{\frac1N\sum_{i=1}^N
  \left[\log(1+e^{z_i})-y_i z_i\right]}_{\text{mean binary cross-entropy}}
  +\underbrace{\frac\lambda2
    \left(\lVert W_1\rVert_F^2+\lVert W_2\rVert_F^2\right)}_{\text{L2 penalty}}.
$$

The Frobenius norm squares and adds every entry of a matrix. Biases are not
penalized. In this convention the data term is averaged over the selected
rows, while the penalty is not divided by $N$. Duplicating every row therefore
leaves both the objective and its gradient unchanged.

Compute each cross-entropy from the logit using

$$
\max(z_i,0)-y_i z_i+\log(1+e^{-|z_i|}).
$$

This equivalent expression avoids forming $e^{1000}$ or taking $\log(0)$
after a sigmoid has rounded to 0 or 1. A confidently wrong prediction can
have a large finite loss even when its displayed probability is rounded.

## 3. Follow the gradient through the layers

The sigmoid and cross-entropy derivatives combine into

$$
D_2=\frac{p-y}{N}.
$$

That identity does not require subtracting rounded probabilities in code.
For $y=1$, compute $-\sigma(-z)/N$; for $y=0$, compute $\sigma(z)/N$.
For example, $z=40,y=1$ retains a derivative of about
$-4.248\times10^{-18}/N$ even though the displayed $p$ rounds to 1.
This avoids falsely reporting a zero gradient in the optimization sandbox.

The remaining derivatives are

$$
\begin{aligned}
\nabla_{W_2}J&=A^TD_2+\lambda W_2,
&\nabla_{b_2}J&=\sum_i(D_2)_i,\\
D_1&=(D_2W_2^T)\odot(1-A^2),\\
\nabla_{W_1}J&=\widetilde X^TD_1+\lambda W_1,
&\nabla_{b_1}J&=\sum_i(D_1)_i.
\end{aligned}
$$

The factor $1/N$ appears once, in $D_2$. Summing bias derivatives over rows
does not introduce a second averaging factor. The L2 derivative is
$\lambda W$, because differentiating $(\lambda/2)W^2$ cancels the 2.

A parameter derivative answers: “If this one weight or bias increases by a
small amount while every other parameter stays fixed, how does this batch's
objective change?” It is different from $\partial p/\partial x$ in the input
sensitivity inspector, which changes an input while keeping the model fixed.

## 4. Check a derivative using only loss evaluations

For selected parameter coordinate $k$, central differences give

$$
g_k^{num}(\varepsilon)=
\frac{J(\theta+\varepsilon e_k)-J(\theta-\varepsilon e_k)}{2\varepsilon}.
$$

Both evaluations use the same rows, labels, standardizer, and $\lambda$.
Otherwise the numerator would mix a parameter change with a different
experiment. The analytic derivative comes from backpropagation; the numerical
one comes from two independent forward-loss evaluations.

Try a sequence such as $10^{-2},10^{-3},10^{-4},10^{-5},10^{-6}$. In exact
arithmetic, a smooth function's central-difference truncation error is
$O(\varepsilon^2)$. In floating-point arithmetic, subtracting almost equal
losses loses significant digits, and division by a tiny $\varepsilon$
amplifies rounding error. An extremely small increment can even round away
when added to the parameter. Smaller epsilon is therefore not always better.

Inspect absolute error when both derivatives are near zero. Relative error
can look large when its denominator is tiny. Agreement at a few tested
coordinates is useful evidence, not a proof that every possible input and
parameter value is correct.

## 5. Read a loss slice and its tangent

The plotted slice changes just one coordinate:

$$
f(s)=J(\theta+s e_k),\qquad
f(s)\approx J(\theta)+s g_k.
$$

The tangent line uses the derivative at the center. The actual curve recomputes
the objective at each sampled offset. Their difference shows curvature along
this direction. Widen the displayed interval to find where the local linear
approximation stops describing the curve well.

A slice is not the full loss surface. Other parameter combinations can behave
differently, and sparse plotted samples do not prove that there are no features
between them. A curved or bowl-shaped slice does not establish global
convexity of the neural-network objective.

## 6. Predict an update, then measure it

For learning rate $\eta>0$, a full gradient-descent step updates every
parameter with its derivative from the **same pre-update model**:

$$
\theta'=\theta-\eta g,
\qquad \Delta J_{linear}=g^T(\theta'-\theta)
=-\eta\lVert g\rVert_2^2.
$$

The norm includes all entries of the weight and bias gradients. A
one-coordinate step changes only coordinate $k$:

$$
\theta'_k=\theta_k-\eta g_k,
\qquad \Delta J_{linear}=-\eta g_k^2.
$$

The measured change is $J(\theta')-J(\theta)$. Negative means the chosen batch's
objective decreased. The first-order prediction omits curvature, so its
nonpositive sign does not guarantee that a finite update improves the loss.
For example, gradient descent on $J(w)=w^2$ gives
$w'=(1-2\eta)w$. At $w=1$ and $\eta=2$, it predicts a change of $-8$ but
actually moves from loss 1 to loss 9: a change of $+8$.

After applying an update, gradients must be recomputed at the new parameters.
The old derivative describes the old model. Changing $\lambda$ or the selected
rows also changes the objective, so loss values across those settings are not
one continuous optimization history.

## 7. A complete one-neuron calculation

Take one raw input equal to the saved mean, so $\widetilde x=[0,0]$, with target
$y=1$. Use one hidden neuron:

$$
W_1=\begin{bmatrix}1\\-1\end{bmatrix},\quad b_1=0,
\quad W_2=2,\quad b_2=0,\quad\lambda=0.1.
$$

The hidden activation and logit are zero, so $p=1/2$. The data loss is
$\log2\approx0.6931471806$. The penalty is
$(0.1/2)(1^2+(-1)^2+2^2)=0.3$, giving $J\approx0.9931471806$.

Because $p-y=-1/2$, the derivatives are

| Parameter | Data derivative | L2 derivative | Total derivative |
| --- | ---: | ---: | ---: |
| $W_{1,1}$ | 0 | 0.1 | 0.1 |
| $W_{1,2}$ | 0 | −0.1 | −0.1 |
| $b_1$ | $(-1/2)\times2\times1=-1$ | 0 | −1 |
| $W_2$ | $0\times(-1/2)=0$ | 0.2 | 0.2 |
| $b_2$ | −0.5 | 0 | −0.5 |

Zero standardized input removes the data derivative of $W_1$ here; it does
not remove the hidden bias derivative or the weight penalty.

Update only $b_2$ with $\eta=0.2$. The new bias is
$0-0.2(-0.5)=0.1$. The penalty stays 0.3 and

$$
J'=\log(1+e^{-0.1})+0.3\approx0.9443966601.
$$

The actual change is approximately $-0.0487505205$, close to the predicted
$-0.2(0.5)^2=-0.05$. This is a calculation for the stated one-row fixture,
not a promised value for the report's moons or circles batch.

For an all-parameter step, the gradient norm squared is
$0.1^2+(-0.1)^2+(-1)^2+0.2^2+(-0.5)^2=1.31$.
At the same rate the predicted change is $-0.262$. Recomputing the forward
pass after changing all parameters gives $J'\approx0.7671798423$ and actual
change approximately $-0.2259673383$. The one-coordinate and all-parameter
steps answer different experiments; neither number should be substituted for
the other.
