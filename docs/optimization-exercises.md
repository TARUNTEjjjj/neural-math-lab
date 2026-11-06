# Optimization lab: predict, check, explain

Use [the optimization mathematics guide](optimization-mathematics.md) as a
reference. For browser experiments, record the starting checkpoint, selected
training rows, L2 coefficient, parameter coordinate, epsilon, and learning
rate. Write a prediction before pressing a control. There is no required
improvement on every model or step size.

## Questions

### 1. Keep the inputs and the parameters separate

A saved standardizer has $\mu=[2,-3]$ and $s=[4,2]$. For raw input $[6,1]$,
calculate the standardized coordinates. What stays fixed when the lab checks
the loss derivative for `W1[0,0]` (zero-based row and column)? How does that differ from checking
$\partial p/\partial x_1$ in the input-sensitivity inspector?

### 2. Predict the complete gradient

Use the guide's one-neuron fixture: standardized input $[0,0]$, target 1,
$W_1=[1,-1]^T$, $b_1=0$, $W_2=2$, $b_2=0$, and $\lambda=0.1$.
Calculate the probability, mean BCE, penalty, objective, and all five scalar
derivatives. Explain why the first-layer weights have nonzero total gradients
even though both standardized input coordinates are zero.

### 3. Compare a coordinate step with a full step

Starting from that fixture, use $\eta=0.2$. First update only $b_2$. Then reset
and update all parameters simultaneously. Calculate each predicted
first-order loss change and each actual loss change. Why must the gradients
for the full step all come from the original parameter values?

### 4. Find a useful finite-difference interval

For the same fixture, check the $b_2$ derivative using central differences at
$\varepsilon=10^{-2},10^{-4},10^{-6}$. Explain a special symmetry of this
example that makes its central difference unusually accurate in exact
arithmetic. Then check a nonzero parameter in a report and vary epsilon over
several orders of magnitude. Why can the smaller epsilon produce a larger
disagreement?

### 5. Change the amount of data without changing the mathematics

Suppose a two-row batch is duplicated to make four rows, preserving the same
input–label pairs and $\lambda$. What happens to mean BCE, the L2 penalty,
the total gradient, and a proposed full step? What changes if you use a sum
of row losses instead of a mean while keeping the same learning rate and
penalty coefficient?

### 6. Discover why a descent direction can still overshoot

For $J(w)=w^2$ at $w=1$, compute the proposed new weight, predicted loss
change, and actual loss change for $\eta=0.1,0.5,1,2$. Determine exactly
which positive learning rates reduce this objective when $w\ne0$.

In the report, compare the actual and tangent curves around one selected
weight. Does a bowl-shaped slice prove that the network has one global
minimum?

### 7. Interpret L2 without changing the reported objective silently

At fixed parameters, change $\lambda$ in the one-neuron fixture from 0.1
to 0.2. Find the new objective and gradients. Has the model's prediction
changed? If a learner compares the loss before changing $\lambda$ with the
loss after one update using the new $\lambda$, what has that comparison
mixed together?

### 8. Decide what evidence an improved sandbox loss provides

A learner starts from the validation-selected checkpoint, chooses the first
16 training rows, and applies ten steps. The sandbox objective falls. Can
they claim that the saved test accuracy improved, that their new parameters
are better on all training rows, or that the original Adam experiment has
been reproduced? Describe a defensible next experiment and how to preserve
the role of held-out data.

---

## Worked answers

### 1. Fixed data, changing weights

The standardized input is $[(6-2)/4,(1-(-3))/2]=[1,2]$.
For a parameter derivative, the raw rows, labels, standardizer, $\lambda$,
and all other parameters stay fixed. Only the selected weight is perturbed.
For input sensitivity, the weights stay fixed and a raw input coordinate
changes; the chain rule includes division by its saved scale. These are
derivatives of different functions with respect to different variables.

### 2. The five derivatives

The activation and logit are zero, so $p=0.5$. Mean BCE is $\log2$; penalty
is $0.3$; objective is approximately $0.9931471806$.
The output error is $p-y=-0.5$, and the hidden error is
$(-0.5)(2)(1-0^2)=-1$.

The full gradients, in the order $W_{1,1},W_{1,2},b_1,W_2,b_2$, are
$[0.1,-0.1,-1,0.2,-0.5]$. Both first-layer data derivatives are zero, but
L2 contributes $\lambda W_1$. Biases receive no L2 term.

### 3. Different directions, different changes

The coordinate step gives $b_2'=0.1$, with every other parameter unchanged.
Its predicted change is $-0.2(-0.5)^2=-0.05$. Its actual new objective is
$\log(1+e^{-0.1})+0.3\approx0.9443966601$, so the change is approximately
$-0.0487505205$.

The full step gives $W_1'=[0.98,-0.98]^T$, $b_1'=0.2$, $W_2'=1.96$, and
$b_2'=0.1$. Its predicted change is $-0.2(1.31)=-0.262$.
The new logit is $1.96\tanh(0.2)+0.1\approx0.4868556276$.
The new penalty is $0.05(0.98^2+(-0.98)^2+1.96^2)=0.28812$.
Together these give $J'\approx0.7671798423$ and change approximately
$-0.2259673383$.

Using partially updated weights to calculate later derivatives would perform
a different, sequential algorithm. The stated update is one vector operation
using the gradient at the original point.

### 4. Numerical evidence and a useful symmetry

For the coordinate $b_2$ in this fixture,
$f(b)=\log(1+e^{-b})+0.3$. The identity
$\log(1+e^{-\varepsilon})-\log(1+e^{\varepsilon})=-\varepsilon$
makes the central difference exactly $-1/2$ in real arithmetic for every
nonzero epsilon. The three suggested checks should be near $-0.5$; their
rounding errors need not decrease monotonically.

This symmetry does not hold for a general parameter and model. Central
differences then balance truncation error against cancellation and rounding.
Changing epsilon should reveal a useful range of agreement, not a rule that
the smallest allowed value is best. Near-zero gradients also make relative
error hard to interpret; inspect absolute error and the perturbed loss values.

### 5. Averaging determines the scale

Duplicating the rows leaves mean BCE unchanged. It also leaves the average
data gradient unchanged. The L2 penalty and its gradient are independent of
the number of duplicated rows, so the full objective, gradient, and proposed
step are unchanged apart from possible floating-point summation effects.

With a sum instead, duplication doubles the data loss and data gradient but
does not double the fixed L2 term. The step changes and the penalty has less
weight relative to the data term. Learning-rate and regularization conventions
must be specified before comparing implementations.

### 6. A local direction is not a finite-step guarantee

At $w=1$, the gradient is 2 and $w'=1-2\eta$.

| $\eta$ | New $w$ | Predicted $\Delta J=-4\eta$ | Actual $\Delta J=(1-2\eta)^2-1$ |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.8 | −0.4 | −0.36 |
| 0.5 | 0 | −2 | −1 |
| 1 | −1 | −4 | 0 |
| 2 | −3 | −8 | 8 |

For any nonzero starting weight, improvement requires
$(1-2\eta)^2<1$, or $0<\eta<1$. The negative derivative prediction ignores
the quadratic term. Likewise, a neural loss slice examines one coordinate
with all others fixed. Its appearance cannot establish the geometry or
number of minima of the full parameter space.

### 7. A new objective, not a new prediction

At $\lambda=0.2$, the unchanged parameters still give $p=0.5$ and BCE
$\log2$. The penalty becomes $0.6$, so $J\approx1.2931471806$.
The gradient vector becomes $[0.2,-0.2,-1,0.4,-0.5]$.

Changing the coefficient modifies the function being minimized, not the
current forward pass. Comparing loss before that setting change with loss
after an update combines an objective change with a parameter change. Compare
before and after on the same rows and the same $\lambda$ to assess the step.

### 8. Describe the observed scope

The evidence is that these temporary parameters have a lower stated objective
on the chosen 16 training rows. That does not establish a lower objective on
the full training set or any improvement on held-out examples. The original
test cards still describe the saved model, not the modified sandbox copy.
Fixed-batch gradient steps also do not recreate Adam's moment estimates,
mini-batch order, early stopping, or checkpoint selection.

A defensible follow-up is a new Python experiment with recorded settings and
seeds, trained on training data and selected using validation results. Compare
learning curves and repeat across sensible seeds. Freeze choices before final
test evaluation; if earlier test results influenced the choices, reserve fresh
evaluation data rather than calling the reused test set untouched.
