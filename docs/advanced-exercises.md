# Advanced exercises: make a prediction, then test it

Read [Beyond XOR](advanced-mathematics.md) first. Use a new output directory for
every run. Keep an experiment notebook with your settings, prediction, measured
result, and explanation. There is no promised score to reproduce across every
platform and dependency version.

Use validation results to choose settings. Inspecting and selecting repeatedly
by test scores turns the test partition into development data.

## Questions and experiments

### 1. Catch a preprocessing leak

The training values of one feature are $[0,2,4]$. A held-out value is 100.

1. Calculate the training mean and population standard deviation.
2. Transform the value 100 using those training statistics.
3. Explain what information leaks if you calculate the mean of all four values
   before splitting the data.
4. Should the trained model's weights change if only the held-out value changes
   from 100 to 1000?

### 2. Count updates, not just epochs

With 600 balanced examples, a 60/20/20 split, and a training batch size of 32:

1. How many training rows are there?
2. How many optimizer updates occur in one complete epoch?
3. How many rows does the final batch contain?
4. After ten full epochs, what is Adam's update counter $t$?

Compare batch sizes 16 and 128 with all other options fixed:

```sh
python -m neural_math_lab experiment --batch-size 16 --seed 7 --output runs/batch-16
python -m neural_math_lab experiment --batch-size 128 --seed 7 --output runs/batch-128
```

Do equal epoch budgets also give equal update budgets? Record the actual stopping
epochs before interpreting the curves.

### 3. Separate data loss from regularization

Consider weights $W_1=[1,-2]$, $W_2=[3]$, biases $b_1=[4,5]$, $b_2=6$, and
$\lambda=0.1$. These simplified arrays are only for this scalar calculation.

1. Calculate the L2 penalty in this lab's objective.
2. Give the extra gradient for each weight.
3. Give the extra gradient for each bias.
4. If the data cross-entropy is 0.4, what is the training objective? What loss
   would the validation plot show if validation cross-entropy were 0.5?

Then compare `--l2 0` with `--l2 0.05`. Use the same dataset and seed. Predict how
the boundary and weight magnitudes might change, and report the actual outcome
without assuming validation must improve.

### 4. Explain an Adam step that follows the current gradient

Reproduce the two-step example from the guide: $\theta_0=1$, $\eta=0.1$,
$g_1=2$, $g_2=-1$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

1. Compute both bias-corrected moment estimates at step 2.
2. Why does the parameter decrease even though the current gradient is negative?
3. What would two ordinary SGD updates do for that supplied gradient sequence?
4. Is the Adam time counter reset to zero at every epoch?

As an experiment, compare `--optimizer adam --learning-rate 0.01` with
`--optimizer sgd --learning-rate 0.05`. Treat this as two specified configurations,
not proof that one optimizer is always better. Give each a new output directory.

### 5. Read a confusion matrix and probability score

Suppose a held-out confusion matrix is

$$
\begin{bmatrix}42&8\\5&45\end{bmatrix}.
$$

Calculate accuracy, precision, recall, and $F_1$ using the convention in the
guide. Separately, calculate Brier score for targets $[0,1]$ and predictions
$[0.1,0.8]$. How would accuracy and Brier score change for predictions
$[0.49,0.51]$ on those same targets?

### 6. Distinguish the best epoch from the final executed epoch

Imagine validation losses at epochs 0 through 6 are:

```text
0.70, 0.55, 0.40, 0.42, 0.405, 0.41, 0.43
```

With patience 3 and `min_delta=0`, when should training stop, and which epoch's
parameters should be returned? If an implementation returns the last executed
weights instead, what selection rule has it violated?

Run one experiment with `--patience 20` and another with `--patience 100`, keeping
everything else fixed. Check `best_epoch` and `stopped_epoch`; they need not be
equal. Do not assume larger patience always finds a better validation loss.

### 7. Investigate capacity and geometry

Run circles with two hidden neurons, then with 16:

```sh
python -m neural_math_lab experiment --dataset circles --hidden 2 --seed 7 --output runs/circles-small
python -m neural_math_lab experiment --dataset circles --hidden 16 --seed 7 --output runs/circles-wide
```

Compare the boundaries, training losses, validation losses, selected epochs,
and logistic baseline. Count the trainable parameters in each neural network.
Why can increasing width help represent curvature while also increasing the
opportunity to fit incidental details? Is a better training score sufficient
to select the wider model?

### 8. Design a result you could defend

Before running anything, choose three seeds, two hidden widths, and a fixed
dataset/noise configuration. State your selection rule using validation loss.

Record every attempted configuration, including failures. Summarize variation
across seeds rather than showing only the best run. Explain why different seeds
can change both data generation and optimization, so they are repeated complete
experiments rather than a clean measurement of initialization alone.

Once you have looked at all corresponding test scores, can you honestly describe
the best score among them as an untouched final evaluation? What additional
evaluation would you need for a stronger claim?

---

## Answers and checks

### 1. Training statistics

The mean is 2; the population variance is $8/3$ and the standard deviation is
$\sqrt{8/3}\approx1.632993$. The transformed held-out value is
$98/\sqrt{8/3}\approx60.012499$.

Using all four rows would make the mean 26.5, allowing the held-out example to
change the representation on which training occurs. The correct training
parameters must not change when only a test coordinate changes, because that
coordinate does not participate in fitting or validation selection. Its final
prediction and test metrics can change.

### 2. Update counts

There are 360 training rows. An epoch has $\lceil360/32\rceil=12$ updates:
eleven batches of 32 and one of 8. After ten complete epochs, Adam has made 120
updates. For batch size 16 there are 23 updates per epoch; for 128 there are 3.
Equal epoch counts therefore do not mean equal numbers of parameter updates.

### 3. Penalty and gradients

The squared weight sum is $1^2+(-2)^2+3^2=14$, so the penalty is
$(0.1/2)14=0.7$. Added weight gradients are $[0.1,-0.2]$ and $[0.3]$. Added bias
gradients are zero. The objective is $0.4+0.7=1.1$, whereas the validation plot
shows its own unpenalized data loss, 0.5.

There is no fixed numerical answer for the experiment. Larger L2 encourages
smaller weights, but the resulting nonlinear training path and validation score
must be measured. Do not call an unobserved improvement a result.

### 4. Adam versus SGD

At step 2, $\widehat m_2\approx0.421052632$ and
$\widehat v_2\approx2.499249625$. The positive retained first moment causes a
positive quantity to be subtracted from the parameter. Adam ends near
0.873366297. SGD gives $1-0.1(2)=0.8$, then $0.8-0.1(-1)=0.9$.

The counter advances at every update and is not reset at epoch boundaries.
Resetting it while retaining the moment buffers would apply inconsistent bias
corrections.

### 5. Metrics

$TN=42$, $FP=8$, $FN=5$, $TP=45$, and $N=100$.

- Accuracy: $87/100=0.87$.
- Precision: $45/53\approx0.849057$.
- Recall: $45/50=0.9$.
- $F_1$: $90/103\approx0.873786$.

The first pair's Brier score is $(0.1^2+(-0.2)^2)/2=0.025$. The second pair's
Brier score is $0.49^2=0.2401$. Both classify the two targets correctly at
threshold 0.5 and therefore have accuracy 1. Probability-sensitive metrics
distinguish predictions that accuracy treats as identical.

### 6. Checkpoint selection

The minimum occurs at epoch 2. Epochs 3, 4, and 5 provide three consecutive
non-improvements, so training stops after epoch 5 and restores epoch 2's
parameters. Epoch 6 would never be reached under this rule. Returning epoch 5
would report a model other than the one selected by lowest validation loss.

With positive `min_delta`, small improvements affect the patience rule; see
the exact distinction in the guide. Always inspect both the selected epoch and
the stopping epoch.

### 7. Capacity

With hidden width $m$, this architecture has $4m+1$ parameters: 9 for width 2
and 65 for width 16. More hidden features can express more varied decision
boundaries. Whether that helps validation depends on the data and optimization.
Training performance measures fit to examples that supplied gradients; it does
not settle model selection on its own.

### 8. Defensible reporting

Choose settings using validation performance and retain the complete experiment
record. A final evaluation needs data not used directly or indirectly in those
choices. After repeatedly inspecting test results and selecting the best, a
fresh independently generated evaluation set would be needed to recover an
untouched assessment for this synthetic task. Even that would support a claim
about this data-generating process, not arbitrary real-world data.

## Experiment record

| Dataset/noise | Seed | Width | Optimizer/rate | L2 | Batch | Best/stopped epoch | Validation loss | What changed? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

Keep the final test comparison in a separate section of your notes. Explain
the observed difference from logistic regression and any remaining uncertainty.
