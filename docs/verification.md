# Release verification

## Numerical review fixes — 5 November 2026

The full local suite passed **143 Python tests** and **19 Node.js tests** after
fixing two reproduced review findings. Eight added regressions cover:

- L2 arithmetic with learner-supplied nested lists and integer matrices. The
  regularizer now uses validated float64 values without replacing or mutating
  the exposed parameter containers. The integer fixture previously returned a
  negative objective; it now matches `log(2) + 80`.
- Decimal cost scaling, symmetric threshold distances, genuine cost differences,
  zero versus tiny positive cost, the exact 0.5 candidate, and comparisons
  anchored to the minimum so approximate ties cannot chain together.

The reproduced failures were observed before their fixes. The separate numerical
gradient check passed with maximum absolute error about `1.10e-11`. Reselecting
thresholds from the committed moons and circles validation scores reproduced
all four saved selections exactly, including the linear baselines; the example
artifacts did not need regeneration. No browser runtime or layout changed.

## Decision lab release (0.3) — 5 November 2026

Checked locally on macOS Apple Silicon with Python 3.9.6, NumPy 2.0.2 and
Matplotlib 3.9.4. The complete suite passed **135 Python tests** and **15 Node.js
tests**. These include the earlier storage and training regressions.

New mathematical checks compare AUC with an independent positive–negative
pair calculation, AP and calibration with hand arithmetic, threshold selection
with exhaustive candidate rules, and raw-input derivatives with central
finite differences. They cover equal scores, missing classes, empty bins,
probabilities at 0 and 1, the all-negative rule, invalid inputs, and strict
floating-point settings. Changing test rows or decision costs cannot change
trained weights, scaling, history, or selected epochs. An additional independent
review checked 1,554 small label/score configurations against exhaustive rules.

Both committed advanced examples were regenerated from the default experiment
settings. At their validation-selected thresholds (equal error costs):

| Dataset | Threshold | Test FP / FN | Test mean cost | Test ROC AUC |
| --- | --- | --- | --- | --- |
| Moons | 0.6830711918697014 | 1 / 1 | 0.016667 | 0.9975 |
| Circles | 0.4430678144727584 | 1 / 2 | 0.025000 | 0.993333 |

The existing fixed-0.5 results remain unchanged. These are descriptive results
on one synthetic split, not population guarantees. The full diagnostics and
exact validation scores are exported; `metrics.json` keeps a compact summary.

Browser checks exercised exact threshold restoration, keyboard adjustment,
all-positive/all-negative rules, invalid-input recovery, baseline switching,
checkpoint replay, and raw-coordinate displacement. The held-out table stayed
unchanged during exploration. At `(0.5, 0.2)` with displacement `(0.01, -0.02)`,
the moons inspector matched Python: first-order output `0.562471044`, actual
output `0.562382115`, absolute error about `8.89e-5`.

Mobile checks at 390- and 320-pixel viewport widths found and fixed hidden
section navigation and page-wide overflow from an expanded summary table.
The final page has no horizontal overflow; wide numeric tables scroll inside
their panels. Expanded ROC tables also passed this check. No browser console
errors or warnings occurred in the tested interactions. Reports have no remote
assets and preserve their inline-script CSP hash.

The 0.3 wheel was built and installed into a temporary directory. Running the
installed experiment and sensitivity CLI outside the checkout succeeded; the
wheel includes the diagnostics module and all offline report assets. GitHub
Actions covers Ubuntu Python 3.9/3.12 and macOS Python 3.12, with Node 22 pinned
for browser-logic tests. Check the exact commit's workflow for remote results.

Checked on 2 November 2026 using macOS on Apple Silicon, Python 3.9.6, NumPy 2.0.2, and Matplotlib 3.9.4. Dependency versions are recorded alongside each example.

## Advanced release (0.2)

- **105 Python tests passed**, including the original 53 regressions and 52 new tests for datasets, optimizers, experiments, pipelines, and the advanced CLI.
- **5 Node.js tests passed**, including agreement within `1e-12` between the browser's inference and every saved Python test prediction in both advanced examples.
- A wheel was built, installed into a temporary directory, and used outside the source checkout to generate a circles report. All three HTML/CSS/JS assets were included. The dataset-specific default path was `runs/circles`.
- Browser checks covered checkpoint keyboard navigation, animated replay, restoring the selected model, switching to the linear baseline, split selection, mistake marking, and raw-coordinate inspection. Baseline mode correctly disables neural replay.
- At raw input `(0.5, 0.2)`, the included moons report displayed `0.538760`, matching the Python explanation. Submitting a high-precision coordinate twice preserved both the inputs and prediction.
- The report was checked at desktop size and a **390 × 844** mobile viewport. A grid sizing issue initially let tables and command text overflow the page; after correction, document width and scroll width were both 390 pixels. Mobile point inspection worked, and the tested page had no browser console errors.

The default advanced runs used seed 7, 600 rows, noise 0.15, 16 hidden neurons, Adam at 0.01, batches of 32, L2 0.001, and at most 1,200 epochs. Each run contained 360 training, 120 validation, and 120 test examples.

| Dataset | Selected / stopped epoch | Neural test accuracy | Baseline test accuracy | Neural test BCE |
| --- | --- | --- | --- | --- |
| Moons | 777 / 877 | 116/120 (96.67%) | 111/120 (92.5%) | 0.07208 |
| Circles | 374 / 474 | 117/120 (97.5%) | 57/120 (47.5%) | 0.10919 |

These results describe one synthetic split per dataset. They do not establish accuracy on another distribution or statistical superiority across seeds. Saved artifacts in `examples/moons` and `examples/circles` contain the complete data, split IDs, history, checkpoints, and predictions needed to inspect these measurements.

The new tests independently check Adam's first two updates, SGD arithmetic, L2 finite-difference gradients and unpenalized biases, stratified/disjoint splits, train-only scaling, deterministic histories, minimum-validation checkpoint restoration, bounded replay checkpoints, and exact metric counts. A test modifies only held-out examples and verifies that fitted parameters, training history, scaler, baseline, and selected epoch do not change. Integration tests reload the pipeline and recompute every test prediction, validate the embedded script's CSP hash, and ensure reports have no remote assets.

Review also caught an inspector rounding issue: the fields displayed five decimals while inference used the original coordinates. Inputs now preserve their actual numeric values, so resubmission is consistent.

## Executed checks

- The original 53 tests remain part of the expanded passing suite above.
- `python -m neural_math_lab gradient-check`: maximum absolute difference between analytic and central-difference gradients **1.10493e-11** at the default initialization.
- `python -m neural_math_lab train`: completed 4,000 updates with seed 7, hidden width 4, and learning rate 0.5. The recorded run was copied to `examples/xor/`.
- `python -m neural_math_lab explain --model runs/xor/model.json --input 1 0`: displayed the learned parameters and actual intermediate layer values.
- Editable package installation succeeded in a fresh project virtual environment after updating the Mac's older bundled pip.

The default run's mean binary cross-entropy decreased from **0.7046521131** to **0.0010746149**. Its outputs were approximately:

| Input | Target | Sigmoid output | Predicted class |
| --- | --- | --- | --- |
| 0, 0 | 0 | 0.000163590 | 0 |
| 0, 1 | 1 | 0.998511904 | 1 |
| 1, 0 | 1 | 0.998900159 | 1 |
| 1, 1 | 0 | 0.001544013 | 0 |

This is accuracy on the complete **training truth table**, not held-out accuracy. Numerical gradient agreement at tested points and finite test cases do not constitute a proof of correctness for all possible floating-point inputs.

## Regression coverage

The 31 core tests cover independently computed forward values, every parameter's numerical gradient on asymmetric batches and single rows, averaging invariants, stable large logits under strict floating-point settings, actual arithmetic overflow, finite update validation, oversized integer learning rates, gradient-check restoration after failures, deterministic seeds, and XOR learning for seeds 0, 7, and 42.

The 12 storage tests cover exact prediction round trips, strict JSON numeric parameters, format/shape/size limits, malformed data, and missing files. The 10 CLI tests execute real subprocess commands, verify metrics against reloaded weights, parse SVG figures and offline HTML, check the entire history, preserve existing outputs, and verify readable error messages.

During visual-report testing, this Mac's NumPy/Apple Accelerate combination emitted floating-point exception flags for a large, finite matrix product. Matrix multiplication now checks the resulting numbers directly; it still rejects genuine non-finite results. A regression test compares a 14,641-point prediction grid to independently computed scalar results. A second regression verifies that true matrix overflow is rejected.

## Continuous checks

`.github/workflows/tests.yml` runs both test suites plus XOR training, gradient checking, advanced experiments, pipeline explanation, and an installed-package smoke check on Ubuntu 24.04 with Python 3.9, current Ubuntu with Python 3.12, and macOS with Python 3.12. The compatibility job is pinned because GitHub's Python distribution manifest does not provide Python 3.9 for Ubuntu 26.04. Consult the workflow run for the exact commit's status; this document records the local checks above rather than assuming remote jobs passed.

The lab does not require network access after dependencies are installed, does not collect telemetry, and does not create a public service. Advanced reports embed the learning figure, interactive plot code, and recorded experiment data; separate SVG exports are also provided. There are no external scripts or remote assets.

The generated report was inspected in a browser at desktop size and at a 390-pixel viewport. The narrow layout had no page-wide horizontal overflow, and the expandable weights panel displayed the saved parameters.
