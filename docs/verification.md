# Release verification

## Toolchain refresh — 7 November 2026

This refresh changes no package code. It updates continuous integration and
records checks against the newest Python, NumPy, and Matplotlib releases.
Python 3.9 and 3.10 have reached end of life. The package still accepts both
(`requires-python = ">=3.9"`), because `/usr/bin/python3` on macOS still runs
Python 3.9 from Apple's Command Line Tools. Python 3.10 was not tested.

The workflow now uses `actions/checkout@v7`, `actions/setup-python@v7`, and
`actions/setup-node@v7`. The previous v4/v5 actions targeted Node.js 20, so
every job reported a deprecation warning. The new matrix is described under
Continuous checks. Each job records its resolved dependency versions.

Checked locally on macOS Apple Silicon. After `npm ci --ignore-scripts`, the 64
Node.js tests and the website build passed with Node.js 26.13.0, and the Python
tests passed in each environment below:

| Python | NumPy | Matplotlib | Python tests |
| --- | --- | --- | --- |
| 3.9.6 (Command Line Tools `/usr/bin/python3`) | 2.0.2 | 3.9.4 | 147 passed |
| 3.11.18 | 1.26.0 (oldest accepted) | 3.7.0 (oldest accepted) | 147 passed |
| 3.11.18 | 2.4.6 | 3.11.2 | 147 passed |
| 3.12.16, 3.13.18, 3.14.10 | 2.5.4 | 3.11.2 | 147 passed each |
| 3.15.2 | 2.5.4 | 3.11.2 | 147 passed, also with warnings as errors |

Each `run:` step of the new workflow was also replayed on this Mac for the
Python 3.9, 3.12, and 3.15 jobs and the minimum-dependency job, in a fresh copy
and a new virtual environment with the workflow's own `pip install`, `npm ci`
and build commands.
Every step passed. The replay used Node.js 26.13.0 throughout and Python 3.9.6
for the 3.9 job, so it did not exercise Ubuntu, Node.js 22 or 24, or the
Python 3.9.25 that CI installs.

On Python 3.15.2 with NumPy 2.5.4 and Matplotlib 3.11.2, the default moons,
circles, and XOR runs reproduce the committed `metrics.json`,
`diagnostics.json`, `pipeline.json`, and `model.json` values exactly, apart
from the recorded environment versions. In `experiment.json` and
`history.csv`, the only other value that changes is the recorded gradient
norm: 162 of 878 moons epochs and 75 of 475 circles epochs differ by one unit
in the last place. The gradients and NumPy's per-parameter sums are bitwise
identical; since Python 3.12, the built-in `sum()` that adds those four sums
uses compensated (Neumaier) summation. The gradient norm is a diagnostic: it
does not affect updates, selection, or stopping. The committed examples are
therefore unchanged and still record the environment that produced them.

Default moons training took 1.45–1.54 s on the new stack versus 2.07–2.22 s on
Python 3.9 with NumPy 2.0.2 (best of three runs, repeated five times). On this
Mac, Matplotlib 3.9 and newer took about 10 seconds to build a fresh font
cache, almost all of it in one `system_profiler SPFontsDataType` call; 3.7
took under one second. Matplotlib then reuses the cache from `~/.matplotlib`
across Python versions and virtual environments until its font-cache version
changes or `MPLCONFIGDIR` points to a new directory. The slow step does not run
on Linux or Windows. The two CLI test classes each set a private
`MPLCONFIGDIR`, so with Matplotlib 3.11.2 their two cache builds took about 19
of the suite's 26 seconds.

A wheel built with Python 3.15 contained every module and report asset, and
its installed `neural-math-lab` command produced a complete experiment outside
the checkout. On Python 3.15.2, 25 invalid CLI inputs each reported a one-line
error and exited with status 2. Four extreme inputs that the CLI accepts
(weights of 1e300, learning rates of 1e6 and 1e300, and 64 hidden units)
exited with status 0 and finite numbers, although both large learning rates
diverged. None produced a traceback or left a partial output directory. On
Python 3.9 and 3.11, `argparse` itself rejects a negative number written in
exponent notation, such as `--input 0.5 -2e-1`, with a usage message.

In headless Chrome 156, each committed report loaded with one network request,
over HTTP and from disk, and logged no console messages. Running the moons and
circles reports' own inference code there reproduced all 240 saved Python test
predictions within 1e-12 (largest difference 3.3e-16), for both the committed
reports and reports regenerated on Python 3.15.

## Performance and distributed capacity — 7 November 2026

The optimized site is live at <https://neural-math-lab.pages.dev/>. GitHub
privacy was rechecked and restored to private before this release was pushed.
The performance commit is attributed to `TARUNTEjjjj`.

The complete local JavaScript suite passed **64 tests**. GitHub Actions also
passed all **147 Python tests** and 64 JavaScript tests on both Linux Python
3.9 and 3.12 configurations, including training and installed-package checks.
See the exact workflow run for macOS status. The recorded experiment data
remain unchanged; the browser handler and hosted asset delivery were optimized.

The two-runner capacity workflow passed with 10,000 verified virtual learners
overlapping for **118.840 seconds**. During that overlap, **19,797** complete
build-matching responses had **zero failures**, **93.027 ms response p95** and
**9.883 ms dispatch-delay p95**. This is an HTTP document-navigation workload
with 60 seconds of reading time, not 10,000 full browser instances. The earlier
single-laptop 10,000-user run failed latency/queue limits and is retained.
See [the performance report](performance.md) for raw evidence and limitations.

All **43 live HTTP checks** passed, including exact deployed assets, cache
headers and five unavailable source paths. Browser checks covered full-gradient
apply/undo during the measured overlap, invalid-input recovery, 390-pixel
mobile layout and 75 rendered mathematics equations. No application console
errors or warnings occurred in the tested interactions.

## Cloudflare Pages website — 5 November 2026

The public learning site is served at <https://neural-math-lab.pages.dev/>.
The GitHub repository remains private. Publication uses an allowlisted `dist/`
build, not the source checkout. It includes a learning home, three reports,
eight linked guides, local equation fonts and a custom 404 page. KaTeX renders
equations and accessible MathML during the build; it does not run in visitors'
browsers. The original report scripts, synthetic data and inline-script CSP
hashes are preserved exactly.

All **40 Node.js tests** passed, including six website build regressions. These
check the public file allowlist, every local link and fragment, math rendering,
bad-equation failures, report/data preservation, and output-directory safety.
The link crawl caught two incorrect home-page anchors, which were corrected.
The dependency audit reported no known vulnerabilities at verification time.
The unchanged Python mathematics suite is also run by the updated CI workflow.

Twenty live HTTP checks passed: the home, all three labs, all eight guides,
stylesheet/font assets, and five missing/private paths. Successful responses
matched the local build byte for byte and included the configured security
headers. Source probes for Python, Git configuration, package metadata and
separate experiment JSON returned the actual 404 page.

Chrome loaded the production domain and successfully applied and undid a
gradient step (moons objective 0.889712 → 0.8895071 → 0.889712). Live browser
checks also exercised baseline/replay switching, the raw-point inspector,
threshold restoration, a full-gradient update, invalid-input recovery,
reset, and the circles sandbox. The selected moons model returned 0.538760
at raw coordinates (0.5, 0.2). Mobile checks at 320 and 390 pixels found no
page-wide overflow in the tested home, guide and lab views, including the
expanded slice table. The optimization guide rendered 75 equations without
math errors. No application console warnings or errors occurred in the
tested live interactions. These are functional checks, not a concurrent-user
load test.

## Gradient lab release (0.4) — 5 November 2026

The complete local suite passed **147 Python tests** and **34 Node.js tests**.
New checks compare every browser gradient with NumPy, central finite differences,
and a hand-computed single-neuron example. They also verify coordinate and full
gradient-descent updates, loss slices and tangents, stable extreme logits,
unpenalized biases, duplicated-batch invariance, and rejection of invalid or
overflowing arithmetic. These finite checks do not prove correctness for every
possible floating-point input.

UI regression tests execute the actual event handlers with a small DOM adapter.
They cover apply/undo/reset, the 50-step limit, invalid-input recovery, changes
to the starting model/batch/L2 coefficient, and recalculation at click time.
Adversarial validation/test rows do not enter updates. All operations preserve
the embedded recorded experiment, and resetting restores the chosen model.

Browser checks on both included reports covered numerical derivative agreement,
coordinate and all-parameter steps, keyboard activation, undo/reset, invalid
controls, and a deliberately large learning rate that increased the objective.
The held-out results stayed fixed. Desktop and 390- and 320-pixel mobile layouts
had no page-wide overflow, including with the 41-row slice table expanded.
No console warnings or errors occurred during these interactions. Browser
checks used a local HTTP preview: direct `file://` navigation was blocked by
the browser tool. Automated checks confirm embedded assets and matching CSP
hashes without remote scripts or styles.

The 0.4.0 wheel was built with the project virtual environment's current pip,
installed into a temporary directory, and used outside the source checkout to
generate a working report. All four new optimization assets were present.
The committed moons/circles HTML reports were refreshed from their recorded
experiments; their experiment data, saved models, and metrics are unchanged.
Worked exercises were checked against NumPy. Local checks used macOS Apple
Silicon, Python 3.9.6, NumPy 2.0.2, and Matplotlib 3.9.4. Consult the exact
commit's Actions run for Linux/macOS CI results.

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
wheel includes the diagnostics module and all offline report assets. At that
release, GitHub Actions covered Ubuntu Python 3.9/3.12 and macOS Python 3.12,
with Node 22 pinned for browser-logic tests. Check the exact commit's workflow
for remote results.

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

`.github/workflows/tests.yml` runs both test suites plus the website build, XOR training, gradient checking, advanced experiments, pipeline explanation, and an installed-package smoke check on Ubuntu 24.04 with Python 3.9 and Node.js 22, current Ubuntu with Python 3.12 and Node.js 24, and both current Ubuntu and macOS with Python 3.15 and Node.js 26. A separate Ubuntu 24.04 job runs the Python tests, with Node.js 22 for their browser-parity check, on Python 3.11 with NumPy 1.26.0 and Matplotlib 3.7.0: the oldest NumPy and Matplotlib releases that `pyproject.toml` accepts. The Python 3.9 job is pinned to Ubuntu 24.04 because GitHub's Python distribution manifest does not provide Python 3.9 for Ubuntu 26.04. CI tests Python 3.9 only on Ubuntu; the macOS Command Line Tools Python 3.9.6 is covered by the local checks above. Consult the workflow run for the exact commit's status; this document records the local checks above rather than assuming remote jobs passed.

The lab does not require network access after dependencies are installed, does not collect telemetry, and does not create a public service. Advanced reports embed the learning figure, interactive plot code, and recorded experiment data; separate SVG exports are also provided. There are no external scripts or remote assets.

The generated report was inspected in a browser at desktop size and at a 390-pixel viewport. The narrow layout had no page-wide horizontal overflow, and the expandable weights panel displayed the saved parameters.
