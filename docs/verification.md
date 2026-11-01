# Release verification

Checked on 2 November 2026 using macOS on Apple Silicon, Python 3.9.6, NumPy 2.0.2, and Matplotlib 3.9.4. Dependency versions are also recorded in `examples/xor/metrics.json`.

## Executed checks

- `python -m unittest discover -s tests -v`: **53 tests passed**.
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

`.github/workflows/tests.yml` runs the suite plus training, gradient checking, and explanation on Linux with Python 3.9 and 3.12, and macOS with Python 3.12. Consult the workflow run for the exact commit's status; this document records the local checks above rather than assuming remote jobs passed.

The lab does not require network access after dependencies are installed, does not collect telemetry, and does not create a public service. The HTML report embeds both figures and has no external scripts or remote assets.

The generated report was inspected in a browser at desktop size and at a 390-pixel viewport. The narrow layout had no page-wide horizontal overflow, and the expandable weights panel displayed the saved parameters.
