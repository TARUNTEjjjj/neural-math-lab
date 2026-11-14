# Import a real, reproducible dataset

The resource catalog links useful projects. The separate Iris adapter imports
actual observations from the high-star **scikit-learn** project into the existing
training pipeline. The adapter has no scikit-learn dependency and downloads no code.

```sh
python -m neural_math_lab fetch-dataset --output datasets/iris
python -m neural_math_lab experiment --dataset iris --data-path datasets/iris --output runs/iris
```

The first command explicitly downloads a 2.7 KB CSV from scikit-learn commit
`25dee604bae18205b01548348388baf7a1cdfe0e` (release 1.7.2). It verifies SHA-256
`f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449`, checks all
150 rows and writes `iris.csv` with `source.json` to a new directory. An existing
directory is preserved. Responses are limited to 16 KiB, a 20-second timeout and
the exact HTTPS URL; redirects or changed bytes fail before any dataset is written.

Training reads only the local file and rechecks the hash. There is no automatic
network call during training, tests, the browser report or ordinary app startup.
To adopt a new source edition, review its measurements, update the pin/hash and
fixture, then rerun the tests. Popularity does not bypass that review.

The network has two inputs and one binary output, so the adapter deliberately
uses **sepal length and sepal width in centimetres**, with **class 1 = setosa** and
**class 0 = versicolor or virginica**. All 150 observations are retained in source
order. This is a transformed educational task, not the original three-class Iris
benchmark. Do not pass the synthetic `--samples` or `--noise` options for Iris.
The existing deterministic stratified split gives 90 training, 30 validation and
30 held-out rows. Scaling uses training rows only; model and threshold selection
use validation rows. A small result from this familiar dataset is not a general
species classifier. The source, revision, hash, features, labels and license
references are saved in the report and experiment JSON.

Source: [scikit-learn Iris CSV](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/sklearn/datasets/data/iris.csv)
and [dataset description](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/sklearn/datasets/descr/iris.rst).
The upstream package has a [BSD-3-Clause license](https://github.com/scikit-learn/scikit-learn/blob/25dee604bae18205b01548348388baf7a1cdfe0e/COPYING),
preserved with the offline test fixture. The original dataset attribution is
Fisher, R. (1936). *Iris* [Dataset], [UCI DOI 10.24432/C56C76](https://doi.org/10.24432/C56C76).
[UCI's dataset page](https://archive.ics.uci.edu/dataset/53/iris) publishes it under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The scikit-learn edition
includes documented corrections relative to the original UCI file.
