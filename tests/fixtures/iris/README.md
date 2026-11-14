# Iris test fixture

Exact `sklearn/datasets/data/iris.csv` from scikit-learn 1.7.2, commit
`25dee604bae18205b01548348388baf7a1cdfe0e`; SHA-256
`f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449`.
The upstream package license is preserved in `SCIKIT-LEARN-LICENSE.txt`.

Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C56C76. UCI publishes Iris under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.
The scikit-learn edition corrects two measurements described in its Iris dataset
documentation. This fixture keeps all four original measurement columns; the
application selects the two sepal columns and maps setosa to class 1 for a binary
educational task. These transformations are not the original three-class task.
