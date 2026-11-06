"""Compare the browser's optimization sandbox with independent Python math."""

import json
import math
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.experiment import objective_and_gradients
from neural_math_lab.storage import load_pipeline


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
MODULE = ROOT / "neural_math_lab" / "assets" / "optimization.js"
NODE_DRIVER = """
const fs = require('node:fs');
const optimization = require(process.argv[1]);
const requests = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(requests.map(({operation, args}) =>
  optimization[operation](...args))));
"""


def parameters(network):
    return {key: value.tolist() for key, value in network.parameters.items()}


def fixture_batch(l2):
    return {
        "X": [[-.8, .3], [2.1, -.9], [.4, -.2], [3.7, .6], [-1.3, -1.1]],
        "y": [0, 1, 1, 0, 1],
        "mean": [[1.25, -.7]],
        "scale": [[2.5, .3]],
        "l2": l2,
    }


def python_objective(network, batch):
    X = standardize(batch["X"], batch["mean"], batch["scale"])
    y = np.asarray(batch["y"], dtype=float).reshape(-1, 1)
    objective, gradients = objective_and_gradients(network, X, y, batch["l2"])
    bce = network.loss(X, y)
    # A scalar reference for the penalty, independent of the JS reduction.
    penalty = batch["l2"] / 2 * sum(
        float(weight) ** 2 for key in ("W1", "W2")
        for weight in network.parameters[key].ravel())
    return objective, bce, penalty, gradients


@unittest.skipUnless(NODE, "Node.js is required for browser/Python parity checks")
class OptimizationParityTests(unittest.TestCase):
    def javascript(self, *requests):
        result = subprocess.run(
            [NODE, "-e", NODE_DRIVER, str(MODULE)],
            input=json.dumps(requests, allow_nan=False), text=True,
            capture_output=True, check=False, cwd=ROOT, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def assert_objective_matches(self, actual, expected):
        objective, bce, penalty, gradients = expected
        np.testing.assert_allclose(
            [actual["loss"], actual["bce"], actual["penalty"]],
            [objective, bce, penalty], rtol=2e-12, atol=2e-13)
        self.assertEqual(set(actual["gradients"]), set(gradients))
        for name, gradient in gradients.items():
            np.testing.assert_allclose(actual["gradients"][name], gradient,
                                       rtol=2e-11, atol=2e-13, err_msg=name)
        norm = math.sqrt(sum(float(g) ** 2 for values in gradients.values()
                             for g in values.ravel()))
        self.assertAlmostEqual(actual["gradientNorm"], norm, places=11)

    def test_objective_and_all_derivatives_match_python_with_scaling_and_l2(self):
        requests, references = [], []
        for hidden in (3, 7):
            network = Network(hidden=hidden, seed=19)
            network.parameters["b1"][:] = np.linspace(-.2, .3, hidden)
            network.parameters["b2"][:] = -.15
            for l2 in (0, .17):
                batch = fixture_batch(l2)
                requests.append({"operation": "objectiveAndGradients",
                                 "args": [parameters(network), batch]})
                references.append(python_objective(network, batch))
        for actual, reference in zip(self.javascript(*requests), references):
            self.assert_objective_matches(actual, reference)

    def test_full_and_coordinate_updates_match_python_descent_and_losses(self):
        network = Network(hidden=4, seed=23)
        batch, rate = fixture_batch(.07), .2
        before, _, _, gradients = python_objective(network, batch)
        coordinates = [None, {"name": "W1", "row": 1, "col": 2}]
        requests = [{"operation": "proposeStep",
                     "args": [parameters(network), batch, rate, coordinate]}
                    for coordinate in coordinates]
        for coordinate, actual in zip(coordinates, self.javascript(*requests)):
            expected = Network(hidden=4, seed=0)
            expected.parameters = {key: value.copy() for key, value in network.parameters.items()}
            if coordinate is None:
                squared_norm = sum(float(g) ** 2 for values in gradients.values()
                                   for g in values.ravel())
                for key in expected.parameters:
                    expected.parameters[key] -= rate * gradients[key]
            else:
                key, row, col = (coordinate[k] for k in ("name", "row", "col"))
                gradient = float(gradients[key][row, col])
                squared_norm = gradient ** 2
                expected.parameters[key][row, col] -= rate * gradient
            after, _, _, _ = python_objective(expected, batch)
            for key in expected.parameters:
                np.testing.assert_allclose(actual["parameters"][key], expected.parameters[key],
                                           rtol=1e-12, atol=2e-13)
            self.assertAlmostEqual(actual["before"]["loss"], before, places=12)
            self.assertAlmostEqual(actual["after"]["loss"], after, places=12)
            self.assertAlmostEqual(actual["predictedChange"], -rate * squared_norm, places=12)
            self.assertAlmostEqual(actual["actualChange"], after - before, places=12)
            self.assertAlmostEqual(actual["stepNorm"], rate * math.sqrt(squared_norm), places=12)

    def test_saved_models_use_training_rows_and_match_python_loss_slices(self):
        coordinate = {"name": "W2", "row": 0, "col": 0}
        for dataset in ("moons", "circles"):
            with self.subTest(dataset=dataset):
                directory = ROOT / "examples" / dataset
                document = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
                network, mean, scale, _ = load_pipeline(directory / "pipeline.json")
                data = document["data"]
                ids = data["train_idx"][:16]
                self.assertTrue(set(ids).isdisjoint(data["val_idx"]))
                self.assertTrue(set(ids).isdisjoint(data["test_idx"]))
                batch = {"X": [data["X"][i] for i in ids],
                         "y": [data["y"][i] for i in ids],
                         "mean": mean.tolist(), "scale": scale.tolist(),
                         "l2": document["config"]["l2"]}
                actual, section = self.javascript(
                    {"operation": "objectiveAndGradients", "args": [parameters(network), batch]},
                    {"operation": "lossSlice", "args": [parameters(network), batch, coordinate, .5, 5]},
                )
                expected = python_objective(network, batch)
                self.assert_objective_matches(actual, expected)
                center, gradient = expected[0], float(expected[3]["W2"][0, 0])
                np.testing.assert_allclose([row["offset"] for row in section],
                                           [-.5, -.25, 0, .25, .5], atol=1e-15)
                for row in section:
                    shifted = Network(hidden=network.hidden, seed=0)
                    shifted.parameters = {key: value.copy() for key, value in network.parameters.items()}
                    shifted.parameters["W2"][0, 0] += row["offset"]
                    reference_loss = python_objective(shifted, batch)[0]
                    self.assertAlmostEqual(row["loss"], reference_loss, places=11)
                    self.assertAlmostEqual(row["tangent"], center + gradient * row["offset"], places=11)


if __name__ == "__main__":
    unittest.main()
