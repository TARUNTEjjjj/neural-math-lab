"""Independent optimizer arithmetic, evaluation, and data-isolation checks."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.experiment import (
    Adam,
    SGD,
    classification_metrics,
    objective_and_gradients,
    run_experiment,
    _Logistic,
)


class OptimizerTests(unittest.TestCase):
    def test_sgd_update_matches_scalar_arithmetic(self):
        parameters = {"w": np.array([[1., -2.]]), "b": np.array([[.5]])}
        gradients = {"w": np.array([[.4, -.8]]), "b": np.array([[.2]])}
        SGD(learning_rate=.25).step(parameters, gradients)
        np.testing.assert_allclose(parameters["w"], [[.9, -1.8]], atol=1e-15)
        np.testing.assert_allclose(parameters["b"], [[.45]], atol=1e-15)

    def test_adam_first_two_steps_include_bias_correction(self):
        rate, beta1, beta2, epsilon = .1, .8, .9, 1e-8
        parameters = {"w": np.array([[2.]])}
        optimizer = Adam(learning_rate=rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        expected, first, second = 2., 0., 0.
        for timestep, gradient in enumerate((.4, -.2), start=1):
            first = beta1 * first + (1 - beta1) * gradient
            second = beta2 * second + (1 - beta2) * gradient * gradient
            corrected_first = first / (1 - beta1 ** timestep)
            corrected_second = second / (1 - beta2 ** timestep)
            expected -= rate * corrected_first / (math.sqrt(corrected_second) + epsilon)
            optimizer.step(parameters, {"w": np.array([[gradient]])})
            self.assertAlmostEqual(float(parameters["w"][0, 0]), expected, places=14)

    def test_zero_gradients_leave_parameters_unchanged(self):
        for optimizer in (SGD(.1), Adam(.1)):
            parameters = {"w": np.array([[1., -2.]])}
            optimizer.step(parameters, {"w": np.zeros((1, 2))})
            np.testing.assert_array_equal(parameters["w"], [[1., -2.]])

    def test_rejected_updates_preserve_all_parameters(self):
        for optimizer in (SGD(.1), Adam(.1)):
            parameters = {"w": np.array([[1., -2.]]), "b": np.array([[.5]])}
            before = {name: value.copy() for name, value in parameters.items()}
            with self.assertRaises(ValueError):
                optimizer.step(parameters, {"w": np.array([[.4, -.8]]), "b": np.array([[np.nan]])})
            for name in before:
                np.testing.assert_array_equal(parameters[name], before[name])

    def test_optimizer_settings_must_be_finite_and_valid(self):
        for constructor in (SGD, Adam):
            for rate in (0, -.1, np.inf, np.nan, True):
                with self.subTest(optimizer=constructor.__name__, rate=rate), self.assertRaises(ValueError):
                    constructor(rate)
        for settings in ({"beta1": 1}, {"beta1": -.1}, {"beta2": 1}, {"beta2": np.nan}, {"epsilon": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Adam(.1, **settings)

    def test_rejected_adam_step_does_not_advance_moments_or_timestep(self):
        optimizer, reference = Adam(.1), Adam(.1)
        actual = {"w": np.array([[2.]]), "b": np.array([[.3]])}
        expected = {name: value.copy() for name, value in actual.items()}
        good = {"w": np.array([[.4]]), "b": np.array([[-.2]])}
        optimizer.step(actual, good)
        reference.step(expected, good)
        with self.assertRaises(ValueError):
            optimizer.step(actual, {"w": np.array([[.8]]), "b": np.array([[np.inf]])})
        self.assertEqual(optimizer.t, reference.t)
        for name in good:
            np.testing.assert_array_equal(optimizer.m[name], reference.m[name])
            np.testing.assert_array_equal(optimizer.v[name], reference.v[name])
        optimizer.step(actual, good)
        reference.step(expected, good)
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])


class RegularizationTests(unittest.TestCase):
    def test_baseline_retains_small_loss_and_gradient_at_correct_saturation(self):
        baseline = _Logistic()
        X, y = np.array([[1., -2.]]), np.array([[1.]])
        baseline.parameters["b"][:] = 40
        tail = math.exp(-40)
        residual = -tail / (1 + tail)
        with np.errstate(all="raise"):
            np.testing.assert_allclose(baseline.loss(X, y), math.log1p(tail), rtol=1e-14, atol=0)
            gradients = baseline.gradients(X, y)
        np.testing.assert_allclose(gradients["b"], [[residual]], rtol=1e-14, atol=0)
        np.testing.assert_allclose(gradients["W"], [[residual], [-2 * residual]], rtol=1e-14, atol=0)

    def test_baseline_averages_large_finite_losses_before_summing(self):
        baseline = _Logistic()
        baseline.parameters["b"][:] = 1e308
        with np.errstate(all="raise"):
            result = baseline.loss(np.zeros((4, 2)), np.zeros((4, 1)))
        self.assertEqual(result, 1e308)

    def test_nested_list_parameters_match_float64_objective_without_mutation(self):
        network = Network(hidden=3, seed=11)
        X = np.array([[.2, -.4], [.7, .3], [-.1, .8]])
        y = np.array([[0.], [1.], [1.]])
        expected_loss = network.loss(X, y)
        expected_objective, expected_gradients = objective_and_gradients(network, X, y, l2=.17)
        network.parameters = {name: value.tolist() for name, value in network.parameters.items()}
        originals = dict(network.parameters)
        before = {name: [row[:] for row in value] for name, value in originals.items()}

        objective, gradients = objective_and_gradients(network, X, y, l2=.17)

        self.assertAlmostEqual(network.loss(X, y), expected_loss, places=14)
        self.assertAlmostEqual(objective, expected_objective, places=14)
        for name in network.parameters:
            np.testing.assert_allclose(gradients[name], expected_gradients[name], atol=1e-14)
            self.assertIs(network.parameters[name], originals[name])
            self.assertEqual(network.parameters[name], before[name])

    def test_integer_weights_are_squared_in_float64_without_mutation(self):
        network = Network(hidden=2, seed=7)
        weights = np.full((2, 2), 200, dtype=np.int16)
        network.parameters["W1"] = weights
        network.parameters["W2"][:] = 0
        X, y = np.zeros((2, 2)), np.array([[0.], [1.]])

        with np.errstate(all="raise"):
            objective, gradients = objective_and_gradients(network, X, y, l2=.001)

        # Four weights contribute (lambda / 2) * 4 * 200**2 = 80.
        self.assertAlmostEqual(objective, math.log(2) + 80, places=13)
        np.testing.assert_allclose(gradients["W1"], np.full((2, 2), .2), atol=1e-14)
        self.assertIs(network.parameters["W1"], weights)
        self.assertEqual(weights.dtype, np.dtype(np.int16))
        np.testing.assert_array_equal(weights, np.full((2, 2), 200, dtype=np.int16))

    def test_disabled_l2_does_not_square_large_but_unused_weights(self):
        network = Network(hidden=2, seed=7)
        network.parameters["W1"][:] = 1e200
        network.parameters["W2"][:] = 0
        X, y = np.zeros((2, 2)), np.array([[0.], [1.]])
        with np.errstate(all="raise"):
            objective, gradients = objective_and_gradients(network, X, y, l2=0)
        self.assertAlmostEqual(objective, math.log(2), places=14)
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))

    def test_l2_penalty_matches_explicit_weight_squares_and_leaves_bias_gradients_alone(self):
        network = Network(hidden=3, seed=11)
        X = np.array([[.2, -.4], [.7, .3], [-.1, .8]])
        y = np.array([[0.], [1.], [1.]])
        strength = .17
        objective, gradients = objective_and_gradients(network, X, y, strength)
        plain = network.gradients(X, y)
        penalty = strength / 2 * sum(float(v) ** 2 for key in ("W1", "W2") for v in network.parameters[key].ravel())
        self.assertAlmostEqual(objective, network.loss(X, y) + penalty, places=13)
        for key in ("b1", "b2"):
            np.testing.assert_allclose(gradients[key], plain[key], atol=1e-14)
        for key in ("W1", "W2"):
            np.testing.assert_allclose(gradients[key] - plain[key], strength * network.parameters[key], atol=1e-14)

    def test_regularized_gradients_match_independent_finite_differences(self):
        network = Network(hidden=3, seed=11)
        X = np.array([[.2, -.4], [.7, .3], [-.1, .8]])
        y = np.array([[0.], [1.], [1.]])
        strength, epsilon = .17, 1e-6
        _, gradients = objective_and_gradients(network, X, y, strength)

        def independent_objective():
            penalty = strength / 2 * sum(float(v) ** 2 for key in ("W1", "W2") for v in network.parameters[key].ravel())
            return network.loss(X, y) + penalty

        for name, parameters in network.parameters.items():
            numeric = np.empty_like(parameters)
            for index in np.ndindex(parameters.shape):
                original = parameters[index]
                try:
                    parameters[index] = original + epsilon
                    plus = independent_objective()
                    parameters[index] = original - epsilon
                    minus = independent_objective()
                    numeric[index] = (plus - minus) / (2 * epsilon)
                finally:
                    parameters[index] = original
            np.testing.assert_allclose(gradients[name], numeric, atol=2e-8, rtol=2e-6, err_msg=name)


class MetricTests(unittest.TestCase):
    def test_confusion_precision_recall_f1_and_brier_match_hand_counts(self):
        y = np.array([[0.], [0.], [1.], [1.], [1.]])
        p = np.array([[.1], [.7], [.5], [.8], [.2]])
        metrics = classification_metrics(y, p, loss=.7)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 3 / 5)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["brier"], (.01 + .49 + .25 + .04 + .64) / 5)
        self.assertEqual(metrics["loss"], .7)

    def test_undefined_positive_class_scores_use_documented_zero_policy(self):
        metrics = classification_metrics(np.zeros((3, 1)), np.array([[.1], [.2], [.3]]), loss=.2)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [0, 0]])
        self.assertEqual(metrics["accuracy"], 1)
        for key in ("precision", "recall", "f1"):
            self.assertEqual(metrics[key], 0)

    def test_invalid_probabilities_or_labels_are_rejected(self):
        for p in (np.array([[-.1], [.5]]), np.array([[1.1], [.5]]),
                  np.array([[np.nan], [.5]]), np.array([[.2, .5]])):
            with self.subTest(p=p), self.assertRaises(ValueError):
                classification_metrics(np.array([[0.], [1.]]), p, loss=.5)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = dict(samples=120, hidden=5, epochs=35, patience=12, batch_size=24, seed=13)
        cls.result = run_experiment(**cls.settings)

    def test_restored_network_matches_best_validation_epoch(self):
        result = self.result
        data, network = result["data"], result["network"]
        Xval = standardize(data["X"][data["val_idx"]], data["mean"], data["scale"])
        best_row = min(result["history"], key=lambda row: row["val_loss"])
        self.assertEqual(result["best_epoch"], best_row["epoch"])
        self.assertAlmostEqual(network.loss(Xval, data["y"][data["val_idx"]]), best_row["val_loss"], places=13)
        self.assertAlmostEqual(result["metrics"]["validation"]["loss"], best_row["val_loss"], places=13)
        self.assertEqual(result["stopped_epoch"], result["history"][-1]["epoch"])

    def test_metrics_count_exactly_the_requested_split_rows(self):
        for split, key in (("train", "train_idx"), ("validation", "val_idx"), ("test", "test_idx")):
            metrics = self.result["metrics"][split]
            self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), len(self.result["data"][key]))
            self.assertTrue(all(math.isfinite(metrics[name]) for name in ("loss", "accuracy", "precision", "recall", "f1", "brier")))

    def test_baseline_metrics_match_the_saved_affine_model(self):
        result = self.result
        data, weights = result["data"], result["baseline"]["parameters"]
        targets = data["y"][data["test_idx"], 0]
        probabilities, losses = [], []
        for raw, target in zip(data["X"][data["test_idx"]], targets):
            x0 = (float(raw[0]) - float(data["mean"][0, 0])) / float(data["scale"][0, 0])
            x1 = (float(raw[1]) - float(data["mean"][0, 1])) / float(data["scale"][0, 1])
            logit = x0 * weights["W"][0][0] + x1 * weights["W"][1][0] + weights["b"][0][0]
            probability = 1 / (1 + math.exp(-logit))
            probabilities.append(probability)
            losses.append(-math.log(probability if target else 1 - probability))
        baseline = result["baseline"]["test"]
        expected_accuracy = sum((p >= .5) == bool(y) for p, y in zip(probabilities, targets)) / len(targets)
        self.assertAlmostEqual(baseline["loss"], sum(losses) / len(losses), places=13)
        self.assertEqual(baseline["accuracy"], expected_accuracy)

    def test_checkpoints_are_bounded_and_include_restored_best_model(self):
        result = self.result
        checkpoints = result["checkpoints"]
        self.assertLessEqual(len(checkpoints), 61)
        self.assertTrue(any(row["epoch"] == result["best_epoch"] for row in checkpoints))
        best = next(row for row in checkpoints if row["epoch"] == result["best_epoch"])
        for name, parameter in result["network"].parameters.items():
            np.testing.assert_array_equal(best["parameters"][name], parameter)

    def test_repeating_seed_reproduces_training_and_baseline(self):
        other = run_experiment(**self.settings)
        self.assertEqual(other["history"], self.result["history"])
        self.assertEqual(other["metrics"], self.result["metrics"])
        for name in other["network"].parameters:
            np.testing.assert_array_equal(other["network"].parameters[name], self.result["network"].parameters[name])
        self.assertEqual(other["baseline"]["parameters"], self.result["baseline"]["parameters"])

    def test_patience_uses_significant_improvement_without_losing_actual_best_weights(self):
        # An impossible improvement threshold makes stopping predictable while
        # small genuine decreases must still be eligible as the best checkpoint.
        result = run_experiment(samples=120, hidden=5, seed=13, epochs=30,
                                learning_rate=1e-8, batch_size=24,
                                patience=3, min_delta=10.)
        self.assertEqual(result["stopped_epoch"], 3)
        best = min(result["history"], key=lambda row: row["val_loss"])
        self.assertEqual(result["best_epoch"], best["epoch"])
        self.assertAlmostEqual(result["metrics"]["validation"]["loss"], best["val_loss"], places=13)

    def test_changing_only_fixed_test_rows_cannot_change_training_or_selection(self):
        original = self.result
        data = original["data"]
        X, y = data["X"].copy(), data["y"].copy()
        X[data["test_idx"]] = X[data["test_idx"]] * 13 + 20
        y[data["test_idx"]] = 1 - y[data["test_idx"]]
        fixed = {key: data[key].copy() for key in ("train_idx", "val_idx", "test_idx")}
        with patch("neural_math_lab.experiment.make_dataset", return_value=(X, y)), \
                patch("neural_math_lab.experiment.stratified_split", return_value=fixed):
            changed = run_experiment(**self.settings)
        self.assertEqual(changed["history"], original["history"])
        self.assertEqual(changed["best_epoch"], original["best_epoch"])
        self.assertEqual(changed["baseline"]["parameters"], original["baseline"]["parameters"])
        self.assertEqual(changed["baseline"]["best_epoch"], original["baseline"]["best_epoch"])
        for key in ("mean", "scale"):
            np.testing.assert_array_equal(changed["data"][key], data[key])
        for name in original["network"].parameters:
            np.testing.assert_array_equal(changed["network"].parameters[name], original["network"].parameters[name])
        self.assertNotEqual(changed["metrics"]["test"]["loss"], original["metrics"]["test"]["loss"])

    def test_both_datasets_learn_useful_held_out_structure(self):
        for dataset in ("moons", "circles"):
            with self.subTest(dataset=dataset):
                result = run_experiment(dataset=dataset, samples=300, noise=.1, hidden=12,
                                        epochs=250, patience=60, batch_size=32, seed=7)
                self.assertGreaterEqual(result["metrics"]["test"]["accuracy"], .75)
                self.assertLess(result["metrics"]["train"]["loss"], result["history"][0]["train_loss"])
                self.assertLessEqual(len(result["checkpoints"]), 61)

    def test_invalid_settings_fail_before_training(self):
        cases = [{"epochs": 0}, {"epochs": True}, {"learning_rate": 0},
                 {"optimizer": "unknown"}, {"batch_size": 0}, {"batch_size": 1.5},
                 {"l2": -.1}, {"l2": np.nan}, {"patience": 0}, {"min_delta": -.01}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(samples=120, **settings)


if __name__ == "__main__":
    unittest.main()
