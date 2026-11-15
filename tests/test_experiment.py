"""Independent optimizer arithmetic, evaluation, and data-isolation checks."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.diagnostics import evaluate_binary
from neural_math_lab.experiment import (
    Adam,
    SGD,
    classification_metrics,
    objective_and_gradients,
    run_experiment,
    _fit,
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

    def test_underflowing_sgd_step_is_rounding_not_overflow(self):
        # 1e-10 * 1e-300 is below the normal float64 range: the step underflows.
        parameters = {"w": np.array([[1., 0.]])}
        with np.errstate(all="raise"):
            SGD(learning_rate=1e-10).step(parameters, {"w": np.array([[1e-300, 1e-300]])})
        np.testing.assert_array_equal(parameters["w"], [[1., -(1e-10 * 1e-300)]])

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

    def test_baseline_regularized_gradients_match_independent_finite_differences(self):
        # Several rows expose a wrong 1/N scale and a positive L2 coefficient
        # exposes a missing penalty: a single row with l2=0 hides both.
        baseline = _Logistic()
        baseline.parameters["W"][:] = [[.8], [-1.3]]
        baseline.parameters["b"][:] = .4
        X = np.array([[.2, -.4], [.7, .3], [-.1, .8], [1.5, -.6], [-.9, -1.2]])
        y = np.array([[0.], [1.], [1.], [0.], [1.]])
        strength, epsilon = .17, 1e-6

        def independent_objective():
            # Mean BCE of sigmoid(x . w + b), plus (l2 / 2) |w|^2; b is not penalized.
            w0, w1 = float(baseline.parameters["W"][0, 0]), float(baseline.parameters["W"][1, 0])
            b = float(baseline.parameters["b"][0, 0])
            total = 0.
            for (x0, x1), target in zip(X.tolist(), y[:, 0]):
                logit = x0 * w0 + x1 * w1 + b
                total += math.log1p(math.exp(-logit if target else logit))
            return total / len(X) + strength / 2 * (w0 ** 2 + w1 ** 2)

        objective, gradients = objective_and_gradients(baseline, X, y, strength)
        self.assertAlmostEqual(objective, independent_objective(), places=13)
        for name, parameters in baseline.parameters.items():
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

    def test_vanishing_scores_are_valid_even_under_strict_numpy_settings(self):
        # Squaring a score below about 1.5e-154 underflows; that is rounding.
        y = np.array([[0.], [1.], [1.], [0.]])
        p = np.array([[1e-200], [.7], [.4], [1e-310]])
        with np.errstate(all="raise"):
            metrics = classification_metrics(y, p, loss=.1)
        self.assertAlmostEqual(metrics["brier"], ((1 - .7) ** 2 + (1 - .4) ** 2) / 4, places=15)
        self.assertAlmostEqual(metrics["brier"], evaluate_binary(y, p)["brier"], places=15)

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

    def test_history_rows_measure_the_weights_saved_at_their_epoch(self):
        # Every row's loss, accuracy and gradient norm describe the weights at
        # the end of that epoch, which its checkpoint saves (one per epoch here).
        result, data = self.result, self.result["data"]
        splits = {name: (standardize(data["X"][data[key]], data["mean"], data["scale"]), data["y"][data[key]])
                  for name, key in (("train", "train_idx"), ("val", "val_idx"))}
        self.assertEqual(len(result["checkpoints"]), len(result["history"]))
        for checkpoint in result["checkpoints"]:
            network = Network(hidden=self.settings["hidden"])
            network.parameters = {name: np.array(value) for name, value in checkpoint["parameters"].items()}
            row = result["history"][checkpoint["epoch"]]
            with self.subTest(epoch=checkpoint["epoch"]):
                for name, (X, y) in splits.items():
                    self.assertAlmostEqual(row[name + "_loss"], network.loss(X, y), places=13)
                    self.assertEqual(row[name + "_accuracy"], float(np.mean((network.predict_proba(X) >= .5) == y)))
                _, gradients = objective_and_gradients(network, *splits["train"], l2=.001)
                norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in gradients.values()))
                self.assertAlmostEqual(row["gradient_norm"] / norm, 1, places=12)

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

    def test_diverging_training_names_the_settings_to_reduce(self):
        # Each SGD update scales the weights by 1 - learning_rate * l2 = -999.
        cases = (({"optimizer": "sgd", "learning_rate": 1000., "l2": 1.}, "the learning rate or the L2 coefficient"),
                 ({"optimizer": "adam", "learning_rate": 1e300}, "the learning rate"))
        for settings, advice in cases:
            with self.subTest(**settings):
                with self.assertRaises(ValueError) as caught:
                    run_experiment(samples=120, hidden=5, epochs=60, seed=13, **settings)
                message = str(caught.exception)
                self.assertRegex(message, r"^Training diverged at epoch [1-9][0-9]*: ")
                self.assertTrue(message.endswith("; reduce " + advice), message)
                self.assertNotIn("smaller weights", message)
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_each_epoch_needs_one_forward_pass_per_batch_and_per_split(self):
        # Updates need one pass over each mini-batch. Each epoch then needs one
        # pass over the training rows, which gives the full-batch gradient (for
        # its norm) and the training loss and accuracy, and one over the
        # validation rows. Nothing computed is discarded or recomputed. The
        # baseline's forward pass is its affine score, so count its logits.
        rng = np.random.default_rng(5)
        X_train, X_val = rng.normal(size=(10, 2)), rng.normal(size=(4, 2))
        y_train, y_val = np.array([[0.], [1.]] * 5), np.array([[0.], [1.]] * 2)
        config = {"optimizer": "adam", "learning_rate": .01, "epochs": 3, "batch_size": 4,
                  "l2": .001, "patience": 10, "min_delta": 0.}
        train, validation = len(X_train), len(X_val)
        for model, owner, method in ((Network(hidden=3, seed=5), Network, "forward"),
                                     (_Logistic(), _Logistic, "_logits")):
            rows = []
            original = getattr(owner, method)

            def counted(instance, X, original=original, rows=rows):
                rows.append(len(X))
                return original(instance, X)

            with self.subTest(model=owner.__name__), patch.object(owner, method, counted):
                history = _fit(model, X_train, y_train, X_val, y_val, config, 1, snapshots=False)[0]
                self.assertEqual(len(history), 4)
                self.assertEqual(sum(rows), (train + validation) + 3 * (2 * train + validation))

    def test_replaced_methods_give_the_recorded_metrics_and_gradient_norm(self):
        # Replaced predict_proba, loss or gradients methods are called, so the
        # logged gradient norm describes the derivatives the updates used, and
        # the recorded metrics use the replaced loss and scores. These return
        # zero derivatives (with l2 = 0 no update changes the weights), a loss
        # one higher than the model's own, and the score 1 for every row.
        rng = np.random.default_rng(5)
        X_train, X_val = rng.normal(size=(10, 2)), rng.normal(size=(4, 2))
        y_train, y_val = np.array([[0.], [1.]] * 5), np.array([[0.], [1.]] * 2)
        config = {"optimizer": "adam", "learning_rate": .01, "epochs": 3, "batch_size": 4,
                  "l2": 0., "patience": 10, "min_delta": 0.}
        for owner, make in ((Network, lambda cls: cls(hidden=3, seed=5)), (_Logistic, lambda cls: cls())):
            class Replaced(owner):
                def predict_proba(self, X):
                    return np.ones((len(X), 1))

                def loss(self, X, y):
                    return super().loss(X, y) + 1.0

                def gradients(self, X, y):
                    return {name: np.zeros_like(value) for name, value in self.parameters.items()}

            plain = make(owner)
            row = {"train_loss": plain.loss(X_train, y_train) + 1.0, "val_loss": plain.loss(X_val, y_val) + 1.0,
                   "train_accuracy": .5, "val_accuracy": .5, "gradient_norm": 0.}
            with self.subTest(model=owner.__name__):
                history = _fit(make(Replaced), X_train, y_train, X_val, y_val, config, 1, snapshots=False)[0]
                self.assertEqual(history, [dict(row, epoch=epoch) for epoch in range(4)])

    def test_vanishing_gradients_train_identically_under_strict_numpy_settings(self):
        # A saturated, correctly classified network has derivatives near 2e-313.
        # Their squares in the gradient norm, and SGD's steps, underflow.
        X, y = np.zeros((6, 2)), np.ones((6, 1))
        config = {"optimizer": "sgd", "learning_rate": .5, "epochs": 2, "batch_size": 4,
                  "l2": .001, "patience": 10, "min_delta": 0.}
        results = []
        for settings in ({}, {"all": "raise"}):
            network = Network(hidden=2, seed=7)
            network.parameters["W1"][:] = 0
            network.parameters["W2"][:] = 0
            network.parameters["b2"][:] = 720.
            with np.errstate(**settings):
                history, _, best, stopped, reason = _fit(network, X, y, X, y, config, 3, snapshots=False)
            results.append((history, best, stopped, reason,
                            {name: value.tolist() for name, value in network.parameters.items()}))
        self.assertEqual(results[1], results[0])
        self.assertEqual(results[1][2], 2)
        self.assertGreater(results[1][0][-1]["train_loss"], 0)

    def test_invalid_settings_fail_before_training(self):
        cases = [{"epochs": 0}, {"epochs": True}, {"learning_rate": 0},
                 {"optimizer": "unknown"}, {"batch_size": 0}, {"batch_size": 1.5},
                 {"l2": -.1}, {"l2": np.nan}, {"patience": 0}, {"min_delta": -.01}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(samples=120, **settings)


if __name__ == "__main__":
    unittest.main()
