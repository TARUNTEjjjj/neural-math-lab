"""Independent checks for the maths, numerical behaviour, and training API.

Run with ``python -m unittest discover -s tests -v`` from the project root.
The finite-difference oracle below deliberately does not call gradient_check.
"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network, gradient_check, train, xor_data


class NetworkMathTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.2, -0.4], [0.7, 0.3], [-0.1, 0.8]])
        self.y = np.array([[0.0], [1.0], [1.0]])

    def test_forward_matches_hand_calculated_values(self):
        net = Network(hidden=2, seed=7)
        a = math.log(3.0) / 2.0  # tanh(a) = 1/2 exactly in real arithmetic.
        net.parameters["W1"][:] = [[a, -a], [-a, a]]
        net.parameters["b1"][:] = 0
        net.parameters["W2"][:] = [[1], [-1]]
        net.parameters["b2"][:] = 0
        result = net.forward(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        np.testing.assert_allclose(result["h"], [[0.5, -0.5], [-0.5, 0.5], [0, 0]], atol=1e-14)
        np.testing.assert_allclose(result["z2"], [[1], [-1], [0]], atol=1e-14)
        np.testing.assert_allclose(
            result["p"], [[math.e / (1 + math.e)], [1 / (1 + math.e)], [0.5]], atol=1e-14
        )

    def test_zero_logits_have_log_two_mean_loss(self):
        net = Network(hidden=3, seed=2)
        net.parameters["W2"][:] = 0
        net.parameters["b2"][:] = 0
        self.assertAlmostEqual(net.loss(self.X, self.y), math.log(2), places=14)

    def test_all_gradients_match_independent_central_differences(self):
        for size in (1, 3):
            with self.subTest(batch_size=size):
                net = Network(hidden=3, seed=11)
                X, y = self.X[:size], self.y[:size]
                analytic = net.gradients(X, y)
                epsilon = 1e-6
                for name, parameter in net.parameters.items():
                    numeric = np.zeros_like(parameter)
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        try:
                            parameter[index] = original + epsilon
                            plus = net.loss(X, y)
                            parameter[index] = original - epsilon
                            minus = net.loss(X, y)
                            numeric[index] = (plus - minus) / (2 * epsilon)
                        finally:
                            parameter[index] = original
                    np.testing.assert_allclose(analytic[name], numeric, atol=2e-8, rtol=2e-6, err_msg=name)

    def test_mean_loss_and_gradients_do_not_scale_with_duplicate_examples(self):
        net = Network(hidden=3, seed=11)
        doubled_X = np.concatenate([self.X, self.X])
        doubled_y = np.concatenate([self.y, self.y])
        self.assertAlmostEqual(net.loss(self.X, self.y), net.loss(doubled_X, doubled_y), places=14)
        expected = net.gradients(self.X, self.y)
        actual = net.gradients(doubled_X, doubled_y)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-14)

    def test_mean_loss_and_gradients_are_invariant_to_example_order(self):
        net = Network(hidden=3, seed=11)
        order = [2, 0, 1]
        self.assertAlmostEqual(net.loss(self.X, self.y), net.loss(self.X[order], self.y[order]), places=14)
        expected = net.gradients(self.X, self.y)
        actual = net.gradients(self.X[order], self.y[order])
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-14)

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        net = Network(hidden=2, seed=7)
        net.parameters["W1"][:] = 0
        net.parameters["W2"][:] = 0
        X = np.zeros((1, 2))
        for logit, target, expected_loss in ((1000.0, 0.0, 1000.0), (-1000.0, 1.0, 1000.0),
                                              (1000.0, 1.0, 0.0), (-1000.0, 0.0, 0.0)):
            with self.subTest(logit=logit, target=target):
                net.parameters["b2"][:] = logit
                with np.errstate(all="raise"):
                    loss = net.loss(X, np.array([[target]]))
                    gradients = net.gradients(X, np.array([[target]]))
                    p = net.predict_proba(X)
                self.assertAlmostEqual(loss, expected_loss, places=12)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(all(np.isfinite(g).all() for g in gradients.values()))

    def test_saturated_hidden_units_retain_representable_gradients(self):
        net = Network(hidden=1)
        net.parameters = {"W1": np.array([[0.], [0.]]), "b1": np.array([[20.]]),
                          "W2": np.array([[1.]]), "b2": np.array([[0.]])}
        for sign in (1, -1):
            net.parameters["b1"][:] = sign * 20
            with np.errstate(all="raise"):
                actual = net.gradients([[1., 0.]], [[0.]])["W1"][0, 0]
            expected = (1 / (1 + math.exp(-sign))) / math.cosh(20) ** 2
            self.assertGreater(actual, 0)
            self.assertAlmostEqual(actual / expected, 1, places=14)

    def test_small_gradient_step_reduces_loss(self):
        net = Network(hidden=3, seed=11)
        before = net.loss(self.X, self.y)
        net.step(net.gradients(self.X, self.y), learning_rate=0.01)
        self.assertLess(net.loss(self.X, self.y), before)

    def test_correct_saturated_logits_retain_representable_loss_and_gradient(self):
        net = Network(hidden=1, seed=7)
        net.parameters["W1"][:] = 0
        net.parameters["b1"][:] = .5
        net.parameters["W2"][:] = 1
        X = np.array([[1., -2.]])
        hidden = math.tanh(.5)
        for logit, target in ((40., 1.), (100., 1.), (-40., 0.)):
            net.parameters["b2"][:] = logit - hidden
            y = np.array([[target]])
            tail = math.exp(-abs(logit))
            residual = (1 if target == 0 else -1) * tail / (1 + tail)
            expected = {"b2": [[residual]], "W2": [[hidden * residual]],
                        "b1": [[(1 - hidden ** 2) * residual]],
                        "W1": [[(1 - hidden ** 2) * residual],
                               [-2 * (1 - hidden ** 2) * residual]]}
            with self.subTest(logit=logit), np.errstate(all="raise"):
                np.testing.assert_allclose(net.loss(X, y), math.log1p(tail), rtol=1e-14, atol=0)
                for name, actual in net.gradients(X, y).items():
                    np.testing.assert_allclose(actual, expected[name], rtol=1e-14, atol=0)

    def test_gradient_checker_restores_every_parameter(self):
        net = Network(hidden=3, seed=11)
        before = {name: p.copy() for name, p in net.parameters.items()}
        result = gradient_check(net, self.X, self.y)
        self.assertLess(result["max_abs_error"], 1e-8)
        self.assertTrue(math.isfinite(result["max_relative_error"]))
        for name in before:
            np.testing.assert_array_equal(net.parameters[name], before[name])

    def test_gradient_checker_restores_state_after_a_failed_loss_evaluation(self):
        net = Network(hidden=3, seed=11)
        parameters = net.parameters
        originals = dict(parameters)
        before = {name: p.copy() for name, p in parameters.items()}
        actual_loss = net.loss
        calls = 0

        def interrupted_loss(X, y):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated failure during a perturbed evaluation")
            return actual_loss(X, y)

        with patch.object(net, "loss", side_effect=interrupted_loss):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                gradient_check(net, self.X, self.y)
        self.assertIs(net.parameters, parameters)
        for name in before:
            self.assertIs(net.parameters[name], originals[name])
            np.testing.assert_array_equal(net.parameters[name], before[name])

    def test_same_seed_reproduces_parameters_without_using_global_rng(self):
        np.random.seed(123)
        expected_global_draw = np.random.random(3)
        np.random.seed(123)
        first = Network(hidden=4, seed=7)
        actual_global_draw = np.random.random(3)
        second = Network(hidden=4, seed=7)
        np.testing.assert_array_equal(actual_global_draw, expected_global_draw)
        for name in first.parameters:
            np.testing.assert_array_equal(first.parameters[name], second.parameters[name])
        other = Network(hidden=4, seed=8)
        self.assertFalse(np.array_equal(first.parameters["W1"], other.parameters["W1"]))

    def test_zero_input_rows_are_valid(self):
        net = Network(hidden=2, seed=7)
        p = net.predict_proba(np.zeros((2, 2)))
        self.assertEqual(p.shape, (2, 1))
        self.assertTrue(np.isfinite(p).all())

    def test_large_prediction_grid_matches_scalar_calculations(self):
        # Some Apple Accelerate builds raise false FP flags on large GEMM calls.
        # Compare against independent scalar arithmetic, not another matrix multiply.
        net = Network(hidden=4, seed=7)
        axis = np.linspace(-0.25, 1.25, 121)
        xx, yy = np.meshgrid(axis, axis)
        X = np.column_stack([xx.ravel(), yy.ravel()])
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            actual = net.predict_proba(X)
        self.assertEqual(actual.shape, (14641, 1))
        self.assertTrue(np.isfinite(actual).all())
        w = net.parameters
        for index in (0, 1, 120, 7300, 14640):
            row = X[index]
            hidden = [math.tanh(sum(float(row[i]) * float(w["W1"][i, j]) for i in range(2))
                                + float(w["b1"][0, j])) for j in range(4)]
            logit = sum(hidden[j] * float(w["W2"][j, 0]) for j in range(4)) + float(w["b2"][0, 0])
            expected = 1.0 / (1.0 + math.exp(-logit))
            self.assertAlmostEqual(float(actual[index, 0]), expected, places=13)

    def test_actual_matrix_overflow_is_still_rejected(self):
        net = Network(hidden=2, seed=7)
        net.parameters["W1"][:] = 1e308
        with self.assertRaisesRegex(ValueError, "non-finite|overflow"):
            net.predict_proba(np.full((3, 2), 2.0))

    def test_saturated_sigmoid_remains_finite_under_strict_numpy_errors(self):
        net = Network(hidden=2, seed=7)
        net.parameters["W1"][:] = 0
        net.parameters["W2"][:] = 0
        for logit, probability in ((1000.0, 1.0), (-1000.0, 0.0)):
            with self.subTest(logit=logit):
                net.parameters["b2"][:] = logit
                with np.errstate(all="raise"):
                    actual = net.predict_proba(np.zeros((1, 2)))
                self.assertEqual(actual[0, 0], probability)


class TrainingTests(unittest.TestCase):
    def test_xor_truth_table_and_independent_returns(self):
        X, y = xor_data()
        np.testing.assert_array_equal(X, [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(y, [[0], [1], [1], [0]])
        X[:] = -1
        y[:] = -1
        fresh_X, fresh_y = xor_data()
        self.assertTrue((fresh_X >= 0).all())
        self.assertTrue((fresh_y >= 0).all())

    def test_training_history_includes_initial_and_final_metrics(self):
        X, y = xor_data()
        net = Network(seed=7)
        initial = net.loss(X, y)
        history = train(net, X, y, epochs=12, learning_rate=0.5)
        self.assertEqual(len(history), 13)
        self.assertEqual([r["epoch"] for r in history], list(range(13)))
        self.assertAlmostEqual(history[0]["loss"], initial, places=14)
        self.assertAlmostEqual(history[-1]["loss"], net.loss(X, y), places=14)
        accuracy = float(np.mean((net.predict_proba(X) >= 0.5) == y))
        self.assertEqual(history[-1]["accuracy"], accuracy)
        self.assertTrue(all(math.isfinite(r["loss"]) and 0 <= r["accuracy"] <= 1 for r in history))

    def test_training_is_reproducible_for_a_fixed_seed(self):
        X, y = xor_data()
        a, b = Network(seed=7), Network(seed=7)
        history_a = train(a, X, y, epochs=100)
        history_b = train(b, X, y, epochs=100)
        self.assertEqual(history_a, history_b)
        for name in a.parameters:
            np.testing.assert_array_equal(a.parameters[name], b.parameters[name])

    def test_xor_is_learned_for_documented_fixed_seeds(self):
        # These are training-set checks on all four truth-table rows, not a
        # claim about generalisation, every random seed, or every dataset.
        X, y = xor_data()
        for seed in (0, 7, 42):
            with self.subTest(seed=seed):
                net = Network(hidden=4, seed=seed)
                history = train(net, X, y, epochs=4000, learning_rate=0.5)
                np.testing.assert_array_equal(net.predict_proba(X) >= 0.5, y.astype(bool))
                self.assertLess(history[-1]["loss"], 0.02)
                self.assertLess(history[-1]["loss"], history[0]["loss"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.net = Network(hidden=3, seed=11)
        self.X, self.y = xor_data()

    def test_invalid_hidden_sizes_rejected(self):
        for hidden in (0, -1, 65, 2.5, True):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                Network(hidden=hidden)

    def test_invalid_seeds_rejected(self):
        for seed in (-1, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Network(seed=seed)

    def test_empty_or_misshaped_inputs_rejected(self):
        for X in (np.empty((0, 2)), np.ones((3, 1)), np.ones((3, 3)), np.ones(2), np.ones((2, 2, 1))):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                self.net.predict_proba(X)

    def test_nonfinite_inputs_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            X = self.X.copy()
            X[0, 0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.net.predict_proba(X)

    def test_misshaped_targets_rejected(self):
        for y in (np.ones(4), np.ones((3, 1)), np.ones((4, 2)), np.ones((4, 1, 1))):
            with self.subTest(shape=y.shape), self.assertRaises(ValueError):
                self.net.loss(self.X, y)

    def test_nonbinary_or_nonfinite_targets_rejected(self):
        for bad in (-1, 2, 0.5, np.nan, np.inf):
            y = self.y.copy()
            y[0, 0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.net.gradients(self.X, y)

    def test_invalid_learning_rates_rejected(self):
        gradients = self.net.gradients(self.X, self.y)
        for rate in (0, -0.1, np.nan, np.inf, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.net.step(gradients, learning_rate=rate)

    def test_integer_learning_rate_too_large_for_float_has_clear_error(self):
        gradients = self.net.gradients(self.X, self.y)
        before = {name: value.copy() for name, value in self.net.parameters.items()}
        with self.assertRaisesRegex(ValueError, "learning_rate must be a positive finite number"):
            self.net.step(gradients, learning_rate=10 ** 1000)
        for name, expected in before.items():
            np.testing.assert_array_equal(self.net.parameters[name], expected)

    def test_invalid_epoch_counts_rejected(self):
        for epochs in (0, -1, 1.5, True, 100001):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                train(self.net, self.X, self.y, epochs=epochs)

    def test_rejected_gradient_update_does_not_partially_change_weights(self):
        before = {name: p.copy() for name, p in self.net.parameters.items()}
        gradients = self.net.gradients(self.X, self.y)
        gradients["b2"][:] = np.nan  # Last parameter makes partial-update bugs visible.
        with self.assertRaises(ValueError):
            self.net.step(gradients, learning_rate=0.5)
        for name in before:
            np.testing.assert_array_equal(self.net.parameters[name], before[name])

    def test_misshaped_and_missing_gradients_rejected(self):
        gradients = self.net.gradients(self.X, self.y)
        gradients["W2"] = np.ones(3)
        with self.assertRaises(ValueError):
            self.net.step(gradients, learning_rate=0.5)
        gradients = self.net.gradients(self.X, self.y)
        del gradients["b2"]
        with self.assertRaises(ValueError):
            self.net.step(gradients, learning_rate=0.5)

    def test_overflowing_update_is_rejected_without_changing_parameters(self):
        before = {name: p.copy() for name, p in self.net.parameters.items()}
        gradients = {name: np.ones_like(p) for name, p in self.net.parameters.items()}
        gradients["b2"][:] = 1e308
        with self.assertRaises(ValueError):
            self.net.step(gradients, learning_rate=1e308)
        for name in before:
            np.testing.assert_array_equal(self.net.parameters[name], before[name])

    def test_gradient_checker_rejects_invalid_step_sizes(self):
        for epsilon in (0, -1e-5, np.nan, np.inf, True):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                gradient_check(self.net, self.X, self.y, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()
