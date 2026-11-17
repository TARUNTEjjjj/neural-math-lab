"""Independent checks for the maths, numerical behaviour, and training API.

Run with ``python -m unittest discover -s tests -v`` from the project root.
The finite-difference oracle below deliberately does not call gradient_check.
"""

import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
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

    def test_every_derivative_through_a_saturated_hidden_unit_matches_its_closed_form(self):
        # tanh(+-20) rounds to exactly +-1, so 1 - h**2 would give 0. The true
        # slope 1/cosh(20)**2 is about 1.7e-17, a normal float64 number.
        net = Network(hidden=1, seed=7)
        net.parameters["W1"][:] = 0
        net.parameters["W2"][:] = 1
        net.parameters["b2"][:] = 0
        X, y = np.array([[1., -2.]]), np.array([[1.]])
        for pre_activation in (20., -20.):
            net.parameters["b1"][:] = pre_activation
            hidden = math.tanh(pre_activation)
            slope = 1 / math.cosh(pre_activation) ** 2
            residual = -1 / (1 + math.exp(hidden))  # sigmoid(z2) - 1 at z2 = hidden
            expected = {"b2": [[residual]], "W2": [[hidden * residual]],
                        "b1": [[slope * residual]],
                        "W1": [[slope * residual], [-2 * slope * residual]]}
            with self.subTest(pre_activation=pre_activation), np.errstate(all="raise"):
                self.assertEqual(hidden * hidden, 1.0)
                for name, actual in net.gradients(X, y).items():
                    np.testing.assert_allclose(actual, expected[name], rtol=1e-14, atol=0, err_msg=name)

    def test_hidden_units_beyond_half_the_float64_range_have_zero_slope(self):
        # For a finite pre-activation above about 8.99e307 in size, 2|z1| is
        # not a float64 number, so a slope formula that doubled z1 would
        # overflow. The slope itself is 0 in float64, as 1 - h**2 gives: a valid
        # result, not an overflow error. The other hidden unit is unaffected.
        net = Network(hidden=2, seed=7)
        net.parameters["W1"][:] = [[0., .5], [0., .25]]
        net.parameters["b1"][:] = [[0., .1]]
        net.parameters["W2"][:] = [[3.], [-2.]]
        net.parameters["b2"][:] = .2
        X, y = np.array([[1., -2.]]), np.array([[1.]])
        for weight in (9e307, -1.5e308, np.finfo(np.float64).max):
            net.parameters["W1"][0, 0] = weight  # z1 = [weight, 0.1]
            hidden = [math.copysign(1., weight), math.tanh(.1)]
            residual = -1 / (1 + math.exp(3 * hidden[0] - 2 * hidden[1] + .2))
            dz1 = [0., -2 * residual / math.cosh(.1) ** 2]
            expected = {"b2": [[residual]], "W2": [[hidden[0] * residual], [hidden[1] * residual]],
                        "b1": [dz1], "W1": [dz1, [-2 * dz1[0], -2 * dz1[1]]]}
            for settings in ({}, {"all": "raise"}):
                with self.subTest(weight=weight, errors=settings), np.errstate(**settings):
                    for name, actual in net.gradients(X, y).items():
                        np.testing.assert_allclose(actual, expected[name], rtol=1e-14, atol=0, err_msg=name)

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

    def test_gradient_checker_differences_come_from_perturbed_losses(self):
        # With backpropagation replaced by zeros, every reported error must be the
        # size of an independently computed central difference. A checker that
        # compared backpropagation with itself would report zero instead.
        net = Network(hidden=3, seed=11)
        epsilon = 1e-6
        expected = {}
        for name, parameter in net.parameters.items():
            numeric = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                try:
                    parameter[index] = original + epsilon
                    plus = net.loss(self.X, self.y)
                    parameter[index] = original - epsilon
                    minus = net.loss(self.X, self.y)
                    numeric[index] = (plus - minus) / (2 * epsilon)
                finally:
                    parameter[index] = original
            expected[name] = float(np.max(np.abs(numeric)))
        zeros = {name: np.zeros_like(parameter) for name, parameter in net.parameters.items()}
        with patch.object(net, "gradients", return_value=zeros):
            result = gradient_check(net, self.X, self.y, epsilon=epsilon)
        for name, size in expected.items():
            with self.subTest(parameter=name):
                self.assertGreater(size, 1e-4)
                self.assertAlmostEqual(result["per_parameter"][name]["max_abs_error"], size, delta=1e-12)
                self.assertEqual(result["per_parameter"][name]["max_relative_error"], 1.0)
        self.assertAlmostEqual(result["max_abs_error"], max(expected.values()), delta=1e-12)

    def test_gradient_checker_flags_and_locates_a_wrong_derivative(self):
        net = Network(hidden=3, seed=11)
        correct = net.gradients(self.X, self.y)
        wrong = dict(correct, W1=2 * correct["W1"])
        with patch.object(net, "gradients", return_value=wrong):
            result = gradient_check(net, self.X, self.y)
        errors = result["per_parameter"]
        # A doubled derivative misses the finite difference by the derivative
        # itself, a relative error of |2g - g| / (|2g| + |g|) = 1/3.
        self.assertAlmostEqual(errors["W1"]["max_abs_error"], float(np.max(np.abs(correct["W1"]))), delta=1e-8)
        self.assertAlmostEqual(errors["W1"]["max_relative_error"], 1 / 3, places=6)
        for name in ("b1", "W2", "b2"):
            self.assertLess(errors[name]["max_abs_error"], 1e-8, name)
        self.assertGreater(result["max_abs_error"], 1e-3)

    def test_gradient_checker_treats_vanishing_errors_as_rounding_under_strict_numpy_errors(self):
        # At b2 = 720 the loss and its derivative are about 2e-313. The tiny
        # absolute error divided by the 1e-8 floor underflows (about 5e-312
        # here), which is rounding, not a failed check.
        net = Network(hidden=2, seed=7)
        net.parameters["W1"][:] = 0
        net.parameters["W2"][:] = 0
        net.parameters["b2"][:] = 720.0
        X, y = np.zeros((1, 2)), np.array([[1.0]])
        expected = gradient_check(net, X, y)
        with np.errstate(all="raise"):
            actual = gradient_check(net, X, y)
        self.assertEqual(actual, expected)
        self.assertLess(actual["max_abs_error"], 1e-300)
        self.assertLess(actual["max_relative_error"], 1e-300)

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

    def test_each_update_records_metrics_from_a_single_forward_pass(self):
        X, y = xor_data()
        calls = []
        forward = Network.forward

        def counted(network, inputs):
            calls.append(len(inputs))
            return forward(network, inputs)

        with patch.object(Network, "forward", counted):
            train(Network(seed=7), X, y, epochs=3)
        # One pass per recorded epoch gives its loss and accuracy, and the
        # following update backpropagates from that same pass.
        self.assertEqual(calls, [4] * (3 + 1))

    def test_training_steps_along_a_replaced_gradients_method(self):
        # A learner may replace Network.gradients, for example to see what a
        # wrong derivative does. train() must then step along the replacement
        # instead of backpropagating from its own forward pass. This one
        # returns zeros, so no update may change the network.
        X, y = xor_data()

        def still(network, X, y):
            return {name: np.zeros_like(value) for name, value in network.parameters.items()}

        class Still(Network):
            gradients = still

        attribute = Network(seed=7)
        attribute.gradients = lambda X, y: still(attribute, X, y)
        cases = (("subclass", Still(seed=7), nullcontext()),
                 ("instance attribute", attribute, nullcontext()),
                 ("patched class", Network(seed=7), patch.object(Network, "gradients", still)))
        for case, network, replacement in cases:
            initial = {name: value.copy() for name, value in network.parameters.items()}
            with self.subTest(case=case), replacement:
                history = train(network, X, y, epochs=3)
                self.assertEqual(history, [dict(history[0], epoch=epoch) for epoch in range(4)])
                for name, value in initial.items():
                    np.testing.assert_array_equal(network.parameters[name], value)

    def test_recorded_metrics_come_from_a_replaced_loss_or_predict_proba(self):
        # The history records loss and the accuracy of predict_proba. When a
        # subclass replaces either method, train() must record what the
        # replacement returns, not what its shared forward pass gives.
        X, y = xor_data()
        expected = train(Network(seed=7), X, y, epochs=3)

        class Shifted(Network):
            def loss(self, X, y):
                return super().loss(X, y) + 1.0

        class Exact(Network):
            def predict_proba(self, X):
                # The XOR truth table itself, whatever the weights.
                return np.abs(X[:, :1] - X[:, 1:])

        self.assertEqual(train(Shifted(seed=7), X, y, epochs=3),
                         [dict(row, loss=row["loss"] + 1.0) for row in expected])
        self.assertEqual(train(Exact(seed=7), X, y, epochs=3),
                         [dict(row, accuracy=1.0) for row in expected])

    def test_training_needs_only_the_public_methods(self):
        # train() calls predict_proba, loss, gradients and step. An object that
        # has only these, here bound to a network, trains exactly like it.
        X, y = xor_data()
        network = Network(seed=7)
        public = SimpleNamespace(predict_proba=network.predict_proba, loss=network.loss,
                                 gradients=network.gradients, step=network.step)
        self.assertEqual(train(public, X, y, epochs=3), train(Network(seed=7), X, y, epochs=3))

    def test_training_is_reproducible_for_a_fixed_seed(self):
        X, y = xor_data()
        a, b = Network(seed=7), Network(seed=7)
        history_a = train(a, X, y, epochs=100)
        history_b = train(b, X, y, epochs=100)
        self.assertEqual(history_a, history_b)
        for name in a.parameters:
            np.testing.assert_array_equal(a.parameters[name], b.parameters[name])

    def test_vanishing_updates_are_rounding_not_overflow_under_strict_numpy_errors(self):
        # A correctly classified saturated logit has a tiny representable
        # derivative (about 2e-313 here). Its update underflows: it rounds to a
        # subnormal step or to no change. A smaller rate makes it smaller still.
        X, y = np.zeros((1, 2)), np.array([[1.0]])

        def saturated():
            net = Network(hidden=2, seed=7)
            net.parameters["W1"][:] = 0
            net.parameters["W2"][:] = 0
            net.parameters["b2"][:] = 720.0
            return net

        for rate in (0.5, 0.05):
            with self.subTest(learning_rate=rate):
                expected = train(saturated(), X, y, epochs=2, learning_rate=rate)
                net = saturated()
                with np.errstate(all="raise"):
                    history = train(net, X, y, epochs=2, learning_rate=rate)
                self.assertEqual(history, expected)
                self.assertGreater(history[-1]["loss"], 0)
                self.assertEqual(net.parameters["b2"][0, 0], 720.0)
        net = Network(hidden=3, seed=11)
        X, y = xor_data()
        gradients = net.gradients(X, y)
        expected = {name: value - 1e-310 * gradients[name] for name, value in net.parameters.items()}
        with np.errstate(all="raise"):
            net.step(gradients, learning_rate=1e-310)
        for name in expected:
            np.testing.assert_array_equal(net.parameters[name], expected[name])
        # Some tanh implementations flag a subnormal argument as underflow.
        net.parameters["W1"][:] = 0
        net.parameters["b1"][:] = 1e-310
        expected = net.forward(X)
        with np.errstate(all="raise"):
            actual = net.forward(X)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

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
        for epsilon in (0, -1e-5, np.nan, np.inf, True, 1e308):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                gradient_check(self.net, self.X, self.y, epsilon=epsilon)

    def test_gradient_checker_rejects_unrepresentable_steps_and_preserves_state(self):
        # This used to compare a parameter with itself, reporting a derivative
        # disagreement even though the finite-difference experiment never ran.
        parameters = self.net.parameters
        originals = dict(parameters)
        before = {name: value.copy() for name, value in parameters.items()}
        with self.assertRaisesRegex(ValueError, "too small to perturb"):
            gradient_check(self.net, self.X, self.y, epsilon=1e-30)
        self.assertIs(self.net.parameters, parameters)
        for name in before:
            self.assertIs(self.net.parameters[name], originals[name])
            np.testing.assert_array_equal(self.net.parameters[name], before[name])

    def test_gradient_checker_rejects_nonfinite_loss_differences_and_preserves_state(self):
        originals = dict(self.net.parameters)
        with patch.object(self.net, "loss", side_effect=[1e308, -1e308]):
            with self.assertRaisesRegex(ValueError, "non-finite derivative"):
                gradient_check(self.net, self.X, self.y)
        for name, value in originals.items():
            self.assertIs(self.net.parameters[name], value)


if __name__ == "__main__":
    unittest.main()
