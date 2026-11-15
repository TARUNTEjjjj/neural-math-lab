"""Independent counts, pairwise ranking, bin arithmetic, and chain-rule checks."""

import json
from decimal import Decimal
import math
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.diagnostics import (
    ABOVE_ONE, evaluate_binary, input_sensitivity, select_threshold,
)
from neural_math_lab.experiment import run_experiment


def column(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


class BinaryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.y, self.p = column([0, 1, 0, 1]), column([.1, .4, .4, .9])

    def test_hand_calculated_ranking_calibration_and_decisions(self):
        actual = evaluate_binary(self.y, self.p, threshold=.4, bins=2, fn_cost=3)
        self.assertEqual(actual["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(actual["tp"], 2)
        self.assertEqual(actual["predicted_positive"], 3)
        self.assertEqual(actual["total_cost"], 1)
        self.assertEqual(actual["mean_cost"], .25)
        self.assertEqual(actual["accuracy"], .75)
        self.assertAlmostEqual(actual["precision"], 2 / 3)
        self.assertEqual(actual["recall"], 1)
        self.assertEqual(actual["specificity"], .5)
        self.assertAlmostEqual(actual["f1"], .8)
        self.assertAlmostEqual(actual["roc_auc"], 7 / 8)
        # Recall jumps 0 -> 1/2 -> 1, with precisions 1 then 2/3.
        self.assertAlmostEqual(actual["average_precision"], 5 / 6)
        self.assertAlmostEqual(actual["brier"], 27 / 200)
        self.assertAlmostEqual(actual["ece"], 1 / 20)
        first, second = actual["reliability_bins"]
        self.assertEqual(first["count"], 3)
        self.assertAlmostEqual(first["mean_probability"], .3)
        self.assertAlmostEqual(first["fraction_positive"], 1 / 3)
        self.assertEqual(second["count"], 1)
        self.assertEqual(second["mean_probability"], .9)
        self.assertEqual(second["fraction_positive"], 1)
        self.assertEqual([row["threshold"] for row in actual["roc"]], [ABOVE_ONE, .9, .4, .1])
        self.assertEqual([(row["fpr"], row["tpr"]) for row in actual["roc"]], [(0, 0), (0, .5), (.5, 1), (1, 1)])
        json.dumps(actual, allow_nan=False)

    def test_auc_matches_independent_positive_negative_pair_count_with_ties(self):
        rng = np.random.default_rng(91)
        for count in range(2, 55):
            labels = rng.integers(0, 2, count)
            labels[:2] = [0, 1]
            # Quantization deliberately makes many ties.
            scores = rng.integers(0, 6, count) / 5
            positives = scores[labels == 1]
            negatives = scores[labels == 0]
            expected = sum(float(a > b) + .5 * float(a == b)
                           for a in positives for b in negatives) / (len(positives) * len(negatives))
            self.assertAlmostEqual(evaluate_binary(column(labels), column(scores))["roc_auc"], expected, places=14)

    def test_all_tied_scores_group_together_and_ap_equals_prevalence(self):
        result = evaluate_binary(column([0, 1, 1, 0, 1]), column([.7] * 5))
        self.assertEqual(len(result["roc"]), 2)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["average_precision"], 3 / 5)
        self.assertEqual(result["precision_recall"][0]["precision"], 1)
        self.assertEqual(result["precision_recall"][0]["recall"], 0)

    def test_one_class_metrics_are_explicitly_undefined_where_needed(self):
        for label in (0, 1):
            with self.subTest(label=label):
                result = evaluate_binary(column([label] * 3), column([0, .5, 1]))
                self.assertIsNone(result["roc_auc"])
                if label:
                    self.assertEqual(result["average_precision"], 1)
                    self.assertTrue(all(row["fpr"] is None for row in result["roc"]))
                else:
                    self.assertIsNone(result["average_precision"])
                    self.assertTrue(all(row["recall"] is None for row in result["precision_recall"]))
                json.dumps(result, allow_nan=False)

    def test_inclusive_boundary_and_empty_bins_have_no_fake_observations(self):
        result = evaluate_binary(column([0, 1, 1, 0]), column([0, .3, .58, 1]), bins=100)
        bins = result["reliability_bins"]
        self.assertEqual([row["count"] for row in bins if row["count"]], [1, 1, 1, 1])
        self.assertEqual([index for index, row in enumerate(bins) if row["count"]], [0, 30, 58, 99])
        self.assertEqual(sum(row["count"] for row in bins), 4)
        self.assertEqual(bins[0]["mean_probability"], 0)
        self.assertEqual(bins[-1]["mean_probability"], 1)
        for row in bins:
            if row["count"] == 0:
                for key in ("mean_probability", "fraction_positive", "absolute_gap"):
                    self.assertIsNone(row[key])

    def test_threshold_equality_is_positive_and_sentinel_is_all_negative(self):
        y, p = column([0, 1, 1]), column([0, .5, 1])
        self.assertEqual(evaluate_binary(y, p, .5)["confusion_matrix"], [[1, 0], [0, 2]])
        self.assertEqual(evaluate_binary(y, p, 0)["predicted_positive"], 3)
        self.assertEqual(evaluate_binary(y, p, ABOVE_ONE)["predicted_positive"], 0)

    def test_permutation_changes_neither_ties_nor_metrics(self):
        expected = evaluate_binary(self.y, self.p, bins=2)
        for ids in ([3, 1, 0, 2], [2, 1, 3, 0]):
            actual = evaluate_binary(self.y[ids], self.p[ids], bins=2)
            self.assertEqual(actual["roc"], expected["roc"])
            self.assertEqual(actual["precision_recall"], expected["precision_recall"])
            for name in ("roc_auc", "average_precision", "brier", "ece"):
                self.assertAlmostEqual(actual[name], expected[name], places=15)

    def test_vanishing_scores_are_valid_even_under_strict_numpy_settings(self):
        with np.errstate(all="raise"):
            result = evaluate_binary(column([0, 1]), column([1e-310, 1]))
        self.assertEqual(result["brier"], 0)
        self.assertTrue(np.isfinite(result["ece"]))

    def test_inputs_are_not_mutated(self):
        y, p = self.y.copy(), self.p.copy()
        evaluate_binary(self.y, self.p)
        select_threshold(self.y, self.p)
        np.testing.assert_array_equal(self.y, y)
        np.testing.assert_array_equal(self.p, p)

    def test_invalid_arrays_are_rejected(self):
        cases = [([], []), ([0, 1], [0, .4]), ([[0], [1]], [[.2, .4]]),
                 ([[0], [2]], [[.2], [.4]]), ([[0]], [[np.nan]]),
                 ([[0]], [[np.inf]]), ([[0]], [[-.1]]), ([[0]], [[1.1]]),
                 ([[0]], [[.1j]]), ([[np.nan]], [[.1]])]
        for y, p in cases:
            for operation in (evaluate_binary, select_threshold):
                with self.subTest(operation=operation.__name__, y=y, p=p), self.assertRaises(ValueError):
                    operation(y, p)

    def test_invalid_threshold_costs_and_bin_counts_are_rejected(self):
        for value in (-1, np.nan, np.inf, True, '0.5', ABOVE_ONE + .1):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                evaluate_binary(self.y, self.p, threshold=value)
        for name in ("fp_cost", "fn_cost"):
            for value in (0, -.1, 1e6 + 1, np.nan, np.inf, True, "1"):
                for operation in (evaluate_binary, select_threshold):
                    with self.subTest(operation=operation.__name__, name=name, value=value), self.assertRaises(ValueError):
                        operation(self.y, self.p, **{name: value})
        for value in (0, 101, 2.5, True, np.nan):
            with self.subTest(bins=value), self.assertRaises(ValueError):
                evaluate_binary(self.y, self.p, bins=value)


class ThresholdTests(unittest.TestCase):
    def test_costs_and_deterministic_ties_on_hand_fixture(self):
        y, p = column([0, 1, 0, 1]), column([.1, .4, .4, .9])
        equal = select_threshold(y, p)
        self.assertEqual(equal["threshold"], .5)
        self.assertEqual(equal["mean_cost"], .25)
        self.assertEqual(equal["candidate_count"], 5)
        expensive_misses = select_threshold(y, p, fn_cost=3)
        self.assertEqual(expensive_misses["threshold"], .4)
        self.assertEqual(expensive_misses["confusion_matrix"], [[1, 1], [0, 2]])

    def test_selection_matches_independent_exhaustive_predictions(self):
        rng = np.random.default_rng(141)
        for _ in range(40):
            labels = rng.integers(0, 2, 20)
            scores = rng.integers(0, 11, 20) / 10
            fp_cost, fn_cost = float(rng.integers(1, 9)), float(rng.integers(1, 9))
            candidates = set(scores.tolist() + [.5, ABOVE_ONE])
            def key(threshold):
                fp = sum(y == 0 and score >= threshold for y, score in zip(labels, scores))
                fn = sum(y == 1 and score < threshold for y, score in zip(labels, scores))
                # Decimal arithmetic is an independent oracle for these decimal
                # fixtures; binary subtraction must not decide symmetric ties.
                return (fp_cost * fp + fn_cost * fn,
                        abs(Decimal(str(threshold)) - Decimal('.5')), -threshold)
            expected = min(candidates, key=key)
            result = select_threshold(column(labels), column(scores), fp_cost, fn_cost)
            self.assertEqual(result["threshold"], expected)
            self.assertEqual(result["total_cost"], key(expected)[0])
            self.assertEqual(result["candidate_count"], len(candidates))

    def test_endpoint_candidates_allow_predicting_none_or_everyone(self):
        all_negative = select_threshold(column([0, 0]), column([1, 1]))
        self.assertEqual(all_negative["threshold"], ABOVE_ONE)
        self.assertEqual(all_negative["total_cost"], 0)
        all_positive = select_threshold(column([1, 1]), column([0, 0]))
        self.assertEqual(all_positive["threshold"], 0)
        self.assertEqual(all_positive["total_cost"], 0)
        json.dumps([all_negative, all_positive], allow_nan=False)

    def test_cost_scaling_preserves_selected_rule(self):
        y, p = column([0, 1, 0, 1]), column([.1, .4, .4, .9])
        first = select_threshold(y, p, 1, 3)
        second = select_threshold(y, p, 100, 300)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(second["mean_cost"], 100 * first["mean_cost"])

    def test_decimal_cost_ties_survive_rescaling_and_row_permutation(self):
        labels, scores = column([1, 0, 0, 0, 1]), column([.5, .5, .5, .5, .9])
        # Three false positives and one false negative have the same cost.
        # Multiplying 0.1 by 3 must not defeat the closest-to-0.5 tie rule.
        for fp_cost, fn_cost in ((1, 3), (.1, .3), (1e-200, 3e-200), (1e5, 3e5)):
            for ids in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]):
                with self.subTest(costs=(fp_cost, fn_cost), ids=ids):
                    result = select_threshold(labels[ids], scores[ids], fp_cost, fn_cost)
                    self.assertEqual(result["threshold"], .5)
                    self.assertEqual((result["fp"], result["fn"]), (3, 0))

    def test_equidistant_decimal_thresholds_prefer_higher_candidate(self):
        # Both candidate thresholds make one mistake and are 0.3 away from 0.5.
        result = select_threshold(column([1, 0, 1]), column([.2, .6, .8]))
        self.assertEqual(result["threshold"], .8)

    def test_cost_ties_measure_distance_at_the_candidate_scores(self):
        # The documented convention (README, docs/decision-mathematics.md):
        # p >= 0.9 and p >= 0.3 both cost 1 and are 0.4 and 0.2 from 0.5.
        # A threshold of 0.53 makes the same validation decisions as 0.9 and
        # is only 0.03 from 0.5, but it is not a candidate.
        y, p = column([1, 0, 1, 0]), column([.9, .52, .3, .1])
        result = select_threshold(y, p)
        self.assertEqual(result["threshold"], .3)
        self.assertEqual((result["fp"], result["fn"], result["total_cost"]), (1, 0, 1.0))
        same_as_top = evaluate_binary(y, p, .53)
        self.assertEqual((same_as_top["fp"], same_as_top["fn"]), (0, 1))
        self.assertEqual(same_as_top["confusion_matrix"], evaluate_binary(y, p, .9)["confusion_matrix"])

    def test_genuinely_different_costs_are_not_rounded_into_ties(self):
        # A small but resolvable advantage still beats proximity to 0.5.
        result = select_threshold(column([1, 0, 0, 0, 1]),
                                  column([.5, .5, .5, .5, .9]), .1, .3 - 1e-12)
        self.assertEqual(result["threshold"], .9)

    def test_zero_cost_beats_tiny_positive_cost_without_absolute_tolerance(self):
        for cost in (1e-200, float(np.nextafter(0., 1.))):
            with self.subTest(cost=cost):
                result = select_threshold(column([0, 1]), column([.6, .8]), cost, cost)
                self.assertEqual(result["threshold"], .8)
                self.assertEqual(result["total_cost"], 0)

    def test_exact_half_remains_closest_despite_adjacent_float_threshold(self):
        adjacent = float(np.nextafter(.5, 1.))
        result = select_threshold(column([0, 1]), column([.1, adjacent]))
        self.assertEqual(result["threshold"], .5)

    def test_roundoff_ties_are_anchored_to_minimum_without_chaining(self):
        # Costs at thresholds .1, .3 and .5 are 2, 2+12eps, 2+24eps.
        # Adjacent pairs are close, but .5 is outside the minimum's tolerance.
        labels, scores = column([0, 1, 0, 1]), column([.4, .3, .2, .1])
        for ids in ([0, 1, 2, 3], [3, 2, 1, 0]):
            result = select_threshold(labels[ids], scores[ids], 1,
                                      1 + 12 * np.finfo(np.float64).eps)
            self.assertEqual(result["threshold"], .3)


class InputSensitivityTests(unittest.TestCase):
    def test_chain_rule_matches_raw_coordinate_central_differences(self):
        network = Network(hidden=5, seed=21)
        raw = np.array([[-1, .5], [.2, -.1], [2, 3]], dtype=float)
        mean, scale = np.array([[.3, -.7]]), np.array([[2.5, .4]])
        before = {key: value.copy() for key, value in network.parameters.items()}
        actual = input_sensitivity(network, raw, mean, scale)
        epsilon = 1e-5
        expected = np.empty_like(raw)
        for row in range(len(raw)):
            for coordinate in range(2):
                plus, minus = raw[row:row + 1].copy(), raw[row:row + 1].copy()
                plus[0, coordinate] += epsilon
                minus[0, coordinate] -= epsilon
                expected[row, coordinate] = (network.predict_proba(standardize(plus, mean, scale))[0, 0]
                    - network.predict_proba(standardize(minus, mean, scale))[0, 0]) / (2 * epsilon)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=2e-10)
        for key in before:
            np.testing.assert_array_equal(network.parameters[key], before[key])

    def test_one_neuron_derivative_includes_inverse_scale(self):
        network = Network(hidden=1)
        network.parameters = {"W1": np.array([[2.], [-3.]]), "b1": np.array([[0.]]),
                              "W2": np.array([[4.]]), "b2": np.array([[0.]])}
        result = input_sensitivity(network, [[5, 7]], [[5, 7]], [[2, 3]])
        # At zero, tanh'=1 and sigmoid'=1/4, so derivative [2,-3]*4/4/[2,3].
        np.testing.assert_array_equal(result, [[1., -1.]])

    def test_saturated_scores_retain_small_input_derivatives(self):
        network = Network(hidden=1)
        network.parameters = {"W1": np.array([[1.], [-2.]]), "b1": np.array([[0.]]),
                              "W2": np.array([[1.]]), "b2": np.array([[40.]])}
        for bias in (40., -40.):
            network.parameters["b2"][:] = bias
            with np.errstate(all="raise"):
                actual = input_sensitivity(network, [[0, 0]], [[0, 0]], [[1, 1]])
            expected = np.exp(-abs(bias)) / (1 + np.exp(-abs(bias))) ** 2
            np.testing.assert_allclose(actual, [[expected, -2 * expected]], rtol=1e-14, atol=0)
        network.parameters["b1"][:] = 20
        network.parameters["b2"][:] = 0
        expected = (np.exp(-1) / (1 + np.exp(-1)) ** 2) / np.cosh(20) ** 2
        np.testing.assert_allclose(input_sensitivity(network, [[0, 0]], [[0, 0]], [[1, 1]]),
                                   [[expected, -2 * expected]], rtol=1e-14, atol=0)

    def test_saturated_units_keep_representable_derivatives(self):
        # sigmoid(40) and tanh(20) round to exactly 1, so p*(1-p) and 1-h**2
        # would return 0 for derivatives that float64 can still represent.
        network = Network(hidden=1)
        network.parameters = {"W1": np.array([[2.], [-1.]]), "b1": np.array([[0.]]),
                              "W2": np.array([[3.]]), "b2": np.array([[0.]])}
        mean, scale = [[1., 2.]], [[2., 4.]]
        for logit in (40., -40.):
            network.parameters["b2"][:] = logit
            # sigmoid(z) * sigmoid(-z), written without 1 - p.
            slope = 1 / ((1 + math.exp(-logit)) * (1 + math.exp(logit)))
            with self.subTest(logit=logit), np.errstate(all="raise"):
                np.testing.assert_allclose(input_sensitivity(network, [[1, 2]], mean, scale),
                                           [[3 * slope, -.75 * slope]], rtol=1e-14, atol=0)
        # tanh(20) rounds to 1, so z2 = 3 * 1 - 3 = 0 and sigmoid' = 1/4.
        network.parameters["b1"][:] = 20
        network.parameters["b2"][:] = -3
        slope = 1 / math.cosh(20.) ** 2
        with np.errstate(all="raise"):
            np.testing.assert_allclose(input_sensitivity(network, [[1, 2]], mean, scale),
                                       [[.75 * slope, -.1875 * slope]], rtol=1e-14, atol=0)

    def test_hidden_units_beyond_half_the_float64_range_have_zero_slope(self):
        # For a finite pre-activation above about 8.99e307 in size, 2|z1| is
        # not a float64 number, so a slope formula that doubled z1 would
        # overflow. That unit's slope is 0 in float64, as 1 - h**2 gives and
        # the browser runtime computes; it is not an overflow error.
        network = Network(hidden=2)
        mean, scale = [[1., 2.]], [[2., 4.]]
        hidden_slope = 1 / math.cosh(.6) ** 2
        for weight in (1e308, -np.finfo(np.float64).max):
            network.parameters = {"W1": np.array([[weight, .5], [0., -1.]]), "b1": np.array([[0., .1]]),
                                  "W2": np.array([[3.], [-2.]]), "b2": np.array([[.2]])}
            # Raw (3, 2) standardizes to (1, 0), so z1 = (weight, 0.6).
            logit = 3 * math.copysign(1., weight) - 2 * math.tanh(.6) + .2
            slope = hidden_slope / ((1 + math.exp(-logit)) * (1 + math.exp(logit)))
            for settings in ({}, {"all": "raise"}):
                with self.subTest(weight=weight, errors=settings), np.errstate(**settings):
                    np.testing.assert_allclose(input_sensitivity(network, [[3, 2]], mean, scale),
                                               [[-.5 * slope, .5 * slope]], rtol=1e-14, atol=0)

    def test_saturated_model_is_finite_with_zero_local_derivative(self):
        network = Network(hidden=1)
        network.parameters["b2"][:] = 1000
        with np.errstate(all="raise"):
            actual = input_sensitivity(network, [[0, 0]], [[0, 0]], [[1, 1]])
        np.testing.assert_array_equal(actual, [[0, 0]])

    def test_invalid_scaler_inputs_and_model_are_rejected(self):
        network = Network(hidden=1)
        cases = [(network, [[np.inf, 0]], [[0, 0]], [[1, 1]]),
                 (network, [[0, 0]], [[0, 0]], [[0, 1]]),
                 (network, [[0, 0]], [[0]], [[1, 1]]),
                 (object(), [[0, 0]], [[0, 0]], [[1, 1]])]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                input_sensitivity(*arguments)


class DiagnosticIsolationTests(unittest.TestCase):
    def test_held_out_label_changes_cannot_affect_weights_or_threshold_selection(self):
        settings = dict(samples=120, epochs=12, patience=5, hidden=4, seed=37,
                        fp_cost=2, fn_cost=5)
        original = run_experiment(**settings)
        data = original["data"]
        changed_y = data["y"].copy()
        changed_y[data["test_idx"]] = 1 - changed_y[data["test_idx"]]
        ids = {key: data[key].copy() for key in ("train_idx", "val_idx", "test_idx")}
        with patch("neural_math_lab.experiment.make_dataset", return_value=(data["X"].copy(), changed_y)), \
                patch("neural_math_lab.experiment.stratified_split", return_value=ids):
            changed = run_experiment(**settings)
        self.assertEqual(changed["history"], original["history"])
        for key, weights in original["network"].parameters.items():
            np.testing.assert_array_equal(changed["network"].parameters[key], weights)
        for model in ("network", "baseline"):
            self.assertEqual(changed["diagnostics"][model]["threshold_selection"], original["diagnostics"][model]["threshold_selection"])
            self.assertEqual(changed["diagnostics"][model]["validation"], original["diagnostics"][model]["validation"])
            self.assertNotEqual(changed["diagnostics"][model]["test"]["confusion_matrix"], original["diagnostics"][model]["test"]["confusion_matrix"])
        self.assertEqual(original["diagnostics"]["selection_split"], "validation")
        json.dumps(original["diagnostics"], allow_nan=False)

    def test_costs_change_only_post_training_decisions(self):
        settings = dict(samples=120, epochs=5, patience=3, hidden=3, seed=10)
        first = run_experiment(**settings)
        changed = run_experiment(**dict(settings, fp_cost=100, fn_cost=1))
        self.assertEqual(first["history"], changed["history"])
        self.assertEqual(first["best_epoch"], changed["best_epoch"])
        self.assertEqual(first["stopped_epoch"], changed["stopped_epoch"])
        for key, value in first["network"].parameters.items():
            np.testing.assert_array_equal(value, changed["network"].parameters[key])
        for key in ("mean", "scale"):
            np.testing.assert_array_equal(first["data"][key], changed["data"][key])
        self.assertEqual(first["metrics"], changed["metrics"])
        self.assertEqual(first["baseline"], changed["baseline"])
        self.assertEqual(changed["diagnostics"]["network"]["threshold_selection"]["fp_cost"], 100)

    def test_stored_validation_scores_reproduce_exact_threshold_counts(self):
        result = run_experiment(samples=120, hidden=4, epochs=7, seed=31)
        ids = result["data"]["val_idx"]
        for model in ("network", "baseline"):
            diagnostic = result["diagnostics"][model]
            records = diagnostic["validation_predictions"]
            self.assertEqual([row["id"] for row in records], ids.tolist())
            self.assertEqual([row["target"] for row in records], result["data"]["y"][ids, 0].tolist())
            actual = evaluate_binary(column([row["target"] for row in records]),
                                     column([row["p"] for row in records]),
                                     diagnostic["threshold_selection"]["threshold"])
            self.assertEqual(actual, diagnostic["validation"])
            if model == "network":
                expected = result["network"].predict_proba(standardize(
                    result["data"]["X"][ids], result["data"]["mean"], result["data"]["scale"]))[:, 0]
                np.testing.assert_array_equal([row["p"] for row in records], expected)

    def test_invalid_costs_fail_before_dataset_generation(self):
        for settings in ({"fp_cost": 0}, {"fn_cost": True}, {"fp_cost": 1e7}):
            with patch("neural_math_lab.experiment.make_dataset") as generator:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    run_experiment(**settings)
                generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
