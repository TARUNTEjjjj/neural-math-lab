"""Checks that data preparation keeps evaluation rows out of fitted statistics."""

import math
import unittest
from fractions import Fraction

import numpy as np

from neural_math_lab.datasets import (
    fit_standardizer,
    make_dataset,
    standardize,
    stratified_split,
)


class SyntheticDataTests(unittest.TestCase):
    def test_generators_return_finite_balanced_binary_examples(self):
        for dataset in ("moons", "circles"):
            for samples in (120, 601):
                with self.subTest(dataset=dataset, samples=samples):
                    X, y = make_dataset(dataset=dataset, samples=samples, noise=0.15, seed=7)
                    self.assertEqual(X.shape, (samples, 2))
                    self.assertEqual(y.shape, (samples, 1))
                    self.assertTrue(np.isfinite(X).all())
                    self.assertEqual(set(y.ravel()), {0, 1})
                    self.assertLessEqual(abs(int(y.sum()) - (samples - int(y.sum()))), 1)

    def test_seed_reproduces_data_without_consuming_global_random_state(self):
        np.random.seed(91)
        expected_global = np.random.random(4)
        np.random.seed(91)
        first = make_dataset(seed=7)
        actual_global = np.random.random(4)
        second = make_dataset(seed=7)
        changed = make_dataset(seed=8)
        np.testing.assert_array_equal(actual_global, expected_global)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0], changed[0]))

    def test_noise_free_circles_follow_documented_radii(self):
        X, y = make_dataset(dataset="circles", samples=120, noise=0, seed=7)
        radii = np.sqrt(np.sum(X * X, axis=1))
        np.testing.assert_allclose(radii[y[:, 0] == 0], 1., atol=1e-14)
        np.testing.assert_allclose(radii[y[:, 0] == 1], .5, atol=1e-14)

    def test_invalid_generator_settings_are_rejected(self):
        cases = [
            {"dataset": "unknown"}, {"samples": 119}, {"samples": 5001},
            {"samples": 120.5}, {"samples": True}, {"noise": -0.1},
            {"noise": 0.61}, {"noise": np.nan}, {"noise": np.inf},
            {"seed": -1}, {"seed": 0.5}, {"seed": True},
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_dataset(**settings)


class SplitTests(unittest.TestCase):
    def test_stratified_splits_are_disjoint_and_cover_every_id_once(self):
        for samples in (120, 601):
            _, y = make_dataset(samples=samples, seed=7)
            splits = stratified_split(y, seed=9)
            self.assertEqual(set(splits), {"train_idx", "val_idx", "test_idx"})
            all_ids = np.concatenate(list(splits.values()))
            np.testing.assert_array_equal(np.sort(all_ids), np.arange(samples))
            for key, fraction in (("train_idx", .6), ("val_idx", .2), ("test_idx", .2)):
                with self.subTest(samples=samples, split=key):
                    ids = np.asarray(splits[key])
                    self.assertEqual(len(ids), len(np.unique(ids)))
                    self.assertTrue(np.issubdtype(ids.dtype, np.integer))
                    self.assertEqual(set(y[ids].ravel()), {0, 1})
                    self.assertLessEqual(abs(len(ids) - samples * fraction), 2)
                    for label in (0, 1):
                        count = int((y[ids] == label).sum())
                        self.assertLessEqual(abs(count - int((y == label).sum()) * fraction), 1)

    def test_split_seed_reproduces_ids_and_other_seed_changes_partition(self):
        _, y = make_dataset(samples=600)
        first = stratified_split(y, seed=7)
        second = stratified_split(y, seed=7)
        changed = stratified_split(y, seed=8)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.array_equal(first["train_idx"], changed["train_idx"]))

    def test_invalid_split_labels_cannot_create_an_evaluation_set(self):
        for labels in (np.zeros((20, 1)), np.ones((4, 1)), np.full((20, 1), 2),
                       np.full((20, 1), np.nan), np.zeros((20, 2))):
            with self.subTest(shape=labels.shape), self.assertRaises(ValueError):
                stratified_split(labels)


class StandardizationTests(unittest.TestCase):
    def test_mean_and_population_scale_match_hand_calculation(self):
        training = np.array([[10., 100.], [12., 100.], [14., 100.]])
        mean, scale = fit_standardizer(training)
        np.testing.assert_allclose(mean, [[12., 100.]], atol=1e-14)
        np.testing.assert_allclose(scale, [[math.sqrt(8. / 3.), 1.]], atol=1e-14)
        transformed = standardize(training, mean, scale)
        np.testing.assert_allclose(transformed[:, 0], [-math.sqrt(1.5), 0, math.sqrt(1.5)], atol=1e-14)
        np.testing.assert_array_equal(transformed[:, 1], [0, 0, 0])

    def test_constant_column_with_rounded_mean_keeps_unit_scale(self):
        training = np.array([[0.1, 1.], [0.1, 2.], [0.1, 3.]])
        mean, scale = fit_standardizer(training)
        self.assertEqual(scale[0, 0], 1.)
        np.testing.assert_allclose(standardize(training, mean, scale)[:, 0], [0, 0, 0], atol=1e-14)
        actual = standardize([[0.2, 2.]], mean, scale)
        np.testing.assert_allclose(actual, [[0.2 - mean[0, 0], 0.]], atol=1e-14)

    @staticmethod
    def exact_moments(column):
        """Population mean and variance of the float64 values, in exact rationals."""
        values = [Fraction(float(value)) for value in column]
        mean = sum(values) / len(values)
        return mean, sum((value - mean) ** 2 for value in values) / len(values)

    def test_large_common_offset_keeps_exact_mean_and_population_scale(self):
        offsets = np.array([0, 0, 0, 0, 2, 2, 4, 8], dtype=np.float64)
        training = np.column_stack((1e16 + offsets, offsets))
        mean, scale = fit_standardizer(training)
        for j in range(2):
            exact_mean, exact_variance = self.exact_moments(training[:, j])
            with self.subTest(column=j):
                self.assertEqual(Fraction(mean[0, j]), exact_mean)
                self.assertEqual(exact_variance, 7)
                self.assertEqual(scale[0, j], math.sqrt(float(exact_variance)))
        self.assertEqual(mean[0, 0], 10000000000000002.)
        np.testing.assert_array_equal(scale[:, 0], scale[:, 1])
        transformed = standardize(training, mean, scale)
        np.testing.assert_allclose(transformed[:, 0], transformed[:, 1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(transformed.mean(axis=0), [0., 0.], rtol=0, atol=1e-15)
        np.testing.assert_allclose(transformed.std(axis=0), [1., 1.], rtol=0, atol=1e-15)

    def test_moments_match_exact_rationals_across_offsets_and_magnitudes(self):
        rng = np.random.default_rng(13)
        for offset in (0., -3.5e9, 1e12, 2.5e15, -1e16, 1e17):
            for spread in (1e-3, 1., 250.):
                training = offset + spread * rng.integers(-40, 41, size=(37, 2)).astype(np.float64)
                if np.any(training.max(axis=0) == training.min(axis=0)):
                    continue
                mean, scale = fit_standardizer(training)
                for j in range(2):
                    exact_mean, exact_variance = self.exact_moments(training[:, j])
                    with self.subTest(offset=offset, spread=spread, column=j):
                        self.assertEqual(mean[0, j], float(exact_mean))
                        expected_scale = math.sqrt(float(exact_variance))
                        self.assertLessEqual(abs(scale[0, j] - expected_scale),
                                             4 * np.spacing(expected_scale))

    def test_mean_is_correctly_rounded_when_the_fsum_remainder_is_inexact(self):
        # The exact sum lies just below a rounding tie. Its one-step fsum
        # remainder, 2**-53 - 2**-200, rounds to 2**-53, which would put
        # total + remainder exactly on the tie and round the mean up.
        column = np.array([1 + 2.0 ** -52, 2.0 ** -53, -2.0 ** -200, 0.])
        training = np.column_stack((column, [0., 1., 2., 3.]))
        mean, _ = fit_standardizer(training)
        exact_mean, _ = self.exact_moments(column)
        self.assertEqual(mean[0, 0], float(exact_mean))
        self.assertEqual(mean[0, 0], (1 + 2.0 ** -52) / 4)

    def test_evaluation_rows_use_frozen_training_statistics(self):
        training = np.array([[1., 2.], [3., 6.]])
        mean, scale = fit_standardizer(training)
        before = (mean.copy(), scale.copy(), training.copy())
        test = np.array([[102., -196.]])
        actual = standardize(test, mean, scale)
        np.testing.assert_allclose(actual, [[100., -100.]])
        np.testing.assert_array_equal(mean, before[0])
        np.testing.assert_array_equal(scale, before[1])
        np.testing.assert_array_equal(training, before[2])

    def test_tiny_coordinates_standardize_identically_under_strict_numpy_settings(self):
        # Squares of tiny deviations and tiny quotients underflow: that is
        # rounding, not coordinates that are too large.
        for training in ([[0., 0.], [1e-170, 1e-170]], [[1., 1e-160], [3., -1e-160]]):
            with self.subTest(training=training):
                mean, scale = fit_standardizer(training)
                expected = standardize(training, mean, scale)
                with np.errstate(all="raise"):
                    strict_mean, strict_scale = fit_standardizer(training)
                    actual = standardize(training, strict_mean, strict_scale)
                np.testing.assert_array_equal(strict_mean, mean)
                np.testing.assert_array_equal(strict_scale, scale)
                np.testing.assert_array_equal(actual, expected)
        with np.errstate(all="raise"):
            actual = standardize([[1e-310, -1e-310]], [[0., 0.]], [[1e10, 1e10]])
        np.testing.assert_array_equal(actual, [[1e-310 / 1e10, -1e-310 / 1e10]])
        for settings in ({}, {"all": "raise"}):
            with self.subTest(settings=settings), np.errstate(**settings):
                with self.assertRaisesRegex(ValueError, "too large to standardize"):
                    fit_standardizer([[1e308, 0.], [-1e308, 1.]])
                with self.assertRaisesRegex(ValueError, "too large to standardize"):
                    standardize([[1e308, 0.]], [[-1e308, 0.]], [[1., 1.]])

    def test_invalid_standardization_inputs_are_rejected(self):
        for X in (np.empty((0, 2)), np.ones((4, 3)), [[np.inf, 1]], [[np.nan, 1]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                fit_standardizer(X)
        for scale in ([[0., 1.]], [[-1., 1.]], [[np.inf, 1.]]):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                standardize([[1., 2.]], [[0., 0.]], scale)


if __name__ == "__main__":
    unittest.main()
