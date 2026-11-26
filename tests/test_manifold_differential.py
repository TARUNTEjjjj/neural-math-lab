import unittest
import numpy as np

from neural_math_lab.core import Network, gradient_check
from neural_math_lab.datasets import standardize


class TestManifoldDifferentialProperties(unittest.TestCase):
    def test_hessian_schwarz_symmetry_and_second_order_consistency(self):
        rng = np.random.default_rng(2026)
        X = rng.normal(size=(12, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(np.float64).reshape(-1, 1)

        net = Network(hidden=3, seed=7)

        def pack(model):
            return np.concatenate([
                model.parameters["W1"].ravel(),
                model.parameters["b1"].ravel(),
                model.parameters["W2"].ravel(),
                model.parameters["b2"].ravel(),
            ])

        def eval_grad(model, theta):
            w1s = model.parameters["W1"].shape
            b1s = model.parameters["b1"].shape
            w2s = model.parameters["W2"].shape
            b2s = model.parameters["b2"].shape
            i1 = model.parameters["W1"].size
            i2 = i1 + model.parameters["b1"].size
            i3 = i2 + model.parameters["W2"].size
            saved = {k: v.copy() for k, v in model.parameters.items()}
            model.parameters = {
                "W1": theta[:i1].reshape(w1s).copy(),
                "b1": theta[i1:i2].reshape(b1s).copy(),
                "W2": theta[i2:i3].reshape(w2s).copy(),
                "b2": theta[i3:].reshape(b2s).copy(),
            }
            try:
                g = model.gradients(X, y)
                return np.concatenate([g["W1"].ravel(), g["b1"].ravel(), g["W2"].ravel(), g["b2"].ravel()])
            finally:
                model.parameters = saved

        theta = pack(net)
        d = theta.size
        eps = 1e-6
        hessian = np.zeros((d, d), dtype=np.float64)
        for k in range(d):
            step = np.zeros_like(theta)
            step[k] = eps
            hessian[:, k] = (eval_grad(net, theta + step) - eval_grad(net, theta - step)) / (2.0 * eps)

        np.testing.assert_allclose(hessian, hessian.T, atol=1e-7)

        # Zero hidden weights produce linear classifier in latent space -> strictly convex (PSD) w.r.t. W2, b2
        tail_idx = net.parameters["W1"].size + net.parameters["b1"].size
        H_out = hessian[tail_idx:, tail_idx:]
        eigvals = np.linalg.eigvalsh(H_out)
        self.assertTrue(np.all(eigvals >= -1e-9), f"Output layer Hessian eigenvalues not PSD: {eigvals}")

    def test_gradient_check_and_affine_standardization_invariance(self):
        rng = np.random.default_rng(11)
        X = rng.normal(scale=1.5, size=(16, 2))
        y = (np.sin(X[:, 0]) + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)
        net = Network(hidden=4, seed=42)
        report = gradient_check(net, X, y, epsilon=1e-6)
        self.assertLess(report["max_abs_error"], 1e-8)
        self.assertLess(report["max_relative_error"], 1e-6)

        mean = np.mean(X, axis=0, keepdims=True)
        scale = np.std(X, axis=0, keepdims=True)
        X_shift = X * np.array([[3.0, -2.5]]) + np.array([[10.0, -4.0]])
        m2 = np.mean(X_shift, axis=0, keepdims=True)
        s2 = np.std(X_shift, axis=0, keepdims=True) * np.array([[1.0, -1.0]])
        Z1 = standardize(X, mean, scale)
        Z2 = standardize(X_shift, m2, np.abs(s2))
        # Flip sign of feature 2 on second transform and inspect invariant symmetry
        Z2[:, 1] *= -1.0
        np.testing.assert_allclose(Z1, Z2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
