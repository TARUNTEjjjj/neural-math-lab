"""A two-layer neural network whose derivatives are written out by hand.

Rows are examples. We use float64 to make numerical gradient checks useful,
not because every machine-learning application needs double precision.
"""

from numbers import Integral, Real
from typing import Dict, List, Tuple

import numpy as np


PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


def _integer(value: int, name: str, minimum: int, maximum=None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer".format(name))
    if value < minimum or (maximum is not None and value > maximum):
        limits = "at least {}".format(minimum)
        if maximum is not None:
            limits = "between {} and {}".format(minimum, maximum)
        raise ValueError("{} must be {}".format(name, limits))
    return int(value)


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("{} must be a positive finite number".format(name))
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("{} must be a positive finite number".format(name)) from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError("{} must be a positive finite number".format(name))
    return result


def _matrix(value, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError("{} must contain real numbers".format(name))
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite numeric matrix".format(name)) from error
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite two-dimensional matrix".format(name))
    return result


def _inputs(value) -> np.ndarray:
    result = _matrix(value, "X")
    if result.shape[0] == 0 or result.shape[1] != 2:
        raise ValueError("X must have shape (number of examples, 2), with at least one example")
    return result


def _targets(value, rows: int) -> np.ndarray:
    result = _matrix(value, "y")
    if result.shape != (rows, 1):
        raise ValueError("y must have shape (number of examples, 1), matching X")
    if not np.all((result == 0) | (result == 1)):
        raise ValueError("y must contain only binary targets 0 or 1")
    return result


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Equivalent branches avoid evaluating exp(1000) for large logits."""
    result = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    # Very small exponentials legitimately round to zero, even when a caller
    # has enabled strict NumPy errors. This is saturation, not invalid input.
    with np.errstate(under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)
    return result


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Some BLAS builds (including Apple Accelerate) set floating-point flags
    # even for finite, correctly bounded products. Check the actual result.
    with np.errstate(all="ignore"):
        result = left @ right
    if not np.all(np.isfinite(result)):
        raise ValueError("matrix multiplication produced a non-finite value; use smaller inputs or weights")
    return result


def xor_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return fresh copies of the complete XOR truth table, in binary order."""
    return (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        np.array([[0], [1], [1], [0]], dtype=np.float64),
    )


class Network:
    """Two inputs -> ``hidden`` tanh units -> one sigmoid output.

    ``parameters`` is deliberately exposed so learners can inspect or change
    the weights. Every calculation checks parameter shapes and finite values.
    """

    def __init__(self, hidden: int = 4, seed: int = 7):
        self.hidden = _integer(hidden, "hidden", 1, 64)
        self.seed = _integer(seed, "seed", 0)
        rng = np.random.default_rng(self.seed)
        # Xavier/Glorot uniform: variance = 2 / (fan_in + fan_out).
        first_bound = np.sqrt(6.0 / (2 + self.hidden))
        second_bound = np.sqrt(6.0 / (self.hidden + 1))
        self.parameters = {
            "W1": rng.uniform(-first_bound, first_bound, (2, self.hidden)),
            "b1": np.zeros((1, self.hidden), dtype=np.float64),
            "W2": rng.uniform(-second_bound, second_bound, (self.hidden, 1)),
            "b2": np.zeros((1, 1), dtype=np.float64),
        }

    def _parameters(self) -> Dict[str, np.ndarray]:
        if not isinstance(self.parameters, dict) or set(self.parameters) != set(PARAMETER_NAMES):
            raise ValueError("parameters must contain exactly W1, b1, W2, b2")
        shapes = {"W1": (2, self.hidden), "b1": (1, self.hidden),
                  "W2": (self.hidden, 1), "b2": (1, 1)}
        checked = {}
        for name, shape in shapes.items():
            value = _matrix(self.parameters[name], name)
            if value.shape != shape:
                raise ValueError("{} must have shape {}".format(name, shape))
            checked[name] = value
        return checked

    def forward(self, X) -> Dict[str, np.ndarray]:
        """Return all intermediate quantities, making the calculation inspectable."""
        X = _inputs(X)
        weights = self._parameters()
        try:
            with np.errstate(over="raise", invalid="raise"):
                z1 = _matmul(X, weights["W1"]) + weights["b1"]
                h = np.tanh(z1)
                z2 = _matmul(h, weights["W2"]) + weights["b2"]
                p = _sigmoid(z2)
        except FloatingPointError as error:
            raise ValueError("forward calculation overflowed; use smaller inputs or weights") from error
        if not all(np.all(np.isfinite(a)) for a in (z1, h, z2, p)):
            raise ValueError("forward calculation produced a non-finite value")
        return {"z1": z1, "h": h, "z2": z2, "p": p}

    def predict_proba(self, X) -> np.ndarray:
        """Return sigmoid scores, shape (N, 1); these are not calibrated confidence."""
        return self.forward(X)["p"]

    def loss(self, X, y) -> float:
        """Mean binary cross-entropy, computed from logits without log(0)."""
        cache = self.forward(X)
        y = _targets(y, cache["p"].shape[0])
        z = cache["z2"]
        # log(1 + exp(z)) - y*z equals BCE(sigmoid(z), y).
        # logaddexp also evaluates vanishing exponentials internally. Rounding
        # those to zero is legitimate for saturated logits under strict settings.
        with np.errstate(under="ignore"):
            losses = np.logaddexp(0.0, z) - y * z
            # Divide first so several very large finite losses do not overflow.
            result = float(np.sum(losses / len(y)))
        if not np.isfinite(result):
            raise ValueError("loss calculation produced a non-finite value")
        return result

    def gradients(self, X, y) -> Dict[str, np.ndarray]:
        """Apply the chain rule to the mean loss; no automatic differentiation."""
        X = _inputs(X)
        y = _targets(y, len(X))
        cache = self.forward(X)
        weights = self._parameters()
        try:
            with np.errstate(over="raise", invalid="raise"):
                # Sigmoid and cross-entropy simplify together to (p - y) / N.
                dz2 = (cache["p"] - y) / len(X)
                dW2 = _matmul(cache["h"].T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)
                dh = _matmul(dz2, weights["W2"].T)
                dz1 = dh * (1.0 - cache["h"] ** 2)
                dW1 = _matmul(X.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)
        except FloatingPointError as error:
            raise ValueError("gradient calculation overflowed; use smaller inputs or weights") from error
        result = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        if not all(np.all(np.isfinite(g)) for g in result.values()):
            raise ValueError("gradient calculation produced a non-finite value")
        return result

    def step(self, grads: Dict[str, np.ndarray], learning_rate: float) -> None:
        """Make one descent step, validating every update before changing any weight."""
        rate = _positive(learning_rate, "learning_rate")
        weights = self._parameters()
        if not isinstance(grads, dict) or set(grads) != set(PARAMETER_NAMES):
            raise ValueError("gradients must contain exactly W1, b1, W2, b2")
        updated = {}
        for name in PARAMETER_NAMES:
            gradient = _matrix(grads[name], "gradient " + name)
            if gradient.shape != weights[name].shape:
                raise ValueError("gradient {} must have shape {}".format(name, weights[name].shape))
            try:
                with np.errstate(over="raise", invalid="raise"):
                    updated[name] = weights[name] - rate * gradient
            except FloatingPointError as error:
                raise ValueError("parameter update overflowed; reduce learning_rate") from error
            if not np.all(np.isfinite(updated[name])):
                raise ValueError("parameter update produced a non-finite value")
        self.parameters.update(updated)


def train(network: Network, X, y, epochs: int = 4000,
          learning_rate: float = 0.5) -> List[dict]:
    """Full-batch descent, recording epoch 0 and each completed update.

    Accuracy is measured on the supplied training examples. This function does
    not create a test set or make a claim about generalization.
    """
    count = _integer(epochs, "epochs", 1, 100000)
    rate = _positive(learning_rate, "learning_rate")
    X = _inputs(X)
    y = _targets(y, len(X))
    history = []
    for epoch in range(count + 1):
        p = network.predict_proba(X)
        history.append({"epoch": epoch, "loss": network.loss(X, y),
                        "accuracy": float(np.mean((p >= 0.5) == y))})
        if epoch < count:
            network.step(network.gradients(X, y), rate)
    return history


def gradient_check(network: Network, X, y, epsilon: float = 1e-5) -> dict:
    """Compare hand derivatives with central finite differences, then restore state.

    A check approximates a derivative as (L(theta+eps)-L(theta-eps))/(2*eps).
    Relative errors use max(1e-8, |analytic|+|numeric|) in the denominator;
    inspect absolute errors too when the derivative is close to zero.
    """
    epsilon = _positive(epsilon, "epsilon")
    X = _inputs(X)
    y = _targets(y, len(X))
    analytic = network.gradients(X, y)
    originals = dict(network.parameters)
    # Perturb private float64 copies: preserve original arrays and dictionary identity.
    temporary = {name: value.copy() for name, value in network._parameters().items()}
    per_parameter = {}
    try:
        network.parameters.update(temporary)
        for name in PARAMETER_NAMES:
            numeric = np.empty_like(temporary[name])
            for index in np.ndindex(temporary[name].shape):
                value = temporary[name][index]
                try:
                    temporary[name][index] = value + epsilon
                    plus = network.loss(X, y)
                    temporary[name][index] = value - epsilon
                    minus = network.loss(X, y)
                    numeric[index] = (plus - minus) / (2.0 * epsilon)
                finally:
                    temporary[name][index] = value
            absolute = np.abs(analytic[name] - numeric)
            scale = np.maximum(1e-8, np.abs(analytic[name]) + np.abs(numeric))
            per_parameter[name] = {"max_abs_error": float(np.max(absolute)),
                                   "max_relative_error": float(np.max(absolute / scale))}
    finally:
        network.parameters.update(originals)
    return {"per_parameter": per_parameter,
            "max_abs_error": max(p["max_abs_error"] for p in per_parameter.values()),
            "max_relative_error": max(p["max_relative_error"] for p in per_parameter.values())}
