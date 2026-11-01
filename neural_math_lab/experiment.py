"""Validation-selected neural-network experiments with an untouched test split.

All derivatives and optimizers are explicit NumPy mathematics. A separate
logistic regression provides a simple affine baseline on exactly the same data.
"""

import math

import numpy as np

from .core import Network, _inputs, _integer, _matmul, _matrix, _positive, _sigmoid, _targets
from .datasets import _bounded_real, fit_standardizer, make_dataset, standardize, stratified_split


def _checked_updates(parameters, gradients):
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("parameters must be a nonempty dictionary of matrices")
    if not isinstance(gradients, dict) or set(parameters) != set(gradients):
        raise ValueError("Each parameter needs exactly one matching gradient")
    weights, grads = {}, {}
    for name, value in parameters.items():
        weights[name] = _matrix(value, "parameter " + str(name))
        grads[name] = _matrix(gradients[name], "gradient " + str(name))
        if grads[name].shape != weights[name].shape:
            raise ValueError("Parameter and gradient shapes must match")
    return weights, grads


class SGD:
    """Stochastic gradient descent: theta <- theta - learning_rate * gradient."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = _positive(learning_rate, "learning_rate")

    def step(self, parameters, gradients):
        weights, gradients = _checked_updates(parameters, gradients)
        try:
            with np.errstate(over="raise", invalid="raise"):
                updated = {k: weights[k] - self.learning_rate * gradients[k] for k in weights}
        except FloatingPointError as exc:
            raise ValueError("SGD update overflowed; reduce the learning rate") from exc
        if not all(np.isfinite(v).all() for v in updated.values()):
            raise ValueError("SGD update produced non-finite parameters")
        parameters.update(updated)


class Adam:
    """Adam with first/second moment bias correction at each mini-batch update."""

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.beta1 = _bounded_real(beta1, "beta1", 0, 1)
        self.beta2 = _bounded_real(beta2, "beta2", 0, 1)
        if self.beta1 == 1 or self.beta2 == 1:
            raise ValueError("Adam beta values must be less than one")
        self.epsilon = _positive(epsilon, "epsilon")
        self.m, self.v, self.t = {}, {}, 0

    def step(self, parameters, gradients):
        weights, gradients = _checked_updates(parameters, gradients)
        if self.m and (set(self.m) != set(weights) or any(self.m[k].shape != weights[k].shape for k in weights)):
            raise ValueError("Adam must be used with the same parameter shapes on every update")
        t = self.t + 1
        m, v, updated = {}, {}, {}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                for key, gradient in gradients.items():
                    previous_m = self.m.get(key, np.zeros_like(gradient))
                    previous_v = self.v.get(key, np.zeros_like(gradient))
                    m[key] = self.beta1 * previous_m + (1 - self.beta1) * gradient
                    v[key] = self.beta2 * previous_v + (1 - self.beta2) * gradient ** 2
                    m_hat = m[key] / (1 - self.beta1 ** t)
                    v_hat = v[key] / (1 - self.beta2 ** t)
                    updated[key] = weights[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        except FloatingPointError as exc:
            raise ValueError("Adam update overflowed; use smaller inputs or gradients") from exc
        if not all(np.isfinite(a).all() for mapping in (m, v, updated) for a in mapping.values()):
            raise ValueError("Adam update produced a non-finite value")
        # A rejected update must not change either model weights or moment state.
        parameters.update(updated)
        self.m, self.v, self.t = m, v, t


def _weight_names(parameters):
    return [name for name in parameters if name.startswith("W")]


def objective_and_gradients(network, X, y, l2=0.001):
    """Mean BCE + (l2 / 2) * squared weight norm; biases are not penalized."""
    l2 = _bounded_real(l2, "l2", 0, 10)
    value = network.loss(X, y)
    gradients = network.gradients(X, y)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for name in _weight_names(network.parameters) if l2 else ():
                weight = network.parameters[name]
                value += 0.5 * l2 * float(np.sum(weight ** 2))
                gradients[name] = gradients[name] + l2 * weight
    except FloatingPointError as exc:
        raise ValueError("Regularization overflowed; use smaller weights") from exc
    if not np.isfinite(value) or not all(np.isfinite(g).all() for g in gradients.values()):
        raise ValueError("Regularized objective produced a non-finite value")
    return float(value), gradients


def classification_metrics(y, p, loss):
    p = _matrix(p, "p")
    y = _targets(y, len(p))
    if not len(p) or p.shape != y.shape or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must have shape (N, 1), be nonempty, and lie in [0, 1]")
    loss = _bounded_real(loss, "loss", 0, np.finfo(np.float64).max)
    predicted = p >= 0.5
    actual = y == 1
    tp = int(np.sum(predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "loss": loss, "accuracy": float((tp + tn) / len(y)),
        "precision": precision, "recall": recall, "f1": f1,
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "samples": len(y),
    }


class _Logistic:
    """An affine score, sigmoid, and BCE: no hidden layer."""

    def __init__(self):
        self.parameters = {"W": np.zeros((2, 1)), "b": np.zeros((1, 1))}

    def _logits(self, X):
        X = _inputs(X)
        value = _matmul(X, self.parameters["W"]) + self.parameters["b"]
        if not np.isfinite(value).all():
            raise ValueError("Baseline logits are not finite")
        return value

    def predict_proba(self, X):
        return _sigmoid(self._logits(X))

    def loss(self, X, y):
        z = self._logits(X)
        y = _targets(y, len(z))
        with np.errstate(under="ignore"):
            return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def gradients(self, X, y):
        X = _inputs(X)
        y = _targets(y, len(X))
        dz = (self.predict_proba(X) - y) / len(X)
        return {"W": _matmul(X.T, dz), "b": dz.sum(axis=0, keepdims=True)}


def _snapshot(model, epoch, row):
    return {"epoch": epoch, "parameters": {k: v.tolist() for k, v in model.parameters.items()},
            "train_loss": row["train_loss"], "val_loss": row["val_loss"]}


def _fit(model, X_train, y_train, X_val, y_val, config, seed, snapshots=True):
    """No test data enters this function, including preprocessing or stopping."""
    optimizer = Adam(config["learning_rate"]) if config["optimizer"] == "adam" else SGD(config["learning_rate"])
    rng = np.random.default_rng(seed)
    history, checkpoints = [], []
    best_loss, significant_anchor = math.inf, math.inf
    best_epoch, best_weights, stale = 0, None, 0
    interval = max(1, math.ceil(config["epochs"] / 58))
    stop_reason = "epoch_limit"
    for epoch in range(config["epochs"] + 1):
        if epoch:
            order = rng.permutation(len(X_train))
            for offset in range(0, len(order), config["batch_size"]):
                ids = order[offset:offset + config["batch_size"]]
                _, gradients = objective_and_gradients(model, X_train[ids], y_train[ids], config["l2"])
                optimizer.step(model.parameters, gradients)
        _, full_gradients = objective_and_gradients(model, X_train, y_train, config["l2"])
        with np.errstate(over="raise", invalid="raise"):
            try:
                gradient_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in full_gradients.values())))
            except FloatingPointError as exc:
                raise ValueError("Gradient norm overflowed") from exc
        if not np.isfinite(gradient_norm):
            raise ValueError("Gradient norm overflowed")
        row = {
            "epoch": epoch, "train_loss": model.loss(X_train, y_train),
            "val_loss": model.loss(X_val, y_val),
            "train_accuracy": float(np.mean((model.predict_proba(X_train) >= 0.5) == y_train)),
            "val_accuracy": float(np.mean((model.predict_proba(X_val) >= 0.5) == y_val)),
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        # Always restore the actual lowest validation BCE, independently of the
        # significant-improvement threshold used to reset the patience counter.
        if row["val_loss"] < best_loss:
            best_loss, best_epoch = row["val_loss"], epoch
            best_weights = {k: v.copy() for k, v in model.parameters.items()}
        if significant_anchor - row["val_loss"] > config["min_delta"]:
            significant_anchor, stale = row["val_loss"], 0
        elif epoch:
            stale += 1
        if snapshots and epoch % interval == 0:
            checkpoints.append(_snapshot(model, epoch, row))
        if stale >= config["patience"]:
            stop_reason = "validation_patience"
            break
    stopped_epoch = history[-1]["epoch"]
    if snapshots and checkpoints[-1]["epoch"] != stopped_epoch:
        checkpoints.append(_snapshot(model, stopped_epoch, history[-1]))
    model.parameters.update(best_weights)
    if snapshots and all(c["epoch"] != best_epoch for c in checkpoints):
        checkpoints.append(_snapshot(model, best_epoch, history[best_epoch]))
    checkpoints.sort(key=lambda c: c["epoch"])
    return history, checkpoints, best_epoch, stopped_epoch, stop_reason


def run_experiment(dataset="moons", samples=600, noise=0.15, hidden=16, seed=7,
                   epochs=1200, learning_rate=0.01, optimizer="adam", batch_size=32,
                   l2=0.001, patience=100, min_delta=1e-5):
    """Fit on training rows, select on validation rows, then evaluate test once.

    Different random streams separate data, split, initialization, and shuffles.
    The configuration is fully validated before constructing any experiment.
    """
    if dataset not in ("moons", "circles"):
        raise ValueError("dataset must be 'moons' or 'circles'")
    if optimizer not in ("adam", "sgd"):
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    config = {
        "dataset": dataset, "samples": _integer(samples, "samples", 120, 5000),
        "noise": _bounded_real(noise, "noise", 0, 0.6),
        "hidden": _integer(hidden, "hidden", 1, 64), "seed": _integer(seed, "seed", 0),
        "epochs": _integer(epochs, "epochs", 1, 10000),
        "learning_rate": _positive(learning_rate, "learning_rate"),
        "optimizer": optimizer, "batch_size": _integer(batch_size, "batch_size", 1, 5000),
        "l2": _bounded_real(l2, "l2", 0, 10), "patience": _integer(patience, "patience", 1, 10000),
        "min_delta": _bounded_real(min_delta, "min_delta", 0, 10),
    }
    seed = config["seed"]
    X, y = make_dataset(dataset, config["samples"], config["noise"], seed)
    ids = stratified_split(y, seed + 1)
    mean, scale = fit_standardizer(X[ids["train_idx"]])
    # Transform only the splits needed for fitting. The held-out set is not even
    # transformed until both models have been selected and restored.
    train_X = standardize(X[ids["train_idx"]], mean, scale)
    val_X = standardize(X[ids["val_idx"]], mean, scale)
    train_y, val_y = y[ids["train_idx"]], y[ids["val_idx"]]
    network = Network(config["hidden"], seed + 2)
    history, checkpoints, best, stopped, reason = _fit(
        network, train_X, train_y, val_X, val_y, config, seed + 3)
    baseline = _Logistic()
    _, _, baseline_best, baseline_stopped, baseline_reason = _fit(
        baseline, train_X, train_y, val_X, val_y, config, seed + 4, snapshots=False)
    test_X, test_y = standardize(X[ids["test_idx"]], mean, scale), y[ids["test_idx"]]
    metrics = {}
    for name, features, targets in (("train", train_X, train_y), ("validation", val_X, val_y), ("test", test_X, test_y)):
        metrics[name] = classification_metrics(targets, network.predict_proba(features), network.loss(features, targets))
    baseline_test = classification_metrics(test_y, baseline.predict_proba(test_X), baseline.loss(test_X, test_y))
    return {
        "config": config, "network": network,
        "data": dict({"X": X, "y": y, "mean": mean, "scale": scale}, **ids),
        "history": history, "checkpoints": checkpoints,
        "best_epoch": best, "stopped_epoch": stopped, "stop_reason": reason,
        "metrics": metrics,
        "baseline": {"parameters": {k: v.tolist() for k, v in baseline.parameters.items()},
                     "best_epoch": baseline_best, "stopped_epoch": baseline_stopped,
                     "stop_reason": baseline_reason, "test": baseline_test},
        "seeds": {"data": seed, "split": seed + 1, "initialization": seed + 2,
                  "network_batches": seed + 3, "baseline_batches": seed + 4},
    }
