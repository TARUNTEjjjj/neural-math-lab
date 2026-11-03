"""Small synthetic classification problems with auditable data boundaries."""

from numbers import Real

import numpy as np

from .core import _inputs, _integer, _matrix, _targets


def _bounded_real(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError("{} must be between {} and {}".format(name, minimum, maximum))
    return result


def make_dataset(dataset="moons", samples=600, noise=0.15, seed=7):
    """Draw independent noisy points; no labels or rows come from a service.

    Moons: upper unit semicircle versus its reflected, translated partner.
    Circles: an outer radius-one ring versus an inner radius-one-half ring.
    The noise is a Gaussian standard deviation in each coordinate.
    """
    if dataset not in ("moons", "circles"):
        raise ValueError("dataset must be 'moons' or 'circles'")
    samples = _integer(samples, "samples", 120, 5000)
    noise = _bounded_real(noise, "noise", 0, 0.6)
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    counts = (samples // 2, samples - samples // 2)
    y = np.concatenate([np.full((n, 1), label, dtype=np.float64)
                        for label, n in enumerate(counts)])
    if dataset == "moons":
        first = rng.uniform(0, np.pi, counts[0])
        second = rng.uniform(0, np.pi, counts[1])
        X = np.concatenate([
            np.column_stack([np.cos(first), np.sin(first)]),
            np.column_stack([1 - np.cos(second), 0.5 - np.sin(second)]),
        ])
    else:
        angles = rng.uniform(0, 2 * np.pi, samples)
        radius = np.where(y[:, 0] == 0, 1.0, 0.5)
        X = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    X += rng.normal(0, noise, X.shape)
    order = rng.permutation(samples)
    return X[order], y[order]


def stratified_split(y, seed=7):
    """Return disjoint row IDs, retaining both classes in every partition.

    Per class: floor(60%) train, floor(20%) validation, remainder test.
    The final remainder matters for sample counts not divisible by ten.
    """
    y = _matrix(y, "y")
    y = _targets(y, len(y))
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    groups = {"train_idx": [], "val_idx": [], "test_idx": []}
    for label in (0, 1):
        ids = np.flatnonzero(y[:, 0] == label)
        if len(ids) < 6:
            raise ValueError("Each class needs at least six rows for stratified splitting.")
        ids = rng.permutation(ids)
        train_end, val_size = int(0.6 * len(ids)), int(0.2 * len(ids))
        groups["train_idx"].append(ids[:train_end])
        groups["val_idx"].append(ids[train_end:train_end + val_size])
        groups["test_idx"].append(ids[train_end + val_size:])
    return {name: rng.permutation(np.concatenate(parts)) for name, parts in groups.items()}


def fit_standardizer(X_train):
    """Compute population moments from training examples only."""
    X_train = _inputs(X_train)
    with np.errstate(over="raise", invalid="raise"):
        try:
            mean = X_train.mean(axis=0, keepdims=True)
            scale = X_train.std(axis=0, keepdims=True)
        except FloatingPointError as exc:
            raise ValueError("Training coordinates are too large to standardize.") from exc
    constant = X_train.max(axis=0, keepdims=True) == X_train.min(axis=0, keepdims=True)
    scale = np.where(constant | (scale == 0), 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Standardization produced non-finite statistics.")
    return mean, scale


def standardize(X, mean, scale):
    X = _inputs(X)
    mean, scale = _matrix(mean, "mean"), _matrix(scale, "scale")
    if mean.shape != (1, 2) or scale.shape != (1, 2) or np.any(scale <= 0):
        raise ValueError("mean and positive scale must both have shape (1, 2)")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = (X - mean) / scale
        except FloatingPointError as exc:
            raise ValueError("Coordinates are too large to standardize.") from exc
    if not np.isfinite(result).all():
        raise ValueError("Standardization produced non-finite coordinates.")
    return result
