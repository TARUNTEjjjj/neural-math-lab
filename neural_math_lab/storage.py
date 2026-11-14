"""Readable model files: JSON numbers only, never executable pickle objects."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .core import Network


def model_document(network):
    return {
        "format": "neural-math-lab-v1",
        "hidden": network.hidden,
        "parameters": {name: value.tolist() for name, value in network._parameters().items()},
    }


def save_model(network, path):
    document = model_document(network)
    _network_from_document(document)
    _write_document(document, path)


def _write_document(document, path):
    """Replace only after a complete JSON file has been written beside the target."""
    encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".neural-math-lab-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_document(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("Model file exceeds the 256 KB limit.")
    try:
        return json.loads(raw)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Model must be valid JSON.") from exc


def _network_from_document(doc):
    try:
        if not isinstance(doc, dict) or set(doc) != {"format", "hidden", "parameters"}:
            raise ValueError("Unexpected model fields.")
        if doc["format"] != "neural-math-lab-v1":
            raise ValueError("Unsupported model format.")
        net = Network(hidden=doc["hidden"], seed=0)
        supplied = doc["parameters"]
        if not isinstance(supplied, dict) or set(supplied) != set(net.parameters):
            raise ValueError("Model must contain W1, b1, W2, and b2.")
        for name, template in net.parameters.items():
            rows = supplied[name]
            if not isinstance(rows, list) or not all(
                isinstance(row, list) and all(type(v) in (int, float) for v in row)
                for row in rows
            ):
                raise ValueError("Model parameters must be matrices of JSON numbers.")
            value = np.asarray(supplied[name], dtype=np.float64)
            if value.shape != template.shape or not np.isfinite(value).all():
                raise ValueError("Invalid shape or non-finite value for " + name)
            net.parameters[name] = value
        return net
    except (TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("Model must be valid JSON with finite numeric parameters.") from exc


def load_model(path):
    return _network_from_document(_read_document(path))


def save_pipeline(network, mean, scale, dataset, path):
    doc = {"format": "neural-math-lab-pipeline-v2", "model": model_document(network),
           "standardizer": {"mean": np.asarray(mean).tolist(), "scale": np.asarray(scale).tolist()},
           "dataset": dataset}
    _pipeline_from_document(doc)
    _write_document(doc, path)


def _pipeline_from_document(doc):
    if not isinstance(doc, dict) or set(doc) != {"format", "model", "standardizer", "dataset"} or doc["format"] != "neural-math-lab-pipeline-v2":
        raise ValueError("Invalid pipeline format or fields.")
    if doc["dataset"] not in ("moons", "circles", "iris"):
        raise ValueError("Pipeline dataset must be moons, circles or iris.")
    scaler = doc["standardizer"]
    if not isinstance(scaler, dict) or set(scaler) != {"mean", "scale"}:
        raise ValueError("Pipeline must contain mean and scale.")
    arrays = {}
    for key, value in scaler.items():
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list) or len(value[0]) != 2 or not all(type(v) in (float, int) for v in value[0]):
            raise ValueError("Standardizer must contain 1 by 2 matrices of JSON numbers.")
        try:
            arrays[key] = np.asarray(value, dtype=np.float64)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Standardizer must contain finite numbers.") from exc
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Standardizer must contain finite numbers.")
    if np.any(arrays["scale"] <= 0):
        raise ValueError("Standardizer scale must be positive.")
    return _network_from_document(doc["model"]), arrays["mean"], arrays["scale"], doc["dataset"]


def load_pipeline(path):
    return _pipeline_from_document(_read_document(path))


def load_for_explanation(path):
    doc = _read_document(path)
    if isinstance(doc, dict) and doc.get("format") == "neural-math-lab-pipeline-v2":
        return _pipeline_from_document(doc)
    return _network_from_document(doc), np.zeros((1, 2)), np.ones((1, 2)), "xor"
