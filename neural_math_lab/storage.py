"""Readable model files: JSON numbers only, never executable pickle objects."""

import json
from pathlib import Path

import numpy as np

from .core import Network


def model_document(network):
    return {
        "format": "neural-math-lab-v1",
        "hidden": network.hidden,
        "parameters": {name: value.tolist() for name, value in network.parameters.items()},
    }


def save_model(network, path):
    Path(path).write_text(json.dumps(model_document(network), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("Model file exceeds the 256 KB limit.")
    try:
        doc = json.loads(raw)
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
