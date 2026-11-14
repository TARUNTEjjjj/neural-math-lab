"""Explicit, bounded import of one reviewed dataset; never executed as code."""

import csv
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

IRIS_COMMIT = "25dee604bae18205b01548348388baf7a1cdfe0e"
IRIS_URL = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/" + IRIS_COMMIT + "/sklearn/datasets/data/iris.csv"
IRIS_SHA256 = "f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449"
MAX_BYTES = 16_384


def iris_provenance():
    return {
        "name": "Iris: setosa versus other species, two sepal measurements",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "commit": IRIS_COMMIT, "url": IRIS_URL, "sha256": IRIS_SHA256,
        "citation": "Fisher, R. (1936). Iris [Dataset]. DOI: 10.24432/C56C76",
        "origin": "https://archive.ics.uci.edu/dataset/53/iris",
        "dataset_license": "CC-BY-4.0",
        "dataset_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_package_license": "BSD-3-Clause",
        "source_package_license_url": "https://github.com/scikit-learn/scikit-learn/blob/" + IRIS_COMMIT + "/COPYING",
        "features": ["sepal length (cm)", "sepal width (cm)"],
        "classes": {"0": "versicolor or virginica", "1": "setosa"},
        "transformation": "Keep all 150 rows in source order, select columns 0 and 1, map species 0 to binary 1 and species 1/2 to binary 0. No synthetic noise or resampling.",
        "edition": "scikit-learn 1.7.2; its corrected Iris edition differs from the original UCI file.",
    }


def parse_iris(data):
    """Validate the exact upstream CSV schema before selecting two features."""
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError("Iris data must be bytes no larger than 16 KiB")
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8")), strict=True))
        if len(rows) != 151 or rows[0] != ["150", "4", "setosa", "versicolor", "virginica"]:
            raise ValueError("Expected the 150-row scikit-learn Iris CSV")
        values = []
        labels = []
        for row in rows[1:]:
            if len(row) != 5 or row[4] not in ("0", "1", "2"):
                raise ValueError("Invalid Iris row or class")
            values.append([float(v) for v in row[:4]])
            labels.append(int(row[4]))
        all_features = np.asarray(values, dtype=np.float64)
        species = np.asarray(labels)
        if not np.isfinite(all_features).all() or np.any(all_features <= 0) or np.any(all_features > 20):
            raise ValueError("Iris measurements must be finite and between 0 and 20 cm")
        if any(np.count_nonzero(species == i) != 50 for i in range(3)):
            raise ValueError("Expected 50 Iris plants per species")
        return all_features[:, :2].copy(), (species == 0).astype(np.float64).reshape(-1, 1)
    except (UnicodeError, csv.Error, OverflowError) as exc:
        raise ValueError("Invalid Iris CSV") from exc


def _verified(data):
    if hashlib.sha256(data).hexdigest() != IRIS_SHA256:
        raise ValueError("Iris checksum differs from the reviewed source; the import was rejected")
    return parse_iris(data)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Dataset redirects are not allowed", headers, fp)


def download_iris(output, opener=None):
    """Fetch only the pinned URL, verify fully, then create a new local dataset."""
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset output already exists; choose a new path to preserve it")
    opener = opener or build_opener(_NoRedirect())
    request = Request(IRIS_URL, headers={"User-Agent": "neural-math-lab-dataset-import/1", "Accept": "text/plain"})
    with opener.open(request, timeout=20) as response:
        if response.geturl() != IRIS_URL:
            raise ValueError("Unexpected dataset response URL")
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Dataset response exceeds 16 KiB")
    _verified(data)
    metadata = iris_provenance()
    metadata["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "iris.csv").write_bytes(data)
        (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    return metadata


def load_iris(directory):
    """Load offline; verify bytes again instead of trusting sidecar metadata."""
    if directory is None:
        raise ValueError("Iris needs --data-path: first run fetch-dataset --output datasets/iris")
    path = Path(directory) / "iris.csv"
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Iris file exceeds 16 KiB")
    X, y = _verified(data)
    return X, y, iris_provenance()
