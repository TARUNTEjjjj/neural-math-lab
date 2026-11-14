"""Pinned data import and real-data experiment boundaries, without network."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from neural_math_lab.cli import main
from neural_math_lab.experiment import run_experiment
from neural_math_lab.experiment_report import write_experiment_report
from neural_math_lab.resource_data import IRIS_URL, download_iris, load_iris, parse_iris
from neural_math_lab.storage import load_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "iris"


class Response(io.BytesIO):
    def geturl(self):
        return IRIS_URL


class ResourceDataTests(unittest.TestCase):
    def test_exact_edition_transform_and_provenance(self):
        X, y, source = load_iris(FIXTURE)
        self.assertEqual(X.shape, (150, 2))
        self.assertEqual(y.shape, (150, 1))
        self.assertEqual(int(y.sum()), 50)
        np.testing.assert_array_equal(X[0], [5.1, 3.5])
        self.assertEqual(y[0, 0], 1)
        self.assertEqual(y[50, 0], 0)
        self.assertEqual(source["features"], ["sepal length (cm)", "sepal width (cm)"])

    def test_download_is_bounded_pinned_and_preserves_existing(self):
        opener = Mock()
        opener.open.return_value = Response((FIXTURE / "iris.csv").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "iris"
            download_iris(output, opener)
            self.assertEqual(load_iris(output)[0].shape, (150, 2))
            self.assertIn("downloaded_at", json.loads((output / "source.json").read_text()))
            self.assertEqual(opener.open.call_args.kwargs["timeout"], 20)
            self.assertEqual(opener.open.call_args.args[0].full_url, IRIS_URL)
            with self.assertRaises(ValueError):
                download_iris(output, opener)
            self.assertEqual(opener.open.call_count, 1)

    def test_changed_bytes_rejected_without_creating_dataset(self):
        opener = Mock()
        opener.open.return_value = Response((FIXTURE / "iris.csv").read_bytes().replace(b"5.1", b"5.2", 1))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "iris"
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_iris(output, opener)
            self.assertFalse(output.exists())

    def test_malformed_numeric_and_label_fields_are_rejected(self):
        good = (FIXTURE / "iris.csv").read_bytes()
        for bad in (good.replace(b"5.1", b"nan", 1), good.replace(b"5.1", b"inf", 1),
                    good.replace(b"0.2,0", b"0.2,9", 1), good[:-20], b"x" * 16385):
            with self.subTest(data=bad[:20]), self.assertRaises(ValueError):
                parse_iris(bad)

    def test_oversized_or_redirected_response_is_rejected(self):
        for content, url in [(b"x" * 16385, IRIS_URL), (b"", "https://example.org/iris")]:
            opener = Mock()
            response = Response(content)
            response.geturl = lambda: url
            opener.open.return_value = response
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    download_iris(Path(directory) / "iris", opener)

    def test_real_data_uses_deterministic_disjoint_splits_and_training_scaler(self):
        first = run_experiment(dataset="iris", data_path=FIXTURE, epochs=3, seed=4)
        second = run_experiment(dataset="iris", data_path=FIXTURE, epochs=3, seed=4)
        data = first["data"]
        indices = [data[name] for name in ("train_idx", "val_idx", "test_idx")]
        self.assertEqual(list(map(len, indices)), [90, 30, 30])
        self.assertEqual(len(set(np.concatenate(indices))), 150)
        np.testing.assert_array_equal(data["train_idx"], second["data"]["train_idx"])
        np.testing.assert_allclose(data["mean"], data["X"][data["train_idx"]].mean(axis=0, keepdims=True))
        self.assertEqual(first["config"]["samples"], 150)
        self.assertIsNone(first["config"]["noise"])
        self.assertIn("sha256", first["config"]["source"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            write_experiment_report(first, output)
            self.assertEqual(load_pipeline(output / "pipeline.json")[3], "iris")
            report = (output / "report.html").read_text()
            self.assertIn("Iris observations", report)
            self.assertIn("--data-path datasets/iris", report)
            self.assertNotIn("--noise None", report)
            self.assertNotIn("These synthetic results", report)
            self.assertNotIn("this synthetic split", report)
            self.assertIn("Replace datasets/iris with the directory you imported", report)
            self.assertIn("fetch-dataset --output datasets/iris", report)

    def test_iris_requires_local_data_and_rejects_synthetic_options(self):
        for kwargs in ({"dataset": "iris"}, {"dataset": "iris", "data_path": FIXTURE, "samples": 200},
                       {"dataset": "moons", "data_path": FIXTURE}):
            with self.assertRaises((ValueError, OSError)):
                run_experiment(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(["experiment", "--dataset", "iris", "--data-path", str(FIXTURE),
                                   "--epochs", "1", "--output", str(Path(directory) / "cli")]), 0)


if __name__ == "__main__":
    unittest.main()
