"""A saved experiment must retain the training-only input transformation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.storage import (
    load_for_explanation,
    load_pipeline,
    model_document,
    save_model,
    save_pipeline,
)


class PipelineStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pipeline.json"
        self.network = Network(hidden=3, seed=11)
        self.mean = np.array([[4., -2.]])
        self.scale = np.array([[2., .5]])
        self.document = {
            "format": "neural-math-lab-pipeline-v2",
            "model": model_document(self.network),
            "standardizer": {"mean": self.mean.tolist(), "scale": self.scale.tolist()},
            "dataset": "moons",
        }

    def write(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_round_trip_preserves_raw_input_predictions_exactly(self):
        raw = np.array([[0., 0.], [4., -2.], [-3., 7.]])
        expected = self.network.predict_proba(standardize(raw, self.mean, self.scale))
        save_pipeline(self.network, self.mean, self.scale, "moons", self.path)
        network, mean, scale, dataset = load_pipeline(self.path)
        self.assertEqual(dataset, "moons")
        np.testing.assert_array_equal(mean, self.mean)
        np.testing.assert_array_equal(scale, self.scale)
        np.testing.assert_array_equal(network.predict_proba(standardize(raw, mean, scale)), expected)
        for name in self.network.parameters:
            np.testing.assert_array_equal(network.parameters[name], self.network.parameters[name])

    def test_explanation_loader_handles_old_xor_models_without_scaling(self):
        save_model(self.network, self.path)
        network, mean, scale, dataset = load_for_explanation(self.path)
        self.assertEqual(dataset, "xor")
        np.testing.assert_array_equal(mean, [[0., 0.]])
        np.testing.assert_array_equal(scale, [[1., 1.]])
        np.testing.assert_array_equal(network.predict_proba([[1., 0.]]), self.network.predict_proba([[1., 0.]]))

    def test_explanation_loader_preserves_pipeline_scaler(self):
        self.write(self.document)
        _, mean, scale, dataset = load_for_explanation(self.path)
        self.assertEqual(dataset, "moons")
        np.testing.assert_array_equal(mean, self.mean)
        np.testing.assert_array_equal(scale, self.scale)

    def test_unknown_format_missing_extra_and_nonobject_documents_rejected(self):
        malformed = [None, [], "pipeline", dict(self.document, format="unknown"), dict(self.document, extra=1)]
        for key in self.document:
            case = copy.deepcopy(self.document)
            del case[key]
            malformed.append(case)
        for document in malformed:
            with self.subTest(document=document):
                self.write(document)
                with self.assertRaises(ValueError):
                    load_pipeline(self.path)

    def test_unknown_datasets_are_rejected(self):
        for dataset in ("xor", "unknown", None, [], True):
            self.write(dict(self.document, dataset=dataset))
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                load_pipeline(self.path)

    def test_scaler_requires_exact_fields_and_one_row_two_columns(self):
        malformed = [None, [], {}, {"mean": [[0, 0]]},
                     {"mean": [[0, 0]], "scale": [[1, 1]], "other": 1}]
        for key in ("mean", "scale"):
            for value in ([], [0, 1], [[0]], [[0, 1], [2, 3]], {"a": 1}):
                scaler = copy.deepcopy(self.document["standardizer"])
                scaler[key] = value
                malformed.append(scaler)
        for scaler in malformed:
            with self.subTest(scaler=scaler):
                self.write(dict(self.document, standardizer=scaler))
                with self.assertRaises(ValueError):
                    load_pipeline(self.path)

    def test_scaler_rejects_strings_booleans_and_nonfinite_values(self):
        for key in ("mean", "scale"):
            for value in ("1.0", True, None, np.nan, np.inf, -np.inf, 10 ** 1000):
                document = copy.deepcopy(self.document)
                document["standardizer"][key][0][0] = value
                self.write(document)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_pipeline(self.path)

    def test_nonpositive_scale_is_rejected(self):
        for value in (0, -1):
            document = copy.deepcopy(self.document)
            document["standardizer"]["scale"][0][1] = value
            self.write(document)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive"):
                load_pipeline(self.path)

    def test_embedded_model_is_validated(self):
        for model in (None, {}, dict(self.document["model"], hidden=True),
                      dict(self.document["model"], format="unknown")):
            self.write(dict(self.document, model=model))
            with self.subTest(model=model), self.assertRaises(ValueError):
                load_pipeline(self.path)

    def test_invalid_save_does_not_replace_existing_pipeline(self):
        self.write(self.document)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            save_pipeline(self.network, self.mean, [[0, 1]], "moons", self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_save_preserves_pipeline_and_cleans_temporary_file(self):
        self.write(self.document)
        before = self.path.read_bytes()
        with patch("neural_math_lab.storage.os.replace", side_effect=OSError("destination unavailable")):
            with self.assertRaisesRegex(OSError, "destination unavailable"):
                save_pipeline(self.network, self.mean, self.scale, "moons", self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        load_pipeline(self.path)

    def test_oversized_and_malformed_json_are_rejected(self):
        for raw in (b" " * 256_001, b"{broken", b"\xff\xfe\xff"):
            self.path.write_bytes(raw)
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                load_pipeline(self.path)


if __name__ == "__main__":
    unittest.main()
