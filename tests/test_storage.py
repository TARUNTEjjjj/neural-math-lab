"""Model persistence must preserve predictions and reject malformed files."""

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neural_math_lab.core import Network, train, xor_data
from neural_math_lab.storage import load_model, model_document, save_model, save_pipeline


def permissions(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.json"
        self.network = Network(hidden=3, seed=11)

    def write_document(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_round_trip_preserves_every_parameter_and_prediction(self):
        X, y = xor_data()
        train(self.network, X, y, epochs=10)
        probes = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.25, 0.75], [-0.1, 1.1]])
        before = self.network.predict_proba(probes)
        save_model(self.network, self.path)
        restored = load_model(self.path)
        self.assertEqual(restored.hidden, self.network.hidden)
        for name in self.network.parameters:
            np.testing.assert_array_equal(restored.parameters[name], self.network.parameters[name])
        np.testing.assert_array_equal(restored.predict_proba(probes), before)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "neural-math-lab-v1")

    def test_model_document_is_an_independent_snapshot(self):
        document = model_document(self.network)
        before = self.network.parameters["W1"].copy()
        document["parameters"]["W1"][0][0] = 999
        np.testing.assert_array_equal(self.network.parameters["W1"], before)

    def test_list_parameters_round_trip_without_mutating_the_learners_containers(self):
        self.network.parameters = {name: value.tolist() for name, value in self.network.parameters.items()}
        original = dict(self.network.parameters)
        before = self.network.predict_proba([[0., 1.]])
        save_model(self.network, self.path)
        np.testing.assert_array_equal(load_model(self.path).predict_proba([[0., 1.]]), before)
        for name, value in original.items():
            self.assertIs(self.network.parameters[name], value)

    def test_invalid_save_preserves_previous_model(self):
        save_model(self.network, self.path)
        before = self.path.read_bytes()
        for invalid in (np.zeros((1, 3)), np.full((2, 3), np.nan)):
            self.network.parameters["W1"] = invalid
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                save_model(self.network, self.path)
            self.assertEqual(self.path.read_bytes(), before)
            load_model(self.path)

    def test_long_valid_destination_name_can_be_saved(self):
        path = self.path.with_name("m" * 240 + ".json")
        save_model(self.network, path)
        restored = load_model(path)
        np.testing.assert_array_equal(restored.predict_proba([[1., 0.]]),
                                      self.network.predict_proba([[1., 0.]]))

    def test_failed_sync_or_replacement_preserves_previous_model(self):
        save_model(self.network, self.path)
        before = self.path.read_bytes()
        self.network.parameters["b2"][0, 0] = 12
        for operation in ("fsync", "replace"):
            with self.subTest(operation=operation), patch(
                    "neural_math_lab.storage.os." + operation,
                    side_effect=OSError("injected save failure")):
                with self.assertRaisesRegex(OSError, "injected save failure"):
                    save_model(self.network, self.path)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(list(self.path.parent.iterdir()), [self.path])
            load_model(self.path)

    def test_nested_list_and_tuple_parameters_save_like_arrays(self):
        # Every calculation accepts nested lists and tuples, so saving must too.
        expected = model_document(self.network)
        probes = np.array([[1., 0.], [0.25, 0.75]])
        before = self.network.predict_proba(probes)
        self.network.parameters = {name: value.tolist() for name, value in self.network.parameters.items()}
        self.network.parameters["b2"] = tuple(tuple(row) for row in self.network.parameters["b2"])
        self.assertEqual(model_document(self.network), expected)
        save_model(self.network, self.path)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), expected)
        np.testing.assert_array_equal(load_model(self.path).predict_proba(probes), before)
        self.assertIsInstance(self.network.parameters["W1"], list, "Saving must not replace the learner's values.")

    def test_invalid_list_parameters_raise_value_errors_and_keep_the_saved_model(self):
        save_model(self.network, self.path)
        before = self.path.read_bytes()
        for invalid in ([[float("nan"), 0, 0], [0, 0, 0]], [[0.5, 0.5, 0.5]], [[0.5], [1.0, 1.0, 1.0]], "W1", None):
            self.network.parameters["W1"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                save_model(self.network, self.path)
            self.assertEqual(self.path.read_bytes(), before)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_saved_files_get_the_permissions_of_an_ordinary_write(self):
        pipeline, shared = self.path.with_name("pipeline.json"), self.path.with_name("shared.json")
        previous = os.umask(0o022)
        try:
            save_model(self.network, self.path)
            save_pipeline(self.network, [[0., 0.]], [[1., 1.]], "moons", pipeline)
            self.assertEqual((permissions(self.path), permissions(pipeline)), (0o644, 0o644))
            os.umask(0o002)
            save_model(self.network, shared)
            self.assertEqual(permissions(shared), 0o664)
            # Replacing a file keeps its permissions, whatever the umask.
            for mode in (0o640, 0o604):
                os.chmod(self.path, mode)
                save_model(self.network, self.path)
                self.assertEqual(permissions(self.path), mode)
        finally:
            os.umask(previous)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["model.json", "pipeline.json", "shared.json"])

    @unittest.skipUnless(os.name == "posix", "symbolic links")
    def test_saving_through_a_symbolic_link_replaces_its_target(self):
        target = self.path.parent / "models" / "target.json"
        target.parent.mkdir()
        save_model(Network(hidden=2, seed=1), target)
        link = self.path.with_name("link.json")
        link.symlink_to(target)
        save_model(self.network, link)
        self.assertTrue(link.is_symlink())
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), model_document(self.network))
        np.testing.assert_array_equal(load_model(link).predict_proba([[1., 0.]]),
                                      self.network.predict_proba([[1., 0.]]))
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["link.json", "models"])
        self.assertEqual([p.name for p in target.parent.iterdir()], ["target.json"])

    def test_unknown_format_is_rejected(self):
        document = model_document(self.network)
        document["format"] = "neural-math-lab-v999"
        self.write_document(document)
        with self.assertRaisesRegex(ValueError, "format"):
            load_model(self.path)

    def test_missing_extra_or_wrong_top_level_fields_are_rejected(self):
        valid = model_document(self.network)
        missing = dict(valid)
        del missing["parameters"]
        extra = dict(valid, source="unrecognised")
        for document in ([], None, "model", missing, extra):
            with self.subTest(document=document):
                self.write_document(document)
                with self.assertRaises(ValueError):
                    load_model(self.path)

    def test_invalid_hidden_sizes_are_rejected(self):
        for hidden in (0, 65, 2.5, True, "3"):
            document = model_document(self.network)
            document["hidden"] = hidden
            self.write_document(document)
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                load_model(self.path)

    def test_missing_extra_or_nonobject_parameters_are_rejected(self):
        valid = model_document(self.network)
        missing = dict(valid["parameters"])
        del missing["b2"]
        extra = dict(valid["parameters"], other=[[0]])
        for parameters in ([], None, missing, extra):
            document = dict(valid, parameters=parameters)
            self.write_document(document)
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                load_model(self.path)

    def test_wrong_parameter_shapes_are_rejected(self):
        for name, invalid in (("W1", [[1, 2, 3]]), ("b1", [0, 0, 0]),
                              ("W2", [[0, 0, 0]]), ("b2", [[0], [0]])):
            document = model_document(self.network)
            document["parameters"][name] = invalid
            self.write_document(document)
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                load_model(self.path)

    def test_nonfinite_parameters_are_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            document = model_document(self.network)
            document["parameters"]["b2"][0][0] = invalid
            self.write_document(document)
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                load_model(self.path)

    def test_non_numeric_or_ragged_parameters_are_rejected(self):
        for invalid in ([["not a number"]], [["0.5"]], [[True]], [[False]],
                        [[0], [0, 1]], [[{"value": 0}]]):
            document = model_document(self.network)
            document["parameters"]["b2"] = invalid
            self.write_document(document)
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                load_model(self.path)

    def test_oversized_files_are_rejected(self):
        self.path.write_bytes(b" " * 256_001)
        with self.assertRaisesRegex(ValueError, "256 KB"):
            load_model(self.path)

    def test_invalid_json_and_encoding_are_rejected(self):
        for invalid in (b"", b"{broken", b"null trailing data", b"\xff\xfe\xff"):
            self.path.write_bytes(invalid)
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                load_model(self.path)

    def test_missing_file_is_reported(self):
        with self.assertRaises(OSError):
            load_model(self.path)


if __name__ == "__main__":
    unittest.main()
