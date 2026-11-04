"""Exercise the experiment command and inspect the saved, offline report."""

import base64
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.storage import load_pipeline, save_pipeline


ROOT = Path(__file__).resolve().parents[1]


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current_script = None
        self.csp = None
        self.references = []
        self.event_attributes = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script":
            self.current_script = {"attrs": attributes, "text": ""}
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "content-security-policy":
            self.csp = attributes.get("content")
        for key, value in attrs:
            if key in ("src", "href", "xlink:href", "poster", "data") and value:
                self.references.append(value)
            if key.startswith("on"):
                self.event_attributes.append(key)

    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.scripts.append(self.current_script)
            self.current_script = None


class AdvancedCommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.base = Path(cls.directory.name)
        cls.environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(cls.base / "matplotlib"))
        cls.output = cls.base / "experiment"
        cls.training = cls.run_command("experiment", "--samples", "120", "--epochs", "10", "--hidden", "3", "--seed", "11", "--fn-cost", "3", "--output", str(cls.output))
        if cls.training.returncode != 0:
            cls.directory.cleanup()
            raise AssertionError("Experiment failed:\n" + cls.training.stdout + cls.training.stderr)
        cls.document = json.loads((cls.output / "experiment.json").read_text(encoding="utf-8"))
        cls.metrics = json.loads((cls.output / "metrics.json").read_text(encoding="utf-8"))
        cls.report = (cls.output / "report.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @classmethod
    def run_command(cls, *arguments):
        return subprocess.run([sys.executable, "-m", "neural_math_lab", *arguments],
                              cwd=ROOT, env=cls.environment, text=True,
                              capture_output=True, timeout=60)

    def test_experiment_outputs_complete_artifacts_and_evaluation_scope(self):
        expected = {"pipeline.json", "experiment.json", "metrics.json", "diagnostics.json", "history.csv", "learning.svg", "decision.svg", "report.html"}
        self.assertEqual({p.name for p in self.output.iterdir()}, expected)
        self.assertIn("72 training / 24 validation / 24 held-out test examples", self.training.stdout)
        self.assertIn("evaluated after model selection", self.training.stdout)
        self.assertIn("one synthetic split", self.training.stdout)
        self.assertIn("Validation-selected decision threshold:", self.training.stdout)

    def test_decision_export_uses_validation_optimum_and_frozen_test_threshold(self):
        diagnostics = json.loads((self.output / "diagnostics.json").read_text())
        self.assertEqual(diagnostics, self.document["diagnostics"])
        for model in ("network", "baseline"):
            summary = self.metrics["diagnostics"][model]
            self.assertEqual(summary["threshold_selection"], diagnostics[model]["threshold_selection"])
            for split in ("validation", "test"):
                expected = {k: v for k, v in diagnostics[model][split].items()
                            if k not in ("roc", "precision_recall", "reliability_bins")}
                self.assertEqual(summary[split], expected)
            self.assertNotIn("validation_predictions", summary)
        self.assertEqual(diagnostics["selection_split"], "validation")
        network, mean, scale, _ = load_pipeline(self.output / "pipeline.json")
        data = self.document["data"]
        X, y = np.asarray(data["X"]), np.asarray(data["y"])[:, 0]
        val = data["val_idx"]
        p = network.predict_proba(standardize(X[val], mean, scale))[:, 0]
        # Independently enumerate prediction rules; do not reuse the selector.
        candidates = set(p.tolist()) | {0.5, float(np.nextafter(1., np.inf))}
        def key(t):
            predicted = p >= t
            cost = int(np.sum(predicted & (y[val] == 0))) + 3 * int(np.sum(~predicted & (y[val] == 1)))
            return cost, abs(t - .5), -t
        threshold = min(candidates, key=key)
        selection = diagnostics["network"]["threshold_selection"]
        self.assertEqual(selection["threshold"], threshold)
        self.assertEqual(selection["total_cost"], key(threshold)[0])
        test = data["test_idx"]
        predicted = network.predict_proba(standardize(X[test], mean, scale))[:, 0] >= threshold
        fp = int(np.sum(predicted & (y[test] == 0)))
        fn = int(np.sum(~predicted & (y[test] == 1)))
        frozen = diagnostics["network"]["test"]
        self.assertEqual(frozen["threshold"], threshold)
        self.assertEqual((frozen["fp"], frozen["fn"]), (fp, fn))
        self.assertEqual(frozen["mean_cost"], (fp + 3 * fn) / len(test))
        self.assertIn("--fp-cost 1.0 --fn-cost 3.0", self.report)

    def test_saved_pipeline_matches_every_test_prediction_and_metric(self):
        network, mean, scale, dataset = load_pipeline(self.output / "pipeline.json")
        records = self.document["test_predictions"]
        raw = np.array([row["input"] for row in records])
        y = np.array([[row["target"]] for row in records])
        transformed = standardize(raw, mean, scale)
        scores = network.predict_proba(transformed)[:, 0]
        np.testing.assert_array_equal(scores, [row["p"] for row in records])
        self.assertEqual(dataset, "moons")
        test = self.metrics["metrics"]["test"]
        self.assertAlmostEqual(test["loss"], network.loss(transformed, y), places=14)
        self.assertEqual(test["accuracy"], float(np.mean((scores >= .5) == y[:, 0])))
        selected = next(c for c in self.document["checkpoints"] if c["epoch"] == self.document["best_epoch"])
        for name, values in network.parameters.items():
            np.testing.assert_array_equal(selected["parameters"][name], values)

    def test_pipeline_statistics_come_only_from_saved_training_ids(self):
        data = self.document["data"]
        train = np.asarray(data["X"])[data["train_idx"]]
        _, mean, scale, _ = load_pipeline(self.output / "pipeline.json")
        np.testing.assert_allclose(mean, train.mean(axis=0, keepdims=True), rtol=0, atol=1e-14)
        np.testing.assert_allclose(scale, train.std(axis=0, keepdims=True), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(mean, data["mean"])
        np.testing.assert_array_equal(scale, data["scale"])

    def test_history_and_metrics_agree_on_selected_validation_checkpoint(self):
        with (self.output / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r["epoch"]) for r in rows], list(range(11)))
        self.assertTrue(all("test" not in key for key in rows[0]))
        selected = rows[self.metrics["best_epoch"]]
        self.assertEqual(float(selected["val_loss"]), min(float(r["val_loss"]) for r in rows))
        self.assertEqual(float(selected["val_loss"]), self.metrics["metrics"]["validation"]["loss"])

    def test_report_inline_code_matches_its_csp_hash_and_embedded_data(self):
        parser = ReportParser()
        parser.feed(self.report)
        executable = [s for s in parser.scripts if s["attrs"].get("type") != "application/json"]
        self.assertEqual(len(executable), 1)
        expected_hash = base64.b64encode(hashlib.sha256(executable[0]["text"].encode("utf-8")).digest()).decode("ascii")
        self.assertIn("script-src 'sha256-" + expected_hash + "'", parser.csp)
        self.assertIn("default-src 'none'", parser.csp)
        self.assertNotIn("'unsafe-inline'", parser.csp.split("script-src", 1)[1])
        self.assertEqual(parser.event_attributes, [])
        embedded = next(s for s in parser.scripts if s["attrs"].get("id") == "experiment-data")
        self.assertEqual(json.loads(embedded["text"]), self.document)

    def test_report_and_exportable_figures_need_no_remote_assets(self):
        parser = ReportParser()
        parser.feed(self.report)
        self.assertTrue(all("src" not in s["attrs"] for s in parser.scripts))
        for reference in parser.references:
            self.assertTrue(reference.startswith(("#", "data:")), reference)
        self.assertNotIn("@import", self.report)
        self.assertNotRegex(self.report, r"\$\{(?:DATASET|HIDDEN|CSS|SCRIPT|DATA|CONFIG|METRIC_CARDS)\}")
        for name in ("learning.svg", "decision.svg"):
            self.assertEqual(ET.parse(self.output / name).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            self.assertGreater((self.output / name).stat().st_size, 1000)

    def test_explain_applies_the_saved_standardizer_before_forward_pass(self):
        network = Network(hidden=2, seed=7)
        for parameter in network.parameters.values():
            parameter[:] = 0
        network.parameters["W1"][0, 0] = math.log(3) / 2
        network.parameters["W2"][0, 0] = 2
        path = self.base / "known-pipeline.json"
        save_pipeline(network, [[10., 20.]], [[2., 5.]], "moons", path)
        result = self.run_command("explain", "--model", str(path), "--input", "12", "20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Raw input = [[12. 20.]]", result.stdout)
        self.assertIn("Standardized input = [[1. 0.]]", result.stdout)
        self.assertIn("p = sigmoid(z2) = [[0.731059]]", result.stdout)

    def test_explain_sensitivity_and_local_approximation_match_independent_values(self):
        network = Network(hidden=1, seed=7)
        network.parameters = {"W1": np.array([[2.], [-1.]]), "b1": np.zeros((1, 1)),
                              "W2": np.array([[3.]]), "b2": np.zeros((1, 1))}
        path = self.base / "sensitivity-pipeline.json"
        save_pipeline(network, [[10., 20.]], [[2., 5.]], "moons", path)
        result = self.run_command("explain", "--model", str(path), "--input", "10", "20", "--delta", "0.01", "-0.02")
        self.assertEqual(result.returncode, 0, result.stderr)
        # At standardized origin: sigmoid'=1/4, tanh'=1; gradient=[.75,-.15].
        self.assertIn("dp/d(raw x) = [[ 0.75 -0.15]]", result.stdout)
        self.assertIn("First-order probability approximation = 0.510500000", result.stdout)
        actual = 1 / (1 + math.exp(-3 * math.tanh(.014)))
        self.assertIn("Actual displaced probability = {:.9f}".format(actual), result.stdout)
        invalid = self.run_command("explain", "--model", str(path), "--input", "10", "20", "--delta", "nan", "0")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("delta must contain two finite", invalid.stderr)
        self.assertNotIn("Traceback", invalid.stderr)

    def test_existing_experiment_directory_is_preserved(self):
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        result = self.run_command("experiment", "--samples", "120", "--epochs", "1", "--output", str(self.output))
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_invalid_experiment_configuration_leaves_no_output(self):
        for key, value in (("--samples", "20"), ("--l2", "-1"), ("--batch-size", "0"), ("--noise", "nan"), ("--fp-cost", "0"), ("--fn-cost", "inf")):
            output = self.base / (key[2:] + "-invalid")
            result = self.run_command("experiment", key, value, "--output", str(output))
            with self.subTest(key=key):
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_invalid_pipeline_is_a_readable_cli_error(self):
        path = self.base / "broken-pipeline.json"
        path.write_text('{"format":"neural-math-lab-pipeline-v2","model":{}}', encoding="utf-8")
        result = self.run_command("explain", "--model", str(path), "--input", "0", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
