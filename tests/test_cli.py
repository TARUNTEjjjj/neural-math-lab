"""End-to-end checks use the real command-line entry point and saved artifacts."""

import csv
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from neural_math_lab.core import Network, xor_data
from neural_math_lab.storage import load_model, save_model


ROOT = Path(__file__).resolve().parents[1]


class ResourceCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.scripts = []
        self.table_depth = 0
        self.rows_outside_tables = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table_depth += 1
        if tag == "tr" and self.table_depth == 0:
            self.rows_outside_tables += 1
        if tag == "script":
            self.scripts.append(dict(attrs))
        for key, value in attrs:
            if key in ("src", "href", "xlink:href", "data", "poster") and value:
                self.references.append(value)

    def handle_endtag(self, tag):
        if tag == "table":
            self.table_depth -= 1


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.base = Path(cls.directory.name)
        cls.environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(cls.base / "matplotlib"))
        cls.output = cls.base / "experiment"
        cls.training = cls.run_command("train", "--epochs", "10", "--hidden", "3", "--seed", "11", "--output", str(cls.output))
        if cls.training.returncode != 0:
            cls.directory.cleanup()
            raise AssertionError("Training command failed:\n" + cls.training.stdout + cls.training.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @classmethod
    def run_command(cls, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "neural_math_lab", *arguments],
            cwd=ROOT, env=cls.environment, text=True, capture_output=True, timeout=60,
        )

    def test_train_creates_complete_artifacts_and_reports_evaluation_scope(self):
        expected = {"model.json", "metrics.json", "history.csv", "loss.svg", "decision.svg", "report.html"}
        self.assertEqual({p.name for p in self.output.iterdir()}, expected)
        self.assertIn("four training rows", self.training.stdout)
        self.assertIn("not a held-out evaluation", self.training.stdout)
        self.assertIn(str(self.output / "report.html"), self.training.stdout)

    def test_saved_metrics_match_reloaded_model(self):
        metrics = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        network = load_model(self.output / "model.json")
        X, y = xor_data()
        scores = network.predict_proba(X)[:, 0]
        self.assertAlmostEqual(metrics["final_loss"], network.loss(X, y), places=14)
        self.assertEqual(metrics["training_accuracy"], float(np.mean((scores >= 0.5) == y[:, 0])))
        self.assertEqual(metrics["config"], {"epochs": 10, "learning_rate": 0.5, "hidden": 3, "seed": 11})
        self.assertIn("no held-out test set", metrics["evaluation"])
        for record, row, target, score in zip(metrics["predictions"], X, y[:, 0], scores):
            self.assertEqual(record["input"], row.tolist())
            self.assertEqual(record["target"], int(target))
            self.assertEqual(record["class"], int(score >= 0.5))
            self.assertEqual(record["p"], float(score))

    def test_history_includes_epoch_zero_and_matches_final_metrics(self):
        with (self.output / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        metrics = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual([int(row["epoch"]) for row in rows], list(range(11)))
        self.assertEqual(float(rows[0]["loss"]), metrics["initial_loss"])
        self.assertEqual(float(rows[-1]["loss"]), metrics["final_loss"])
        self.assertEqual(float(rows[-1]["accuracy"]), metrics["training_accuracy"])

    def test_figures_are_valid_svg_and_html_needs_no_remote_assets(self):
        for filename in ("loss.svg", "decision.svg"):
            path = self.output / filename
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(ET.parse(path).getroot().tag, "{http://www.w3.org/2000/svg}svg")
        report = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("TRAINING ACCURACY ON FOUR EXAMPLES", report)
        self.assertIn("not performance on unseen data", report)
        self.assertEqual(report.count("<svg"), 2)
        parser = ResourceCollector()
        parser.feed(report)
        self.assertEqual(parser.scripts, [])
        self.assertEqual(parser.rows_outside_tables, 0)
        for reference in parser.references:
            self.assertTrue(reference.startswith(("#", "data:")), reference)
        self.assertNotIn("@import", report)
        for placeholder in ("FINAL_LOSS", "INITIAL_LOSS", "LOSS_FIGURE", "DECISION_FIGURE"):
            self.assertNotIn(placeholder, report)

    def test_explain_shows_actual_layer_values_for_known_model(self):
        network = Network(hidden=2, seed=7)
        for parameter in network.parameters.values():
            parameter[:] = 0
        path = self.base / "known-model.json"
        save_model(network, path)
        result = self.run_command("explain", "--model", str(path), "--input", "0", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ("z1 = x @ W1 + b1", "h = tanh(z1)", "z2 = h @ W2 + b2", "p = sigmoid(z2) = [[0.5]]"):
            self.assertIn(expected, result.stdout)
        self.assertIn("Predicted class at threshold 0.5: 1", result.stdout)

    def test_gradient_check_command_passes_with_small_absolute_error(self):
        result = self.run_command("gradient-check", "--hidden", "3", "--seed", "11")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary, end = json.JSONDecoder().raw_decode(result.stdout)
        self.assertLess(summary["max_abs_error"], 1e-7)
        self.assertIn("PASS", result.stdout[end:])

    def test_existing_experiment_is_preserved(self):
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        result = self.run_command("train", "--epochs", "1", "--output", str(self.output))
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        after = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.assertEqual(before, after)

    def test_invalid_learning_rate_fails_without_creating_output(self):
        output = self.base / "invalid-rate"
        result = self.run_command("train", "--learning-rate", "-1", "--epochs", "1", "--output", str(output))
        self.assertEqual(result.returncode, 2)
        self.assertIn("learning_rate", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_invalid_model_is_reported_without_traceback(self):
        model = self.base / "broken.json"
        model.write_text('{"format":"unknown"}', encoding="utf-8")
        result = self.run_command("explain", "--model", str(model), "--input", "0", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_model_is_reported_without_traceback(self):
        result = self.run_command("explain", "--model", str(self.base / "missing.json"), "--input", "0", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
