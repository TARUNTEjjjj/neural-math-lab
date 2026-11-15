"""Exercise the experiment command and inspect the saved, offline report."""

import base64
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from neural_math_lab.cli import main
from neural_math_lab.core import Network
from neural_math_lab.datasets import standardize
from neural_math_lab.experiment import run_experiment
from neural_math_lab.experiment_report import write_experiment_report
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


class RenderedReport(HTMLParser):
    """Collect the text a reader sees in metric cards, pills, captions and tables."""

    def __init__(self):
        super().__init__()
        self.cards, self.pills, self.captions, self.tables = [], [], [], []
        self.card = self.table = self.row = self.target = self.target_tag = None
        self.section = "rows"

    def start_text(self, tag, container, key):
        container[key] = ""
        self.target, self.target_tag = (container, key), tag

    def handle_starttag(self, tag, attrs):
        css = dict(attrs).get("class") or ""
        if tag == "div" and css == "metric-card":
            self.card = {}
        elif self.card is not None and tag in ("span", "strong", "small"):
            self.start_text(tag, self.card, tag)
        elif tag == "span" and css == "pill":
            self.pills.append(None)
            self.start_text(tag, self.pills, -1)
        elif tag == "p" and css == "caption":
            self.captions.append(None)
            self.start_text(tag, self.captions, -1)
        elif tag == "table":
            self.table, self.section = {"class": css, "head": [], "rows": []}, "rows"
        elif self.table is not None and tag in ("thead", "tbody"):
            self.section = "head" if tag == "thead" else "rows"
        elif self.table is not None and tag == "tr":
            self.row = []
            self.table[self.section].append(self.row)
        elif self.row is not None and tag in ("th", "td"):
            self.row.append(None)
            self.start_text(tag, self.row, -1)

    def handle_endtag(self, tag):
        if tag == self.target_tag:
            self.target = self.target_tag = None
        if tag == "small" and self.card is not None:
            self.cards.append(self.card)
            self.card = None
        elif tag == "tr":
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None

    def handle_data(self, data):
        if self.target is not None:
            container, key = self.target
            container[key] += data

    def select(self, *headers):
        return [t for t in self.tables if t["head"] and t["head"][0] == list(headers)]


def assert_shows(test, text, value, suffix="", scale=1):
    """A rendered number must equal value * scale at its displayed precision."""
    text = text.strip()
    if value is None:
        test.assertEqual(text, "Undefined")
        return
    test.assertTrue(text.endswith(suffix), text)
    number = text[:len(text) - len(suffix)]
    test.assertRegex(number, r"^[+-]?\d+(?:\.\d+)?$")
    places = len(number.partition(".")[2])
    test.assertLessEqual(abs(float(number) - value * scale), .5 * 10 ** -places + 1e-12,
                         "{!r} does not display {!r}".format(text, value * scale))


def assert_threshold(test, text, value):
    if value > 1:
        test.assertEqual(text, "Above 1 (all negative)")
    else:
        test.assertEqual(float(text), value)


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
        # Training, validation and test results all differ in this longer run,
        # so a report or summary that shows the wrong split cannot pass by chance.
        cls.selected_output = cls.base / "selected-epoch"
        cls.selected = cls.run_command("experiment", "--samples", "120", "--epochs", "40", "--seed", "11", "--output", str(cls.selected_output))
        if cls.selected.returncode != 0:
            cls.directory.cleanup()
            raise AssertionError("Experiment failed:\n" + cls.selected.stdout + cls.selected.stderr)

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

    def assert_report_matches_saved_results(self, directory, stdout=None):
        """Each number a reader sees must come from the saved split it names."""
        def read(name):
            return json.loads((directory / name).read_text(encoding="utf-8"))
        metrics, diagnostics, document = read("metrics.json"), read("diagnostics.json"), read("experiment.json")
        report = (directory / "report.html").read_text(encoding="utf-8")
        page = RenderedReport()
        page.feed(report)
        test, baseline, data = metrics["metrics"]["test"], metrics["baseline"]["test"], document["data"]

        cards = {card["span"]: card for card in page.cards}
        self.assertEqual(len(page.cards), 4)
        accuracy_card = cards["Held-out test accuracy"]
        assert_shows(self, accuracy_card["strong"], test["accuracy"], "%", 100)
        self.assertEqual(accuracy_card["small"], "{} unseen examples · selected model".format(len(data["test_idx"])))
        assert_shows(self, cards["Class 1 · F1 score"]["strong"], test["f1"])
        gap_card = cards["Against linear baseline"]
        assert_shows(self, gap_card["strong"], test["accuracy"] - baseline["accuracy"], " pp", 100)
        assert_shows(self, gap_card["small"].replace("Baseline accuracy: ", ""), baseline["accuracy"], "%", 100)
        self.assertEqual(cards["Selected epoch"]["strong"], str(metrics["best_epoch"]))
        self.assertEqual(cards["Selected epoch"]["small"], "Lowest validation BCE · stopped at {}".format(metrics["stopped_epoch"]))
        self.assertIn("Stopped at epoch {}".format(metrics["stopped_epoch"]), page.pills)
        self.assertIn("The saved model restores epoch {}.".format(metrics["best_epoch"]), report)

        comparison, = page.select("Model", "Accuracy", "Precision", "Recall", "F1", "Brier ↓")
        self.assertEqual([row[0] for row in comparison["rows"]], ["Neural network", "Linear baseline"])
        for row, model in zip(comparison["rows"], (test, baseline)):
            assert_shows(self, row[1], model["accuracy"], "%", 100)
            for cell, key in zip(row[2:], ("precision", "recall", "f1", "brier")):
                assert_shows(self, cell, model[key])
        confusion, = [table for table in page.tables if table["class"] == "confusion"]
        self.assertEqual([row[0] for row in confusion["rows"]], ["Class 0", "Class 1"])
        self.assertEqual([[int(cell) for cell in row[1:]] for row in confusion["rows"]], test["confusion_matrix"])
        splits, = page.select("Split", "Examples", "Accuracy", "BCE", "Role")
        self.assertEqual([row[0] for row in splits["rows"]], ["Training", "Validation", "Test"])
        for row, ids, name in zip(splits["rows"], ("train_idx", "val_idx", "test_idx"), ("train", "validation", "test")):
            self.assertEqual(int(row[1]), len(data[ids]))
            assert_shows(self, row[2], metrics["metrics"][name]["accuracy"], "%", 100)
            assert_shows(self, row[3], metrics["metrics"][name]["loss"])

        frozen_table, = page.select("Model", "Selected τ", "FP", "FN", "Mean cost", "Accuracy", "ROC AUC", "AP", "Brier")
        selections = page.select("Selected threshold", "FP cost", "FN cost", "Validation mean cost", "Candidates")
        bins = page.select("Bin (left inclusive)", "Count", "Mean p", "Positive fraction", "Absolute gap")
        rocs = page.select("Threshold", "False-positive rate", "True-positive rate", "TP", "FP")
        curves = page.select("Threshold", "Recall", "Precision", "TP", "FP")
        captions = [text for text in page.captions if text.startswith("ROC AUC: ")]
        models = ("network", "baseline")
        for collection in (frozen_table["rows"], selections, bins, rocs, curves, captions):
            self.assertEqual(len(collection), len(models))
        for model, frozen_row, selection, bin_table, roc, curve, caption in zip(
                models, frozen_table["rows"], selections, bins, rocs, curves, captions):
            chosen, frozen, validation = (diagnostics[model][key] for key in ("threshold_selection", "test", "validation"))
            self.assertEqual(metrics["diagnostics"][model]["threshold_selection"], chosen)
            assert_threshold(self, frozen_row[1], chosen["threshold"])
            self.assertEqual((int(frozen_row[2]), int(frozen_row[3])), (frozen["fp"], frozen["fn"]))
            for cell, key in zip(frozen_row[4:], ("mean_cost", "accuracy", "roc_auc", "average_precision", "brier")):
                assert_shows(self, cell, frozen[key])
            row, = selection["rows"]
            assert_threshold(self, row[0], chosen["threshold"])
            self.assertEqual((float(row[1]), float(row[2]), int(row[4])), (chosen["fp_cost"], chosen["fn_cost"], chosen["candidate_count"]))
            assert_shows(self, row[3], chosen["mean_cost"])
            shown = re.fullmatch(r"ROC AUC: (\S+) · Average precision: (\S+) · Brier: (\S+) · ECE: (\S+)\. Curves below use validation scores\.", caption)
            for text, key in zip(shown.groups(), ("roc_auc", "average_precision", "brier", "ece")):
                assert_shows(self, text, validation[key])
            self.assertEqual(len(bin_table["rows"]), len(validation["reliability_bins"]))
            for row, expected in zip(bin_table["rows"], validation["reliability_bins"]):
                self.assertEqual(int(row[1]), expected["count"])
                for cell, key in zip(row[2:], ("mean_probability", "fraction_positive", "absolute_gap")):
                    assert_shows(self, cell, expected[key])
            for table, points, rates in ((roc, validation["roc"], ("fpr", "tpr")),
                                         (curve, validation["precision_recall"], ("recall", "precision"))):
                self.assertEqual(len(table["rows"]), len(points))
                for row, point in zip(table["rows"], points):
                    assert_threshold(self, row[0], point["threshold"])
                    assert_shows(self, row[1], point[rates[0]])
                    assert_shows(self, row[2], point[rates[1]])
                    self.assertEqual((int(row[3]), int(row[4])), (point["tp"], point["fp"]))

        errors = [r for r in document["test_predictions"] if int(r["p"] >= .5) != r["target"]]
        error_tables = [table for table in page.tables if table["class"] == "error-table"]
        self.assertEqual(len(error_tables), 1 if errors else 0)
        for row, record in zip(error_tables[0]["rows"] if errors else [], errors):
            self.assertEqual((int(row[0]), int(row[3]), int(row[5])), (record["id"], record["target"], int(record["p"] >= .5)))
            for cell, value in zip((row[1], row[2], row[4]), record["input"] + [record["p"]]):
                assert_shows(self, cell, value)
        if errors:
            self.assertEqual(len(error_tables[0]["rows"]), min(len(errors), 50))

        if stdout is not None:
            printed = stdout.splitlines()
            self.assertIn("{}: {} training / {} validation / {} held-out test examples".format(
                metrics["config"]["dataset"].title(), len(data["train_idx"]), len(data["val_idx"]), len(data["test_idx"])), printed)
            self.assertIn("Selected epoch {} by validation loss; stopped at {} ({})".format(
                metrics["best_epoch"], metrics["stopped_epoch"], metrics["stop_reason"]), printed)
            value = {line.split(": ")[0]: line.split(": ", 1)[1] for line in printed if ": " in line}
            neural, linear = re.fullmatch(r"neural (\S+); linear baseline (\S+)", value["Held-out accuracy"]).groups()
            assert_shows(self, neural, test["accuracy"], "%", 100)
            assert_shows(self, linear, baseline["accuracy"], "%", 100)
            network_choice = diagnostics["network"]["threshold_selection"]
            threshold = re.fullmatch(r"(\S+) \(FP cost (\S+); FN cost (\S+)\)", value["Validation-selected decision threshold"]).groups()
            self.assertEqual(tuple(map(float, threshold)),
                             (network_choice["threshold"], network_choice["fp_cost"], network_choice["fn_cost"]))
            assert_shows(self, value["Frozen-threshold held-out mean cost"], diagnostics["network"]["test"]["mean_cost"])

    def test_report_and_printed_summary_show_the_saved_results(self):
        for directory, run in ((self.output, self.training), (self.selected_output, self.selected)):
            with self.subTest(directory=directory.name):
                self.assert_report_matches_saved_results(directory, run.stdout)

    def test_committed_example_reports_show_their_saved_results(self):
        for dataset in ("moons", "circles"):
            with self.subTest(dataset=dataset):
                self.assert_report_matches_saved_results(ROOT / "examples" / dataset)

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

    def test_report_embeds_gradient_lab_assets_and_preserves_saved_model(self):
        assets = ROOT / "neural_math_lab" / "assets"
        for name in ("optimization.js", "optimization-ui.js", "optimization.css"):
            self.assertIn((assets / name).read_text(encoding="utf-8"), self.report)
        self.assertIn('id="gradient-lab"', self.report)
        self.assertIn('id="optimization-apply"', self.report)
        self.assertIn('id="optimization-undo"', self.report)
        self.assertIn('id="optimization-history"', self.report)
        self.assertIn("temporary copy in this tab", self.report)
        self.assertIn("No test or validation rows enter these calculations", self.report)
        self.assertNotIn("${OPTIMIZATION_LAB}", self.report)
        network, _, _, _ = load_pipeline(self.output / "pipeline.json")
        for name, values in network.parameters.items():
            np.testing.assert_array_equal(values, self.document["model"]["parameters"][name])

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

    def test_failed_report_leaves_no_directory_to_block_a_rerun(self):
        output = self.base / "runs" / "retry"
        result = run_experiment(samples=120, epochs=2, hidden=3, seed=11)
        # Without Matplotlib the writer fails after mkdir.
        with patch.dict(sys.modules, {"matplotlib": None}), self.assertRaises(ImportError):
            write_experiment_report(result, output)
        self.assertFalse(output.exists())
        # A later failure, after figures and diagnostics were written.
        def stub_figures(result, directory, bounds):
            for name in ("learning.svg", "decision.svg"):
                (directory / name).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n', encoding="utf-8")
            return "stub"
        with patch("neural_math_lab.experiment_report._figures", stub_figures), \
                patch("neural_math_lab.experiment_report.save_pipeline", side_effect=OSError("No space left on device")), \
                self.assertRaisesRegex(OSError, "No space left"):
            write_experiment_report(result, output)
        self.assertFalse(output.exists())
        # Cleanup never touches a directory the writer did not create.
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            write_experiment_report(result, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_missing_matplotlib_is_a_one_line_error_before_training(self):
        output = self.base / "runs" / "no-matplotlib"
        printed, errors = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"matplotlib": None}), \
                patch("neural_math_lab.experiment.run_experiment") as training, \
                redirect_stdout(printed), redirect_stderr(errors):
            status = main(["experiment", "--samples", "120", "--epochs", "2", "--hidden", "3", "--output", str(output)])
        self.assertEqual(status, 2)
        training.assert_not_called()
        self.assertEqual(printed.getvalue(), "")
        self.assertRegex(errors.getvalue(), r"\Aerror: Matplotlib could not be imported \([^\n]+\); "
                                            r"it is needed to draw the report\.[^\n]*\n\Z")
        self.assertFalse(output.exists())

    def test_report_writer_leaves_the_callers_matplotlib_state_alone(self):
        # A learner may write a report from a session that is already plotting.
        script = """if True:
            import json, sys
            import matplotlib
            matplotlib.use("svg")
            import matplotlib.pyplot as plt
            figure = plt.figure()
            from neural_math_lab.experiment import run_experiment
            from neural_math_lab.experiment_report import write_experiment_report
            write_experiment_report(run_experiment(samples=120, epochs=2, hidden=3, seed=11), sys.argv[1])
            print(json.dumps({"backend": matplotlib.get_backend(), "figures": plt.get_fignums(), "caller": figure.number}))
        """
        output = self.base / "in-process"
        result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=ROOT, env=self.environment,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(state["backend"], "svg")
        self.assertEqual(state["figures"], [state["caller"]], "The caller's figure must stay open.")
        self.assertTrue((output / "report.html").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_pipeline_file_has_the_same_permissions_as_the_other_outputs(self):
        modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in self.output.iterdir()}
        self.assertEqual(set(modes.values()), {modes["metrics.json"]}, modes)

    def test_invalid_experiment_configuration_leaves_no_output(self):
        for key, value in (("--samples", "20"), ("--l2", "-1"), ("--batch-size", "0"), ("--noise", "nan"), ("--fp-cost", "0"), ("--fn-cost", "inf")):
            output = self.base / (key[2:] + "-invalid")
            result = self.run_command("experiment", key, value, "--output", str(output))
            with self.subTest(key=key):
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_diverging_learning_rate_is_a_readable_cli_error(self):
        output = self.base / "diverged"
        result = self.run_command("experiment", "--samples", "120", "--optimizer", "sgd",
                                  "--learning-rate", "1000", "--l2", "1", "--output", str(output))
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stderr, r"Training diverged at epoch [1-9]")
        self.assertIn("reduce the learning rate or the L2 coefficient", result.stderr)
        self.assertNotIn("smaller weights", result.stderr)
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
