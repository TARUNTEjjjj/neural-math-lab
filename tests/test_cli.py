"""End-to-end checks use the real command-line entry point and saved artifacts."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from html.parser import HTMLParser
import io
import json
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
from neural_math_lab.core import Network, train, xor_data
from neural_math_lab.report import write_report
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


class XorReportText(HTMLParser):
    """Visible text of the XOR report's metric cards, answer rows and code blocks."""

    def __init__(self):
        super().__init__()
        self.metrics, self.labels, self.blocks, self.rows = [], [], [], []
        self.target = self.target_tag = None

    def handle_starttag(self, tag, attrs):
        css = dict(attrs).get("class")
        if tag == "tr":
            self.rows.append([])
            return
        if tag == "div" and css in ("metric", "label"):
            collection = self.metrics if css == "metric" else self.labels
        elif tag == "pre":
            collection = self.blocks
        elif tag == "td":
            collection = self.rows[-1]
        else:
            return
        collection.append("")
        self.target, self.target_tag = collection, tag

    def handle_endtag(self, tag):
        if tag == self.target_tag:
            self.target = self.target_tag = None

    def handle_data(self, data):
        if self.target is not None:
            self.target[-1] += data


def assert_shows(test, text, value, suffix="", scale=1):
    """A rendered number must equal value * scale at its displayed precision."""
    text = text.strip()
    test.assertTrue(text.endswith(suffix), text)
    number = text[:len(text) - len(suffix)]
    test.assertRegex(number, r"^[+-]?\d+(?:\.\d+)?$")
    places = len(number.partition(".")[2])
    test.assertLessEqual(abs(float(number) - value * scale), .5 * 10 ** -places + 1e-12,
                         "{!r} does not display {!r}".format(text, value * scale))


def assert_shows_array(test, text, values):
    """np.array2string(..., precision=6) text must show every value."""
    shown = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?", text)]
    expected = np.ravel(values)
    test.assertEqual(len(shown), expected.size, text)
    for number, value in zip(shown, expected):
        test.assertLessEqual(abs(number - value), 5.000001e-7 * max(1.0, abs(value)), text)


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
        # Loss and accuracy both change in these five updates, so a report
        # that shows the first epoch as the final one cannot pass by chance.
        cls.short_output = cls.base / "five-updates"
        cls.short_training = cls.run_command("train", "--epochs", "5", "--output", str(cls.short_output))
        if cls.short_training.returncode != 0:
            cls.directory.cleanup()
            raise AssertionError("Training command failed:\n" + cls.short_training.stdout + cls.short_training.stderr)

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

    def assert_report_matches_saved_results(self, directory, stdout=None):
        """Each number a reader sees must come from the saved model and metrics."""
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        network = load_model(directory / "model.json")
        page = XorReportText()
        page.feed((directory / "report.html").read_text(encoding="utf-8"))
        config, predictions = metrics["config"], metrics["predictions"]
        final_loss, accuracy = page.metrics
        assert_shows(self, final_loss, metrics["final_loss"])
        assert_shows(self, accuracy, metrics["training_accuracy"], "%", 100)
        started, = [label for label in page.labels if label.startswith("Started at ")]
        assert_shows(self, started[len("Started at "):], metrics["initial_loss"])
        self.assertIn("{} full-batch gradient updates · {} hidden neurons".format(config["epochs"], config["hidden"]), page.labels)
        rows = [row for row in page.rows if row]
        self.assertEqual(len(rows), len(predictions))
        for row, record in zip(rows, predictions):
            self.assertEqual(row[0], "{}, {}".format(*map(int, record["input"])))
            self.assertEqual((int(row[1]), int(row[3])), (record["target"], record["class"]))
            assert_shows(self, row[2], record["p"])
        trace, weights = page.blocks[:2]
        values = network.forward(np.array([[1.0, 0.0]]))
        labels = ("z1 = x @ W1 + b1 = ", "h = tanh(z1) = ", "z2 = h @ W2 + b2 = ", "p = sigmoid(z2) = ")
        parts = re.split("|".join(map(re.escape, labels)), trace)
        self.assertEqual((parts[0], len(parts)), ("", 5))
        for text, key in zip(parts[1:], ("z1", "h", "z2", "p")):
            assert_shows_array(self, text, values[key])
        self.assertEqual(len(weights.split("\n\n")), len(network.parameters))
        for block, (name, value) in zip(weights.split("\n\n"), network.parameters.items()):
            label, numbers = block.split(" =\n")
            self.assertEqual(label, name)
            assert_shows_array(self, numbers, value)
        self.assertIn("python -m neural_math_lab train --epochs {epochs} --learning-rate {learning_rate} "
                      "--hidden {hidden} --seed {seed} --output runs/reproduction".format(**config), page.blocks)
        if stdout is not None:
            printed = stdout.splitlines()
            initial, final = re.fullmatch(r"XOR training loss: (\S+) -> (\S+)", printed[0]).groups()
            assert_shows(self, initial, metrics["initial_loss"])
            assert_shows(self, final, metrics["final_loss"])
            shown, = re.fullmatch(r"Accuracy on the four training rows: (\S+)", printed[1]).groups()
            assert_shows(self, shown, metrics["training_accuracy"], "%", 100)
            for line, record in zip(printed[2:6], predictions):
                a, b, target, p, prediction = re.fullmatch(r"(\d) XOR (\d): target=(\d) p=(\S+) prediction=(\d)", line).groups()
                self.assertEqual([float(a), float(b)], record["input"])
                self.assertEqual((int(target), int(prediction)), (record["target"], record["class"]))
                assert_shows(self, p, record["p"])

    def test_report_and_printed_summary_show_the_saved_results(self):
        for directory, run in ((self.output, self.training), (self.short_output, self.short_training)):
            with self.subTest(directory=directory.name):
                self.assert_report_matches_saved_results(directory, run.stdout)

    def test_committed_example_report_shows_its_saved_results(self):
        self.assert_report_matches_saved_results(ROOT / "examples" / "xor")

    def test_failed_report_removes_its_partial_directory(self):
        # The four XOR rows give finite outputs, but the decision grid reaches
        # x1 = 1.25, where 1.25 * 1.5e308 overflows after model.json,
        # metrics.json, history.csv and loss.svg were written.
        script = """if True:
            import sys
            from neural_math_lab.core import Network, xor_data
            from neural_math_lab.report import write_report
            network = Network(hidden=2, seed=7)
            network.parameters["W1"][:] = [[1.5e308, 0.], [0., 0.]]
            X, y = xor_data()
            write_report(network, X, y, [{"epoch": 0, "loss": .5, "accuracy": .5}], sys.argv[1],
                         {"epochs": 0, "learning_rate": .5, "hidden": 2, "seed": 7})
        """
        output = self.base / "runs" / "overflowing-grid"
        result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=ROOT, env=self.environment,
                                text=True, capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: matrix multiplication produced a non-finite value", result.stderr)
        self.assertFalse(output.exists())
        # Without Matplotlib the writer fails after mkdir. Cleanup never
        # touches a directory the writer did not create.
        X, y = xor_data()
        arguments = (Network(hidden=2), X, y, [{"epoch": 0, "loss": .5, "accuracy": .5}])
        config = {"epochs": 0, "learning_rate": .5, "hidden": 2, "seed": 7}
        with patch.dict(sys.modules, {"matplotlib": None}), self.assertRaises(ImportError):
            write_report(*arguments, output, config)
        self.assertFalse(output.exists())
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            write_report(*arguments, self.output, config)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_missing_matplotlib_is_a_one_line_error_before_training(self):
        output = self.base / "no-matplotlib"
        printed, errors = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"matplotlib": None}), patch("neural_math_lab.cli.train") as training, \
                redirect_stdout(printed), redirect_stderr(errors):
            status = main(["train", "--epochs", "2", "--output", str(output)])
        self.assertEqual(status, 2)
        training.assert_not_called()
        self.assertEqual(printed.getvalue(), "")
        self.assertRegex(errors.getvalue(), r"\Aerror: Matplotlib could not be imported \([^\n]+\); "
                                            r"it is needed to draw the report\.[^\n]*\n\Z")
        self.assertFalse(output.exists())

    def test_identical_runs_write_identical_files(self):
        again = self.base / "same-settings"
        result = self.run_command("train", "--epochs", "10", "--hidden", "3", "--seed", "11", "--output", str(again))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in again.iterdir()), sorted(p.name for p in self.output.iterdir()))
        for path in sorted(self.output.iterdir()):
            with self.subTest(file=path.name):
                self.assertTrue((again / path.name).read_bytes() == path.read_bytes(), path.name + " differs between identical runs")
        # Both figures are inlined in report.html: every id they reference must be defined once there.
        report = (again / "report.html").read_text(encoding="utf-8")
        defined = re.findall(r'\bid="([^"]+)"', report)
        referenced = set(re.findall(r"url\(#([^)]+)\)", report)) | set(re.findall(r'href="#([^"]+)"', report))
        self.assertGreater(len(referenced), 2)
        for name in sorted(referenced):
            self.assertEqual(defined.count(name), 1, name)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_model_file_has_the_same_permissions_as_the_other_outputs(self):
        modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in self.output.iterdir()}
        self.assertEqual(set(modes.values()), {modes["metrics.json"]}, modes)

    def run_writer(self, script, output):
        result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=ROOT, env=self.environment,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_report_writer_leaves_the_callers_matplotlib_state_alone(self):
        # A learner may write a report from a session that is already plotting.
        script = """if True:
            import json, sys
            import matplotlib
            matplotlib.use("svg")
            import matplotlib.pyplot as plt
            figure = plt.figure()
            from neural_math_lab.core import Network, train, xor_data
            from neural_math_lab.report import write_report
            network = Network(hidden=2, seed=7)
            X, y = xor_data()
            history = train(network, X, y, epochs=2)
            write_report(network, X, y, history, sys.argv[1], {"epochs": 2, "learning_rate": .5, "hidden": 2, "seed": 7})
            print(json.dumps({"backend": matplotlib.get_backend(), "figures": plt.get_fignums(), "caller": figure.number}))
        """
        output = self.base / "in-process"
        state = json.loads(self.run_writer(script, output).splitlines()[-1])
        self.assertEqual(state["backend"], "svg")
        self.assertEqual(state["figures"], [state["caller"]], "The caller's figure must stay open.")
        self.assertTrue((output / "report.html").exists())

    def test_report_saves_a_network_with_list_parameters(self):
        script = """if True:
            import sys
            from neural_math_lab.core import Network, train, xor_data
            from neural_math_lab.report import write_report
            network = Network(hidden=2, seed=7)
            X, y = xor_data()
            history = train(network, X, y, epochs=3)
            network.parameters = {name: value.tolist() for name, value in network.parameters.items()}
            write_report(network, X, y, history, sys.argv[1], {"epochs": 3, "learning_rate": .5, "hidden": 2, "seed": 7})
        """
        output = self.base / "list-parameters"
        self.run_writer(script, output)
        network = Network(hidden=2, seed=7)
        X, y = xor_data()
        train(network, X, y, epochs=3)
        saved = load_model(output / "model.json")
        for name, value in network.parameters.items():
            np.testing.assert_array_equal(saved.parameters[name], value)
        self.assert_report_matches_saved_results(output)

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

    def test_underflowing_delta_term_explains_identically_under_strict_numpy_settings(self):
        # At b2 = 40 the output's slope p(1-p) is about 2.3e-18 and dp/dx1 about
        # 6.6e-19, so its product with a 1e-300 displacement, about 6.6e-319, is
        # below the smallest normal float64 (2.2e-308). That underflow is
        # rounding, as under NumPy's defaults, so strict settings must print
        # the same explanation.
        network = Network(hidden=2, seed=7)
        network.parameters["b2"][:] = 40.0
        path = self.base / "saturated-model.json"
        save_model(network, path)
        arguments = ["explain", "--model", str(path), "--input", "1", "0", "--delta", "1e-300", "0"]

        def explain(**settings):
            printed, errors = io.StringIO(), io.StringIO()
            with np.errstate(**settings), redirect_stdout(printed), redirect_stderr(errors):
                status = main(arguments)
            return status, printed.getvalue(), errors.getvalue()

        expected = explain()
        self.assertEqual((expected[0], expected[2]), (0, ""))
        self.assertIn("First-order probability approximation = 1.000000000", expected[1])
        self.assertEqual(explain(all="raise"), expected)

    def test_gradient_check_command_passes_with_small_absolute_error(self):
        result = self.run_command("gradient-check", "--hidden", "3", "--seed", "11")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary, end = json.JSONDecoder().raw_decode(result.stdout)
        self.assertLess(summary["max_abs_error"], 1e-7)
        self.assertIn("PASS", result.stdout[end:])

    def test_gradient_check_command_fails_for_a_wrong_derivative(self):
        correct = Network.gradients

        def doubled_first_layer(network, X, y):
            gradients = correct(network, X, y)
            gradients["W1"] = 2 * gradients["W1"]
            return gradients

        printed = io.StringIO()
        with patch.object(Network, "gradients", doubled_first_layer), redirect_stdout(printed):
            status = main(["gradient-check", "--hidden", "3", "--seed", "11"])
        self.assertEqual(status, 1)
        summary, end = json.JSONDecoder().raw_decode(printed.getvalue())
        self.assertGreater(summary["max_abs_error"], 1e-3)
        self.assertGreater(summary["per_parameter"]["W1"]["max_abs_error"], 1e-3)
        self.assertEqual(printed.getvalue()[end:].strip(), "FAIL: gradients disagree.")

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
