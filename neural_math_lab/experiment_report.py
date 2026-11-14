"""Offline experiment artifacts, with interactive replay of actual checkpoints."""

import base64
import csv
import hashlib
import html
import json
import platform
from pathlib import Path
from string import Template

import numpy as np

from .datasets import standardize
from .diagnostics import input_sensitivity
from .storage import model_document, save_pipeline


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def json_text(value):
    return json.dumps(serializable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _table(headers, rows, css=""):
    def cells(tag, values):
        return "".join("<{}>{}</{}>".format(tag, html.escape(str(v)), tag) for v in values)
    return '<table class="' + css + '"><thead><tr>' + cells("th", headers) + "</tr></thead><tbody>" + "".join("<tr>" + cells("td", row) + "</tr>" for row in rows) + "</tbody></table>"


def _diagnostic_tables(diagnostics):
    """Render numerical evidence even without JavaScript or a canvas."""
    def number(value, places=4):
        return "Undefined" if value is None else format(value, ".{}f".format(places))

    def threshold(value):
        return "Above 1 (all negative)" if value > 1 else str(value)

    summary, evidence = [], []
    for key, title in (("network", "Neural network"), ("baseline", "Linear baseline")):
        item = diagnostics[key]
        selected, test, val = item["threshold_selection"], item["test"], item["validation"]
        summary.append([title, threshold(selected["threshold"]), test["fp"], test["fn"],
                        number(test["mean_cost"]), number(test["accuracy"]),
                        number(test["roc_auc"]), number(test["average_precision"]), number(test["brier"])])
        content = '<div class="diagnostic-static"><h3>' + title + ' · saved validation diagnostics</h3>'
        content += '<div class="table-scroll">' + _table(["Selected threshold", "FP cost", "FN cost", "Validation mean cost", "Candidates"],
                          [[threshold(selected["threshold"]), selected["fp_cost"], selected["fn_cost"], number(selected["mean_cost"]), selected["candidate_count"]]]) + '</div>'
        content += '<p class="caption">ROC AUC: {} · Average precision: {} · Brier: {} · ECE: {}. Curves below use validation scores.</p>'.format(
            number(val["roc_auc"]), number(val["average_precision"]), number(val["brier"]), number(val["ece"]))
        content += '<h4>Reliability bins</h4><div class="table-scroll">' + _table(
            ["Bin (left inclusive)", "Count", "Mean p", "Positive fraction", "Absolute gap"],
            [["[{:.1f}, {:.1f}{}".format(b["lower"], b["upper"], "]" if i == len(val["reliability_bins"]) - 1 else ")"),
              b["count"], number(b["mean_probability"]), number(b["fraction_positive"]), number(b["absolute_gap"])]
             for i, b in enumerate(val["reliability_bins"])]) + '</div>'
        content += '<p class="caption">Empty bins have undefined means and are omitted from the plot. ECE is the count-weighted absolute bin gap.</p>'
        content += '<details><summary>ROC points</summary><div class="curve-points">' + _table(
            ["Threshold", "False-positive rate", "True-positive rate", "TP", "FP"],
            [[threshold(r["threshold"]), number(r["fpr"]), number(r["tpr"]), r["tp"], r["fp"]] for r in val["roc"]]) + '</div></details>'
        content += '<details><summary>Precision–recall points</summary><div class="curve-points">' + _table(
            ["Threshold", "Recall", "Precision", "TP", "FP"],
            [[threshold(r["threshold"]), number(r["recall"]), number(r["precision"]), r["tp"], r["fp"]] for r in val["precision_recall"]]) + '</div></details></div>'
        evidence.append(content)
    return _table(["Model", "Selected τ", "FP", "FN", "Mean cost", "Accuracy", "ROC AUC", "AP", "Brier"], summary), "".join(evidence)


def _figures(result, output, bounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, net, history = result["data"], result["network"], result["history"]
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "neural-math-lab-v2"}):
        fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], color="#087f78", label="Training BCE")
        ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], color="#b94d24", label="Validation BCE")
        ax.axvline(result["best_epoch"], color="#637d69", linestyle="--", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel="Mean binary cross-entropy", title="Model selection uses validation loss")
        ax.legend(frameon=False)
        ax.grid(alpha=0.12)
        fig.savefig(output / "learning.svg", metadata={"Date": None})
        plt.close(fig)
        xs, ys = np.meshgrid(np.linspace(bounds[0], bounds[1], 160), np.linspace(bounds[2], bounds[3], 120))
        probabilities = net.predict_proba(standardize(np.column_stack([xs.ravel(), ys.ravel()]), data["mean"], data["scale"])).reshape(xs.shape)
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
        mesh = ax.pcolormesh(xs, ys, probabilities, cmap="BrBG", vmin=0, vmax=1, shading="auto", rasterized=True)
        if probabilities.min() < 0.5 < probabilities.max():
            ax.contour(xs, ys, probabilities, levels=[0.5], colors="#183e3b", linewidths=1)
        ids = data["test_idx"]
        for label, marker, color in ((0, "s", "#b76a43"), (1, "o", "#00866c")):
            selected = ids[data["y"][ids, 0] == label]
            ax.scatter(data["X"][selected, 0], data["X"][selected, 1], marker=marker, s=27, c=color, edgecolors="white", linewidths=0.7, label="Test class " + str(label))
        ax.set(xlabel="Raw input x1", ylabel="Raw input x2", title="Validation-selected network · held-out test points")
        ax.legend(frameon=False, loc="upper right")
        fig.colorbar(mesh, ax=ax, label="Model output p")
        fig.savefig(output / "decision.svg", metadata={"Date": None})
        plt.close(fig)
    for name in ("learning.svg", "decision.svg"):
        path = output / name
        text = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
        path.write_text(text, encoding="utf-8")
    return matplotlib.__version__


def write_experiment_report(result, output):
    """Write a new run; an existing directory is never overwritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, config, network = result["data"], result["config"], result["network"]
    low, high = data["X"].min(axis=0), data["X"].max(axis=0)
    padding = np.maximum((high - low) * 0.12, 0.1)
    bounds = [float(low[0] - padding[0]), float(high[0] + padding[0]), float(low[1] - padding[1]), float(high[1] + padding[1])]
    plot_version = _figures(result, output, bounds)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": plot_version}
    document = {k: v for k, v in result.items() if k != "network"}
    document.update({"format": "neural-math-lab-experiment-v2", "bounds": bounds, "environment": environment, "model": model_document(network)})
    predictions = network.predict_proba(standardize(data["X"][data["test_idx"]], data["mean"], data["scale"]))[:, 0]
    document["test_predictions"] = [
        {"id": int(idx), "input": data["X"][idx].tolist(), "target": int(data["y"][idx, 0]), "p": float(p)}
        for idx, p in zip(data["test_idx"], predictions)
    ]
    probe_inputs = data["X"][data["test_idx"][:8]]
    document["sensitivity_probes"] = [
        {"input": row.tolist(), "gradient": gradient.tolist()}
        for row, gradient in zip(probe_inputs, input_sensitivity(network, probe_inputs, data["mean"], data["scale"]))
    ]
    (output / "diagnostics.json").write_text(json.dumps(serializable(result["diagnostics"]), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_pipeline(network, data["mean"], data["scale"], config["dataset"], output / "pipeline.json")
    (output / "experiment.json").write_text(json_text(document) + "\n", encoding="utf-8")
    metrics = {k: document[k] for k in ("config", "metrics", "baseline", "best_epoch", "stopped_epoch", "stop_reason", "seeds", "environment")}
    # Keep the quick summary readable. Full curves, bins, and row-level scores
    # live in diagnostics.json and the self-contained experiment document.
    diagnostics = document["diagnostics"]
    metrics["diagnostics"] = {k: diagnostics[k] for k in ("selection_split", "decision_rule", "bin_count")}
    for model in ("network", "baseline"):
        metrics["diagnostics"][model] = {"threshold_selection": diagnostics[model]["threshold_selection"]}
        for split in ("validation", "test"):
            metrics["diagnostics"][model][split] = {
                k: v for k, v in diagnostics[model][split].items()
                if k not in ("roc", "precision_recall", "reliability_bins")
            }
    (output / "metrics.json").write_text(json.dumps(serializable(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["history"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["history"])
    test = result["metrics"]["test"]
    baseline = result["baseline"]["test"]
    count = len(data["test_idx"])
    gap = (test["accuracy"] - baseline["accuracy"]) * 100
    card_data = [
        ("Held-out test accuracy", "{:.1%}".format(test["accuracy"]), "{} unseen examples · selected model".format(count)),
        ("Class 1 · F1 score", "{:.3f}".format(test["f1"]), "Combines precision and recall"),
        ("Against linear baseline", "{:+.1f} pp".format(gap), "Baseline accuracy: {:.1%}".format(baseline["accuracy"])),
        ("Selected epoch", str(result["best_epoch"]), "Lowest validation BCE · stopped at {}".format(result["stopped_epoch"])),
    ]
    cards = "".join('<div class="metric-card"><span>{}</span><strong>{}</strong><small>{}</small></div>'.format(*map(html.escape, card)) for card in card_data)
    comparison = _table(["Model", "Accuracy", "Precision", "Recall", "F1", "Brier ↓"], [
        [name, "{:.1%}".format(m["accuracy"]), "{:.3f}".format(m["precision"]), "{:.3f}".format(m["recall"]), "{:.3f}".format(m["f1"]), "{:.4f}".format(m["brier"])]
        for name, m in (("Neural network", test), ("Linear baseline", baseline))
    ], "comparison")
    cm = test["confusion_matrix"]
    confusion = '<table class="confusion"><caption>Actual ↓ / predicted →</caption><thead><tr><th></th><th>Class 0</th><th>Class 1</th></tr></thead><tbody><tr><th>Class 0</th><td class="correct">{}</td><td>{}</td></tr><tr><th>Class 1</th><td>{}</td><td class="correct">{}</td></tr></tbody></table>'.format(cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    split_table = _table(["Split", "Examples", "Accuracy", "BCE", "Role"], [
        [label, len(data[key]), "{:.1%}".format(result["metrics"][name]["accuracy"]), "{:.5f}".format(result["metrics"][name]["loss"]), role]
        for label, key, name, role in (("Training", "train_idx", "train", "Gradient updates + scaler"), ("Validation", "val_idx", "validation", "Choose weights and stop"), ("Test", "test_idx", "test", "Final evaluation only"))
    ])
    errors = [r for r in document["test_predictions"] if int(r["p"] >= 0.5) != r["target"]]
    error_table = _table(["Row ID", "x1", "x2", "Target", "Output p", "Predicted"], [[r["id"], "{:.4f}".format(r["input"][0]), "{:.4f}".format(r["input"][1]), r["target"], "{:.6f}".format(r["p"]), int(r["p"] >= 0.5)] for r in errors[:50]], "error-table") if errors else '<p class="empty-state">No misclassified examples in this test split. This does not imply a zero population error rate.</p>'
    if len(errors) > 50:
        error_table += '<p class="caption">Showing 50 of {} misclassified examples. Full test predictions are in experiment.json.</p>'.format(len(errors))
    diagnostic_summary, diagnostic_tables = _diagnostic_tables(result["diagnostics"])
    assets = Path(__file__).parent / "assets"
    script = "\n".join((assets / name).read_text(encoding="utf-8")
                       for name in ("experiment.js", "optimization.js", "optimization-ui.js"))
    script_hash = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    svg = (output / "learning.svg").read_text(encoding="utf-8")
    command_keys = ("dataset", "samples", "noise", "hidden", "seed", "epochs", "learning_rate", "optimizer", "batch_size", "l2", "patience", "min_delta", "fp_cost", "fn_cost")
    if config["dataset"] == "iris":
        command_keys = tuple(key for key in command_keys if key not in ("samples", "noise"))
    command = "python -m neural_math_lab experiment " + " ".join("--{} {}".format(key.replace("_", "-"), config[key]) for key in command_keys) + (" --data-path datasets/iris" if config["dataset"] == "iris" else "") + " --output runs/new-experiment"
    replacements = {
        "CSS": "\n".join((assets / name).read_text(encoding="utf-8")
                         for name in ("experiment.css", "optimization.css")),
        "OPTIMIZATION_LAB": (assets / "optimization.html").read_text(encoding="utf-8"),
        "SCRIPT": script, "SCRIPT_HASH": script_hash, "DATA": json_text(document).replace("<", "\\u003c"),
        "DATA_SCOPE": ("Iris observations: x₁ = sepal length (cm), x₂ = sepal width (cm); class 1 = setosa, class 0 = versicolor or virginica. This is a small educational split." if config["dataset"] == "iris" else "These synthetic results describe this split and seed."),
        "DATASET": html.escape(config["dataset"].title()), "HIDDEN": str(config["hidden"]),
        "TAGS": "".join("<span>{}</span>".format(html.escape(str(tag))) for tag in (config["optimizer"].upper(), "Batch " + str(config["batch_size"]), "λ " + str(config["l2"]), "Seed " + str(config["seed"]))),
        "METRIC_CARDS": cards, "STOP_SUMMARY": "Stopped at epoch " + str(result["stopped_epoch"]),
        "EARLY_STOP_TEXT": "This run stopped because: {}. The saved model restores epoch {}. The selected checkpoint is included in the replay.".format(html.escape(result["stop_reason"].replace("_", " ")), result["best_epoch"]),
        "STATIC_FIGURE": svg[svg.index("<svg"):], "COMPARISON_TABLE": comparison,
        "DIAGNOSTIC_SUMMARY": diagnostic_summary, "DIAGNOSTIC_TABLES": diagnostic_tables,
        "CONFUSION": confusion, "SPLIT_TABLE": split_table, "ERROR_TABLE": error_table,
        "REPRO_NOTE": ("The command runs locally in Python. Replace datasets/iris with the directory you imported, or first run: python -m neural_math_lab fetch-dataset --output datasets/iris. Training reads local data only." if config["dataset"] == "iris" else "The command runs locally in Python."),
        "REPRO_COMMAND": html.escape(command), "CONFIG": html.escape(json.dumps(serializable({"config": config, "seeds": result["seeds"], "environment": environment}), indent=2)),
    }
    report = Template((assets / "experiment.html").read_text(encoding="utf-8")).substitute(replacements)
    (output / "report.html").write_text(report, encoding="utf-8")
