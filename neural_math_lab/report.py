"""Generate a self-contained reading report and reusable scientific figures."""

import csv
import html
import json
import platform
from string import Template

import numpy as np

from .storage import new_output_directory, save_model


def write_report(network, X, y, history, output, config):
    """Write a new run; an existing directory is never overwritten.

    If any step fails, the partly written directory is removed so the path
    stays free for a corrected run.
    """
    with new_output_directory(output) as directory:
        _write_report_files(network, X, y, history, directory, config)


def _write_report_files(network, X, y, history, output, config):
    # Figures made without pyplot need no backend: the caller's backend and
    # open figures stay untouched when the report is written in-process.
    import matplotlib
    from matplotlib.figure import Figure

    scores = network.predict_proba(X)[:, 0]
    summary = {
        "task": "XOR truth-table learning",
        "evaluation": "training rows only; no held-out test set",
        "config": config,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
        "training_accuracy": history[-1]["accuracy"],
        "predictions": [{"input": row.tolist(), "target": int(target), "p": float(p), "class": int(p >= 0.5)} for row, target, p in zip(X, y[:, 0], scores)],
    }
    save_model(network, output / "model.json")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "accuracy"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    def save_svg(fig, name, title):
        # Matplotlib salts SVG ids with a random uuid unless svg.hashsalt is
        # set. Fixed salts make identical runs write identical files; one salt
        # per figure keeps ids distinct when both are inlined in report.html.
        with matplotlib.rc_context({"svg.hashsalt": "neural-math-lab-xor-" + name}):
            fig.savefig(output / name, format="svg", metadata={"Date": None, "Title": title})

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}):
        fig = Figure(figsize=(6.4, 4), layout="constrained")
        ax = fig.subplots()
        ax.plot([r["epoch"] for r in history], [r["loss"] for r in history], color="#087f8c", linewidth=2)
        ax.set(xlabel="Completed gradient updates", ylabel="Mean binary cross-entropy", title="Learning from four examples")
        ax.grid(alpha=0.15)
        save_svg(fig, "loss.svg", "XOR training loss")

        axis = np.linspace(-0.25, 1.25, 121)
        xx, yy = np.meshgrid(axis, axis)
        probability = network.predict_proba(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        fig = Figure(figsize=(6.4, 4), layout="constrained")
        ax = fig.subplots()
        mesh = ax.pcolormesh(xx, yy, probability, vmin=0, vmax=1, cmap="BrBG", shading="auto", rasterized=True)
        if probability.min() < 0.5 < probability.max():
            ax.contour(xx, yy, probability, levels=[0.5], colors="#222222", linewidths=1)
        for row, target in zip(X, y[:, 0]):
            ax.scatter(*row, marker="s" if target == 0 else "o", color="white", edgecolor="black", s=115, linewidth=1.5)
            ax.annotate("target " + str(int(target)), row, xytext=(8, 9), textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1})
        fig.colorbar(mesh, ax=ax, label="Model output p")
        ax.set(xlabel="Input x1", ylabel="Input x2", title="What the fitted network computes", aspect="equal")
        save_svg(fig, "decision.svg", "XOR fitted model output")

    def figure(name):
        path = output / name
        svg = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
        path.write_text(svg, encoding="utf-8")
        return svg[svg.index("<svg"):]

    rows = "".join("<tr><td>{}, {}</td><td>{}</td><td>{:.6f}</td><td>{}</td></tr>".format(int(r[0]), int(r[1]), int(t), p, int(p >= 0.5)) for r, t, p in zip(X, y[:, 0], scores))
    values = network.forward(np.array([[1.0, 0.0]]))
    trace = "\n".join(label + " = " + np.array2string(values[key], precision=6) for label, key in
                      (("z1 = x @ W1 + b1", "z1"), ("h = tanh(z1)", "h"), ("z2 = h @ W2 + b2", "z2"), ("p = sigmoid(z2)", "p")))
    weights = "\n\n".join(key + " =\n" + np.array2string(value, precision=6) for key, value in network._parameters().items())
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<title>Neural Math Lab — XOR experiment</title><style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#183433;background:#f4f5ef}*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:44px 24px 70px}.eyebrow{font-size:12px;letter-spacing:.16em;font-weight:750;color:#087f8c}h1{font-size:clamp(32px,6vw,60px);letter-spacing:-.045em;line-height:1.05;margin:18px 0}h2{font-size:23px;letter-spacing:-.02em}p{line-height:1.7;max-width:79ch}.lead{font-size:19px;color:#44605f}.metrics,.plots{display:grid;gap:18px;grid-template-columns:repeat(2,minmax(0,1fr))}.card,section{background:#fff;border:1px solid #dce3dc;border-radius:15px;padding:24px;margin:22px 0}.metrics .card{margin:0}.metric{font-size:36px;font-weight:750;color:#087f8c;letter-spacing:-.04em}.label{font-size:13px;color:#526865}.note{background:#e6efdf;border-left:4px solid #557945;padding:15px 18px;line-height:1.65}.plots section{margin:0;padding:15px}.plots svg{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #dde4df;padding:12px 9px}th{font-size:13px;color:#496462}pre{background:#edf2f0;padding:18px;border-radius:10px;overflow:auto;line-height:1.7;font-size:13px}code{font-family:ui-monospace,monospace}summary{cursor:pointer;padding:12px 0;font-weight:650}footer{color:#526865;font-size:13px;margin-top:30px}a{color:#086a79}li{line-height:1.7;margin:7px 0}@media(max-width:650px){main{padding:25px 16px}.metrics,.plots{grid-template-columns:1fr}section{padding:18px}th,td{padding:10px 5px;font-size:13px}}@media print{body{background:white}main{padding:0}.plots{grid-template-columns:1fr}details{display:block}section{break-inside:avoid}}</style>
<main><div class="eyebrow">NEURAL MATH LAB / YOUR FIRST EXPERIMENT</div>
<h1>A tiny network.<br>Every calculation visible.</h1>
<p class="lead">Two inputs, one hidden layer, and one question: are the bits different?</p>
<p>XOR returns 1 for (0, 1) and (1, 0), and 0 for (0, 0) and (1, 1). This report records an actual NumPy training run using manually written derivatives.</p>
<div class="metrics"><div class="card"><div class="label">FINAL MEAN LOSS · LOWER IS BETTER</div><div class="metric">${FINAL_LOSS}</div><div class="label">Started at ${INITIAL_LOSS}</div></div><div class="card"><div class="label">TRAINING ACCURACY ON FOUR EXAMPLES</div><div class="metric">${ACCURACY}</div><div class="label">${EPOCHS} full-batch gradient updates · ${HIDDEN} hidden neurons</div></div></div>
<p class="note">All four XOR rows were used for training. These results show fitting this truth table, not performance on unseen data. The output p is not calibrated confidence.</p>
<div class="plots"><section aria-label="Training loss figure">${LOSS_FIGURE}<p>The loss measures how well the probabilities match the targets. One epoch here is one update using all four examples.</p></section><section aria-label="Network prediction figure">${DECISION_FIGURE}<p>Squares mark target 0, circles target 1. The black curve, when present, is p = 0.5. Values between the four binary points illustrate the fitted function; they have no XOR ground truth.</p></section></div>
<section><div class="eyebrow">01 / CHECK THE ANSWERS</div><h2>What did the model learn?</h2><table><thead><tr><th>Inputs</th><th>Target</th><th>Output p</th><th>Class (p ≥ 0.5)</th></tr></thead><tbody>${ROWS}</tbody></table></section>
<section><div class="eyebrow">02 / FOLLOW ONE EXAMPLE</div><h2>Trace input (1, 0)</h2><p>Each column of W1 describes a hidden neuron. Multiply inputs by its weights, add its bias, and apply tanh. The output neuron combines those activations, then sigmoid maps the result into (0, 1), subject to floating-point rounding.</p><pre>${TRACE}</pre><details><summary>Inspect this run’s learned weights and biases</summary><pre>${WEIGHTS}</pre></details></section>
<section><div class="eyebrow">03 / HOW LEARNING HAPPENS</div><h2>The chain rule supplies the direction.</h2><pre>Z1 = X @ W1 + b1       H = tanh(Z1)
Z2 = H @ W2 + b2       P = sigmoid(Z2)
L  = mean(logaddexp(0, where(Y == 1, -Z2, Z2)))

dZ2 = where(Y == 1, -sigmoid(-Z2), sigmoid(Z2)) / N
dW2 = H.T @ dZ2        db2 = sum(dZ2, axis=0)
E   = exp(-abs(Z1))
dZ1 = (dZ2 @ W2.T) * (2 * E / (1 + E * E))**2
dW1 = X.T @ dZ1        db1 = sum(dZ1, axis=0)

parameter = parameter - learning_rate * gradient</pre><p>N is the number of examples, four here. In exact arithmetic the loss equals <code>mean(logaddexp(0, Z2) - Y * Z2)</code>, <code>dZ2</code> equals <code>(P - Y) / N</code>, and <code>(2 * E / (1 + E * E))**2</code> equals <code>1 - H**2</code>. The code evaluates these forms from Z2 and Z1, so a P that rounds to exactly 1 or an H that rounds to ±1 cannot turn a tiny loss or derivative into 0. A positive derivative means increasing that parameter locally increases the loss; subtracting the gradient takes a small step in the opposite direction. The learning rate determines the step size.</p><p>Read <code>docs/mathematics.md</code> in the repository for the full derivation, a worked numerical example, and the reason one linear classifier cannot solve XOR. Then try <code>docs/exercises.md</code>.</p></section>
<section><div class="eyebrow">04 / REPRODUCE AND EXPERIMENT</div><h2>Change one setting at a time.</h2><pre>${COMMAND}</pre><p>Use a new output directory for each run. Compare loss curves with learning rates 0.05 and 0.5, or change the number of hidden neurons. Failed learning is useful evidence too.</p><details><summary>Run configuration and environment</summary><pre>${CONFIG}</pre></details><p>Alongside this report: <code>model.json</code> contains the weights; <code>metrics.json</code> records the results; <code>history.csv</code> records every epoch. Both figures are separate SVG files as well as embedded here.</p></section>
<footer>Generated locally by Neural Math Lab · No remote assets, accounts, or model API. This report opens offline. Seeded runs are reproducible in the same numerical environment; small floating-point differences can occur across systems.</footer></main></html>"""
    replacements = {
        "FINAL_LOSS": "{:.6f}".format(history[-1]["loss"]),
        "INITIAL_LOSS": "{:.6f}".format(history[0]["loss"]),
        "ACCURACY": "{:.0%}".format(history[-1]["accuracy"]),
        "EPOCHS": str(config["epochs"]), "HIDDEN": str(config["hidden"]),
        "LOSS_FIGURE": figure("loss.svg"), "DECISION_FIGURE": figure("decision.svg"),
        "ROWS": rows, "TRACE": html.escape(trace), "WEIGHTS": html.escape(weights),
        "COMMAND": "python -m neural_math_lab train --epochs {epochs} --learning-rate {learning_rate} --hidden {hidden} --seed {seed} --output runs/reproduction".format(**config),
        "CONFIG": html.escape(json.dumps({"config": config, "environment": summary["environment"]}, indent=2)),
    }
    content = Template(content).substitute(replacements)
    (output / "report.html").write_text(content, encoding="utf-8")
