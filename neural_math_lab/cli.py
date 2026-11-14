"""Commands deliberately mirror the steps in docs/mathematics.md."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .core import Network, gradient_check, train, xor_data
from .storage import load_for_explanation


def parser():
    result = argparse.ArgumentParser(description="Train and inspect a tiny NumPy neural network.")
    commands = result.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Learn the four XOR rows and create an offline report.")
    fit.add_argument("--epochs", type=int, default=4000)
    fit.add_argument("--learning-rate", type=float, default=0.5)
    fit.add_argument("--hidden", type=int, default=4)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--output", type=Path, default=Path("runs/xor"), help="New output directory; existing directories are preserved.")
    fetch = commands.add_parser("fetch-dataset", help="Import the pinned Iris data after checksum verification.")
    fetch.add_argument("--dataset", choices=("iris",), default="iris")
    fetch.add_argument("--output", type=Path, default=Path("datasets/iris"))
    experiment = commands.add_parser("experiment", help="Train moons, circles or imported Iris with validation, a held-out test set, and interactive replay.")
    experiment.add_argument("--dataset", choices=("moons", "circles", "iris"), default="moons")
    experiment.add_argument("--data-path", type=Path, help="Local verified Iris dataset directory; training never downloads.")
    experiment.add_argument("--samples", type=int, default=600)
    experiment.add_argument("--noise", type=float, default=0.15)
    experiment.add_argument("--hidden", type=int, default=16)
    experiment.add_argument("--seed", type=int, default=7)
    experiment.add_argument("--epochs", type=int, default=1200)
    experiment.add_argument("--learning-rate", type=float, default=0.01)
    experiment.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    experiment.add_argument("--batch-size", type=int, default=32)
    experiment.add_argument("--l2", type=float, default=0.001)
    experiment.add_argument("--patience", type=int, default=100)
    experiment.add_argument("--min-delta", type=float, default=1e-5)
    experiment.add_argument("--fp-cost", type=float, default=1.0, help="Positive false-positive cost for validation-only decision threshold selection (at most 1e6).")
    experiment.add_argument("--fn-cost", type=float, default=1.0, help="Positive false-negative cost for validation-only decision threshold selection (at most 1e6).")
    experiment.add_argument("--output", type=Path, help="New output directory (default: runs/<dataset>); existing directories are preserved.")
    check = commands.add_parser("gradient-check", help="Compare backpropagation with finite differences.")
    check.add_argument("--hidden", type=int, default=4)
    check.add_argument("--seed", type=int, default=7)
    inspect = commands.add_parser("explain", help="Print each layer's actual numbers for one input.")
    inspect.add_argument("--model", type=Path, required=True)
    inspect.add_argument("--input", type=float, nargs=2, required=True, metavar=("X1", "X2"))
    inspect.add_argument("--delta", type=float, nargs=2, metavar=("DX1", "DX2"), help="Compare a local linear approximation with the actual probability after this raw-input displacement.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch-dataset":
            from .resource_data import download_iris
            metadata = download_iris(args.output)
            print("Imported 150 verified Iris rows into", args.output)
            print("SHA-256:", metadata["sha256"])
            print("Two features: sepal length and width (cm). Class 1: setosa; class 0: other species.")
            return 0
        if args.command == "gradient-check":
            network = Network(hidden=args.hidden, seed=args.seed)
            X, y = xor_data()
            errors = gradient_check(network, X, y)
            print(json.dumps(errors, indent=2, allow_nan=False))
            passed = errors["max_abs_error"] < 1e-7
            print("PASS: numerical and analytic gradients agree." if passed else "FAIL: gradients disagree.")
            return 0 if passed else 1
        if args.command == "explain":
            from .datasets import standardize
            from .diagnostics import input_sensitivity
            network, mean, scale, dataset = load_for_explanation(args.model)
            raw = np.asarray([args.input], dtype=np.float64)
            X = standardize(raw, mean, scale)
            values = network.forward(X)
            with np.printoptions(precision=6, suppress=True):
                if dataset != "xor":
                    print("Raw input =", raw)
                    print("Training mean =", mean, "scale =", scale)
                    print("Standardized input =", X)
                print("x =", X)
                for name in ("W1", "b1", "W2", "b2"):
                    print(name + " =", network.parameters[name])
                for label, name in (("z1 = x @ W1 + b1", "z1"), ("h = tanh(z1)", "h"), ("z2 = h @ W2 + b2", "z2"), ("p = sigmoid(z2)", "p")):
                    print(label, "=", values[name])
                sensitivity = input_sensitivity(network, raw, mean, scale)
                print("dp/d(raw x) =", sensitivity)
                if args.delta is not None:
                    delta = np.asarray([args.delta], dtype=np.float64)
                    if not np.isfinite(delta).all():
                        raise ValueError("delta must contain two finite raw-input displacements")
                    with np.errstate(over="raise", invalid="raise"):
                        moved = raw + delta
                        approximation = float(values["p"][0, 0] + np.sum(sensitivity * delta))
                    actual = float(network.predict_proba(standardize(moved, mean, scale))[0, 0])
                    if not np.isfinite(approximation):
                        raise ValueError("local approximation overflowed; use a smaller displacement")
                    print("Raw displacement =", delta)
                    print("First-order probability approximation = {:.9f}".format(approximation))
                    print("Actual displaced probability = {:.9f}".format(actual))
                    print("Absolute approximation error = {:.9g}".format(abs(approximation - actual)))
                    print("A local approximation can leave [0, 1]; it is not clipped or a causal effect.")
            print("Predicted class at threshold 0.5:", int(values["p"][0, 0] >= 0.5))
            print("p is a model output, not calibrated confidence.")
            if dataset == "xor":
                print("XOR is defined only on binary inputs.")
            return 0
        if args.command == "experiment" and args.output is None:
            args.output = Path("runs") / args.dataset
        if args.output.exists():
            raise ValueError("Output directory already exists. Choose a new --output path to preserve previous experiments.")
        if args.command == "experiment":
            from .experiment import run_experiment
            from .experiment_report import write_experiment_report
            config = {name: value for name, value in vars(args).items() if name not in ("command", "output")}
            result = run_experiment(**config)
            write_experiment_report(result, args.output)
            print("{}: {} training / {} validation / {} held-out test examples".format(args.dataset.title(), len(result["data"]["train_idx"]), len(result["data"]["val_idx"]), len(result["data"]["test_idx"])))
            print("Selected epoch {} by validation loss; stopped at {} ({})".format(result["best_epoch"], result["stopped_epoch"], result["stop_reason"]))
            print("Held-out accuracy: neural {:.2%}; linear baseline {:.2%}".format(result["metrics"]["test"]["accuracy"], result["baseline"]["test"]["accuracy"]))
            print("The test set is evaluated after model selection. These are results on one " + ("small Iris split, not a general species classifier." if args.dataset == "iris" else "synthetic split."))
            decision = result["diagnostics"]["network"]
            print("Validation-selected decision threshold: {:.17g} (FP cost {}; FN cost {})".format(decision["threshold_selection"]["threshold"], args.fp_cost, args.fn_cost))
            print("Frozen-threshold held-out mean cost: {:.6f}".format(decision["test"]["mean_cost"]))
            print("Open the interactive offline report:", (args.output / "report.html").resolve())
            return 0
        network = Network(hidden=args.hidden, seed=args.seed)
        X, y = xor_data()
        history = train(network, X, y, epochs=args.epochs, learning_rate=args.learning_rate)
        from .report import write_report
        write_report(network, X, y, history, args.output, {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "hidden": args.hidden, "seed": args.seed,
        })
        print("XOR training loss: {:.6f} -> {:.6f}".format(history[0]["loss"], history[-1]["loss"]))
        print("Accuracy on the four training rows: {:.0%}".format(history[-1]["accuracy"]))
        for row, target, score in zip(X, y[:, 0], network.predict_proba(X)[:, 0]):
            print("{} XOR {}: target={} p={:.6f} prediction={}".format(int(row[0]), int(row[1]), int(target), score, int(score >= 0.5)))
        print("All four rows were used for training; this is not a held-out evaluation.")
        print("Open the local report:", (args.output / "report.html").resolve())
        return 0
    except (ValueError, OSError, FloatingPointError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
