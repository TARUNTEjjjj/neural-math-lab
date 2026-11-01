"""Commands deliberately mirror the steps in docs/mathematics.md."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .core import Network, gradient_check, train, xor_data
from .storage import load_model


def parser():
    result = argparse.ArgumentParser(description="Train and inspect a tiny NumPy neural network.")
    commands = result.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Learn the four XOR rows and create an offline report.")
    fit.add_argument("--epochs", type=int, default=4000)
    fit.add_argument("--learning-rate", type=float, default=0.5)
    fit.add_argument("--hidden", type=int, default=4)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--output", type=Path, default=Path("runs/xor"), help="New output directory; existing directories are preserved.")
    check = commands.add_parser("gradient-check", help="Compare backpropagation with finite differences.")
    check.add_argument("--hidden", type=int, default=4)
    check.add_argument("--seed", type=int, default=7)
    inspect = commands.add_parser("explain", help="Print each layer's actual numbers for one input.")
    inspect.add_argument("--model", type=Path, required=True)
    inspect.add_argument("--input", type=float, nargs=2, required=True, metavar=("X1", "X2"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "gradient-check":
            network = Network(hidden=args.hidden, seed=args.seed)
            X, y = xor_data()
            errors = gradient_check(network, X, y)
            print(json.dumps(errors, indent=2, allow_nan=False))
            passed = errors["max_abs_error"] < 1e-7
            print("PASS: numerical and analytic gradients agree." if passed else "FAIL: gradients disagree.")
            return 0 if passed else 1
        if args.command == "explain":
            network = load_model(args.model)
            X = np.asarray([args.input], dtype=np.float64)
            values = network.forward(X)
            with np.printoptions(precision=6, suppress=True):
                print("x =", X)
                for name in ("W1", "b1", "W2", "b2"):
                    print(name + " =", network.parameters[name])
                for label, name in (("z1 = x @ W1 + b1", "z1"), ("h = tanh(z1)", "h"), ("z2 = h @ W2 + b2", "z2"), ("p = sigmoid(z2)", "p")):
                    print(label, "=", values[name])
            print("Predicted class at threshold 0.5:", int(values["p"][0, 0] >= 0.5))
            print("p is a model output, not calibrated confidence. XOR is defined only on binary inputs.")
            return 0
        if args.output.exists():
            raise ValueError("Output directory already exists. Choose a new --output path to preserve previous experiments.")
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
