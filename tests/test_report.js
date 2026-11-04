"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { infer, inputSensitivity, thresholdMetrics, sensitivityProbe } = require("../neural_math_lab/assets/experiment.js");

test("browser inference follows a hand-calculated standardized hidden layer", () => {
  const weights = { W1: [[1, -1], [0.5, 2]], b1: [[0.1, -0.2]], W2: [[0.7], [-0.4]], b2: [[0.3]] };
  const result = infer(weights, [3, 1], [[1, -1]], [[2, 4]]);
  const a = Math.tanh(1.35), b = Math.tanh(-0.2);
  const z = 0.7 * a - 0.4 * b + 0.3;
  assert.deepEqual(result.x, [1, 0.5]);
  assert.ok(Math.abs(result.h[0] - a) < 1e-14);
  assert.ok(Math.abs(result.h[1] - b) < 1e-14);
  assert.ok(Math.abs(result.p - 1 / (1 + Math.exp(-z))) < 1e-14);
});

test("linear baseline applies the identical training standardizer", () => {
  const result = infer({ W: [[2], [-3]], b: [[0.5]] }, [5, 7], [[1, 3]], [[2, 4]], true);
  assert.deepEqual(result.x, [2, 1]);
  assert.equal(result.z, 1.5);
  assert.deepEqual(result.h, []);
  assert.ok(Math.abs(result.p - 1 / (1 + Math.exp(-1.5))) < 1e-14);
});

test("extreme positive and negative logits remain finite", () => {
  for (const z of [-1000, 1000]) {
    const result = infer({ W: [[0], [0]], b: [[z]] }, [0, 0], [[0, 0]], [[1, 1]], true);
    assert.ok(Number.isFinite(result.p));
    assert.equal(result.p, z > 0 ? 1 : 0);
  }
});

for (const dataset of ["moons", "circles"]) {
  const file = path.join(__dirname, "..", "examples", dataset, "experiment.json");
  test(dataset + ": browser probabilities match every saved Python test prediction", () => {
    const run = JSON.parse(fs.readFileSync(file, "utf8"));
    const chosen = run.checkpoints.find((frame) => frame.epoch === run.best_epoch);
    assert.deepEqual(chosen.parameters, run.model.parameters);
    let correct = 0, baselineCorrect = 0;
    for (const row of run.test_predictions) {
      const p = infer(chosen.parameters, row.input, run.data.mean, run.data.scale).p;
      const baseline = infer(run.baseline.parameters, row.input, run.data.mean, run.data.scale, true).p;
      assert.ok(Math.abs(p - row.p) < 1e-12, "row " + row.id + " differs from Python");
      correct += Number(Number(p >= 0.5) === row.target);
      baselineCorrect += Number(Number(baseline >= 0.5) === row.target);
    }
    assert.equal(correct / run.test_predictions.length, run.metrics.test.accuracy);
    assert.equal(baselineCorrect / run.test_predictions.length, run.baseline.test.accuracy);
  });
}


test("raw-coordinate chain rule agrees with a hand-calculated network", () => {
  const weights = { W1: [[2], [-1]], b1: [[0]], W2: [[3]], b2: [[0]] };
  const mean = [[1, 2]], scale = [[2, 4]];
  assert.deepEqual(inputSensitivity(weights, [1, 2], mean, scale), [0.75, -0.1875]);
  const r = sensitivityProbe(weights, [1, 2], [0.02, -0.04], mean, scale);
  assert.equal(r.p, 0.5);
  assert.equal(r.approximation, 0.5225);
  assert.ok(Math.abs(r.actual - 0.5224780908622351) < 1e-14);
  assert.ok(Math.abs(r.absolute_error - 0.0000219091377649) < 1e-14);
});

test("raw input gradients agree with independent central differences", () => {
  const mean = [[1, -2]], scale = [[0.2, 4]];
  const models = [
    [{ W1: [[1, -1], [0.5, 2]], b1: [[0.1, -0.2]], W2: [[0.7], [-0.4]], b2: [[0.3]] }, false],
    [{ W: [[2], [-3]], b: [[0.5]] }, true]
  ];
  for (const [weights, linear] of models) for (const raw of [[1, -2], [1.15, 0.5], [0.8, -4]]) {
    const grad = inputSensitivity(weights, raw, mean, scale, linear);
    for (let i = 0; i < 2; i++) {
      const left = raw.slice(), right = raw.slice(), epsilon = 1e-6;
      left[i] -= epsilon; right[i] += epsilon;
      const numerical = (infer(weights, right, mean, scale, linear).p - infer(weights, left, mean, scale, linear).p) / (2 * epsilon);
      assert.ok(Math.abs(numerical - grad[i]) < 1e-9);
    }
  }
});

test("sensitivity keeps a large linear estimate visible rather than clipping", () => {
  const model = { W: [[8], [0]], b: [[0]] };
  const r = sensitivityProbe(model, [0, 0], [1, 0], [[0, 0]], [[1, 1]], true);
  assert.equal(r.approximation, 2.5);
  assert.ok(r.actual < 1 && r.actual > 0.99);
  assert.throws(() => sensitivityProbe(model, [0, 0], [NaN, 0], [[0, 0]], [[1, 1]], true), RangeError);
});

test("threshold costs follow the worked tied-score example", () => {
  const y = [0, 1, 0, 1], p = [0.1, 0.4, 0.4, 0.9];
  const high = thresholdMetrics(y, p, 0.5, 1, 3);
  assert.deepEqual([high.tp, high.fp, high.fn, high.tn], [1, 0, 1, 2]);
  assert.equal(high.mean_cost, 0.75);
  const lower = thresholdMetrics(y, p, 0.4, 1, 3);
  assert.deepEqual([lower.tp, lower.fp, lower.fn, lower.tn], [2, 1, 0, 1]);
  assert.equal(lower.mean_cost, 0.25);
  assert.equal(lower.precision, 2 / 3);
  assert.equal(lower.recall, 1);
});

test("threshold endpoints include exact zeros and ones and an all-negative rule", () => {
  const y = [0, 1], p = [0, 1];
  const all = thresholdMetrics(y, p, 0);
  assert.deepEqual([all.tp, all.fp, all.fn, all.tn], [1, 1, 0, 0]);
  assert.equal(thresholdMetrics(y, p, 1).accuracy, 1);
  const none = thresholdMetrics(y, p, 1 + Number.EPSILON);
  assert.deepEqual([none.tp, none.fp, none.fn, none.tn], [0, 0, 1, 1]);
  assert.equal(none.precision, 0);
  assert.equal(none.recall, 0);
});

test("invalid threshold inputs fail without silently changing decisions", () => {
  for (const args of [ [[], [], 0.5], [[1], [NaN], 0.5], [[2], [0.4], 0.5], [[1], [2], 0.5],
    [[1], [0.4], NaN], [[1], [0.4], -0.1], [[1], [0.4], 1.1], [[1], [0.4], 0.5, 0], [[1], [0.4], 0.5, 1, Infinity] ]) {
    assert.throws(() => thresholdMetrics(...args), RangeError);
  }
});

for (const dataset of ["moons", "circles"]) {
  test(dataset + ": browser raw-input derivatives match saved Python derivatives", () => {
    const run = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "examples", dataset, "experiment.json"), "utf8"));
    assert.ok(run.sensitivity_probes.length > 0);
    for (const row of run.sensitivity_probes) {
      const gradient = inputSensitivity(run.model.parameters, row.input, run.data.mean, run.data.scale);
      gradient.forEach((g, i) => assert.ok(Math.abs(g - row.gradient[i]) < 1e-11));
    }
  });
  test(dataset + ": validation exploration exactly reproduces Python selected costs", () => {
    const run = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "examples", dataset, "experiment.json"), "utf8"));
    for (const key of ["network", "baseline"]) {
      const d = run.diagnostics[key], rows = d.validation_predictions, policy = d.threshold_selection;
      assert.deepEqual(rows.map((r) => r.id), run.data.val_idx);
      const actual = thresholdMetrics(rows.map((r) => r.target), rows.map((r) => r.p), policy.threshold, policy.fp_cost, policy.fn_cost);
      for (const field of ["tp", "tn", "fp", "fn", "mean_cost", "total_cost", "precision", "recall", "f1"]) {
        assert.equal(actual[field], d.validation[field], key + " " + field);
      }
    }
  });
}
