"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { infer } = require("../neural_math_lab/assets/experiment.js");

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
