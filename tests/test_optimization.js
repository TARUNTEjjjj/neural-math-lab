"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { objectiveAndGradients, parameterKeys, gradientCheck, proposeStep, lossSlice } = require("../neural_math_lab/assets/optimization.js");
const clone = (v) => JSON.parse(JSON.stringify(v));
const near = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const model = { W1: [[.3, -.5], [.7, .2]], b1: [[.1, -.2]], W2: [[.8], [-.6]], b2: [[.15]] };
const batch = { X: [[2, -1], [0, 3], [1, 1]], y: [0, 1, 1], mean: [[1, 1]], scale: [[2, 4]], l2: .17 };

test("hand-computed single neuron derivatives and unpenalized biases", () => {
  const weights = { W1: [[2], [-1]], b1: [[0]], W2: [[3]], b2: [[0]] };
  const data = { X: [[1, 2]], y: [1], mean: [[1, 2]], scale: [[2, 4]], l2: .1 };
  const r = objectiveAndGradients(weights, data);
  near(r.bce, Math.log(2)); near(r.penalty, .7); near(r.loss, Math.log(2) + .7);
  assert.deepEqual(r.gradients, { W1: [[.2], [-.1]], b1: [[-1.5]], W2: [[.30000000000000004]], b2: [[-.5]] });
});

test("all analytic derivatives agree with independently perturbed objectives", () => {
  for (const c of parameterKeys(model)) {
    const left = clone(model), right = clone(model), epsilon = 1e-5;
    left[c.name][c.row][c.col] -= epsilon; right[c.name][c.row][c.col] += epsilon;
    const numerical = (objectiveAndGradients(right, batch).loss - objectiveAndGradients(left, batch).loss) / (2 * epsilon);
    const analytic = objectiveAndGradients(model, batch).gradients[c.name][c.row][c.col];
    near(analytic, numerical, 3e-10);
    const check = gradientCheck(model, batch, c, epsilon);
    near(check.analytic, analytic); near(check.numerical, numerical); assert.ok(check.absoluteError < 3e-10);
  }
});

test("saturated hidden neurons preserve representable training derivatives", () => {
  const weights = {W1: [[0], [0]], b1: [[20]], W2: [[1]], b2: [[0]]};
  const data = {X: [[1, 0]], y: [0], mean: [[0, 0]], scale: [[1, 1]], l2: 0};
  const gradient = objectiveAndGradients(weights, data).gradients.W1[0][0];
  const expected = (1 / (1 + Math.exp(-1))) / Math.cosh(20) ** 2;
  assert.ok(gradient > 0);
  assert.ok(Math.abs(gradient / expected - 1) < 1e-14);
});

test("mean gradients are invariant to duplicated training rows", () => {
  const original = objectiveAndGradients(model, batch);
  const doubled = objectiveAndGradients(model, { ...batch, X: batch.X.concat(batch.X), y: batch.y.concat(batch.y) });
  near(original.loss, doubled.loss);
  for (const c of parameterKeys(model)) near(original.gradients[c.name][c.row][c.col], doubled.gradients[c.name][c.row][c.col]);
});

test("zero and saturated logits have stable finite BCE and derivatives", () => {
  for (const z of [0, -1000, 1000]) for (const y of [0, 1]) {
    const weights = { W1: [[0], [0]], b1: [[0]], W2: [[0]], b2: [[z]] };
    const r = objectiveAndGradients(weights, { X: [[0, 0]], y: [[y]], mean: [[0, 0]], scale: [[1, 1]] });
    near(r.loss, z === 0 ? Math.log(2) : Number((z > 0) !== Boolean(y)) * 1000);
    assert.ok(Number.isFinite(r.gradientNorm));
  }
});

test("SGD proposals match arithmetic for one coordinate or all parameters", () => {
  const before = objectiveAndGradients(model, batch), coordinate = { name: "W1", row: 0, col: 1 }, rate = .1;
  for (const selected of [coordinate, null]) {
    const proposed = proposeStep(model, batch, rate, selected);
    for (const c of parameterKeys(model)) {
      const moves = !selected || (c.name === selected.name && c.row === selected.row && c.col === selected.col);
      near(proposed.parameters[c.name][c.row][c.col], model[c.name][c.row][c.col] - (moves ? rate * before.gradients[c.name][c.row][c.col] : 0));
    }
    near(proposed.actualChange, proposed.after.loss - before.loss);
    assert.ok(proposed.after.loss < before.loss);
    assert.ok(proposed.predictedChange < 0);
  }
  assert.deepEqual(proposeStep(model, batch, 0).parameters, model);
});

test("correct saturated outputs retain tiny derivatives instead of a false stationary point", () => {
  for (const z of [40, 100, -40]) {
    const y = z > 0 ? 1 : 0, h = Math.tanh(.5);
    const weights = { W1: [[0], [0]], b1: [[.5]], W2: [[1]], b2: [[z - h]] };
    const result = objectiveAndGradients(weights, { X: [[1, -2]], y: [y], mean: [[0, 0]], scale: [[1, 1]] });
    const tail = Math.exp(-Math.abs(z)), dz = (1 - 2 * y) * tail / (1 + tail);
    const expected = { W1: [[(1 - h * h) * dz], [-2 * (1 - h * h) * dz]],
      b1: [[(1 - h * h) * dz]], W2: [[h * dz]], b2: [[dz]] };
    for (const c of parameterKeys(weights)) near(result.gradients[c.name][c.row][c.col] / expected[c.name][c.row][c.col], 1, 1e-14);
    near(result.bce / Math.log1p(tail), 1, 1e-14);
    assert.ok(result.gradientNorm > 0);
    const check = gradientCheck(weights, { X: [[1, -2]], y: [y], mean: [[0, 0]], scale: [[1, 1]] }, { name: "b2", row: 0, col: 0 }, .0001);
    near(check.analytic / check.numerical, 1, 1e-7);
  }
});

test("a hidden unit whose tanh rounds to +-1 keeps tiny weight and bias derivatives", () => {
  for (const a of [20, -20]) {
    const h = Math.tanh(a), slope = 1 / Math.cosh(a) ** 2, dz = -1 / (1 + Math.exp(h));
    assert.equal(h * h, 1);
    const weights = { W1: [[0], [0]], b1: [[a]], W2: [[1]], b2: [[0]] };
    const result = objectiveAndGradients(weights, { X: [[1, -2]], y: [1], mean: [[0, 0]], scale: [[1, 1]] });
    const expected = { W1: [[slope * dz], [-2 * slope * dz]], b1: [[slope * dz]], W2: [[h * dz]], b2: [[dz]] };
    for (const c of parameterKeys(weights)) near(result.gradients[c.name][c.row][c.col] / expected[c.name][c.row][c.col], 1, 1e-14);
  }
});

test("large step can increase true loss despite negative first-order change", () => {
  const weights = { W1: [[0], [0]], b1: [[0]], W2: [[0]], b2: [[0]] };
  const data = { X: [[0, 0], [0, 0], [0, 0], [0, 0]], y: [1, 1, 1, 0], mean: [[0, 0]], scale: [[1, 1]] };
  const r = proposeStep(weights, data, 10, { name: "b2", row: 0, col: 0 });
  assert.ok(r.predictedChange < 0); assert.ok(r.actualChange > 0);
});

test("loss slice includes exact center and its correct tangent", () => {
  const coordinate = { name: "b2", row: 0, col: 0 }, base = objectiveAndGradients(model, batch);
  const rows = lossSlice(model, batch, coordinate, .5, 5);
  assert.deepEqual(rows.map((r) => r.offset), [-.5, -.25, 0, .25, .5]);
  near(rows[2].loss, base.loss);
  for (const row of rows) near(row.tangent, base.loss + base.gradients.b2[0][0] * row.offset);
});

test("all sandbox operations preserve saved parameters, batch and nested arrays", () => {
  const weights = clone(model), data = clone(batch), before = JSON.stringify({ weights, data });
  for (const row of Object.values(weights).flat()) Object.freeze(row);
  Object.values(weights).forEach(Object.freeze); Object.freeze(weights);
  parameterKeys(weights); objectiveAndGradients(weights, data);
  gradientCheck(weights, data, { name: "b2", row: 0, col: 0 });
  proposeStep(weights, data, .1); lossSlice(weights, data, { name: "b2", row: 0, col: 0 });
  assert.equal(JSON.stringify({ weights, data }), before);
});

test("invalid and oversized requests fail instead of silently changing math", () => {
  for (const bad of [null, {}, { ...model, W1: [[1]] }, { ...model, b2: [[NaN]] }, { ...model, extra: 1 }]) assert.throws(() => objectiveAndGradients(bad, batch), RangeError);
  for (const bad of [{ ...batch, X: [] }, { ...batch, X: Array(65).fill([0, 0]), y: Array(65).fill(0) }, { ...batch, y: [true, 1, 0] }, { ...batch, scale: [[0, 1]] }, { ...batch, l2: -1 }, { ...batch, X: [[Infinity, 0]] }]) assert.throws(() => objectiveAndGradients(model, bad), RangeError);
  for (const coordinate of [null, { name: "W1", row: -1, col: 0 }, { name: "b2", row: 0, col: 1 }]) assert.throws(() => gradientCheck(model, batch, coordinate), RangeError);
  for (const rate of [-1, 11, NaN]) assert.throws(() => proposeStep(model, batch, rate), RangeError);
  assert.throws(() => gradientCheck(model, batch, { name: "b2", row: 0, col: 0 }, 0), RangeError);
  assert.throws(() => lossSlice(model, batch, { name: "b2", row: 0, col: 0 }, 1, 4), RangeError);
  assert.throws(() => objectiveAndGradients(model, { ...batch, X: Array(3) }), RangeError);
  assert.throws(() => objectiveAndGradients(model, { ...batch, y: Array(3) }), RangeError);
});

test("finite arithmetic overflow is reported and inputs are preserved", () => {
  const weights = { W1: [[1e308], [1e308]], b1: [[0]], W2: [[1]], b2: [[0]] };
  const data = { X: [[2, 2]], y: [1], mean: [[0, 0]], scale: [[1, 1]], l2: 0 };
  assert.throws(() => objectiveAndGradients(weights, data), /overflowed/);
  const large = { W1: [[0], [0]], b1: [[0]], W2: [[0]], b2: [[1e100]] };
  assert.throws(() => gradientCheck(large, data, { name: "b2", row: 0, col: 0 }), /too small/);
});
