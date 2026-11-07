(function (root) {
  "use strict";
  // A bounded, disposable SGD experiment. Saved models and batches are never mutated.
  const names = ["W1", "b1", "W2", "b2"];
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  function fail(message) { throw new RangeError(message); }
  function number(value, low, high, label) {
    if (!finite(value) || value < low || value > high) fail(label + " must be finite and between " + low + " and " + high + ".");
    return value;
  }
  function matrix(value, rows, cols, label) {
    if (!Array.isArray(value) || value.length !== rows || !Array.from(value).every(
      (row) => Array.isArray(row) && row.length === cols && Array.from(row).every(finite))) fail(label + " has invalid dimensions or non-finite values.");
    return value;
  }
  function model(parameters) {
    if (!parameters || typeof parameters !== "object" || Object.keys(parameters).length !== 4 ||
        !names.every((name) => Object.hasOwn(parameters, name))) fail("Use a network with W1, b1, W2 and b2.");
    const hidden = parameters.b1?.[0]?.length;
    if (!Number.isInteger(hidden) || hidden < 1 || hidden > 64) fail("Use between 1 and 64 hidden neurons.");
    matrix(parameters.W1, 2, hidden, "W1"); matrix(parameters.b1, 1, hidden, "b1");
    matrix(parameters.W2, hidden, 1, "W2"); matrix(parameters.b2, 1, 1, "b2");
    return hidden;
  }
  function batchData(batch) {
    if (!batch || !Array.isArray(batch.X) || batch.X.length < 1 || batch.X.length > 64) fail("Choose between 1 and 64 training rows.");
    matrix(batch.X, batch.X.length, 2, "Raw inputs");
    matrix(batch.mean, 1, 2, "Training mean"); matrix(batch.scale, 1, 2, "Training scale");
    if (!batch.scale[0].every((v) => v > 0)) fail("Training scales must be positive.");
    if (!Array.isArray(batch.y) || batch.y.length !== batch.X.length) fail("Supply one binary target per training row.");
    const y = Array.from(batch.y, (value) => Array.isArray(value) && value.length === 1 ? value[0] : value);
    if (!y.every((value) => value === 0 || value === 1)) fail("Targets must be binary numbers, 0 or 1.");
    const l2 = number(batch.l2 === undefined ? 0 : batch.l2, 0, 10, "L2 coefficient");
    const X = batch.X.map((row) => row.map((value, i) => (value - batch.mean[0][i]) / batch.scale[0][i]));
    if (!X.every((row) => row.every(finite))) fail("Standardization overflowed; use smaller raw inputs.");
    return { X, y, l2 };
  }
  const copy = (parameters) => Object.fromEntries(names.map((name) => [name, parameters[name].map((row) => row.slice())]));
  function parameterKeys(parameters) {
    model(parameters);
    return names.flatMap((name) => parameters[name].flatMap((row, r) =>
      row.map((_, c) => ({ name, row: r, col: c, label: name + "[" + r + "," + c + "]" }))));
  }
  function coordinateKey(parameters, coordinate) {
    if (!coordinate || !names.includes(coordinate.name) || !Number.isInteger(coordinate.row) || !Number.isInteger(coordinate.col) ||
        coordinate.row < 0 || coordinate.row >= parameters[coordinate.name].length ||
        coordinate.col < 0 || coordinate.col >= parameters[coordinate.name][coordinate.row].length) fail("Choose an existing parameter coordinate.");
    return coordinate;
  }
  function objectiveAndGradients(parameters, batch) {
    const hidden = model(parameters), { X, y, l2 } = batchData(batch), n = X.length;
    const gradients = Object.fromEntries(names.map((name) => [name, parameters[name].map((row) => row.map(() => 0))]));
    let bce = 0, penalty = 0;
    X.forEach((x, i) => {
      const z1 = Array.from({ length: hidden }, (_, j) => x[0] * parameters.W1[0][j] + x[1] * parameters.W1[1][j] + parameters.b1[0][j]);
      if (!z1.every(finite)) fail("Hidden-layer arithmetic overflowed; use smaller weights.");
      const h = z1.map(Math.tanh);
      const z = h.reduce((sum, value, j) => sum + value * parameters.W2[j][0], parameters.b2[0][0]);
      if (!finite(z)) fail("Output arithmetic overflowed; use smaller weights.");
      const e = Math.exp(-Math.abs(z));
      // BCE from logits stays finite when sigmoid rounds to exactly zero or one.
      bce += (Math.max(z, 0) - y[i] * z + Math.log1p(e)) / n;
      // Equivalent to (sigmoid(z)-y)/N, without subtracting a rounded 1.
      const residual = y[i] === 1 ? -(z >= 0 ? e : 1) / (1 + e)
        : (z >= 0 ? 1 : e) / (1 + e);
      const dz = residual / n;
      gradients.b2[0][0] += dz;
      for (let j = 0; j < hidden; j++) {
        gradients.W2[j][0] += h[j] * dz;
        const dz1 = dz * parameters.W2[j][0] * (1 - h[j] * h[j]);
        gradients.b1[0][j] += dz1;
        for (let k = 0; k < 2; k++) gradients.W1[k][j] += x[k] * dz1;
      }
    });
    if (l2) for (const name of ["W1", "W2"]) parameters[name].forEach((row, r) => row.forEach((weight, c) => {
      penalty += l2 / 2 * weight * weight;
      gradients[name][r][c] += l2 * weight;
    }));
    const flat = names.flatMap((name) => gradients[name].flat());
    const loss = bce + penalty, gradientNorm = Math.hypot(...flat);
    if (![bce, penalty, loss, gradientNorm, ...flat].every(finite)) fail("Objective or derivative overflowed; reduce weights or L2.");
    return { loss, bce, penalty, gradients, gradientNorm };
  }
  function gradientCheck(parameters, batch, coordinate, epsilon = 1e-5) {
    model(parameters); const { name, row, col } = coordinateKey(parameters, coordinate);
    number(epsilon, 1e-8, 1, "Finite-difference epsilon");
    const objective = objectiveAndGradients(parameters, batch), analytic = objective.gradients[name][row][col];
    const plus = copy(parameters), minus = copy(parameters), value = parameters[name][row][col];
    plus[name][row][col] = value + epsilon; minus[name][row][col] = value - epsilon;
    if (plus[name][row][col] === value || minus[name][row][col] === value) fail("Epsilon is too small to perturb this weight at floating-point precision.");
    const plusLoss = objectiveAndGradients(plus, batch).loss, minusLoss = objectiveAndGradients(minus, batch).loss;
    const numerical = (plusLoss - minusLoss) / (2 * epsilon), absoluteError = Math.abs(analytic - numerical);
    const relativeError = absoluteError / Math.max(1e-8, Math.abs(analytic) + Math.abs(numerical));
    if (![numerical, absoluteError, relativeError].every(finite)) fail("Finite differences overflowed; reduce weights or change epsilon.");
    return { analytic, numerical, absoluteError, relativeError, plusLoss, minusLoss };
  }
  function proposeStep(parameters, batch, learningRate, coordinate = null) {
    model(parameters); number(learningRate, 0, 10, "Learning rate");
    const keys = coordinate === null ? parameterKeys(parameters) : [coordinateKey(parameters, coordinate)];
    const before = objectiveAndGradients(parameters, batch), updated = copy(parameters), deltas = [];
    let predictedChange = 0;
    keys.forEach(({ name, row, col }) => {
      const g = before.gradients[name][row][col], delta = -learningRate * g;
      updated[name][row][col] += delta; deltas.push(delta); predictedChange += g * delta;
    });
    const after = objectiveAndGradients(updated, batch), actualChange = after.loss - before.loss, stepNorm = Math.hypot(...deltas);
    if (![predictedChange, actualChange, stepNorm].every(finite)) fail("The update overflowed; choose a smaller learning rate.");
    return { parameters: updated, before, after, predictedChange, actualChange, stepNorm };
  }
  function lossSlice(parameters, batch, coordinate, radius = 1, count = 41) {
    model(parameters); const { name, row, col } = coordinateKey(parameters, coordinate);
    number(radius, 1e-4, 10, "Slice radius");
    if (!Number.isInteger(count) || count < 3 || count > 101 || count % 2 !== 1) fail("Use an odd slice count between 3 and 101.");
    const base = objectiveAndGradients(parameters, batch), gradient = base.gradients[name][row][col];
    return Array.from({ length: count }, (_, i) => {
      const offset = (i - (count - 1) / 2) * (2 * radius / (count - 1)), altered = copy(parameters);
      altered[name][row][col] += offset;
      const loss = objectiveAndGradients(altered, batch).loss, tangent = base.loss + gradient * offset;
      if (!finite(tangent)) fail("The tangent approximation overflowed; choose a smaller slice.");
      return { offset, loss, tangent };
    });
  }
  const api = { objectiveAndGradients, parameterKeys, gradientCheck, proposeStep, lossSlice };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NeuralOptimization = api;
})(typeof window !== "undefined" ? window : globalThis);
