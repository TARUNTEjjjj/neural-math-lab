(function () {
  "use strict";
  if (typeof document === "undefined") return;
  const root = document.getElementById("gradient-lab");
  if (!root) return;
  const math = window.NeuralOptimization;
  const data = JSON.parse(document.getElementById("experiment-data").textContent);
  const el = (name) => document.getElementById("optimization-" + name);
  const copy = (value) => JSON.parse(JSON.stringify(value));
  const fmt = (value) => value === 0 ? "0" : Number(value.toPrecision(7)).toString();
  const MAX_STEPS = 50;
  let current, history = [], undo = [];
  // Parameters are replaced, never mutated. Keep only the latest computation
  // for each dependency set; typing a rate need not rebuild an unchanged curve.
  let calculations, drawnSlice, drawnHistory;
  const canvas = el("slice"), ctx = canvas.getContext("2d");
  const initial = data.checkpoints[0];
  el("origin").options[0].textContent = "Initial checkpoint · epoch " + initial.epoch;
  el("origin").options[1].textContent = "Selected model · epoch " + data.best_epoch;
  el("l2").value = String(data.config.l2);
  const coordinates = math.parameterKeys(data.model.parameters);
  coordinates.forEach((coordinate, i) => {
    const option = document.createElement("option");
    option.value = String(i); option.textContent = coordinate.label; el("parameter").append(option);
  });
  el("parameter").value = String(coordinates.findIndex((c) => c.name === "b2"));
  function number(name, min, max) {
    const value = el(name).valueAsNumber;
    if (!Number.isFinite(value) || value < min || value > max) {
      const titles = { l2: "L2 coefficient", rate: "Learning rate", radius: "Slice radius", epsilon: "Finite-difference ε" };
      throw new RangeError(titles[name] + " must be a finite number between " + min + " and " + max + ".");
    }
    return value;
  }
  function batch() {
    const ids = data.data.train_idx.slice(0, Number(el("batch").value));
    return { X: ids.map((id) => data.data.X[id]), y: ids.map((id) => data.data.y[id][0]),
      mean: data.data.mean, scale: data.data.scale, l2: number("l2", 0, 10) };
  }
  function row(body, values) {
    const tr = document.createElement("tr");
    values.forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
    body.append(tr);
  }
  function historyTable() {
    const last = history[history.length - 1];
    if (!drawnHistory || drawnHistory.length !== history.length || drawnHistory.last !== last) {
      const body = el("history"); body.replaceChildren();
      history.forEach((item, i) => row(body, [i, fmt(item.bce), fmt(item.penalty), fmt(item.loss), i ? fmt(item.change) : "Starting point"]));
      drawnHistory = { length: history.length, last };
    }
    el("state").textContent = "Temporary sandbox · " + undo.length + " / " + MAX_STEPS + " steps";
    el("undo").disabled = undo.length === 0;
  }
  function drawSlice(points) {
    const area = { left: 65, right: canvas.width - 18, top: 22, bottom: canvas.height - 48 };
    const low = Math.min(...points.flatMap((p) => [p.loss, p.tangent]));
    const high = Math.max(...points.flatMap((p) => [p.loss, p.tangent]));
    const pad = Math.max((high - low) * .12, 1e-6), bottom = low - pad, top = high + pad;
    const radius = points[points.length - 1].offset;
    const x = (offset) => area.left + (offset + radius) / (2 * radius) * (area.right - area.left);
    const y = (loss) => area.bottom - (loss - bottom) / (top - bottom) * (area.bottom - area.top);
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.font = "11px system-ui";
    for (let i = 0; i <= 4; i++) {
      const value = bottom + (top - bottom) * i / 4;
      ctx.strokeStyle = "#dfe8da"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(area.left, y(value)); ctx.lineTo(area.right, y(value)); ctx.stroke();
      ctx.fillStyle = "#456151"; ctx.textAlign = "right"; ctx.fillText(value.toPrecision(3), area.left - 8, y(value) + 4);
      const offset = -radius + 2 * radius * i / 4;
      ctx.textAlign = "center"; ctx.fillText(fmt(offset), x(offset), area.bottom + 20);
    }
    for (const [key, color, dash] of [["tangent", "#b94d24", [5, 4]], ["loss", "#087f78", []]]) {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
      points.forEach((p, i) => i ? ctx.lineTo(x(p.offset), y(p[key])) : ctx.moveTo(x(p.offset), y(p[key]))); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.fillStyle = "#193e3b";
    ctx.beginPath(); ctx.arc(x(0), y(points[Math.floor(points.length / 2)].loss), 4, 0, 2 * Math.PI); ctx.fill();
    ctx.textAlign = "center"; ctx.fillText("Change δ in the selected parameter", canvas.width / 2, canvas.height - 8);
    ctx.save(); ctx.translate(13, canvas.height / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Training objective J", 0, 0); ctx.restore();
  }
  function calculate() {
    const samples = batch(), coordinate = coordinates[Number(el("parameter").value)];
    const rate = number("rate", 1e-6, 10), epsilon = number("epsilon", 1e-8, .1), radius = number("radius", .001, 5);
    if (!calculations || calculations.parameters !== current || calculations.rows !== samples.X.length || calculations.l2 !== samples.l2) {
      calculations = { parameters: current, rows: samples.X.length, l2: samples.l2 };
    }
    const direction = el("direction").value === "full" ? null : coordinate;
    if (!calculations.check || calculations.check.coordinate !== coordinate || calculations.check.epsilon !== epsilon) {
      calculations.check = { coordinate, epsilon, value: math.gradientCheck(current, samples, coordinate, epsilon) };
    }
    if (!calculations.proposal || calculations.proposal.direction !== direction || calculations.proposal.rate !== rate) {
      calculations.proposal = { direction, rate, value: math.proposeStep(current, samples, rate, direction) };
    }
    if (!calculations.slice || calculations.slice.coordinate !== coordinate || calculations.slice.radius !== radius) {
      calculations.slice = { coordinate, radius, value: math.lossSlice(current, samples, coordinate, radius, 41) };
    }
    const check = calculations.check.value, proposed = calculations.proposal.value, slice = calculations.slice.value;
    return { samples, coordinate, rate, epsilon, check, proposed, slice };
  }
  function invalidate(error) {
    el("error").textContent = error.message;
    el("apply").disabled = true;
    ["analytic", "numerical", "gradient-error", "bce", "penalty", "before", "after", "predicted", "actual", "norm"].forEach((id) => { el(id).textContent = "—"; });
    el("verdict").textContent = "Correct the controls to calculate a new proposal.";
    el("slice-caption").textContent = "No slice shown for invalid controls.";
    el("slice-rows").replaceChildren(); ctx.clearRect(0, 0, canvas.width, canvas.height); drawnSlice = null;
  }
  function render() {
    try {
      const { samples, coordinate, check, proposed, slice } = calculate();
      if (!history.length) history.push({ ...proposed.before, change: 0 });
      const fields = { analytic: check.analytic, numerical: check.numerical, "gradient-error": check.absoluteError,
        bce: proposed.before.bce, penalty: proposed.before.penalty, before: proposed.before.loss,
        after: proposed.after.loss, predicted: proposed.predictedChange, actual: proposed.actualChange, norm: proposed.before.gradientNorm };
      Object.entries(fields).forEach(([id, value]) => { el(id).textContent = fmt(value); });
      const value = current[coordinate.name][coordinate.row][coordinate.col];
      el("slice-caption").textContent = coordinate.label + " = " + fmt(value) + ". The dot is the current value (δ = 0). This is a one-coordinate slice, not the path of an all-parameter update.";
      const count = samples.X.length;
      el("batch-detail").textContent = count + " fixed training row" + (count === 1 ? "" : "s") + ", using the saved training standardizer. Changing the starting model, batch or λ resets the sandbox.";
      if (drawnSlice !== slice) {
        const body = el("slice-rows"); body.replaceChildren();
        slice.forEach((point) => row(body, [fmt(point.offset), fmt(point.loss), fmt(point.tangent)]));
        drawSlice(slice); drawnSlice = slice;
      }
      const uphill = proposed.actualChange > 0;
      el("verdict").classList.toggle("uphill", uphill);
      el("verdict").textContent = uphill ? "This proposal increases the objective. The local tangent is not a guarantee for this step size; try a smaller learning rate or apply and undo to explore." :
        proposed.actualChange < 0 ? "This proposal reduces this batch’s objective. That does not establish better held-out performance." : "No representable objective change for this proposal. Inspect the gradient and try another parameter or learning rate.";
      el("error").textContent = "";
      el("apply").disabled = undo.length >= MAX_STEPS;
      if (undo.length >= MAX_STEPS) el("verdict").textContent += " Step limit reached: undo or reset to continue.";
    } catch (error) { invalidate(error); }
    historyTable();
  }
  function reset() {
    current = copy(el("origin").value === "initial" ? initial.parameters : data.model.parameters);
    undo = []; history = []; render();
  }
  ["origin", "batch"].forEach((id) => el(id).addEventListener("change", reset));
  el("l2").addEventListener("input", reset);
  ["parameter", "direction"].forEach((id) => el(id).addEventListener("change", render));
  ["rate", "epsilon", "radius"].forEach((id) => el(id).addEventListener("input", render));
  el("reset").addEventListener("click", reset);
  el("large").addEventListener("click", () => { el("rate").value = "10"; render(); });
  el("apply").addEventListener("click", () => {
    if (undo.length >= MAX_STEPS) return;
    try {
      // Re-read controls at click time; a stale preview can never be applied.
      const { proposed } = calculate();
      undo.push(current); current = proposed.parameters;
      history.push({ ...proposed.after, change: proposed.actualChange }); render();
    } catch (error) { invalidate(error); }
  });
  el("undo").addEventListener("click", () => { if (undo.length) { current = undo.pop(); history.pop(); render(); } });
  reset();
})();
