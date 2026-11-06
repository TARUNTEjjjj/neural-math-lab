"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const math = require("../neural_math_lab/assets/optimization.js");

const assets = path.join(__dirname, "..", "neural_math_lab", "assets");
const html = fs.readFileSync(path.join(assets, "optimization.html"), "utf8");
const script = fs.readFileSync(path.join(assets, "optimization-ui.js"), "utf8");

// A narrow DOM adapter runs the actual event handlers without a browser package.
// Visual layout, focus and native input behavior still need browser checks.
class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.listeners = new Map();
    this.value = "";
    this.disabled = false;
    this.width = 720;
    this.height = 360;
    this._text = "";
    this.classList = { toggle() {} };
  }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get valueAsNumber() { return this.value.trim() === "" ? NaN : Number(this.value); }
  get options() { return this.children; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this._text = ""; this.children = items; }
  addEventListener(name, listener) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(listener);
  }
  dispatch(name) {
    if (name === "click" && this.disabled) return;
    for (const listener of this.listeners.get(name) || []) listener({ preventDefault() {} });
  }
  getContext() { return new Proxy({}, { get: () => () => {}, set: () => true }); }
}

function start() {
  const elements = new Map();
  for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element(match[1]);
    const value = match[2].match(/\bvalue="([^"]*)"/);
    if (value) element.value = value[1];
    element.disabled = /\bdisabled\b/.test(match[2]);
    elements.set(match[3], element);
  }
  for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const select = elements.get(match[1]);
    for (const optionMatch of match[2].matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)) {
      const option = new Element("option");
      option.value = optionMatch[1].match(/value="([^"]*)"/)[1];
      option.textContent = optionMatch[2];
      if (!select.children.length || /(?:^|\s)selected(?:\s|$)/.test(optionMatch[1])) select.value = option.value;
      select.append(option);
    }
  }
  const initial = { W1: [[1], [-1]], b1: [[0]], W2: [[2]], b2: [[0]] };
  const selected = { W1: [[.4], [-.3]], b1: [[.2]], W2: [[.7]], b2: [[-.1]] };
  const data = {
    checkpoints: [{ epoch: 0, parameters: initial }], best_epoch: 20,
    model: { parameters: selected }, config: { l2: .1 },
    data: { X: [[0, 0], [.3, -.2], [-.6, .4], [99, 99], [-99, -99]],
      y: [[1], [0], [1], [0], [1]], mean: [[0, 0]], scale: [[1, 1]],
      train_idx: [0, 1, 2], val_idx: [3], test_idx: [4] },
  };
  const recorded = JSON.stringify(data);
  const experiment = new Element("script"); experiment.textContent = recorded;
  elements.set("experiment-data", experiment);
  let parsedExperiment;
  const batchCalls = [];
  const mathCalls = { gradientCheck: 0, proposeStep: 0, lossSlice: 0 };
  const observedMath = { ...math };
  for (const name of ["gradientCheck", "proposeStep", "lossSlice"]) {
    observedMath[name] = (...args) => {
      mathCalls[name]++;
      batchCalls.push(JSON.parse(JSON.stringify(args[1])));
      return math[name](...args);
    };
  }
  const context = {
    document: { getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag) },
    window: { NeuralOptimization: observedMath },
    JSON: { stringify: JSON.stringify, parse(value) {
      const parsed = JSON.parse(value);
      if (value === recorded) parsedExperiment = parsed;
      return parsed;
    } },
  };
  vm.runInNewContext(script, context, { filename: "optimization-ui.js" });
  const get = (id) => elements.get("optimization-" + id);
  const set = (id, value, event = "input") => { get(id).value = String(value); get(id).dispatch(event); };
  const rows = () => get("history").children.map((row) => row.children.map((cell) => cell.textContent));
  return { get, set, rows, batchCalls, mathCalls, initial, selected, data,
    checkRecorded: () => assert.equal(JSON.stringify(parsedExperiment), recorded) };
}

test("sandbox applies a proposal, undoes it exactly, and leaves recorded data untouched", () => {
  const ui = start();
  assert.equal(ui.rows().length, 1);
  assert.equal(ui.get("undo").disabled, true);
  const starting = ui.get("before").textContent;
  const proposed = ui.get("after").textContent;
  ui.get("apply").dispatch("click");
  assert.equal(ui.get("before").textContent, proposed);
  assert.equal(ui.rows().length, 2);
  assert.equal(ui.get("undo").disabled, false);
  ui.get("undo").dispatch("click");
  assert.equal(ui.get("before").textContent, starting);
  assert.equal(ui.rows().length, 1);
  assert.equal(ui.get("undo").disabled, true);
  ui.get("apply").dispatch("click");
  ui.get("reset").dispatch("click");
  assert.equal(ui.get("before").textContent, starting);
  assert.equal(ui.rows().length, 1);
  ui.checkRecorded();
});

test("batch, L2 and starting model changes reset history and use training rows only", () => {
  const ui = start();
  for (const [id, value, event] of [["batch", 1, "change"], ["l2", .2, "input"], ["origin", "selected", "change"]]) {
    ui.get("apply").dispatch("click");
    assert.equal(ui.rows().length, 2);
    ui.set(id, value, event);
    assert.equal(ui.rows().length, 1, id + " must reset history");
    assert.equal(ui.get("undo").disabled, true);
  }
  const expected = math.objectiveAndGradients(ui.selected, {
    X: [[0, 0]], y: [1], mean: [[0, 0]], scale: [[1, 1]], l2: .2,
  });
  assert.ok(Math.abs(Number(ui.get("before").textContent) - expected.loss) < 1e-6);
  for (const batch of ui.batchCalls) {
    assert.ok(batch.X.length >= 1 && batch.X.length <= 3);
    assert.ok(batch.X.every((row) => Math.abs(row[0]) < 1 && Math.abs(row[1]) < 1));
  }
  ui.checkRecorded();
});

test("invalid controls disable applying without changing sandbox history", () => {
  const ui = start();
  ui.get("apply").dispatch("click");
  const before = ui.rows();
  ui.set("rate", "");
  assert.equal(ui.get("apply").disabled, true);
  assert.match(ui.get("error").textContent, /Learning rate/);
  assert.equal(ui.get("after").textContent, "—");
  ui.get("apply").dispatch("click");
  assert.deepEqual(ui.rows(), before);
  ui.set("rate", .1);
  assert.equal(ui.get("apply").disabled, false);
  assert.equal(ui.get("error").textContent, "");
  assert.deepEqual(ui.rows(), before);
  ui.set("epsilon", 0);
  assert.equal(ui.get("apply").disabled, true);
  assert.equal(ui.get("slice-rows").children.length, 0);
  ui.checkRecorded();
});

test("apply recalculates controls at click time rather than using an old preview", () => {
  const ui = start();
  // Simulate a delayed input event: change the numeric value without notifying
  // render, then click. The applied update must use the current control value.
  ui.get("rate").value = ".5";
  const expected = math.proposeStep(ui.initial, {
    X: ui.data.data.X.slice(0, 3), y: [1, 0, 1],
    mean: [[0, 0]], scale: [[1, 1]], l2: .1,
  }, .5, { name: "b2", row: 0, col: 0 });
  ui.get("apply").dispatch("click");
  assert.ok(Math.abs(Number(ui.get("before").textContent) - expected.after.loss) < 1e-6);
  assert.equal(ui.rows().length, 2);
  ui.checkRecorded();
});

test("sandbox enforces its 50-step limit and undo restores one available step", () => {
  const ui = start();
  ui.set("rate", .01);
  for (let i = 0; i < 50; i++) ui.get("apply").dispatch("click");
  assert.equal(ui.rows().length, 51);
  assert.equal(ui.get("apply").disabled, true);
  assert.match(ui.get("state").textContent, /50 \/ 50/);
  ui.get("apply").dispatch("click");
  assert.equal(ui.rows().length, 51);
  ui.get("undo").dispatch("click");
  assert.equal(ui.rows().length, 50);
  assert.equal(ui.get("apply").disabled, false);
  ui.get("apply").dispatch("click");
  assert.equal(ui.rows().length, 51);
  assert.equal(ui.get("apply").disabled, true);
  ui.get("reset").dispatch("click");
  assert.equal(ui.rows().length, 1);
  assert.equal(ui.get("apply").disabled, false);
  ui.checkRecorded();
});

test("rate and update-direction edits preserve the unchanged curve, derivative and table nodes", () => {
  const ui = start();
  const sliceRow = ui.get("slice-rows").children[0], historyRow = ui.get("history").children[0];
  const check = ui.get("analytic").textContent;
  for (const rate of [.2, .3, .4]) ui.set("rate", rate);
  ui.set("direction", "full", "change");
  assert.deepEqual(ui.mathCalls, { gradientCheck: 1, proposeStep: 5, lossSlice: 1 });
  assert.equal(ui.get("slice-rows").children[0], sliceRow);
  assert.equal(ui.get("history").children[0], historyRow);
  assert.equal(ui.get("analytic").textContent, check);
  const expected = math.proposeStep(ui.initial, {
    X: ui.data.data.X.slice(0, 3), y: [1, 0, 1], mean: [[0, 0]], scale: [[1, 1]], l2: .1,
  }, .4);
  assert.ok(Math.abs(Number(ui.get("after").textContent) - expected.after.loss) < 1e-6);
  ui.checkRecorded();
});

test("epsilon and radius invalidate only their dependent calculations, and invalid inputs restore the curve", () => {
  const ui = start();
  const firstRow = ui.get("slice-rows").children[0];
  ui.set("epsilon", .0001);
  assert.deepEqual(ui.mathCalls, { gradientCheck: 2, proposeStep: 1, lossSlice: 1 });
  assert.equal(ui.get("slice-rows").children[0], firstRow);
  ui.set("radius", .5);
  assert.deepEqual(ui.mathCalls, { gradientCheck: 2, proposeStep: 1, lossSlice: 2 });
  assert.notEqual(ui.get("slice-rows").children[0], firstRow);
  assert.equal(ui.get("slice-rows").children[0].children[0].textContent, "-0.5");
  ui.set("radius", "");
  assert.equal(ui.get("slice-rows").children.length, 0);
  assert.equal(ui.get("apply").disabled, true);
  ui.set("radius", .5);
  assert.equal(ui.get("slice-rows").children.length, 41);
  assert.equal(ui.get("apply").disabled, false);
  assert.deepEqual(ui.mathCalls, { gradientCheck: 2, proposeStep: 1, lossSlice: 2 });
});

test("parameter changes and new sandbox models invalidate cached results without redoing the old proposal", () => {
  const ui = start();
  ui.set("parameter", 0, "change");
  assert.deepEqual(ui.mathCalls, { gradientCheck: 2, proposeStep: 2, lossSlice: 2 });
  const before = ui.get("before").textContent, after = ui.get("after").textContent;
  ui.get("apply").dispatch("click");
  assert.equal(ui.get("before").textContent, after);
  assert.deepEqual(ui.mathCalls, { gradientCheck: 3, proposeStep: 3, lossSlice: 3 });
  ui.get("undo").dispatch("click");
  assert.equal(ui.get("before").textContent, before);
  assert.deepEqual(ui.mathCalls, { gradientCheck: 4, proposeStep: 4, lossSlice: 4 });
  // Only the current model is cached, so undo recomputes its original values.
  for (const [id, value, event] of [["batch", 1, "change"], ["l2", .2, "input"], ["origin", "selected", "change"]]) {
    const oldCalls = { ...ui.mathCalls };
    ui.set(id, value, event);
    for (const key of Object.keys(oldCalls)) assert.equal(ui.mathCalls[key], oldCalls[key] + 1);
  }
  ui.checkRecorded();
});
