(function () {
  "use strict";
  // This is inference only. Python performs all fitting and model selection.
  function infer(parameters, raw, mean, scale, linear = false) {
    const x = raw.map((v, i) => (v - mean[0][i]) / scale[0][i]);
    let h = [], z;
    if (linear) {
      z = x[0] * parameters.W[0][0] + x[1] * parameters.W[1][0] + parameters.b[0][0];
    } else {
      h = parameters.b1[0].map((b, j) => Math.tanh(x[0] * parameters.W1[0][j] + x[1] * parameters.W1[1][j] + b));
      z = parameters.b2[0][0] + h.reduce((sum, a, j) => sum + a * parameters.W2[j][0], 0);
    }
    const e = Math.exp(z < 0 ? z : -z);
    const p = z >= 0 ? 1 / (1 + e) : e / (1 + e);
    return { x, h, z, p };
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { infer };
    return;
  }
  const data = JSON.parse(document.getElementById("experiment-data").textContent);
  const byId = (id) => document.getElementById(id);
  const frames = data.checkpoints;
  const slider = byId("checkpoint");
  const bestIndex = frames.findIndex((f) => f.epoch === data.best_epoch);
  slider.max = String(frames.length - 1);
  slider.value = String(bestIndex);
  const bounds = data.bounds;
  let probe = data.data.X[data.data.test_idx[0]].slice();
  let timer = null;
  const canvas = byId("boundary"), ctx = canvas.getContext("2d");
  const chart = byId("learning-curve"), chartCtx = chart.getContext("2d");
  const box = { left: 48, top: 16, right: canvas.width - 18, bottom: canvas.height - 39 };
  const colors = { zero: "#aa5839", one: "#087c6b", ink: "#264a3e", grid: "#dae4d4" };
  const px = (x) => box.left + (x - bounds[0]) / (bounds[1] - bounds[0]) * (box.right - box.left);
  const py = (y) => box.bottom - (y - bounds[2]) / (bounds[3] - bounds[2]) * (box.bottom - box.top);
  const selectedFrame = () => frames[Number(slider.value)];
  const isLinear = () => byId("model-view").value === "baseline";
  const parameters = () => isLinear() ? data.baseline.parameters : selectedFrame().parameters;
  const predict = (point) => infer(parameters(), point, data.data.mean, data.data.scale, isLinear());
  const fixed = (n) => n.toFixed(4);

  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    byId("play").textContent = "Play training";
  }
  function color(p) {
    const a = p < 0.5 ? [235, 207, 182] : [245, 246, 233];
    const b = p < 0.5 ? [245, 246, 233] : [125, 188, 164];
    const t = p < 0.5 ? p * 2 : (p - 0.5) * 2;
    return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",") + ")";
  }
  function drawBoundary() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const nx = 85, ny = 53, dx = (box.right - box.left) / nx, dy = (box.bottom - box.top) / ny;
    const points = [];
    for (let j = 0; j <= ny; j++) {
      const row = [];
      for (let i = 0; i <= nx; i++) {
        row.push(predict([bounds[0] + (bounds[1] - bounds[0]) * i / nx, bounds[3] - (bounds[3] - bounds[2]) * j / ny]).p);
      }
      points.push(row);
    }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      ctx.fillStyle = color((points[j][i] + points[j + 1][i] + points[j][i + 1] + points[j + 1][i + 1]) / 4);
      ctx.fillRect(box.left + i * dx, box.top + j * dy, dx + 0.6, dy + 0.6);
      const corners = [[i, j, points[j][i]], [i + 1, j, points[j][i + 1]], [i + 1, j + 1, points[j + 1][i + 1]], [i, j + 1, points[j + 1][i]]];
      const crossings = [];
      for (let k = 0; k < 4; k++) {
        const a = corners[k], b = corners[(k + 1) % 4];
        if ((a[2] < 0.5) !== (b[2] < 0.5)) {
          const t = (0.5 - a[2]) / (b[2] - a[2]);
          crossings.push([box.left + (a[0] + t * (b[0] - a[0])) * dx, box.top + (a[1] + t * (b[1] - a[1])) * dy]);
        }
      }
      ctx.strokeStyle = "#315c4e"; ctx.lineWidth = 1.2;
      for (let k = 0; k + 1 < crossings.length; k += 2) {
        ctx.beginPath(); ctx.moveTo(...crossings[k]); ctx.lineTo(...crossings[k + 1]); ctx.stroke();
      }
    }
    const split = byId("split-view").value;
    const ids = split === "none" ? [] : split === "all" ? data.data.X.map((_, i) => i) : data.data[split === "validation" ? "val_idx" : split + "_idx"];
    for (const id of ids) {
      const point = data.data.X[id], actual = data.data.y[id][0], x = px(point[0]), y = py(point[1]);
      const p = predict(point).p;
      ctx.fillStyle = actual === 0 ? colors.zero : colors.one; ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
      ctx.beginPath();
      if (actual === 0) ctx.rect(x - 3, y - 3, 6, 6); else ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      if (byId("show-errors").checked && Number(p >= 0.5) !== actual) {
        ctx.strokeStyle = "#9a1525"; ctx.lineWidth = 1.6; ctx.beginPath();
        ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x - 5, y + 5); ctx.lineTo(x + 5, y - 5); ctx.stroke();
      }
    }
    ctx.strokeStyle = "#193e3b"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px(probe[0]), py(probe[1]), 8, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px(probe[0]) - 12, py(probe[1])); ctx.lineTo(px(probe[0]) + 12, py(probe[1])); ctx.moveTo(px(probe[0]), py(probe[1]) - 12); ctx.lineTo(px(probe[0]), py(probe[1]) + 12); ctx.stroke();
    ctx.strokeStyle = "#a5b9ac"; ctx.lineWidth = 1; ctx.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    ctx.fillStyle = colors.ink; ctx.font = "11px system-ui";
    for (let i = 0; i <= 4; i++) {
      ctx.textAlign = "center"; ctx.fillText((bounds[0] + (bounds[1] - bounds[0]) * i / 4).toFixed(1), box.left + (box.right - box.left) * i / 4, box.bottom + 17);
      ctx.textAlign = "right"; ctx.fillText((bounds[2] + (bounds[3] - bounds[2]) * i / 4).toFixed(1), box.left - 9, box.bottom - (box.bottom - box.top) * i / 4 + 4);
    }
    ctx.textAlign = "center"; ctx.fillText("Raw input x₁", (box.left + box.right) / 2, canvas.height - 6);
    ctx.save(); ctx.translate(12, (box.top + box.bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Raw input x₂", 0, 0); ctx.restore();
  }
  function drawCurve() {
    const c = chartCtx, w = chart.width, h = chart.height;
    const area = { left: 49, right: w - 16, top: 22, bottom: h - 38 };
    const maxLoss = Math.max(...data.history.map((r) => Math.max(r.train_loss, r.val_loss))) * 1.08 || 1;
    const sx = (e) => area.left + e / Math.max(1, data.stopped_epoch) * (area.right - area.left);
    const sy = (v) => area.bottom - v / maxLoss * (area.bottom - area.top);
    c.clearRect(0, 0, w, h); c.font = "11px system-ui";
    for (let i = 0; i <= 4; i++) {
      const v = maxLoss * i / 4; c.strokeStyle = "#e3e9df"; c.lineWidth = 1; c.beginPath(); c.moveTo(area.left, sy(v)); c.lineTo(area.right, sy(v)); c.stroke();
      c.fillStyle = colors.ink; c.textAlign = "right"; c.fillText(v.toFixed(2), area.left - 8, sy(v) + 4);
      c.textAlign = "center"; c.fillText(String(Math.round(data.stopped_epoch * i / 4)), sx(data.stopped_epoch * i / 4), area.bottom + 19);
    }
    for (const [key, stroke] of [["train_loss", "#087f78"], ["val_loss", "#b94d24"]]) {
      c.strokeStyle = stroke; c.lineWidth = 2; c.beginPath();
      data.history.forEach((r, i) => i ? c.lineTo(sx(r.epoch), sy(r[key])) : c.moveTo(sx(r.epoch), sy(r[key]))); c.stroke();
    }
    c.strokeStyle = "#7e9283"; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(sx(data.best_epoch), area.top); c.lineTo(sx(data.best_epoch), area.bottom); c.stroke(); c.setLineDash([]);
    if (!isLinear()) { c.fillStyle = "#183e3b"; c.beginPath(); c.arc(sx(selectedFrame().epoch), sy(data.history[selectedFrame().epoch].val_loss), 4, 0, Math.PI * 2); c.fill(); }
    c.fillStyle = colors.ink; c.textAlign = "center"; c.fillText("Epoch · one pass through the training split", w / 2, h - 5);
    c.save(); c.translate(12, h / 2); c.rotate(-Math.PI / 2); c.fillText("Mean binary cross-entropy", 0, 0); c.restore();
  }
  function updateInspector() {
    const r = predict(probe);
    byId("probe-p").textContent = r.p.toFixed(6);
    byId("probe-class").textContent = "Predicted class " + Number(r.p >= 0.5) + " · threshold 0.5";
    byId("trace-standard").textContent = "[" + r.x.map(fixed).join(", ") + "]";
    byId("trace-logit").textContent = fixed(r.z);
    byId("trace-sigmoid").textContent = "σ(" + fixed(r.z) + ") = " + r.p.toFixed(6);
    const bars = byId("neuron-bars"); bars.replaceChildren();
    if (isLinear()) { bars.textContent = "No hidden layer in the baseline."; }
    r.h.forEach((v, j) => {
      const row = document.createElement("div"); row.className = "neuron";
      const label = document.createElement("span"); label.textContent = "h" + (j + 1);
      const track = document.createElement("span"); track.className = "neuron-track";
      const fill = document.createElement("span"); fill.className = "neuron-fill" + (v < 0 ? " negative" : "");
      fill.style.width = Math.abs(v) * 50 + "%"; fill.style.left = (v < 0 ? 50 + v * 50 : 50) + "%"; track.append(fill);
      const value = document.createElement("output"); value.textContent = v.toFixed(3); row.append(label, track, value); bars.append(row);
    });
  }
  function update() {
    const frame = selectedFrame();
    byId("frame-label").textContent = isLinear() ? "Linear baseline · epoch " + data.baseline.best_epoch : "Epoch " + frame.epoch + (frame.epoch === data.best_epoch ? " · selected" : " · replay");
    byId("frame-detail").textContent = isLinear() ? "The baseline’s validation-selected model. Its parameters remain fixed during comparison." : "Checkpoint " + (Number(slider.value) + 1) + " of " + frames.length + ". The dark green contour approximates p = 0.5. Metric cards show the selected model at epoch " + data.best_epoch + ", not this replay frame.";
    slider.setAttribute("aria-valuetext", "Epoch " + frame.epoch);
    drawBoundary(); drawCurve(); updateInspector();
  }
  function setProbe(point) {
    probe = point;
    byId("input-x").value = String(probe[0]);
    byId("input-y").value = String(probe[1]);
    byId("probe-error").textContent = "";
    update();
  }
  byId("probe-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const point = [byId("input-x").valueAsNumber, byId("input-y").valueAsNumber];
    if (!point.every(Number.isFinite) || point[0] < bounds[0] || point[0] > bounds[1] || point[1] < bounds[2] || point[1] > bounds[3]) {
      byId("probe-error").textContent = "Choose finite coordinates within the displayed map."; return;
    }
    setProbe(point);
  });
  canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect(), x = (event.clientX - rect.left) / rect.width * canvas.width, y = (event.clientY - rect.top) / rect.height * canvas.height;
    if (x < box.left || x > box.right || y < box.top || y > box.bottom) return;
    setProbe([bounds[0] + (x - box.left) / (box.right - box.left) * (bounds[1] - bounds[0]), bounds[3] - (y - box.top) / (box.bottom - box.top) * (bounds[3] - bounds[2])]);
  });
  slider.addEventListener("input", () => { stop(); update(); });
  byId("model-view").addEventListener("change", () => { stop(); slider.disabled = isLinear(); byId("play").disabled = isLinear(); update(); });
  byId("split-view").addEventListener("change", update);
  byId("show-errors").addEventListener("change", update);
  byId("best").addEventListener("click", () => { stop(); byId("model-view").value = "network"; slider.disabled = false; byId("play").disabled = false; slider.value = String(bestIndex); update(); });
  byId("play").addEventListener("click", () => {
    if (timer !== null) { stop(); return; }
    if (Number(slider.value) >= frames.length - 1 || Number(slider.value) === bestIndex) slider.value = "0";
    byId("play").textContent = "Pause"; update();
    timer = setInterval(() => { if (Number(slider.value) >= frames.length - 1) { stop(); return; } slider.value = String(Number(slider.value) + 1); update(); }, 220);
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  window.addEventListener("pagehide", stop);
  setProbe(probe);
})();
