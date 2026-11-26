"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../site/site3d-math.js");

test("SO(3) lookAt camera basis preserves orthonormality and det = +1 across orbit sweep", () => {
  for (let i = 0; i < 24; i++) {
    const yaw = -Math.PI + (2 * Math.PI * i) / 24;
    const pitch = -1.2 + (2.4 * i) / 23;
    const eye = M.orbitCameraEye(yaw, pitch, 4.35);
    assert.ok(Math.abs(Math.hypot(...eye) - 4.35) < 1e-12);
    const V = M.lookAtMatrix4(eye, [0, 0, 0], [0, 1, 0]);
    const r0 = [V[0], V[4], V[8]];
    const r1 = [V[1], V[5], V[9]];
    const r2 = [V[2], V[6], V[10]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert.ok(Math.abs(dot(r0, r0) - 1) < 1e-6);
    assert.ok(Math.abs(dot(r1, r1) - 1) < 1e-6);
    assert.ok(Math.abs(dot(r2, r2) - 1) < 1e-6);
    assert.ok(Math.abs(dot(r0, r1)) < 1e-6);
    assert.ok(Math.abs(dot(r0, r2)) < 1e-6);
    assert.ok(Math.abs(dot(r1, r2)) < 1e-6);
    const det =
      r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) -
      r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) +
      r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
    assert.ok(Math.abs(det - 1) < 1e-6, `det=${det}`);
  }
});

test("differential verification: analytical gradient & Hessian of J(w1, w2) match central finite differences", () => {
  const h = 1e-6;
  for (let i = -3; i <= 3; i++) {
    for (let j = -3; j <= 3; j++) {
      const w1 = i * 0.45, w2 = j * 0.42;
      const res = M.evaluateLossSurface(w1, w2, 0.22, 1.05);
      const d1Num = (M.evaluateLossSurface(w1 + h, w2, 0.22, 1.05).loss - M.evaluateLossSurface(w1 - h, w2, 0.22, 1.05).loss) / (2 * h);
      const d2Num = (M.evaluateLossSurface(w1, w2 + h, 0.22, 1.05).loss - M.evaluateLossSurface(w1, w2 - h, 0.22, 1.05).loss) / (2 * h);
      assert.ok(Math.abs(res.gradient[0] - d1Num) < 1e-8, `d1 mismatch at (${w1}, ${w2})`);
      assert.ok(Math.abs(res.gradient[1] - d2Num) < 1e-8, `d2 mismatch at (${w1}, ${w2})`);

      // Hessian Schwarz symmetry & second derivative parity
      assert.ok(Math.abs(res.hessian[0][1] - res.hessian[1][0]) < 1e-12);
      const h11Num = (M.evaluateLossSurface(w1 + h, w2, 0.22, 1.05).gradient[0] - M.evaluateLossSurface(w1 - h, w2, 0.22, 1.05).gradient[0]) / (2 * h);
      const h22Num = (M.evaluateLossSurface(w1, w2 + h, 0.22, 1.05).gradient[1] - M.evaluateLossSurface(w1, w2 - h, 0.22, 1.05).gradient[1]) / (2 * h);
      const h12Num = (M.evaluateLossSurface(w1, w2 + h, 0.22, 1.05).gradient[0] - M.evaluateLossSurface(w1, w2 - h, 0.22, 1.05).gradient[0]) / (2 * h);
      assert.ok(Math.abs(res.hessian[0][0] - h11Num) < 1e-7);
      assert.ok(Math.abs(res.hessian[1][1] - h22Num) < 1e-7);
      assert.ok(Math.abs(res.hessian[0][1] - h12Num) < 1e-7);
    }
  }
});

test("differential verification: 3D MLP manifold gradient matches central differences across presets", () => {
  const h = 1e-6;
  for (const key of Object.keys(M.PRESETS)) {
    const preset = M.PRESETS[key];
    for (const [x1, x2] of [[-1.1, 0.7], [0.0, 0.0], [0.85, -0.45], [-0.4, -0.9]]) {
      const out = M.evaluateMLP(preset, x1, x2);
      assert.ok(out.p > 0 && out.p < 1);
      const gx1 = (M.evaluateMLP(preset, x1 + h, x2).p - M.evaluateMLP(preset, x1 - h, x2).p) / (2 * h);
      const gx2 = (M.evaluateMLP(preset, x1, x2 + h).p - M.evaluateMLP(preset, x1, x2 - h).p) / (2 * h);
      assert.ok(Math.abs(out.gradient[0] - gx1) < 1e-9, `${key} gx1`);
      assert.ok(Math.abs(out.gradient[1] - gx2) < 1e-9, `${key} gx2`);
    }
  }
});

test("gradient descent path achieves strict descent under small step sizes and all 3D scenes emit finite geometry", () => {
  const path = M.traceDescentPath([1.25, 1.15], 0.08, 25, 0.22, 1.05);
  assert.ok(path.at(-1).loss < path[0].loss);
  for (let i = 1; i < path.length; i++) {
    assert.ok(path[i].loss <= path[i - 1].loss + 1e-9);
  }
  for (const mode of M.MODES) {
    for (const preset of Object.keys(M.PRESETS)) {
      const scene = M.buildSceneData({ mode, preset, rate: 0.16, phase: 0.5 });
      assert.ok(scene.buffers.ptPos.length > 0 || scene.buffers.pos.length > 0);
      assert.ok(scene.summary.formula.length > 5);
      for (const v of scene.buffers.pos) assert.ok(Number.isFinite(v));
      for (const n of scene.buffers.norm) assert.ok(Number.isFinite(n));
    }
  }
});
