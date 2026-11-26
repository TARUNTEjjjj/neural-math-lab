(function (root) {
  "use strict";

  const TAU = Math.PI * 2;
  const MODES = ["landscape", "manifold", "network"];
  const PRESETS = {
    moons: {
      id: "moons",
      label: "Moons wave",
      description: "Non-linear crescent separation requiring curved hidden-unit boundaries.",
      // Compact 2 -> 4 -> 1 parameters producing an S-shaped classification ridge
      W1: [[2.4, -2.1, 1.7, -1.5], [1.9, 2.3, -1.6, -2.2]],
      b1: [0.35, -0.45, 0.25, 0.55],
      W2: [2.6, -2.4, 2.1, -1.9],
      b2: -0.15,
      biasW: [0.65, -0.45],
      points: [
        [-1.1, 0.4, 0], [-0.7, 0.9, 0], [-0.1, 1.0, 0], [0.5, 0.65, 0], [0.85, 0.15, 0],
        [-0.45, -0.15, 1], [0.05, -0.75, 1], [0.65, -0.95, 1], [1.15, -0.55, 1], [1.35, 0.05, 1]
      ]
    },
    circles: {
      id: "circles",
      label: "Circles dome",
      description: "Concentric radial classes forming a smooth bell-shaped probability peak.",
      W1: [[2.2, -2.2, 0.1, -0.1], [0.1, -0.1, 2.2, -2.2]],
      b1: [-1.1, -1.1, -1.1, -1.1],
      W2: [-2.4, -2.4, -2.4, -2.4],
      b2: 3.1,
      biasW: [-0.2, 0.75],
      points: [
        [0.0, 0.0, 1], [0.35, 0.25, 1], [-0.3, -0.35, 1], [0.4, -0.2, 1], [-0.35, 0.3, 1],
        [-1.2, 0.1, 0], [1.2, -0.1, 0], [0.05, 1.25, 0], [-0.1, -1.25, 0], [0.9, 0.9, 0]
      ]
    },
    xor: {
      id: "xor",
      label: "XOR saddle",
      description: "Hyperbolic saddle parity where quadrants swap output polarity across axes.",
      W1: [[2.6, -2.6, 2.3, -2.3], [2.6, -2.6, -2.3, 2.3]],
      b1: [-1.4, -1.4, 1.2, 1.2],
      W2: [2.7, 2.7, -2.4, -2.4],
      b2: -0.2,
      biasW: [0.55, 0.55],
      points: [
        [-0.95, -0.95, 0], [-0.85, 0.95, 1], [0.95, -0.95, 1], [0.95, 0.95, 0],
        [-0.6, -0.7, 0], [-0.7, 0.65, 1], [0.7, -0.65, 1], [0.7, 0.7, 0]
      ]
    }
  };

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
  }

  function sigmoid(z) {
    if (z >= 0) {
      const e = Math.exp(-z);
      return 1 / (1 + e);
    }
    const e = Math.exp(z);
    return e / (1 + e);
  }

  function vecSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function vecCross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }
  function vecNormalize(v) {
    const len = Math.hypot(v[0], v[1], v[2]);
    return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 1];
  }

  function lookAtMatrix4(eye, target, up) {
    const zAxis = vecNormalize(vecSub(eye, target));
    const xAxis = vecNormalize(vecCross(up || [0, 1, 0], zAxis));
    const yAxis = vecCross(zAxis, xAxis);
    return new Float32Array([
      xAxis[0], yAxis[0], zAxis[0], 0,
      xAxis[1], yAxis[1], zAxis[1], 0,
      xAxis[2], yAxis[2], zAxis[2], 0,
      -(xAxis[0] * eye[0] + xAxis[1] * eye[1] + xAxis[2] * eye[2]),
      -(yAxis[0] * eye[0] + yAxis[1] * eye[1] + yAxis[2] * eye[2]),
      -(zAxis[0] * eye[0] + zAxis[1] * eye[1] + zAxis[2] * eye[2]),
      1
    ]);
  }

  function perspectiveMatrix4(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY * 0.5);
    const rangeInv = 1 / (near - far);
    return new Float32Array([
      f / Math.max(0.2, aspect), 0, 0, 0,
      0, f, 0, 0,
      0, 0, (near + far) * rangeInv, -1,
      0, 0, near * far * rangeInv * 2, 0
    ]);
  }

  function mat4Mul(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        out[col * 4 + row] =
          a[0 * 4 + row] * b[col * 4 + 0] +
          a[1 * 4 + row] * b[col * 4 + 1] +
          a[2 * 4 + row] * b[col * 4 + 2] +
          a[3 * 4 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  function orbitCameraEye(yaw, pitch, distance) {
    const cp = Math.cos(pitch);
    return [
      distance * Math.sin(yaw) * cp,
      distance * Math.sin(pitch),
      distance * Math.cos(yaw) * cp
    ];
  }

  function projectPoint3D(point, mvp, width, height) {
    const x = point[0], y = point[1], z = point[2];
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cz = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0.05) return { visible: false, x: 0, y: 0, depth: cw };
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    return {
      visible: Math.abs(ndcX) <= 1.8 && Math.abs(ndcY) <= 1.8,
      x: (ndcX * 0.5 + 0.5) * width,
      y: (0.5 - ndcY * 0.5) * height,
      depth: cz / cw
    };
  }

  /** Evaluate 2 -> 4 -> 1 forward probability and exact d(p)/dx using overflow-safe derivatives. */
  function evaluateMLP(preset, x1, x2) {
    const h = new Array(4);
    const sech2 = new Array(4);
    let logit = preset.b2;
    for (let j = 0; j < 4; j++) {
      const z1 = x1 * preset.W1[0][j] + x2 * preset.W1[1][j] + preset.b1[j];
      const act = Math.tanh(z1);
      h[j] = act;
      const e = Math.exp(-Math.abs(z1));
      const d = (2 * e) / (1 + e * e);
      sech2[j] = d * d;
      logit += act * preset.W2[j];
    }
    const p = sigmoid(logit);
    const e2 = Math.exp(-Math.abs(logit));
    const dpDz2 = (e2 / ((1 + e2) * (1 + e2)));
    let gx1 = 0;
    let gx2 = 0;
    for (let j = 0; j < 4; j++) {
      const chain = dpDz2 * preset.W2[j] * sech2[j];
      gx1 += chain * preset.W1[0][j];
      gx2 += chain * preset.W1[1][j];
    }
    return { h, logit, p, gradient: [gx1, gx2] };
  }

  /**
   * Analytic loss surface J(w1, w2) with analytical gradient and Hessian:
   * Combines non-convex sinusoidal ripple (saddle regions) + softmax-like ridge + L2 well.
   */
  function evaluateLossSurface(w1, w2, lambda = 0.22, coupling = 1.0) {
    const c = clamp(coupling, 0.2, 2.5);
    const ridge1 = Math.sin(1.7 * w1) * Math.cos(1.35 * w2);
    const base = 0.28 * c * (w1 * w1 + 1.35 * w2 * w2) +
      lambda * 0.5 * (w1 * w1 + w2 * w2) +
      0.32 * Math.exp(-((w1 - 0.55) * (w1 - 0.55) + (w2 + 0.4) * (w2 + 0.4)) * 1.4) +
      0.24 * (1 - ridge1);

    const kQuad = 0.56 * c + lambda;
    const kQuadY = 0.756 * c + lambda;
    const dxG = (w1 - 0.55), dyG = (w2 + 0.4);
    const gauss = 0.32 * Math.exp(-(dxG * dxG + dyG * dyG) * 1.4);

    const d1 = kQuad * w1 - 2.8 * dxG * gauss - 0.24 * 1.7 * Math.cos(1.7 * w1) * Math.cos(1.35 * w2);
    const d2 = kQuadY * w2 - 2.8 * dyG * gauss + 0.24 * 1.35 * Math.sin(1.7 * w1) * Math.sin(1.35 * w2);

    const h11 = kQuad - 2.8 * gauss * (1 - 2.8 * dxG * dxG) + 0.24 * 1.7 * 1.7 * Math.sin(1.7 * w1) * Math.cos(1.35 * w2);
    const h22 = kQuadY - 2.8 * gauss * (1 - 2.8 * dyG * dyG) + 0.24 * 1.35 * 1.35 * Math.sin(1.7 * w1) * Math.cos(1.35 * w2);
    const h12 = 2.8 * 2.8 * dxG * dyG * gauss + 0.24 * 1.7 * 1.35 * Math.cos(1.7 * w1) * Math.sin(1.35 * w2);

    return {
      loss: base,
      gradient: [d1, d2],
      gradNorm: Math.hypot(d1, d2),
      hessian: [[h11, h12], [h12, h22]]
    };
  }

  function traceDescentPath(startW, lr = 0.14, steps = 28, lambda = 0.22, coupling = 1.0) {
    const trail = [];
    let w1 = startW[0], w2 = startW[1];
    for (let i = 0; i <= steps; i++) {
      const s = evaluateLossSurface(w1, w2, lambda, coupling);
      trail.push({ w1, w2, loss: s.loss, gradNorm: s.gradNorm, gradient: s.gradient, hessian: s.hessian });
      const stepScale = clamp(lr, 0.01, 0.45);
      w1 = clamp(w1 - stepScale * s.gradient[0], -1.75, 1.75);
      w2 = clamp(w2 - stepScale * s.gradient[1], -1.75, 1.75);
    }
    return trail;
  }

  function heatmapColor(t) {
    const u = clamp(t, 0, 1);
    if (u < 0.5) {
      const k = u * 2;
      return [
        0.10 + k * 0.22,
        0.48 + k * 0.26,
        0.47 + k * 0.12,
        1
      ];
    }
    const k = (u - 0.5) * 2;
    return [
      0.32 + k * 0.45,
      0.74 - k * 0.36,
      0.59 - k * 0.42,
      1
    ];
  }

  function pushTri(buf, p0, n0, c0, p1, n1, c1, p2, n2, c2) {
    buf.pos.push(...p0, ...p1, ...p2);
    buf.norm.push(...n0, ...n1, ...n2);
    buf.col.push(...c0, ...c1, ...c2);
    buf.tris.push({ p0, p1, p2, col: [(c0[0] + c1[0] + c2[0]) / 3, (c0[1] + c1[1] + c2[1]) / 3, (c0[2] + c1[2] + c2[2]) / 3, (c0[3] + c1[3] + c2[3]) / 3] });
  }

  function pushSeg(buf, a, b, col) {
    buf.linePos.push(...a, ...b);
    buf.lineCol.push(...col, ...col);
    buf.lines.push({ a, b, col });
  }

  function pushPt(buf, pos, col, size) {
    buf.ptPos.push(...pos);
    buf.ptCol.push(...col);
    buf.ptSize.push(size);
    buf.pts.push({ pos, col, size });
  }

  function addGridSurface(buf, cols, rows, evaluator, scaleY = 1.0) {
    const span = 1.65;
    const grid = [];
    let minZ = Infinity, maxZ = -Infinity;
    for (let j = 0; j <= rows; j++) {
      const row = [];
      const u2 = -span + (2 * span * j) / rows;
      for (let i = 0; i <= cols; i++) {
        const u1 = -span + (2 * span * i) / cols;
        const val = evaluator(u1, u2);
        minZ = Math.min(minZ, val.h);
        maxZ = Math.max(maxZ, val.h);
        row.push({ x: u1, z: u2, y: val.h, gx: val.gx, gz: val.gz, raw: val.raw });
      }
      grid.push(row);
    }
    const denom = Math.max(1e-5, maxZ - minZ);
    for (let j = 0; j <= rows; j++) {
      for (let i = 0; i <= cols; i++) {
        const pt = grid[j][i];
        pt.yn = ((pt.y - minZ) / denom - 0.5) * 1.45 * scaleY;
        const n = vecNormalize([-pt.gx * 0.55, 1, -pt.gz * 0.55]);
        pt.norm = n;
        pt.col = heatmapColor((pt.y - minZ) / denom);
      }
    }
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i + 1], d = grid[j + 1][i];
        const pa = [a.x, a.yn, a.z], pb = [b.x, b.yn, b.z], pc = [c.x, c.yn, c.z], pd = [d.x, d.yn, d.z];
        pushTri(buf, pa, a.norm, a.col, pb, b.norm, b.col, pc, c.norm, c.col);
        pushTri(buf, pa, a.norm, a.col, pc, c.norm, c.col, pd, d.norm, d.col);
      }
    }
    for (let j = 0; j <= rows; j += 3) {
      for (let i = 0; i < cols; i++) {
        const a = grid[j][i], b = grid[j][i + 1];
        pushSeg(buf, [a.x, a.yn + 0.01, a.z], [b.x, b.yn + 0.01, b.z], [0.85, 0.94, 0.89, 0.25]);
      }
    }
    for (let i = 0; i <= cols; i += 3) {
      for (let j = 0; j < rows; j++) {
        const a = grid[j][i], b = grid[j + 1][i];
        pushSeg(buf, [a.x, a.yn + 0.01, a.z], [b.x, b.yn + 0.01, b.z], [0.85, 0.94, 0.89, 0.25]);
      }
    }
    return { minZ, maxZ, denom };
  }

  function buildSceneData(state = {}) {
    const mode = MODES.includes(state.mode) ? state.mode : "landscape";
    const presetKey = PRESETS[state.preset] ? state.preset : "moons";
    const preset = PRESETS[presetKey];
    const rate = clamp(Number(state.rate ?? 0.16), 0.02, 0.42);
    const phase = Number(state.phase ?? 0.4);

    const buf = { pos: [], norm: [], col: [], linePos: [], lineCol: [], ptPos: [], ptCol: [], ptSize: [], tris: [], lines: [], pts: [] };
    // Subtle 3D floor reference frame
    for (let k = -2; k <= 2; k++) {
      const c = k === 0 ? [0.46, 0.72, 0.66, 0.38] : [0.45, 0.64, 0.6, 0.16];
      pushSeg(buf, [k * 0.8, -0.78, -1.65], [k * 0.8, -0.78, 1.65], c);
      pushSeg(buf, [-1.65, -0.78, k * 0.8], [1.65, -0.78, k * 0.8], c);
    }

    let summary = { title: "", subtitle: "", formula: "", readout: "" };

    if (mode === "landscape") {
      const coupling = presetKey === "xor" ? 0.85 : presetKey === "circles" ? 1.25 : 1.05;
      const lambda = 0.22;
      const stats = addGridSurface(buf, 26, 26, (w1, w2) => {
        const s = evaluateLossSurface(w1, w2, lambda, coupling);
        return { h: s.loss, gx: s.gradient[0], gz: s.gradient[1], raw: s.loss };
      });
      const trail = traceDescentPath([1.25, 1.15], rate, 26, lambda, coupling);
      const yMap = (loss) => ((loss - stats.minZ) / stats.denom - 0.5) * 1.45;
      for (let i = 0; i < trail.length - 1; i++) {
        const a = trail[i], b = trail[i + 1];
        pushSeg(buf, [a.w1, yMap(a.loss) + 0.04, a.w2], [b.w1, yMap(b.loss) + 0.04, b.w2], [0.98, 0.76, 0.38, 0.96]);
        pushPt(buf, [a.w1, yMap(a.loss) + 0.04, a.w2], [0.98, 0.82, 0.45, 0.9], i % 4 === 0 ? 7 : 4.5);
      }
      const idx = clamp(Math.floor(((phase * 0.45) % 1) * (trail.length - 1)), 0, trail.length - 1);
      const head = trail[idx];
      const hy = yMap(head.loss) + 0.05;
      pushPt(buf, [head.w1, hy, head.w2], [0.99, 0.99, 0.82, 1], 14);
      const gScale = 0.25 / Math.max(0.4, head.gradNorm);
      pushSeg(buf, [head.w1, hy, head.w2], [head.w1 - head.gradient[0] * gScale, hy, head.w2 - head.gradient[1] * gScale], [0.42, 0.96, 0.75, 1]);
      summary = {
        title: "Non-Convex Loss Surface & Gradient Flow",
        subtitle: "Saddle ridges and regularised bowls over parameter coordinates (w₁, w₂) with live −η∇J trajectory.",
        formula: "w′ = w − η∇J(w) · Hᵢⱼ = ∂²J/∂wᵢ∂wⱼ",
        readout: `J = ${head.loss.toFixed(3)} · ‖∇J‖ = ${head.gradNorm.toFixed(3)} · tr(H) = ${(head.hessian[0][0] + head.hessian[1][1]).toFixed(2)} · step ${idx}/${trail.length - 1}`
      };
    } else if (mode === "manifold") {
      const amp = 0.72 + rate * 1.4;
      const stats = addGridSurface(buf, 26, 26, (x1, x2) => {
        const res = evaluateMLP(preset, x1, x2);
        return { h: res.p * amp, gx: res.gradient[0] * amp, gz: res.gradient[1] * amp, raw: res.p };
      }, 0.95);
      const yMap = (val) => ((val - stats.minZ) / stats.denom - 0.5) * 1.45 * 0.95;
      const midY = yMap(0.5 * amp);
      pushSeg(buf, [-1.65, midY, -1.65], [1.65, midY, -1.65], [0.96, 0.85, 0.45, 0.65]);
      pushSeg(buf, [1.65, midY, -1.65], [1.65, midY, 1.65], [0.96, 0.85, 0.45, 0.65]);
      pushSeg(buf, [1.65, midY, 1.65], [-1.65, midY, 1.65], [0.96, 0.85, 0.45, 0.65]);
      pushSeg(buf, [-1.65, midY, 1.65], [-1.65, midY, -1.65], [0.96, 0.85, 0.45, 0.65]);
      for (const [px, pz, target] of preset.points) {
        const py = yMap(evaluateMLP(preset, px, pz).p * amp) + 0.05;
        const c = target === 1 ? [0.22, 0.92, 0.78, 1] : [0.92, 0.48, 0.30, 1];
        pushPt(buf, [px, py, pz], c, 10.5);
        pushSeg(buf, [px, -0.78, pz], [px, py, pz], [c[0], c[1], c[2], 0.42]);
      }
      const probeX = 0.4 * Math.cos(phase * 0.8);
      const probeZ = 0.4 * Math.sin(phase * 0.8);
      const probe = evaluateMLP(preset, probeX, probeZ);
      const probeY = yMap(probe.p * amp) + 0.06;
      pushPt(buf, [probeX, probeY, probeZ], [0.99, 0.98, 0.72, 1], 13);
      summary = {
        title: `Decision Manifold · ${preset.label}`,
        subtitle: preset.description,
        formula: "p(x) = σ(tanh(xW₁ + b₁)W₂ + b₂) · ∇ₓp = p(1−p)·JₓᵀW₂",
        readout: `x = (${probeX.toFixed(2)}, ${probeZ.toFixed(2)}) · p = ${probe.p.toFixed(3)} · ‖∇ₓp‖ = ${Math.hypot(probe.gradient[0], probe.gradient[1]).toFixed(3)}`
      };
    } else {
      // 3D Feedforward Computation Graph: 2 -> 4 -> 1 across z planes
      const inputs = [[-0.65, 0.35, -1.35], [0.65, -0.35, -1.35]];
      const hiddens = [
        [-1.05, 0.62, 0.0], [-0.35, -0.58, 0.0], [0.35, 0.62, 0.0], [1.05, -0.58, 0.0]
      ];
      const out = [0, 0.05, 1.4];
      const probe = evaluateMLP(preset, Math.cos(phase * 0.7), Math.sin(phase * 0.7));
      for (let i = 0; i < inputs.length; i++) {
        pushPt(buf, inputs[i], [0.88, 0.95, 0.86, 1], 15);
        for (let j = 0; j < hiddens.length; j++) {
          const w = preset.W1[i][j];
          const col = w >= 0 ? [0.18, 0.88, 0.74, 0.78] : [0.89, 0.46, 0.28, 0.78];
          pushSeg(buf, inputs[i], hiddens[j], col);
        }
      }
      for (let j = 0; j < hiddens.length; j++) {
        const act = Math.abs(probe.h[j]);
        pushPt(buf, hiddens[j], [0.16 + act * 0.25, 0.62 + act * 0.33, 0.68, 1], 14 + act * 5);
        const w = preset.W2[j];
        const col = w >= 0 ? [0.22, 0.92, 0.78, 0.86] : [0.92, 0.48, 0.30, 0.86];
        pushSeg(buf, hiddens[j], out, col);
      }
      pushPt(buf, out, [0.98, 0.84, 0.42, 1], 20);
      // Animated signal pulse packet
      const u = (phase * 0.45) % 1;
      for (let j = 0; j < 4; j++) {
        const src = j < 2 ? inputs[0] : inputs[1];
        const mid = hiddens[j];
        const pos = u < 0.5
          ? [src[0] + (mid[0] - src[0]) * (u * 2), src[1] + (mid[1] - src[1]) * (u * 2), src[2] + (mid[2] - src[2]) * (u * 2)]
          : [mid[0] + (out[0] - mid[0]) * ((u - 0.5) * 2), mid[1] + (out[1] - mid[1]) * ((u - 0.5) * 2), mid[2] + (out[2] - mid[2]) * ((u - 0.5) * 2)];
        pushPt(buf, pos, [0.99, 0.96, 0.76, 1], 9);
      }
      summary = {
        title: "3D Feedforward & Backpropagation Graph",
        subtitle: "2 inputs → 4 tanh hidden units → 1 sigmoid output rendered in 3D weight space.",
        formula: "dZ₁ = (dZ₂W₂ᵀ) ⊙ sech²(Z₁) · dW₁ = XᵀdZ₁",
        readout: `Forward p = ${probe.p.toFixed(3)} · max|h| = ${Math.max(...probe.h.map(Math.abs)).toFixed(3)} · η = ${rate.toFixed(2)}`
      };
    }

    return { mode, preset: presetKey, buffers: buf, summary };
  }

  const api = {
    TAU,
    MODES,
    PRESETS,
    clamp,
    sigmoid,
    vecNormalize,
    vecCross,
    lookAtMatrix4,
    perspectiveMatrix4,
    mat4Mul,
    orbitCameraEye,
    projectPoint3D,
    evaluateMLP,
    evaluateLossSurface,
    traceDescentPath,
    buildSceneData
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NeuralManifoldMath = api;
})(typeof window !== "undefined" ? window : globalThis);
