(function (root) {
  "use strict";
  if (typeof document === "undefined") return;

  const M = root.NeuralManifoldMath;
  if (!M) return;

  const SURFACE_VS = `#version 300 es
    in vec3 aPos;
    in vec3 aNorm;
    in vec4 aCol;
    uniform mat4 uMvp;
    out vec3 vPos;
    out vec3 vNorm;
    out vec4 vCol;
    void main() {
      vPos = aPos;
      vNorm = normalize(aNorm);
      vCol = aCol;
      gl_Position = uMvp * vec4(aPos, 1.0);
    }`;

  const SURFACE_FS = `#version 300 es
    precision highp float;
    in vec3 vPos;
    in vec3 vNorm;
    in vec4 vCol;
    uniform vec3 uEye;
    out vec4 fragColor;
    void main() {
      vec3 N = normalize(vNorm);
      vec3 V = normalize(uEye - vPos);
      vec3 L = normalize(vec3(2.2, 3.6, 2.8));
      vec3 H = normalize(L + V);
      float diff = abs(dot(N, L)) * 0.72 + 0.28;
      float spec = pow(max(abs(dot(N, H)), 0.0), 32.0) * 0.42;
      float fres = pow(1.0 - max(abs(dot(N, V)), 0.0), 3.0) * 0.22;
      vec3 rgb = vCol.rgb * diff + vec3(0.86, 0.96, 0.91) * (spec + fres);
      fragColor = vec4(rgb, vCol.a);
    }`;

  const LINE_VS = `#version 300 es
    in vec3 aPos;
    in vec4 aCol;
    uniform mat4 uMvp;
    uniform float uPointSize;
    out vec4 vCol;
    void main() {
      vCol = aCol;
      gl_Position = uMvp * vec4(aPos, 1.0);
      gl_PointSize = uPointSize;
    }`;

  const LINE_FS = `#version 300 es
    precision highp float;
    in vec4 vCol;
    uniform float uIsPoint;
    out vec4 fragColor;
    void main() {
      if (uIsPoint > 0.5) {
        vec2 c = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(c, c);
        if (r2 > 1.0) discard;
        float glow = exp(-r2 * 2.2);
        fragColor = vec4(vCol.rgb * (0.7 + 0.5 * glow), vCol.a * (1.0 - smoothstep(0.65, 1.0, r2)));
      } else {
        fragColor = vCol;
      }
    }`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      gl.deleteShader(sh);
      throw new Error("Shader compile error");
    }
    return sh;
  }

  function linkProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      gl.deleteProgram(prog);
      throw new Error("Program link error");
    }
    return prog;
  }

  function createWebGL2Backend(canvas) {
    let gl = null;
    try {
      gl = canvas.getContext("webgl2", { antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    } catch {
      return null;
    }
    if (!gl) return null;

    let surfProg, lineProg;
    try {
      surfProg = linkProgram(gl, SURFACE_VS, SURFACE_FS);
      lineProg = linkProgram(gl, LINE_VS, LINE_FS);
    } catch {
      return null;
    }

    const posB = gl.createBuffer(), normB = gl.createBuffer(), colB = gl.createBuffer();
    const lPosB = gl.createBuffer(), lColB = gl.createBuffer();
    const pPosB = gl.createBuffer(), pColB = gl.createBuffer();

    const vaoSurf = gl.createVertexArray();
    gl.bindVertexArray(vaoSurf);
    const bindAttr = (prog, name, buf, size) => {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    bindAttr(surfProg, "aPos", posB, 3);
    bindAttr(surfProg, "aNorm", normB, 3);
    bindAttr(surfProg, "aCol", colB, 4);

    const vaoLine = gl.createVertexArray();
    gl.bindVertexArray(vaoLine);
    bindAttr(lineProg, "aPos", lPosB, 3);
    bindAttr(lineProg, "aCol", lColB, 4);

    const vaoPt = gl.createVertexArray();
    gl.bindVertexArray(vaoPt);
    bindAttr(lineProg, "aPos", pPosB, 3);
    bindAttr(lineProg, "aCol", pColB, 4);
    gl.bindVertexArray(null);

    return {
      kind: "webgl2",
      render(scene, mvp, eye, width, height, dpr) {
        if (gl.isContextLost()) return false;
        gl.viewport(0, 0, width, height);
        gl.clearColor(0.06, 0.16, 0.15, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const b = scene.buffers;
        if (b.pos.length) {
          gl.useProgram(surfProg);
          gl.uniformMatrix4fv(gl.getUniformLocation(surfProg, "uMvp"), false, mvp);
          gl.uniform3f(gl.getUniformLocation(surfProg, "uEye"), eye[0], eye[1], eye[2]);
          gl.bindVertexArray(vaoSurf);
          gl.bindBuffer(gl.ARRAY_BUFFER, posB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.pos), gl.DYNAMIC_DRAW);
          gl.bindBuffer(gl.ARRAY_BUFFER, normB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.norm), gl.DYNAMIC_DRAW);
          gl.bindBuffer(gl.ARRAY_BUFFER, colB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.col), gl.DYNAMIC_DRAW);
          gl.drawArrays(gl.TRIANGLES, 0, b.pos.length / 3);
        }
        if (b.linePos.length) {
          gl.useProgram(lineProg);
          gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uMvp"), false, mvp);
          gl.uniform1f(gl.getUniformLocation(lineProg, "uIsPoint"), 0.0);
          gl.uniform1f(gl.getUniformLocation(lineProg, "uPointSize"), 1.0);
          gl.bindVertexArray(vaoLine);
          gl.bindBuffer(gl.ARRAY_BUFFER, lPosB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.linePos), gl.DYNAMIC_DRAW);
          gl.bindBuffer(gl.ARRAY_BUFFER, lColB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.lineCol), gl.DYNAMIC_DRAW);
          gl.drawArrays(gl.LINES, 0, b.linePos.length / 3);
        }
        if (b.ptPos.length) {
          gl.useProgram(lineProg);
          gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uMvp"), false, mvp);
          gl.uniform1f(gl.getUniformLocation(lineProg, "uIsPoint"), 1.0);
          gl.uniform1f(gl.getUniformLocation(lineProg, "uPointSize"), 11.5 * dpr);
          gl.bindVertexArray(vaoPt);
          gl.bindBuffer(gl.ARRAY_BUFFER, pPosB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.ptPos), gl.DYNAMIC_DRAW);
          gl.bindBuffer(gl.ARRAY_BUFFER, pColB); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.ptCol), gl.DYNAMIC_DRAW);
          gl.drawArrays(gl.POINTS, 0, b.ptPos.length / 3);
        }
        gl.bindVertexArray(null);
        return true;
      }
    };
  }

  function createCanvas2DBackend(canvas) {
    let ctx = null;
    try { ctx = canvas.getContext("2d"); } catch { return null; }
    if (!ctx) return null;
    const rgba = (c) => `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)}, ${(c[3] ?? 1).toFixed(3)})`;
    return {
      kind: "canvas2d",
      render(scene, mvp, _eye, width, height, dpr) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        const grad = ctx.createRadialGradient(width * 0.35, height * 0.25, width * 0.05, width * 0.5, height * 0.5, width * 0.8);
        grad.addColorStop(0, "#19433e"); grad.addColorStop(1, "#0d2220");
        ctx.fillStyle = grad; ctx.fillRect(0, 0, width, height);

        const queue = [];
        for (const t of scene.buffers.tris) {
          const a = M.projectPoint3D(t.p0, mvp, width, height);
          const b = M.projectPoint3D(t.p1, mvp, width, height);
          const c = M.projectPoint3D(t.p2, mvp, width, height);
          if (!a.visible || !b.visible || !c.visible) continue;
          queue.push({ k: "t", d: (a.depth + b.depth + c.depth) / 3, a, b, c, col: t.col });
        }
        for (const l of scene.buffers.lines) {
          const a = M.projectPoint3D(l.a, mvp, width, height);
          const b = M.projectPoint3D(l.b, mvp, width, height);
          if (!a.visible && !b.visible) continue;
          queue.push({ k: "l", d: (a.depth + b.depth) * 0.5, a, b, col: l.col });
        }
        for (const p of scene.buffers.pts) {
          const q = M.projectPoint3D(p.pos, mvp, width, height);
          if (!q.visible) continue;
          queue.push({ k: "p", d: q.depth, q, col: p.col, size: p.size * dpr });
        }
        queue.sort((u, v) => v.d - u.d);
        for (const item of queue) {
          if (item.k === "t") {
            ctx.beginPath(); ctx.moveTo(item.a.x, item.a.y); ctx.lineTo(item.b.x, item.b.y); ctx.lineTo(item.c.x, item.c.y); ctx.closePath();
            ctx.fillStyle = rgba(item.col); ctx.fill();
          } else if (item.k === "l") {
            ctx.beginPath(); ctx.moveTo(item.a.x, item.a.y); ctx.lineTo(item.b.x, item.b.y);
            ctx.strokeStyle = rgba(item.col); ctx.lineWidth = Math.max(1, 1.25 * dpr); ctx.stroke();
          } else {
            ctx.beginPath(); ctx.arc(item.q.x, item.q.y, Math.max(2, item.size * 0.42), 0, M.TAU);
            ctx.fillStyle = rgba(item.col); ctx.fill();
          }
        }
        return true;
      }
    };
  }

  function mountManifoldStudio(section) {
    if (!section) return null;
    const stage = section.querySelector(".nml3d-stage");
    const canvas = section.querySelector("#nml3d-canvas");
    const rendererBadge = section.querySelector("#nml3d-renderer");
    const motionBadge = section.querySelector("#nml3d-motion-state");
    const titleEl = section.querySelector("#nml3d-title");
    const descEl = section.querySelector("#nml3d-desc");
    const formulaEl = section.querySelector("#nml3d-formula");
    const readoutEl = section.querySelector("#nml3d-readout");
    const rateInput = section.querySelector("#nml3d-rate");
    const rateVal = section.querySelector("#nml3d-rate-value");
    const pauseBtn = section.querySelector("#nml3d-pause");

    const reduceQuery = typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : { matches: false };

    let state = { mode: "landscape", preset: "moons", rate: 0.16, userPaused: false };
    let camera = { yaw: 0.62, pitch: 0.48, distance: 4.35 };
    let phase = 0.35;
    let rafId = 0;
    let lastTime = 0;
    let resizeRaf = 0;

    let backend = canvas ? (createWebGL2Backend(canvas) || createCanvas2DBackend(canvas)) : null;

    const isPaused = () => Boolean(reduceQuery.matches || state.userPaused);

    function syncUi(scene) {
      section.dataset.mode = state.mode;
      section.dataset.preset = state.preset;
      section.dataset.renderer = backend ? backend.kind : "none";
      section.dataset.motion = isPaused() ? "paused" : "running";
      if (rendererBadge) rendererBadge.textContent = backend?.kind === "webgl2" ? "WebGL2 3D" : "Canvas 2D";
      if (motionBadge) motionBadge.textContent = reduceQuery.matches ? "Reduced motion" : state.userPaused ? "Paused" : "Live flow";
      if (pauseBtn) {
        pauseBtn.setAttribute("aria-pressed", String(isPaused()));
        pauseBtn.textContent = reduceQuery.matches ? "Motion paused" : state.userPaused ? "Resume 3D" : "Pause 3D";
        pauseBtn.disabled = Boolean(reduceQuery.matches);
      }
      section.querySelectorAll("[data-nml-mode]").forEach((btn) => {
        btn.setAttribute("aria-pressed", String(btn.dataset.nmlMode === state.mode));
      });
      section.querySelectorAll("[data-nml-preset]").forEach((btn) => {
        btn.setAttribute("aria-pressed", String(btn.dataset.nmlPreset === state.preset));
      });
      if (rateVal) rateVal.textContent = state.rate.toFixed(2);
      if (titleEl) titleEl.textContent = scene.summary.title;
      if (descEl) descEl.textContent = scene.summary.subtitle;
      if (formulaEl) formulaEl.textContent = scene.summary.formula;
      if (readoutEl) readoutEl.textContent = scene.summary.readout;
    }

    function drawFrame(updateCopy = false) {
      if (!canvas || !backend) return;
      const host = stage || canvas;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const cssW = Math.max(240, Math.round(host.clientWidth || 560));
      const cssH = Math.max(200, Math.round(host.clientHeight || 340));
      const width = Math.round(cssW * dpr);
      const height = Math.round(cssH * dpr);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const scene = M.buildSceneData({ ...state, phase });
      const eye = M.orbitCameraEye(camera.yaw, camera.pitch, camera.distance);
      const view = M.lookAtMatrix4(eye, [0, 0, 0], [0, 1, 0]);
      const proj = M.perspectiveMatrix4(0.68, width / Math.max(1, height), 0.1, 30);
      const mvp = M.mat4Mul(proj, view);
      const ok = backend.render(scene, mvp, eye, width, height, dpr);
      if (!ok && backend.kind === "webgl2") {
        backend = createCanvas2DBackend(canvas);
        if (backend) backend.render(scene, mvp, eye, width, height, dpr);
      }
      if (updateCopy) syncUi(scene);
    }

    function tick(now) {
      rafId = 0;
      if (isPaused()) return;
      const dt = lastTime ? Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000)) : 1 / 60;
      lastTime = now;
      phase += dt * 0.45;
      camera.yaw += dt * 0.16;
      drawFrame(false);
      rafId = requestAnimationFrame(tick);
    }

    function scheduleLoop() {
      if (rafId || isPaused()) return;
      lastTime = performance.now();
      rafId = requestAnimationFrame(tick);
    }

    function stopLoop() {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      lastTime = 0;
    }

    section.addEventListener("click", (event) => {
      const mBtn = event.target.closest?.("[data-nml-mode]");
      if (mBtn) { state.mode = mBtn.dataset.nmlMode; drawFrame(true); return; }
      const pBtn = event.target.closest?.("[data-nml-preset]");
      if (pBtn) { state.preset = pBtn.dataset.nmlPreset; drawFrame(true); return; }
      const aBtn = event.target.closest?.("[data-nml-action]");
      if (!aBtn) return;
      const act = aBtn.dataset.nmlAction;
      if (act === "pause" && !reduceQuery.matches) {
        state.userPaused = !state.userPaused;
        drawFrame(true);
        if (isPaused()) stopLoop(); else scheduleLoop();
      } else if (act === "zoom-in") { camera.distance = M.clamp(camera.distance - 0.45, 2.4, 7.6); drawFrame(false); }
      else if (act === "zoom-out") { camera.distance = M.clamp(camera.distance + 0.45, 2.4, 7.6); drawFrame(false); }
      else if (act === "reset") { camera = { yaw: 0.62, pitch: 0.48, distance: 4.35 }; phase = 0.35; state.rate = 0.16; if (rateInput) rateInput.value = "0.16"; drawFrame(true); }
    });

    if (rateInput) {
      rateInput.addEventListener("input", () => {
        state.rate = M.clamp(Number(rateInput.value), 0.02, 0.42);
        drawFrame(true);
      });
    }

    let dragging = false, lx = 0, ly = 0;
    if (stage) {
      stage.addEventListener("pointerdown", (e) => {
        if (e.button !== undefined && e.button !== 0) return;
        dragging = true; lx = e.clientX; ly = e.clientY;
        stage.classList.add("is-dragging");
        stage.setPointerCapture?.(e.pointerId);
      });
      stage.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        camera.yaw -= (e.clientX - lx) * 0.009;
        camera.pitch = M.clamp(camera.pitch + (e.clientY - ly) * 0.008, -1.35, 1.35);
        lx = e.clientX; ly = e.clientY;
        drawFrame(false);
      });
      const endDrag = (e) => {
        if (!dragging) return;
        dragging = false;
        stage.classList.remove("is-dragging");
        stage.releasePointerCapture?.(e.pointerId);
      };
      stage.addEventListener("pointerup", endDrag);
      stage.addEventListener("pointercancel", endDrag);
      stage.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") { e.preventDefault(); camera.yaw -= 0.15; drawFrame(false); }
        else if (e.key === "ArrowRight") { e.preventDefault(); camera.yaw += 0.15; drawFrame(false); }
        else if (e.key === "ArrowUp") { e.preventDefault(); camera.pitch = M.clamp(camera.pitch + 0.12, -1.35, 1.35); drawFrame(false); }
        else if (e.key === "ArrowDown") { e.preventDefault(); camera.pitch = M.clamp(camera.pitch - 0.12, -1.35, 1.35); drawFrame(false); }
        else if (e.key === "+" || e.key === "=") { e.preventDefault(); camera.distance = M.clamp(camera.distance - 0.4, 2.4, 7.6); drawFrame(false); }
        else if (e.key === "-" || e.key === "_") { e.preventDefault(); camera.distance = M.clamp(camera.distance + 0.4, 2.4, 7.6); drawFrame(false); }
        else if (e.key === " " && !reduceQuery.matches) { e.preventDefault(); state.userPaused = !state.userPaused; drawFrame(true); if (isPaused()) stopLoop(); else scheduleLoop(); }
        else if (e.key === "r" || e.key === "R") { e.preventDefault(); camera = { yaw: 0.62, pitch: 0.48, distance: 4.35 }; phase = 0.35; drawFrame(true); }
      });
    }

    const ro = typeof ResizeObserver === "function" && stage
      ? new ResizeObserver(() => {
          if (resizeRaf) return;
          resizeRaf = requestAnimationFrame(() => { resizeRaf = 0; drawFrame(false); });
        })
      : null;
    ro?.observe(stage);

    reduceQuery.addEventListener?.("change", () => {
      if (reduceQuery.matches) stopLoop(); else if (!state.userPaused) scheduleLoop();
      drawFrame(true);
    });

    drawFrame(true);
    if (!isPaused()) scheduleLoop();
    return { getState: () => ({ ...state, camera: { ...camera }, renderer: backend?.kind || "none" }) };
  }

  function bindInteractiveMotion() {
    const reduceQuery = typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : { matches: false };
    const coarse = typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(pointer: coarse)")
      : { matches: false };
    const cards = [...document.querySelectorAll(".experiment-card, .learning-card, .network-figure")];
    if (!reduceQuery.matches && !coarse.matches) {
      for (const card of cards) {
        card.classList.add("tilt-3d");
        card.addEventListener("pointermove", (e) => {
          if (reduceQuery.matches) return;
          const r = card.getBoundingClientRect();
          if (!r.width || !r.height) return;
          const px = M.clamp((e.clientX - r.left) / r.width, 0, 1);
          const py = M.clamp((e.clientY - r.top) / r.height, 0, 1);
          card.style.setProperty("--rx", `${((0.5 - py) * 6.5).toFixed(2)}deg`);
          card.style.setProperty("--ry", `${((px - 0.5) * 6.5).toFixed(2)}deg`);
          card.style.setProperty("--mx", `${(px * 100).toFixed(1)}%`);
          card.style.setProperty("--my", `${(py * 100).toFixed(1)}%`);
          card.style.setProperty("--sheen", "1");
        });
        card.addEventListener("pointerleave", () => {
          card.style.setProperty("--rx", "0deg");
          card.style.setProperty("--ry", "0deg");
          card.style.setProperty("--sheen", "0");
        });
      }
    }
  }

  function init() {
    const studio = document.getElementById("manifold-studio");
    if (studio) studio.__manifoldStudio = mountManifoldStudio(studio);
    bindInteractiveMotion();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
