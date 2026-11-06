"use strict";

// Bounded HTTP/2 document-navigation load, not a browser or distributed load farm.
const http2 = require("node:http2");
const zlib = require("node:zlib");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { performance, monitorEventLoopDelay } = require("node:perf_hooks");
const ROUTES = ["/", "/labs/moons/", "/labs/circles/", "/learn/optimization-mathematics/"];
// One row for every completed request; maxRequests bounds retained samples.
const REQUEST_SAMPLE_FIELDS = ["dispatchOffsetMs", "completionOffsetMs", "responseMs", "dispatchDelayMs", "success"];
const DEFAULTS = { url: "https://neural-math-lab.pages.dev", vus: 100, rampSeconds: 10,
  holdSeconds: 65, thinkSeconds: 60, maxInflight: 128, maxRps: 400, connections: 8,
  timeoutSeconds: 15, readyGraceSeconds: 30, maxRequests: 200000, maxWireBytes: 3 * 1024 ** 3,
  maxFailureRate: 0.01, maxP95Ms: 2000, maxDispatchP95Ms: 1000, maxEventLoopP95Ms: 100 };

function options(input = {}) {
  const value = { ...DEFAULTS, ...input };
  const target = new URL(value.url);
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(target.hostname);
  if (target.username || target.password || target.pathname !== "/" || target.search || target.hash ||
      (!local && (target.hostname !== "neural-math-lab.pages.dev" || target.protocol !== "https:")) ||
      !["http:", "https:"].includes(target.protocol)) {
    throw new Error("Target must be the owned neural-math-lab.pages.dev origin or a loopback test server.");
  }
  value.url = target.origin;
  const ranges = { vus: [1, 10000, true], rampSeconds: [0, 600], holdSeconds: [.05, 600],
    thinkSeconds: [.01, 600], maxInflight: [1, 256, true], maxRps: [1, 1000], connections: [1, 16, true],
    timeoutSeconds: [.05, 60], readyGraceSeconds: [.05, 120], maxRequests: [1, 200000, true],
    maxWireBytes: [1024, 3 * 1024 ** 3, true], maxFailureRate: [0, .1], maxP95Ms: [1, 60000],
    maxDispatchP95Ms: [1, 60000], maxEventLoopP95Ms: [1, 1000] };
  for (const [name, [low, high, integer]] of Object.entries(ranges)) {
    if (!Number.isFinite(value[name]) || value[name] < low || value[name] > high || (integer && !Number.isInteger(value[name]))) {
      throw new Error(`${name} must be ${integer ? "an integer" : "a finite number"} between ${low} and ${high}.`);
    }
  }
  return value;
}

function quantiles(samples) {
  if (!samples.length) return { count: 0, p50: null, p95: null, p99: null, max: null };
  const sorted = samples.slice().sort((a, b) => a - b);
  const at = (fraction) => +sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)].toFixed(3);
  return { count: sorted.length, p50: at(.5), p95: at(.95), p99: at(.99), max: +sorted[sorted.length - 1].toFixed(3) };
}

function verifyHeaders(headers) {
  if (headers["cf-mitigated"] === "challenge") return "cloudflare_challenge";
  if (headers[":status"] !== 200) return "http_" + headers[":status"];
  if (!/^text\/html(?:;|$)/i.test(headers["content-type"] || "")) return "not_html";
  if (![undefined, "identity", "br", "gzip", "deflate"].includes(headers["content-encoding"])) return "unsupported_encoding";
  return null;
}

function verifyTitle(route, prefix) {
  const title = (prefix.match(/<title>([^<]+)<\/title>/i) || [])[1]?.replaceAll("&amp;", "&");
  if (!title || !title.includes("Neural Math Lab")) return false;
  const expected = { "/": "Understand the learning", "/labs/moons/": "Moons experiment",
    "/labs/circles/": "Circles experiment", "/learn/optimization-mathematics/": "Gradients & optimization" };
  return !Object.hasOwn(expected, route) || title.includes(expected[route]);
}

function loadExpectedHashes(directory) {
  return Object.fromEntries(ROUTES.map((route) => [route, crypto.createHash("sha256")
    .update(fs.readFileSync(path.join(directory, route.slice(1), "index.html"))).digest("hex")]));
}

function holdChecks(hold, cfg) {
  return {
    holdHadNavigationRequests: !!hold && hold.requests > 0,
    holdFailureRateWithinLimit: !!hold && hold.requests > 0 && hold.failures / hold.requests <= cfg.maxFailureRate,
    holdResponseP95WithinLimit: !!hold && hold.responseMs.p95 !== null && hold.responseMs.p95 <= cfg.maxP95Ms,
    holdDispatchP95WithinLimit: !!hold && hold.dispatchMs.p95 !== null && hold.dispatchMs.p95 <= cfg.maxDispatchP95Ms
  };
}

function requestDocument(session, route, timeoutMs, expectedHash = null) {
  return new Promise((resolve) => {
    const began = performance.now();
    let stream, decoder, headers = {}, wireBytes = 0, decodedBytes = 0, prefix = Buffer.alloc(0), ended = false;
    const hash = crypto.createHash("sha256");
    const finish = (failure) => {
      if (ended) return; ended = true; clearTimeout(timer);
      if (failure) { stream?.close(http2.constants.NGHTTP2_CANCEL); decoder?.destroy(); }
      resolve({ ok: !failure, failure: failure || null, status: headers[":status"] || null,
        latencyMs: performance.now() - began, wireBytes, decodedBytes,
        cache: String(headers["cf-cache-status"] || "unknown"),
        pop: String(headers["cf-ray"] || "").split("-").slice(1).join("-") || "unknown",
        etag: headers.etag ? String(headers.etag) : null });
    };
    const timer = setTimeout(() => finish("timeout"), timeoutMs);
    function decoded(chunk) {
      if (ended) return;
      decodedBytes += chunk.length;
      if (decodedBytes > 4 * 1024 ** 2) { finish("decoded_body_too_large"); return; }
      hash.update(chunk);
      if (prefix.length < 8192) prefix = Buffer.concat([prefix, chunk.subarray(0, 8192 - prefix.length)]);
    }
    function complete() {
      const expectedLength = headers["content-length"];
      if (expectedLength !== undefined && Number(expectedLength) !== wireBytes) return finish("content_length_mismatch");
      if (wireBytes < 32 || decodedBytes < 100) return finish("empty_or_short_body");
      if (!verifyTitle(route, prefix.toString("utf8"))) return finish("wrong_page_title");
      if (expectedHash && hash.digest("hex") !== expectedHash) return finish("body_hash_mismatch");
      finish(null);
    }
    try {
      stream = session.request({ ":method": "GET", ":path": route, "accept": "text/html",
        "accept-encoding": "br, gzip", "user-agent": "NeuralMathLab-owned-capacity-check/1.0" });
      stream.on("response", (received) => {
        headers = received;
        const error = verifyHeaders(headers); if (error) return finish(error);
        const encoding = headers["content-encoding"];
        decoder = encoding === "br" ? zlib.createBrotliDecompress() : encoding === "gzip" ? zlib.createGunzip() : encoding === "deflate" ? zlib.createInflate() : null;
        if (decoder) {
          decoder.on("data", decoded); decoder.on("end", complete);
          decoder.on("error", () => finish("decompression_error"));
          decoder.on("drain", () => { if (!ended) stream.resume(); });
        }
      });
      stream.on("data", (chunk) => {
        if (ended) return; wireBytes += chunk.length;
        if (wireBytes > 2 * 1024 ** 2) return finish("wire_body_too_large");
        if (decoder) { if (!decoder.write(chunk)) stream.pause(); } else decoded(chunk);
      });
      stream.on("end", () => { if (!ended) { if (decoder) decoder.end(); else complete(); } });
      stream.on("error", () => finish("stream_error"));
      stream.on("aborted", () => finish("stream_aborted"));
      stream.on("close", () => { if (!ended && !stream.readableEnded) finish("stream_closed"); });
      stream.end();
    } catch (_) { finish("request_start_error"); }
  });
}

async function runLoadTest(input = {}, onProgress = () => {}, signal = null) {
  const cfg = options(input), startedAt = new Date().toISOString(), start = performance.now();
  if (!cfg.expectedHashes || ROUTES.some((route) => !/^[a-f0-9]{64}$/.test(cfg.expectedHashes[route] || ""))) {
    throw new Error("A known SHA-256 expected body hash is required for every workload route.");
  }
  const loop = monitorEventLoopDelay({ resolution: 20 }); loop.enable();
  const users = Array.from({ length: cfg.vus }, (_, id) => ({ id, activated: false, ready: false, busy: false,
    nextAt: start + id * cfg.rampSeconds * 1000 / cfg.vus, iteration: 0 }));
  const stats = { scheduledUsers: cfg.vus, activatedUsers: 0, readyUsers: 0, peakActiveUsers: 0,
    requests: 0, completed: 0, successes: 0, failures: 0, inflight: 0, peakInflight: 0, peakQueuedUsers: 0,
    wireBytes: 0, decodedBytes: 0, peakRssBytes: process.memoryUsage().rss,
    readyAtMs: null, targetStartedAtMs: null, holdStartedAtMs: null, holdFinishedAtMs: null };
  const latencies = [], dispatchDelays = [], requestSamples = [], failures = {}, cache = {}, pops = {}, routeStats = {}, stageStats = {};
  const recent = [], sessions = Array.from({ length: cfg.connections }, () => ({ session: null, active: 0, ready: false, retryAt: 0, maxStreams: 100 }));
  let tokens = 0, lastTick = start, holdStart = null, stopping = false, stoppedAt = null, abortReason = null, lastProgress = start;
  let readyDeadline = start + (cfg.rampSeconds + cfg.readyGraceSeconds) * 1000;
  const increment = (object, key) => { object[key] = (object[key] || 0) + 1; };
  function abort(reason) { if (!stopping) { stopping = true; abortReason = reason; stoppedAt = performance.now(); } }
  function phase(now) { return holdStart !== null ? "hold" : now - start < cfg.rampSeconds * 1000 ? "ramp" : "waiting_for_verified_users"; }
  function openConnections(now) {
    for (const slot of sessions) {
      if (slot.session || now < slot.retryAt) continue;
      const session = http2.connect(cfg.url); slot.session = session; slot.ready = false;
      const connectTimer = setTimeout(() => session.destroy(), cfg.timeoutSeconds * 1000);
      session.on("connect", () => { clearTimeout(connectTimer); slot.ready = true; });
      session.on("remoteSettings", (settings) => { slot.maxStreams = settings.maxConcurrentStreams || 100; });
      session.on("error", () => {}); // Per-request failures retain the observable error count.
      session.on("goaway", () => { slot.ready = false; session.close(); });
      session.on("close", () => { clearTimeout(connectTimer); slot.ready = false; slot.session = null; slot.retryAt = performance.now() + 1000; });
    }
  }
  function record(user, route, stage, slot, dispatchedAt, dispatchDelay, result) {
    const now = performance.now(); user.busy = false; slot.active--; stats.inflight--; stats.completed++;
    requestSamples.push([dispatchedAt - start, now - start, result.latencyMs, dispatchDelay, result.ok]);
    stats.wireBytes += result.wireBytes; stats.decodedBytes += result.decodedBytes; latencies.push(result.latencyMs);
    const row = routeStats[route] ||= { requests: 0, successes: 0, failures: 0, wireBytes: 0, latencies: [], cache: {}, etags: new Set() };
    const stageRow = stageStats[stage] ||= { requests: 0, successes: 0, failures: 0, latencies: [], dispatchDelays: [] };
    row.requests++; row.wireBytes += result.wireBytes; row.latencies.push(result.latencyMs); stageRow.requests++; stageRow.latencies.push(result.latencyMs);
    stageRow.dispatchDelays.push(dispatchDelay);
    increment(cache, result.cache); increment(row.cache, result.cache); increment(pops, result.pop);
    if (result.etag && row.etags.size < 20) row.etags.add(result.etag);
    if (result.ok) {
      stats.successes++; row.successes++; stageRow.successes++;
      if (!user.ready) { user.ready = true; stats.readyUsers++; }
    } else { stats.failures++; row.failures++; stageRow.failures++; increment(failures, result.failure); }
    if ([403, 429].includes(result.status) || result.failure === "cloudflare_challenge") {
      abort("Access block, rate limit or Cloudflare challenge received; no further requests will be started.");
    }
    recent.push(result.ok); if (recent.length > 100) recent.shift();
    user.iteration++; user.nextAt = now + (user.ready ? cfg.thinkSeconds * 1000 : Math.min(1000, cfg.thinkSeconds * 1000));
    if (recent.length >= 50 && recent.filter((success) => !success).length / recent.length >= .1) abort("At least 10% failures in the latest 50–100 completed requests.");
    if (stats.wireBytes > cfg.maxWireBytes) abort("Transfer budget reached.");
  }
  const cancel = () => { abort("Cancelled by the operator."); for (const slot of sessions) slot.session?.destroy(); };
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  await new Promise((resolve) => {
    const tick = () => {
      const now = performance.now();
      stats.peakRssBytes = Math.max(stats.peakRssBytes, process.memoryUsage().rss);
      tokens = Math.min(cfg.maxInflight, tokens + (now - lastTick) * cfg.maxRps / 1000); lastTick = now;
      if (!stopping) {
        openConnections(now);
        const pending = [];
        for (const user of users) {
          if (!user.activated && now >= user.nextAt) { user.activated = true; stats.activatedUsers++; }
          if (user.activated && !user.busy && now >= user.nextAt) pending.push(user);
        }
        stats.peakActiveUsers = Math.max(stats.peakActiveUsers, stats.activatedUsers);
        stats.peakQueuedUsers = Math.max(stats.peakQueuedUsers, pending.length);
        if (stats.activatedUsers === cfg.vus && stats.targetStartedAtMs === null) stats.targetStartedAtMs = now - start;
        if (stats.readyUsers === cfg.vus && holdStart === null) {
          holdStart = now; stats.readyAtMs = now - start; stats.holdStartedAtMs = now - start;
        }
        if (holdStart === null && now >= readyDeadline) abort("Not all virtual users completed a verified initial load within the readiness deadline.");
        if (holdStart !== null && now - holdStart >= cfg.holdSeconds * 1000) {
          stopping = true; stoppedAt = now; stats.holdFinishedAtMs = now - start;
        }
        if (!stopping) {
          pending.sort((a, b) => a.nextAt - b.nextAt);
          for (const user of pending) {
            if (stats.inflight >= cfg.maxInflight || tokens < 1) break;
            if (stats.requests >= cfg.maxRequests) { abort("Request budget reached."); break; }
            const available = sessions.filter((slot) => slot.ready && !slot.session.destroyed && !slot.session.closed && slot.active < slot.maxStreams).sort((a, b) => a.active - b.active);
            if (!available.length) break;
            const slot = available[0], route = ROUTES[(user.id + user.iteration) % ROUTES.length];
            const dispatchedAt = performance.now(), dispatchDelay = dispatchedAt - user.nextAt, stage = phase(dispatchedAt);
            tokens--; user.busy = true; slot.active++; stats.requests++; stats.inflight++;
            stats.peakInflight = Math.max(stats.peakInflight, stats.inflight); dispatchDelays.push(dispatchDelay);
            requestDocument(slot.session, route, cfg.timeoutSeconds * 1000, cfg.expectedHashes[route]).then((result) => record(user, route, stage, slot, dispatchedAt, dispatchDelay, result));
          }
        }
      }
      if (now - lastProgress >= 10000) {
        onProgress({ elapsedSeconds: +((now - start) / 1000).toFixed(1), phase: stopping ? "draining" : phase(now),
          activatedUsers: stats.activatedUsers, verifiedUsers: stats.readyUsers, inflight: stats.inflight,
          completed: stats.completed, failures: stats.failures, wireMiB: +(stats.wireBytes / 1024 ** 2).toFixed(1) });
        lastProgress = now;
      }
      if (stopping && stats.inflight === 0) { clearInterval(interval); resolve(); }
    };
    const interval = setInterval(tick, 10); tick();
  });
  for (const slot of sessions) slot.session?.destroy();
  signal?.removeEventListener("abort", cancel);
  loop.disable();
  const durationMs = performance.now() - start;
  const responseMs = quantiles(latencies), dispatchMs = quantiles(dispatchDelays);
  const eventLoopP95Ms = +(loop.percentile(95) / 1e6).toFixed(3);
  const fullTargetHoldSeconds = stats.holdStartedAtMs !== null && stats.holdFinishedAtMs !== null ? (stats.holdFinishedAtMs - stats.holdStartedAtMs) / 1000 : 0;
  const failureRate = stats.completed ? stats.failures / stats.completed : 1;
  const stages = Object.fromEntries(Object.entries(stageStats).map(([stage, value]) => [stage, {
    ...value, latencies: undefined, dispatchDelays: undefined,
    responseMs: quantiles(value.latencies), dispatchMs: quantiles(value.dispatchDelays)
  }]));
  const checks = {
    noAbort: abortReason === null,
    allVirtualUsersVerified: stats.readyUsers === cfg.vus,
    fullTargetHoldCompleted: fullTargetHoldSeconds >= cfg.holdSeconds,
    failureRateWithinLimit: failureRate <= cfg.maxFailureRate,
    responseP95WithinLimit: responseMs.p95 !== null && responseMs.p95 <= cfg.maxP95Ms,
    dispatchP95WithinLimit: dispatchMs.p95 !== null && dispatchMs.p95 <= cfg.maxDispatchP95Ms,
    generatorEventLoopHealthy: eventLoopP95Ms <= cfg.maxEventLoopP95Ms,
    ...holdChecks(stages.hold, cfg)
  };
  return {
    schemaVersion: 2, startedAt, finishedAt: new Date().toISOString(), config: cfg,
    scenario: { routes: ROUTES, workload: "One HTML navigation per virtual user, followed by think time; routes rotate.",
      excluded: ["CSS/JavaScript/font subrequests", "real browser execution and interaction CPU", "independent devices/IP addresses", "multi-region delivery", "10,000 simultaneous in-flight HTTP requests"],
      readiness: "Full-target hold starts only after every VU completes a verified successful first page response.",
      responseVerification: "HTTP 200, HTML content type, streamed decompression, route-specific title, SHA-256 of the entire decoded response matching the known build, minimum bytes and matching Content-Length when supplied." },
    passed: Object.values(checks).every(Boolean), checks, abortReason,
    summary: { ...stats, durationSeconds: +(durationMs / 1000).toFixed(3),
      measuredTrafficSeconds: +(((stoppedAt || performance.now()) - start) / 1000).toFixed(3),
      fullTargetHoldSeconds: +fullTargetHoldSeconds.toFixed(3), failureRate,
      requestsPerSecond: +(stats.completed / (durationMs / 1000)).toFixed(3), responseMs, dispatchMs,
      eventLoopP95Ms, eventLoopMaxMs: +(loop.max / 1e6).toFixed(3) },
    failures, cache, cloudflarePops: pops,
    requestSampleFields: REQUEST_SAMPLE_FIELDS, requestSamples,
    routes: Object.fromEntries(Object.entries(routeStats).map(([route, value]) => [route, { ...value, latencies: undefined, etags: [...value.etags], responseMs: quantiles(value.latencies) }])),
    stages
  };
}

function parseArgs(argv) {
  const aliases = { "url": "url", "vus": "vus", "ramp-seconds": "rampSeconds", "hold-seconds": "holdSeconds",
    "think-seconds": "thinkSeconds", "max-inflight": "maxInflight", "max-rps": "maxRps", "connections": "connections",
    "timeout-seconds": "timeoutSeconds", "ready-grace-seconds": "readyGraceSeconds", "max-requests": "maxRequests" };
  const input = {}; let output = null, expectedDirectory = path.resolve(__dirname, "../dist");
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index]; if (!flag.startsWith("--") || argv[index + 1] === undefined) throw new Error("Expected --option value pairs.");
    const name = flag.slice(2), value = argv[index + 1];
    if (name === "output") output = value;
    else if (name === "expected-directory") expectedDirectory = path.resolve(value);
    else if (Object.hasOwn(aliases, name)) input[aliases[name]] = name === "url" ? value : Number(value);
    else throw new Error("Unknown option: " + flag);
  }
  return { config: options(input), output, expectedDirectory };
}

module.exports = { DEFAULTS, ROUTES, REQUEST_SAMPLE_FIELDS, options, quantiles, verifyHeaders, verifyTitle, loadExpectedHashes, holdChecks, requestDocument, runLoadTest, parseArgs };
if (require.main === module) {
  (async () => {
    const { config, output, expectedDirectory } = parseArgs(process.argv.slice(2));
    config.expectedHashes = loadExpectedHashes(expectedDirectory);
    console.error("Document-navigation test only; this does not launch real browsers. Target:", config.url);
    const controller = new AbortController();
    const stop = () => controller.abort(); process.once("SIGINT", stop); process.once("SIGTERM", stop);
    const report = await runLoadTest(config, (progress) => console.error(JSON.stringify(progress)), controller.signal);
    process.removeListener("SIGINT", stop); process.removeListener("SIGTERM", stop);
    const json = JSON.stringify(report, null, 2) + "\n";
    if (output) fs.writeFileSync(output, json); else process.stdout.write(json);
    console.error(JSON.stringify({ passed: report.passed, checks: report.checks, summary: report.summary, abortReason: report.abortReason, output }));
    if (!report.passed) process.exitCode = 1;
  })().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
