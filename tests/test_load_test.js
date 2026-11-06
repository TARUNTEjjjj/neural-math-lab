"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const http2 = require("node:http2");
const zlib = require("node:zlib");
const crypto = require("node:crypto");
const { options, parseArgs, quantiles, requestDocument, runLoadTest, holdChecks, ROUTES, REQUEST_SAMPLE_FIELDS } = require("../scripts/load-test.cjs");

const html = Buffer.from('<!doctype html><html><head><title>Neural Math Lab · Understand the learning</title></head><body>' + 'Learning. '.repeat(200) + '</body></html>');
const routeHtml = (route) => Buffer.from(html.toString().replace("Understand the learning", {
  "/labs/moons/": "Moons experiment", "/labs/circles/": "Circles experiment",
  "/learn/optimization-mathematics/": "Gradients &amp; optimization"
}[route] || "Understand the learning"));
const expectedHashes = Object.fromEntries(ROUTES.map((route) => [route, crypto.createHash("sha256").update(routeHtml(route)).digest("hex")]));
async function fixture(handler) {
  const server = http2.createServer(); const sessions = new Set();
  server.on("session", (session) => { sessions.add(session); session.on("error", () => {}); session.on("close", () => sessions.delete(session)); });
  server.on("stream", (stream, headers) => { stream.on("error", () => {}); handler(stream, headers); });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: "http://127.0.0.1:" + server.address().port,
    async close() { for (const session of sessions) session.destroy(); await new Promise((resolve) => server.close(resolve)); } };
}
const respond = (stream, body = html, extra = {}) => {
  stream.respond({ ":status": 200, "content-type": "text/html; charset=utf-8", "content-length": body.length,
    "cf-cache-status": "HIT", "cf-ray": "local-test-DEL", "etag": '"fixture"', ...extra }); stream.end(body);
};

test("capacity harness rejects unrelated hosts, unsafe origins, and unbounded configurations", () => {
  for (const url of ["https://example.com", "https://neural-math-lab.pages.dev/anything", "http://neural-math-lab.pages.dev", "https://a:b@neural-math-lab.pages.dev", "https://neural-math-lab.pages.dev/?x=1"]) {
    assert.throws(() => options({ url }), /Target/);
  }
  assert.equal(options({ url: "http://127.0.0.1:8099" }).url, "http://127.0.0.1:8099");
  for (const invalid of [{ vus: 10001 }, { vus: 1.5 }, { maxInflight: 10000 }, { maxRps: Infinity }, { connections: 30 }, { thinkSeconds: 0 }]) {
    assert.throws(() => options(invalid));
  }
  assert.deepEqual(parseArgs(["--vus", "10000", "--ramp-seconds", "120", "--hold-seconds", "60", "--output", "result.json"]).output, "result.json");
  assert.throws(() => parseArgs(["--unknown", "5"]), /Unknown/);
});

test("quantiles use all completed samples and leave their input untouched", () => {
  const samples = Array.from({ length: 100 }, (_, i) => 100 - i);
  assert.deepEqual(quantiles(samples), { count: 100, p50: 50, p95: 95, p99: 99, max: 100 });
  assert.equal(samples[0], 100); assert.equal(quantiles([]).p95, null);
});

test("response verification streams Brotli and gzip, checks title and rejects successful error pages", async () => {
  const server = await fixture((stream, headers) => {
    const route = headers[":path"];
    if (route === "/br") return respond(stream, zlib.brotliCompressSync(html), { "content-encoding": "br" });
    if (route === "/gzip") return respond(stream, zlib.gzipSync(html), { "content-encoding": "gzip" });
    if (route === "/wrong") return respond(stream, Buffer.from(html.toString().replace("Neural Math Lab", "Upstream failure")));
    if (route === "/error") return respond(stream, html, { ":status": 503 });
    if (route === "/not-html") return respond(stream, html, { "content-type": "application/json" });
    respond(stream);
  });
  const session = http2.connect(server.url);
  try {
    for (const route of ["/", "/br", "/gzip"]) {
      const result = await requestDocument(session, route, 1000);
      assert.equal(result.ok, true, route); assert.equal(result.decodedBytes, html.length);
      assert.ok(result.wireBytes > 32); assert.equal(result.cache, "HIT"); assert.equal(result.pop, "test-DEL");
    }
    assert.equal((await requestDocument(session, "/wrong", 1000)).failure, "wrong_page_title");
    assert.equal((await requestDocument(session, "/error", 1000)).failure, "http_503");
    assert.equal((await requestDocument(session, "/not-html", 1000)).failure, "not_html");
  } finally { session.destroy(); await server.close(); }
});

test("capacity run counts verified users, sustains the full target, and respects in-flight bounds", async () => {
  let inflight = 0, peak = 0; const observed = new Set();
  const server = await fixture((stream, headers) => {
    observed.add(headers[":path"]); inflight++; peak = Math.max(peak, inflight);
    setImmediate(() => { respond(stream, routeHtml(headers[":path"])); inflight--; });
  });
  try {
    const report = await runLoadTest({ expectedHashes, url: server.url, vus: 8, rampSeconds: .05, holdSeconds: .12, thinkSeconds: .03,
      maxInflight: 2, maxRps: 500, connections: 1, readyGraceSeconds: 3, maxEventLoopP95Ms: 1000 });
    assert.equal(report.passed, true, JSON.stringify(report.checks));
    assert.equal(report.summary.readyUsers, 8); assert.equal(report.summary.peakActiveUsers, 8);
    assert.ok(report.summary.fullTargetHoldSeconds >= .12); assert.ok(report.summary.peakInflight <= 2); assert.ok(peak <= 2);
    assert.equal(report.summary.failures, 0); assert.equal(report.summary.completed, report.summary.requests);
    assert.ok(report.summary.completed > 8); assert.deepEqual([...observed].sort(), [...ROUTES].sort());
    assert.ok(report.stages.hold.successes > 0); assert.equal(report.scenario.excluded.includes("real browser execution and interaction CPU"), true);
    assert.ok(report.stages.hold.dispatchMs.count > 0); assert.deepEqual(report.config.expectedHashes, expectedHashes);
    assert.deepEqual(report.requestSampleFields, REQUEST_SAMPLE_FIELDS);
    assert.equal(report.requestSamples.length, report.summary.completed);
    assert.ok(report.requestSamples.length <= report.config.maxRequests);
    for (const [dispatch, completion, latency, queue, success] of report.requestSamples) {
      assert.ok(dispatch >= 0 && completion >= dispatch && latency <= completion - dispatch + .01);
      assert.ok(queue >= 0 && queue <= dispatch); assert.equal(success, true);
    }
    assert.deepEqual(quantiles(report.requestSamples.map((row) => row[2])), report.summary.responseMs);
    assert.deepEqual(quantiles(report.requestSamples.map((row) => row[3])), report.summary.dispatchMs);
    const holdSamples = report.requestSamples.filter((row) => row[0] >= report.summary.holdStartedAtMs && row[0] < report.summary.holdFinishedAtMs);
    assert.equal(holdSamples.length, report.stages.hold.requests);
    assert.deepEqual(quantiles(holdSamples.map((row) => row[2])), report.stages.hold.responseMs);
  } finally { await server.close(); }
});

test("scheduled users without successful responses cannot pass as 10k-style verified capacity", async () => {
  const server = await fixture((stream) => respond(stream, Buffer.from(html.toString().replace("Neural Math Lab", "Not the learning site"))));
  try {
    const report = await runLoadTest({ expectedHashes, url: server.url, vus: 4, rampSeconds: 0, holdSeconds: .05, thinkSeconds: .03,
      maxInflight: 2, maxRps: 500, connections: 1, readyGraceSeconds: .15, maxEventLoopP95Ms: 1000 });
    assert.equal(report.passed, false); assert.equal(report.summary.activatedUsers, 4); assert.equal(report.summary.readyUsers, 0);
    assert.equal(report.summary.fullTargetHoldSeconds, 0); assert.equal(report.checks.allVirtualUsersVerified, false);
    assert.ok(report.failures.wrong_page_title > 0); assert.match(report.abortReason, /verified initial load/);
    assert.equal(report.requestSamples.filter((row) => row[4] === false).length, report.summary.failures);
  } finally { await server.close(); }
});

test("stalled streams time out instead of leaving the capacity run hanging", async () => {
  const server = await fixture(() => {}); const session = http2.connect(server.url);
  try { assert.equal((await requestDocument(session, "/", 50)).failure, "timeout"); }
  finally { session.destroy(); await server.close(); }
});

test("branded wrong pages fail route validation and rate limits stop dispatch immediately", async () => {
  const server = await fixture((stream) => respond(stream)); const session = http2.connect(server.url);
  try { assert.equal((await requestDocument(session, "/labs/moons/", 1000)).failure, "wrong_page_title"); }
  finally { session.destroy(); await server.close(); }
  let seen = 0;
  const blocked = await fixture((stream) => { seen++; respond(stream, html, { ":status": 429 }); });
  try {
    const report = await runLoadTest({ expectedHashes, url: blocked.url, vus: 20, rampSeconds: 0, holdSeconds: 1,
      maxInflight: 1, connections: 1, maxRps: 100, readyGraceSeconds: 2 });
    assert.equal(report.passed, false); assert.equal(seen, 1); assert.equal(report.summary.requests, 1);
    assert.match(report.abortReason, /rate limit/);
  } finally { await blocked.close(); }
});

test("operator cancellation yields a failed partial report and closes pending streams", async () => {
  const server = await fixture(() => {}); const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 50);
  try {
    const report = await runLoadTest({ expectedHashes, url: server.url, vus: 5, rampSeconds: 0, holdSeconds: 1 }, () => {}, controller.signal);
    assert.equal(report.passed, false); assert.match(report.abortReason, /Cancelled/);
    assert.equal(report.summary.inflight, 0);
    assert.equal(report.requestSamples.length, report.summary.completed);
  } finally { clearTimeout(timer); await server.close(); }
});

test("full-body hashes reject a correct-title truncated page despite consistent Content-Length", async () => {
  const server = await fixture((stream) => respond(stream, html.subarray(0, 250)));
  const session = http2.connect(server.url);
  try {
    const result = await requestDocument(session, "/", 1000, expectedHashes["/"]);
    assert.equal(result.failure, "body_hash_mismatch");
    await assert.rejects(() => runLoadTest({ url: server.url }), /expected body hash/);
  } finally { session.destroy(); await server.close(); }
});

test("degraded or idle full-target hold cannot be hidden by healthy ramp results", () => {
  const cfg = options();
  const healthy = { requests: 1000, failures: 0, responseMs: { p95: 100 }, dispatchMs: { p95: 20 } };
  assert.equal(Object.values(holdChecks(healthy, cfg)).every(Boolean), true);
  for (const bad of [undefined,
    { ...healthy, requests: 0 },
    { ...healthy, failures: 11 },
    { ...healthy, responseMs: { p95: 2001 } },
    { ...healthy, dispatchMs: { p95: 1001 } }]) {
    assert.equal(Object.values(holdChecks(bad, cfg)).every(Boolean), false);
  }
});
