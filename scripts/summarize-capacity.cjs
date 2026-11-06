"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { ROUTES, REQUEST_SAMPLE_FIELDS, loadExpectedHashes, quantiles } = require("./load-test.cjs");
const TARGET = "https://neural-math-lab.pages.dev";
const MINIMUM_OVERLAP_SECONDS = 65;
const nonnegative = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const integer = (value) => nonnegative(value) && Number.isInteger(value);
const REQUIRED_CHECKS = ["noAbort", "allVirtualUsersVerified", "fullTargetHoldCompleted", "failureRateWithinLimit",
  "responseP95WithinLimit", "dispatchP95WithinLimit", "generatorEventLoopHealthy", "holdHadNavigationRequests",
  "holdFailureRateWithinLimit", "holdResponseP95WithinLimit", "holdDispatchP95WithinLimit"];

function sampleMetrics(samples) {
  const failures = samples.filter((sample) => !sample[4]).length;
  return { requests: samples.length, successes: samples.length - failures, failures,
    failureRate: samples.length ? failures / samples.length : null,
    responseMs: quantiles(samples.map((sample) => sample[2])),
    dispatchMs: quantiles(samples.map((sample) => sample[3])) };
}

function validateSamples(report, bad) {
  const samples = report?.requestSamples, summary = report?.summary || {}, cfg = report?.config || {};
  if (JSON.stringify(report?.requestSampleFields) !== JSON.stringify(REQUEST_SAMPLE_FIELDS) ||
      !Array.isArray(samples) || !integer(cfg.maxRequests) || cfg.maxRequests > 200000 ||
      samples.length > cfg.maxRequests || samples.length !== summary.completed || samples.length === 0) {
    bad("Missing, incomplete, unbounded or incompatible per-request timing samples."); return null;
  }
  const durationMs = summary.durationSeconds * 1000;
  if (!nonnegative(durationMs) || !nonnegative(summary.holdFinishedAtMs) || summary.holdFinishedAtMs > durationMs + 1) {
    bad("Invalid run duration or hold outside the run."); return null;
  }
  // Requests are selected by dispatch time, including their complete response if
  // it finishes after a cohort's end. A late slow response must not disappear.
  if (samples.some((sample) => !Array.isArray(sample) || sample.length !== REQUEST_SAMPLE_FIELDS.length ||
      !sample.slice(0, 4).every(nonnegative) || typeof sample[4] !== "boolean" ||
      sample[1] < sample[0] || sample[1] > durationMs + 1 || sample[0] >= summary.holdFinishedAtMs ||
      sample[2] > sample[1] - sample[0] + .01 || sample[3] > sample[0] + .01)) {
    bad("Invalid request sample timing, success flag or row shape."); return null;
  }
  const checkMetrics = (actual, stated, label) => {
    for (const key of ["requests", "successes", "failures"]) {
      if (actual[key] !== stated?.[key]) bad(`${label} ${key} does not match complete request samples.`);
    }
    for (const key of ["responseMs", "dispatchMs"]) {
      if (!nonnegative(stated?.[key]?.p95) || Math.abs(actual[key].p95 - stated[key].p95) > .001) {
        bad(`${label} ${key} p95 does not match complete request samples.`);
      }
    }
  };
  checkMetrics(sampleMetrics(samples), summary, "Whole-run");
  checkMetrics(sampleMetrics(samples.filter((sample) => sample[0] >= summary.holdStartedAtMs &&
    sample[0] < summary.holdFinishedAtMs)), report?.stages?.hold, "Full-target hold");
  return samples;
}

function summarizeCapacity(reports, { expectedHashes, labels = ["shard-1", "shard-2"] } = {}) {
  const validationErrors = [];
  const fail = (text) => validationErrors.push(text);
  if (!Array.isArray(reports) || reports.length !== 2) {
    return { schemaVersion: 2, passed: false, validationErrors: ["Exactly two shard reports are required."] };
  }
  const validHashes = (hashes) => hashes && Object.keys(hashes).length === ROUTES.length &&
    ROUTES.every((route) => /^[a-f0-9]{64}$/.test(hashes[route] || ""));
  if (!validHashes(expectedHashes)) fail("Expected hashes for all four routes must come from the current known build.");
  const validatedSamples = [];
  const perRun = reports.map((report, index) => {
    const label = labels[index] || `shard-${index + 1}`, cfg = report?.config || {}, summary = report?.summary || {}, hold = report?.stages?.hold;
    const bad = (message) => fail(`${label}: ${message}`);
    if (report?.schemaVersion !== 2) bad("Expected capacity report schema 2.");
    if (report?.passed !== true || !report.checks || REQUIRED_CHECKS.some((name) => report.checks[name] !== true) ||
        Object.values(report.checks).some((value) => value !== true) || report.abortReason !== null) bad("Shard did not pass all of its own checks.");
    if (cfg.url !== TARGET) bad("Wrong target origin.");
    if (JSON.stringify(report?.scenario?.routes) !== JSON.stringify(ROUTES)) bad("Wrong document workload routes or order.");
    if (cfg.thinkSeconds !== 60) bad("Expected the same 60-second think-time workload.");
    if (cfg.vus !== 5000 || summary.readyUsers !== 5000 || summary.activatedUsers !== 5000) bad("Shard must contain exactly 5,000 successfully initialized virtual users.");
    if (!validHashes(cfg.expectedHashes) || !validHashes(expectedHashes) || ROUTES.some((route) => cfg.expectedHashes?.[route] !== expectedHashes?.[route])) bad("Response hashes do not match the current known build.");
    for (const key of ["requests", "completed", "successes", "failures", "wireBytes", "decodedBytes", "peakInflight"]) {
      if (!integer(summary[key])) bad(`Invalid ${key} count.`);
    }
    if (summary.completed !== summary.requests || summary.successes + summary.failures !== summary.completed || summary.successes < 5000) bad("Completed response counts are inconsistent or insufficient.");
    if (!(summary.completed > 0) || summary.failures / summary.completed > .01) bad("Whole-run failure rate exceeds 1%.");
    for (const [metric, maximum] of [["responseMs", 2000], ["dispatchMs", 1000]]) {
      if (!nonnegative(summary[metric]?.p95) || summary[metric].p95 > maximum) bad(`Whole-run ${metric} p95 exceeds ${maximum} ms or is missing.`);
    }
    if (!nonnegative(summary.eventLoopP95Ms) || summary.eventLoopP95Ms > 100) bad("Generator event-loop p95 exceeds 100 ms or is missing.");
    if (!hold || !integer(hold.requests) || hold.requests <= 0 || !integer(hold.failures) || !integer(hold.successes) || hold.successes + hold.failures !== hold.requests) bad("No valid full-target hold navigation results.");
    if (!hold || hold.failures / hold.requests > .01) bad("Full-target hold failure rate exceeds 1%.");
    if (!nonnegative(hold?.responseMs?.p95) || hold.responseMs.p95 > 2000) bad("Full-target hold response p95 exceeds 2 seconds or is missing.");
    if (!nonnegative(hold?.dispatchMs?.p95) || hold.dispatchMs.p95 > 1000) bad("Full-target hold dispatch p95 exceeds 1 second or is missing.");
    const runStart = Date.parse(report?.startedAt), holdStart = summary.holdStartedAtMs, holdEnd = summary.holdFinishedAtMs;
    const validWindow = Number.isFinite(runStart) && nonnegative(holdStart) && nonnegative(holdEnd) && holdEnd > holdStart &&
      holdEnd <= 24 * 60 * 60 * 1000 && Math.abs(runStart + holdEnd) <= 8.64e15;
    const holdSeconds = validWindow ? (holdEnd - holdStart) / 1000 : null;
    if (!validWindow || holdSeconds < MINIMUM_OVERLAP_SECONDS || !nonnegative(summary.fullTargetHoldSeconds) || Math.abs(summary.fullTargetHoldSeconds - holdSeconds) > .01) bad("Invalid or inconsistent full-target hold window.");
    validatedSamples.push(validateSamples(report, bad));
    return {
      label, passed: report?.passed === true, verifiedUsers: integer(summary.readyUsers) ? summary.readyUsers : 0,
      startedAt: report?.startedAt || null,
      fullTargetHoldStart: validWindow ? new Date(runStart + holdStart).toISOString() : null,
      fullTargetHoldEnd: validWindow ? new Date(runStart + holdEnd).toISOString() : null,
      holdStartEpochMs: validWindow ? runStart + holdStart : null,
      holdEndEpochMs: validWindow ? runStart + holdEnd : null,
      fullTargetHoldSeconds: holdSeconds,
      responseP95Ms: summary.responseMs?.p95 ?? null, holdResponseP95Ms: hold?.responseMs?.p95 ?? null,
      dispatchP95Ms: summary.dispatchMs?.p95 ?? null, holdDispatchP95Ms: hold?.dispatchMs?.p95 ?? null,
      peakHTTPInflight: summary.peakInflight ?? null, requestsPerSecond: summary.requestsPerSecond ?? null,
      failures: summary.failures ?? null, holdFailures: hold?.failures ?? null, holdRequests: hold?.requests ?? null,
      eventLoopP95Ms: summary.eventLoopP95Ms ?? null, peakRssBytes: summary.peakRssBytes ?? null,
      cloudflarePops: report?.cloudflarePops || {}, cache: report?.cache || {}
    };
  });
  const allWindowsValid = perRun.every((run) => run.holdStartEpochMs !== null && run.holdEndEpochMs !== null);
  const overlapStart = allWindowsValid ? Math.max(...perRun.map((run) => run.holdStartEpochMs)) : null;
  const overlapEnd = allWindowsValid ? Math.min(...perRun.map((run) => run.holdEndEpochMs)) : null;
  const overlapSeconds = allWindowsValid ? Math.max(0, (overlapEnd - overlapStart) / 1000) : 0;
  if (overlapSeconds < MINIMUM_OVERLAP_SECONDS) fail("The two verified full-target hold windows overlap for less than 65 seconds.");
  const totalVerifiedUsers = perRun.reduce((total, run) => total + run.verifiedUsers, 0);
  if (totalVerifiedUsers !== 10000) fail("The two shards do not total 10,000 verified virtual users.");
  const overlapSamples = [];
  for (let index = 0; index < reports.length; index++) {
    const runStart = Date.parse(reports[index]?.startedAt);
    const cohort = allWindowsValid && validatedSamples[index] ? validatedSamples[index].filter((sample) =>
      runStart + sample[0] >= overlapStart && runStart + sample[0] < overlapEnd) : [];
    perRun[index].overlap = sampleMetrics(cohort);
    if (!cohort.length) fail(`${perRun[index].label}: No valid navigation traffic dispatched inside the shared hold window.`);
    for (const sample of cohort) overlapSamples.push(sample);
  }
  const overlap = sampleMetrics(overlapSamples);
  const overlapChecks = {
    bothShardsHadNavigationRequests: perRun.every((run) => run.overlap.requests > 0),
    failureRateWithinLimit: overlap.failureRate !== null && overlap.failureRate <= .01,
    responseP95WithinLimit: overlap.responseMs.p95 !== null && overlap.responseMs.p95 <= 2000,
    dispatchP95WithinLimit: overlap.dispatchMs.p95 !== null && overlap.dispatchMs.p95 <= 1000
  };
  for (const [name, passed] of Object.entries(overlapChecks)) if (!passed) fail(`Shared-overlap SLO failed: ${name}.`);
  const totals = Object.fromEntries(["requests", "completed", "successes", "failures", "wireBytes", "decodedBytes"].map((key) => [key,
    reports.reduce((sum, report) => sum + (integer(report?.summary?.[key]) ? report.summary[key] : 0), 0)]));
  const pops = {};
  for (const report of reports) for (const [pop, count] of Object.entries(report?.cloudflarePops || {})) if (integer(count)) pops[pop] = (pops[pop] || 0) + count;
  const worst = (field) => perRun.every((run) => nonnegative(run[field])) ? Math.max(...perRun.map((run) => run[field])) : null;
  return {
    schemaVersion: 2, passed: validationErrors.length === 0, validationErrors,
    target: TARGET, expectedHashes, thinkSeconds: 60, shardCount: 2, totalVerifiedUsers,
    minimumOverlapSeconds: MINIMUM_OVERLAP_SECONDS,
    overlapStart: overlapStart !== null ? new Date(overlapStart).toISOString() : null,
    overlapEnd: overlapEnd !== null ? new Date(overlapEnd).toISOString() : null,
    overlapSeconds: +overlapSeconds.toFixed(3), errors: totals.failures,
    overlap, overlapChecks,
    wholeRunTotals: totals,
    worstShardResponseP95Ms: worst("responseP95Ms"), worstShardHoldP95Ms: worst("holdResponseP95Ms"),
    worstShardDispatchP95Ms: worst("dispatchP95Ms"), worstShardHoldDispatchP95Ms: worst("holdDispatchP95Ms"),
    cloudflarePops: pops, perRun,
    interpretation: [
      "The overlap uses each generator's reported UTC start and monotonic hold offsets; it assumes their wall clocks agree.",
      "Traffic totals include the entire two runs, including ramps and time outside the shared hold window.",
      "Overlap metrics pool every request dispatched at or after overlapStart and before overlapEnd, retaining complete latency and outcome even if the response finishes after overlapEnd.",
      "The pooled overlap response/dispatch percentiles and failure rate must independently meet the 2-second, 1-second and 1% limits; each shard must contribute traffic in that window.",
      "Worst-shard percentiles are maxima of shard percentiles, not a pooled/global p95; no percentile average is used.",
      "This measures two generators' HTTP document-navigation workload, not 10,000 real browsers or browser-side interactions."
    ]
  };
}

module.exports = { summarizeCapacity, MINIMUM_OVERLAP_SECONDS };
if (require.main === module) {
  try {
    const argv = process.argv.slice(2), files = [], flags = {};
    for (let index = 0; index < argv.length; index++) {
      if (!argv[index].startsWith("--")) files.push(argv[index]);
      else if (["--output", "--expected-directory"].includes(argv[index]) && argv[index + 1]) flags[argv[index]] = argv[++index];
      else throw new Error("Usage: node scripts/summarize-capacity.cjs shard1.json shard2.json [--expected-directory dist] [--output summary.json]");
    }
    if (files.length !== 2) throw new Error("Supply exactly two shard JSON reports.");
    const report = summarizeCapacity(files.map((file) => JSON.parse(fs.readFileSync(file, "utf8"))), {
      labels: files.map((file) => path.basename(file)),
      expectedHashes: loadExpectedHashes(flags["--expected-directory"] || path.resolve(__dirname, "../dist"))
    });
    const json = JSON.stringify(report, null, 2) + "\n";
    if (flags["--output"]) fs.writeFileSync(flags["--output"], json); else process.stdout.write(json);
    if (!report.passed) { console.error(report.validationErrors.join("\n")); process.exitCode = 1; }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
