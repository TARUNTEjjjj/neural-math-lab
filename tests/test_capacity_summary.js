"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { summarizeCapacity } = require("../scripts/summarize-capacity.cjs");
const { ROUTES, REQUEST_SAMPLE_FIELDS, quantiles } = require("../scripts/load-test.cjs");
const hashes = Object.fromEntries(ROUTES.map((route) => [route, "a".repeat(64)]));

function shard(offsetSeconds = 0, responseP95 = 200) {
  const requestSamples = Array.from({ length: 15000 }, (_, index) => {
    const dispatch = index < 5000 ? 50 + index * 23.98 : 120000 + (index - 5000) * 11.99;
    return [dispatch, dispatch + responseP95 + .1, responseP95, 25, true];
  });
  return {
    schemaVersion: 2, passed: true, abortReason: null,
    startedAt: new Date(Date.UTC(2026, 10, 5, 10) + offsetSeconds * 1000).toISOString(),
    config: { url: "https://neural-math-lab.pages.dev", vus: 5000, thinkSeconds: 60, maxRequests: 200000, expectedHashes: { ...hashes } },
    scenario: { routes: [...ROUTES] }, checks: Object.fromEntries(["noAbort", "allVirtualUsersVerified", "fullTargetHoldCompleted",
      "failureRateWithinLimit", "responseP95WithinLimit", "dispatchP95WithinLimit", "generatorEventLoopHealthy", "holdHadNavigationRequests",
      "holdFailureRateWithinLimit", "holdResponseP95WithinLimit", "holdDispatchP95WithinLimit"].map((key) => [key, true])),
    summary: { readyUsers: 5000, activatedUsers: 5000, requests: 15000, completed: 15000,
      successes: 15000, failures: 0, wireBytes: 500000, decodedBytes: 2000000, peakInflight: 40,
      holdStartedAtMs: 120000, holdFinishedAtMs: 240000, fullTargetHoldSeconds: 120, durationSeconds: 245,
      responseMs: { p95: responseP95 }, dispatchMs: { p95: 25 }, eventLoopP95Ms: 21,
      peakRssBytes: 80000000, requestsPerSecond: 62.5 },
    stages: { hold: { requests: 10000, successes: 10000, failures: 0, responseMs: { p95: responseP95 }, dispatchMs: { p95: 25 } } },
    requestSampleFields: [...REQUEST_SAMPLE_FIELDS], requestSamples,
    cloudflarePops: { LHR: 15000 }, cache: { HIT: 15000 }
  };
}

test("two verified 5k shards require a real shared hold and pool individual overlap samples", () => {
  const result = summarizeCapacity([shard(0, 100), shard(10, 400)], { expectedHashes: hashes });
  assert.equal(result.passed, true, JSON.stringify(result.validationErrors));
  assert.equal(result.totalVerifiedUsers, 10000); assert.equal(result.overlapSeconds, 110);
  assert.equal(result.overlapStart, "2026-11-05T10:02:10.000Z");
  assert.equal(result.overlapEnd, "2026-11-05T10:04:00.000Z");
  assert.equal(result.worstShardResponseP95Ms, 400); assert.equal(result.worstShardHoldP95Ms, 400);
  assert.equal(result.wholeRunTotals.requests, 30000); assert.equal(result.errors, 0);
  assert.equal(result.cloudflarePops.LHR, 30000);
  assert.equal(Object.hasOwn(result, "globalP95"), false);
  assert.equal(result.overlap.responseMs.p95, 400); assert.equal(result.overlap.dispatchMs.p95, 25);
  assert.ok(result.overlap.requests < result.wholeRunTotals.requests);
  assert.equal(result.overlap.requests, result.perRun.reduce((count, run) => count + run.overlap.requests, 0));
  assert.equal(Object.values(result.overlapChecks).every(Boolean), true);
});

test("successful runs without 65 seconds of overlap cannot establish simultaneous 10k capacity", () => {
  for (const offset of [56, 200]) {
    const result = summarizeCapacity([shard(), shard(offset)], { expectedHashes: hashes });
    assert.equal(result.passed, false); assert.ok(result.validationErrors.some((text) => /overlap/.test(text)));
  }
  assert.equal(summarizeCapacity([shard(), shard(55)], { expectedHashes: hashes }).passed, true);
});

test("a failed shard or unverified users cannot be hidden by the other shard", () => {
  const failed = shard(); failed.passed = false; failed.checks.holdHadNavigationRequests = false;
  assert.equal(summarizeCapacity([shard(), failed], { expectedHashes: hashes }).passed, false);
  const incomplete = shard(); incomplete.summary.readyUsers = 4999;
  const result = summarizeCapacity([shard(), incomplete], { expectedHashes: hashes });
  assert.equal(result.passed, false); assert.equal(result.totalVerifiedUsers, 9999);
});

test("mismatched report hashes and a stale current build both fail source verification", () => {
  const stale = shard(); stale.config.expectedHashes["/"] = "b".repeat(64);
  assert.equal(summarizeCapacity([shard(), stale], { expectedHashes: hashes }).passed, false);
  const newBuildHashes = { ...hashes, "/": "b".repeat(64) };
  assert.equal(summarizeCapacity([shard(), shard()], { expectedHashes: newBuildHashes }).passed, false);
  assert.equal(summarizeCapacity([shard(), shard()]).passed, false);
});

test("aggregation rejects different workloads, invalid windows and degraded hold results independently", () => {
  const mutations = [
    (r) => { r.config.thinkSeconds = 30; },
    (r) => { r.config.url = "https://example.com"; },
    (r) => { r.scenario.routes.pop(); },
    (r) => { r.schemaVersion = 1; },
    (r) => { r.summary.holdFinishedAtMs = r.summary.holdStartedAtMs; },
    (r) => { r.summary.fullTargetHoldSeconds = 119; },
    (r) => { r.stages.hold.responseMs.p95 = 2001; },
    (r) => { r.stages.hold.dispatchMs.p95 = 1001; },
    (r) => { r.stages.hold.failures = 101; r.stages.hold.successes -= 101; },
    (r) => { r.stages.hold.requests = 0; r.stages.hold.successes = 0; }
  ];
  for (const mutate of mutations) {
    const report = shard(); mutate(report);
    assert.equal(summarizeCapacity([shard(), report], { expectedHashes: hashes }).passed, false, mutate.toString());
  }
  assert.equal(summarizeCapacity([shard()], { expectedHashes: hashes }).passed, false);
});

test("overlap-only latency or dispatch degradation cannot be diluted by healthy traffic outside it", () => {
  for (const field of [2, 3]) {
    const reports = [shard(), shard(55)];
    // The common hold is 175000–240000 ms from shard 1's start. Degrade
    // 4.5% of each full hold but >5% of the actual common-window cohort.
    for (const [index, report] of reports.entries()) {
      const cohort = report.requestSamples.filter((sample) => sample[0] + index * 55000 >= 175000 && sample[0] + index * 55000 < 240000);
      for (const sample of cohort.slice(0, 450)) {
        sample[field] = field === 2 ? 2500 : 1200;
        if (field === 2) sample[1] = sample[0] + sample[2] + .1;
      }
      assert.equal(quantiles(report.requestSamples.filter((sample) => sample[0] >= 120000).map((sample) => sample[field])).p95,
        field === 2 ? 200 : 25);
    }
    const result = summarizeCapacity(reports, { expectedHashes: hashes });
    assert.equal(result.passed, false);
    assert.equal(result.overlapChecks[field === 2 ? "responseP95WithinLimit" : "dispatchP95WithinLimit"], false);
    assert.ok(result.validationErrors.every((message) => message.startsWith("Shared-overlap SLO failed:")), JSON.stringify(result.validationErrors));
  }
});

test("overlap-only failures cannot be diluted by healthy traffic outside it", () => {
  const reports = [shard(), shard(55)];
  for (const [index, report] of reports.entries()) {
    const cohort = report.requestSamples.filter((sample) => sample[0] + index * 55000 >= 175000 && sample[0] + index * 55000 < 240000);
    for (const sample of cohort.slice(0, 100)) sample[4] = false;
    report.summary.failures = 100; report.summary.successes -= 100;
    report.stages.hold.failures = 100; report.stages.hold.successes -= 100;
  }
  const result = summarizeCapacity(reports, { expectedHashes: hashes });
  assert.equal(result.passed, false); assert.equal(result.overlapChecks.failureRateWithinLimit, false);
  assert.ok(result.overlap.failureRate > .01);
  assert.ok(result.validationErrors.every((message) => message.startsWith("Shared-overlap SLO failed:")), JSON.stringify(result.validationErrors));
});

test("overlap cohort includes full latency of responses completing after the overlap ends", () => {
  const reports = [shard(), shard(55)];
  const sample = reports[1].requestSamples.find((row) => row[0] > 184800 && row[0] < 185000);
  assert.ok(sample); sample[1] = sample[0] + 4000; sample[2] = 4000;
  const result = summarizeCapacity(reports, { expectedHashes: hashes });
  assert.equal(result.passed, true, JSON.stringify(result.validationErrors));
  assert.equal(result.overlap.responseMs.max, 4000);
});

test("missing, incomplete, malformed or inconsistent timing samples cannot establish capacity", () => {
  const mutations = [
    (r) => { delete r.requestSamples; },
    (r) => { r.requestSampleFields.reverse(); },
    (r) => { r.requestSamples.pop(); },
    (r) => { r.requestSamples[0][4] = 1; },
    (r) => { r.requestSamples[0][2] = Infinity; },
    (r) => { r.requestSamples[0][1] = r.requestSamples[0][0] - 1; },
    (r) => { r.requestSamples[0][2] = 5000; },
    (r) => { r.requestSamples[0][3] = 5000; },
    (r) => { r.requestSamples[0][4] = false; },
    (r) => { r.requestSamples[5000][0] = 119999; },
    (r) => { r.requestSamples[0][1] = 250000; },
    (r) => { r.summary.durationSeconds = 239; },
    (r) => { r.summary.responseMs.p95 = 50; },
    (r) => { r.stages.hold.dispatchMs.p95 = 50; },
    (r) => { r.config.maxRequests = 200001; },
    (r) => { r.checks = {}; }
  ];
  for (const mutate of mutations) {
    const report = shard(); mutate(report);
    assert.equal(summarizeCapacity([shard(), report], { expectedHashes: hashes }).passed, false, mutate.toString());
  }
});
