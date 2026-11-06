# Performance measurements and capacity scope

Initial measurements on 5 November 2026; final verification on 7 November 2026 against the public
[Neural Math Lab website](https://neural-math-lab.pages.dev/). The source
repository remains private. Raw measurements are in `benchmarks/`.

## What changed

The gradient UI previously recalculated a numerical derivative and 41-point
loss slice whenever the learning rate changed, although neither depends on
that rate. It now retains only the latest result for each dependency set and
reuses unchanged curve/table nodes. Changing weights, the selected coordinate,
batch, L2 coefficient, epsilon or slice radius invalidates the affected work.
Apply still reads the current controls, invalid values disable it, and undo
restores the exact previous parameters. Recorded model/data bytes are unchanged.

The actual handler benchmark used the recorded Moons model, 64 training rows
and 200 warmed rate edits in the test DOM adapter:

| Handler duration | Before | After |
| --- | ---: | ---: |
| Median | 4.010 ms | 0.241 ms |
| p95 | 4.100 ms | 0.312 ms |

This is about 94% less measured handler time. It is not an INP measurement,
browser-paint benchmark, or a claim that every device runs at this speed.
`gradient-ui.json` records the method and measurements. Correctness tests
assert cache invalidation and numerical equivalence rather than fragile timing
thresholds.

The hosted build now shares content-hashed JavaScript and CSS across the two
advanced labs, with one-year immutable browser caching. The shared files total
59,687 uncompressed bytes (about 16.7 KB with the measured local gzip settings).
They are reusable across lab navigations. Moons HTML drops from 540,986 to
482,820 bytes; Circles drops from 437,577 to 379,411 bytes. A first visit must
also fetch the shared assets: these reductions are not an equivalent reduction
in total cold-load bytes. The offline HTML reports remain self-contained.

The site stylesheet and KaTeX font directory are also content-addressed. HTML
keeps revalidation so new releases are discoverable. WOFF2-only fonts reduce
the deployment bundle; modern visitors already selected WOFF2, so the removed
legacy variants are not counted as visitor bandwidth savings. Live checks
confirmed immutable asset headers, revalidated HTML and exact deployed bytes.

## What the capacity test means

This is an HTTP/2 **document-navigation workload**. The initial checks used
one Mac and network through Cloudflare's MAA edge; the final distributed check
uses two hosted Ubuntu runners. It does not launch full browsers. Each
virtual learner downloads one complete HTML document, then spends 60 seconds
reading or using the already-loaded page before navigating again. Routes rotate
between the home, Moons, Circles and optimization guide. Interactions such as
gradient updates run in the visitor's browser and make no backend requests.

All target learners must complete a verified first response before the timed
full-population hold starts. A response must be HTTP 200, have the expected
content type and title, decompress correctly, and match the complete known
build's SHA-256. Counting scheduled timers is not sufficient to pass.

The generator uses eight reusable HTTP/2 connections, at most 128 requests in
flight and a 400-request/second dispatch ceiling. These are generator bounds,
not claims that the server's maximum capacity is 128 requests or 400 requests
per second. It records actual concurrency, queue delay, latency, memory and
event-loop delay. The 3 GiB transfer budget is a soft stop: requests already
in flight can finish while it drains. A 403, 429 or Cloudflare challenge stops
new requests immediately; the test does not bypass those controls.

Overall results and the full-population hold independently require no more
than 1% failed responses, p95 response time at most 2 seconds and p95 dispatch
delay at most 1 second. The hold must contain actual navigation traffic, and
the generator's event-loop p95 must be at most 100 ms.

The workload excludes CSS/JavaScript/font subrequests, real browser rendering
and interaction CPU, independent devices/IPs, multi-region conditions and
10,000 simultaneous requests in flight. It represents navigation with shared
assets already cached. Functional browser checks and asset-integrity tests
cover those omitted application features separately. It is not evidence for
a synchronized cold start of 10,000 complete browsers, a long soak, or a global
availability guarantee.

## Reproduce the bounded live checks

Build and deploy the exact source before testing. The CLI defaults to the
repository's `dist/` for expected page hashes. Testing a different deployed
revision correctly fails content verification.

```sh
npm ci --ignore-scripts
npm test
npm run build

# Small sustained smoke: deliberately faster navigation than the main model.
npm run test:capacity -- --vus 100 --ramp-seconds 5 --hold-seconds 10 \
  --think-seconds 2 --output benchmarks/live-optimized-100.json

# Main reading-time model, staged before the largest run.
npm run test:capacity -- --vus 1000 --ramp-seconds 30 --hold-seconds 65 \
  --think-seconds 60 --output benchmarks/live-optimized-1000.json
npm run test:capacity -- --vus 10000 --ramp-seconds 120 --hold-seconds 65 \
  --think-seconds 60 --output benchmarks/live-optimized-10000.json
```

Only the owned production origin and loopback fixture servers are accepted.
Run stages sequentially and inspect each report before increasing load.
Ctrl-C writes a failed partial report. Regular `npm test` only uses local
HTTP/2 fixtures; CI does not generate load against the public site.

The original `live-baseline-100.json` is an initialization smoke from schema 1,
with a ten-second idle hold and no repeated navigations. It predates full-body
hash and hold-specific checks and is not used as evidence for the final
capacity result or as a controlled before/after throughput comparison.

Cloudflare documents [automatic Pages caching and compression](https://developers.cloudflare.com/pages/configuration/serving-pages/)
and [static-asset request pricing](https://developers.cloudflare.com/pages/functions/pricing/).
Those platform descriptions are not a concurrent-user service-level guarantee.

## Single-generator results

| Profile | Verified learners | Complete responses | Failures | Overall response p95 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 s ramp / 10 s hold / 2 s think | 100 | 657 | 0 | 111.477 ms | Passed |
| 30 s ramp / 65 s hold / 60 s think | 1,000 | 2,000 | 0 | 109.331 ms | Passed |
| 120 s ramp / 65 s hold / 60 s think | 10,000 | 23,410 | 0 | 2,075.924 ms | Failed latency and dispatch limits |

The local 10,000-user run is retained as a **failed** result, not recast as a
success because its HTTP responses were correct. It reached all 10,000 verified
learners and sustained the population for 65.003 seconds, but the generator
reached its 128-request ceiling, queued up to 2,375 navigations and recorded
18.746 seconds p95 dispatch delay. The generator event-loop p95 stayed at
21.135 ms. The run transferred about 1.25 GiB over 206.741 seconds. These results
alone cannot distinguish local network, connection-pool and delivery limits.

## Independent hosted generators

The manual `capacity.yml` workflow runs two independent Ubuntu generators with
5,000 virtual learners each: 60-second ramps, 120-second holds and 60-second
reading times. Each retains the full-body checks and limits above, with a
200-request/second ceiling per runner. The summary requires both reports to
pass and their verified full-population windows to overlap by at least 65
seconds. It checks identical routes, current build hashes and browsing settings.

The generators retain bounded per-request dispatch/completion times, response
latency, queue delay and success status. The summary combines requests
dispatched inside the shared hold window and computes overlap-only latency,
dispatch-delay percentiles and error rate directly from those samples. Requests
are counted by dispatch time, including their complete response time even if
they finish just after the window. The overlap must independently satisfy the
same 2-second response p95, 1-second dispatch p95 and 1% error limits.
Percentiles are calculated from samples, never averaged across runners.
Whole-run traffic totals are labeled separately. This separates the single-laptop
constraint from two independent generators, without claiming global coverage
or full-browser simulation. The workflow runs only when manually requested;
it does not run on pushes or pull requests.

## Final distributed result — 7 November 2026 (India)

[The manual workflow](https://github.com/TARUNTEjjjj/neural-math-lab/actions/runs/42272532458)
passed on source commit `244091de7fa57c1ec77dbdf1e0ab20159bd39800`.
Both generators initialized all 5,000 learners and completed their 120-second
holds. Their shared 10,000-learner window lasted **118.840 seconds**, from
`2026-11-06T18:39:36.388Z` to `2026-11-06T18:41:35.228Z` (UTC).

| Shared-overlap measure | Observed | Limit |
| --- | ---: | ---: |
| Concurrent initialized virtual learners | 10,000 | 10,000 |
| Verified complete document responses | 19,797 | Actual navigation traffic required |
| Failed responses | 0 (0%) | ≤1% |
| Response p50 / p95 / p99 | 46.702 / 93.027 / 119.160 ms | p95 ≤2,000 ms |
| Dispatch-delay p95 | 9.883 ms | ≤1,000 ms |
| Verified overlap duration | 118.840 s | ≥65 s |

The complete runs returned **29,992** valid documents, with zero failures,
including their ramps. The reported Cloudflare edges were IAD, MSP and ORD.
Peak requests in flight were 16 and 28 on the respective generators; the
test did not put 10,000 HTTP requests in flight at once. Event-loop p95 was
21.135 ms for each runner. Each runner independently passed all overall and
hold-specific checks. The pooled overlap checks also all passed.

Raw reports and the exact summary are retained in
[`benchmarks/distributed-10000/`](../benchmarks/distributed-10000/README.md).
Recomputing the downloaded reports locally against the current build produced
the same summary. This supports the stated 60-second-reading navigation model
from those generators and edges. It does not establish cold-start, long-soak,
full-browser or worldwide performance, nor remove the failed laptop result.

During the verified overlap, the production lab reloaded successfully and
applied/undid a 64-row full-gradient step at learning rate 0.2. The displayed
objective changed from 0.9182262 to 0.6848585 and back to 0.9182262. No browser
warnings or errors appeared in those interactions. Separate 390-pixel mobile
checks covered apply/undo, invalid-rate recovery and absence of horizontal
page overflow. The optimization guide rendered 75 MathML equations without
math errors. These are functional browser observations, not timing benchmarks.

All 43 final HTTP checks passed: expected pages and assets matched the built
bytes, hashed assets had immutable cache headers, HTML was revalidated, and
five missing/private-source paths returned the exact custom 404 page.
`live-browser-checks.json` and `live-release-http.json` record these checks.
