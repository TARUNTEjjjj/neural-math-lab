"use strict";

const assert = require("node:assert/strict");
const { before, after, test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const crypto = require("node:crypto");
const { buildSite, renderLesson } = require("../scripts/build-site.cjs");

const root = path.resolve(__dirname, "..");
const lessonSlugs = ["mathematics", "exercises", "advanced-mathematics", "advanced-exercises",
  "decision-mathematics", "decision-exercises", "optimization-mathematics", "optimization-exercises"];
let temporary, output, files, offlineSnapshots;

function walk(directory, prefix = "") {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relative = prefix + entry.name;
    return entry.isDirectory() ? walk(path.join(directory, entry.name), relative + "/") : [relative];
  });
}
const read = (relative) => fs.readFileSync(path.join(output, relative), "utf8");
const scripts = (html) => [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map((match) => match[0]);
const assetLinks = (html, extension) => [...html.matchAll(new RegExp(`(?:href|src)="(/assets/[^"?#]+\\.${extension})"`, "g"))].map((match) => match[1]);
const digest = (content, encoding = "hex") => crypto.createHash("sha256").update(content).digest(encoding);

before(() => {
  temporary = fs.mkdtempSync(path.join(os.tmpdir(), "neural-public-site-"));
  offlineSnapshots = Object.fromEntries(["moons", "circles", "xor"].map((name) => [name,
    fs.readFileSync(path.join(root, `examples/${name}/report.html`), "utf8")]));
  output = buildSite(path.join(temporary, "public"));
  files = walk(output);
});
after(() => { if (temporary) fs.rmSync(temporary, { recursive: true, force: true }); });

test("public build contains only learning pages and explicitly required static assets", () => {
  const expected = new Set(["index.html", "404.html", "_headers", "offline-resources-worker.js", "app-version.json",
    ...['index.html', 'explorer.css', 'explorer.mjs', 'catalog.json', 'catalog.mjs', 'storage.mjs', 'live.mjs', 'background.mjs', 'offline.mjs', 'app-updates.mjs', 'offline-worker.js'].map(file => 'resources/' + file),
    ...["moons", "circles", "xor"].map((name) => `labs/${name}/index.html`),
    ...lessonSlugs.map((slug) => `learn/${slug}/index.html`)]);
  for (const file of expected) assert.ok(files.includes(file), "Missing public page/asset: " + file);
  for (const file of files) {
    assert.ok(expected.has(file) || /^assets\/(?:site|lab)\.[a-f0-9]{16}\.(?:css|js)$/.test(file) ||
      /^assets\/katex\/[a-f0-9]{16}\/(?:katex\.min\.css|LICENSE\.txt|fonts\/KaTeX_[\w-]+\.woff2)$/.test(file), "Unexpected published file: " + file);
    if (file.endsWith(".json")) assert.ok(["resources/catalog.json", "app-version.json"].includes(file));
    assert.doesNotMatch(file, /(^|\/)(?:\.git|\.env|node_modules|neural_math_lab|tests)(\/|$)|\.(?:py|map|md)$/);
  }
  assert.equal(files.filter((name) => name.endsWith(".html")).length, 14);
  assert.ok(files.includes("resources/index.html"));
  assert.ok(files.includes("resources/catalog.json"));
  const release=JSON.parse(read("app-version.json"));
  assert.equal(release.project,"neural-math-lab");
  assert.match(release.version,/^[a-f0-9]{64}$/);
  for(const file of files.filter(file=>file.endsWith('.html'))) assert.ok(read(file).includes(`name="app-version" content="${release.version}"`),file+' must carry the actual running release');
  for(const worker of ['offline-resources-worker.js','resources/offline-worker.js']) {
    assert.ok(read(worker).includes(`const APP_VERSION = "${release.version}"`));
    assert.doesNotMatch(read(worker),/APP_UPDATE_PROTOCOL|APP_VERSION \*\//);
  }
  assert.deepEqual(read('resources/catalog.json'),fs.readFileSync(path.join(root,'resources/catalog.json'),'utf8'),'Code updates preserve bundled collection bytes');
  assert.match(read('_headers'),/\/app-version\.json\n  Cache-Control: no-store/);
  assert.match(read("_headers"), /\/offline-resources-worker\.js\n  Cache-Control: no-cache/);
});

test("every generated local page, asset and fragment link resolves", () => {
  let checked = 0;
  for (const file of files.filter((name) => name.endsWith(".html"))) {
    const html = read(file);
    const base = new URL(file.replace(/index\.html$/, ""), "https://neural.test/");
    for (const match of html.matchAll(/<(?:a|link|img|script|use)\b[^>]*?\b(?:href|src)="([^"]+)"/g)) {
      const target = new URL(match[1].replaceAll("&amp;", "&"), base);
      if (target.origin !== base.origin) continue;
      let relative = decodeURIComponent(target.pathname).slice(1);
      if (!relative || relative.endsWith("/")) relative += "index.html";
      assert.ok(files.includes(relative), `${file} links to missing ${relative}`);
      if (target.hash) {
        const fragment = decodeURIComponent(target.hash.slice(1));
        const ids = [...read(relative).matchAll(/\bid="([^"]+)"/g)].map((item) => item[1]);
        assert.ok(ids.includes(fragment), `${file} links to missing #${fragment} in ${relative}`);
      }
      checked++;
    }
    assert.doesNotMatch(html, /<a\b[^>]*href="[^"]*(?:neural_math_lab\/|\.md(?:#|"))/);
    assert.doesNotMatch(html, /<a\b[^>]*href="https:\/\/github\.com\/TARUNTEjjjj\//i);
  }
  assert.ok(checked > 100, "The crawl should cover navigation, lessons, figures and report anchors.");
});

test("hosted reports preserve exact data and runtime bytes with deferred integrity-checked scripts", () => {
  for (const name of ["moons", "circles", "xor"]) {
    const original = offlineSnapshots[name];
    const hosted = read(`labs/${name}/index.html`);
    assert.match(hosted, /aria-label="Learning website"/);
    assert.equal(fs.readFileSync(path.join(root, `examples/${name}/report.html`), "utf8"), original,
      "Building the hosted site must not rewrite the offline report.");
    const originalStyle = original.match(/<style>([\s\S]*?)<\/style>/)[1];
    const styleFiles = assetLinks(hosted, "css");
    assert.equal(styleFiles.length, 1);
    assert.ok(read(styleFiles[0].slice(1)).startsWith(originalStyle), "Original report styles must be preserved.");
    assert.match(hosted, /style-src 'self' 'unsafe-inline'/);
    if (name !== "xor") {
      const dataTag = /<script id="experiment-data" type="application\/json">[\s\S]*?<\/script>/;
      assert.equal(hosted.match(dataTag)[0], original.match(dataTag)[0], "Recorded data must stay byte-identical.");
      assert.doesNotMatch(hosted, /<script>/, "Hosted runtime should not be duplicated inline.");
      const originalRuntime = original.match(/<script>([\s\S]*?)<\/script>/)[1];
      const runtimeFiles = assetLinks(hosted, "js");
      assert.equal(runtimeFiles.length, 1);
      assert.equal(read(runtimeFiles[0].slice(1)), originalRuntime);
      assert.match(hosted, /<script defer src="\/assets\/lab\.[a-f0-9]{16}\.js"/);
      assert.ok(hosted.includes(`integrity="sha256-${digest(originalRuntime, "base64")}"`));
      assert.ok(hosted.includes(`script-src 'self'`));
      assert.ok(original.includes(`script-src 'sha256-${digest(originalRuntime, "base64")}'`), "Offline CSP remains valid.");
      assert.match(hosted, /href="\/learn\/optimization-mathematics\/"/);
      assert.match(hosted, /href="\/learn\/optimization-exercises\/"/);
    } else assert.deepEqual(scripts(hosted), ['<script type="module" src="/resources/background.mjs"></script>'], "XOR calculations remain pre-rendered; only resource saving runs in the background.");
    assert.match(hosted, /connect-src 'self' https:\/\/raw.githubusercontent.com/);
    assert.match(hosted, /worker-src 'self'/);
  }
});

test("lesson equations are pre-rendered with accessible MathML and local font assets", () => {
  for (const slug of lessonSlugs) {
    const html = read(`learn/${slug}/index.html`);
    assert.doesNotMatch(html, /katex-error|KaTeX parse error|No math renderer found/);
    assert.match(html, /<math\b/);
    assert.match(html, /class="katex-html"/);
    assert.match(html, /<link rel="stylesheet" href="\/assets\/katex\/[a-f0-9]{16}\/katex.min.css">/);
    assert.deepEqual(scripts(html), ['<script type="module" src="/resources/background.mjs"></script>'], "Lesson equations stay pre-rendered; only resource saving runs in the background.");
  }
  const mathCss = files.find((name) => /^assets\/katex\/[a-f0-9]{16}\/katex\.min\.css$/.test(name));
  assert.ok(mathCss, "Locally served maths stylesheet is required.");
  let references = 0;
  for (const match of read(mathCss).matchAll(/url\(["']?([^)'" ]+)["']?\)/g)) {
    const font = path.posix.normalize(path.posix.join(path.posix.dirname(mathCss), match[1]));
    assert.ok(files.includes(font), "Missing locally served math font: " + font);
    assert.match(font, /\.woff2$/);
    references++;
  }
  assert.ok(references > 10, "Check the complete set of maths font references.");
  assert.match(read(path.posix.join(path.posix.dirname(mathCss), "LICENSE.txt")), /MIT License/);
});

test("shared assets are content-addressed and immutable caching never applies to HTML", () => {
  for (const file of files.filter((name) => /^assets\/(?:site|lab)\./.test(name))) {
    const hash = file.match(/\.([a-f0-9]{16})\.(?:js|css)$/)[1];
    assert.equal(hash, digest(fs.readFileSync(path.join(output, file))).slice(0, 16), "Stale asset fingerprint: " + file);
  }
  const mathCss = files.find((name) => /^assets\/katex\/[a-f0-9]{16}\/katex\.min\.css$/.test(name));
  const mathDirectory = path.posix.dirname(mathCss);
  const mathHash = crypto.createHash("sha256").update(read(mathCss));
  for (const file of files.filter((name) => name.startsWith(mathDirectory + "/fonts/")).sort()) {
    mathHash.update(path.posix.basename(file)).update(fs.readFileSync(path.join(output, file)));
  }
  assert.equal(mathDirectory.split("/").at(-1), mathHash.digest("hex").slice(0, 16), "Maths font or CSS changes must invalidate its directory URL.");
  const headerBlocks = read("_headers").trim().split(/\n\s*\n/);
  assert.ok(headerBlocks.some((block) => block.startsWith("/assets/*\n") && /Cache-Control: public, max-age=31536000, immutable/.test(block)));
  for (const block of headerBlocks) if (/immutable/.test(block)) assert.ok(block.startsWith("/assets/*\n"), "HTML must remain revalidatable after deployment.");
  assert.ok(files.filter((file) => file.endsWith(".html")).every((file) => !file.startsWith("assets/")));
});

test("switching advanced labs reuses one script and stylesheet; reading pages share their stylesheet", () => {
  const moons = read("labs/moons/index.html"), circles = read("labs/circles/index.html");
  assert.deepEqual(assetLinks(moons, "js"), assetLinks(circles, "js"));
  assert.deepEqual(assetLinks(moons, "css"), assetLinks(circles, "css"));
  const siteCss = assetLinks(read("index.html"), "css").find((name) => /^\/assets\/site\./.test(name));
  assert.ok(siteCss);
  for (const page of ["404.html", ...lessonSlugs.map((slug) => `learn/${slug}/index.html`)]) {
    assert.ok(assetLinks(read(page), "css").includes(siteCss), page + " must reuse the home stylesheet.");
  }
});

test("bad equations and unresolved guide references fail the build instead of publishing broken lessons", () => {
  assert.throws(() => renderLesson("mathematics", "# Broken\n\n$\\notARealMathCommand{x}$"), /Undefined control sequence/);
  assert.throws(() => renderLesson("mathematics", "[Missing](unknown-guide.md)"), /Unknown lesson/);
  const rendered = renderLesson("mathematics", "# Guide\n\nSee [`core.py`](../neural_math_lab/core.py) and [exercises](exercises.md).\n");
  assert.match(rendered, /<code>core\.py<\/code>/);
  assert.doesNotMatch(rendered, /href="[^\"]*core\.py/);
  assert.match(rendered, /href="\/learn\/exercises\/"/);
});

test("hosting has a real not-found document and preserves unrelated output directories", () => {
  assert.match(read("404.html"), /This page is not in the lab/);
  assert.match(read("404.html"), /href="\/"/);
  assert.ok(!files.includes("_redirects"), "Do not turn source or missing paths into a successful SPA fallback.");
  assert.match(read("_headers"), /X-Content-Type-Options: nosniff/);
  assert.match(read("_headers"), /frame-ancestors 'none'/);
  const existing = path.join(temporary, "keep-me");
  fs.mkdirSync(existing);
  fs.writeFileSync(path.join(existing, "sentinel.txt"), "Keep this");
  assert.throws(() => buildSite(existing), /Custom output must not already exist/);
  assert.equal(fs.readFileSync(path.join(existing, "sentinel.txt"), "utf8"), "Keep this");
});
