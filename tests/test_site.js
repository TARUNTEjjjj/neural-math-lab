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
let temporary, output, files;

function walk(directory, prefix = "") {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relative = prefix + entry.name;
    return entry.isDirectory() ? walk(path.join(directory, entry.name), relative + "/") : [relative];
  });
}
const read = (relative) => fs.readFileSync(path.join(output, relative), "utf8");
const scripts = (html) => [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map((match) => match[0]);

before(() => {
  temporary = fs.mkdtempSync(path.join(os.tmpdir(), "neural-public-site-"));
  output = buildSite(path.join(temporary, "public"));
  files = walk(output);
});
after(() => { if (temporary) fs.rmSync(temporary, { recursive: true, force: true }); });

test("public build contains only learning pages and explicitly required static assets", () => {
  const expected = new Set(["index.html", "404.html", "site.css", "_headers",
    "assets/katex/katex.min.css", "assets/katex/LICENSE.txt",
    ...["moons", "circles", "xor"].map((name) => `labs/${name}/index.html`),
    ...lessonSlugs.map((slug) => `learn/${slug}/index.html`)]);
  for (const file of expected) assert.ok(files.includes(file), "Missing public page/asset: " + file);
  for (const file of files) {
    assert.ok(expected.has(file) || /^assets\/katex\/fonts\/KaTeX_[\w-]+\.(woff2?|ttf)$/.test(file), "Unexpected published file: " + file);
    assert.doesNotMatch(file, /(^|\/)(?:\.git|\.env|node_modules|neural_math_lab|tests)(\/|$)|\.(?:py|json|map|md)$/);
  }
  assert.equal(files.filter((name) => name.endsWith(".html")).length, 13);
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

test("hosted reports preserve recorded data and runtime bytes with valid inline-script CSP", () => {
  for (const name of ["moons", "circles", "xor"]) {
    const original = fs.readFileSync(path.join(root, `examples/${name}/report.html`), "utf8");
    const hosted = read(`labs/${name}/index.html`);
    assert.deepEqual(scripts(hosted), scripts(original), `${name} runtime or recorded data changed`);
    assert.match(hosted, /aria-label="Learning website"/);
    if (name !== "xor") {
      const runtime = hosted.match(/<script>([\s\S]*?)<\/script>/)[1];
      const expectedHash = crypto.createHash("sha256").update(runtime).digest("base64");
      assert.ok(hosted.includes(`script-src 'sha256-${expectedHash}'`));
      assert.match(hosted, /href="\/learn\/optimization-mathematics\/"/);
      assert.match(hosted, /href="\/learn\/optimization-exercises\/"/);
    }
  }
});

test("lesson equations are pre-rendered with accessible MathML and local font assets", () => {
  for (const slug of lessonSlugs) {
    const html = read(`learn/${slug}/index.html`);
    assert.doesNotMatch(html, /katex-error|KaTeX parse error|No math renderer found/);
    assert.match(html, /<math\b/);
    assert.match(html, /class="katex-html"/);
    assert.match(html, /<link rel="stylesheet" href="\/assets\/katex\/katex.min.css">/);
    assert.equal(scripts(html).length, 0, "Lesson rendering must not need browser JavaScript.");
  }
  for (const match of read("assets/katex/katex.min.css").matchAll(/url\(["']?([^)'" ]+)["']?\)/g)) {
    const font = path.posix.normalize(path.posix.join("assets/katex", match[1]));
    assert.ok(files.includes(font), "Missing locally served math font: " + font);
  }
  assert.match(read("assets/katex/LICENSE.txt"), /MIT License/);
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
