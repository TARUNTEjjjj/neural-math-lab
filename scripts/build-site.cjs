"use strict";

// Publish an explicit allowlist, never the checkout or the whole examples folder.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const {execFileSync} = require("node:child_process");
const {pathToFileURL} = require("node:url");
const MarkdownIt = require("markdown-it");
const katex = require("katex");
const texmath = require("markdown-it-texmath");
const ROOT = path.resolve(__dirname, "..");
const LESSONS = {
  mathematics: "The network, from first principles",
  exercises: "First principles · exercises",
  "advanced-mathematics": "Training beyond XOR",
  "advanced-exercises": "Training · exercises",
  "decision-mathematics": "Decisions, ranking & calibration",
  "decision-exercises": "Decisions · exercises",
  "optimization-mathematics": "Gradients & optimization",
  "optimization-exercises": "Optimization · exercises"
};
const CSP = "default-src 'none'; script-src 'self'; worker-src 'self'; connect-src 'self' https://raw.githubusercontent.com; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'";
const escape = (text) => String(text).replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const links = () => Object.entries(LESSONS).map(([slug, title]) => `<li><a href="/learn/${slug}/">${title}</a></li>`).join("");
const header = `<a class="skip-link" href="#main">Skip to content</a><header class="site-header"><a class="site-brand" href="/">∇ Neural Math Lab</a><nav class="site-nav" aria-label="Main navigation"><a href="/#experiments">Experiments</a><a href="/#learning-path">Maths guides</a><a href="/labs/moons/#gradient-lab">Gradient lab</a><a href="/resources/">Resources</a></nav></header>`;
const footer = `<footer class="site-footer"><span>Neural Math Lab · Learn through experiments.</span><span>Calculations stay in your browser · No sign-in required</span></footer>`;

function page(title, content, extraHead = "") {
  extraHead += '<script type="module" src="/resources/background.mjs"></script>';
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="${CSP}"><title>${escape(title)} · Neural Math Lab</title><link rel="stylesheet" href="/site.css">${extraHead}</head><body>${header}${content}${footer}</body></html>`;
}

function renderLesson(slug, source) {
  // Source-file mentions remain readable, but private Python files aren't exposed.
  const prepared = source.replace(/\[([^\]]+)\]\(\.\.\/neural_math_lab\/[^)]+\)/g, "$1")
    .replace(/\]\(([\w-]+)\.md(#[^)]*)?\)/g, (_, target, hash = "") => {
      if (!Object.hasOwn(LESSONS, target)) throw new Error("Unknown lesson: " + target);
      return `](/learn/${target}/${hash})`;
    });
  const md = new MarkdownIt({ html: false }).use(texmath, {
    engine: katex, delimiters: ["dollars", "brackets"],
    katexOptions: { throwOnError: true, trust: false, output: "htmlAndMathml", strict: "error" }
  });
  md.renderer.rules.table_open = () => '<div class="lesson-table"><table>';
  md.renderer.rules.table_close = () => '</table></div>';
  const tokens = md.parse(prepared, {});
  const headings = [];
  const ids = new Set();
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type === "heading_open") {
      const label = tokens[i + 1].content;
      let id = label.toLowerCase().replace(/[^a-z0-9\s-]/g, "").trim().replace(/\s+/g, "-");
      const base = id; let suffix = 1;
      while (ids.has(id)) id = base + "-" + suffix++;
      ids.add(id); tokens[i].attrSet("id", id);
      if (tokens[i].tag === "h2") headings.push(`<li><a href="#${id}">${escape(label)}</a></li>`);
    }
  }
  function verifyMath(items) {
    for (const token of items) {
      if (token.type.startsWith("math")) katex.renderToString(token.content, { throwOnError: true, trust: false, strict: "error", displayMode: token.block });
      if (token.children) verifyMath(token.children);
    }
  }
  verifyMath(tokens); // texmath otherwise turns a rendering error into visible text.
  const rendered = md.renderer.render(tokens, md.options, {});
  return page(LESSONS[slug], `<main id="main" class="site-main lesson-layout"><article class="lesson-content"><p class="eyebrow">THE MATHS / LEARNING NOTES</p>${rendered}<p><a class="button" href="/labs/moons/#gradient-lab">Try the gradient lab →</a></p></article><aside class="lesson-sidebar"><h2>On this page</h2><ol>${headings.join("")}</ol><h2>Keep learning</h2><ul>${links()}</ul></aside></main>`, '<link rel="stylesheet" href="/assets/katex/katex.min.css">');
}

function hostedReport(name, original, publishAsset) {
  const nav = `<nav class="hosted-nav" aria-label="Learning website"><a href="/">← Learning home</a><a href="/labs/moons/">Moons</a><a href="/labs/circles/">Circles</a><a href="/labs/xor/">XOR</a><a href="/learn/optimization-mathematics/">Maths guides</a></nav>`;
  const style = '.hosted-nav{display:flex;flex-wrap:wrap;gap:12px 24px;padding:16px max(16px,calc((100% - 1280px)/2));background:#193e3b;color:#fff;font:600 13px system-ui}.hosted-nav a{color:#fff;text-decoration:none}.hosted-nav a:hover{text-decoration:underline}.hosted-nav a:focus-visible{outline:2px solid #fff;outline-offset:4px}';
  let html = original.replace("</style>", style + "</style>");
  // The standalone XOR report omits optional head/body tags. Make those
  // boundaries explicit in its hosted form before adding release metadata.
  if (publishAsset && name === "xor" && !/<head(?:\s|>)/i.test(html)) {
    html = html.replace(/(<html\b[^>]*>)/i, '$1<head>').replace('<main>', '</head><body><main>').replace('</html>', '</body></html>');
  }
  html = name === "xor" ? html.replace("<main>", nav + "<main>") : html.replace("<body>", "<body>" + nav);
  html = html.replace('href="#" class="brand"', 'href="/" class="brand"');
  for (const [slug, title] of Object.entries(LESSONS)) {
    html = html.replaceAll(`<code>docs/${slug}.md</code>`, `<a href="/learn/${slug}/">${title}</a>`);
  }
  html = html.replace("OFFLINE REPORT", "BROWSER LEARNING LAB");
  html = html.replace("Generated locally · No telemetry · No remote assets", "Browser calculations · Temporary sandbox · No sign-in required");
  if (publishAsset) {
    // Share cacheable bytes between hosted labs. Offline reports stay self-contained.
    html = html.replace(/<style>([\s\S]*?)<\/style>/, (_, css) => `<link rel="stylesheet" href="${publishAsset("lab", "css", css)}">`);
    html = html.replace(/<script>([\s\S]*?)<\/script>/, (_, script) => {
      const integrity = crypto.createHash("sha256").update(script).digest("base64");
      return `<script defer src="${publishAsset("lab", "js", script)}" integrity="sha256-${integrity}" crossorigin="anonymous"></script>`;
    });
    html = html.replace("style-src 'unsafe-inline'", "style-src 'self' 'unsafe-inline'")
      .replace(/script-src 'sha256-[^']+'/g, "script-src 'self'");
  }
  // Only hosted labs load the public resource background client; exported reports stay self-contained.
  if (publishAsset) {
    const background = '<script type="module" src="/resources/background.mjs"></script>';
    html = html.includes('</head>') ? html.replace('</head>', background + '</head>') : html.replace('<main>', background + '<main>');
    html = html.replace(/(<meta http-equiv="Content-Security-Policy" content=")([^"]+)(")/, (_, start, policy, end) => {
      const kept = policy.split(';').map(value => value.trim()).filter(value => value && !/^(?:script-src|worker-src|connect-src)\b/.test(value));
      return start + [...kept, "script-src 'self'", "worker-src 'self'", "connect-src 'self' https://raw.githubusercontent.com"].join('; ') + end;
    });
  }
  return html;
}

function buildSite(output = path.join(ROOT, "dist")) {
  const destination = path.resolve(output);
  // Don't erase an arbitrary path. Fresh temp output is supported by tests.
  if (fs.existsSync(destination)) {
    if (destination !== path.join(ROOT, "dist")) throw new Error("Custom output must not already exist.");
    fs.rmSync(destination, { recursive: true });
  }
  function write(relative, content) {
    const file = path.join(destination, relative);
    fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, content);
  }
  function publishAsset(name, extension, content) {
    const hash = crypto.createHash("sha256").update(content).digest("hex").slice(0, 16);
    const relative = `assets/${name}.${hash}.${extension}`;
    write(relative, content); return "/" + relative;
  }
  const siteStyle = publishAsset("site", "css", fs.readFileSync(path.join(ROOT, "site/site.css")));
  function withSharedStyle(html) { return html.replaceAll('href="/site.css"', `href="${siteStyle}"`); }
  const home = fs.readFileSync(path.join(ROOT, "site/index.html"), "utf8");
  write("index.html", withSharedStyle(home.replace("<head>", `<head><meta http-equiv="Content-Security-Policy" content="${CSP}">`)));
  for (const name of ["moons", "circles", "xor"]) {
    write(`labs/${name}/index.html`, hostedReport(name, fs.readFileSync(path.join(ROOT, `examples/${name}/report.html`), "utf8"), publishAsset));
  }
  const katexRoot = path.resolve(path.dirname(require.resolve("katex")), "..");
  // Modern browsers use WOFF2. Include only those files and fingerprint the whole
  // dependency directory so a future KaTeX upgrade cannot reuse stale font URLs.
  const fontNames = fs.readdirSync(path.join(katexRoot, "dist/fonts")).filter(name => name.endsWith(".woff2")).sort();
  const mathCss = fs.readFileSync(path.join(katexRoot, "dist/katex.min.css"), "utf8")
    .replace(/src:([^;]+);/g, (_, sources) => {
      const woff2 = sources.split(",").filter(source => source.includes(".woff2"));
      if (woff2.length !== 1) throw new Error("Expected one WOFF2 source per font face.");
      return "src:" + woff2[0] + ";";
    });
  const mathHash = crypto.createHash("sha256").update(mathCss);
  for (const name of fontNames) mathHash.update(name).update(fs.readFileSync(path.join(katexRoot, "dist/fonts", name)));
  const mathDirectory = "assets/katex/" + mathHash.digest("hex").slice(0, 16);
  write(mathDirectory + "/katex.min.css", mathCss);
  write(mathDirectory + "/LICENSE.txt", fs.readFileSync(path.join(katexRoot, "LICENSE")));
  for (const font of fontNames) write(mathDirectory + "/fonts/" + font, fs.readFileSync(path.join(katexRoot, "dist/fonts", font)));
  for (const slug of Object.keys(LESSONS)) {
    const html = withSharedStyle(renderLesson(slug, fs.readFileSync(path.join(ROOT, `docs/${slug}.md`), "utf8")))
      .replace('href="/assets/katex/katex.min.css"', `href="/${mathDirectory}/katex.min.css"`);
    write(`learn/${slug}/index.html`, html);
  }
  write("404.html", withSharedStyle(page("Page not found", '<main id="main" class="site-main"><p class="eyebrow">404 / FIND YOUR WAY BACK</p><h1>This page is not in the lab.</h1><p>Choose an experiment or a maths guide from the learning home.</p><a class="button" href="/">Back to Neural Math Lab →</a></main>')));
  write("_headers", "/*\n  X-Content-Type-Options: nosniff\n  X-Frame-Options: DENY\n  Referrer-Policy: strict-origin-when-cross-origin\n  Permissions-Policy: camera=(), microphone=(), geolocation=()\n  Content-Security-Policy: frame-ancestors 'none'; base-uri 'none'; form-action 'none'\n\n/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n/offline-resources-worker.js\n  Cache-Control: no-cache\n/app-version.json\n  Cache-Control: no-store\n/resources/offline-worker.js\n  Cache-Control: no-cache\n");
  // Stamp every app page and the verified resource shell from one source identity.
  const stageModule = pathToFileURL(path.join(ROOT, 'scripts/stage-resource-explorer.mjs')).href;
  const releaseModule = pathToFileURL(path.join(ROOT, 'scripts/app-release.mjs')).href;
  const stage = `import {readFile,writeFile,readdir} from 'node:fs/promises';
    import {resolve} from 'node:path';
    import {readAppRelease,stampAppVersion,writeAppVersion} from ${JSON.stringify(releaseModule)};
    import {stageResourceExplorer,stageRootOfflineWorker} from ${JSON.stringify(stageModule)};
    const root=process.argv[1], output=process.argv[2];
    const release=await readAppRelease(root,{project:'neural-math-lab',sourcePaths:[
      'site','resources','neural_math_lab/assets','docs','examples/moons/report.html',
      'examples/circles/report.html','examples/xor/report.html','offline-resources-worker.js',
      'scripts/build-site.cjs','scripts/app-release.mjs','scripts/app-update-protocol.mjs',
      'scripts/stage-resource-explorer.mjs','package.json','package-lock.json']});
    async function stamp(dir){for(const entry of await readdir(dir,{withFileTypes:true})){
      const file=resolve(dir,entry.name);
      if(entry.isDirectory()) await stamp(file);
      else if(entry.name.endsWith('.html')) await writeFile(file,stampAppVersion(await readFile(file,'utf8'),release));
    }}
    await stamp(output);
    await stageResourceExplorer(root,resolve(output,'resources'),{release});
    await stageRootOfflineWorker(root,output,{release});
    await writeAppVersion(output,release,{workers:[{scope:'/',script:'/offline-resources-worker.js'},{scope:'/resources/',script:'/resources/offline-worker.js'}]});`;
  execFileSync(process.execPath, ['--input-type=module','--eval',stage,ROOT,destination], {stdio:'pipe'});
  return destination;
}

module.exports = { buildSite, hostedReport, renderLesson, LESSONS };
if (require.main === module) console.log("Built public assets in " + buildSite());
