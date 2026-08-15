#!/usr/bin/env node
/**
 * eval_math.mjs — can KaTeX actually render the math we kept?
 *
 * Extracts every math span from the index and renders it with the macros pulled
 * from the thesis preamble, reporting what fails and why. Run after any change
 * to the LaTeX handling in build_rag.mjs or to rag/macros.json.
 *
 * Usage: node tools/eval_math.mjs [--verbose]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import katex from '../scripts/ask/vendor/katex.mjs';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const VERBOSE = process.argv.includes('--verbose');

const index = JSON.parse(readFileSync(path.join(ROOT, 'rag', 'all.json'), 'utf8'));
const sets = JSON.parse(readFileSync(path.join(ROOT, 'rag', 'macros.json'), 'utf8'));

const RE = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
let total = 0;
let ok = 0;
const failures = new Map();

const perSource = new Map();
for (const chunk of index.chunks) {
  if (!chunk.text.includes('$')) continue;
  const macros = { ...(sets._shared ?? {}), ...(sets[chunk.source.m] ?? {}) };
  const src = chunk.source.title.slice(0, 40);
  if (!perSource.has(src)) perSource.set(src, { ok: 0, fail: 0 });
  const tally = perSource.get(src);
  RE.lastIndex = 0;
  let m;
  while ((m = RE.exec(chunk.text))) {
    const body = m[1] ?? m[2];
    const display = m[1] != null;
    total++;
    try {
      // Fresh copy per call: KaTeX may write \gdef definitions into the map.
      katex.renderToString(body, {
        displayMode: display, output: 'mathml', macros: { ...macros },
        throwOnError: true, strict: 'ignore',
      });
      ok++; tally.ok++;
    } catch (err) {
      const reason = String(err.message).replace(/ at position \d+.*/s, '').slice(0, 80);
      if (!failures.has(reason)) failures.set(reason, []);
      failures.get(reason).push(body.slice(0, 90)); tally.fail++;
    }
  }
}

console.log(`\nmath spans: ${total}`);
console.log(`rendered:   ${ok} (${((100 * ok) / Math.max(1, total)).toFixed(1)}%)`);
console.log(`failed:     ${total - ok}\n`);

console.log('by source:');
for (const [src, t] of [...perSource.entries()].sort((a, b) => (b[1].ok + b[1].fail) - (a[1].ok + a[1].fail))) {
  const tot = t.ok + t.fail;
  console.log(`  ${String(tot).padStart(5)} spans  ${((100 * t.ok) / tot).toFixed(1).padStart(5)}%  ${src}`);
}
console.log();

const ranked = [...failures.entries()].sort((a, b) => b[1].length - a[1].length);
for (const [reason, samples] of ranked.slice(0, 12)) {
  console.log(`${String(samples.length).padStart(5)}  ${reason}`);
  if (VERBOSE) for (const s of samples.slice(0, 2)) console.log(`         e.g. ${s}`);
}
