#!/usr/bin/env node
/**
 * probe.mjs — run a query through exactly the runtime's retrieval path.
 *
 * Mirrors scripts/ask/retrieval.js: per-corpus noise floors, ranking by margin
 * above a chunk's own floor, the reject/weak thresholds. Use it to find out why
 * a query returns "No close match" — the answer is usually the gate, not the
 * index.
 *
 * Usage: node tools/probe.mjs "your question" ["another" …]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);

const REJECT_MARGIN = 0.005;
const WEAK_MARGIN = 0.05;
const MARGIN = 0.10;
const PER_DOC = 2;
const K = 4;

const index = JSON.parse(readFileSync(path.join(ROOT, 'rag', 'all.json'), 'utf8'));
const bin = readFileSync(path.join(ROOT, 'rag', index.vectors.file));
const vecs = new Int8Array(bin.buffer, bin.byteOffset, bin.byteLength);
const { dims, pooling, normalize, query_prefix, model, dtype } = index.embedding;

const { env, pipeline } = await import('@huggingface/transformers');
env.cacheDir = path.join(TOOLS, '.cache', 'models');
const ex = await pipeline('feature-extraction', model, { dtype });

const queries = process.argv.slice(2);
if (!queries.length) {
  console.error('usage: node tools/probe.mjs "your question"');
  process.exit(1);
}

for (const q of queries) {
  const out = await ex([(query_prefix ?? '') + q], { pooling, normalize });
  const qv = out.data;

  const scored = index.chunks.map((c, i) => {
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += qv[j] * vecs[i * dims + j];
    const score = dot / 127;
    return { i, c, score, margin: score - (index.noise_floors[c.c] ?? 0) };
  }).sort((a, b) => b.margin - a.margin);

  const best = scored[0].margin;
  const verdict = best <= REJECT_MARGIN ? 'REJECTED ("No close match")'
    : best < WEAK_MARGIN ? 'weak (shown, flagged)' : 'ok';

  console.log(`\nQ: ${q}`);
  console.log(`   best margin ${best.toFixed(4)} → ${verdict}`);
  console.log(`   floors: ${JSON.stringify(index.noise_floors)}`);

  const cutoff = Math.max(0, best - MARGIN);
  const perDoc = new Map();
  let shown = 0;
  for (const h of scored.slice(0, 12)) {
    const doc = h.c.source.url.split('#')[0];
    const n = perDoc.get(doc) ?? 0;
    const display = h.margin >= cutoff && n < PER_DOC && shown < K && best > REJECT_MARGIN;
    if (display) { perDoc.set(doc, n + 1); shown++; }
    const where = `${h.c.source.title.slice(0, 34)} · ${h.c.source.loc ?? ''}`.slice(0, 60);
    console.log(`   ${display ? '✓' : ' '} ${h.margin.toFixed(4)} [${h.c.c}] ${where.padEnd(60)} ${h.c.text.replace(/\s+/g, ' ').slice(0, 50)}…`);
  }
}
