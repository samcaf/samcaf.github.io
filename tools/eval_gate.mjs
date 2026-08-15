#!/usr/bin/env node
/**
 * eval_gate.mjs — find a rule that tells a real question from an off-topic one.
 *
 * A fixed score floor cannot do it: BGE cosines are compressed high, and their
 * absolute level shifts with corpus size and homogeneity (the 47-chunk CV scores
 * everything alike because it is all about one person). So this measures several
 * candidate statistics on known-good and known-irrelevant queries, per corpus,
 * and prints whether each separates them.
 *
 * Usage: node tools/eval_gate.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);

const GOOD = {
  cv: ['What did your PhD research cover?', 'What programming languages do you use?',
       'Where did you go to graduate school?', 'Do you have teaching experience?',
       'Who was your PhD advisor?'],
  publications: ['What is PIRANHA?', 'What are energy correlators?', 'qubit noise and decoherence',
                 'long-lived particles at colliders', 'How does jet grooming remove pileup?'],
  notes: ['What does the optical theorem say?', 'What is the TT-bar deformation?',
          'supersymmetric quantum mechanics', 'inflation and the cosmological collider',
          'Grassmannian and Plucker coordinates'],
};

const BAD = [
  'zebra parking meter',
  'how do I bake sourdough bread',
  'best pizza places in Chicago',
  'how to change a flat car tire',
  'what is the capital of Peru',
  'my cat keeps knocking things off the table',
];

const { env, pipeline } = await import('@huggingface/transformers');
env.cacheDir = path.join(TOOLS, '.cache', 'models');

function stats(scores) {
  const n = scores.length;
  const sorted = [...scores].sort((a, b) => b - a);
  const top1 = sorted[0];
  const median = sorted[Math.floor(n / 2)];
  const mean = scores.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1e-9;
  const top10 = sorted.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  return { top1, median, spread: top1 - median, z: (top1 - mean) / sd, zTop10: (top10 - mean) / sd };
}

const rows = [];
for (const corpus of ['cv', 'publications', 'notes']) {
  const index = JSON.parse(readFileSync(path.join(ROOT, 'rag', `${corpus}.json`), 'utf8'));
  const bin = readFileSync(path.join(ROOT, 'rag', index.vectors.file));
  const vecs = new Int8Array(bin.buffer, bin.byteOffset, bin.byteLength);
  const { dims, pooling, normalize, query_prefix, model, dtype } = index.embedding;
  const ex = await pipeline('feature-extraction', model, { dtype });

  const run = async (q) => {
    const out = await ex([(query_prefix ?? '') + q], { pooling, normalize });
    const qv = out.data;
    const scores = new Array(index.count);
    for (let i = 0; i < index.count; i++) {
      let dot = 0;
      for (let j = 0; j < dims; j++) dot += qv[j] * vecs[i * dims + j];
      scores[i] = dot / 127;
    }
    return stats(scores);
  };

  for (const q of GOOD[corpus]) rows.push({ corpus, kind: 'good', q, ...(await run(q)) });
  for (const q of BAD) rows.push({ corpus, kind: 'bad', q, ...(await run(q)) });
}

const f = (x) => x.toFixed(3).padStart(6);
console.log('\ncorpus       kind  top1   median spread   z    zTop10  query');
for (const r of rows) {
  console.log(`${r.corpus.padEnd(12)} ${r.kind.padEnd(4)} ${f(r.top1)} ${f(r.median)} ${f(r.spread)} ${f(r.z)} ${f(r.zTop10)}  ${r.q.slice(0, 40)}`);
}

console.log('\n--- separation per statistic (worst good vs best bad, per corpus) ---');
for (const corpus of ['cv', 'publications', 'notes']) {
  const g = rows.filter((r) => r.corpus === corpus && r.kind === 'good');
  const b = rows.filter((r) => r.corpus === corpus && r.kind === 'bad');
  for (const stat of ['top1', 'spread', 'z', 'zTop10']) {
    const worstGood = Math.min(...g.map((r) => r[stat]));
    const bestBad = Math.max(...b.map((r) => r[stat]));
    const gap = worstGood - bestBad;
    console.log(`${corpus.padEnd(12)} ${stat.padEnd(7)} worst-good ${f(worstGood)}  best-bad ${f(bestBad)}  gap ${f(gap)} ${gap > 0 ? '  ✓ separable' : '  ✗ overlap'}`);
  }
}
