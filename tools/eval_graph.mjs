#!/usr/bin/env node
/**
 * eval_graph.mjs — A/B: plain vector search vs graph-boosted search.
 *
 * Uses the same graphboost.js the browser ships, so the numbers describe what
 * visitors actually get. Reports, per query, how the displayed set changes and
 * which concepts were activated.
 *
 * Usage: node tools/eval_graph.mjs [corpus]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { prepareGraph, activate, boostRanking, conceptsFromQuery } from '../scripts/ask/graphboost.js';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);

const FLOOR = 0.42, MARGIN = 0.10, PER_DOC = 2, K = 4;

// Relational and multi-hop questions — where a concept graph should earn its
// keep. Plain vector search already handles "what is X?" well.
const QUERIES = {
  publications: [
    'How does jet grooming relate to energy correlators?',
    'What does pileup do to jet substructure observables?',
    'How is optimal transport used in grooming?',
    'What is the connection between parton showers and resummation?',
    'How do energy correlators probe hadronization?',
    'What is PIRANHA?',
    'What are energy correlators?',
  ],
  notes: [
    'How does unitarity constrain scattering amplitudes?',
    'What is the relationship between supersymmetry and the harmonic oscillator?',
    'How does the cosmological collider relate to inflation?',
  ],
};

function pick(ranked) {
  const best = ranked[0]?.score ?? 0;
  if (best < FLOOR) return [];
  const cutoff = Math.max(FLOOR, best - MARGIN);
  const perDoc = new Map();
  const out = [];
  for (const h of ranked) {
    if (h.score < cutoff) break;
    const doc = h.chunk.source.url.split('#')[0];
    const n = perDoc.get(doc) ?? 0;
    if (n >= PER_DOC) continue;
    perDoc.set(doc, n + 1);
    out.push(h);
    if (out.length >= K) break;
  }
  return out;
}

const corpus = process.argv[2] ?? 'publications';
const index = JSON.parse(readFileSync(path.join(ROOT, 'rag', `${corpus}.json`), 'utf8'));
const bin = readFileSync(path.join(ROOT, 'rag', index.vectors.file));
const vecs = new Int8Array(bin.buffer, bin.byteOffset, bin.byteLength);
const graph = prepareGraph(JSON.parse(readFileSync(path.join(ROOT, 'rag', 'graph.json'), 'utf8')));

const { env, pipeline } = await import('@huggingface/transformers');
env.cacheDir = path.join(TOOLS, '.cache', 'models');
const ex = await pipeline('feature-extraction', index.embedding.model, { dtype: index.embedding.dtype });

const { dims, pooling, normalize, query_prefix } = index.embedding;
let changed = 0;
let total = 0;

console.log(`\n=== ${corpus}: plain vs graph-boosted (${index.count} chunks, ${graph.nodes.length} concepts) ===\n`);

for (const q of QUERIES[corpus] ?? []) {
  const out = await ex([(query_prefix ?? '') + q], { pooling, normalize });
  const qv = out.data;
  const ranked = index.chunks.map((c, i) => {
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += qv[j] * vecs[i * dims + j];
    return { i, chunk: c, score: dot / 127 };
  }).sort((a, b) => b.score - a.score);

  const plain = pick(ranked);
  const concepts = activate(graph, corpus, ranked.map((h) => h.i), q);
  const boosted = pick(boostRanking(graph, corpus, ranked, concepts));

  const idsA = plain.map((h) => h.i).join(',');
  const idsB = boosted.map((h) => h.i).join(',');
  total++;
  if (idsA !== idsB) changed++;

  const named = conceptsFromQuery(graph, q).map((i) => graph.nodes[i].name);
  console.log(`Q: ${q}`);
  console.log(`   named in query : ${named.length ? named.join(', ') : '(none)'}`);
  console.log(`   activated      : ${concepts.slice(0, 8).map((i) => graph.nodes[i].name).join(', ') || '(none)'}`);
  console.log(`   result set     : ${idsA === idsB ? 'unchanged' : 'CHANGED'}`);
  if (idsA !== idsB) {
    const label = (h) => `${h.chunk.source.title.slice(0, 34)} · ${h.chunk.source.loc ?? ''}`.trim();
    console.log(`     plain : ${plain.map(label).join(' | ')}`);
    console.log(`     graph : ${boosted.map((h) => label(h) + (h.boosted ? ` (+${h.boosted})` : '')).join(' | ')}`);
  }
  console.log();
}
console.log(`${changed}/${total} result sets changed.`);
