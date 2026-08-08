#!/usr/bin/env node
/**
 * query_rag.mjs — retrieval smoke test against a built index.
 * Mirrors the browser runtime exactly: same model, dtype, pooling,
 * query prefix, and int8 scoring.
 *
 * Usage: node tools/query_rag.mjs rag/publications.json "what is PIRANHA?"
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const [file, ...q] = process.argv.slice(2);
const query = q.join(' ');
if (!file || !query) {
  console.error('usage: node tools/query_rag.mjs rag/<corpus>.json "question"');
  process.exit(1);
}

const index = JSON.parse(readFileSync(path.resolve(file), 'utf8'));
const { model, dims, dtype, pooling, normalize, query_prefix } = index.embedding;

const { env, pipeline } = await import('@huggingface/transformers');
env.cacheDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '.cache', 'models');
const ex = await pipeline('feature-extraction', model, { dtype });
const out = await ex([query_prefix + query], { pooling, normalize });
const qv = out.data;

const raw = Buffer.from(index.vectors, 'base64');
const vecs = new Int8Array(raw.buffer, raw.byteOffset, raw.length);

const scored = index.chunks.map((chunk, i) => {
  let dot = 0;
  for (let j = 0; j < dims; j++) dot += qv[j] * vecs[i * dims + j];
  return { chunk, score: dot / 127 };
});
scored.sort((a, b) => b.score - a.score);

for (const { chunk, score } of scored.slice(0, 5)) {
  const s = chunk.source;
  console.log(`\n[${score.toFixed(3)}] ${s.title}${s.loc ? ` (${s.loc})` : ''} — ${s.url}`);
  console.log('  ' + chunk.text.slice(0, 240).replace(/\s+/g, ' '));
}
