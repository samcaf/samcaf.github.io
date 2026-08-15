#!/usr/bin/env node
/**
 * eval_rag.mjs — retrieval quality probe for rag/*.json.
 *
 * Prints the ranked hits for a set of queries, plus the score spread. A healthy
 * index separates the relevant chunk from the rest; a flat spread means the
 * index is too small or the chunks are too coarse to discriminate.
 *
 * Usage: node tools/eval_rag.mjs [corpus] ["custom query"]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);

const QUERIES = {
  cv: [
    'What did your PhD research cover?',
    'What programming languages do you use?',
    'Where did you go to graduate school?',
    'Do you have teaching experience?',
  ],
  publications: [
    'What is PIRANHA?',
    'What are energy correlators?',
    'qubit noise and decoherence',
    'long-lived particles at colliders',
  ],
  notes: [
    'What does the optical theorem say?',
    'What is the TT-bar deformation?',
    'supersymmetric quantum mechanics and the Witten index',
    'inflation and the cosmological collider',
    'Grassmann numbers and superspace',
  ],
};

async function main() {
  const corpus = process.argv[2] ?? 'publications';
  const custom = process.argv[3];
  const index = JSON.parse(readFileSync(path.join(ROOT, 'rag', `${corpus}.json`), 'utf8'));
  const { model, dims, dtype, pooling, normalize, query_prefix } = index.embedding;

  const bin = typeof index.vectors === 'string'
    ? Buffer.from(index.vectors, 'base64')                                  // v1 inline
    : readFileSync(path.join(ROOT, 'rag', index.vectors.file));             // v2 sidecar
  const vecs = new Int8Array(bin.buffer, bin.byteOffset, bin.byteLength);

  // Mirror the runtime cutoff so the eval shows what a visitor would actually see.
  const FLOOR = 0.42, MARGIN = 0.10, PER_DOC = 2;

  const { env, pipeline, AutoTokenizer } = await import('@huggingface/transformers');
  env.cacheDir = path.join(TOOLS, '.cache', 'models');
  const ex = await pipeline('feature-extraction', model, { dtype });
  const tok = await AutoTokenizer.from_pretrained(model);

  // How many chunks exceed the encoder's 512-token window (silently truncated)?
  const lens = index.chunks.map((c) => tok.encode(c.text).length);
  const over = lens.filter((n) => n > 512).length;
  const near = lens.filter((n) => n > 400).length;
  console.log(`\n=== ${corpus}: ${index.count} chunks ===`);
  console.log(`tokens/chunk: p50 ${lens.slice().sort((a, b) => a - b)[Math.floor(lens.length / 2)]}, max ${Math.max(...lens)}`);
  console.log(`truncated (>512 tok): ${over} chunks (${(100 * over / lens.length).toFixed(0)}%) | >400 tok: ${near}\n`);

  for (const q of custom ? [custom] : QUERIES[corpus]) {
    const out = await ex([(query_prefix ?? '') + q], { pooling, normalize });
    const qv = out.data;
    const scored = index.chunks.map((c, i) => {
      let dot = 0;
      for (let j = 0; j < dims; j++) dot += qv[j] * vecs[i * dims + j];
      return { c, s: dot / 127 };
    }).sort((a, b) => b.s - a.s);

    const top = scored.slice(0, 6);
    const median = scored[Math.floor(scored.length / 2)].s;
    const cutoff = Math.max(FLOOR, top[0].s - MARGIN);
    console.log(`Q: ${q}`);
    console.log(`   top1 ${top[0].s.toFixed(3)} | median ${median.toFixed(3)} | spread ${(top[0].s - median).toFixed(3)} | cutoff ${cutoff.toFixed(3)}`);
    const perDoc = new Map();
    for (const [i, h] of top.entries()) {
      const doc = h.c.source.url.split('#')[0];
      const n = perDoc.get(doc) ?? 0;
      let shown = h.s >= cutoff && n < PER_DOC;
      if (shown) perDoc.set(doc, n + 1);
      const where = [h.c.source.title, h.c.source.loc].filter(Boolean).join(' · ');
      console.log(`   ${shown ? '✓' : ' '} ${i + 1}. ${h.s.toFixed(3)}  ${where.slice(0, 46).padEnd(46)} ${h.c.text.replace(/\s+/g, ' ').slice(0, 58)}…`);
    }
    console.log();
  }
}

main();
