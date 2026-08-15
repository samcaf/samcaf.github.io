#!/usr/bin/env node
/**
 * build_graph.mjs — concept knowledge graph for the "Ask" assistant.
 *
 * The thesis ships its own ontology: 129 hand-written \newglossaryentry terms,
 * each with a name and a definition. So the graph is *extracted*, not guessed by
 * an LLM — no API calls, deterministic, and the vocabulary is the author's.
 *
 *   nodes  concepts (glossary entries + terms mined from the notes)
 *   edges  co-occurrence within a chunk, weighted by PMI so that "these two
 *          ideas genuinely travel together" beats "both are common"
 *   layout a deterministic 3D force-directed embedding, precomputed here so the
 *          browser only has to project and spin it
 *
 * Reads the chunk text already sitting in rag/*.json, so it never re-parses a
 * PDF. Writes rag/graph.json.
 *
 * Usage: node tools/build_graph.mjs [--no-fetch]
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);
const RAG = path.join(ROOT, 'rag');
const CACHE = path.join(TOOLS, '.cache');
const NO_FETCH = process.argv.includes('--no-fetch');

const GLOSSARY_URL = 'https://raw.githubusercontent.com/samcaf/Thesis/HEAD/includes/glossary.tex';

const CORPORA = ['cv', 'publications', 'notes'];

// A concept appearing in more than this fraction of chunks is a stopword in
// disguise ("accuracy", "energy") — it links everything to everything.
const MAX_DF = 0.10;
const MIN_MENTIONS = 3;      // ignore concepts the corpus barely uses
const MIN_EDGE_PMI = 0.30;   // weak association is noise at galaxy scale
const MAX_EDGES_PER_NODE = 8;

/* ---------------- inputs ---------------- */

async function glossary() {
  const dest = path.join(CACHE, 'thesis/glossary.tex');
  if (!existsSync(dest)) {
    if (NO_FETCH) throw new Error('glossary.tex not cached and --no-fetch given');
    mkdirSync(path.dirname(dest), { recursive: true });
    const res = await fetch(GLOSSARY_URL);
    if (!res.ok) throw new Error(`glossary fetch failed: ${res.status}`);
    writeFileSync(dest, await res.text());
  }
  const tex = readFileSync(dest, 'utf8');

  const out = [];
  const re = /\\newglossaryentry\s*\{([^}]+)\}\s*\{/g;
  let m;
  while ((m = re.exec(tex))) {
    // Walk braces to find the entry body (descriptions contain nested braces).
    let depth = 1;
    let i = re.lastIndex;
    while (i < tex.length && depth > 0) {
      if (tex[i] === '{') depth++;
      else if (tex[i] === '}') depth--;
      i++;
    }
    const body = tex.slice(re.lastIndex, i - 1);
    const name = (body.match(/\bname\s*=\s*\{?([^,}\n]+)/) || [])[1]?.trim();
    const desc = (body.match(/\bdescription\s*=\s*\{([\s\S]*?)\}\s*,?\s*$/m) || [])[1]
      ?? (body.match(/\bdescription\s*=\s*\{([\s\S]*)/) || [])[1];
    if (!name) continue;
    out.push({
      key: m[1],
      name: clean(name),
      description: clean(desc ?? '').replace(/\s+/g, ' ').trim(),
    });
  }
  return out;
}

function clean(s) {
  return String(s ?? '')
    .replace(/\\gls[a-z]*\s*\{([^}]*)\}/g, '$1')
    .replace(/\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?/g, ' ')
    .replace(/[{}$\\]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** The corpora are the source of truth for embedding config — never duplicate it. */
function embeddingConfig() {
  for (const corpus of CORPORA) {
    const file = path.join(RAG, `${corpus}.json`);
    if (existsSync(file)) return JSON.parse(readFileSync(file, 'utf8')).embedding;
  }
  throw new Error('no rag/*.json found — run tools/build_rag.mjs first');
}

function loadChunks() {
  const chunks = [];
  for (const corpus of CORPORA) {
    const file = path.join(RAG, `${corpus}.json`);
    if (!existsSync(file)) continue;
    const idx = JSON.parse(readFileSync(file, 'utf8'));
    // `li` is the chunk's index *within its own corpus* — the runtime searches
    // one corpus at a time, so global indices would be meaningless there.
    idx.chunks.forEach((c, li) => chunks.push({ ...c, corpus, li }));
  }
  return chunks;
}

/* ---------------- concept matching ---------------- */

const norm = (s) => s.toLowerCase().replace(/[‐-―]/g, '-').replace(/\s+/g, ' ');

/** Word-boundary matcher tolerant of plurals and hyphen/space variation. */
function matcher(name) {
  const esc = norm(name).replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/[-\s]+/g, '[-\\s]+');
  return new RegExp(`(?<![\\w-])${esc}(?:e?s)?(?![\\w-])`, 'g');
}

/**
 * Glossary names carry their acronym: "pileup (PU)", "Energy-Energy Correlator
 * (EEC)". Matching the literal string finds neither the bare term nor the
 * acronym as they actually appear in prose, so match on all three.
 */
function aliases(name) {
  const out = new Set([name]);
  const paren = name.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (paren) {
    if (paren[1].trim().length > 2) out.add(paren[1].trim());
    const acr = paren[2].trim();
    if (/^[A-Za-z][A-Za-z0-9-]{1,7}$/.test(acr)) out.add(acr);
  }
  return [...out];
}

function matchers(name) {
  return aliases(name).map(matcher);
}

/**
 * Terms the notes care about that the thesis glossary never names (TT̄, the
 * Grassmannian, the Witten index). Picked by "distinctive to one document":
 * frequent inside it, rare everywhere else.
 */
function mineTerms(chunks, existing) {
  const STOP = new Set(('the a an of in on for to and or is are be been we our this that these those it its as at by with from can may will '
    + 'which where when what how why not but if then than there here their they them he she his her one two three first second new old '
    + 'more most less least very much many some any all each other another such same different using used use also thus hence therefore '
    + 'however moreover furthermore given since while about into over under between among during before after above below case cases '
    + 'section chapter figure table equation appendix note notes example examples result results value values form forms term terms '
    + 'function functions number numbers order orders point points part parts way ways time times set sets').split(' '));
  const isWord = (w) => w.length > 2 && !STOP.has(w) && /^[a-z][a-z-]*$/.test(w);

  const perDoc = new Map();   // doc -> {corpus, terms: Map(term -> chunk count)}
  const globalDf = new Map(); // term -> number of docs containing it
  for (const c of chunks) {
    const doc = c.source.title;
    // Form bigrams only *within* a punctuation-delimited run, or "Santa Barbara,
    // Santa Barbara, CA" yields the phantom term "barbara santa".
    const grams = new Set();
    for (const seg of norm(c.text).split(/[^a-z0-9\s-]+/)) {
      const words = seg.split(/\s+/).filter(Boolean);
      for (let i = 0; i < words.length; i++) {
        if (isWord(words[i])) grams.add(words[i]);
        if (i + 1 < words.length && isWord(words[i]) && isWord(words[i + 1])
            && words[i].length >= 4 && words[i + 1].length >= 4) {
          grams.add(`${words[i]} ${words[i + 1]}`);
          // Trigrams too, else "infinite square well" fragments into the two
          // meaningless halves "infinite square" and "square well".
          if (i + 2 < words.length && isWord(words[i + 2]) && words[i + 2].length >= 4) {
            grams.add(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
          }
        }
      }
    }
    if (!perDoc.has(doc)) perDoc.set(doc, { corpus: c.corpus, terms: new Map() });
    const dm = perDoc.get(doc).terms;
    for (const g of grams) dm.set(g, (dm.get(g) ?? 0) + 1);
  }
  for (const { terms } of perDoc.values()) for (const g of terms.keys()) globalDf.set(g, (globalDf.get(g) ?? 0) + 1);

  const taken = new Set(existing.map((c) => norm(c.name)));
  const mined = [];
  for (const [doc, { corpus, terms }] of perDoc) {
    // Only the notes need mining: the thesis and papers are already covered by
    // the author's own glossary, and the CV has no concepts worth graphing.
    if (corpus !== 'notes') continue;
    const ranked = [...terms]
      .filter(([g, n]) => n >= 4 && globalDf.get(g) <= 2 && !taken.has(g) && g.includes(' '))
      .map(([g, n]) => [g, n / globalDf.get(g)])
      .sort((a, b) => b[1] - a[1]);
    // Prefer the longest phrase: drop any candidate contained in a better one.
    const chosen = [];
    for (const [g, score] of ranked) {
      if (chosen.some(([o]) => o.includes(g) || g.includes(o))) continue;
      chosen.push([g, score]);
      if (chosen.length >= 6) break;
    }
    for (const [g] of chosen) {
      taken.add(g);
      mined.push({ key: `mined:${g.replace(/\s+/g, '-')}`, name: g, description: '', mined: true, home: doc });
    }
  }
  return mined;
}

/* ---------------- graph ---------------- */

function buildGraph(concepts, chunks) {
  const hits = concepts.map(() => []);          // concept index -> chunk indices
  const res = concepts.map((c) => matchers(c.name));

  chunks.forEach((chunk, ci) => {
    const hay = norm(chunk.text);
    res.forEach((alts, i) => {
      for (const re of alts) {
        re.lastIndex = 0;
        if (re.test(hay)) { hits[i].push(ci); return; }
      }
    });
  });

  // Drop the too-rare and the too-generic.
  const keep = [];
  concepts.forEach((c, i) => {
    const df = hits[i].length;
    if (df < MIN_MENTIONS) return;
    if (df > MAX_DF * chunks.length) return;
    keep.push({ ...c, df, chunkIds: hits[i] });
  });

  // Co-occurrence with PMI weighting.
  const N = chunks.length;
  const byChunk = new Map();
  keep.forEach((c, i) => c.chunkIds.forEach((ci) => {
    if (!byChunk.has(ci)) byChunk.set(ci, []);
    byChunk.get(ci).push(i);
  }));

  const pair = new Map();
  for (const list of byChunk.values()) {
    for (let a = 0; a < list.length; a++) {
      for (let b = a + 1; b < list.length; b++) {
        const k = `${list[a]},${list[b]}`;
        pair.set(k, (pair.get(k) ?? 0) + 1);
      }
    }
  }

  let edges = [];
  for (const [k, co] of pair) {
    if (co < 2) continue;
    const [a, b] = k.split(',').map(Number);
    const pmi = Math.log((co / N) / ((keep[a].df / N) * (keep[b].df / N)));
    if (pmi < MIN_EDGE_PMI) continue;
    edges.push({ a, b, w: pmi, co });
  }

  // Keep each node's strongest links so the galaxy stays legible.
  const perNode = new Map();
  edges.sort((x, y) => y.w - x.w);
  const kept = [];
  for (const e of edges) {
    const na = perNode.get(e.a) ?? 0;
    const nb = perNode.get(e.b) ?? 0;
    if (na >= MAX_EDGES_PER_NODE && nb >= MAX_EDGES_PER_NODE) continue;
    perNode.set(e.a, na + 1);
    perNode.set(e.b, nb + 1);
    kept.push(e);
  }
  return { nodes: keep, edges: kept };
}

/** Dominant document for a concept — used for clustering colour. */
function assignHomes(nodes, chunks) {
  const dominant = (counts) => {
    let best = null;
    let bestN = 0;
    for (const [k, c] of counts) if (c > bestN) { best = k; bestN = c; }
    return best;
  };
  for (const n of nodes) {
    const docs = new Map();
    const corpora = new Map();
    for (const ci of n.chunkIds) {
      const t = chunks[ci].source.title;
      docs.set(t, (docs.get(t) ?? 0) + 1);
      corpora.set(chunks[ci].corpus, (corpora.get(chunks[ci].corpus) ?? 0) + 1);
    }
    // Where a concept *mostly* lives — not merely where it was first seen.
    n.home = dominant(docs) ?? n.home ?? null;
    n.corpus = dominant(corpora) ?? 'publications';
  }
}

/* ---------------- concept embeddings ---------------- */

/**
 * Embed each concept ("name. definition") in the same space as the passages, so
 * the runtime can ask "is this concept what the question is about?" directly.
 * Without this, concepts light by mere co-occurrence and generic entries like
 * "algorithm" glow for every query that happens to touch a passage naming them.
 */
async function embedConcepts(nodes, EMBED) {
  const { env, pipeline } = await import('@huggingface/transformers');
  env.cacheDir = path.join(CACHE, 'models');
  const ex = await pipeline('feature-extraction', EMBED.model, { dtype: EMBED.dtype });

  const texts = nodes.map((n) => (n.description ? `${n.name}. ${n.description}` : n.name));
  const arr = new Int8Array(nodes.length * EMBED.dims);
  for (let i = 0; i < texts.length; i += 32) {
    const out = await ex(texts.slice(i, i + 32), { pooling: EMBED.pooling, normalize: EMBED.normalize });
    const [n, d] = out.dims;
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < d; k++) {
        arr[(i + j) * d + k] = Math.max(-127, Math.min(127, Math.round(out.data[j * d + k] * 127)));
      }
    }
  }
  return arr;
}

/* ---------------- deterministic 3D layout ---------------- */

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function layout3d(nodes, edges, { iterations = 400 } = {}) {
  const rand = mulberry32(20260815);
  const n = nodes.length;
  const pos = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {          // start on a sphere
    const u = rand() * 2 - 1;
    const th = rand() * Math.PI * 2;
    const r = Math.sqrt(1 - u * u);
    pos[i * 3] = r * Math.cos(th);
    pos[i * 3 + 1] = r * Math.sin(th);
    pos[i * 3 + 2] = u;
  }

  const deg = new Float64Array(n);
  for (const e of edges) { deg[e.a] += e.w; deg[e.b] += e.w; }

  const disp = new Float64Array(n * 3);
  const K = 0.9;
  for (let it = 0; it < iterations; it++) {
    disp.fill(0);
    const temp = 0.10 * (1 - it / iterations) + 0.002;

    for (let i = 0; i < n; i++) {          // repulsion
      for (let j = i + 1; j < n; j++) {
        let dx = pos[i * 3] - pos[j * 3];
        let dy = pos[i * 3 + 1] - pos[j * 3 + 1];
        let dz = pos[i * 3 + 2] - pos[j * 3 + 2];
        let d2 = dx * dx + dy * dy + dz * dz + 1e-6;
        const f = (K * K) / d2;
        const d = Math.sqrt(d2);
        dx /= d; dy /= d; dz /= d;
        disp[i * 3] += dx * f; disp[i * 3 + 1] += dy * f; disp[i * 3 + 2] += dz * f;
        disp[j * 3] -= dx * f; disp[j * 3 + 1] -= dy * f; disp[j * 3 + 2] -= dz * f;
      }
    }
    for (const e of edges) {               // attraction along edges
      const i = e.a, j = e.b;
      let dx = pos[i * 3] - pos[j * 3];
      let dy = pos[i * 3 + 1] - pos[j * 3 + 1];
      let dz = pos[i * 3 + 2] - pos[j * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
      const f = (d * d) / K * Math.min(2, e.w);
      dx = (dx / d) * f; dy = (dy / d) * f; dz = (dz / d) * f;
      disp[i * 3] -= dx; disp[i * 3 + 1] -= dy; disp[i * 3 + 2] -= dz;
      disp[j * 3] += dx; disp[j * 3 + 1] += dy; disp[j * 3 + 2] += dz;
    }
    for (let i = 0; i < n; i++) {          // step + mild gravity
      let dx = disp[i * 3], dy = disp[i * 3 + 1], dz = disp[i * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;
      const s = Math.min(d, temp) / d;
      pos[i * 3] += dx * s - pos[i * 3] * 0.012;
      pos[i * 3 + 1] += dy * s - pos[i * 3 + 1] * 0.012;
      pos[i * 3 + 2] += dz * s - pos[i * 3 + 2] * 0.012;
    }
  }

  // Normalise into the unit sphere, flattened slightly so it reads as a galaxy.
  let max = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    if (d > max) max = d;
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push([
      +(pos[i * 3] / max).toFixed(4),
      +(pos[i * 3 + 1] / max * 0.72).toFixed(4),
      +(pos[i * 3 + 2] / max).toFixed(4),
    ]);
  }
  return out;
}

function components(nodes, edges) {
  const parent = nodes.map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (const e of edges) { const a = find(e.a), b = find(e.b); if (a !== b) parent[a] = b; }
  const seen = new Map();
  nodes.forEach((_, i) => {
    const r = find(i);
    if (!seen.has(r)) seen.set(r, seen.size);
  });
  return { count: seen.size, of: nodes.map((_, i) => seen.get(find(i))) };
}

/* ---------------- main ---------------- */

const gloss = await glossary();
console.log(`glossary: ${gloss.length} entries`);

const chunks = loadChunks();
console.log(`chunks: ${chunks.length} across ${CORPORA.length} corpora`);

const mined = mineTerms(chunks, gloss);
console.log(`mined ${mined.length} note-specific terms`);
console.log('  e.g.', mined.slice(0, 12).map((m) => m.name).join(' | '));

// The glossary occasionally defines the same term under two keys; one star per
// concept, keeping whichever entry carries the fuller definition.
const merged = [];
const byName = new Map();
for (const c of [...gloss, ...mined]) {
  const k = c.name.toLowerCase().trim();
  const prev = byName.get(k);
  if (!prev) { byName.set(k, c); merged.push(c); continue; }
  if ((c.description?.length ?? 0) > (prev.description?.length ?? 0)) prev.description = c.description;
}
if (merged.length !== gloss.length + mined.length) {
  console.log(`  merged ${gloss.length + mined.length - merged.length} duplicate concept name(s)`);
}

const { nodes, edges } = buildGraph(merged, chunks);
assignHomes(nodes, chunks);
const comp = components(nodes, edges);
console.log(`graph: ${nodes.length} nodes, ${edges.length} edges, ${comp.count} components`);

const dropped = gloss.length + mined.length - nodes.length;
console.log(`  dropped ${dropped} (too rare < ${MIN_MENTIONS} mentions, or too generic > ${(MAX_DF * 100).toFixed(0)}% of chunks)`);
console.log('  busiest:', [...nodes].sort((a, b) => b.df - a.df).slice(0, 8).map((n) => `${n.name}(${n.df})`).join(', '));

console.log('embedding concepts…');
const EMBED = embeddingConfig();
const conceptVecs = await embedConcepts(nodes, EMBED);
writeFileSync(path.join(RAG, 'graph.vec'), Buffer.from(conceptVecs.buffer, conceptVecs.byteOffset, conceptVecs.byteLength));

console.log('laying out…');
const pos = layout3d(nodes, edges);

const out = {
  version: 2,
  built_from: { glossary: GLOSSARY_URL, corpora: CORPORA },
  embedding: EMBED,
  vectors: { file: 'graph.vec', encoding: 'int8', dims: EMBED.dims },
  nodes: nodes.map((n, i) => ({
    id: n.key,
    name: n.name,
    def: n.description || undefined,
    mined: n.mined || undefined,
    home: n.home,
    corpus: n.corpus,
    df: n.df,
    group: comp.of[i],
    p: pos[i],
    // { corpus: [local chunk indices] } so a page can map its own hits to
    // concepts. `all` holds indices into the unified index, which concatenates
    // the corpora in this same CORPORA order — keep the two in step.
    chunks: n.chunkIds.reduce((acc, ci) => {
      const { corpus, li } = chunks[ci];
      (acc[corpus] ??= []).push(li);
      (acc.all ??= []).push(ci);
      return acc;
    }, {}),
  })),
  edges: edges.map((e) => ({ a: e.a, b: e.b, w: +e.w.toFixed(3) })),
};
mkdirSync(RAG, { recursive: true });
const file = path.join(RAG, 'graph.json');
writeFileSync(file, JSON.stringify(out));
console.log(`wrote rag/graph.json — ${(statSync(file).size / 1024).toFixed(0)} KB`);
