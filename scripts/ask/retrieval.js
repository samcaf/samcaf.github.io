/**
 * retrieval.js — client-side semantic search over a prebuilt index.
 *
 * Everything runs on the visitor's device: the index is a static JSON file,
 * the embedder is a vendored ONNX model. No network calls after load, and
 * no question ever leaves the browser.
 *
 * The index carries its own embedding config (model, dtype, pooling, query
 * prefix), so query vectors are guaranteed to match the corpus vectors that
 * tools/build_rag.mjs produced.
 */

const VENDOR = new URL('./vendor/', import.meta.url).href;
const MODELS = new URL('../../assets/models/', import.meta.url).href;

// Retrieval cutoffs — see search() for why the relative margin matters.
// FLOOR is only a backstop: each index carries its own `noise_floor`, measured
// at build time as the best score that corpus gives a deliberately irrelevant
// question. That has to be per-corpus — see tools/eval_gate.mjs.
const FLOOR = 0.42;
const MARGIN = 0.10;         // keep hits within this much of the best one
const PER_DOC = 2;           // at most this many excerpts from any single document
const REJECT_MARGIN = 0.005; // at or below its corpus's noise floor: not an answer
const WEAK_MARGIN = 0.05;    // cleared the floor, but only just — say so

let embedderPromise = null;

/** Fetch an index and decode its int8 vectors into a typed array. */
export async function loadIndex(url) {
  // Default caching: the browser revalidates via ETag, so a rebuilt index is
  // picked up without shipping a stale copy forever.
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load index (${res.status})`);
  const index = await res.json();

  // Vectors live in a sibling .vec file: raw int8, ~33% smaller than base64
  // inline and parsed in one step instead of char-by-char.
  const vecUrl = new URL(index.vectors.file, new URL(url, location.href)).href;
  const vecRes = await fetch(vecUrl);
  if (!vecRes.ok) throw new Error(`could not load vectors (${vecRes.status})`);
  index.vecs = new Int8Array(await vecRes.arrayBuffer());

  const expected = index.count * index.embedding.dims;
  if (index.vecs.length !== expected) {
    throw new Error(`index/vector mismatch: ${index.vecs.length} != ${expected} (stale build?)`);
  }
  return index;
}

/**
 * Load the embedding model. Reports coarse progress via onProgress(fraction,
 * label) so the UI can show a bar for the one-time download.
 */
export function loadEmbedder(index, onProgress) {
  if (embedderPromise) return embedderPromise;

  embedderPromise = (async () => {
    const { env, pipeline } = await import(VENDOR + 'transformers.min.js');

    // Self-hosted: model weights and the ONNX runtime ship with the site.
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = MODELS;
    env.backends.onnx.wasm.wasmPaths = VENDOR + 'ort/';
    env.backends.onnx.wasm.numThreads = 1; // no cross-origin isolation on Pages

    const files = new Map();
    const cb = (p) => {
      if (!onProgress) return;
      if (p.status === 'progress' && p.total) files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      if (total) onProgress(loaded / total, `${(loaded / 1048576).toFixed(0)} of ${(total / 1048576).toFixed(0)} MB`);
    };

    const { model, dtype } = index.embedding;
    const name = model.includes('/') ? model.split('/').pop() : model;
    return pipeline('feature-extraction', name, { dtype, progress_callback: cb });
  })();

  return embedderPromise;
}

export function isEmbedderReady() {
  return embedderPromise !== null;
}

/** Embed arbitrary passages with the already-loaded search model. */
export async function embedTexts(index, texts) {
  const embedder = await loadEmbedder(index);
  const { pooling, normalize } = index.embedding;
  const out = await embedder(texts, { pooling, normalize });
  const [n, d] = out.dims;
  return Array.from({ length: n }, (_, i) => out.data.slice(i * d, (i + 1) * d));
}

/**
 * Embed a query and return the top-k chunks by cosine similarity.
 *
 * Returns the displayed `hits` plus the full `ranked` list, because concept
 * activation reads deeper than the four excerpts a visitor sees.
 */
export async function search(index, query, k = 4) {
  const embedder = await loadEmbedder(index);
  const { dims, pooling, normalize, query_prefix } = index.embedding;

  const out = await embedder([(query_prefix || '') + query], { pooling, normalize });
  const q = out.data;

  /*
   * Score each chunk against the noise floor of the corpus it came from, and
   * rank by that margin rather than the raw cosine.
   *
   * The unified index is heterogeneous: the CV is small and homogeneous so
   * everything in it scores low (floor 0.557), while the notes score high
   * (0.607). Comparing raw cosines across them is apples to oranges — a global
   * floor would either reject real CV questions or admit nonsense on the notes.
   * The margin puts every source on the same footing.
   */
  const floors = index.noise_floors ?? null;
  const globalFloor = Math.max(FLOOR, index.noise_floor ?? 0);
  const floorOf = (chunk) => (floors ? (floors[chunk.c] ?? globalFloor) : globalFloor);

  const scored = new Array(index.count);
  for (let i = 0; i < index.count; i++) {
    let dot = 0;
    const base = i * dims;
    for (let j = 0; j < dims; j++) dot += q[j] * index.vecs[base + j];
    const chunk = index.chunks[i];
    const score = dot / 127;
    scored[i] = { i, chunk, score, margin: score - floorOf(chunk) };
  }
  scored.sort((a, b) => b.margin - a.margin);

  // Cutoff is *relative* to the best hit, not a fixed floor. BGE cosines are
  // compressed high — genuinely irrelevant passages still score ~0.5 — so an
  // absolute threshold either admits everything or nothing. Keeping only what
  // is close to the top hit is what stops "vaguely same field" results.
  /*
   * Reject only what clearly missed; flag what barely cleared.
   *
   * Measured margins (tools/, unified index) overlap: "how do I tune a guitar"
   * scores 0.049 — the papers discuss *tuning* — while the perfectly legitimate
   * "do you have teaching experience?" scores 0.012, because the CV is small and
   * telegraphic. No single cutoff separates them, so pretending one exists would
   * either lose real answers or wave nonsense through. Everything in between is
   * returned and labelled a loose match.
   */
  const best = scored[0]?.margin ?? -1;
  if (best <= REJECT_MARGIN) return result([], scored, q);
  const cutoff = Math.max(0, best - MARGIN);

  const perDoc = new Map();
  const picked = [];
  for (const hit of scored) {
    if (hit.margin < cutoff) break;
    const doc = hit.chunk.source.url.split('#')[0];
    const n = perDoc.get(doc) || 0;
    if (n >= PER_DOC) continue; // don't let one long PDF fill the whole list
    perDoc.set(doc, n + 1);
    picked.push(hit);
    if (picked.length >= k) break;
  }
  return result(picked, scored, q, best < WEAK_MARGIN);
}

/**
 * The result is an *array of hits* that also carries `.hits` and `.ranked`.
 *
 * Browsers cache ES modules independently, so a visitor can end up with a fresh
 * retrieval.js beside a stale ask.js. When this returned a bare object, that
 * pairing read `.length` off an object, got undefined, and reported "No close
 * match" for every query — a silent, unrecoverable failure. Staying array-like
 * means an out-of-date caller still gets its results; only the newer features
 * go missing. Safe to simplify once no stale copies can be in flight.
 */
function result(hits, ranked, queryVec, weak = false) {
  const out = hits.slice();
  out.hits = hits;
  out.ranked = ranked;
  out.queryVec = queryVec;
  out.weak = weak;   // cleared the noise floor only narrowly — surface the doubt
  return out;
}
