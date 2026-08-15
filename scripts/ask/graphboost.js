/**
 * graphboost.js — GraphRAG "local search" over the concept graph.
 *
 * Plain vector search ranks passages in isolation. This adds the author's own
 * ontology back in: work out which concepts the question is about (directly by
 * name, and indirectly via the best-matching passages), expand one hop through
 * the co-occurrence graph, then nudge passages that carry those concepts.
 *
 * The nudge is deliberately small — semantic similarity still decides the
 * ranking, the graph only breaks ties between passages that are already close.
 *
 * No DOM, no fetch: tools/eval_rag.mjs imports this too, so the A/B measures
 * exactly what ships.
 */

const WIDE = 24;        // passages consulted for concept activation
const MAX_ACTIVE = 10;   // concepts lit at once
const MIN_SUPPORT = 1.2; // ≈ named in the query, or present in 2+ top passages
/*
 * Measured concept-similarity distributions (bge-small, this graph):
 *   on-point   "optical theorem" → 0.863, next best 0.667
 *              "optimal transport in grooming" → PIRANHA 0.705, continuous grooming 0.703
 *   no match   "supersymmetric quantum mechanics" → best is 0.673 (conformal
 *              symmetry, QCD) — all wrong; the glossary has no SUSY-QM concept
 * So an on-point concept clears ~0.70 while the best-of-a-bad-lot sits ~0.67.
 * The floor is absolute on purpose: a relative-to-best rule always keeps a
 * winner, however irrelevant, which is exactly the "lights up when nothing is
 * relevant" failure. Fitted to a handful of probes — retune with the sweep in
 * tools/eval_gate.mjs if the galaxy goes dark too often.
 */
const SIM_FLOOR = 0.68;
const SIM_MARGIN = 0.12; // among concepts that clear the floor, keep the close ones
const BOOST = 0.012;    // score added per activated concept found in a passage
const MAX_BOOST = 3;    // ...counted at most this many times

/** Index a raw graph.json for lookup. Idempotent, field by field — a partly
 *  prepared graph (e.g. neighbours built elsewhere) must still get the rest. */
export function prepareGraph(g) {
  if (!g.neighbors) {
    g.neighbors = g.nodes.map(() => []);
    g.edges.forEach((e) => { g.neighbors[e.a].push(e.b); g.neighbors[e.b].push(e.a); });
  }
  if (!g._chunkMap) g._chunkMap = new Map();
  return g;
}

const norm = (s) => String(s).toLowerCase().replace(/[‐-―]/g, '-').replace(/\s+/g, ' ');

/** Names the concept answers to: "pileup (PU)" also answers to "pileup", "PU". */
function aliases(name) {
  const out = [name];
  const paren = name.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (paren) {
    if (paren[1].trim().length > 2) out.push(paren[1].trim());
    if (/^[A-Za-z][A-Za-z0-9-]{1,7}$/.test(paren[2].trim())) out.push(paren[2].trim());
  }
  return out.map(norm);
}

/** chunk index -> concept indices, for one corpus (cached per graph). */
function chunkConcepts(graph, corpus) {
  if (graph._chunkMap.has(corpus)) return graph._chunkMap.get(corpus);
  const map = new Map();
  graph.nodes.forEach((n, i) => {
    for (const li of n.chunks?.[corpus] ?? []) {
      if (!map.has(li)) map.set(li, []);
      map.get(li).push(i);
    }
  });
  graph._chunkMap.set(corpus, map);
  return map;
}

/**
 * Cosine of the query against every concept's own embedding, or null when the
 * graph predates concept vectors (older graph.json — degrade, don't crash).
 */
export function conceptSimilarity(graph, queryVec) {
  if (!queryVec || !graph.vecs) return null;
  const dims = graph.vectors?.dims ?? 384;
  const out = new Float32Array(graph.nodes.length);
  for (let i = 0; i < graph.nodes.length; i++) {
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += queryVec[j] * graph.vecs[i * dims + j];
    out[i] = dot / 127;
  }
  return out;
}

/** Concepts the question names outright. */
export function conceptsFromQuery(graph, query) {
  const q = ` ${norm(query).replace(/[^\w\s-]/g, ' ')} `;
  const found = [];
  graph.nodes.forEach((n, i) => {
    for (const a of aliases(n.name)) {
      if (a.length < 4) continue;                      // "PU" would match anything
      if (q.includes(` ${a} `) || q.includes(` ${a}s `)) { found.push(i); return; }
    }
  });
  return found;
}

/**
 * Concepts this question is about: named in the query, or carried by the
 * passages that already rank highest. Ordered by how much support they have.
 */
export function activate(graph, corpus, rankedIndices, query, { max = MAX_ACTIVE, queryVec = null } = {}) {
  const map = chunkConcepts(graph, corpus);
  const score = new Map();
  const bump = (i, by) => score.set(i, (score.get(i) ?? 0) + by);

  const named = new Set(conceptsFromQuery(graph, query));
  for (const i of named) bump(i, 3);
  rankedIndices.slice(0, WIDE).forEach((li, rank) => {
    for (const ci of map.get(li) ?? []) bump(ci, 1 - rank / (WIDE * 2));
  });

  /*
   * Co-occurrence is not relevance. Generic entries ("algorithm", "limit",
   * "region") appear in so many passages that they light for any question that
   * happens to retrieve one. Concepts carry their own embedding, so require the
   * concept itself to be about the question. A concept the query named outright
   * is exempt — that is relevance by definition.
   */
  const sims = conceptSimilarity(graph, queryVec);
  if (sims) {
    let bestSim = 0;
    for (const [i] of score) if (sims[i] > bestSim) bestSim = sims[i];
    const cutoff = Math.max(SIM_FLOOR, bestSim - SIM_MARGIN);
    for (const [i] of [...score]) {
      if (named.has(i)) continue;
      if (sims[i] < cutoff) score.delete(i);
      else score.set(i, score.get(i) * (0.5 + sims[i]));   // prefer the closest
    }
  }

  // Demand real support: named in the query, or carried by 2+ top passages.
  // Better a dark galaxy than a confidently wrong one.
  return [...score.entries()]
    .filter(([, s]) => s >= MIN_SUPPORT)
    .sort((a, b) => b[1] - a[1])
    .slice(0, max)
    .map(([i]) => i);
}

/** Nudge passages carrying activated concepts; returns a new sorted array. */
export function boostRanking(graph, corpus, ranked, activeConcepts) {
  if (!activeConcepts.length) return ranked;
  const map = chunkConcepts(graph, corpus);
  const active = new Set(activeConcepts);
  return ranked
    .map((hit) => {
      let n = 0;
      for (const ci of map.get(hit.i) ?? []) if (active.has(ci)) n++;
      return n ? { ...hit, score: hit.score + BOOST * Math.min(MAX_BOOST, n), boosted: n } : hit;
    })
    .sort((a, b) => b.score - a.score);
}
