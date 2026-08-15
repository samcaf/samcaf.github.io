/**
 * extractive.js — synthesis with nothing to download.
 *
 * The neural tier needs hundreds of megabytes of weights for prose a 0.5B model
 * often gets wrong anyway. This does the useful half with what is already in
 * memory: the search embedder scores each *sentence* of the retrieved passages
 * against the question, and the best few become the answer.
 *
 * It is extractive, so every word is the author's and a citation is exact by
 * construction — it cannot hallucinate, only fail to find. That makes it the
 * right default; the language model is an upgrade for phrasing, not for truth.
 */

import { embedTexts } from './retrieval.js';

const MIN_CHARS = 45;
const MAX_CHARS = 320;

/** Sentence split that tolerates the abbreviations physics prose is full of. */
function sentences(text) {
  const ABBR = /(?:[A-Z]|Eq|Eqs|Fig|Figs|Ref|Refs|Sec|Ch|Chap|Tab|App|Dr|Prof|vs|cf|etc|al|e\.g|i\.e|Phys|Rev|Lett|JHEP|No|pp)$/;
  const out = [];
  let start = 0;
  const re = /([.!?])["')\]]*\s+/g;
  let m;
  while ((m = re.exec(text))) {
    const lastWord = text.slice(start, m.index).split(/[\s(]/).pop() ?? '';
    if (m[1] === '.' && ABBR.test(lastWord)) continue;
    if (!/[A-Z(“"'$]/.test(text[re.lastIndex] ?? '')) continue;
    out.push(text.slice(start, re.lastIndex).trim());
    start = re.lastIndex;
  }
  const tail = text.slice(start).trim();
  if (tail) out.push(tail);
  return out;
}

/** Strip the source's own reference markers so quotes read cleanly. */
const tidy = (s) => s
  .replace(/^\s*(?:Abstract|Summary)\s*[:.]\s*/i, '')   // the label is not part of the sentence
  .replace(/\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]/g, '')
  .replace(/\(\s*\)/g, '')
  .replace(/\s+([.,;:!?])/g, '$1')     // "GitHub ." once the citation is gone
  .replace(/\s{2,}/g, ' ')
  .trim();

/**
 * Typeset acronyms come out of the PDF letter-spaced — "P ileup and I nfrared
 * R adiation A n N i H il A tion" is one sentence in the thesis. It reads as
 * gibberish when quoted, so drop anything with a suspicious run of stray single
 * letters.
 */
function isGarbled(s) {
  const words = s.split(/\s+/);
  if (words.length < 6) return false;
  const singles = words.filter((w) => /^[A-Za-z]$/.test(w)).length;
  return singles / words.length > 0.12;
}

const FUNCTION_WORDS = new Set(('the a an of in on at to and or is are was were be been for with from that this these those '
  + 'which who whose it its as by we our they their can may will would could should not but if then than there').split(' '));

/**
 * Prose, not a list. Card blurbs and nav strips retrieve well ("Energy
 * correlators Monte-Carlo Data visualization C++ / Python GitHub arXiv:…") but
 * are unreadable as an answer. Real sentences are full of function words;
 * keyword soup has almost none.
 */
function isSentence(s) {
  const words = s.toLowerCase().split(/\W+/).filter(Boolean);
  if (words.length < 8) return false;
  const fn = words.filter((w) => FUNCTION_WORDS.has(w)).length;
  return fn >= 3 && fn / words.length >= 0.10;
}

/**
 * Pick the sentences that actually answer the question.
 * Returns [{ text, n }] where n is the 1-based number of the source it came from.
 */
export async function extractiveAnswer({ index, hits, queryVec, max = 3 }) {
  if (!hits.length || !queryVec) return [];

  const candidates = [];
  hits.forEach((hit, i) => {
    for (const raw of sentences(hit.chunk.text)) {
      const text = tidy(raw);
      if (text.length < MIN_CHARS || text.length > MAX_CHARS) continue;
      if (!/[a-z]{3}/.test(text)) continue;                 // skip equation debris
      if (isGarbled(text) || !isSentence(text)) continue;
      candidates.push({ text, n: i + 1, order: candidates.length });
    }
  });
  if (!candidates.length) return [];

  const vecs = await embedTexts(index, candidates.map((c) => c.text));
  const dims = index.embedding.dims;
  candidates.forEach((c, i) => {
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += queryVec[j] * vecs[i][j];
    // Nudge toward the better-ranked passages: retrieval already judged which
    // source answers the question, and a sentence's own embedding is a noisy
    // signal on its own.
    c.score = dot + 0.03 * (hits.length - (c.n - 1)) / hits.length;
  });

  const picked = [];
  for (const c of [...candidates].sort((a, b) => b.score - a.score)) {
    if (picked.length >= max) break;
    // Don't say the same thing twice: overlapping passages repeat sentences.
    const words = new Set(c.text.toLowerCase().split(/\W+/).filter((w) => w.length > 4));
    const dup = picked.some((p) => {
      const other = new Set(p.text.toLowerCase().split(/\W+/).filter((w) => w.length > 4));
      let shared = 0;
      for (const w of words) if (other.has(w)) shared++;
      return shared / Math.max(1, Math.min(words.size, other.size)) > 0.6;
    });
    if (!dup) picked.push(c);
  }

  // Read in the order the sources present them, not in score order.
  return picked.sort((a, b) => a.order - b.order).map(({ text, n }) => ({ text, n }));
}
