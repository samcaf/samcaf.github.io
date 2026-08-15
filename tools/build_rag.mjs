#!/usr/bin/env node
/**
 * build_rag.mjs — offline index builder for the site's "Ask" assistant.
 *
 * Reads site content (HTML pages, CV, note PDFs, arXiv papers, thesis LaTeX),
 * chunks it, embeds every chunk with the same model the browser runtime uses,
 * and writes static retrieval indexes to rag/*.json + rag/*.vec. GitHub Pages
 * serves those as-is; nothing here runs at deploy time.
 *
 * Usage:  node tools/build_rag.mjs [cv|publications|notes|all] [--no-fetch]
 * Needs:  node >= 18, pdftotext (poppler). Network on first run only —
 *         model, arXiv PDFs, and thesis sources are cached in tools/.cache.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);
const RAG = path.join(ROOT, 'rag');
const CACHE = path.join(TOOLS, '.cache');
const NO_FETCH = process.argv.includes('--no-fetch');

// Embedding config — the browser runtime reads this back from the index files,
// so build-time and query-time vectors are guaranteed to live in the same space.
const EMBED = {
  model: 'Xenova/bge-small-en-v1.5',
  dims: 384,
  dtype: 'q8',
  pooling: 'cls',
  normalize: true,
  // BGE v1.5 wants this prefix on *queries* only; passages are embedded bare.
  query_prefix: 'Represent this sentence for searching relevant passages: ',
  vector_encoding: 'int8x127-bin',
  max_tokens: 512,
};

// Chunks are deliberately small: one topic per vector. Anything approaching the
// encoder's 512-token window averages several topics together and then matches
// everything weakly and nothing well.
const CHUNK = { target: 520, hardMax: 1100, min: 140, overlapSentences: 1 };

/**
 * Deliberately irrelevant questions, used to calibrate a per-corpus "noise
 * floor": the best score this corpus produces for something it knows nothing
 * about. Anything at or below that is not an answer.
 *
 * This has to be measured per corpus, not fixed. BGE cosines sit high and their
 * level depends on corpus size and homogeneity — measured floors run ~0.56 for
 * the CV and ~0.60 for the notes. (Spread and z-score were tried and rejected:
 * see tools/eval_gate.mjs — on the small CV corpus, off-topic queries score
 * *higher* z than real ones.)
 */
const NOISE_PROBES = [
  'zebra parking meter', 'how do I bake sourdough bread', 'best pizza places in Chicago',
  'how to change a flat car tire', 'what is the capital of Peru', 'when does the train leave',
  'my cat keeps knocking things off the table', 'recipe for chicken soup',
  'how tall is the Eiffel Tower', 'who won the world cup in 1998',
];
const NOISE_MARGIN = 0.005;

/* ---------------- text helpers ---------------- */

const ENTITIES = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ',
  mdash: '—', ndash: '–', rsquo: '’', lsquo: '‘',
  ldquo: '“', rdquo: '”', hellip: '…', middot: '·',
};

function stripHtml(html) {
  return html
    .replace(/<(script|style|svg)\b[\s\S]*?<\/\1>/gi, ' ')
    // Block-level tags become paragraph breaks so the chunker can see structure.
    .replace(/<\/(p|div|li|h[1-6]|section|article|tr)>/gi, '\n\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&([a-z]+);/gi, (m, name) => ENTITIES[name.toLowerCase()] ?? ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^[ \t]+|[ \t]+$/gm, '')
    .trim();
}

// Top-level <section> blocks (the site never nests sections).
function sections(html, pageUrl) {
  return (html.match(/<section\b[^>]*>[\s\S]*?<\/section>/g) ?? []).map((m) => {
    const id = (m.match(/\bid="([^"]+)"/) || [])[1] ?? null;
    const title = stripHtml((m.match(/<h[12][^>]*>([\s\S]*?)<\/h[12]>/) || [])[1] ?? '') || null;
    return { id, title, url: id ? `${pageUrl}#${id}` : pageUrl, html: m, text: stripHtml(m) };
  });
}

/** Split into sentences, tolerating the abbreviations physics prose is full of. */
function splitSentences(text) {
  const ABBR = /(?:[A-Z]|Eq|Eqs|Fig|Figs|Ref|Refs|Sec|Secs|Ch|Chap|Tab|App|Prof|Dr|Mr|Ms|St|vs|cf|resp|approx|etc|al|e\.g|i\.e|Phys|Rev|Lett|Nucl|Mod|Int|J|Vol|No|pp)$/;
  const out = [];
  let start = 0;
  const re = /([.!?])["')\]]*\s+/g;
  let m;
  while ((m = re.exec(text))) {
    const head = text.slice(start, m.index);
    const lastWord = head.split(/[\s(]/).pop() ?? '';
    if (m[1] === '.' && ABBR.test(lastWord)) continue;      // "Eq. (3)", "e.g. this"
    if (!/[A-Z(“"'$\\]/.test(text[re.lastIndex] ?? '')) continue; // next must look like a start
    out.push(text.slice(start, re.lastIndex).trim());
    start = re.lastIndex;
  }
  const tail = text.slice(start).trim();
  if (tail) out.push(tail);
  return out.filter(Boolean);
}

/** Sentence-aware chunker: ~CHUNK.target chars, one sentence of overlap. */
function chunkText(text) {
  const clean = String(text ?? '').replace(/\r/g, '').trim();
  if (!clean) return [];

  const sents = [];
  for (const para of clean.split(/\n\s*\n+/)) {
    const flat = para.replace(/\s+/g, ' ').trim();
    if (flat) sents.push(...splitSentences(flat));
  }

  const chunks = [];
  let cur = [];
  let len = 0;
  const flush = () => {
    const t = cur.join(' ').trim();
    if (t) chunks.push(t);
    cur = [];
    len = 0;
  };

  for (const s of sents) {
    if (s.length > CHUNK.hardMax) {           // runaway "sentence" (tables, math dumps)
      if (len) flush();
      for (let i = 0; i < s.length; i += CHUNK.target) chunks.push(s.slice(i, i + CHUNK.target).trim());
      continue;
    }
    if (len && len + s.length > CHUNK.target) {
      const tail = cur.slice(-CHUNK.overlapSentences);
      flush();
      cur = [...tail];
      len = tail.join(' ').length;
    }
    cur.push(s);
    len += s.length + 1;
  }
  flush();

  // Fold a too-short trailing chunk back into its predecessor.
  const out = [];
  for (const c of chunks) {
    if (c.length < CHUNK.min && out.length) out[out.length - 1] += ' ' + c;
    else out.push(c);
  }
  return out.filter((c) => c.length >= 60);
}

/* ---------------- PDF text ---------------- */

/**
 * `collapse` squeezes runs of spaces — right for flowing prose, but fatal for
 * -layout output, where a run of 3+ spaces *is* the column separator that tells
 * an entry apart from a section heading. The CV parser needs them intact.
 */
function pdfPages(file, { layout = false, collapse = true } = {}) {
  const args = ['-enc', 'UTF-8'];
  if (layout) args.push('-layout');
  const txt = execFileSync('pdftotext', [...args, file, '-'], {
    encoding: 'utf8', maxBuffer: 128 * 1024 * 1024,
  });
  return txt.split('\f').map((p) => (collapse ? p.replace(/[ \t]+/g, ' ') : p).replace(/\s+$/, ''));
}

/** Drop running heads/footers and page numbers that repeat across a document. */
function stripRunningHeads(pages) {
  if (pages.length < 4) return pages;
  const seen = new Map();
  for (const p of pages) {
    const lines = p.split('\n').map((l) => l.trim()).filter(Boolean);
    for (const l of [lines[0], lines[1], lines[lines.length - 1]]) {
      if (l && l.length < 90) seen.set(l, (seen.get(l) ?? 0) + 1);
    }
  }
  const threshold = Math.max(3, Math.floor(pages.length * 0.35));
  const common = new Set([...seen].filter(([, n]) => n >= threshold).map(([l]) => l));
  return pages.map((p) => p.split('\n')
    .filter((l) => {
      const t = l.trim();
      return t && !common.has(t) && !/^\d{1,4}$/.test(t) && !/^[-–—\s|]+$/.test(t);
    })
    .join('\n'));
}

/** Rejoin hyphenated line breaks and unwrap hard-wrapped lines into paragraphs. */
function unwrap(pageText) {
  return pageText
    .replace(/(\p{L})[-‐‑]\n(\p{L})/gu, '$1$2')
    .replace(/\n{2,}/g, '\u0000')   // park real paragraph breaks
    .replace(/\n/g, ' ')            // undo the PDF's hard line wrapping
    .replace(/\u0000/g, '\n\n')     // and restore them
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

/* ---------------- LaTeX text ---------------- */

const KEEP_ARG = /\\(?:emph|textit|textbf|textsc|texttt|text|mbox|underline|uline|title|caption|footnote|subsection|paragraph|item)\b\s*(?:\[[^\]]*\])?\{/;

/**
 * Reduce LaTeX to readable prose, *keeping the mathematics*.
 *
 * Math is set aside behind placeholders before the macro-stripping runs, then
 * restored as `$…$` / `$$…$$` for the browser to typeset. It is stripped again
 * at embedding time (see stripMath) because LaTeX embeds badly — so retrieval
 * sees prose while the reader sees equations. On a physics site, an excerpt
 * with its equations deleted is not really the author's sentence.
 */
function texToProse(tex) {
  let s = tex;

  // Comments (but not \%). LaTeX comments run to end of line.
  s = s.replace(/(^|[^\\])%.*$/gm, '$1');

  // Keep figure/table captions, drop the floats themselves.
  const captions = [];
  s = s.replace(/\\begin\{(figure|table|wrapfigure)\*?\}[\s\S]*?\\end\{\1\*?\}/g, (block) => {
    const cap = block.match(/\\caption\s*\{([\s\S]*?)\}\s*(?:\\label|\n|$)/);
    if (cap) captions.push(cap[1]);
    return ' ';
  });

  // --- park the math where the macro stripper cannot reach it ---------------
  const math = [];
  const stash = (body, display) => {
    const cleaned = String(body)
      .replace(/\\(?:label|nonumber|notag)\s*(?:\{[^{}]*\})?/g, ' ')  // KaTeX rejects \label
      .replace(/\s+/g, ' ')
      .trim();
    if (!cleaned) return ' ';
    math.push({ tex: cleaned, display });
    return ` ⟦M${math.length - 1}⟧ `;
  };

  s = s.replace(/\\begin\{(equation\*?)\}([\s\S]*?)\\end\{\1\}/g, (_, __, body) => stash(body, true));
  // align/gather/… become `aligned`, which KaTeX renders inside display math.
  s = s.replace(/\\begin\{(align\*?|gather\*?|multline\*?|eqnarray\*?|split)\}([\s\S]*?)\\end\{\1\}/g,
    (_, __, body) => stash(`\\begin{aligned}${body}\\end{aligned}`, true));
  s = s.replace(/\$\$([\s\S]*?)\$\$/g, (_, body) => stash(body, true));
  s = s.replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => stash(body, true));
  s = s.replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => stash(body, false));
  s = s.replace(/\$([^$]+)\$/g, (_, body) => stash(body, false));

  // Non-prose environments that carry no mathematics.
  const DROP = 'tikzpicture|tabular|verbatim|lstlisting|thebibliography';
  s = s.replace(new RegExp(`\\\\begin\\{(${DROP})\\*?\\}[\\s\\S]*?\\\\end\\{\\1\\*?\\}`, 'g'), ' ');

  // Cross-references and citations: drop entirely, argument included.
  s = s.replace(/\\(?:label|ref|eqref|cite[a-z]*|Reff?|Chap|Sec|Fig|Tab|App|Eq|Eqs|index|glsadd|nocite|markboth|epigraph|vspace|hspace|includegraphics)\b\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})*/g, ' ');

  // Unwrap content-bearing macros (repeat for nesting).
  for (let i = 0; i < 4; i++) {
    s = s.replace(new RegExp(KEEP_ARG.source + '([^{}]*)\\}', 'g'), ' $1 ');
  }

  // Remaining environments: keep their bodies.
  s = s.replace(/\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?/g, '\n\n');

  // Anything left over.
  s = s.replace(/\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?/g, ' ');
  s = s.replace(/[{}]/g, ' ');
  s = s.replace(/\\[%&#_]/g, '');
  s = s.replace(/~/g, ' ');

  // A dropped citation leaves its punctuation behind: "introduced by the author in ,"
  s = s.replace(/\s+([,.;:!?])/g, '$1').replace(/\(\s*\)/g, '');
  s = s.replace(/[ \t]+/g, ' ').replace(/ ?\n ?/g, '\n').replace(/\n{3,}/g, '\n\n').trim();

  // --- put the mathematics back ---------------------------------------------
  s = s.replace(/⟦M(\d+)⟧/g, (_, i) => {
    const m = math[Number(i)];
    if (!m) return ' ';
    return m.display ? `\n\n$$${m.tex}$$\n\n` : `$${m.tex}$`;
  });
  s = s.replace(/\n{3,}/g, '\n\n').trim();

  return { prose: s, captions: captions.map((c) => texToProse(c).prose) };
}

/** Math as written is unreadable to the embedder; it only ever sees the prose. */
function stripMath(text) {
  return String(text ?? '')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$\n]*\$/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/** Headings carry layout macros (\phantom, \\) that must not reach a citation. */
function cleanTexTitle(s) {
  return String(s ?? '')
    .replace(/\\phantom\s*\{[^{}]*\}/g, ' ')
    .replace(/\\\\/g, ' ')
    .replace(/\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?/g, ' ')
    .replace(/[{}|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Split a chapter .tex into {section, text} blocks, keeping headings. */
function texSections(tex) {
  const chapter = (tex.match(/\\chapter(?:\[([^\]]*)\])?\{([\s\S]*?)\}\s*$/m) || []);
  const chapterTitle = cleanTexTitle(chapter[1] || chapter[2] || '');

  const out = [];
  const re = /\\section(?:\[[^\]]*\])?\{([\s\S]*?)\}/g;
  let m;
  let cursor = 0;
  let current = null;
  while ((m = re.exec(tex))) {
    if (current) out.push({ section: current, tex: tex.slice(cursor, m.index) });
    else out.push({ section: null, tex: tex.slice(cursor, m.index) });
    current = cleanTexTitle(m[1]);
    cursor = re.lastIndex;
  }
  out.push({ section: current, tex: tex.slice(cursor) });
  return { chapterTitle, blocks: out };
}

/* ---------------- downloads (cached) ---------------- */

async function cachedFetch(url, file, { binary = false } = {}) {
  const dest = path.join(CACHE, file);
  if (existsSync(dest)) return dest;
  if (NO_FETCH) throw new Error(`missing cache ${file} and --no-fetch was given`);
  mkdirSync(path.dirname(dest), { recursive: true });
  process.stdout.write(`  fetching ${url} … `);
  const res = await fetch(url, { headers: { 'User-Agent': 'samcaf-site-rag/1.0 (personal site index build)' } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(dest, buf);
  console.log(`${(buf.length / 1024).toFixed(0)} KB`);
  await new Promise((r) => setTimeout(r, 3000)); // be polite to arXiv/GitHub
  return dest;
}

/* ---------------- embedding ---------------- */

let _extractor = null;
async function extractor() {
  if (!_extractor) {
    const { env, pipeline } = await import('@huggingface/transformers');
    env.cacheDir = path.join(CACHE, 'models');
    console.log(`  loading embedder ${EMBED.model} (cached in tools/.cache/models)…`);
    _extractor = await pipeline('feature-extraction', EMBED.model, { dtype: EMBED.dtype });
  }
  return _extractor;
}

async function embedAll(texts) {
  const ex = await extractor();
  const vecs = [];
  for (let i = 0; i < texts.length; i += 32) {
    const out = await ex(texts.slice(i, i + 32), { pooling: EMBED.pooling, normalize: EMBED.normalize });
    const [n, d] = out.dims;
    for (let j = 0; j < n; j++) vecs.push(out.data.slice(j * d, (j + 1) * d));
    process.stdout.write(`\r  embedded ${Math.min(i + 32, texts.length)}/${texts.length} chunks`);
  }
  console.log();
  return vecs;
}

/**
 * Each chunk is embedded with a short provenance header ("<title> — <loc>: …")
 * so a passage from page 7 still knows which paper it belongs to. The header is
 * not shipped: the UI shows the verbatim chunk and renders provenance as a
 * citation.
 */
function embedText(c) {
  const head = [c.source.title, c.source.loc].filter(Boolean).join(' — ');
  const body = stripMath(c.text);   // the index is built on prose only
  return head ? `${head}: ${body}` : body;
}

/** Best score this corpus gives a deliberately irrelevant question. */
async function noiseFloor(vecs) {
  const probes = await embedAll(NOISE_PROBES.map((p) => (EMBED.query_prefix ?? '') + p));
  let worst = 0;
  for (const pv of probes) {
    let best = -1;
    for (const v of vecs) {
      let dot = 0;
      for (let j = 0; j < EMBED.dims; j++) dot += pv[j] * v[j];
      if (dot > best) best = dot;
    }
    if (best > worst) worst = best;
  }
  return +(worst + NOISE_MARGIN).toFixed(4);
}

async function writeCorpus(name, rawChunks) {
  if (_dropped) { console.log(`  filtered ${_dropped} non-prose chunks (equations, name lists, boilerplate)`); _dropped = 0; }
  const chunks = rawChunks.filter((c) => c.text && c.text.length >= 60);
  const seen = new Set();
  const unique = chunks.filter((c) => {
    const key = c.text.slice(0, 160);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  unique.forEach((c, i) => { c.id = `${name}-${i}`; });

  const vecs = await embedAll(unique.map(embedText));
  const arr = new Int8Array(unique.length * EMBED.dims);
  vecs.forEach((v, i) => {
    for (let j = 0; j < EMBED.dims; j++) {
      arr[i * EMBED.dims + j] = Math.max(-127, Math.min(127, Math.round(v[j] * 127)));
    }
  });

  const floor = await noiseFloor(vecs);
  console.log(`  noise floor ${floor} — the best this corpus scores on a question it knows nothing about`);

  mkdirSync(RAG, { recursive: true });
  const vecFile = `${name}.vec`;
  writeFileSync(path.join(RAG, vecFile), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
  const out = {
    version: 2,
    corpus: name,
    embedding: EMBED,
    count: unique.length,
    noise_floor: floor,
    vectors: { file: vecFile, encoding: 'int8', dims: EMBED.dims },
    chunks: unique,
  };
  const jsonFile = path.join(RAG, `${name}.json`);
  writeFileSync(jsonFile, JSON.stringify(out));
  const kb = (f) => (statSync(f).size / 1024).toFixed(0);
  console.log(`  wrote rag/${name}.json (${kb(jsonFile)} KB) + rag/${vecFile} (${kb(path.join(RAG, vecFile))} KB) — ${unique.length} chunks`);
}

/* ---------------- corpora ---------------- */

// Boilerplate that retrieves well (it is topically on-point) but reads terribly
// as a quote: author lists, thanks, reference dumps, preprint stamps.
const BOILERPLATE = /^(acknowledg|we (?:would like to )?thank\b|references\b|bibliography\b|appendix\s*[a-z]?\s*$|contents\b|\[?\d+\]\s|arxiv:\s*\d|mit-ctp)/i;

/**
 * Is this chunk readable prose? Equation dumps, tables of symbols, and name
 * lists match queries about their topic but are useless as displayed excerpts.
 */
function isProse(t) {
  // Judge the prose, not the equations: a sentence carrying real mathematics
  // would fail every ratio below on the raw text, and it is exactly the kind of
  // sentence worth keeping.
  const hasMath = /\$/.test(t);
  const plain = hasMath ? stripMath(t) : t;
  if (!plain) return false;
  if (BOILERPLATE.test(plain.trim())) return false;
  const words = plain.split(/\s+/).filter(Boolean);
  if (words.length < (hasMath ? 6 : 10)) return false;
  const alpha = (plain.match(/[A-Za-z]/g) ?? []).length / plain.length;
  if (alpha < 0.55) return false;                       // math / table soup
  const wordy = words.filter((w) => /^[A-Za-z][a-z]{2,}$/.test(w)).length / words.length;
  return wordy >= 0.35;                                 // symbol or surname lists
}

let _dropped = 0;
function pushChunks(list, text, source, { filter = true } = {}) {
  for (const t of chunkText(text)) {
    if (filter && !isProse(t)) { _dropped++; continue; }
    list.push({ text: t, source: { ...source } });
  }
}

/** Smallcaps come out of the PDF as "PH.D. iN THEORETiCAL PHYSiCS" — restore
 *  ordinary casing so the tokenizer sees real words. */
function tidyCase(s) {
  const letters = s.replace(/[^A-Za-z]/g, '');
  if (letters.length < 4) return s;
  if ((s.match(/[A-Z]/g) ?? []).length / letters.length < 0.6) return s;
  return s
    .replace(/[A-Za-z][A-Za-z.'’-]*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
    .replace(/\.([a-z])/g, (_, c) => '.' + c.toUpperCase());   // Ph.d. → Ph.D.
}

const tidyLine = (l) => tidyCase(l.trim().replace(/\s{3,}/g, ' · ')).replace(/\s+/g, ' ');

/**
 * Parse the CV into one entry per role/degree/award, tagged with its section.
 * A résumé is telegraphic, not prose: "Massachusetts Institute of Technology
 * Cambridge, MA PH.D." embeds nothing like the question "where did you go to
 * grad school?". Carrying the section heading into each chunk is what makes
 * those questions retrievable.
 */
function parseCvEntries(fullText) {
  const out = [];
  let section = null;
  let cur = null;
  const flush = () => { if (cur && (cur.head || cur.body)) out.push(cur); cur = null; };

  for (const raw of fullText.split('\n')) {
    const l = raw.replace(/\s+$/, '');
    if (!l.trim()) continue;
    const t = l.trim();
    const bullet = /^[•·]/.test(t);
    const columned = /\s{3,}/.test(l);   // -layout's right-aligned date/location column

    if (!bullet && !columned && /^[A-Z][A-Za-z&,'’ ]{2,44}$/.test(t)) {
      flush(); section = t; continue;
    }
    if (!bullet && columned && /^\S/.test(l)) {
      if (cur && !cur.body) { cur.head += ' · ' + tidyLine(l); continue; }  // role under institution
      flush(); cur = { section, head: tidyLine(l), body: '' }; continue;
    }
    if (!cur) cur = { section, head: '', body: '' };
    cur.body += (cur.body ? ' ' : '') + tidyLine(t.replace(/^[•·]\s*/, ''));
  }
  flush();
  return out;
}

/**
 * Turn "MIT · Cambridge, MA · Ph.D. In Physics · Aug. 2019 – May 2025" into
 * "Ph.D. In Physics at MIT, Cambridge, MA (Aug. 2019 – May 2025)." Sentences
 * retrieve far better than column-separated fragments for questions phrased as
 * questions.
 */
function entrySentence(head) {
  const parts = head.split(' · ').map((p) => p.trim()).filter(Boolean);
  if (parts.length < 2) return head;
  const dates = /\d{4}|present/i.test(parts[parts.length - 1]) ? parts.pop() : null;
  const place = parts.length > 1 && /,\s*[A-Z]{2}$|,\s*[A-Z][a-z]+$/.test(parts[1]) ? parts.splice(1, 1)[0] : null;
  const org = parts.shift();
  const role = parts.join(', ');
  let s = role ? `${role} at ${org}` : org;
  if (place) s += `, ${place}`;
  if (dates) s += ` (${dates})`;
  return s + '.';
}

async function buildCv() {
  console.log('cv:');
  const chunks = [];
  // -layout keeps each role with its bullets; without it the two-column header
  // scatters into orphan lines like "Cambridge, MA".
  const cvText = pdfPages(path.join(ROOT, 'pdfs/cv.pdf'), { layout: true, collapse: false }).join('\n');
  const entries = parseCvEntries(cvText);
  for (const e of entries) {
    const source = { title: 'Curriculum Vitae', url: 'pdfs/cv.pdf', loc: e.section ?? 'Contact' };
    const sentence = e.head ? entrySentence(e.head) : '';
    if (!e.body) { pushChunks(chunks, sentence, source, { filter: false }); continue; }
    // Repeat the institution/role on every piece so a split entry keeps its context.
    const head = sentence ? `${sentence} ` : '';
    for (const piece of chunkText(e.body)) chunks.push({ text: head + piece, source: { ...source } });
  }
  console.log(`  CV: ${entries.length} entries → ${chunks.length} chunks (${[...new Set(entries.map((e) => e.section).filter(Boolean))].length} sections)`);

  const html = readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  for (const s of sections(html, 'index.html')) {
    pushChunks(chunks, s.text, { title: s.title ?? 'Home', url: s.url, loc: 'homepage' });
  }
  await writeCorpus('cv', chunks);
}

function parsePubs(html) {
  return (html.match(/<li class="pub">[\s\S]*?<\/li>/g) ?? []).map((li) => {
    const get = (re) => stripHtml((li.match(re) || [])[1] ?? '').replace(/\s+/g, ' ');
    const links = [...li.matchAll(/<a href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)]
      .map((m) => ({ url: m[1], label: stripHtml(m[2]) }));
    const arxiv = ((li.match(/arxiv\.org\/abs\/([^"<]+)/) || [])[1] ?? null);
    return {
      title: get(/<h3[^>]*>([\s\S]*?)<\/h3>/),
      year: (li.match(/<div class="pub-year">\s*(\d{4})/) || [])[1] ?? '',
      tag: get(/<span class="pub-venue-tag">([\s\S]*?)<\/span>/),
      authors: get(/<div class="authors">([\s\S]*?)<\/div>/),
      venue: get(/<div class="venue">([\s\S]*?)<\/div>/),
      arxiv,
      url: (links.find((l) => /arxiv/i.test(l.url)) ?? links[0])?.url ?? 'publications.html',
    };
  });
}

async function arxivAbstracts(ids) {
  const file = path.join(CACHE, 'arxiv.json');
  const cache = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : {};
  const missing = ids.filter((id) => id && !cache[id.replace(/v\d+$/, '')]);
  if (missing.length && !NO_FETCH) {
    console.log(`  fetching ${missing.length} abstract(s) from arXiv…`);
    const url = `https://export.arxiv.org/api/query?id_list=${missing.join(',')}&max_results=${missing.length}`;
    const xml = await (await fetch(url)).text();
    for (const entry of xml.split(/<entry>/).slice(1)) {
      const id = ((entry.match(/<id>https?:\/\/arxiv\.org\/abs\/([^<]+)<\/id>/) || [])[1] ?? '').replace(/v\d+$/, '');
      const summary = stripHtml((entry.match(/<summary[^>]*>([\s\S]*?)<\/summary>/) || [])[1] ?? '');
      if (id && summary) cache[id] = summary;
    }
    mkdirSync(CACHE, { recursive: true });
    writeFileSync(file, JSON.stringify(cache, null, 1));
  }
  return cache;
}

// Large multi-author community reports are indexed by abstract only: their full
// text would swamp the corpus (the CLIC report alone is 283 pages / ~100 authors)
// and answer questions about work that isn't really Sam's. Add or remove ids here.
const ABSTRACT_ONLY = new Set(['1812.02093']);

/** Full text of each arXiv paper, chunked with page-level deep links. */
async function paperFullText(pub, chunks) {
  if (!pub.arxiv) return;
  const id = pub.arxiv.replace(/v\d+$/, '');
  if (ABSTRACT_ONLY.has(id)) {
    console.log(`  ${id}: abstract only (large multi-author report)`);
    return;
  }
  let file;
  try {
    file = await cachedFetch(`https://arxiv.org/pdf/${id}`, `papers/${id.replace('/', '_')}.pdf`, { binary: true });
  } catch (err) {
    console.warn(`\n  ! skipping full text for ${id}: ${err.message}`);
    return;
  }
  const pages = stripRunningHeads(pdfPages(file));
  if (pages.length > 120) {
    console.warn(`  ! ${id}: ${pages.length} pages — consider adding it to ABSTRACT_ONLY, it will dominate the index`);
  }
  const loc = [pub.tag, pub.year].filter(Boolean).join(' ');
  let kept = 0;
  pages.forEach((page, i) => {
    const text = unwrap(page);
    // References sections are citation soup — they retrieve badly and read worse.
    if (/^\s*(references|bibliography)\b/i.test(text) && i > pages.length / 2) return;
    const before = chunks.length;
    pushChunks(chunks, text, {
      title: pub.title,
      url: `https://arxiv.org/pdf/${id}#page=${i + 1}`,
      loc: `${loc} · p. ${i + 1}`,
    });
    kept += chunks.length - before;
  });
  console.log(`  ${id}: ${pages.length} pages → ${kept} chunks`);
}

/**
 * The thesis defines ~180 of its own macros (\alphas, \acomm, \le/\ri …).
 * Quoted equations are written in them, so they are extracted here and handed
 * to KaTeX at render time — otherwise every excerpt with math shows an error.
 */
function parseMacros(tex) {
  const out = {};
  const group = (str, i) => {                       // i points at the opening brace
    let depth = 0;
    for (let j = i; j < str.length; j++) {
      if (str[j] === '{') depth++;
      else if (str[j] === '}' && !--depth) return str.slice(i + 1, j);
    }
    return null;
  };
  // Layout macros would only break the math renderer.
  const LAYOUT = /\\(vspace|par|noindent|hrule|clearpage|newpage|section|chapter|centering|makebox|raisebox|phantom|xspace)\b/;
  // KaTeX has no \mathpalette; a macro built on it can never render.
  const UNSUPPORTED = /\\(mathpalette|mathchoice|DHLhksqrt|newlength|setlength|settowidth|hbox|vbox)\b/;

  const re = /\\(?:newcommand|renewcommand|providecommand)\*?\s*(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))\s*(?:\[(\d)\])?(?:\[[^\]]*\])?\s*\{/g;
  let m;
  while ((m = re.exec(tex))) {
    const name = m[1] || m[2];
    let body = group(tex, re.lastIndex - 1);
    if (!name || body == null) continue;
    body = body.replace(/\\xspace/g, '').trim()
      .replace(/^\\ensuremath\s*\{([\s\S]*)\}$/, '$1').trim();
    if (!body || LAYOUT.test(body) || UNSUPPORTED.test(body)) continue;
    if (KATEX_BUILTINS.has(name)) continue;   // never shadow a working built-in
    out['\\' + name] = body;
  }
  // \def\le{\left} — TeX primitives the preamble uses alongside \newcommand.
  const def = /\\def\s*\\([A-Za-z@]+)\s*((?:#\d)*)\s*\{/g;
  while ((m = def.exec(tex))) {
    const body = group(tex, def.lastIndex - 1);
    if (body == null || LAYOUT.test(body) || UNSUPPORTED.test(body)) continue;
    if (KATEX_BUILTINS.has(m[1])) continue;
    out['\\' + m[1]] = body.trim();
  }

  const op = /\\DeclareMathOperator\*?\s*\{\s*\\([A-Za-z@]+)\s*\}\s*\{/g;
  while ((m = op.exec(tex))) {
    const body = group(tex, op.lastIndex - 1);
    if (body != null && !KATEX_BUILTINS.has(m[1])) out['\\' + m[1]] = `\\operatorname{${body.trim()}}`;
  }

  for (const name of BLOCKED_MACROS) delete out[name];
  return { ...PHYSICS_PACKAGE, ...out };   // the preamble wins over the fallbacks
}

// Captured from the preamble but meaningless (or harmful) inside math.
// A preamble that "improves" \sqrt must not break the one KaTeX ships.
const KATEX_BUILTINS = new Set(('sqrt frac dfrac tfrac text textbf textit mathrm mathbf mathcal mathbb vec hat bar dot ddot '
  + 'tilde overline underline left right sum prod int oint lim log ln exp sin cos tan sec csc cot sinh cosh tanh '
  + 'max min sup inf det dim deg arg gcd ker Pr binom cdot times div pm mp leq geq neq approx equiv propto').split(' '));

const BLOCKED_MACROS = ['\\label', '\\ref', '\\eqref', '\\cite', '\\caption', '\\item', '\\footnote'];

// The thesis loads the `physics` package, whose commands are defined by the
// package rather than the preamble, so KaTeX has never heard of them.
const PHYSICS_PACKAGE = {
  '\\ensuremath': '#1',                       // strips any that survived extraction
  '\\Tr': '\\operatorname{Tr}',
  '\\tr': '\\operatorname{tr}',
  '\\indices': '#1',                          // tensor pkg: the arg already has ^ and _
  '\\slashed': '#1\\!\\!\\!/',
  '\\SumInt': '\\sum\\!\\!\\!\\!\\int',
  '\\Vec': '\\vec{#1}',
  '\\comm': '\\left[#1,#2\\right]',
  '\\acomm': '\\left\\{#1,#2\\right\\}',
  '\\pb': '\\left\\{#1,#2\\right\\}',
  '\\dd': '\\mathrm{d}',
  '\\abs': '\\left|#1\\right|',
  '\\norm': '\\left\\|#1\\right\\|',
  '\\ev': '\\left\\langle#1\\right\\rangle',
  '\\bra': '\\left\\langle#1\\right|',
  '\\ket': '\\left|#1\\right\\rangle',
  '\\braket': '\\left\\langle#1\\middle|#2\\right\\rangle',
  '\\order': '\\mathcal{O}',
  '\\cross': '\\times',
  '\\dv': '\\frac{\\mathrm{d}#1}{\\mathrm{d}#2}',
  '\\pdv': '\\frac{\\partial#1}{\\partial#2}',
  '\\ketbra': '\\left|#1\\right\\rangle\\!\\left\\langle#2\\right|',
  '\\Bar': '\\bar{#1}',
  '\\textsc': '\\text{#1}',
};

const THESIS = {
  title: 'Particles Inside Particles (Ph.D. thesis)',
  url: 'https://dspace.mit.edu/entities/publication/66c4727b-11b7-46be-99b1-db991e1ba5b6',
  repo: 'https://raw.githubusercontent.com/samcaf/Thesis/HEAD/',
  files: [
    ['frontmatter/abstract.tex', 'Abstract'],
    ['frontmatter/preface.tex', 'Preface'],
    ['frontmatter/notation.tex', 'Notation'],
    ['chapters/1-qcd.tex', 'Ch. 1'],
    ['chapters/2-particles.tex', 'Ch. 2'],
    ['chapters/3-jets.tex', 'Ch. 3'],
    ['chapters/4-substructure.tex', 'Ch. 4'],
    ['chapters/5-event_shapes.tex', 'Ch. 5'],
    ['chapters/conclusion.tex', 'Conclusion'],
  ],
  macroFiles: ['includes/thesis_utils.tex', 'includes/paper_preamble.tex'],
};

/** Pull the thesis macro definitions into rag/macros.json for the renderer. */
async function writeMacros() {
  const macros = {};
  for (const rel of THESIS.macroFiles) {
    try {
      const file = await cachedFetch(THESIS.repo + rel, `thesis/${path.basename(rel)}`);
      Object.assign(macros, parseMacros(readFileSync(file, 'utf8')));
    } catch (err) {
      console.warn(`  ! macros from ${rel}: ${err.message}`);
    }
  }
  mkdirSync(RAG, { recursive: true });
  writeFileSync(path.join(RAG, 'macros.json'), JSON.stringify(macros));
  console.log(`  wrote rag/macros.json — ${Object.keys(macros).length} macros`);
  return macros;
}

async function thesisChunks(chunks) {
  console.log('  thesis (LaTeX sources):');
  await writeMacros();
  for (const [rel, label] of THESIS.files) {
    let file;
    try {
      file = await cachedFetch(THESIS.repo + rel, `thesis/${path.basename(rel)}`);
    } catch (err) {
      console.warn(`  ! skipping ${rel}: ${err.message}`);
      continue;
    }
    const tex = readFileSync(file, 'utf8');
    const { chapterTitle, blocks } = texSections(tex);
    const before = chunks.length;
    for (const b of blocks) {
      const { prose, captions } = texToProse(b.tex);
      // A citation gets the chapter number plus *one* title — chapter title and
      // section title chained together runs to 110 characters.
      const where = b.section
        ? `${label} · ${b.section}`
        : [label, chapterTitle && chapterTitle !== label ? chapterTitle : null].filter(Boolean).join(' · ');
      pushChunks(chunks, prose, { title: THESIS.title, url: THESIS.url, loc: where });
      for (const cap of captions) {
        pushChunks(chunks, cap, { title: THESIS.title, url: THESIS.url, loc: `${where} · figure` });
      }
    }
    console.log(`    ${label} ${chapterTitle ? `“${chapterTitle}”` : ''} → ${chunks.length - before} chunks`);
  }
}

async function buildPublications() {
  console.log('publications:');
  const html = readFileSync(path.join(ROOT, 'publications.html'), 'utf8');
  const pubs = parsePubs(html);
  const abstracts = await arxivAbstracts(pubs.map((p) => p.arxiv).filter(Boolean));

  const chunks = [];
  for (const p of pubs) {
    let text = `"${p.title}" — ${p.authors} (${p.venue || p.tag || p.year}).`;
    if (p.arxiv) text += ` arXiv:${p.arxiv}.`;
    const abs = p.arxiv && abstracts[p.arxiv.replace(/v\d+$/, '')];
    if (abs) text += `\n\nAbstract: ${abs}`;
    pushChunks(chunks, text, { title: p.title, url: p.url, loc: [p.tag, p.year].filter(Boolean).join(' ') }, { filter: false });
  }
  for (const p of pubs) await paperFullText(p, chunks);
  await thesisChunks(chunks);

  for (const s of sections(html, 'publications.html')) {
    if (/pub-list/.test(s.html)) continue; // covered per-paper above
    pushChunks(chunks, s.text, { title: s.title ?? 'Publications & talks', url: s.url, loc: 'publications page' });
  }
  await writeCorpus('publications', chunks);
}

function noteCards(html) {
  const sec = (html.match(/<section[^>]*id="notes"[\s\S]*?<\/section>/) || [])[0] ?? '';
  const map = {};
  for (const card of sec.match(/<article class="card[\s\S]*?<\/article>/g) ?? []) {
    const file = (card.match(/href="(pdfs\/notes\/[^"]+\.pdf)"/) || [])[1];
    if (!file) continue;
    map[path.basename(file)] = {
      title: stripHtml((card.match(/<h3[^>]*>([\s\S]*?)<\/h3>/) || [])[1] ?? ''),
      desc: stripHtml((card.match(/<p>([\s\S]*?)<\/p>/) || [])[1] ?? ''),
      tag: stripHtml((card.match(/<div class="card-lang">([\s\S]*?)<\/div>/) || [])[1] ?? ''),
    };
  }
  return map;
}

async function buildNotes() {
  console.log('notes:');
  const cards = noteCards(readFileSync(path.join(ROOT, 'projects.html'), 'utf8'));
  const dir = path.join(ROOT, 'pdfs/notes/traditional');
  const chunks = [];
  for (const f of readdirSync(dir).filter((f) => f.endsWith('.pdf')).sort()) {
    const rel = `pdfs/notes/traditional/${f}`;
    const card = cards[f];
    const title = card?.title || f.replace(/\.pdf$/, '').replace(/_/g, ' ');
    if (card?.desc) {
      chunks.push({ text: `${title} — ${card.tag || 'note'}. ${card.desc}`, source: { title, url: rel, loc: 'summary' } });
    }
    const pages = stripRunningHeads(pdfPages(path.join(dir, f)));
    const before = chunks.length;
    pages.forEach((page, i) => {
      pushChunks(chunks, unwrap(page), { title, url: `${rel}#page=${i + 1}`, loc: `p. ${i + 1}` });
    });
    console.log(`  ${title}: ${pages.length} pages → ${chunks.length - before} chunks`);
  }
  await writeCorpus('notes', chunks);
}

/**
 * Merge the per-page corpora into one index the whole site shares, so a visitor
 * can ask anything from anywhere. Concatenates the already-built artifacts —
 * nothing is re-embedded.
 *
 * Each chunk keeps a `c` tag naming its source corpus, and the index keeps every
 * corpus's own noise floor. A single global floor would be wrong: the CV's floor
 * is 0.557 and the notes' is 0.607, so one number either rejects real CV
 * questions or waves nonsense through on the notes. The runtime scores each
 * chunk against the floor of the corpus it came from.
 */
function mergeCorpora() {
  console.log('all (unified index):');
  const parts = [];
  for (const name of ['cv', 'publications', 'notes']) {
    const file = path.join(RAG, `${name}.json`);
    if (!existsSync(file)) { console.warn(`  ! ${name}.json missing — skipping`); continue; }
    const json = JSON.parse(readFileSync(file, 'utf8'));
    const vec = readFileSync(path.join(RAG, `${name}.vec`));
    parts.push({ name, json, vec });
  }
  if (!parts.length) throw new Error('nothing to merge');

  const dims = parts[0].json.embedding.dims;
  const total = parts.reduce((n, p) => n + p.json.count, 0);
  const arr = new Int8Array(total * dims);
  const chunks = [];
  const floors = {};
  let offset = 0;
  for (const p of parts) {
    floors[p.name] = p.json.noise_floor;
    const v = new Int8Array(p.vec.buffer, p.vec.byteOffset, p.vec.byteLength);
    arr.set(v, offset * dims);
    for (const c of p.json.chunks) chunks.push({ ...c, c: p.name });
    offset += p.json.count;
    console.log(`  + ${p.name}: ${p.json.count} chunks (floor ${p.json.noise_floor})`);
  }

  writeFileSync(path.join(RAG, 'all.vec'), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
  const out = {
    version: 2,
    corpus: 'all',
    embedding: parts[0].json.embedding,
    count: chunks.length,
    noise_floors: floors,
    parts: parts.map((p) => ({ corpus: p.name, count: p.json.count })),
    vectors: { file: 'all.vec', encoding: 'int8', dims },
    chunks,
  };
  const jsonFile = path.join(RAG, 'all.json');
  writeFileSync(jsonFile, JSON.stringify(out));
  const kb = (f) => (statSync(f).size / 1024).toFixed(0);
  console.log(`  wrote rag/all.json (${kb(jsonFile)} KB) + rag/all.vec (${kb(path.join(RAG, 'all.vec'))} KB) — ${chunks.length} chunks`);
}

function writeManifest() {
  const manifest = {
    version: 2,
    embedding: EMBED,
    chunking: CHUNK,
    generator: {
      note: 'WebLLM prebuilt ids — provisional until wired up in the generation milestone.',
      primary: 'Qwen3-1.7B-q4f16_1-MLC',
      small: 'Qwen3-0.6B-q4f16_1-MLC',
    },
    // One index for the whole site; the per-corpus files are its inputs.
    corpus: { file: 'rag/all.json', vectors: 'rag/all.vec', pages: ['index.html', 'publications.html', 'projects.html'] },
    parts: ['rag/cv.json', 'rag/publications.json', 'rag/notes.json'],
  };
  mkdirSync(RAG, { recursive: true });
  writeFileSync(path.join(RAG, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log('  wrote rag/manifest.json');
}

const what = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : 'all';
if (!['cv', 'publications', 'notes', 'all', 'merge', 'macros'].includes(what)) {
  console.error('usage: node tools/build_rag.mjs [cv|publications|notes|all|merge] [--no-fetch]');
  process.exit(1);
}
if (what === 'macros') { await writeMacros(); process.exit(0); }
if (['cv', 'all'].includes(what)) await buildCv();
if (['publications', 'all'].includes(what)) await buildPublications();
if (['notes', 'all'].includes(what)) await buildNotes();
// The site shares one index; rebuild it whenever any part changed.
mergeCorpora();
writeManifest();
console.log('done.');
