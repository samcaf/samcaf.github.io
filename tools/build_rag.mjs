#!/usr/bin/env node
/**
 * build_rag.mjs — offline index builder for the site's "Ask" assistant.
 *
 * Reads site content (HTML pages, CV, note PDFs), chunks it, embeds every
 * chunk with the same model the browser runtime uses, and writes static
 * retrieval indexes to rag/*.json. GitHub Pages serves those files as-is;
 * nothing here runs at deploy time.
 *
 * Usage:  node tools/build_rag.mjs [cv|publications|notes|all]
 * Needs:  node >= 18, pdftotext (poppler). Network on first run only
 *         (embedding-model download + arXiv abstracts; both cached in tools/.cache).
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(TOOLS);
const RAG = path.join(ROOT, 'rag');
const CACHE = path.join(TOOLS, '.cache');

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
  vector_encoding: 'int8x127-base64',
};

/* ---------------- text helpers ---------------- */

const ENTITIES = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ',
  mdash: '—', ndash: '–', rsquo: '’', lsquo: '‘',
  ldquo: '“', rdquo: '”', hellip: '…', middot: '·',
};

function stripHtml(html) {
  return html
    .replace(/<(script|style|svg)\b[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&([a-z]+);/gi, (m, name) => ENTITIES[name.toLowerCase()] ?? ' ')
    .replace(/\s+/g, ' ')
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

function chunkParas(text, target = 1400, overlap = 200) {
  const clean = text.replace(/\r/g, '').trim();
  if (!clean) return [];
  if (clean.length <= target * 1.5) return [clean];
  // Units are paragraphs; hard-split any single unit longer than target.
  const units = [];
  for (const p of clean.split(/\n\s*\n+/)) {
    if (p.length <= target) { units.push(p); continue; }
    for (let i = 0; i < p.length; i += target - overlap) units.push(p.slice(i, i + target));
  }
  const chunks = [];
  let cur = '';
  for (const u of units) {
    if (cur && cur.length + u.length + 2 > target * 1.2) { chunks.push(cur.trim()); cur = ''; }
    cur += (cur ? '\n\n' : '') + u;
  }
  if (cur.trim()) chunks.push(cur.trim());
  return chunks.filter((c) => c.length > 40);
}

function pdfPages(file) {
  const txt = execFileSync('pdftotext', ['-enc', 'UTF-8', file, '-'], {
    encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
  });
  return txt.split('\f').map((p) => p.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim());
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
  for (let i = 0; i < texts.length; i += 16) {
    const out = await ex(texts.slice(i, i + 16), { pooling: EMBED.pooling, normalize: EMBED.normalize });
    const [n, d] = out.dims;
    for (let j = 0; j < n; j++) vecs.push(out.data.slice(j * d, (j + 1) * d));
    process.stdout.write(`\r  embedded ${Math.min(i + 16, texts.length)}/${texts.length} chunks`);
  }
  console.log();
  return vecs;
}

function packInt8(vecs, dims) {
  const arr = new Int8Array(vecs.length * dims);
  vecs.forEach((v, i) => {
    for (let j = 0; j < dims; j++) arr[i * dims + j] = Math.max(-127, Math.min(127, Math.round(v[j] * 127)));
  });
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
}

async function writeCorpus(name, rawChunks) {
  const chunks = rawChunks.filter((c) => c.text && c.text.length > 40);
  chunks.forEach((c, i) => { c.id = `${name}-${i}`; });
  const vecs = await embedAll(chunks.map((c) => c.text));
  const out = {
    version: 1,
    corpus: name,
    embedding: EMBED,
    count: chunks.length,
    chunks,
    vectors: packInt8(vecs, EMBED.dims),
  };
  mkdirSync(RAG, { recursive: true });
  const file = path.join(RAG, `${name}.json`);
  writeFileSync(file, JSON.stringify(out));
  console.log(`  wrote rag/${name}.json — ${chunks.length} chunks, ${(statSync(file).size / 1024).toFixed(0)} KB`);
}

/* ---------------- corpora ---------------- */

async function buildCv() {
  console.log('cv:');
  const chunks = [];
  pdfPages(path.join(ROOT, 'pdfs/cv.pdf')).forEach((page, i) => {
    for (const text of chunkParas(page)) {
      chunks.push({ text, source: { title: 'Curriculum Vitae', url: 'pdfs/cv.pdf', loc: `p. ${i + 1}` } });
    }
  });
  const html = readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  for (const s of sections(html, 'index.html')) {
    for (const text of chunkParas(s.text)) {
      chunks.push({ text, source: { title: s.title ?? 'Home', url: s.url, loc: 'homepage' } });
    }
  }
  await writeCorpus('cv', chunks);
}

function parsePubs(html) {
  return (html.match(/<li class="pub">[\s\S]*?<\/li>/g) ?? []).map((li) => {
    const get = (re) => stripHtml((li.match(re) || [])[1] ?? '');
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
  if (missing.length) {
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

async function buildPublications() {
  console.log('publications:');
  const html = readFileSync(path.join(ROOT, 'publications.html'), 'utf8');
  const pubs = parsePubs(html);
  const abstracts = await arxivAbstracts(pubs.map((p) => p.arxiv).filter(Boolean));
  const chunks = pubs.map((p) => {
    let text = `"${p.title}" — ${p.authors} (${p.venue || p.tag || p.year}).`;
    if (p.arxiv) text += ` arXiv:${p.arxiv}.`;
    const abs = p.arxiv && abstracts[p.arxiv.replace(/v\d+$/, '')];
    if (abs) text += `\nAbstract: ${abs}`;
    return { text, source: { title: p.title, url: p.url, loc: [p.tag, p.year].filter(Boolean).join(' ') } };
  });
  for (const s of sections(html, 'publications.html')) {
    if (/class="pub-list"/.test(s.html)) continue; // covered per-paper above
    for (const text of chunkParas(s.text)) {
      chunks.push({ text, source: { title: s.title ?? 'Publications & talks', url: s.url, loc: 'publications page' } });
    }
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
    pdfPages(path.join(dir, f)).forEach((page, i) => {
      for (const text of chunkParas(page)) {
        chunks.push({ text, source: { title, url: `${rel}#page=${i + 1}`, loc: `p. ${i + 1}` } });
      }
    });
  }
  await writeCorpus('notes', chunks);
}

function writeManifest() {
  const manifest = {
    version: 1,
    embedding: EMBED,
    generator: {
      note: 'WebLLM prebuilt ids — provisional until wired up in the generation milestone.',
      primary: 'Qwen3-1.7B-q4f16_1-MLC',
      small: 'Qwen3-0.6B-q4f16_1-MLC',
    },
    corpora: {
      cv: { file: 'rag/cv.json', pages: ['index.html'], hint: 'Ask about my background…' },
      publications: { file: 'rag/publications.json', pages: ['publications.html'], hint: 'Ask about my publications…' },
      notes: { file: 'rag/notes.json', pages: ['projects.html'], hint: 'Ask about my notes…' },
    },
  };
  mkdirSync(RAG, { recursive: true });
  writeFileSync(path.join(RAG, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log('  wrote rag/manifest.json');
}

const what = process.argv[2] ?? 'all';
if (!['cv', 'publications', 'notes', 'all'].includes(what)) {
  console.error('usage: node tools/build_rag.mjs [cv|publications|notes|all]');
  process.exit(1);
}
if (['cv', 'all'].includes(what)) await buildCv();
if (['publications', 'all'].includes(what)) await buildPublications();
if (['notes', 'all'].includes(what)) await buildNotes();
writeManifest();
console.log('done.');
