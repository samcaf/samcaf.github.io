/**
 * ask.js — the "Ask" widget: a floating bar that opens a panel (desktop
 * sidebar / mobile bottom sheet) for page-scoped question answering.
 *
 * Retrieval renders first and verbatim; generated synthesis is a separate
 * zone, so quotes can never be paraphrased into something you didn't write.
 *
 * The same assistant runs on every page against one unified index, so any
 * question can be asked from anywhere on the site.
 */

import { loadIndex, search, loadEmbedder, isEmbedderReady } from './retrieval.js';
import { createGalaxy } from './galaxy.js';
import { prepareGraph, activate } from './graphboost.js';
import { capability, loadEngine, isEngineReady, streamAnswer, MODELS } from './generate.js';
import { extractiveAnswer } from './extractive.js';
import { renderMath, hasMath, preloadMath } from './math.js';

// One assistant, one corpus, every page: a visitor shouldn't have to guess
// which part of the site knows the answer.
const CONFIG = {
  title: 'Ask about my work',
  blurb: 'Searches everything here — my CV, papers, thesis, and lecture notes.',
  hint: 'Ask about my research, papers, or notes…',
  suggestions: [
    'What is PIRANHA?',
    'What are energy correlators?',
    'What did your PhD research cover?',
    'What does the optical theorem say?',
  ],
};

const script = document.querySelector('script[data-ask-corpus]');
const corpusUrl = script?.dataset.askCorpus ?? new URL('../../rag/all.json', import.meta.url).href;
const cfg = CONFIG;

const ICON = {
  spark: '<svg class="ask-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h13M12 5l7 7-7 7"/></svg>',
};

/* ---------------- markup ---------------- */

const root = document.createElement('div');
root.className = 'ask';
root.dataset.state = 'closed';
root.innerHTML = `
  <button class="ask-bar" type="button" aria-expanded="false" aria-controls="ask-panel">
    ${ICON.spark}<span class="ask-bar-hint">${cfg.hint}</span><kbd>/</kbd>
  </button>
  <div class="ask-scrim"></div>
  <aside class="ask-panel" id="ask-panel" role="dialog" aria-modal="true" aria-label="${cfg.title}" inert>
    <span class="ask-grip" aria-hidden="true"></span>
    <header class="ask-head">
      <div>
        <div class="eyebrow">Ask</div>
        <h2>${cfg.title}</h2>
        <p>${cfg.blurb} Runs entirely in your browser.</p>
      </div>
      <button class="ask-close" type="button" aria-label="Close">${ICON.close}</button>
    </header>
    <div class="ask-body">
      <figure class="ask-galaxy" hidden>
        <canvas aria-label="Concept map of this page's sources"></canvas>
        <figcaption>Concepts from my glossary. Lit stars are the ones your answer came from &mdash; drag to spin, click a star to explore.</figcaption>
      </figure>
      <div class="ask-results"></div>
    </div>
    <form class="ask-form">
      <input class="ask-input" type="text" placeholder="${cfg.hint}" autocomplete="off"
             spellcheck="false" aria-label="Your question">
      <button class="ask-send" type="submit" aria-label="Search">${ICON.send}</button>
    </form>
    <p class="ask-foot">Your question never leaves this device.</p>
  </aside>`;
document.body.appendChild(root);

const bar = root.querySelector('.ask-bar');
const panel = root.querySelector('.ask-panel');
const scrim = root.querySelector('.ask-scrim');
const body = root.querySelector('.ask-results');
const scroller = root.querySelector('.ask-body');
const galaxyFig = root.querySelector('.ask-galaxy');
const galaxyCanvas = galaxyFig.querySelector('canvas');
const form = root.querySelector('.ask-form');
const input = root.querySelector('.ask-input');
const send = root.querySelector('.ask-send');

/* ---------------- open / close ---------------- */

let lastFocus = null;

function open() {
  if (root.dataset.state === 'open') return;
  lastFocus = document.activeElement;
  root.dataset.state = 'open';
  bar.setAttribute('aria-expanded', 'true');
  panel.removeAttribute('inert');
  document.body.style.overflow = 'hidden';
  ensureGalaxy();
  galaxy?.resume();
  preloadMath();
  setTimeout(() => input.focus(), 60);
}

function close() {
  if (root.dataset.state !== 'open') return;
  root.dataset.state = 'closed';
  bar.setAttribute('aria-expanded', 'false');
  panel.setAttribute('inert', '');
  document.body.style.overflow = '';
  galaxy?.pause();
  if (lastFocus instanceof HTMLElement) lastFocus.focus();
}

bar.addEventListener('click', open);
scrim.addEventListener('click', close);
root.querySelector('.ask-close').addEventListener('click', close);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && root.dataset.state === 'open') { close(); return; }
  if (e.key !== '/' || root.dataset.state === 'open') return;
  const el = document.activeElement;
  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
  e.preventDefault();
  open();
});

// Keep tab focus inside the panel while it is open.
panel.addEventListener('keydown', (e) => {
  if (e.key !== 'Tab') return;
  const items = panel.querySelectorAll('button, input, a[href], [tabindex]:not([tabindex="-1"])');
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
});

/* ---------------- galaxy ---------------- */

let galaxy = null;

/** Built on first open, not on page load: the graph is a 86 KB extra. */
function ensureGalaxy() {
  if (galaxy) return;
  try {
    galaxy = createGalaxy(galaxyCanvas, { onSelect: showConcept });
    galaxyFig.hidden = false;
    galaxyFig._galaxy = galaxy;   // handle for debugging from the console
  } catch (err) {
    console.warn('[ask] galaxy unavailable', err);   // the widget works without it
  }
}

/** Clicking a star: show what the concept means, offer to go read about it. */
function showConcept(node) {
  const card = el('div', 'ask-concept');
  card.appendChild(el('h4', null, node.name));
  card.appendChild(el('p', null, node.def
    || `A term that recurs in ${node.home ?? 'these sources'} — mined from the text rather than my glossary, so it has no definition.`));
  const go = el('button', 'ask-concept-go', `Find passages about “${node.name}”`);
  go.type = 'button';
  go.addEventListener('click', () => { input.value = node.name; run(node.name); });
  card.appendChild(go);
  const existing = body.querySelector('.ask-concept');
  if (existing) existing.replaceWith(card);
  else body.prepend(card);
}

/* ---------------- scrolling ---------------- */

/*
 * Follow new content, but stop the moment the reader scrolls up to read
 * something — and resume as soon as they come back to the bottom. Without this
 * a streaming answer keeps yanking the view away mid-sentence.
 */
let stick = true;
scroller.addEventListener('scroll', () => {
  stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 24;
}, { passive: true });

function keepAtBottom() {
  if (stick) scroller.scrollTop = scroller.scrollHeight;
}

/* ---------------- rendering ---------------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text; // never innerHTML for corpus text
  return node;
}

function showIdle() {
  body.replaceChildren();
  body.appendChild(el('p', 'ask-intro', `${cfg.blurb} Answers quote the source directly, so you can check every claim.`));
  body.appendChild(el('div', 'ask-sugg-label', 'Try asking'));
  const wrap = el('div', 'ask-sugg');
  for (const q of cfg.suggestions) {
    const b = el('button', null, q);
    b.type = 'button';
    b.addEventListener('click', () => { input.value = q; run(q); });
    wrap.appendChild(b);
  }
  body.appendChild(wrap);
}

function showStatus(title, detail, { progress = null, error = false } = {}) {
  body.replaceChildren();
  const box = el('div', 'ask-status' + (error ? ' ask-error' : ''));
  const line = el('div');
  line.appendChild(el('b', null, title));
  box.appendChild(line);
  if (progress !== null) {
    const bar = el('div', 'ask-progress');
    const fill = el('i');
    fill.style.width = `${Math.round(progress * 100)}%`;
    bar.appendChild(fill);
    box.appendChild(bar);
  }
  if (detail) box.appendChild(el('small', null, detail));
  body.appendChild(box);
}

function sourceCard(hit, n, collapsed = false) {
  const card = el('div', 'ask-src');
  const quote = el('p', 'ask-src-quote');
  renderMath(quote, hit.chunk.text);   // async; falls back to the LaTeX source
  card.appendChild(quote);

  const more = el('button', 'ask-src-more', 'Show full excerpt');
  more.type = 'button';
  more.addEventListener('click', () => {
    const open = card.classList.toggle('expanded');
    more.textContent = open ? 'Show less' : 'Show full excerpt';
  });
  card.appendChild(more);

  const cite = el('div', 'ask-cite');
  cite.appendChild(el('span', 'ask-cite-n', String(n)));
  const src = hit.chunk.source;
  const body = el('span');
  const link = el('a', null, src.title);
  link.href = src.url;
  if (!src.url.startsWith('#')) { link.target = '_blank'; link.rel = 'noopener'; }
  body.appendChild(link);
  if (src.loc) body.appendChild(el('span', 'ask-cite-loc', ` · ${src.loc}`));
  cite.appendChild(body);
  card.appendChild(cite);

  // Beyond the first, sources start as just their citation line: the list stays
  // scannable and the answer is not buried under four blocks of quotation.
  if (collapsed) {
    card.classList.add('ask-src--min');
    card.tabIndex = 0;
    card.setAttribute('role', 'button');
    card.setAttribute('aria-expanded', 'false');
    const reveal = (ev) => {
      if (ev.target.closest('a')) return;          // let the citation link work
      ev.preventDefault();
      card.classList.remove('ask-src--min');
      card.setAttribute('aria-expanded', 'true');
      card._fitMore();
    };
    card.addEventListener('click', reveal);
    card.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') reveal(ev);
    });
  }

  // Only offer the toggle when the quote is actually clamped (measured once
  // the card is in the document).
  card._fitMore = () => {
    if (!card.classList.contains('ask-src--min')
        && quote.scrollHeight <= quote.clientHeight + 2) more.style.display = 'none';
  };
  return card;
}

function showResults(hits, weak = false) {
  body.replaceChildren();

  // Say so when the match is thin, rather than presenting a near-miss as an answer.
  if (weak) {
    body.appendChild(el('div', 'ask-status ask-weak',
      'Nothing here is a close match. These are the nearest passages I found — they may not answer the question.'));
  }

  const sources = el('div', 'ask-zone');
  const head = el('div', 'ask-zone-head');
  head.appendChild(el('h3', null, `Sources · ${hits.length}`));
  head.appendChild(el('span', 'ask-zone-note', 'verbatim'));
  sources.appendChild(head);
  const cards = hits.map((hit, i) => sourceCard(hit, i + 1, i > 0));
  cards.forEach((c) => sources.appendChild(c));
  body.appendChild(sources);
  cards.forEach((c) => c._fitMore());

  const synth = el('div', 'ask-zone');
  const shead = el('div', 'ask-zone-head');
  shead.appendChild(el('h3', null, 'Synthesis'));
  shead.appendChild(el('span', 'ask-zone-note', 'generated'));
  synth.appendChild(shead);
  const box = el('div', 'ask-synth');
  synth.appendChild(box);
  body.appendChild(synth);
  renderSynthesis(box, hits, lastQueryVec);

  keepAtBottom();
}

/* ---------------- synthesis ---------------- */

const AUTO_KEY = 'ask:auto-synthesis';
let cap = null;                                  // cached capability probe
const autoOn = () => { try { return localStorage.getItem(AUTO_KEY) === '1'; } catch { return false; } };
const setAuto = (v) => { try { localStorage.setItem(AUTO_KEY, v ? '1' : '0'); } catch {} };

/**
 * The model is a big download, so it is never fetched behind the visitor's
 * back: the first answer is opt-in and the choice is remembered. Where WebGPU
 * is unavailable the quotes above are the whole feature, which is why they are
 * rendered first and in full.
 */
async function renderSynthesis(box, hits, queryVec) {
  // The zero-download answer first: it needs nothing the search did not already
  // load, so every visitor gets one immediately.
  box.replaceChildren(el('p', 'ask-synth-pending', 'Reading the passages…'));
  let picked = [];
  try {
    picked = await extractiveAnswer({ index: await indexPromise, hits, queryVec });
  } catch (err) {
    console.warn('[ask] extractive', err);
  }

  box.replaceChildren();
  if (picked.length) {
    const p = el('p', 'ask-synth-text');
    picked.forEach((s, i) => {
      if (i) p.appendChild(document.createTextNode(' '));
      if (hasMath(s.text)) {
        const span = el('span');
        renderMath(span, s.text + ' ');
        p.appendChild(span);
      } else {
        p.appendChild(document.createTextNode(s.text + ' '));
      }
      p.appendChild(el('span', 'ask-synth-cite', `[${s.n}]`));
    });
    box.appendChild(p);
    box.appendChild(el('p', 'ask-synth-caveat',
      'Assembled from the passages above — the sentences that best match your question, quoted exactly and in their original order. Nothing here is paraphrased.'));
  } else {
    box.appendChild(el('p', 'ask-synth-pending', 'No single passage answers this directly — the sources above are the closest material.'));
  }

  if (isEngineReady() || autoOn()) return runSynthesis(box, hits);
  box.appendChild(offerModel(box, hits));
}

/** Optional upgrade: real prose, at the cost of a large one-time download. */
function offerModel(box, hits) {
  const wrap = el('div', 'ask-synth-upgrade');
  const slow = cap?.mode === 'cpu';
  wrap.appendChild(el('p', 'ask-synth-pending', slow
    ? `Prefer it rewritten as prose? A small language model can do that on your CPU — ${cap.model.download} to download once, then roughly half a minute per answer. It paraphrases, so check it against the quotes.`
    : `Prefer it rewritten as prose? ${cap?.model?.label ?? 'A small model'} can do that on your GPU — ${cap?.model?.download ?? ''} to download once. It paraphrases, so check it against the quotes.`));
  const go = el('button', 'ask-concept-go', 'Rewrite as prose');
  go.type = 'button';
  go.addEventListener('click', () => { setAuto(true); runSynthesis(box, hits); });
  wrap.appendChild(go);
  return wrap;
}

async function runSynthesis(box, hits) {
  cap ??= await capability();
  const status = el('p', 'ask-synth-pending', 'Loading the model…');
  const bar = el('div', 'ask-progress');
  const fill = el('i');
  bar.appendChild(fill);
  box.replaceChildren(status, bar);

  try {
    const engine = await loadEngine(cap.model, cap.mode, ({ fraction, text }) => {
      fill.style.width = `${Math.round((fraction || 0) * 100)}%`;
      if (text) status.textContent = text;
    });

    const out = el('p', 'ask-synth-text');
    box.replaceChildren(out);
    await streamAnswer({
      engine,
      question: input.value,
      hits,
      onToken: (t) => { out.textContent += t; keepAtBottom(); },
    });
    if (!out.textContent.trim()) out.textContent = 'The model returned nothing. The quoted passages above still stand.';
    box.appendChild(el('p', 'ask-synth-caveat',
      `Written by ${cap.model.label} running in your browser, from the excerpts above. Check it against them.`));
  } catch (err) {
    console.error('[ask] synthesis', err);
    box.replaceChildren(el('p', 'ask-synth-pending',
      `The model could not run: ${err?.message ?? err}. The quoted passages above are unaffected.`));
  }
}

/* ---------------- query flow ---------------- */

let indexPromise = null;
let busy = false;
let lastQueryVec = null;   // reused by the extractive tier to score sentences

async function run(query) {
  if (busy || !query.trim()) return;
  busy = true;
  send.disabled = true;
  stick = true;

  try {
    if (!indexPromise) indexPromise = loadIndex(corpusUrl);
    showStatus('Loading index…', 'A small static file of prewritten excerpts.');
    const index = await indexPromise;

    if (!isEmbedderReady()) {
      showStatus('Loading the search model…', 'One-time download, then cached by your browser. Nothing is sent to a server.', { progress: 0 });
      await loadEmbedder(index, (fraction, label) => {
        showStatus('Loading the search model…', `${label} — cached after this.`, { progress: fraction });
      });
    }

    showStatus('Searching…', 'Embedding your question and comparing it against the index.');
    const { hits, ranked, queryVec, weak } = await search(index, query, 4);
    lastQueryVec = queryVec;

    if (!hits.length) {
      galaxy?.clear();   // nothing matched: the galaxy must not stay lit from before
      showStatus('No close match', 'Nothing in this page\'s sources looks relevant. Try different wording, or ask on another page — each page searches its own material.');
      return;
    }
    showResults(hits, weak);

    // Light the concepts this question is about — named in the query, or
    // carried by the passages that ranked highest (deeper than the four shown).
    if (galaxy?.graph) {
      const g = prepareGraph(galaxy.graph);
      galaxy.setActive(activate(g, index.corpus, ranked.map((h) => h.i), query, { queryVec }));
    }
  } catch (err) {
    galaxy?.clear();
    console.error('[ask]', err);
    showStatus('Something went wrong', String(err && err.message ? err.message : err), { error: true });
  } finally {
    busy = false;
    send.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  run(input.value);
});

showIdle();
