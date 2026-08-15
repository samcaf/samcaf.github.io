/**
 * math.js — typeset the LaTeX the index preserves.
 *
 * KaTeX is rendered to **MathML**, not HTML: the browser draws it with its own
 * math fonts, so nothing here ships a font file. The macros come from the
 * thesis preamble (rag/macros.json) — the equations are written in the author's
 * own shorthand (\alphas, \le, \acomm), and without those definitions almost
 * every excerpt with math would render as an error.
 *
 * Failure is always soft: anything KaTeX cannot parse is shown as its LaTeX
 * source, which is still more informative than the silence we had before.
 */

const MACROS_URL = new URL('../../rag/macros.json', import.meta.url).href;

let katexPromise = null;
let macrosPromise = null;

function load() {
  katexPromise ??= import('./vendor/katex.mjs').then((m) => m.default ?? m);
  macrosPromise ??= fetch(MACROS_URL)
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));
  return Promise.all([katexPromise, macrosPromise]);
}

/** Warm the renderer so the first answer doesn't wait on it. */
export function preloadMath() {
  load().catch(() => {});
}

// $$…$$ first: otherwise the single-$ branch would match its two halves.
const MATH = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;

export function hasMath(text) {
  return typeof text === 'string' && text.includes('$');
}

/**
 * Replace `el`'s contents with `text`, typesetting any math it contains.
 * Plain text goes in as text nodes — only KaTeX's own output is ever markup.
 */
export async function renderMath(el, text) {
  if (!hasMath(text)) { el.textContent = text; return; }

  let katex;
  let macros;
  try {
    [katex, macros] = await load();
  } catch {
    el.textContent = text;   // renderer unavailable: show the source
    return;
  }

  const frag = document.createDocumentFragment();
  let last = 0;
  let m;
  MATH.lastIndex = 0;
  while ((m = MATH.exec(text))) {
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    const body = m[1] ?? m[2];
    const display = m[1] != null;
    const span = document.createElement('span');
    span.className = display ? 'ask-math ask-math--display' : 'ask-math';

    // A body containing '$' means the delimiters were mispaired — usually a
    // display block clipped by a chunk boundary. Show it rather than guess.
    if (body.includes('$')) {
      span.textContent = m[0];
    } else {
      try {
        span.innerHTML = katex.renderToString(body, {
          displayMode: display,
          output: 'mathml',            // no fonts to ship; the browser draws it
          macros: { ...macros },       // copied: KaTeX writes \gdef into this
          throwOnError: false,
          strict: 'ignore',
        });
      } catch {
        span.textContent = m[0];
      }
    }
    frag.appendChild(span);
    last = MATH.lastIndex;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));

  el.replaceChildren(frag);
}
