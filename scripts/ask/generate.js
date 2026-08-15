/**
 * generate.js — the synthesis tier: a small language model, in the browser.
 *
 * Nothing is trained here. This loads pre-trained Qwen3 weights and prompts them
 * with the passages retrieval already found, so the model's job is to *read and
 * summarise*, never to recall. Quotes come from the index and are rendered
 * separately; anything the model writes is confined to its own zone and must
 * cite the excerpts by number.
 *
 * The library is vendored (same-origin), but the weights — ~1.1 GB for the 1.7B
 * — stream from the MLC CDN on first use and are then cached by the browser.
 * That is the one runtime dependency on a third party in the whole assistant;
 * the visitor's question still never leaves their machine.
 */

const VENDOR = new URL('./vendor/', import.meta.url).href;

export const MODELS = {
  primary: { id: 'Qwen3-1.7B-q4f16_1-MLC', label: 'Qwen3 1.7B', download: '~1.1 GB', vramMB: 2037 },
  small: { id: 'Qwen3-0.6B-q4f16_1-MLC', label: 'Qwen3 0.6B', download: '~500 MB', vramMB: 1403 },
  // No WebGPU? Fall back to ONNX on the CPU. Much smaller and much slower —
  // tens of seconds rather than a couple — but it runs in any browser, so the
  // written answer is not a privilege of the right graphics stack.
  // dtype matters more than it looks: 'q4' resolves to a 786 MB file that the
  // Cache API refuses to persist, so it re-downloaded on every visit. 'q4f16'
  // is 483 MB and caches.
  cpu: { id: 'onnx-community/Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 0.5B (CPU)', download: '~480 MB', dtype: 'q4f16', cpu: true },
};

const SYSTEM = `You answer questions about Samuel Alipour-fard's research using ONLY the numbered excerpts given.

Rules:
- End every sentence with the bracketed number of the excerpt it came from, like [1].
- Copy names, acronyms, and numbers exactly as they are spelled in the excerpts.
- If the excerpts do not answer the question, say exactly that in one sentence. Never fill the gap from memory.
- 2-3 sentences. No preamble, no restating the question. /no_think`;

/*
 * A one-shot exchange, because describing the citation format is not enough for
 * a 0.5B model — the CPU fallback followed every other instruction but silently
 * dropped the [n] markers until shown one. Kept tiny: on the CPU path every
 * prompt token costs real seconds.
 */
const SHOT = [
  {
    role: 'user',
    content: 'Excerpts:\n\n[1] (Example Paper, 2020)\nThe detector records proton collisions at 13 TeV.\n\n[2] (Example Paper, 2020)\nCalibration runs used lead ions instead.\n\nQuestion: What does the detector record?',
  },
  { role: 'assistant', content: 'It records proton collisions at 13 TeV [1]. Calibration runs used lead ions instead [2].' },
];

let enginePromise = null;
let loadedModelId = null;

/**
 * Can this browser run the model at all? A `navigator.gpu` object is not enough
 * — Chrome exposes it on machines with no usable adapter, so the adapter has to
 * be requested to know.
 */
export async function capability() {
  let adapter = null;
  if (navigator.gpu) {
    try {
      adapter = await navigator.gpu.requestAdapter();
    } catch { /* treated as no adapter */ }
  }

  // No usable GPU is not the end of the road — it just means the slow path.
  if (!adapter) {
    return {
      ok: true,
      mode: 'cpu',
      model: MODELS.cpu,
      reason: navigator.gpu ? 'no-adapter' : 'no-webgpu',
    };
  }

  // Pick by what the device can actually hold, not by hope.
  const budgetMB = (adapter.limits?.maxBufferSize ?? 0) / 1048576;
  const model = budgetMB && budgetMB < MODELS.primary.vramMB ? MODELS.small : MODELS.primary;
  return { ok: true, mode: 'webgpu', model, budgetMB: Math.round(budgetMB) };
}

export function isEngineReady() {
  return enginePromise !== null;
}

export function loadedModel() {
  return loadedModelId;
}

/** Load (and cache) an engine for either mode. onProgress gets {fraction, text}. */
export function loadEngine(model, mode, onProgress) {
  if (enginePromise && loadedModelId === model.id) return enginePromise;
  loadedModelId = model.id;

  enginePromise = (mode === 'cpu' ? loadCpuEngine : loadGpuEngine)(model, onProgress);
  enginePromise.catch(() => { enginePromise = null; loadedModelId = null; });
  return enginePromise;
}

async function loadGpuEngine(model, onProgress) {
  const webllm = await import(VENDOR + 'web-llm.js');
  const engine = await webllm.CreateMLCEngine(model.id, {
    initProgressCallback: (p) => onProgress?.({ fraction: p.progress ?? 0, text: p.text ?? '' }),
  });
  return { mode: 'webgpu', engine };
}

async function loadCpuEngine(model, onProgress) {
  const worker = new Worker(new URL('./gen-worker.js', import.meta.url), { type: 'module' });
  const listeners = new Set();
  worker.onmessage = (ev) => { for (const fn of listeners) fn(ev.data); };

  await new Promise((resolve, reject) => {
    const onMsg = (m) => {
      if (m.type === 'progress') onProgress?.({ fraction: m.fraction, text: m.text });
      else if (m.type === 'ready') { listeners.delete(onMsg); resolve(); }
      else if (m.type === 'error') { listeners.delete(onMsg); reject(new Error(m.message)); }
    };
    listeners.add(onMsg);
    worker.postMessage({ type: 'load', model: model.id });
  });

  return { mode: 'cpu', worker, listeners, modelId: model.id, dtype: model.dtype };
}

/**
 * Numbered excerpts, in the order the UI shows them, so [n] lines up.
 *
 * The papers carry their own bracketed reference markers, and a small model
 * cannot tell "[113]" in the body text from the "[3]" we asked it to cite —
 * it duly cited excerpt 113, which does not exist. Strip them, and state the
 * valid range so the only brackets in play are ours.
 */
function buildPrompt(question, hits) {
  const excerpts = hits.map((h, i) => {
    const src = h.chunk.source;
    const where = [src.title, src.loc].filter(Boolean).join(', ');
    const text = h.chunk.text.replace(/\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]/g, '').replace(/\s{2,}/g, ' ');
    return `[${i + 1}] (${where})\n${text}`;
  }).join('\n\n');
  return `Excerpts:\n\n${excerpts}\n\nQuestion: ${question}\n`
    + `(Cite only with numbers 1 to ${hits.length}.)`;
}

/**
 * Qwen3 emits chain-of-thought inside <think>…</think>. It is not an answer and
 * must not reach the page, but it arrives token by token, possibly with the tags
 * split across chunks — so filter the stream rather than the finished string.
 */
export function makeThinkFilter() {
  const OPEN = '<think>';
  const CLOSE = '</think>';
  let buffer = '';
  let thinking = false;

  // How many trailing characters could still turn out to be the start of `tag`.
  // Only those are held back; holding a fixed number would swallow the end of
  // every answer, and short answers whole.
  const heldFor = (buf, tag) => {
    for (let n = Math.min(buf.length, tag.length - 1); n > 0; n--) {
      if (buf.endsWith(tag.slice(0, n))) return n;
    }
    return 0;
  };

  const push = (delta) => {
    buffer += delta;
    let out = '';
    for (;;) {
      if (thinking) {
        const end = buffer.indexOf(CLOSE);
        if (end === -1) {
          buffer = buffer.slice(buffer.length - heldFor(buffer, CLOSE)); // discard thoughts
          return out;
        }
        buffer = buffer.slice(end + CLOSE.length);
        thinking = false;
      } else {
        const start = buffer.indexOf(OPEN);
        if (start === -1) {
          const hold = heldFor(buffer, OPEN);
          out += buffer.slice(0, buffer.length - hold);
          buffer = hold ? buffer.slice(buffer.length - hold) : '';
          return out;
        }
        out += buffer.slice(0, start);
        buffer = buffer.slice(start + OPEN.length);
        thinking = true;
      }
    }
  };

  // At end of stream nothing more can arrive, so any held-back tail is real text.
  push.flush = () => {
    if (thinking) { buffer = ''; return ''; }
    const rest = buffer;
    buffer = '';
    return rest;
  };
  return push;
}

/** Stream an answer. Calls onToken(text) with think-blocks already stripped. */
export async function streamAnswer({ engine, question, hits, onToken, signal }) {
  const filter = makeThinkFilter();
  let full = '';
  const emit = (text) => { if (text) { full += text; onToken?.(text); } };
  const messages = [
    { role: 'system', content: SYSTEM },
    ...SHOT,
    { role: 'user', content: buildPrompt(question, hits) },
  ];

  if (engine.mode === 'cpu') {
    const id = `${Date.now()}-${messages.length}`;
    await new Promise((resolve, reject) => {
      const onMsg = (m) => {
        if (m.id !== id) return;
        if (m.type === 'token') { if (!signal?.aborted) emit(filter(m.text)); }
        else if (m.type === 'done') { engine.listeners.delete(onMsg); resolve(); }
        else if (m.type === 'error') { engine.listeners.delete(onMsg); reject(new Error(m.message)); }
      };
      engine.listeners.add(onMsg);
      engine.worker.postMessage({ type: 'generate', id, model: engine.modelId, dtype: engine.dtype, messages });
    });
  } else {
    const stream = await engine.engine.chat.completions.create({
      stream: true,
      temperature: 0.3,
      max_tokens: 400,
      messages,
    });
    for await (const part of stream) {
      if (signal?.aborted) break;
      const delta = part.choices?.[0]?.delta?.content ?? '';
      if (delta) emit(filter(delta));
    }
  }

  emit(filter.flush());   // release the tail held back for a possible split tag
  return full.trim();
}
