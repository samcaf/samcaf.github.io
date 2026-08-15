/**
 * gen-worker.js — CPU text generation, off the main thread.
 *
 * WASM inference is synchronous and takes tens of seconds, so running it in the
 * page freezes everything: the panel stops scrolling, the galaxy stops turning,
 * the close button stops responding. Here it blocks only this worker, and tokens
 * arrive by message as they are produced.
 *
 * Protocol
 *   in   {type:'load', model}            {type:'generate', id, messages, maxTokens}
 *   out  {type:'progress', fraction, text} {type:'ready'}
 *        {type:'token', id, text}          {type:'done', id}  {type:'error', message}
 */

import { pipeline, TextStreamer, env } from './vendor/transformers.min.js';

const VENDOR = new URL('./vendor/', import.meta.url).href;

env.allowLocalModels = true;
env.allowRemoteModels = true;   // the generator is fetched from the HF CDN
env.localModelPath = new URL('../../assets/models/', import.meta.url).href;
env.backends.onnx.wasm.wasmPaths = VENDOR + 'ort/';
env.backends.onnx.wasm.numThreads = 1;   // no cross-origin isolation on Pages

let generator = null;

async function load(model, dtype = 'q4f16') {
  if (generator) return;
  const files = new Map();
  generator = await pipeline('text-generation', model, {
    dtype,
    device: 'wasm',
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      if (total) {
        self.postMessage({
          type: 'progress',
          fraction: loaded / total,
          text: `Downloading ${(loaded / 1048576).toFixed(0)} of ${(total / 1048576).toFixed(0)} MB`,
        });
      }
    },
  });
  self.postMessage({ type: 'ready' });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') {
      await load(msg.model, msg.dtype);
      return;
    }
    if (msg.type === 'generate') {
      await load(msg.model, msg.dtype);
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => self.postMessage({ type: 'token', id: msg.id, text }),
      });
      await generator(msg.messages, { max_new_tokens: msg.maxTokens ?? 260, do_sample: false, streamer });
      self.postMessage({ type: 'done', id: msg.id });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err?.message ?? err) });
  }
};
