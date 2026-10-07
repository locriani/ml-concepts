// Loading SmolLM2-135M: weights straight from the Hugging Face hub (kept in the browser's private file storage),
// tokenizer from transformers.js (Hugging Face, Apache-2.0), pinned CDN build.

import { parseSafetensors } from './llama.js';

const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const REPO = 'HuggingFaceTB/SmolLM2-135M';
const URL = `https://huggingface.co/${REPO}/resolve/main/model.safetensors`;
const BYTES = 269060552; // exact file size: also how a stored copy is recognised as complete
const FILE = 'smollm2-135m.safetensors';

// Origin Private File System: unlike the Cache API it takes a file this size without complaint.
const stored = async () => {
  try { const f = await (await (await navigator.storage.getDirectory()).getFileHandle(FILE)).getFile(); return f.size === BYTES ? f : null; } catch { return null; }
};
export const isCached = async () => !!(await stored());

// onProgress(0..1); resolves to { tensorName: Float32Array, cfg }
export async function loadWeights(onProgress) {
  const hit = await stored();
  if (hit) return parseSafetensors(await hit.arrayBuffer());

  const res = await fetch(URL);
  if (!res.ok) throw new Error(`weights download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || BYTES;
  const reader = res.body.getReader();
  const buf = new Uint8Array(BYTES); // filled in place: no second copy
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    onProgress(Math.min(1, got / total));
  }
  if (got !== BYTES) throw new Error(`weights download incomplete: ${got} of ${BYTES} bytes`);
  try {
    const w = await (await (await navigator.storage.getDirectory()).getFileHandle(FILE, { create: true })).createWritable();
    await w.write(buf); await w.close(); // committed atomically on close
  } catch { /* storage unavailable or full: fine, download again next visit */ }
  return parseSafetensors(buf.buffer);
}

let tok;
// text -> { ids, pieces }. Byte-pair tokens, no special tokens; pieces keep their leading space.
export async function tokenize(text) {
  if (!tok) {
    const { AutoTokenizer, env } = await import(TJS);
    env.allowLocalModels = false;
    tok = await AutoTokenizer.from_pretrained(REPO);
  }
  const ids = Array.from(tok.encode(text), Number);
  return { ids, pieces: ids.map(piece) };
}

// one token id -> its text (needs one prior tokenize() call to have loaded the tokenizer)
export const piece = (id) => tok.decode([id]);
