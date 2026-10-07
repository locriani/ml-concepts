// SmolLM2-135M loader. Weights download from Hugging Face and persist in Origin Private File System.
// Tokenizer loads from a pinned transformers.js CDN build.

import { parseSafetensors } from './llama.js';

const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const REPO = 'HuggingFaceTB/SmolLM2-135M';
const URL = `https://huggingface.co/${REPO}/resolve/main/model.safetensors`;
const BYTES = 269060552; // Expected byte count used for verification of cached weights.
const FILE = 'smollm2-135m.safetensors';

// Origin Private File System supports storage for 269 MB weight buffers.
const stored = async () => {
  try { const f = await (await (await navigator.storage.getDirectory()).getFileHandle(FILE)).getFile(); return f.size === BYTES ? f : null; } catch { return null; }
};
export const isCached = async () => !!(await stored());

// onProgress receives float values between 0 and 1. Resolves to parsed tensors and config.
export async function loadWeights(onProgress) {
  const hit = await stored();
  if (hit) return parseSafetensors(await hit.arrayBuffer());

  const res = await fetch(URL);
  if (!res.ok) throw new Error(`weights download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || BYTES;
  const reader = res.body.getReader();
  const buf = new Uint8Array(BYTES); // Filled in place without intermediate copy.
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
    await w.write(buf); await w.close(); // Committed atomically on close.
  } catch { /* Storage write failures fall back to downloading again on next visit. */ }
  return parseSafetensors(buf.buffer);
}

let tok;
// Encodes text into byte-pair tokens without special tokens. Retains leading whitespace in pieces.
export async function tokenize(text) {
  if (!tok) {
    const { AutoTokenizer, env } = await import(TJS);
    env.allowLocalModels = false;
    tok = await AutoTokenizer.from_pretrained(REPO);
  }
  const ids = Array.from(tok.encode(text), Number);
  return { ids, pieces: ids.map(piece) };
}

// Decodes a single token id to string representation.
export const piece = (id) => tok.decode([id]);
