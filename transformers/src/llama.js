// SmolLM2-135M implementation reading published safetensors weights.
// Architecture uses pre-RMSNorm blocks and causal grouped-query attention with RoPE.
// Layers include SwiGLU feed-forward networks without bias parameters.
// The output projection ties to the token embedding matrix.
// Linear weight tensors use row-major layout where each row forms an output dimension.
// A KV cache provides constant compute cost per generated token.

export const SMOLLM2_135M = { layers: 30, d: 576, heads: 9, kv: 3, ff: 1536, theta: 1e5, eps: 1e-5 };

// Parses safetensors into named Float32Array tensors and configuration. Widening bf16 shifts 16 bits into the upper half of float32.
export function parseSafetensors(buf, cfg = SMOLLM2_135M) {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  let base = 8 + n;
  if (base % 4) { buf = buf.slice(base); base = 0; } // Aligns buffer offset for typed array views.
  const W = { cfg };
  for (const [k, v] of Object.entries(header)) {
    if (k === '__metadata__') continue;
    const [a, b] = v.data_offsets;
    if (v.dtype === 'F32') W[k] = new Float32Array(buf, base + a, (b - a) / 4);
    else if (v.dtype === 'BF16') {
      const src = new Uint16Array(buf, base + a, (b - a) / 2), out = new Float32Array(src.length), u = new Uint32Array(out.buffer);
      for (let i = 0; i < src.length; i++) u[i] = src[i] << 16;
      W[k] = out;
    }
  }
  return W;
}

const dot = (a, ao, b, bo, n) => { let s = 0; for (let k = 0; k < n; k++) s += a[ao + k] * b[bo + k]; return s; };

function rmsNorm(x, g, eps) {
  const d = x.length, y = new Float32Array(d);
  let s = 0; for (let k = 0; k < d; k++) s += x[k] * x[k];
  const inv = 1 / Math.sqrt(s / d + eps);
  for (let k = 0; k < d; k++) y[k] = x[k] * inv * g[k];
  return y;
}

// Computes matrix-vector product for weight dimensions outD by input length.
function matvec(W, x, outD) {
  const y = new Float32Array(outD), n = x.length;
  for (let o = 0; o < outD; o++) y[o] = dot(W, o * n, x, 0, n);
  return y;
}

const silu = (x) => x / (1 + Math.exp(-x));

// Rotary position embedding rotates paired dimensions across each head by pos times theta^(-2i/dh).
// Inner products between rotated vectors depend strictly on relative token distance.
export function rope(v, pos, dh, theta) {
  const h = dh / 2;
  for (let off = 0; off < v.length; off += dh) {
    for (let i = 0; i < h; i++) {
      const a = pos * Math.pow(theta, -2 * i / dh), c = Math.cos(a), s = Math.sin(a), x = v[off + i], y = v[off + i + h];
      v[off + i] = x * c - y * s;
      v[off + i + h] = y * c + x * s;
    }
  }
  return v;
}

const EMB = 'model.embed_tokens.weight';

export const newState = (W) => ({ n: 0, K: Array.from({ length: W.cfg.layers }, () => []), V: Array.from({ length: W.cfg.layers }, () => []) });

// Appends token ids and advances state. Returns next-token logits and attention weights.
// Array attn stores attention distributions over all visible tokens.
// Optional parameter rec records intermediate layer activations for tracing.
export function step(W, st, ids, rec = null) {
  const { d, heads, kv, ff, theta, eps } = W.cfg, dh = d / heads, group = heads / kv, emb = W[EMB];
  const keep = (slot, v) => { if (rec) slot.push(Float32Array.from(v)); };
  let x, attn;
  for (const id of ids) {
    const p = st.n;
    x = Float32Array.from(emb.subarray(id * d, (id + 1) * d));
    keep(rec?.embed.word, x);
    attn = [];
    for (let l = 0; l < st.K.length; l++) {
      const P = `model.layers.${l}.`, R = rec?.layers[l];
      keep(R?.input, x);
      const a = rmsNorm(x, W[P + 'input_layernorm.weight'], eps);
      keep(R?.ln1, a);
      const q = rope(matvec(W[P + 'self_attn.q_proj.weight'], a, d), p, dh, theta);
      keep(R?.q, q);
      st.K[l].push(rope(matvec(W[P + 'self_attn.k_proj.weight'], a, kv * dh), p, dh, theta));
      st.V[l].push(matvec(W[P + 'self_attn.v_proj.weight'], a, kv * dh));
      const ctx = new Float32Array(d), rows = [];
      for (let h = 0; h < heads; h++) {
        const g = Math.floor(h / group) * dh, sc = new Float32Array(p + 1), w = new Float32Array(p + 1); // Slice index for shared key and value head.
        let m = -Infinity;
        for (let j = 0; j <= p; j++) { sc[j] = dot(q, h * dh, st.K[l][j], g, dh) / Math.sqrt(dh); m = Math.max(m, sc[j]); }
        let s = 0; for (let j = 0; j <= p; j++) { w[j] = Math.exp(sc[j] - m); s += w[j]; }
        for (let j = 0; j <= p; j++) { w[j] /= s; for (let c = 0; c < dh; c++) ctx[h * dh + c] += w[j] * st.V[l][j][g + c]; }
        rows.push(w);
        if (R) { R.scores[h].push(sc); R.attn[h].push(w); }
      }
      attn.push(rows);
      keep(R?.ctx, ctx);
      const o = matvec(W[P + 'self_attn.o_proj.weight'], ctx, d);
      keep(R?.attnOut, o);
      for (let k = 0; k < d; k++) x[k] += o[k];
      keep(R?.resAttn, x);
      const m2 = rmsNorm(x, W[P + 'post_attention_layernorm.weight'], eps);
      keep(R?.ln2, m2);
      const gate = matvec(W[P + 'mlp.gate_proj.weight'], m2, ff), up = matvec(W[P + 'mlp.up_proj.weight'], m2, ff);
      keep(R?.gate, gate); keep(R?.up, up);
      for (let k = 0; k < ff; k++) gate[k] = silu(gate[k]) * up[k]; // SwiGLU activation: gate values modulate transmission of up-projection features.
      keep(R?.ffHidden, gate);
      const f = matvec(W[P + 'mlp.down_proj.weight'], gate, d);
      keep(R?.ffOut, f);
      for (let k = 0; k < d; k++) x[k] += f[k];
      keep(R?.out, x);
    }
    st.n++;
  }
  return { logits: logitsOf(W, x), attn };
}

// Computes vocabulary logits from residual stream vectors via final RMSNorm and tied embeddings.
export function logitsOf(W, h) {
  const emb = W[EMB], d = h.length, vocab = emb.length / d;
  const hf = rmsNorm(h, W['model.norm.weight'], W.cfg.eps), logits = new Float32Array(vocab);
  for (let v = 0; v < vocab; v++) logits[v] = dot(hf, 0, emb, v * d, d);
  return logits;
}

// Evaluates prompt tokens and records all intermediate activations as flat arrays.
// Score and attention matrices use dimensions n by n. Masked future positions evaluate to -Infinity and zero weight.
export function trace(W, ids) {
  const { d, heads } = W.cfg, n = ids.length, st = newState(W);
  const names = ['input', 'ln1', 'q', 'ctx', 'attnOut', 'resAttn', 'ln2', 'gate', 'up', 'ffHidden', 'ffOut', 'out'];
  const rec = { embed: { word: [] }, layers: st.K.map(() => ({ ...Object.fromEntries(names.map((k) => [k, []])), scores: Array.from({ length: heads }, () => []), attn: Array.from({ length: heads }, () => []) })) };
  const { logits } = step(W, st, ids, rec);
  const flat = (rows) => { const w = rows[0].length, a = new Float32Array(n * w); rows.forEach((r, i) => a.set(r, i * w)); return a; };
  const square = (rows, fill) => { const a = new Float32Array(n * n).fill(fill); rows.forEach((r, i) => a.set(r, i * n)); return a; };
  return {
    n, d, heads, kv: W.cfg.kv, ids, logits,
    embed: { word: flat(rec.embed.word) },
    layers: rec.layers.map((R, l) => ({
      ...Object.fromEntries(names.map((k) => [k, flat(R[k])])), k: flat(st.K[l]), v: flat(st.V[l]), // Arrays store rotated keys and values matching KV cache layout.
      scores: R.scores.map((r) => square(r, -Infinity)), attn: R.attn.map((r) => square(r, 0)),
    })),
  };
}

// Returns top k token predictions under temperature scaling via full vocabulary softmax.
export function topK(logits, k, t = 1) {
  let m = -Infinity;
  for (const x of logits) if (x > m) m = x;
  let s = 0;
  const e = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { e[i] = Math.exp((logits[i] - m) / t); s += e[i]; }
  const idx = Array.from(e.keys()).sort((a, b) => e[b] - e[a]).slice(0, k);
  return { top: idx.map((id) => ({ id, p: e[id] / s })), probs: e, sum: s };
}

// Samples a token id from top K candidates proportionally to normalised probabilities.
export function sample({ top }, rnd = Math.random()) {
  let r = rnd * top.reduce((s, x) => s + x.p, 0);
  for (const x of top) { r -= x.p; if (r <= 0) return x.id; }
  return top.at(-1).id;
}
