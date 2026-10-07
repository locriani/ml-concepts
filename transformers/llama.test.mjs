// Synthetic-weight checks for the Llama forward pass: no download needed.
import assert from 'node:assert';
import { newState, step, trace, topK, sample, rope, parseSafetensors } from './src/llama.js';

const cfg = { layers: 2, d: 16, heads: 4, kv: 2, ff: 12, theta: 1e4, eps: 1e-5 }, { d, heads, kv, ff, layers: L } = cfg, dh = d / heads, vocab = 11;
let seed = 7; const r = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
const rnd = (n, s = 0.5) => Float32Array.from({ length: n }, () => r() * s);
const ones = (n) => new Float32Array(n).fill(1);
const W = { cfg, 'model.embed_tokens.weight': rnd(vocab * d, 2), 'model.norm.weight': ones(d) };
for (let l = 0; l < L; l++) {
  const P = `model.layers.${l}.`;
  Object.assign(W, { [P + 'input_layernorm.weight']: ones(d), [P + 'post_attention_layernorm.weight']: ones(d),
    [P + 'self_attn.q_proj.weight']: rnd(d * d), [P + 'self_attn.k_proj.weight']: rnd(kv * dh * d), [P + 'self_attn.v_proj.weight']: rnd(kv * dh * d), [P + 'self_attn.o_proj.weight']: rnd(d * d),
    [P + 'mlp.gate_proj.weight']: rnd(ff * d), [P + 'mlp.up_proj.weight']: rnd(ff * d), [P + 'mlp.down_proj.weight']: rnd(d * ff) });
}

const ids = [3, 1, 4, 1, 5];
// one token at a time through the KV cache must equal feeding the whole prompt in one call
const a = newState(W), full = step(W, a, ids);
const b = newState(W); let inc;
for (const id of ids) inc = step(W, b, [id]);
full.logits.forEach((x, i) => assert(Math.abs(x - inc.logits[i]) < 1e-5, 'cache mismatch'));
// causal: the last token attends over exactly n tokens and each row sums to 1
assert.equal(full.attn[0][0].length, ids.length);
full.attn.forEach((rows) => rows.forEach((w) => assert(Math.abs(w.reduce((s, x) => s + x, 0) - 1) < 1e-5)));
// distribution + sampling
const tk = topK(full.logits, 3, 1);
assert(Math.abs(Array.from(tk.probs).reduce((s, x) => s + x, 0) / tk.sum - 1) < 1e-5);
assert(tk.top[0].p >= tk.top[1].p);
assert(sample(tk, 0.999999) < vocab);
// top-K: sampling never leaves the K kept tokens, and K=1 is the argmax
const t3 = topK(full.logits, 3, 1), kept = new Set(t3.top.map((x) => x.id));
for (const u of [0, 0.3, 0.7, 0.999999]) assert(kept.has(sample(t3, u)));
assert.equal(sample(topK(full.logits, 1, 1), 0.9), t3.top[0].id);

// RoPE: position 0 is the identity, rotation keeps length, and q.k depends only on the distance between positions
const v0 = rnd(dh, 2), w0 = rnd(dh, 2), len = (v) => Math.hypot(...v), dotp = (x, y) => x.reduce((s, t, i) => s + t * y[i], 0);
assert.deepEqual(Array.from(rope(Float32Array.from(v0), 0, dh, 1e4)), Array.from(v0));
assert(Math.abs(len(rope(Float32Array.from(v0), 9, dh, 1e4)) - len(v0)) < 1e-5);
const rel = (i, j) => dotp(rope(Float32Array.from(v0), i, dh, 1e4), rope(Float32Array.from(w0), j, dh, 1e4));
assert(Math.abs(rel(7, 4) - rel(13, 10)) < 1e-4, 'relative positions only');

// full pass keeping every intermediate: logits agree, masking and sums are right, the residual stream chains
const A = trace(W, ids), n = ids.length;
A.logits.forEach((x, i) => assert(Math.abs(x - full.logits[i]) < 1e-5, 'trace logits'));
for (let h = 0; h < heads; h++) for (let i = 0; i < n; i++) {
  let s = 0;
  for (let j = 0; j < n; j++) {
    const w = A.layers[1].attn[h][i * n + j];
    if (j > i) { assert.equal(w, 0); assert.equal(A.layers[1].scores[h][i * n + j], -Infinity); }
    s += w;
  }
  assert(Math.abs(s - 1) < 1e-5, 'row sums to 1');
}
A.layers[1].input.forEach((x, i) => assert.equal(x, A.layers[0].out[i])); // residual stream chains layer to layer
A.layers[0].resAttn.forEach((x, i) => assert(Math.abs(x - A.layers[0].input[i] - A.layers[0].attnOut[i]) < 1e-5));
A.layers[0].out.forEach((x, i) => assert(Math.abs(x - A.layers[0].resAttn[i] - A.layers[0].ffOut[i]) < 1e-5));
assert.equal(A.layers[0].ffHidden.length, n * ff);
A.layers[0].ffHidden.forEach((x, i) => assert(Math.abs(x - A.layers[0].gate[i] / (1 + Math.exp(-A.layers[0].gate[i])) * A.layers[0].up[i]) < 1e-5, 'hidden = silu(gate) * up'));
// attention walk-through: q, k, v reproduce scores, and weights x values reproduce each head's slice of ctx
for (const l of [0, 1]) for (let h = 0; h < heads; h++) for (let i = 0; i < n; i++) {
  const Lr = A.layers[l], g = Math.floor(h / (heads / kv)) * dh, out = new Float32Array(dh);
  for (let j = 0; j <= i; j++) {
    const qv = Lr.q.subarray(i * d + h * dh, i * d + (h + 1) * dh), kj = Lr.k.subarray(j * kv * dh + g, j * kv * dh + g + dh), vj = Lr.v.subarray(j * kv * dh + g, j * kv * dh + g + dh);
    assert(Math.abs(dotp(qv, kj) / Math.sqrt(dh) - Lr.scores[h][i * n + j]) < 1e-4, 'q.k/sqrt(dk) is the score');
    for (let c = 0; c < dh; c++) out[c] += Lr.attn[h][i * n + j] * vj[c];
  }
  out.forEach((x, c) => assert(Math.abs(x - Lr.ctx[i * d + h * dh + c]) < 1e-4, 'sum of weight x value is the head output'));
}

// safetensors: bf16 widens exactly, f32 stays f32, and a misaligned header still yields correct views
const bf = new Uint16Array([0x3f80, 0xc000, 0x4040]); // 1, -2, 3 in bf16
let hdr = JSON.stringify({ a: { dtype: 'BF16', shape: [3], data_offsets: [0, 6] }, b: { dtype: 'F32', shape: [1], data_offsets: [8, 12] } });
for (const pad of [0, 1]) {
  let h = hdr; while ((8 + h.length) % 4 !== (pad ? 1 : 0)) h += ' ';
  const buf = new ArrayBuffer(8 + h.length + 12);
  new DataView(buf).setBigUint64(0, BigInt(h.length), true);
  new Uint8Array(buf, 8, h.length).set(new TextEncoder().encode(h));
  const body = new Uint8Array(buf, 8 + h.length, 12);
  body.set(new Uint8Array(bf.buffer), 0); body.set(new Uint8Array(new Float32Array([2.5]).buffer), 8);
  const t = parseSafetensors(buf);
  assert.deepEqual(Array.from(t.a), [1, -2, 3]);
  assert.deepEqual(Array.from(t.b), [2.5]);
}
console.log('llama.test: ok');
