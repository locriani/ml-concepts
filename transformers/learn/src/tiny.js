// One-layer causal LM with the same block as SmolLM2: RMSNorm, RoPE, causal attention, SwiGLU, tied embeddings.
// ponytail: one layer and no grouped-query attention. Stack this block, or share K/V, only if the lesson is depth or GQA.
import { rope } from '../../src/llama.js';

export const CFG = { d: 16, heads: 2, ff: 32, theta: 1e4, eps: 1e-5 };
export const SEED = 7;
export const LINES = [
  'The cat sat on the mat because it was tired.',
  'A dog ran across the park after the ball.',
  'Some birds fly south before the winter.',
  'Those fish swim deep under the ice.',
];
export const PARAMS = ['embed', 'norm', 'ln1', 'ln2', 'wq', 'wk', 'wv', 'wo', 'wgate', 'wup', 'wdown'];

const z = (n) => new Float64Array(n);

export const silu = (x) => x / (1 + Math.exp(-x));
export const siluPrime = (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); };

export function tokenize(text) {
  const words = text.trim().toLowerCase().split(/\s+/).map((w) => w.replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, '')).filter(Boolean).slice(0, 24);
  const vocab = [];
  const ids = words.map((w) => {
    let i = vocab.indexOf(w);
    if (i < 0) { i = vocab.length; vocab.push(w); }
    return i;
  });
  return { words, ids, vocab };
}

export function sentences(lines) {
  const vocab = [];
  const sents = [];
  for (const line of lines) {
    const { words } = tokenize(line);
    if (words.length < 2) continue;
    const ids = words.map((w) => {
      let i = vocab.indexOf(w);
      if (i < 0) { i = vocab.length; vocab.push(w); }
      return i;
    });
    sents.push({ words, ids });
  }
  return { vocab, sents };
}

export function dims(W, key) {
  const { d, ff, vocab } = W.cfg;
  if (key === 'embed') return { rows: vocab, cols: d };
  if (key === 'wgate' || key === 'wup') return { rows: ff, cols: d };
  if (key === 'wdown') return { rows: d, cols: ff };
  if (key === 'ln1' || key === 'ln2' || key === 'norm') return { rows: 1, cols: d };
  return { rows: d, cols: d };
}

export function init(vocab, seed = SEED) {
  let s = seed;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 * 2 - 1; };
  const arr = (n, scale) => Float64Array.from({ length: n }, () => rnd() * scale);
  const ones = (n) => { const a = z(n); a.fill(1); return a; };
  const { d, ff } = CFG, scale = 0.2;
  return {
    cfg: { ...CFG, vocab, dh: d / CFG.heads },
    embed: arr(vocab * d, scale),
    norm: ones(d), ln1: ones(d), ln2: ones(d),
    wq: arr(d * d, scale), wk: arr(d * d, scale), wv: arr(d * d, scale), wo: arr(d * d, scale),
    wgate: arr(ff * d, scale), wup: arr(ff * d, scale), wdown: arr(d * ff, scale),
  };
}

function matvec(W, x, outD) {
  const y = z(outD), n = x.length;
  for (let o = 0; o < outD; o++) {
    let s = 0;
    const row = o * n;
    for (let k = 0; k < n; k++) s += W[row + k] * x[k];
    y[o] = s;
  }
  return y;
}

function rmsNorm(x, g, eps) {
  const d = x.length, y = z(d);
  let s = 0;
  for (let k = 0; k < d; k++) s += x[k] * x[k];
  const inv = 1 / Math.sqrt(s / d + eps);
  for (let k = 0; k < d; k++) y[k] = x[k] * inv * g[k];
  return y;
}

function softmax(logits) {
  let m = -Infinity;
  for (let v = 0; v < logits.length; v++) if (logits[v] > m) m = logits[v];
  const prob = z(logits.length);
  let s = 0;
  for (let v = 0; v < logits.length; v++) { prob[v] = Math.exp(logits[v] - m); s += prob[v]; }
  for (let v = 0; v < logits.length; v++) prob[v] /= s;
  return prob;
}

// Teacher-forced forward. Position i predicts ids[i + 1]. The last position has no target.
export function forward(W, ids) {
  const { d, heads, dh, ff, theta, eps, vocab } = W.cfg, n = ids.length, scale = 1 / Math.sqrt(dh);
  const scores = Array.from({ length: heads }, () => { const a = z(n * n); a.fill(-Infinity); return a; });
  const attn = Array.from({ length: heads }, () => z(n * n));
  const pos = [];
  for (let i = 0; i < n; i++) {
    const x0 = Float64Array.from(W.embed.subarray(ids[i] * d, (ids[i] + 1) * d));
    const ln1 = rmsNorm(x0, W.ln1, eps);
    const qPre = matvec(W.wq, ln1, d), kPre = matvec(W.wk, ln1, d);
    const q = rope(Float64Array.from(qPre), i, dh, theta);
    const k = rope(Float64Array.from(kPre), i, dh, theta);
    const v = matvec(W.wv, ln1, d);
    pos.push({ x0, ln1, qPre, kPre, q, k, v });
  }
  for (let i = 0; i < n; i++) {
    const ctx = z(d);
    for (let h = 0; h < heads; h++) {
      let m = -Infinity;
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let c = 0; c < dh; c++) s += pos[i].q[h * dh + c] * pos[j].k[h * dh + c];
        scores[h][i * n + j] = s * scale;
        if (scores[h][i * n + j] > m) m = scores[h][i * n + j];
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) { attn[h][i * n + j] = Math.exp(scores[h][i * n + j] - m); sum += attn[h][i * n + j]; }
      for (let j = 0; j <= i; j++) {
        attn[h][i * n + j] /= sum;
        for (let c = 0; c < dh; c++) ctx[h * dh + c] += attn[h][i * n + j] * pos[j].v[h * dh + c];
      }
    }
    const attnOut = matvec(W.wo, ctx, d);
    const res1 = z(d);
    for (let k = 0; k < d; k++) res1[k] = pos[i].x0[k] + attnOut[k];
    const ln2 = rmsNorm(res1, W.ln2, eps);
    const gate = matvec(W.wgate, ln2, ff), up = matvec(W.wup, ln2, ff), hidden = z(ff);
    for (let k = 0; k < ff; k++) hidden[k] = silu(gate[k]) * up[k];
    const ffOut = matvec(W.wdown, hidden, d);
    const x1 = z(d);
    for (let k = 0; k < d; k++) x1[k] = res1[k] + ffOut[k];
    const lnF = rmsNorm(x1, W.norm, eps);
    const logits = z(vocab);
    for (let v = 0; v < vocab; v++) {
      let s = 0;
      const row = v * d;
      for (let k = 0; k < d; k++) s += lnF[k] * W.embed[row + k];
      logits[v] = s;
    }
    const prob = softmax(logits);
    const target = i + 1 < n ? ids[i + 1] : -1;
    const ce = target < 0 ? null : -Math.log(prob[target]);
    Object.assign(pos[i], { ctx, attnOut, res1, ln2, gate, up, hidden, ffOut, x1, lnF, logits, prob, target, ce });
  }
  const nPred = Math.max(n - 1, 0);
  let loss = 0;
  for (let i = 0; i < nPred; i++) loss += pos[i].ce;
  if (nPred) loss /= nPred;
  return { ids, n, loss, nPred, scores, attn, pos };
}

function matvecBackward(W, x, dy, dW, dx) {
  const n = x.length;
  for (let o = 0; o < dy.length; o++) {
    const go = dy[o], row = o * n;
    for (let k = 0; k < n; k++) {
      dW[row + k] += go * x[k];
      dx[k] += W[row + k] * go;
    }
  }
}

function rmsBackward(x, g, eps, dy, dg, dx) {
  const d = x.length;
  let ss = 0;
  for (let k = 0; k < d; k++) ss += x[k] * x[k];
  const inv = 1 / Math.sqrt(ss / d + eps);
  const dz = z(d);
  let dot = 0;
  for (let k = 0; k < d; k++) {
    dz[k] = dy[k] * g[k];
    dg[k] += dy[k] * x[k] * inv;
    dot += dz[k] * x[k];
  }
  const inv3 = inv * inv * inv;
  for (let m = 0; m < d; m++) dx[m] += dz[m] * inv - inv3 * x[m] * dot / d;
}

// Inverse of rope(): the rotation is orthogonal, so the gradient rotates by the opposite angle.
function ropeBackward(dRot, pos, dh, theta, dPre) {
  const h = dh / 2;
  for (let off = 0; off < dRot.length; off += dh) {
    for (let i = 0; i < h; i++) {
      const a = pos * Math.pow(theta, -2 * i / dh), c = Math.cos(a), s = Math.sin(a);
      const dx = dRot[off + i], dy = dRot[off + i + h];
      dPre[off + i] += dx * c + dy * s;
      dPre[off + i + h] += -dx * s + dy * c;
    }
  }
}

function addScaled(dst, src, off, n, s) {
  for (let c = 0; c < n; c++) dst[off + c] += s * src[off + c];
}

function dotAt(a, b, off, n) {
  let s = 0;
  for (let c = 0; c < n; c++) s += a[off + c] * b[off + c];
  return s;
}

export function backward(W, tape) {
  const { d, heads, dh, ff, theta, eps, vocab } = W.cfg, { n, nPred, pos, attn } = tape;
  const g = Object.fromEntries(PARAMS.map((k) => [k, z(W[k].length)]));
  const dEmbedLookup = z(vocab * d), dEmbedUnembed = z(vocab * d);
  const dScores = Array.from({ length: heads }, () => z(n * n));
  const dAttn = Array.from({ length: heads }, () => z(n * n));
  const fields = ['x0', 'ln1', 'qPre', 'kPre', 'q', 'k', 'v', 'ctx', 'attnOut', 'res1', 'ln2', 'gate', 'up', 'hidden', 'ffOut', 'x1', 'lnF', 'logits', 'skipMlp', 'branchMlp', 'skipAttn', 'branchAttn'];
  for (const p of pos) {
    p.d = {};
    for (const f of fields) p.d[f] = z(f === 'gate' || f === 'up' || f === 'hidden' ? ff : f === 'logits' ? vocab : d);
  }
  for (let i = 0; i < nPred; i++) {
    const p = pos[i], prob = p.prob;
    for (let v = 0; v < vocab; v++) p.d.logits[v] = (prob[v] - (v === p.target ? 1 : 0)) / nPred;
    for (let v = 0; v < vocab; v++) {
      const gl = p.d.logits[v], row = v * d;
      for (let k = 0; k < d; k++) {
        p.d.lnF[k] += gl * W.embed[row + k];
        dEmbedUnembed[row + k] += gl * p.lnF[k];
      }
    }
    rmsBackward(p.x1, W.norm, eps, p.d.lnF, g.norm, p.d.x1);
  }
  for (let i = 0; i < n; i++) {
    const p = pos[i];
    p.d.skipMlp.set(p.d.x1);
    p.d.ffOut.set(p.d.x1);
    p.d.res1.set(p.d.x1);
    matvecBackward(W.wdown, p.hidden, p.d.ffOut, g.wdown, p.d.hidden);
    for (let k = 0; k < ff; k++) {
      const sg = silu(p.gate[k]);
      p.d.up[k] = p.d.hidden[k] * sg;
      p.d.gate[k] = p.d.hidden[k] * p.up[k] * siluPrime(p.gate[k]);
    }
    matvecBackward(W.wup, p.ln2, p.d.up, g.wup, p.d.ln2);
    matvecBackward(W.wgate, p.ln2, p.d.gate, g.wgate, p.d.ln2);
    rmsBackward(p.res1, W.ln2, eps, p.d.ln2, g.ln2, p.d.res1);
    for (let k = 0; k < d; k++) p.d.branchMlp[k] = p.d.res1[k] - p.d.skipMlp[k];
  }
  for (let i = 0; i < n; i++) {
    const p = pos[i];
    p.d.skipAttn.set(p.d.res1);
    p.d.attnOut.set(p.d.res1);
    p.d.x0.set(p.d.res1);
    matvecBackward(W.wo, p.ctx, p.d.attnOut, g.wo, p.d.ctx);
  }
  const scale = 1 / Math.sqrt(dh);
  for (let h = 0; h < heads; h++) {
    const off = h * dh;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        dAttn[h][i * n + j] = dotAt(pos[i].d.ctx, pos[j].v, off, dh);
        addScaled(pos[j].d.v, pos[i].d.ctx, off, dh, attn[h][i * n + j]);
      }
      let mix = 0;
      for (let j = 0; j <= i; j++) mix += attn[h][i * n + j] * dAttn[h][i * n + j];
      for (let j = 0; j <= i; j++) {
        const ds = attn[h][i * n + j] * (dAttn[h][i * n + j] - mix);
        dScores[h][i * n + j] = ds;
        const s = ds * scale;
        addScaled(pos[i].d.q, pos[j].k, off, dh, s);
        addScaled(pos[j].d.k, pos[i].q, off, dh, s);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const p = pos[i];
    ropeBackward(p.d.q, i, dh, theta, p.d.qPre);
    ropeBackward(p.d.k, i, dh, theta, p.d.kPre);
    matvecBackward(W.wq, p.ln1, p.d.qPre, g.wq, p.d.ln1);
    matvecBackward(W.wk, p.ln1, p.d.kPre, g.wk, p.d.ln1);
    matvecBackward(W.wv, p.ln1, p.d.v, g.wv, p.d.ln1);
    const skip = Float64Array.from(p.d.x0);
    rmsBackward(p.x0, W.ln1, eps, p.d.ln1, g.ln1, p.d.x0);
    for (let k = 0; k < d; k++) p.d.branchAttn[k] = p.d.x0[k] - skip[k];
    const row = tape.ids[i] * d;
    for (let k = 0; k < d; k++) dEmbedLookup[row + k] += p.d.x0[k];
  }
  for (let i = 0; i < g.embed.length; i++) g.embed[i] = dEmbedLookup[i] + dEmbedUnembed[i];
  tape.g = g;
  tape.dScores = dScores;
  tape.dAttn = dAttn;
  tape.dEmbedLookup = dEmbedLookup;
  tape.dEmbedUnembed = dEmbedUnembed;
  return tape;
}

export function learn(W, ids) {
  const tape = forward(W, ids);
  return backward(W, tape);
}

export function loss(W, ids) { return forward(W, ids).loss; }

export function sgd(W, g, lr) {
  for (const k of PARAMS) {
    const w = W[k], dw = g[k];
    for (let i = 0; i < w.length; i++) w[i] -= lr * dw[i];
  }
}

// Each sentence is one example. The training step uses the mean of their gradients.
export function meanGrad(tapes) {
  const g = {};
  for (const k of PARAMS) {
    const n = tapes[0].g[k].length, a = new Float64Array(n);
    for (const t of tapes) {
      const src = t.g[k];
      for (let i = 0; i < n; i++) a[i] += src[i];
    }
    for (let i = 0; i < n; i++) a[i] /= tapes.length;
    g[k] = a;
  }
  return g;
}
