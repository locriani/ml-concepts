// Learning series: one panel per step of the backward pass, then the update and the training loop.
import { createScene } from '../../src/space.js';
import { CFG, PARAMS, SEED, LINES, init, learn, sgd, meanGrad, dims, silu, siluPrime, tokenize, sentences } from './tiny.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const l2 = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0));
const fmt = (x) => { if (!Number.isFinite(x)) return '—'; const a = Math.abs(x); if (a !== 0 && (a < 0.001 || a >= 1000)) return x.toExponential(2); return x.toFixed(4); };
const argmax = (p) => { let m = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[m]) m = i; return m; };
const argmaxAbs = (a) => { let m = 0; for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[m])) m = i; return m; };

const BASE = [10, 13, 16], TEAL = [63, 197, 189], ORANGE = [255, 159, 69];
const C = { accent: '#3fc5bd', high: '#ff9f45', ok: '#4fc98a', crit: '#ff6b63' };
const mixv = (to, t) => BASE.map((x, i) => x + (to[i] - x) * t);
const mix = (to, t) => `rgb(${mixv(to, t).map(Math.round)})`;

const COLS = ['#3fc5bd', '#ff9f45', '#4fc98a', '#e8c44a', '#ff6b63', '#b07a8a'];
const RUN = 40;

let lines = LINES.slice();
let vocab, sents;
function repack() {
  ({ vocab, sents } = sentences(lines));
  lines = sents.map((s) => s.words.join(' '));
  if (si >= sents.length) si = Math.max(0, sents.length - 1);
  const n = sents[si] ? sents[si].words.length : 1;
  q = Math.min(q, n - 1);
}

let W, tapes, tok, tape, hist = [], stage = 0, si = 0, q = 0, head = 0, busy = false;
let walk;

const lr = () => +$('lr').value;
const word = (i) => tok.words[i];
const followed = () => tape.pos[q];
const correctCount = () => tape.pos.filter((p) => p.target >= 0 && argmax(p.prob) === p.target).length;

const STAGES = [
  ['Loss', 'Cross-entropy, the loss', 'Each row is one word predicting the next. The bar is the probability of the true next word. The number at the right is the loss at that position: the cross-entropy, how surprised the model was. Zero would mean it was certain and correct. The height on the graph is the mean loss over the training set.'],
  ['Logits', 'The gradient starts at the logits', 'These bars are ∂loss/∂logit. Logits are the scores before the softmax turns them into probabilities. The tall downward bar is the true next word: its gradient is probability minus one, so gradient descent raises that logit. Every other word gets a gradient equal to its probability, so those logits fall. Backpropagation starts here. The last word has no target, so its gradient is zero.'],
  ['Unembed', 'The unembedding writes the gradient onto a row', 'The unembedding scores the final vector against every row of the embedding table. The gradient on the true next word’s row is the logit gradient times that vector, which is why that row is the tall one. The same table is the embedding used for lookup, so the weights are tied. This tab is only the unembed half. The Embed tab adds the lookup half.'],
  ['Residual', 'A residual copies the gradient down the shortcut', 'A residual connection adds a branch onto a skip. Backpropagation sends one copy of the gradient straight back along the skip. The branch can add a further gradient, and the gradient that continues is those two added together. The three charts share a scale so the addition is visible.'],
  ['MLP', 'SwiGLU, and a closed gate blocks the gradient', 'The MLP hidden unit is silu(gate) times up. The product rule splits the incoming gradient across the gate and the up projection. The three bars are the unit with the largest gradient. A nearly closed gate stops the gradient, so that unit’s up-gradient stays small.'],
  ['Attention', 'The attention gradient says which earlier word should count', 'Attention mixes the tokens already read. The weights are a softmax, and the gradient on a weight says whether the loss wants that earlier token up-weighted or down-weighted. Future tokens are masked, so their gradient stays zero.'],
  ['Embed', 'Tied embeddings add both gradients into one row', 'The embedding row for this token receives the gradient from the lookup, after attention and the residual. It also receives the unembed gradient from every position that scored this word. Tied weights add those two gradients into the update.'],
  ['Step', 'One parameter, four gradients', 'These bars are the gradient of a single parameter, the entry where the sentences disagree most. Orange is a negative gradient. Gradient descent adds to that weight, because the update is weight minus learning rate times gradient. Teal is a positive gradient, so the weight shrinks. Train all applies the mean gradient. Train this sentence applies only that sentence’s gradient.'],
  ['Train', 'The mean gradient is what fits the training set', 'Forty steps of gradient descent on one sentence fit that sentence, and any next-word pair it shares with another. The sentences left out stay near the loss of a uniform guess. Forty steps on the mean gradient fit every sentence. Adding a sentence reinitializes the weights, because new words add embedding rows.'],
];

function dimLabels(vector, mark) {
  return Array.from({ length: vector.length }, (_, i) => (vector.length <= 18 || i === mark ? String(i) : ''));
}

function vbars(values, labels, mark = -1, scale) {
  let m = scale;
  if (m == null) {
    m = 0;
    for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]));
  }
  m = m || 1e-12;
  const words = labels && [...labels].some((s) => String(s).length > 2);
  const box = el('div', 'vbars' + (words ? ' words' : ''));
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const col = el('div', 'vcol' + (i === mark ? ' on' : ''));
    const name = labels && labels[i] ? labels[i] : String(i);
    col.title = `${name}  ${fmt(v)}`;
    const up = el('div', 'half up'), dn = el('div', 'half dn');
    if (v) {
      const bar = document.createElement('i');
      bar.style.height = `${Math.abs(v) / m * 100}%`;
      (v < 0 ? dn : up).append(bar);
    }
    col.append(up, dn);
    if (labels && labels[i]) col.append(el('span', words ? 'vlbl vert' : 'vlbl', esc(labels[i])));
    box.append(col);
  }
  return box;
}

function vec(name, desc, vector) {
  const mark = argmaxAbs(vector);
  const box = el('div', 'stack tight');
  box.append(section(name, desc));
  box.append(vbars(vector, dimLabels(vector, mark), mark));
  return box;
}

function vecs(name, desc, rows, labels, on = q) {
  let m = 0;
  for (const r of rows) for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i]));
  const box = el('div', 'stack tight');
  box.append(section(name, desc));
  rows.forEach((r, i) => {
    const hot = argmaxAbs(r);
    const line = el('div', 'stack tight');
    line.append(el('span', 'lbl' + (i === on ? ' hot' : ''), esc(labels[i][0])));
    line.append(vbars(r, i === 0 ? dimLabels(r, hot) : null, i === on ? hot : -1, m || 1e-12));
    box.append(line);
  });
  return box;
}

function addPicture(names, parts, mark) {
  let m = 0;
  for (const v of parts) for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
  const labs = dimLabels(parts[0], mark);
  const row = el('div', 'addpic');
  parts.forEach((v, i) => {
    if (i) row.append(el('span', 'bop', i === parts.length - 1 ? '=' : '+'));
    const cell = el('div', 'stack tight');
    cell.append(el('span', 'lbl', esc(names[i])));
    cell.append(vbars(v, labs, mark, m || 1e-12));
    row.append(cell);
  });
  return row;
}

function withPath(nodes) {
  return [
    el('p', 'note', `Gradient of “${esc(word(q))}” in “${esc(sents[si].words.join(' '))}”. Backpropagation walks it from the logits to the embedding.`),
    legend(),
    ...nodes,
  ];
}

const labelsOf = () => tok.words.map((w, i) => [w, String(i)]);
const note = (html) => el('p', 'note mono', html);
const legend = () => el('div', 'legend', `<span><i style="background:${C.high}"></i>negative gradient, descent increases it</span><span><i style="background:${C.accent}"></i>positive gradient, descent decreases it</span>`);
const section = (title, desc) => el('div', 'controls', `<span class="step-name">${esc(title)}</span><span class="muted small">${esc(desc)}</span>`);

function heat(at, n, title, signed) {
  let m = 1e-12;
  if (signed) for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) m = Math.max(m, Math.abs(at[i * n + j]));
  const wrap = el('div', 'scroll'), g = el('div', 'heat');
  const hd = el('div', 'heat-r hrow');
  hd.append(el('div', 'rowbtn'));
  tok.words.forEach((w) => hd.append(el('div', 'hd', esc(w))));
  g.append(hd);
  for (let i = 0; i < n; i++) {
    const r = el('div', 'heat-r'), b = el('button', 'rowbtn' + (i === q ? ' on' : ''), esc(word(i)));
    b.onclick = () => { q = i; render(); };
    r.append(b);
    for (let j = 0; j < n; j++) {
      const c = el('div', 'hc');
      if (j > i) c.style.background = 'var(--sunk)';
      else if (!signed) {
        const v = at[i * n + j];
        c.textContent = v >= 0.05 ? (v > 0.995 ? '1' : v.toFixed(2).slice(1)) : '';
        c.style.background = mix(TEAL, Math.sqrt(v));
        c.style.color = v > 0.45 ? '#0a0d10' : 'var(--fg)';
      } else {
        const v = at[i * n + j], t = Math.sqrt(Math.min(1, Math.abs(v) / m));
        c.style.background = mix(v >= 0 ? TEAL : ORANGE, t);
        c.style.color = t > 0.55 ? '#0a0d10' : 'var(--fg)';
      }
      r.append(c);
    }
    g.append(r);
  }
  wrap.append(g);
  const box = el('div', 'stack tight');
  box.append(el('span', 'step-name', title), wrap);
  return box;
}

function headChips() {
  const row = el('div', 'chips');
  for (let h = 0; h < W.cfg.heads; h++) {
    const b = el('button', 'chip small' + (h === head ? ' on' : ''), 'head ' + h);
    b.onclick = () => { head = h; render(); };
    row.append(b);
  }
  return row;
}

function lossPanel() {
  const nPred = tape.nPred, uniform = Math.log(tok.vocab.length);
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl" style="flex:0 0 36px">#</span><span class="lbl" style="flex:1">word</span><span class="lbl" style="flex:1">true next</span><span class="lbl" style="flex:1">top guess</span><span class="lbl" style="flex:1.4">P(true)</span><span class="lbl num">loss</span>`));
  tape.pos.forEach((p, i) => {
    const last = p.target < 0;
    const guess = argmax(p.prob), hit = !last && guess === p.target;
    const pTrue = last ? 0 : p.prob[p.target];
    const r = el('button', 'tr' + (i === q ? ' on' : ''), `<span class="mono" style="flex:0 0 36px">${i}</span><span class="mono" style="flex:1">${esc(word(i))}</span><span class="mono" style="flex:1">${last ? '—' : esc(tok.vocab[p.target])}</span><span class="mono" style="flex:1;color:${last ? 'inherit' : hit ? C.ok : C.crit}">${esc(tok.vocab[guess])}</span><div class="bar"><i style="width:${last ? 0 : pTrue * 100}%"></i></div><span class="num">${last ? '—' : fmt(p.ce)}</span>`);
    r.onclick = () => { q = i; render(); };
    t.append(r);
  });
  const wrap = el('div', 'scroll');
  wrap.append(t);
  return [
    el('p', 'note', `This sentence’s loss is ${fmt(tape.loss)} over ${nPred} predictions. A uniform guess over ${tok.vocab.length} words has loss ${fmt(uniform)}. The graph’s height, the mean loss of the training set, is ${fmt(meanLoss())}. ${correctCount()} of ${nPred} next words are the model’s top guess.`),
    wrap,
  ];
}

function signalPanel() {
  const p = followed();
  if (p.target < 0) {
    return withPath([el('p', 'note', `${esc(word(q))} is the last token, so it has no target. The loss does not depend on it, its logit gradient is zero, and it sends no gradient backward.`)]);
  }
  const target = tok.vocab[p.target];
  return withPath([
    note(`∂logit(${esc(target)}) = (${fmt(p.prob[p.target])} − 1) / ${tape.nPred} = ${fmt(p.d.logits[p.target])}. The other words take probability / ${tape.nPred}. The bars sum to zero.`),
    vbars(p.d.logits, tok.vocab, p.target),
  ]);
}

function unembedPanel() {
  const p = followed(), d = W.cfg.d;
  const rows = tape.pos.map((x) => x.d.lnF);
  const out = tape.pos.map((x) => x.d.x1);
  const gain = vec('Final norm gain', 'Gradient on the learned scale applied after the last RMSNorm. One value per dimension, summed over every predicted position.', tape.g.norm);
  let call = el('p', 'note', 'Follow a position that has a next word to see that word’s unembed contribution.');
  if (p.target >= 0) {
    const gl = p.d.logits[p.target];
    const k = argmaxAbs(p.lnF);
    call = note(`${esc(word(q))} scores “${esc(tok.vocab[p.target])}”. At dim ${k}, ∂row = ∂logit ${fmt(gl)} × state ${fmt(p.lnF[k])} = ${fmt(gl * p.lnF[k])}. The row below sums this contribution from every position.`);
  }
  const embedRows = tok.vocab.map((_, v) => tape.dEmbedUnembed.subarray(v * d, (v + 1) * d));
  return withPath([
    vecs('∂ final state', 'Gradient on the normalized vector that dots with the embedding table. One row per position. The rows share a scale, so a flat row is a position the loss does not score.', rows, labelsOf()),
    gain,
    vecs('∂ layer output', 'The same gradient after the final RMSNorm. This is what enters the residual stream.', out, labelsOf()),
    call,
    vecs('∂ embedding, unembed half', 'Each row is one vocabulary word, on one shared scale. The true next word’s row is the tall one. Lookup gradients are on the Embed panel and add to these rows because the table is tied.', embedRows, tok.vocab.map((w) => [w, '']), p.target),
  ]);
}

function splitBlock(title, desc, arriving, returned, leaving) {
  const k = argmaxAbs(returned);
  return [
    section(title, desc),
    addPicture(['arriving', 'returned by the branch', 'leaving'], [arriving, returned, leaving], k),
    note(`dim ${k}: ${fmt(arriving[k])} + ${fmt(returned[k])} = ${fmt(leaving[k])}. The three charts share a scale.`),
  ];
}

function residualPanel() {
  const p = followed();
  return withPath([
    ...splitBlock('MLP add', `${word(q)} adds the MLP output onto the stream. The branch returns an extra gradient because RMSNorm 2 reads that stream.`, p.d.skipMlp, p.d.branchMlp, p.d.res1),
    ...splitBlock('Attention add', 'The attention output adds on in the same way. Its branch returns through the first RMSNorm, which reads the token vector.', p.d.skipAttn, p.d.branchAttn, p.d.x0),
  ]);
}

function mlpPanel() {
  const p = followed(), u = argmaxAbs(p.d.hidden);
  const sg = silu(p.gate[u]), sp = siluPrime(p.gate[u]);
  const dh = p.d.hidden[u], dup = dh * sg, dg = dh * p.up[u] * sp;
  const max = Math.max(Math.abs(dh), Math.abs(dup), Math.abs(dg), 1e-9);
  const prod = el('div', 'table');
  [['∂hidden', dh], ['∂up', dup], ['∂gate', dg]].forEach(([lab, v]) => prod.append(pullRow(lab, v, max, false)));
  return withPath([
    el('p', 'note', `Hidden unit ${u} has the largest absolute hidden gradient. The three bars share a scale.`),
    note(`∂up = ${fmt(dh)} × silu(gate) ${fmt(sg)} = ${fmt(dup)}`),
    note(`∂gate = ${fmt(dh)} × up ${fmt(p.up[u])} × silu′(gate) ${fmt(sp)} = ${fmt(dg)}`),
    prod,
    vec('∂ MLP output', 'Gradient on the vector added into the residual stream.', p.d.ffOut),
    vec('∂ hidden', 'Gradient before the down projection. silu(gate) × up lives here.', p.d.hidden),
    vec('∂ gate', 'Hidden gradient times the up value times the slope of silu.', p.d.gate),
    vec('∂ up', 'Hidden gradient times silu(gate). A closed gate blocks this path.', p.d.up),
    vec('∂ RMSNorm 2', 'Both projections read this normalized vector, so their gradients add here.', p.d.ln2),
  ]);
}

function attentionPanel() {
  const n = tape.n, at = tape.attn[head], da = tape.dAttn[head], ds = tape.dScores[head];
  let mix = 0, best = 0;
  for (let j = 0; j <= q; j++) {
    mix += at[q * n + j] * da[q * n + j];
    if (Math.abs(ds[q * n + j]) > Math.abs(ds[q * n + best])) best = j;
  }
  const a = at[q * n + best], g = da[q * n + best];
  const jac = note(`key “${esc(word(best))}”: ∂score = weight ${fmt(a)} × (∂weight ${fmt(g)} − ${fmt(mix)}) = ${fmt(a * (g - mix))}`);
  const keys = tok.words.slice(0, q + 1);
  const take = (src) => Float64Array.from({ length: q + 1 }, (_, j) => src[q * n + j]);
  return withPath([
    headChips(),
    el('p', 'note', `Query “${esc(word(q))}”, head ${head}. Rows attend to columns. Cells above the diagonal are masked. Orange is a negative gradient, teal a positive one.`),
    heat(at, n, 'Attention weights', false),
    heat(da, n, '∂loss / ∂attention weight', true),
    heat(ds, n, '∂loss / ∂score', true),
    section(`Followed row · ${word(q)}`, 'Score gradient on the keys this query can see. The tallest bar is the key the loss most wants to reweight.'),
    vbars(take(ds), keys, best),
    jac,
  ]);
}

function projectPanel() {
  const p = followed(), d = W.cfg.d, dh = W.cfg.dh, id = tok.ids[q];
  const angle = q * Math.pow(W.cfg.theta, 0);
  const c = Math.cos(angle), s = Math.sin(angle);
  const dq0 = p.d.q[0], dqh = p.d.q[dh / 2];
  const pre0 = dq0 * c + dqh * s, preh = -dq0 * s + dqh * c;
  const lookup = tape.dEmbedLookup.subarray(id * d, (id + 1) * d);
  const unemb = tape.dEmbedUnembed.subarray(id * d, (id + 1) * d);
  const total = tape.g.embed.subarray(id * d, (id + 1) * d);
  const k = argmaxAbs(total);
  const places = tok.ids.reduce((n, x) => n + (x === id ? 1 : 0), 0);
  const table = el('div', 'table');
  table.append(el('div', 'tr th', `<span class="lbl" style="flex:1">word</span><span class="lbl" style="flex:1">lookup</span><span class="lbl num">‖·‖</span><span class="lbl" style="flex:1">unembed</span><span class="lbl num">‖·‖</span>`));
  const scale = Math.max(...tok.vocab.map((_, v) => l2(tape.g.embed.subarray(v * d, (v + 1) * d)))) || 1;
  tok.vocab.forEach((w, v) => {
    const lu = l2(tape.dEmbedLookup.subarray(v * d, (v + 1) * d));
    const un = l2(tape.dEmbedUnembed.subarray(v * d, (v + 1) * d));
    table.append(el('div', 'tr' + (v === id ? ' on' : ''), `<span class="mono" style="flex:1">${esc(w)}</span><div class="bar"><i style="width:${lu / scale * 100}%"></i></div><span class="num">${fmt(lu)}</span><div class="bar"><i style="width:${un / scale * 100}%;background:${C.high}"></i></div><span class="num">${fmt(un)}</span>`));
  });
  return withPath([
    el('p', 'note', `Query and key gradients are shown after RoPE and before it. The inverse rotation uses angle ${fmt(angle)} on pair (0, ${dh / 2}) of head 0.`),
    note(`∂pre₀ = ${fmt(pre0)}, stored ${fmt(p.d.qPre[0])}. ∂pre₁ = ${fmt(preh)}, stored ${fmt(p.d.qPre[dh / 2])}.`),
    vec('∂ rotated query', 'Gradient in the space where the score was computed.', p.d.q),
    vec('∂ query before RoPE', 'What multiplies the query projection. Position 0 leaves this equal to the rotated gradient.', p.d.qPre),
    vec('∂ rotated key', 'Summed over every query that attended to this token.', p.d.k),
    vec('∂ value', 'Scaled by the attention weight of each query that read this token.', p.d.v),
    vec('∂ token vector', 'Skip gradient plus the gradient returned through the first RMSNorm.', p.d.x0),
    section('Embedding row', `“${word(q)}” occurs at ${places} position${places === 1 ? '' : 's'}. The lookup row sums the token-vector gradient at each of them. The unembed row sums every position that scored this word.`),
    note(`dim ${k}: lookup ${fmt(lookup[k])} + unembed ${fmt(unemb[k])} = total ${fmt(total[k])}. The three charts share a scale.`),
    addPicture(['lookup', 'unembed', 'total'], [lookup, unemb, total], k),
    table,
  ]);
}

function cellName(key, index) {
  const { rows, cols } = dims(W, key);
  const r = Math.floor(index / cols), c = index % cols;
  if (key === 'embed') return `${vocab[r]} · dim ${c}`;
  if (rows === 1) return `dim ${c}`;
  return `out ${r} · in ${c}`;
}

function biggestSplit() {
  let best = null;
  for (const key of PARAMS) {
    const n = tapes[0].g[key].length;
    for (let i = 0; i < n; i++) {
      let lo = Infinity, hi = -Infinity;
      for (const t of tapes) {
        const v = t.g[key][i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (!best || hi - lo > best.spread) best = { key, i, spread: hi - lo };
    }
  }
  return best;
}

function pullRow(label, value, max, on) {
  const t = max ? Math.min(1, Math.abs(value) / max) * 100 : 0;
  const neg = value < 0 ? t : 0, pos = value > 0 ? t : 0;
  return el('div', 'tr' + (on ? ' on' : ''), `<span class="mono clip">${esc(label)}</span><div class="signed"><div class="half neg"><i style="width:${neg}%"></i></div><div class="half pos"><i style="width:${pos}%"></i></div></div><span class="num">${fmt(value)}</span>`);
}

function stepPanel() {
  const spot = biggestSplit();
  const pulls = tapes.map((t) => t.g[spot.key][spot.i]);
  const mean = pulls.reduce((s, v) => s + v, 0) / pulls.length;
  const max = Math.max(...pulls.map(Math.abs), Math.abs(mean), 1e-9);
  const rate = lr();
  const w = W[spot.key][spot.i];
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl" style="flex:0 0 150px">sentence</span><span class="lbl" style="flex:1">gradient on ${esc(cellName(spot.key, spot.i))}</span><span class="lbl num">∂</span>`));
  sents.forEach((s, i) => t.append(pullRow(s.words.join(' '), pulls[i], max, i === si)));
  t.append(pullRow('mean', mean, max, false));
  const actions = el('div', 'row');
  const one = el('button', 'btn', 'Step this sentence');
  const all = el('button', 'btn pri', 'Step all');
  one.onclick = () => steps(1, false);
  all.onclick = () => steps(1, true);
  actions.append(one, all);
  return [
    el('p', 'note', `Orange is a negative gradient, so gradient descent increases the weight. Teal is a positive gradient, so the weight decreases. The update subtracts learning rate times the gradient. Train all uses the mean.`),
    t,
    note(`${esc(cellName(spot.key, spot.i))}: ${fmt(w)} − ${fmt(rate)} × ${fmt(mean)} = ${fmt(w - rate * mean)} if the step uses every sentence.`),
    actions,
  ];
}

function chart(rows) {
  const w = 960, h = 220, pad = 16;
  const c = el('canvas');
  c.width = w; c.height = h;
  c.style.width = '100%'; c.style.height = '180px';
  c.style.background = 'var(--sunk)';
  c.style.border = '1px solid var(--rule)';
  c.style.borderRadius = '3px';
  const g = c.getContext('2d');
  if (rows.length < 2) return c;
  let mx = 0.5;
  for (const row of rows) for (const v of row) if (Number.isFinite(v)) mx = Math.max(mx, v);
  rows[0].forEach((_, s) => {
    g.strokeStyle = COLS[s % COLS.length];
    g.lineWidth = 2;
    g.beginPath();
    rows.forEach((row, i) => {
      const x = pad + (w - 2 * pad) * i / (rows.length - 1);
      const y = h - pad - (h - 2 * pad) * (Math.min(row[s], mx) / mx);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  });
  return c;
}

function trainPanel() {
  const row = el('div', 'row');
  const one = el('button', 'btn', 'Train this sentence');
  const all = el('button', 'btn pri', 'Train all');
  one.onclick = () => steps(RUN, false);
  all.onclick = () => steps(RUN, true);
  row.append(one, all);
  const stepsDone = hist.length - 1;
  const legendRow = el('div', 'legend');
  sents.forEach((s, i) => {
    const sw = el('span');
    sw.innerHTML = `<i style="background:${COLS[i % COLS.length]}"></i>${esc(s.words.join(' '))}`;
    legendRow.append(sw);
  });
  const curve = hist.length < 2
    ? [el('p', 'note', 'A line per sentence appears after the first step. Each line is that sentence’s loss.')]
    : [chart(hist), legendRow];
  return [
    el('p', 'note', `${stepsDone} gradient step${stepsDone === 1 ? '' : 's'} so far. Mean loss ${fmt(meanLoss())}. Training this sentence fits “${esc(sents[si].words.join(' '))}”, and any next-word pair another sentence shares. Training all of them applies the mean gradient.`),
    row,
    ...curve,
  ];
}

const PANELS = [lossPanel, signalPanel, unembedPanel, residualPanel, mlpPanel, attentionPanel, projectPanel, stepPanel, trainPanel];

const meanLoss = () => tapes.reduce((s, t) => s + t.loss, 0) / tapes.length;

function focus() {
  tape = tapes[si];
  const s = sents[si];
  tok = { words: s.words, ids: s.ids, vocab };
}

function evalAll() {
  tapes = sents.map((s) => learn(W, s.ids));
  focus();
}

function corpus() {
  const box = el('div', 'sents');
  sents.forEach((s, i) => {
    const row = el('div', 'sent' + (i === si ? ' on' : ''));
    const lab = el('button', 'chip small' + (i === si ? ' on' : ''), String(i + 1));
    lab.onclick = () => { si = i; q = Math.min(q, s.words.length - 1); focus(); render(); };
    const toks = el('div', 'toks');
    s.words.forEach((w, j) => {
      const p = tapes[i].pos[j], last = p.target < 0;
      const hit = !last && argmax(p.prob) === p.target;
      const prob = last ? 0 : p.prob[p.target];
      const b = el('button', 'tok' + (i === si && j === q ? ' on' : ''), `<span>${esc(w)}</span>${last ? '' : `<span class="meter"><i style="width:${(prob * 100).toFixed(1)}%;background:${hit ? C.ok : C.high}"></i></span>`}`);
      b.onclick = () => { si = i; q = j; focus(); render(); };
      toks.append(b);
    });
    row.append(lab, toks);
    if (sents.length > 1) {
      const x = el('button', 'chip small', '×');
      x.title = 'Remove sentence';
      x.onclick = () => drop(i);
      row.append(x);
    }
    box.append(row);
  });
  $('corpus').replaceChildren(box);
}

function render() {
  $('lrText').textContent = lr().toFixed(2);
  [...$('tabs').children].forEach((b, i) => b.classList.toggle('on', i === stage));
  $('stageTitle').textContent = STAGES[stage][1];
  $('stageText').textContent = STAGES[stage][2];
  $('lossNow').textContent = fmt(meanLoss());
  $('spec').textContent = `1 layer · d=${CFG.d} · ${CFG.heads} heads · ${PARAMS.reduce((s, k) => s + W[k].length, 0)} parameters · ${sents.length} sentences`;
  corpus();
  $('out').replaceChildren(...PANELS[stage]());
  paint();
}

function addLine(text) {
  if (busy) return;
  const words = tokenize(text).words;
  if (words.length < 2) return;
  lines.push(words.join(' '));
  si = lines.length - 1;
  q = 0;
  reset();
  $('sentence').value = '';
}

function drop(i) {
  if (busy || lines.length < 2) return;
  lines.splice(i, 1);
  if (si > i) si--;
  else if (si >= lines.length) si = lines.length - 1;
  q = 0;
  reset();
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function vnorm(a) { return Math.sqrt(dot(a, a)); }
function packGrad(g, scale) {
  let n = 0;
  for (const k of PARAMS) n += g[k].length;
  const o = new Float64Array(n);
  let p = 0;
  for (const k of PARAMS) {
    const src = g[k];
    for (let i = 0; i < src.length; i++) o[p + i] = src[i] * scale;
    p += src.length;
  }
  return o;
}

function armWalk() {
  const z = meanLoss();
  walk = {
    prev: null, side: null, x: 0, y: 0,
    box: { x0: -2.6, x1: 2.6, y0: -1.4, y1: 3.2, z0: 0, z1: Math.max(4, z + 0.5) },
    pts: [{ x: 0, y: 0, z }],
    marker: { x: 0, y: 0, z },
  };
}

function growBox(p) {
  const b = walk.box, m = 0.45;
  if (p.x < b.x0 + 0.15) b.x0 = p.x - m;
  if (p.x > b.x1 - 0.15) b.x1 = p.x + m;
  if (p.y < b.y0 + 0.15) b.y0 = p.y - m;
  if (p.y > b.y1 - 0.15) b.y1 = p.y + m;
  if (p.z > b.z1 - 0.15) b.z1 = p.z + m;
}

function pushStep(g, rate) {
  const step = packGrad(g, -rate);
  const n = vnorm(step);
  if (n < 1e-12) return;
  if (!walk.prev) walk.x += n;
  else {
    const pn = vnorm(walk.prev);
    const along = dot(step, walk.prev) / pn;
    const lat = step.slice();
    for (let i = 0; i < lat.length; i++) lat[i] -= walk.prev[i] * (along / pn);
    const ln = vnorm(lat);
    if (!walk.side && ln > 1e-8) {
      walk.side = new Float64Array(lat.length);
      for (let i = 0; i < lat.length; i++) walk.side[i] = lat[i] / ln;
    }
    walk.x += along;
    if (walk.side) walk.y += dot(lat, walk.side);
  }
  walk.prev = step;
  const p = { x: walk.x, y: walk.y, z: meanLoss() };
  growBox(p);
  walk.pts.push(p);
}

function world(p) {
  const b = walk.box;
  const x = (p.x - (b.x0 + b.x1) / 2) / ((b.x1 - b.x0) / 2) * 1.2;
  const z = (p.y - (b.y0 + b.y1) / 2) / ((b.y1 - b.y0) / 2) * 1.2;
  const y = ((p.z - b.z0) / (b.z1 - b.z0) - 0.5) * 1.6;
  return [x, y, z];
}

function drawHill({ g, proj, line }) {
  if (!walk) return;
  g.font = '10px "IBM Plex Mono", monospace';
  g.fillStyle = '#66727c';
  [['gradient', [1.4, 0, 0]], ['loss', [0, 1.4, 0]], ['later steps', [0, 0, 1.4]]].forEach(([name, v]) => {
    line(v.map((x) => -x), v, '#232c33', 1);
    const p = proj(v);
    g.fillText(name, p.x + 4, p.y - 4);
  });
  const y0 = (0 - 0.5) * 1.6;
  for (let i = 0; i <= 4; i++) {
    const t = -1.2 + i * 0.6;
    line([t, y0, -1.2], [t, y0, 1.2], '#232c33', 1);
    line([-1.2, y0, t], [1.2, y0, t], '#232c33', 1);
  }
  const shown = walk.pts.slice(0, -1);
  shown.push(walk.marker);
  shown.forEach((p) => {
    const t = (p.z - walk.box.z0) / (walk.box.z1 - walk.box.z0);
    const [x, , z] = world(p);
    line([x, y0, z], world(p), `rgba(255,159,69,${0.25 + 0.55 * t})`, 1);
  });
  for (let i = 1; i < shown.length; i++) line(world(shown[i - 1]), world(shown[i]), '#3fc5bd', 2);
  const m = proj(world(walk.marker));
  g.beginPath();
  g.arc(m.x, m.y, 6 * m.k, 0, 7);
  g.fillStyle = '#e8edf1';
  g.fill();
  g.strokeStyle = '#3fc5bd';
  g.lineWidth = 2;
  g.stroke();
}

function reset() {
  if (busy) return;
  repack();
  W = init(vocab.length, SEED);
  evalAll();
  hist = [tapes.map((t) => t.loss)];
  armWalk();
  render();
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function steps(n, all) {
  if (busy) return;
  busy = true;
  const rate = lr();
  for (let i = 0; i < n; i++) {
    const g = all ? meanGrad(tapes) : tapes[si].g;
    const from = { ...walk.marker };
    sgd(W, g, rate);
    evalAll();
    hist.push(tapes.map((t) => t.loss));
    pushStep(g, rate);
    const to = walk.pts[walk.pts.length - 1];
    const frames = 6;
    for (let f = 1; f <= frames; f++) {
      const t = f / frames;
      walk.marker = { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t, z: from.z + (to.z - from.z) * t };
      paint();
      await wait(48);
    }
    render();
  }
  busy = false;
  render();
}

STAGES.forEach(([name], i) => {
  const b = el('button', 'tab', `<i>${i + 1}</i>${esc(name)}`);
  b.onclick = () => { stage = i; render(); };
  $('tabs').append(b);
});
$('reset').onclick = reset;
$('add').onsubmit = (e) => { e.preventDefault(); addLine($('sentence').value); };
$('trainOne').onclick = () => steps(RUN, false);
$('trainAll').onclick = () => steps(RUN, true);
$('lr').oninput = () => render();
const hill = createScene($('hill'), drawHill);
function paint() { hill.redraw(); }
reset();
