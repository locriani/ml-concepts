// Transformer explorer: every number shown is computed live from SmolLM2-135M's published weights.
import { isCached, loadWeights, tokenize, piece } from './models.js';
import { newState, step, trace, logitsOf, topK, sample } from './llama.js';
import { pca3, cosine } from './pca.js';
import { createScene } from './space.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tick = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); }); // yield to the UI; unlike setTimeout it is not throttled in background tabs
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const l2 = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0));
const unit = (a) => { const n = l2(a) || 1; return a.map((x) => x / n); };
const row = (a, i, d) => a.subarray(i * d, (i + 1) * d);
const vis = (p) => p.replace(/^ /, '␣').replace(/\n/g, '↵'); // make a leading space visible
const dotv = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

const BASE = [10, 13, 16], TEAL = [63, 197, 189], ORANGE = [255, 159, 69]; // rAIdteam accent and high
const C = { accent: '#3fc5bd', crit: '#ff6b63', high: '#ff9f45', med: '#e8c44a', low: '#7f8b95', ok: '#4fc98a', absent: '#b07a8a', fg: '#e8edf1', muted: '#98a4ae' };
const mixv = (to, t) => BASE.map((x, i) => x + (to[i] - x) * t);
const mix = (to, t) => `rgb(${mixv(to, t).map(Math.round)})`;

const WTE = 'model.embed_tokens.weight';
let W, tr, pieces = [];
let stage = 0, q = 1, layer = 12, head = 0, scaled = true, kj = null;
const key = () => (kj != null && kj <= q ? kj : Math.max(0, q - 1)); // the key token of the followed pair: any token up to the query; by default the one just before it

const STAGES = [
  ['Embed', 'Text to vectors', 'The text is split into byte-pair tokens (a leading space belongs to the token, shown as ␣). Each token id looks up a 576-number vector in the embedding table, and that vector enters layer 1 as is. There is no position vector here: this model marks position inside attention instead (see the next stage). One row per token, one column per dimension: teal is positive, orange is negative.'],
  ['Attention', 'All heads in a layer, then one head', 'Each layer runs 9 heads in parallel and each learns its own pattern. To save memory, each group of 3 heads shares one set of keys and values (heads 0 to 2, 3 to 5, 6 to 8). The cards are every head\'s token-by-token attention matrix (rows look at columns); the outlined row is the token you follow. A token may only look at itself and earlier tokens, so everything above the diagonal is blocked. Click a card to open that head below. A head gives every token a query and a key, both rotated by an angle that grows with the token\'s position, so a score depends on how far apart two tokens are: score = query · key / √d<sub>k</sub>, softmax turns scores into weights that sum to 1, and the token\'s new vector is the weighted mix of the earlier tokens\' values.'],
  ['Heads', 'What each head does', 'Measured on the sentence above: for every head in the model, how much attention goes to the previous token, the token itself and the first token, and how sharply it focuses. Pick a measure to colour the layer-by-head grid, then click a cell to inspect that head. These are properties of this sentence, not fixed roles: the same head can behave differently on other text.'],
  ['Layer', 'Inside one layer', 'One layer applied to the followed token. A normalisation (RMSNorm) comes first, attention output is added back to the token\'s vector (a residual), then RMSNorm again, a 1536-wide gated MLP (SwiGLU: one branch decides how much of the other passes), and a second residual. The residuals are why a token\'s vector changes gradually from layer to layer.'],
  ['Predict', 'Every token predicts the next one', 'The model\'s real job. At each position it scores all 49,152 tokens for what comes next, from only the tokens up to there. The first table compares those guesses with what you actually wrote. The second reads the followed token\'s guess off at every depth by applying the final normalisation and output matrix to every third layer\'s vector, so you can watch the prediction form.'],
  ['Space', 'Tokens in 3D', 'The same tokens at every depth of the model, projected to 3 dimensions. Each vector is scaled to length 1 and each depth is centred on its own average, so the picture shows how the tokens relate to each other at that depth, not how far the whole sentence has travelled.'],
  ['Words', 'Word map in 3D', 'Single words as the model first sees them: each word\'s row in the embedding table, centered on the vocabulary average, projected to 3 dimensions. Words that are several tokens long use the average of their tokens and carry an asterisk.'],
  ['Generate', 'One new token at a time', 'Writing is the Predict stage on repeat: score the next token, choose one, append it, run again with one more token. Nothing else is happening when it writes a paragraph. Temperature flattens or sharpens the scores before choosing; top K cuts everything but the K likely tokens, so a wild low-probability pick becomes impossible.'],
];
const GROUPS = [
  ['animals', C.high, 'cat dog kitten puppy horse tiger lion fish'],
  ['vehicles', C.med, 'car truck bus train bicycle airplane'],
  ['food', C.ok, 'apple banana orange pizza bread cheese'],
  ['places', C.absent, 'paris london tokyo france japan'],
  ['feelings', C.crit, 'happy sad angry joyful'],
  ['people', C.low, 'king queen man woman'],
];
const DEFAULT_WORDS = GROUPS.flatMap((g) => g[2].split(' '));
const hueOf = (w) => GROUPS.find((g) => g[2].split(' ').includes(w))?.[1] ?? C.fg;

// ---------- drawing helpers ----------
function stripCanvas(arr, n, d) {
  const rms = Math.sqrt(arr.reduce((s, x) => s + x * x, 0) / arr.length) || 1, S = 3 * rms;
  const c = el('canvas'); c.width = d; c.height = n;
  const g = c.getContext('2d'), im = g.createImageData(d, n);
  for (let k = 0; k < arr.length; k++) {
    const v = clamp(arr[k] / S, -1, 1), [r, gg, b] = mixv(v >= 0 ? TEAL : ORANGE, Math.abs(v));
    im.data.set([r, gg, b, 255], k * 4);
  }
  g.putImageData(im, 0, 0);
  return c;
}

// arr is [n, d] row-major; labels[i] = [main, small]
function strip(name, desc, arr, n, d, labels, on, rowH = 17, hue = C.accent) {
  const box = el('div', 'stack tight');
  box.append(el('div', 'controls', `<span class="step-name" style="color:${hue}">${name}</span><span class="muted small">${desc}</span>`));
  const s = el('div', 'strip'), l = el('div', 'strip-l');
  labels.forEach((t, i) => {
    const r = el('div', i === on ? 'on' : '', `<span>${esc(t[0])}</span><span class="faint">${esc(t[1] ?? '')}</span>`);
    r.style.height = rowH + 'px';
    l.append(r);
  });
  const c = stripCanvas(arr, n, d);
  c.style.height = n * rowH + 'px';
  s.append(l, c);
  box.append(s);
  return box;
}

const tokLabels = () => pieces.map((p, i) => [vis(p), tr.ids[i]]);

function heat(at, n, onRow, onCell) {
  const wrap = el('div', 'scroll'), g = el('div', 'heat');
  const hd = el('div', 'heat-r hrow');
  hd.append(el('div', 'rowbtn'));
  pieces.forEach((p) => hd.append(el('div', 'hd', esc(vis(p)))));
  g.append(hd);
  for (let i = 0; i < n; i++) {
    const r = el('div', 'heat-r'), b = el('button', 'rowbtn' + (i === q ? ' on' : ''), esc(vis(pieces[i])));
    b.onclick = () => onRow(i);
    r.append(b);
    for (let j = 0; j < n; j++) {
      const v = at[i * n + j];
      const c = el('div', 'hc', v >= 0.05 ? (v > 0.995 ? '1' : v.toFixed(2).slice(1)) : '');
      c.style.background = mix(TEAL, Math.sqrt(v));
      c.style.color = v > 0.45 ? '#0a0d10' : 'var(--fg)';
      if (j <= i && onCell) { c.style.cursor = 'pointer'; c.title = `${vis(pieces[i])} → ${vis(pieces[j])}`; c.onclick = () => onCell(i, j); }
      if (i === q && j === key()) c.style.outline = '2px solid var(--accent)', c.style.outlineOffset = '-2px';
      r.append(c);
    }
    g.append(r);
  }
  wrap.append(g);
  return wrap;
}

function miniHeat(at, n, qrow) {
  const k = 12, c = el('canvas');
  c.width = c.height = n * k;
  c.style.width = '100%';
  const g = c.getContext('2d');
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { g.fillStyle = mix(TEAL, Math.sqrt(at[i * n + j])); g.fillRect(j * k, i * k, k, k); }
  g.strokeStyle = '#3fc5bd'; g.lineWidth = 2;
  g.strokeRect(1, qrow * k + 1, n * k - 2, k - 2);
  return c;
}

const barRow = (label, val, text, on, w = 110, color = C.accent) =>
  el('div', 'tr' + (on ? ' on' : ''), `<span class="mono" style="flex:0 0 ${w}px">${esc(label)}</span><div class="bar"><i style="width:${clamp(val, 0, 1) * 100}%;background:${color}"></i></div><span class="num">${text}</span>`);

// ---------- stages ----------
const sqrtDk = () => Math.sqrt(tr.d / tr.heads);

function attnView(l, hd) {
  const L = tr.layers[l], n = tr.n;
  if (scaled) return { sc: L.scores[hd], at: L.attn[hd] };
  const sc = L.scores[hd].map((x) => x * sqrtDk()), at = new Float32Array(n * n); // masked scores stay -Infinity
  for (let i = 0; i < n; i++) {
    let m = -Infinity, s = 0;
    for (let j = 0; j <= i; j++) m = Math.max(m, sc[i * n + j]);
    for (let j = 0; j <= i; j++) { at[i * n + j] = Math.exp(sc[i * n + j] - m); s += at[i * n + j]; }
    for (let j = 0; j <= i; j++) at[i * n + j] /= s;
  }
  return { sc, at };
}

function s1() {
  const { n, d, embed } = tr, lab = tokLabels();
  $('s1out').replaceChildren(
    strip('word', 'row of the embedding table for this token id: the input to layer 1', embed.word, n, d, lab, q),
  );
}

function headCards() {
  const n = tr.n, grid = el('div');
  grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px';
  for (let hd = 0; hd < tr.heads; hd++) {
    const b = el('button', 'chip' + (hd === head ? ' on' : ''), `<span class="lbl">head ${hd}</span>`);
    b.style.cssText = 'width:100%;padding:8px;gap:6px';
    b.append(miniHeat(attnView(layer - 1, hd).at, n, q));
    b.onclick = () => { head = hd; render(); };
    grid.append(b);
  }
  return grid;
}

function s2() {
  const n = tr.n, { sc, at } = attnView(layer - 1, head), k = sqrtDk();
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl" style="flex:0 0 110px">key token</span><span class="lbl num">score</span><span class="lbl" style="flex:1">weight</span><span class="lbl num">weight</span>`));
  for (let j = 0; j < n; j++) {
    const open = j <= q, v = sc[q * n + j];
    t.append(el('div', 'tr' + (j === q ? ' on' : '') + (open ? '' : ' faint'), `<span class="mono" style="flex:0 0 110px">${esc(vis(pieces[j]))}</span><span class="num">${open ? v.toFixed(2) : 'blocked'}</span><div class="bar"><i style="width:${at[q * n + j] * 100}%"></i></div><span class="num">${at[q * n + j].toFixed(2)}</span>`));
  }
  $('s2out').replaceChildren(
    el('span', 'lbl', `Layer ${layer} · all ${tr.heads} heads`),
    headCards(),
    el('p', 'small', `Layer ${layer} · head ${head} · query <span class="step-name">${esc(vis(pieces[q]))}</span>: how much it reads from each token. Later tokens are blocked: they have not been written yet.`),
    t,
    el('p', 'note', scaled
      ? `Scores are divided by √d<sub>k</sub> = √${tr.d / tr.heads} = ${k} before the softmax. Untick the box to see why.`
      : `Without the scale, dot products over ${tr.d / tr.heads} dimensions are about ${k}× larger, the softmax saturates, and each token puts nearly all its weight on one other token.`),
    el('span', 'lbl', 'Every token as the query (click a row label to follow it, or a cell to follow that pair)'),
    heat(at, n, (i) => { q = i; render(); }, (i, j) => { q = i; kj = j; render(); }),
    pairPanel(),
  );
}

// one pair of tokens (query reads from key), followed through every layer and head
function pairPanel() {
  const n = tr.n, i = q, j = key(), box = el('div', 'col');
  const pick = el('div', 'chips');
  for (let k = 0; k <= i; k++) {
    const b = el('button', 'chip small' + (k === j ? ' on' : ''), esc(vis(pieces[k])));
    b.onclick = () => { kj = k; render(); };
    pick.append(b);
  }
  const g = el('div', 'heat'), hr = el('div', 'heat-r hrow');
  hr.append(el('div', 'rowbtn'));
  for (let h = 0; h < tr.heads; h++) hr.append(el('div', 'hh', 'h' + h));
  hr.append(el('div', 'hh', 'mean'));
  g.append(hr);
  for (let l = tr.layers.length - 1; l >= 0; l--) {
    const r = el('div', 'heat-r'), vals = tr.layers[l].attn.map((a) => a[i * n + j]), mean = vals.reduce((x, y) => x + y, 0) / vals.length;
    r.append(el('div', 'rowbtn' + (l + 1 === layer ? ' on' : ''), `layer ${l + 1}`));
    [...vals, mean].forEach((v, h) => {
      const c = el('button', 'hc' + (l + 1 === layer && h === head ? ' on' : ''), v >= 0.005 ? (v > 0.995 ? '1' : v.toFixed(2).replace(/^0/, '')) : '');
      c.style.background = mix(h === vals.length ? ORANGE : TEAL, Math.sqrt(clamp(v, 0, 1)));
      c.style.color = v > 0.3 ? '#0a0d10' : 'var(--fg)';
      c.title = `layer ${l + 1}, ${h === vals.length ? 'mean of heads' : 'head ' + h}: ${v.toFixed(3)}`;
      c.onclick = () => { layer = l + 1; if (h < vals.length) head = h; render(); };
      r.append(c);
    });
    g.append(r);
  }
  const wrap = el('div', 'scroll'); wrap.append(g);
  box.append(
    el('span', 'lbl', 'Follow one pair through the whole model'),
    el('p', 'small', `How much <span class="step-name">${esc(vis(pieces[i]))}</span> (the query, chosen above) reads from <span class="step-name">${esc(vis(pieces[j]))}</span> in every layer and head. Pick the other token of the pair:`),
    pick, wrap,
    el('p', 'note', 'The same pair is drawn in white in the 3D view, where every pair of tokens is a line and brighter means more attention at that depth.'),
  );
  return box;
}

// ---------- heads: what each head does, measured on this sentence ----------
const METRICS = [
  ['prev', 'previous token', 'mean weight each token puts on the token just before it'],
  ['self', 'itself', 'mean weight each token puts on itself'],
  ['first', 'first token', 'mean weight on the very first token, where heads often park attention when nothing else is relevant'],
  ['focus', 'focus', '1 means each token looks at one token, 0 means it spreads evenly over everything it can see'],
  ['reach', 'reach', 'mean distance to the tokens attended to, as a share of how far back each token could look'],
];
let metric = 'prev';

function headStats(l, hd) {
  const key = l * 100 + hd;
  tr.stats ??= {};
  if (tr.stats[key]) return tr.stats[key];
  const n = tr.n, at = tr.layers[l].attn[hd], m = { prev: 0, self: 0, first: 0, focus: 0, reach: 0 };
  for (let i = 1; i < n; i++) { // row 0 can only see itself, so it says nothing about the head
    let ent = 0, reach = 0;
    for (let j = 0; j <= i; j++) {
      const w = at[i * n + j];
      if (w > 0) ent -= w * Math.log(w);
      reach += w * (i - j) / i;
    }
    m.prev += at[i * n + i - 1]; m.self += at[i * n + i]; m.first += at[i * n];
    m.focus += 1 - ent / Math.log(i + 1); m.reach += reach;
  }
  for (const k of Object.keys(m)) m[k] /= Math.max(1, n - 1);
  const top = [['previous-token head', m.prev], ['self head', m.self]].sort((a, b) => b[1] - a[1])[0];
  m.tag = m.first >= 0.5 ? 'first-token sink' : top[1] >= 0.4 ? top[0] : m.focus >= 0.45 ? 'focused on specific tokens' : 'broad, mixed';
  return (tr.stats[key] = m);
}

function s3() {
  const [, label, hint] = METRICS.find((x) => x[0] === metric), out = $('s3out'), n = tr.n;
  const mk = el('div', 'chips');
  METRICS.forEach(([k, name]) => {
    const b = el('button', 'chip small' + (k === metric ? ' on' : ''), esc(name));
    b.onclick = () => { metric = k; render(); };
    mk.append(b);
  });
  const g = el('div', 'heat'), hr = el('div', 'heat-r hrow');
  hr.append(el('div', 'rowbtn'));
  for (let h = 0; h < tr.heads; h++) hr.append(el('div', 'hh', 'h' + h));
  g.append(hr);
  for (let l = tr.layers.length - 1; l >= 0; l--) {
    const r = el('div', 'heat-r');
    r.append(el('div', 'rowbtn' + (l + 1 === layer ? ' on' : ''), `layer ${l + 1}`));
    for (let h = 0; h < tr.heads; h++) {
      const v = headStats(l, h)[metric], c = el('button', 'hc' + (l + 1 === layer && h === head ? ' on' : ''), v.toFixed(2).replace(/^0/, ''));
      c.style.background = mix(TEAL, Math.sqrt(clamp(v, 0, 1)));
      c.style.color = v > 0.3 ? '#0a0d10' : 'var(--fg)';
      c.title = `layer ${l + 1}, head ${h}: ${v.toFixed(3)}`;
      c.onclick = () => { layer = l + 1; head = h; render(); };
      r.append(c);
    }
    g.append(r);
  }
  const wrap = el('div', 'scroll'); wrap.append(g);

  // detail for the selected head
  const st = headStats(layer - 1, head), at = tr.layers[layer - 1].attn[head];
  const prof = el('div', 'table');
  prof.append(el('div', 'tr th', `<span class="lbl">Layer ${layer} · head ${head}: ${esc(st.tag)}</span>`));
  METRICS.forEach(([k, name]) => prof.append(barRow(name, st[k], st[k].toFixed(2), k === metric, 130)));
  const pairs = [];
  for (let i = 1; i < n; i++) for (let j = 0; j <= i; j++) pairs.push([i, j, at[i * n + j]]);
  pairs.sort((a, b) => b[2] - a[2]);
  const links = el('div', 'table');
  links.append(el('div', 'tr th', `<span class="lbl">Strongest links in this head · token → earlier token it reads from</span>`));
  pairs.slice(0, 10).forEach(([i, j, w]) => links.append(barRow(`${vis(pieces[i])} → ${vis(pieces[j])}`, w, w.toFixed(2), false, 210)));
  const open = el('button', 'btn', 'Open in Attention');
  open.onclick = () => { stage = 1; render(); };
  out.replaceChildren(
    el('div', 'stack tight', `<span class="lbl">Colour by</span>`),
    mk,
    el('p', 'note', `${esc(label)}: ${esc(hint)}. Even attention over everything visible would give a token only 1 / (its position + 1), so anything near or above 0.4 is a clear habit.`),
    wrap,
    prof,
    links,
    el('div', 'row').appendChild(open).parentElement,
  );
}

function s4() {
  const L = tr.layers[layer - 1], { d } = tr, ffD = L.ffHidden.length / tr.n;
  const r = (a, w) => row(a, q, w), lab = [[vis(pieces[q])]];
  const stp = (name, desc, v, hue) => {
    const box = strip(name, desc, v, 1, v.length, lab, -1, 22, hue);
    box.append(el('span', 'note', `length ${l2(v).toFixed(2)}${v.length === d ? ` · cosine to layer input ${cosine(v, r(L.input, d)).toFixed(3)}` : ''}`));
    return box;
  };
  $('s4out').replaceChildren(
    el('p', 'small', `Layer ${layer}, following <span class="step-name">${esc(vis(pieces[q]))}</span>. Cosine to the layer input shows how far each step has moved the token.`),
    el('div', 'legend', `<span><i style="background:${C.fg}"></i>token vector</span><span><i style="background:${C.accent}"></i>attention</span><span><i style="background:${C.ok}"></i>residual add</span><span><i style="background:${C.high}"></i>MLP</span><span><i style="background:${C.muted}"></i>RMSNorm</span>`),
    stp('input', 'the token\'s vector so far (the residual stream)', r(L.input, d), C.fg),
    stp('RMSNorm 1', 'normalised before attention reads it', r(L.ln1, d), C.muted),
    stp('attention context', `all ${tr.heads} heads\' weighted value mixes, side by side`, r(L.ctx, d), C.accent),
    stp('attention output', 'context projected back by the output matrix', r(L.attnOut, d), C.accent),
    stp('+ residual', 'input + attention output', r(L.resAttn, d), C.ok),
    stp('RMSNorm 2', 'normalised before the MLP reads it', r(L.ln2, d), C.muted),
    stp('MLP hidden', `expanded to ${ffD} numbers; gate × up, with SiLU on the gate`, r(L.ffHidden, ffD), C.high),
    stp('MLP output', `projected back to ${d}`, r(L.ffOut, d), C.high),
    stp('+ residual', 'previous residual + MLP output: this is the layer output', r(L.out, d), C.ok),
  );
}

// ---------- predict: what the model expects next, and how that forms across depth ----------
const states = () => [tr.embed.word, ...tr.layers.map((l) => l.out)];

async function s5() {
  const out = $('s5out'), myQ = q, n = tr.n, d = tr.d;
  out.replaceChildren(el('p', 'muted small', 'Scoring every position…'));
  await tick();
  if (!tr.pred) {
    tr.pred = [];
    for (let i = 0; i < n; i++) {
      tr.pred[i] = topK(logitsOf(W, row(tr.layers.at(-1).out, i, d)), 3);
      if (stage === 4) await tick();
    }
  }
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl" style="flex:0 0 100px">token</span><span class="lbl" style="flex:0 0 100px">you wrote next</span><span class="lbl" style="flex:1">probability the model gave it</span><span class="lbl" style="flex:0 0 190px">its top guess</span>`));
  for (let i = 0; i < n; i++) {
    const pr = tr.pred[i], g1 = pr.top[0], nxt = tr.ids[i + 1], pa = nxt == null ? null : pr.probs[nxt] / pr.sum;
    const b = el('button', 'tr' + (i === q ? ' on' : ''), `<span class="mono" style="flex:0 0 100px">${esc(vis(pieces[i]))}</span><span class="mono" style="flex:0 0 100px">${nxt == null ? '–' : esc(vis(pieces[i + 1]))}</span><div class="bar"><i style="width:${(pa ?? 0) * 100}%;background:${nxt != null && nxt === g1.id ? C.ok : C.accent}"></i></div><span class="num">${pa == null ? '' : (pa * 100).toFixed(1) + '%'}</span><span class="mono" style="flex:0 0 190px">${esc(vis(piece(g1.id)))} ${(g1.p * 100).toFixed(0)}%</span>`);
    b.onclick = () => { q = i; render(); };
    t.append(b);
  }
  out.replaceChildren(
    el('p', 'small', 'Each row sees only its own token and the ones before it. Green means the model\'s top guess is exactly what you wrote.'),
    t,
    el('p', 'muted small', `Reading the guess for <span class="step-name">${esc(vis(pieces[q]))}</span> at every depth…`),
  );

  // logit lens for the followed token
  tr.lens ??= {};
  if (!tr.lens[q]) {
    const got = []; // cached only once complete, so an interrupted render never leaves a partial table
    for (const S of states().filter((_, i) => i % 3 === 0)) { got.push(topK(logitsOf(W, row(S, q, d)), 3)); await tick(); if (stage !== 4 || q !== myQ) return; }
    tr.lens[q] = got;
  }
  if (stage !== 4 || q !== myQ) return;
  const nxt = tr.ids[q + 1], lt = el('div', 'table');
  lt.append(el('div', 'tr th', `<span class="lbl">After <b>${esc(vis(pieces[q]))}</b> · top 3 guesses read off at each depth${nxt == null ? '' : ` · you wrote “${esc(vis(pieces[q + 1]))}”`}</span>`));
  tr.lens[q].forEach((tk, s) => {
    const cells = tk.top.map((x, k) => `<span class="mono" style="flex:1;${k === 0 ? `color:${x.id === nxt ? C.ok : C.accent}` : ''}">${esc(vis(piece(x.id)))} <span class="faint">${(x.p * 100).toFixed(0)}%</span></span>`).join('');
    lt.append(el('div', 'tr', `<span class="mono" style="flex:0 0 110px">${s === 0 ? 'embeddings' : 'layer ' + s * 3}</span>${cells}`));
  });
  out.lastChild.replaceWith(lt, el('p', 'note', 'Only the last row is what the model actually outputs. Earlier rows apply the final LayerNorm and output matrix to a half-finished vector, which was never trained for that, so they are rough but they show where the answer comes from.'));
}

// ---------- 3D views ----------
const AXES = [['PC1', [1.5, 0, 0]], ['PC2', [0, 1.5, 0]], ['PC3', [0, 0, 1.5]]];
const normalize = (c) => { const mx = Math.max(...c.flat().map(Math.abs)) || 1; return c.map((p) => p.map((x) => (x * 1.3) / mx)); };
function axes({ g, proj, line }) {
  g.font = '10px "IBM Plex Mono", monospace'; g.fillStyle = '#66727c';
  AXES.forEach(([name, v]) => {
    line(v.map((x) => -x), v, '#232c33', 1);
    const p = proj(v); g.fillText(name, p.x + 4, p.y - 4);
  });
}

let P6 = null;
function build6() {
  const S = states().filter((_, i) => i % 3 === 0), n = tr.n, vecs = []; // every third depth keeps the projection quick
  S.forEach((m) => { // directions only, centred on this depth's own average so the picture shows how tokens differ from each other
    const u = Array.from({ length: n }, (_, i) => unit(row(m, i, tr.d))), mu = Array.from(u[0], (_, k) => u.reduce((t, v) => t + v[k], 0) / n);
    u.forEach((v) => vecs.push(v.map((x, k) => x - mu[k])));
  });
  const c = pca3(vecs);
  P6 = S.map((_, s) => { // each depth is scaled so its typical token sits the same distance out; a far outlier (usually the first token) is held inside the box
    const pts = c.slice(s * n, (s + 1) * n), r = pts.map((p) => Math.hypot(...p)), k = 0.6 / ([...r].sort((x, y) => x - y)[n >> 1] || 1);
    return pts.map((p, i) => p.map((x) => x * k * Math.min(1, 1.6 / (r[i] * k))));
  });
}

const scene6 = createScene($('space3d'), (f) => {
  if (!P6) return;
  const { g, proj, line } = f, depth = +$('prog').value, t = depth / 3, L = P6.length - 1, s0 = Math.floor(t), s1_ = Math.min(s0 + 1, L), k = t - s0, n = tr.n;
  axes(f);
  const pos = (i) => P6[s0][i].map((x, a) => x + (P6[s1_][i][a] - x) * k);
  for (let s = 0; s < L; s++) line(P6[s][q], P6[s + 1][q], 'rgba(79,201,138,.7)', 1.5);
  const A = tr.layers[clamp(Math.ceil(depth), 1, tr.layers.length) - 1].attn, mean = (i, j) => A.reduce((t, a) => t + a[i * n + j], 0) / tr.heads, kk = key();
  for (let i = 1; i < n; i++) for (let j = 0; j < i; j++) { // every pair, always: weight is this depth's attention averaged over heads
    if (i === q && j === kk) continue;
    const w = mean(i, j);
    line(pos(i), pos(j), `rgba(63,197,189,${0.04 + 0.9 * Math.sqrt(w)})`, 0.4 + 7 * w);
  }
  line(pos(q), pos(kk), '#e8edf1', 1.5 + 8 * mean(q, kk)); // the followed pair
  g.font = '11px "IBM Plex Mono", monospace';
  const pts = Array.from({ length: n }, (_, i) => ({ i, p: proj(pos(i)) })).sort((a, b) => a.p.z - b.p.z);
  pts.forEach(({ i, p }) => {
    g.beginPath(); g.arc(p.x, p.y, (i === q ? 6 : 4) * p.k, 0, 7);
    g.fillStyle = i === q ? C.accent : i === 0 ? C.med : C.muted; g.fill();
    if (i === kk) { g.strokeStyle = C.fg; g.lineWidth = 2; g.stroke(); }
    g.fillStyle = i === q ? C.accent : C.fg;
    g.fillText(vis(pieces[i]), p.x + 8, p.y - 6);
  });
});

function progText() {
  const t = +$('prog').value;
  $('progText').textContent = t === 0 ? 'embeddings' : Number.isInteger(t) ? `after layer ${t}` : `between ${t < 1 ? 'embeddings' : 'layer ' + Math.floor(t)} and layer ${Math.ceil(t)}`;
}

function s6() {
  if (!P6) build6();
  progText();
  scene6.redraw();
}

// ---------- words: the embedding table itself ----------
let words = [], P7 = null, sel = 0, hits = [], probe = null;
let E7 = null; // { mu, cn, cand }: vocabulary mean, centered row lengths, ids of plain whole-word tokens
const neighbors = () => words.map((w, i) => [i, dotv(words[sel].v, w.v)]).filter(([i]) => i !== sel).sort((a, b) => b[1] - a[1]).slice(0, 6);

async function prepWords(onProgress) {
  const E = W[WTE], d = tr.d, V = E.length / d, mu = new Float32Array(d), cn = new Float32Array(V), cand = [];
  const slice = 4000;
  for (let i = 0; i < V; i++) { const o = i * d; for (let k = 0; k < d; k++) mu[k] += E[o + k] / V; if (i % slice === 0) { onProgress(0.3 * i / V, 'Averaging the embedding table…'); await tick(); } }
  for (let i = 0; i < V; i++) {
    const o = i * d; let s = 0; for (let k = 0; k < d; k++) s += (E[o + k] - mu[k]) ** 2; cn[i] = Math.sqrt(s);
    if (i % slice === 0) { onProgress(0.3 + 0.3 * i / V, 'Measuring every token vector…'); await tick(); }
  }
  for (let i = 0; i < V; i++) {
    if (/^ [A-Za-z]{2,}$/.test(piece(i))) cand.push(i);
    if (i % slice === 0) { onProgress(0.6 + 0.4 * i / V, 'Decoding the vocabulary…'); await tick(); }
  }
  E7 = { mu, cn, cand };
}

// a word (or phrase) -> { v: unit centered vector, ids }. The leading space matters: " cat" is its own token.
async function wordVec(w) {
  let { ids } = await tokenize(' ' + w);
  if (ids.length > 1) { const cap = (await tokenize(' ' + w[0].toUpperCase() + w.slice(1))).ids; if (cap.length === 1) ids = cap; } // names are usually one token when capitalised
  const E = W[WTE], d = tr.d, v = new Float32Array(d);
  ids.forEach((id) => { for (let k = 0; k < d; k++) v[k] += (E[id * d + k] - E7.mu[k]) / ids.length; });
  return { v: unit(v), ids };
}

const scene7 = createScene($('wordSpace'), (f) => {
  if (!P7) return;
  const { g, proj, line } = f;
  axes(f);
  const nb = new Set(neighbors().map(([i]) => i));
  nb.forEach((i) => line(P7[sel], P7[i], '#33404a', 1));
  g.font = '11px "IBM Plex Mono", monospace';
  const pts = P7.slice(0, words.length).map((c, i) => ({ i, p: proj(c) })).sort((a, b) => a.p.z - b.p.z);
  hits = pts.map(({ i, p }) => ({ i, x: p.x, y: p.y }));
  pts.forEach(({ i, p }) => {
    g.beginPath(); g.arc(p.x, p.y, (i === sel ? 6 : 4) * p.k, 0, 7);
    g.fillStyle = hueOf(words[i].w); g.fill();
    if (i === sel || nb.has(i)) { g.strokeStyle = i === sel ? C.accent : C.fg; g.lineWidth = 2; g.stroke(); }
    g.fillStyle = i === sel ? C.accent : nb.has(i) ? C.fg : C.muted;
    g.fillText(words[i].w + (words[i].multi ? '*' : ''), p.x + 8, p.y - 6);
  });
  if (probe) {
    const p = proj(P7[words.length]);
    g.fillStyle = C.fg; g.strokeStyle = C.accent; g.lineWidth = 2;
    g.fillRect(p.x - 5, p.y - 5, 10, 10); g.strokeRect(p.x - 5, p.y - 5, 10, 10);
    g.fillStyle = C.accent; g.fillText('= ' + probe.label, p.x + 10, p.y - 8);
  }
});

async function addWord(w) {
  if (words.some((x) => x.w === w)) { sel = words.findIndex((x) => x.w === w); return; }
  const { v, ids } = await wordVec(w);
  words.push({ w, v, multi: ids.length > 1 });
  sel = words.length - 1;
}

function list7() {
  P7 = normalize(pca3([...words.map((x) => x.v), ...(probe ? [probe.v] : [])]));
  const chips = el('div', 'chips');
  words.forEach((x, i) => {
    const b = el('button', 'chip small' + (i === sel ? ' on' : ''), esc(x.w + (x.multi ? '*' : '')));
    b.style.borderLeft = `3px solid ${hueOf(x.w)}`;
    b.onclick = () => { sel = i; list7(); };
    chips.append(b);
  });
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl">Nearest to “${esc(words[sel].w)}” by cosine</span>`));
  neighbors().forEach(([i, s]) => t.append(barRow(words[i].w, s, s.toFixed(2), false, 110, hueOf(words[i].w))));
  const legend = el('div', 'legend', GROUPS.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('') + `<span><i style="background:${C.fg}"></i>yours</span><span><i style="background:transparent;border:2px solid ${C.accent}"></i>selected</span><span><i style="background:transparent;border:2px solid ${C.fg}"></i>nearest 6</span>`);
  $('s7out').replaceChildren(legend, chips, t);
  scene7.redraw();
}

let ready7 = false, loading7 = false;
async function load7() {
  const set = (f, t) => { $('idxFill').style.width = f * 100 + '%'; $('idxText').textContent = t; };
  await prepWords(set);
  for (let k = 0; k < DEFAULT_WORDS.length; k++) await addWord(DEFAULT_WORDS[k]);
  sel = words.findIndex((x) => x.w === 'cat');
  ready7 = true;
  $('s7gate').hidden = true;
  $('s7body').hidden = false;
  if (stage === 6) list7();
}

async function s7() {
  if (ready7) return list7();
  if (loading7) return;
  loading7 = true;
  await load7();
}

// vector arithmetic: sum signed word vectors, then search EVERY token in the vocabulary
async function calc() {
  const parts = $('expr').value.toLowerCase().split(/([+-])/).map((s) => s.trim()).filter(Boolean);
  const terms = []; let sign = 1;
  for (const p of parts) { if (p === '+') sign = 1; else if (p === '-') sign = -1; else { terms.push([sign, p]); sign = 1; } }
  if (!terms.length) return;
  const v = new Float32Array(tr.d), skip = new Set();
  for (const [s, t] of terms) {
    const { v: tv, ids } = await wordVec(t);
    tv.forEach((x, k) => { v[k] += s * x; });
    ids.forEach((id) => skip.add(id));
  }
  probe = { v: unit(v), label: terms.map(([s, t], i) => (i ? (s > 0 ? ' + ' : ' - ') : s < 0 ? '-' : '') + t).join(''), terms };
  list7();
  showCalc(skip, terms);
}

function showCalc(skip, terms) {
  const E = W[WTE], d = tr.d, { mu, cn, cand } = E7, muv = dotv(mu, probe.v), words_ = new Set(terms.map((x) => x[1]));
  const near = (list) => list.sort((a, b) => b[1] - a[1]).slice(0, 8);
  const mapT = el('div', 'table'), vocT = el('div', 'table');
  mapT.append(el('div', 'tr th', `<span class="lbl">Closest map words to ${esc(probe.label)}</span>`));
  near(words.filter((x) => !words_.has(x.w)).map((x) => [x.w, dotv(probe.v, x.v)])).forEach(([w, s]) => mapT.append(barRow(w, s, s.toFixed(2), false, 110, hueOf(w))));
  vocT.append(el('div', 'tr th', `<span class="lbl">Closest words in the model's whole vocabulary (${cand.length.toLocaleString()} whole-word tokens searched)</span>`));
  const all = [];
  for (const id of cand) {
    if (skip.has(id)) continue;
    let s = 0; const o = id * d;
    for (let k = 0; k < d; k++) s += E[o + k] * probe.v[k];
    all.push([id, (s - muv) / cn[id]]); // cosine with the centered row, without building the centered matrix
  }
  near(all).forEach(([id, s]) => vocT.append(barRow(piece(id).trim(), s, s.toFixed(2), false, 110, C.accent)));
  $('s7calc').replaceChildren(
    el('p', 'small', `<span class="step-name">${esc(probe.label)}</span> as a vector (white square on the map). Each word is its row in the embedding table, centered on the vocabulary average. The signed vectors are summed, scaled to length 1, and compared by cosine with every candidate. Embedding tables were not trained for analogies, so expect related words more than exact answers.`),
    mapT, vocT);
}

// ---------- generate: one token at a time ----------
const g = { st: null, ids: [], pieces: [], nPrompt: 0, logits: null, attn: null, l: 15, h: 0, cands: null };

const gK = () => +$('gk').value;

function gRender() {
  const T = +$('gtemp').value, K = gK(), tk = topK(g.logits, Math.max(10, K), T), n = g.ids.length;
  const kept = tk.top.slice(0, K), keptP = kept.reduce((s, x) => s + x.p, 0);
  $('gtempText').textContent = T.toFixed(1);
  $('gkText').textContent = K;
  $('gtext').replaceChildren(...g.pieces.map((p, i) => el('span', 'chip small' + (i >= g.nPrompt ? ' gen' : '') + (i === n - 1 ? ' on' : ''), esc(vis(p)))));
  const t = el('div', 'table');
  t.append(el('div', 'tr th', `<span class="lbl">Next token · the ${K} kept tokens hold ${(keptP * 100).toFixed(1)}% of the probability and share all of it · bars: model's probability, right: chance of being picked · click one to choose it</span>`));
  g.cands = { top: kept }; // greedy takes the first kept token
  tk.top.slice(0, 10).forEach((x, rank) => {
    const cut = rank >= K, r = el('button', 'tr' + (cut ? ' faint' : ''), `<span class="mono" style="flex:0 0 110px">${esc(vis(piece(x.id)))}</span><div class="bar"><i style="width:${x.p / tk.top[0].p * 100}%;${cut ? `background:${C.low}` : ''}"></i></div><span class="num">${cut ? 'cut' : (x.p / keptP * 100).toFixed(1) + '%'}</span>`);
    r.onclick = () => gAdvance(x.id);
    t.append(r);
  });
  $('gout').replaceChildren(el('p', 'small', `<span class="mono">${esc(g.pieces.join(''))}</span><span class="step-name">▍</span>`), t);
  const w = g.attn[g.l][g.h], a = el('div', 'table');
  a.append(el('div', 'tr th', `<span class="lbl">Layer ${g.l + 1} · head ${g.h} · where “${esc(vis(g.pieces[n - 1]))}” looks (it can only see itself and earlier tokens)</span>`));
  g.pieces.forEach((p, j) => a.append(barRow(vis(p), w[j], w[j].toFixed(2), j === n - 1)));
  $('gattn').replaceChildren(a);
  const num = (count, cur, set, from) => Array.from({ length: count }, (_, k) => {
    const b = el('button', 'chip small' + (k === cur ? ' on' : ''), String(k + from));
    b.onclick = () => { set(k); gRender(); };
    return b;
  });
  $('glayers').replaceChildren(...num(g.attn.length, g.l, (v) => { g.l = v; }, 1));
  $('gheads').replaceChildren(...num(g.attn[0].length, g.h, (v) => { g.h = v; }, 0));
}

const gBusy = async (fn) => {
  const bs = ['gstart', 'gsample', 'ggreedy'].map($);
  bs.forEach((b) => { b.disabled = true; });
  await tick();
  try { await fn(); } finally { bs.forEach((b) => { b.disabled = false; }); }
};

const gStart = () => gBusy(async () => {
  let t = await tokenize($('gprompt').value);
  if (!t.ids.length) t = { ids: [0], pieces: ['<|endoftext|>'] }; // empty prompt: start-of-text token
  const ids = t.ids.slice(-40); // recent context only keeps a click quick
  g.st = newState(W);
  g.ids = ids; g.pieces = t.pieces.slice(-40); g.nPrompt = ids.length;
  ({ logits: g.logits, attn: g.attn } = step(W, g.st, ids));
  gRender();
});

const gAdvance = (id) => gBusy(async () => {
  g.pieces.push(piece(id));
  g.ids.push(id);
  ({ logits: g.logits, attn: g.attn } = step(W, g.st, [id])); // KV cache: only the new token is computed
  gRender();
});

let gInit = false;
function s8() {
  if (gInit) return;
  gInit = true;
  gStart();
}

// ---------- shell ----------
function chips() {
  $('chips').replaceChildren(...pieces.map((p, i) => {
    const b = el('button', 'chip' + (i === q ? ' on' : ''), `<span>${esc(vis(p))}</span><span class="lbl">${tr.ids[i]}</span>`);
    b.onclick = () => { q = i; render(); };
    return b;
  }));
  const num = (count, cur, set, from = 0) => Array.from({ length: count }, (_, k) => {
    const b = el('button', 'chip small' + (k + from === cur ? ' on' : ''), String(k + from));
    b.onclick = () => { set(k + from); render(); };
    return b;
  });
  $('layers').replaceChildren(...num(tr.layers.length, layer, (v) => (layer = v), 1));
}

function render() {
  if (!tr) return;
  STAGES.forEach((_, i) => { $('s' + (i + 1)).hidden = i !== stage; });
  [...$('tabs').children].forEach((b, i) => b.classList.toggle('on', i === stage));
  $('stageTitle').textContent = STAGES[stage][1];
  $('stageText').innerHTML = STAGES[stage][2] + (stage === 5 ? ` The real vectors have ${tr.d} numbers; PCA picks the 3 directions along which all tokens at all depths differ most, so the axes have no names. Every teal line is attention from a token to an earlier one it can see: brighter and thicker where the ${tr.heads} heads attend more at this depth. The white line is the pair chosen on the Attention page. Slide from the embeddings to layer ${tr.layers.length} and watch the tokens regroup as they take in context (every third layer is computed, in between is interpolated). The first token (yellow) is special: heads park attention on it. Drag to rotate, scroll or pinch to zoom, shift-drag (or right-drag) to pan, Reset view to start over.` : stage === 6 ? ' Distances are only roughly faithful, since 576 dimensions are squashed to 3. Click a point or a word to list its nearest neighbours. Drag to rotate, scroll or pinch to zoom, shift-drag (or right-drag) to pan, Reset view to start over.' : '');
  $('ctx').hidden = stage === 7;
  $('layers').parentElement.hidden = ![1, 2, 3].includes(stage);
  chips();
  [s1, s2, s3, s4, s5, s6, s7, s8][stage]();
}

async function run() {
  const text = $('text').value.trim();
  if (!text) return;
  $('run').disabled = true;
  await tick();
  try {
    const t = await tokenize(text), prev = pieces[q];
    const ids = t.ids.slice(0, 26); // keep matrices readable
    pieces = t.pieces.slice(0, 26);
    tr = trace(W, ids);
    q = pieces.indexOf(prev) > 0 ? pieces.indexOf(prev) : Math.max(1, pieces.indexOf(' it'));
    P6 = null;
    render();
  } finally { $('run').disabled = false; }
}

async function boot() {
  const bar = $('gateBar'), fill = $('gateFill'), txt = $('gateText');
  $('loadModel').disabled = true;
  bar.hidden = false;
  txt.textContent = 'Loading weights…';
  try {
    W = await loadWeights((p) => { fill.style.width = p * 100 + '%'; txt.textContent = `Downloading weights… ${Math.round(p * 100)}% of 269 MB`; });
  } catch (e) {
    txt.textContent = `Could not load the model: ${e.message}`;
    $('loadModel').disabled = false;
    return;
  }
  $('gate').hidden = true;
  $('app').hidden = false;
  await run();
}

STAGES.forEach(([name], i) => {
  const b = el('button', 'tab', `<i>${i + 1}</i>${name}`);
  b.onclick = () => { stage = i; render(); };
  $('tabs').append(b);
});
$('loadModel').onclick = boot;
$('run').onclick = run;
$('text').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
$('scale').onchange = (e) => { scaled = e.target.checked; render(); };
$('prog').oninput = () => { progText(); scene6.redraw(); };
$('reset6').onclick = () => scene6.reset();
$('reset7').onclick = () => scene7.reset();
$('gstart').onclick = gStart;
$('gprompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') gStart(); });
$('gsample').onclick = () => gAdvance(sample(topK(g.logits, gK(), +$('gtemp').value)));
$('ggreedy').onclick = () => gAdvance(g.cands.top[0].id);
$('gtemp').oninput = () => gRender();
$('gk').oninput = () => gRender();
$('calc').onclick = calc;
$('expr').addEventListener('keydown', (e) => { if (e.key === 'Enter') calc(); });
['king - man + woman', 'paris - france + japan', 'puppy - dog + cat', 'happy - sad'].forEach((x) => {
  const b = el('button', 'chip small', esc(x));
  b.onclick = () => { $('expr').value = x; calc(); };
  $('examples').append(b);
});
const addIn = async () => {
  const w = $('wordIn').value.trim().toLowerCase();
  if (!w || !ready7) return;
  $('wordIn').value = '';
  await addWord(w);
  list7();
};
$('addWord').onclick = addIn;
$('wordIn').addEventListener('keydown', (e) => { if (e.key === 'Enter') addIn(); });

// click-to-select a word, ignoring the end of a drag
let down = [0, 0];
$('wordSpace').addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
$('wordSpace').addEventListener('click', (e) => {
  if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = e.currentTarget.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const best = hits.map((p) => ({ ...p, d: Math.hypot(p.x - x, p.y - y) })).sort((a, b) => a.d - b.d)[0];
  if (best && best.d < 18) { sel = best.i; list7(); }
});

isCached().then((hit) => { if (hit) boot(); });
