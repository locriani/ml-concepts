// Finite-difference check of the hand-written backward pass, plus a descent check on one sentence.
import assert from 'node:assert';
import { tokenize, sentences, init, learn, loss, sgd, meanGrad, PARAMS, SEED, LINES } from './src/tiny.js';

const text = 'The cat sat on the mat because it was tired.';
const { ids } = tokenize(text);
const near = (a, b, tol, msg) => assert(Math.abs(a - b) < tol, `${msg}: ${a} vs ${b}`);

const W0 = init(tokenize(text).vocab.length, SEED);
const tape = learn(W0, ids);
const { d } = W0.cfg;

assert(Number.isFinite(tape.loss) && tape.loss > 0, 'finite loss');
assert.equal(tape.pos.at(-1).target, -1);
tape.pos.at(-1).d.logits.forEach((x) => assert.equal(x, 0));
tape.pos.at(-1).d.x1.forEach((x) => assert.equal(x, 0));

for (const p of tape.pos) {
  for (let k = 0; k < d; k++) {
    near(p.d.skipMlp[k] + p.d.branchMlp[k], p.d.res1[k], 1e-9, 'mlp residual split');
    near(p.d.skipAttn[k] + p.d.branchAttn[k], p.d.x0[k], 1e-9, 'attn residual split');
  }
  if (p.target >= 0) {
    let s = 0;
    for (const g of p.d.logits) s += g;
    near(s, 0, 1e-9, 'logit grad sums to 0');
  }
}
for (let i = 0; i < tape.g.embed.length; i++) near(tape.g.embed[i], tape.dEmbedLookup[i] + tape.dEmbedUnembed[i], 1e-9, 'tied embed');

const before = tape.loss;
sgd(W0, tape.g, 1e-3);
assert(loss(W0, ids) < before, 'a short step lowers the loss');

const W = init(tokenize(text).vocab.length, SEED);
const analytic = learn(W, ids);
const eps = 1e-5;
for (const k of PARAMS) {
  const w = W[k], g = analytic.g[k];
  for (let i = 0; i < w.length; i++) {
    w[i] += eps;
    const lp = loss(W, ids);
    w[i] -= 2 * eps;
    const lm = loss(W, ids);
    w[i] += eps;
    const num = (lp - lm) / (2 * eps);
    const err = Math.abs(num - g[i]);
    assert(err < 1e-4, `${k}[${i}] analytic ${g[i]} numeric ${num} err ${err}`);
  }
}

const trained = init(tokenize(text).vocab.length, SEED);
let L = loss(trained, ids), steps = 0;
for (; steps < 80 && L > 0.2; steps++) {
  const t = learn(trained, ids);
  sgd(trained, t.g, 0.4);
  L = loss(trained, ids);
}
assert(L < 0.5, `loss should fall, ended at ${L} after ${steps} steps`);

// The default sentences. The step on the set is the mean gradient. One sentence does not teach the others.
const { vocab, sents } = sentences(LINES);
const batch = sents.map((s) => s.ids);
const fresh = () => init(vocab.length, SEED);
const meanOf = (M) => batch.reduce((s, ids) => s + loss(M, ids), 0) / batch.length;
const B0 = fresh();
const bt = batch.map((ids) => learn(B0, ids));
const beforeMean = bt.reduce((s, t) => s + t.loss, 0) / bt.length;
sgd(B0, meanGrad(bt), 1e-3);
assert(meanOf(B0) < beforeMean, 'a short step on the mean gradient lowers the mean loss');

const all = fresh();
for (let i = 0; i < 40; i++) sgd(all, meanGrad(batch.map((ids) => learn(all, ids))), 0.35);
for (const ids of batch) assert(loss(all, ids) < 0.2, 'every sentence is learned when the mean gradient is used');

const one = fresh();
for (let i = 0; i < 40; i++) sgd(one, learn(one, batch[0]).g, 0.35);
assert(loss(one, batch[0]) < 0.1, 'the trained sentence is learned');
assert(loss(one, batch[3]) > 1, 'a sentence left out of the update stays unlearned');
