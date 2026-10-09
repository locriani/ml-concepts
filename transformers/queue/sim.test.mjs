// M/M/1 formulas, plus one sample path: the time average settles on L = ρ/(1−ρ),
// Little's law holds, and ρ > 1 grows a line.
import assert from 'node:assert';
import { lcg, mm1, create, advance, stats, mmc, splitMeans, createPair, advancePair, pairStats, configurePair } from './src/sim.js';

const near = (a, b, tol, msg) => assert(Math.abs(a - b) < tol, `${msg}: ${a} vs ${b}`);

const m = mm1(0.5, 1);
assert.equal(m.stable, true);
near(m.L, 1, 1e-12, 'L');
near(m.Lq, 0.5, 1e-12, 'Lq');
near(m.W, 2, 1e-12, 'W');
near(m.Wq, 1, 1e-12, 'Wq');
assert.equal(mm1(1, 1).stable, false);
assert.equal(mm1(1.2, 1).stable, false);

const s = create(0.5, 1, lcg(7));
advance(s, 8000);
const st = stats(s);
const lam = st.arrivals / s.t;
near(st.L, 1, 0.15, 'measured L');
near(st.L, lam * st.W, 0.08, "Little's law");

const u = create(1.5, 1, lcg(7));
advance(u, 300);
assert(stats(u).n > 50, `unstable line grew, n=${stats(u).n}`);

const shared = mmc(2.4, 1, 3);
const lanes = splitMeans(2.4, 1, 3);
near(shared.rho, 0.8, 1e-12, 'ρ');
near(shared.Wq, 1.0786516853932584, 1e-9, 'M/M/3 wait');
near(lanes.Wq, 4, 1e-12, 'separate-line wait');
assert(shared.Wq < lanes.Wq, 'one line waits less');
near(mmc(0.5, 1, 1).Wq, mm1(0.5, 1).Wq, 1e-12, 'one register matches M/M/1');
assert.equal(mmc(3.6, 1, 3).stable, false);

const pair = createPair(2.4, 1, 3, 7);
advancePair(pair, 80000);
const ps = pairStats(pair);
assert.equal(ps.bad, 0, 'shared line never waits while a register is idle');
assert(ps.stuck > 10, `separate lines left a register idle, stuck=${ps.stuck}`);
near(ps.shared.Wq, shared.Wq, 0.12, 'measured shared wait');
near(ps.lanes.Wq, lanes.Wq, 0.25, 'measured separate wait');
assert(ps.shared.Wq < ps.lanes.Wq, 'sample path: one line is shorter');
near(pair.gapSum / pair.gapN, 1 / 2.4, 0.02, 'mean gap is 1/λ');
near(pair.svcSum / pair.svcN, 1, 0.02, 'mean checkout is 1/μ');
assert(pair.gaps.length <= 180 && pair.svcs.length <= 180, 'histogram keeps a window');

const live = createPair(2.4, 1, 3, 7);
advancePair(live, 40);
const laneN = live.lanes.reduce((a, ln) => a + ln.q.length + (ln.serving ? 1 : 0), 0);
const sharedN = live.shared.line.length + live.shared.servers.filter(Boolean).length;
const done = live.shared.nDone;
assert(done > 0, 'the run has finished someone');
configurePair(live, 1.2, 0.8, 4);
assert.equal(live.t, 40);
assert.equal(live.shared.nDone, done);
assert.equal(live.c, 4);
assert.equal(live.lambda, 1.2);
assert.equal(live.mu, 0.8);
assert.equal(live.shared.bad, 0);
configurePair(live, 1.2, 0.8, 1);
const laneAfter = live.lanes.reduce((a, ln) => a + ln.q.length + (ln.serving ? 1 : 0), 0);
const sharedAfter = live.shared.line.length + live.shared.servers.filter(Boolean).length;
assert.equal(laneAfter, laneN, 'closing registers keeps the separate-line people');
assert.equal(sharedAfter, sharedN, 'closing registers keeps the shared-line people');
assert.equal(live.shared.bad, 0);
advancePair(live, 70);
assert.equal(live.shared.bad, 0, 'a changed store still never waits while a register is idle');
