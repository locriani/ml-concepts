// Poisson arrivals, exponential service.
// One register is M/M/1. Several registers are either c separate lines or one shared line (M/M/c).
// ponytail: a separate line sends each arrival to a random register and they stay there.
// Joining the shortest line is a third rule; add it when the lesson is that looking still loses to one line.

export function lcg(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

function expTime(rng, rate) {
  return -Math.log(1 - rng()) / rate;
}

// Steady means. None exist at ρ ≥ 1: the line grows without a bound.
export function mm1(lambda, mu) {
  const rho = lambda / mu;
  if (!(rho < 1)) return { rho, stable: false };
  return {
    rho,
    stable: true,
    L: rho / (1 - rho),
    Lq: rho * rho / (1 - rho),
    W: 1 / (mu - lambda),
    Wq: rho / (mu - lambda),
  };
}

export function create(lambda, mu, rng) {
  const s = {
    t: 0, lambda, mu, rng,
    nextA: 0, spanA: 0, nextD: Infinity,
    queue: [], serving: null,
    areaN: 0, areaQ: 0, arrivals: 0, completions: 0, sojourn: 0,
  };
  scheduleA(s);
  return s;
}

function scheduleA(s) {
  s.spanA = expTime(s.rng, s.lambda);
  s.nextA = s.t + s.spanA;
}

export function nSys(s) {
  return s.queue.length + (s.serving == null ? 0 : 1);
}

function arrive(s) {
  s.arrivals++;
  scheduleA(s);
  if (s.serving == null) {
    s.serving = s.t;
    s.nextD = s.t + expTime(s.rng, s.mu);
  } else s.queue.push(s.t);
}

function depart(s) {
  s.completions++;
  s.sojourn += s.t - s.serving;
  if (s.queue.length) {
    s.serving = s.queue.shift();
    s.nextD = s.t + expTime(s.rng, s.mu);
  } else {
    s.serving = null;
    s.nextD = Infinity;
  }
}

// Advance the clock to t1, folding in every arrival and departure on the way.
export function advance(s, t1) {
  if (!(t1 > s.t)) return;
  let guard = 0;
  while (Math.min(s.nextA, s.nextD) <= t1) {
    const te = Math.min(s.nextA, s.nextD);
    const n = nSys(s);
    s.areaN += n * (te - s.t);
    s.areaQ += s.queue.length * (te - s.t);
    s.t = te;
    if (s.nextA <= s.nextD) arrive(s);
    else depart(s);
    if (++guard > 1e6) break;
  }
  const n = nSys(s);
  s.areaN += n * (t1 - s.t);
  s.areaQ += s.queue.length * (t1 - s.t);
  s.t = t1;
}

export function stats(s) {
  const t = s.t || 1e-9;
  return {
    L: s.areaN / t,
    Lq: s.areaQ / t,
    W: s.completions ? s.sojourn / s.completions : 0,
    n: nSys(s),
    q: s.queue.length,
    busy: s.serving != null,
    arrivals: s.arrivals,
    completions: s.completions,
  };
}

// M/M/c. ρ = λ/(cμ). C is Erlang's C formula, the chance every register is busy.
export function mmc(lambda, mu, c) {
  const rho = lambda / (c * mu);
  if (!(rho < 1)) return { rho, stable: false, erlangC: 1 };
  const a = lambda / mu;
  let sum = 0, term = 1;
  for (let k = 0; k < c; k++) {
    sum += term;
    term *= a / (k + 1);
  }
  const tail = term / (1 - rho);
  const erlangC = tail / (sum + tail);
  const Wq = erlangC / (c * mu - lambda);
  return { rho, stable: true, erlangC, Wq, Lq: lambda * Wq, W: Wq + 1 / mu };
}

// c separate M/M/1 queues, each with arrival rate λ/c. Same ρ as the shared line.
export function splitMeans(lambda, mu, c) {
  const one = mm1(lambda / c, mu);
  const rho = lambda / (c * mu);
  if (!one.stable) return { rho, stable: false };
  return { rho, stable: true, Wq: one.Wq, Lq: c * one.Lq, W: one.W };
}

export function createPair(lambda, mu, c, seed) {
  const s = {
    t: 0, lambda, mu, c, stuck: 0,
    arrRng: lcg(seed), svcRng: lcg(seed + 1), pickRng: lcg(seed + 2),
    nextA: 0, spanA: 0,
    gaps: [], svcs: [], gapSum: 0, gapN: 0, svcSum: 0, svcN: 0,
    shared: emptyPool(c),
    lanes: Array.from({ length: c }, emptyLane),
  };
  schedulePair(s);
  return s;
}

// ponytail: the histogram keeps the last 180 draws. The mean uses every draw.
function remember(buf, x, s, sum, n) {
  buf.push(x);
  s[sum] += x;
  s[n]++;
  if (buf.length > 180) buf.shift();
}

function emptyLane() {
  return { q: [], serving: null, nextD: Infinity, wait: 0, nDone: 0 };
}

function emptyPool(c) {
  return {
    line: [], servers: Array(c).fill(null), nextD: Array(c).fill(Infinity),
    wait: 0, nDone: 0, bad: 0,
  };
}

function schedulePair(s) {
  s.spanA = expTime(s.arrRng, s.lambda);
  s.nextA = s.t + s.spanA;
  remember(s.gaps, s.spanA, s, 'gapSum', 'gapN');
}

function arrivePair(s) {
  const svc = expTime(s.svcRng, s.mu);
  remember(s.svcs, svc, s, 'svcSum', 'svcN');
  giveShared(s, { born: s.t, svc });
  giveLane(s, Math.floor(s.pickRng() * s.c), { born: s.t, svc });
  schedulePair(s);
  checkShared(s);
}

function giveShared(s, cust) {
  const i = s.shared.servers.findIndex((x) => x == null);
  if (i >= 0) startShared(s, i, cust);
  else s.shared.line.push(cust);
}

function startShared(s, i, cust) {
  cust.started = s.t;
  s.shared.servers[i] = cust;
  s.shared.nextD[i] = s.t + cust.svc;
}

function giveLane(s, i, cust) {
  const ln = s.lanes[i];
  if (ln.serving == null) startLane(s, i, cust);
  else ln.q.push(cust);
}

function startLane(s, i, cust) {
  const ln = s.lanes[i];
  cust.started = s.t;
  ln.serving = cust;
  ln.nextD = s.t + cust.svc;
}

function departShared(s, i) {
  const cust = s.shared.servers[i];
  s.shared.nDone++;
  s.shared.wait += cust.started - cust.born + (cust.held || 0);
  if (s.shared.line.length) startShared(s, i, s.shared.line.shift());
  else { s.shared.servers[i] = null; s.shared.nextD[i] = Infinity; }
  checkShared(s);
}

function departLane(s, i) {
  const ln = s.lanes[i];
  ln.nDone++;
  ln.wait += ln.serving.started - ln.serving.born + (ln.serving.held || 0);
  if (ln.q.length) startLane(s, i, ln.q.shift());
  else { ln.serving = null; ln.nextD = Infinity; }
}

function checkShared(s) {
  const idle = s.shared.servers.some((x) => x == null);
  if (s.shared.line.length && idle) s.shared.bad++;
}

// Keep the people. New λ and μ apply to the time still left; a new register
// opens empty; a closed one parks its customer on the last register that stays.
// ponytail: a closed register's queue joins the last open line. It does not rebalance the others.
export function configurePair(s, lambda, mu, c) {
  if (lambda !== s.lambda) {
    const left = Math.max(0, s.nextA - s.t);
    const scale = s.lambda / lambda;
    const realized = Math.max(0, s.spanA - left) + left * scale;
    if (s.gaps.length) {
      s.gapSum += realized - s.gaps[s.gaps.length - 1];
      s.gaps[s.gaps.length - 1] = realized;
    }
    s.spanA = realized;
    s.nextA = s.t + left * scale;
    s.lambda = lambda;
  }
  if (mu !== s.mu) applyMu(s, mu);
  if (c !== s.c) resizePair(s, c);
}

function applyMu(s, mu) {
  const scale = s.mu / mu;
  const serving = (cust, nextD) => {
    const left = Math.max(0, nextD - s.t) * scale;
    cust.svc = (s.t - cust.started) + left;
    return s.t + left;
  };
  for (const ln of s.lanes) {
    for (const cust of ln.q) cust.svc *= scale;
    if (ln.serving) ln.nextD = serving(ln.serving, ln.nextD);
  }
  for (const cust of s.shared.line) cust.svc *= scale;
  s.shared.servers.forEach((cust, i) => {
    if (cust) s.shared.nextD[i] = serving(cust, s.shared.nextD[i]);
  });
  s.mu = mu;
}

function park(cust, now, left) {
  cust.held = (cust.held || 0) + (cust.started - cust.born);
  cust.born = now;
  cust.started = null;
  cust.svc = Math.max(left, 1e-9);
}

function drainShared(s) {
  for (;;) {
    const i = s.shared.servers.findIndex((x) => x == null);
    if (i < 0 || !s.shared.line.length) break;
    startShared(s, i, s.shared.line.shift());
  }
}

function resizePair(s, c) {
  if (c > s.c) {
    for (let i = s.c; i < c; i++) {
      s.lanes.push(emptyLane());
      s.shared.servers.push(null);
      s.shared.nextD.push(Infinity);
    }
  } else {
    const dest = s.lanes[c - 1];
    for (let i = s.c - 1; i >= c; i--) {
      const ln = s.lanes[i];
      if (ln.serving) {
        park(ln.serving, s.t, Math.max(0, ln.nextD - s.t));
        dest.q.unshift(ln.serving);
      }
      for (const cust of ln.q) dest.q.push(cust);
      const cust = s.shared.servers[i];
      if (cust) {
        park(cust, s.t, Math.max(0, s.shared.nextD[i] - s.t));
        s.shared.line.unshift(cust);
      }
    }
    s.lanes.length = c;
    s.shared.servers.length = c;
    s.shared.nextD.length = c;
    if (dest.serving == null && dest.q.length) startLane(s, c - 1, dest.q.shift());
  }
  s.c = c;
  drainShared(s);
  checkShared(s);
}

function nextEvent(s) {
  let te = s.nextA, kind = 'a', who = -1;
  s.shared.nextD.forEach((t, i) => { if (t < te) { te = t; kind = 'sd'; who = i; } });
  s.lanes.forEach((ln, i) => { if (ln.nextD < te) { te = ln.nextD; kind = 'ld'; who = i; } });
  return { te, kind, who };
}

function accrue(s, dt) {
  if (!(dt > 0)) return;
  const waiting = s.lanes.some((ln) => ln.q.length > 0);
  const idle = s.lanes.some((ln) => ln.serving == null);
  if (waiting && idle) s.stuck += dt;
}

export function advancePair(s, t1) {
  if (!(t1 > s.t)) return;
  let guard = 0;
  for (;;) {
    const ev = nextEvent(s);
    if (!(ev.te <= t1)) break;
    accrue(s, ev.te - s.t);
    s.t = ev.te;
    if (ev.kind === 'a') arrivePair(s);
    else if (ev.kind === 'sd') departShared(s, ev.who);
    else departLane(s, ev.who);
    if (++guard > 1e6) break;
  }
  accrue(s, t1 - s.t);
  s.t = t1;
}

export function pairStats(s) {
  const lanesDone = s.lanes.reduce((a, ln) => a + ln.nDone, 0);
  const lanesWait = s.lanes.reduce((a, ln) => a + ln.wait, 0);
  return {
    t: s.t,
    stuck: s.stuck,
    bad: s.shared.bad,
    shared: {
      Wq: s.shared.nDone ? s.shared.wait / s.shared.nDone : 0,
      q: s.shared.line.length,
      busy: s.shared.servers.map((x) => x != null),
      nDone: s.shared.nDone,
    },
    lanes: {
      Wq: lanesDone ? lanesWait / lanesDone : 0,
      qs: s.lanes.map((ln) => ln.q.length),
      busy: s.lanes.map((ln) => ln.serving != null),
      nDone: lanesDone,
    },
  };
}
