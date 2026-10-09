import { mmc, splitMeans, createPair, advancePair, pairStats } from './sim.js';

const $ = (id) => document.getElementById(id);
const css = getComputedStyle(document.documentElement);
const C = {};
for (const k of ['fg', 'muted', 'faint', 'rule', 'rule-hi', 'accent', 'accent-dim', 'high', 'crit', 'sunk', 'raised']) {
  C[k.replace('-', '')] = css.getPropertyValue('--' + k).trim();
}
const font = '12px IBM Plex Mono, ui-monospace, monospace';

let lambda = 2.4, mu = 1, c = 3, speed = 2, running = true;
let locked = 'rho', focus = 'rho';
let sim = createPair(lambda, mu, c, 7);
let hits = [];
let timer = 0;
let samples = [];
let lastMark = 0;

function restart() {
  sim = createPair(lambda, mu, c, 7);
  samples = [];
  lastMark = 0;
}

function mark() {
  if (sim.t - lastMark < 0.2 && samples.length) return;
  const st = pairStats(sim);
  samples.push({
    t: sim.t,
    lanes: st.lanes.qs.reduce((a, n) => a + n, 0),
    shared: st.shared.q,
  });
  lastMark = sim.t;
  const cut = sim.t - 40;
  while (samples.length && samples[0].t < cut) samples.shift();
}

function fmt(x) {
  return Number.isFinite(x) ? x.toFixed(2) : '—';
}

function note() {
  const rho = lambda / (c * mu);
  const shared = mmc(lambda, mu, c);
  const past = rho >= 1;
  if (focus === 'rho') {
    return past
      ? `ρ = ${fmt(rho)} is past 1. The registers cannot keep up, and neither layout has a steady wait.`
      : `ρ = ${fmt(rho)} is the fraction of time each register is busy. Both layouts have this same ρ. Total checkout speed is cμ = ${fmt(c * mu)}.`;
  }
  if (focus === 'lam') return `λ = ${fmt(lambda)} people arrive per unit time, on average. The gaps are messy and their mean is 1/λ = ${fmt(1 / lambda)}. Split across the registers, each line’s share is λ/c = ${fmt(lambda / c)}.`;
  if (focus === 'mu') return `μ = ${fmt(mu)} checkouts per unit time, on average, while someone is at the register. Each checkout is messy and the mean length is 1/μ = ${fmt(1 / mu)}. The bar under the person is how much of this one is left.`;
  if (focus === 'c') return `c = ${c} registers, outlined below. Together they finish cμ = ${fmt(c * mu)} people per unit time when every register is busy.`;
  if (focus === 'lanes') return past
    ? 'Each person picks a register at random and stays. At this ρ the lines grow without a steady wait.'
    : 'Each person picks a register at random and stays. That is c separate M/M/1 queues. A register can be idle while another line still has people. Mean wait Wq = ρ / (μ(1−ρ)).';
  if (focus === 'shared') return past
    ? 'One line feeds every register. At this ρ that line grows without a steady wait.'
    : 'One line. The next free register takes the next person, so a register is idle only when the line is empty. This is an M/M/c queue.';
  const er = shared.stable ? fmt(shared.erlangC) : '1';
  return `C(c, λ/μ) = ${er} is the probability every register is busy. It is called Erlang’s C formula. In the single line, that is the only time the next person has to wait.`;
}

function paintReadout() {
  const rho = lambda / (c * mu);
  const shared = mmc(lambda, mu, c);
  const lanes = splitMeans(lambda, mu, c);
  const st = pairStats(sim);
  $('nRho').textContent = fmt(rho);
  $('nLam').textContent = fmt(lambda);
  $('nMu').textContent = fmt(mu);
  $('nC').textContent = String(c);
  $('nErlang').textContent = shared.stable ? fmt(shared.erlangC) : '1';
  $('lamText').textContent = fmt(lambda);
  $('muText').textContent = fmt(mu);
  $('wLanes').textContent = lanes.stable ? fmt(lanes.Wq) : 'none';
  $('wShared').textContent = shared.stable ? fmt(shared.Wq) : 'none';
  $('formulaNote').textContent = note();
  $('measLanes').textContent = `Waiting now ${st.lanes.qs.join(' · ')}. Measured mean wait ${fmt(st.lanes.Wq)}.`;
  $('measShared').textContent = `Waiting now ${st.shared.q}. Measured mean wait ${fmt(st.shared.Wq)}.`;
  document.querySelectorAll('[data-part]').forEach((b) => {
    const on = b.dataset.part === focus;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  document.querySelectorAll('#counts .chip').forEach((b) => b.classList.toggle('on', +b.dataset.c === c));
  document.querySelectorAll('#presets .chip').forEach((b) => {
    const on = Math.abs(+b.dataset.l - lambda) < 1e-9 && Math.abs(+b.dataset.m - mu) < 1e-9 && +b.dataset.c === c;
    b.classList.toggle('on', on);
  });
}

function fit(cv) {
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

function shopper(g, x, y, color) {
  g.fillStyle = color;
  g.beginPath();
  g.arc(x, y - 8, 5, 0, Math.PI * 2);
  g.fill();
  g.beginPath();
  g.roundRect(x - 6, y - 2, 12, 11, 2);
  g.fill();
}

function counter(g, x, y, w, h, hot) {
  g.strokeStyle = hot ? C.accent : C.rulehi;
  g.lineWidth = hot ? 2 : 1;
  g.strokeRect(x + 0.5, y + 0.5, w, h);
  g.lineWidth = 1;
}

function drawStore() {
  const { g, w, h } = fit($('store'));
  g.clearRect(0, 0, w, h);
  g.font = font;
  hits = [];
  const gap = 18;
  const half = (w - gap * 3) / 2;
  const left = { x: gap, y: 10, w: half, h: h - 20 };
  const right = { x: gap * 2 + half, y: 10, w: half, h: h - 20 };
  g.strokeStyle = C.rule;
  g.beginPath();
  g.moveTo(w / 2, 8);
  g.lineTo(w / 2, h - 8);
  g.stroke();
  drawLanes(g, left);
  drawShared(g, right);
  if (focus === 'lanes' || focus === 'shared') {
    const b = focus === 'lanes' ? left : right;
    g.strokeStyle = C.accent;
    g.strokeRect(b.x + 0.5, b.y + 0.5, b.w, b.h);
  }
}

function chute(g, x, y, w, h, hot) {
  g.fillStyle = C.raised;
  g.fillRect(x, y, w, h);
  g.strokeStyle = hot ? C.accent : C.rulehi;
  g.lineWidth = hot ? 2 : 1;
  g.strokeRect(x + 0.5, y + 0.5, w, h);
  g.lineWidth = 1;
  g.strokeStyle = C.rule;
  g.setLineDash([3, 5]);
  g.beginPath();
  g.moveTo(x + w / 2, y + 6);
  g.lineTo(x + w / 2, y + h - 6);
  g.stroke();
  g.setLineDash([]);
}

function drawLanes(g, box) {
  g.fillStyle = C.faint;
  g.fillText('a line at each register', box.x, box.y + 14);
  const regH = 44, regY = box.y + box.h - regH - 8;
  const colW = box.w / c;
  const waiting = sim.lanes.some((ln) => ln.q.length);
  const laneTop = box.y + 36;
  hits.push({ part: 'lam', x: box.x, y: box.y, w: box.w, h: 26 });
  hits.push({ part: 'lanes', x: box.x, y: laneTop, w: box.w, h: regY - laneTop });
  hits.push({ part: 'c', x: box.x, y: regY, w: box.w, h: regH + 8 });

  for (let i = 0; i < c; i++) {
    const ln = sim.lanes[i];
    const cx = box.x + colW * i + colW / 2;
    const laneW = Math.min(68, colW - 10);
    const lx = cx - laneW / 2;
    const hotLane = focus === 'lanes' || focus === 'c';
    chute(g, lx, laneTop, laneW, regY - laneTop, hotLane);
    const rw = Math.min(laneW, 56);
    const rx = cx - rw / 2;
    const hot = focus === 'c' || focus === 'mu' || focus === 'rho';
    if (ln.serving && (focus === 'rho' || focus === 'mu')) {
      g.fillStyle = C.accentdim;
      g.fillRect(rx, regY, rw, regH);
    }
    counter(g, rx, regY, rw, regH, hot);
    hits.push({ part: 'mu', x: rx, y: regY, w: rw, h: regH });
    if (ln.serving) {
      shopper(g, cx, regY + 26, C.accent);
      const left = Math.max(0, (ln.nextD - sim.t) / ln.serving.svc);
      g.fillStyle = focus === 'mu' ? C.accent : C.faint;
      g.fillRect(rx + 4, regY + regH - 6, (rw - 8) * left, 3);
    }
    g.fillStyle = !ln.serving && waiting ? C.high : C.faint;
    g.fillText(ln.serving ? (focus === 'mu' ? 'μ' : String(i + 1)) : 'idle', lx + 4, laneTop + 14);

    const step = 28;
    const room = Math.max(1, Math.floor((regY - laneTop - 24) / step));
    const shown = Math.min(ln.q.length, room);
    for (let j = 0; j < shown; j++) shopper(g, cx, regY - 16 - j * step, C.high);
    if (ln.q.length > shown) {
      g.fillStyle = C.high;
      g.fillText(`+${ln.q.length - shown}`, lx + 4, laneTop + 28);
    }
  }
  if (focus === 'lam') {
    g.fillStyle = C.accent;
    g.fillText('λ', box.x + 4, box.y + 16);
  }
}

function drawShared(g, box) {
  g.fillStyle = C.faint;
  g.fillText(c > 1 ? `one line for all ${c}` : 'one line', box.x, box.y + 14);
  const regH = 44, regY = box.y + box.h - regH - 8;
  const colW = box.w / c;
  const line = sim.shared.line;
  const pool = focus === 'pool' || focus === 'shared';
  const laneTop = box.y + 36;
  const laneW = 68;
  const xs = Array.from({ length: c }, (_, i) => box.x + colW * i + colW / 2);
  const cx = box.x + box.w / 2;
  const lx = cx - laneW / 2;
  const stem = 16, bandH = 22, drop = 26;
  const bandY = c > 1 ? regY - drop - bandH : regY;
  const floor = c > 1 ? bandY : regY;
  hits.push({ part: 'lam', x: box.x, y: box.y, w: box.w, h: 26 });
  hits.push({ part: 'shared', x: box.x, y: laneTop, w: box.w, h: floor - laneTop });
  hits.push({ part: 'c', x: box.x, y: regY, w: box.w, h: regH + 8 });
  if (c > 1) {
    const x0 = xs[0] - stem / 2, span = xs[c - 1] - xs[0] + stem;
    g.fillStyle = C.raised;
    g.fillRect(x0, bandY, span, bandH);
    for (const x of xs) g.fillRect(x - stem / 2, bandY, stem, regY - bandY);
    g.strokeStyle = pool ? C.accent : C.rulehi;
    g.lineWidth = pool ? 2 : 1;
    g.strokeRect(x0 + 0.5, bandY + 0.5, span, bandH);
    for (const x of xs) g.strokeRect(x - stem / 2 + 0.5, bandY + 0.5, stem, regY - bandY);
    g.lineWidth = 1;
  }
  chute(g, lx, laneTop, laneW, floor - laneTop, pool);

  const step = 28;
  const room = Math.max(1, Math.floor((floor - laneTop - 24) / step));
  const shown = Math.min(line.length, room);
  for (let j = 0; j < shown; j++) shopper(g, cx, floor - 16 - j * step, C.high);
  if (line.length > shown) {
    g.fillStyle = C.high;
    g.fillText(`+${line.length - shown}`, lx + 4, laneTop + 28);
  }

  for (let i = 0; i < c; i++) {
    const busy = sim.shared.servers[i];
    const x = xs[i];
    const rw = Math.min(52, colW - 8);
    const rx = x - rw / 2;
    const hot = focus === 'c' || focus === 'mu' || focus === 'rho' || pool;
    if (busy && (focus === 'rho' || focus === 'mu' || pool)) {
      g.fillStyle = C.accentdim;
      g.fillRect(rx, regY, rw, regH);
    }
    counter(g, rx, regY, rw, regH, hot);
    hits.push({ part: 'mu', x: rx, y: regY, w: rw, h: regH });
    g.fillStyle = busy ? C.faint : C.muted;
    g.fillText(String(i + 1), rx + 6, regY + 16);
    if (busy) {
      shopper(g, x, regY + 26, C.accent);
      const left = Math.max(0, (sim.shared.nextD[i] - sim.t) / busy.svc);
      g.fillStyle = focus === 'mu' ? C.accent : C.faint;
      g.fillRect(rx + 4, regY + regH - 6, (rw - 8) * left, 3);
      if (focus === 'mu') {
        g.fillStyle = C.accent;
        g.fillText('μ', rx + 4, regY - 6);
      }
    } else {
      g.fillStyle = C.faint;
      g.fillText('idle', rx + 4, regY + 32);
    }
  }
  if (focus === 'lam') {
    g.fillStyle = C.accent;
    g.fillText('λ', box.x + 4, box.y + 16);
  }
}

function drawTrace() {
  const { g, w, h } = fit($('trace'));
  g.clearRect(0, 0, w, h);
  g.font = font;
  const x = 16, y = 12, pw = w - 32, ph = h - 32;
  const now = sim.t, t0 = now - 40;
  const lanes = splitMeans(lambda, mu, c);
  const shared = mmc(lambda, mu, c);
  let ymax = 4;
  for (const s of samples) if (s.t >= t0) ymax = Math.max(ymax, s.lanes, s.shared);
  if (lanes.stable) ymax = Math.max(ymax, Math.min(lanes.Lq, 40));
  if (shared.stable) ymax = Math.max(ymax, Math.min(shared.Lq, 40));
  ymax = Math.ceil(ymax);
  const X = (t) => x + pw * (t - t0) / 40;
  const Y = (n) => y + ph - ph * Math.min(n, ymax) / ymax;
  g.strokeStyle = C.rule;
  g.strokeRect(x + 0.5, y + 0.5, pw, ph);
  g.fillStyle = C.faint;
  g.fillText('people waiting', x + 8, y + 14);
  g.fillText(String(ymax), x + 4, y + 28);
  const mean = (n, color, label) => {
    const yy = Y(n);
    g.strokeStyle = color;
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(x, yy);
    g.lineTo(x + pw, yy);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = color;
    g.fillText(label, x + 8, Math.min(y + ph - 4, Math.max(y + 26, yy - 4)));
  };
  if (lanes.stable) mean(lanes.Lq, C.high, 'mean, lines');
  if (shared.stable) mean(shared.Lq, C.accent, 'mean, one line');
  else {
    g.fillStyle = C.crit;
    g.fillText('no steady mean', x + pw - 120, y + 14);
  }
  const stroke = (key, color) => {
    g.strokeStyle = color;
    g.lineWidth = 1.5;
    g.beginPath();
    let on = false;
    for (const s of samples) {
      if (s.t < t0) continue;
      const px = X(s.t), py = Y(s[key]);
      if (!on) { g.moveTo(px, py); on = true; }
      else g.lineTo(px, py);
    }
    g.stroke();
    g.lineWidth = 1;
  };
  stroke('lanes', C.high);
  stroke('shared', C.accent);
}

function drawKnee() {
  const { g, w, h } = fit($('knee'));
  g.clearRect(0, 0, w, h);
  g.font = font;
  const x0 = 40, y0 = 16, x1 = w - 16, y1 = h - 28, ymax = 12;
  const X = (r) => x0 + (x1 - x0) * r;
  const Y = (v) => y1 - (y1 - y0) * Math.min(Math.max(v, 0), ymax) / ymax;
  g.strokeStyle = C.rule;
  g.beginPath();
  g.moveTo(x0, y0);
  g.lineTo(x0, y1);
  g.lineTo(x1, y1);
  g.stroke();
  g.fillStyle = C.faint;
  g.fillText('Wq', 6, y0 + 10);
  g.fillText('12', 12, Y(12) + 4);
  g.fillText('0', x0, y1 + 16);
  g.fillText('ρ', (x0 + x1) / 2 - 4, y1 + 16);
  g.fillText('1', x1 - 8, y1 + 16);
  const curve = (fn, color) => {
    g.strokeStyle = color;
    g.lineWidth = 1.5;
    g.beginPath();
    for (let i = 0; i <= 120; i++) {
      const r = (i / 120) * 0.98;
      const m = fn(r * c * mu, mu, c);
      const px = X(r), py = Y(m.stable ? m.Wq : ymax);
      if (i) g.lineTo(px, py);
      else g.moveTo(px, py);
    }
    g.stroke();
    g.lineWidth = 1;
  };
  curve(splitMeans, C.high);
  curve(mmc, C.accent);
  const rho = lambda / (c * mu);
  if (rho < 1 && rho >= 0) {
    const Ls = splitMeans(lambda, mu, c), Sh = mmc(lambda, mu, c);
    g.strokeStyle = C.faint;
    g.setLineDash([3, 3]);
    g.beginPath();
    g.moveTo(X(rho), y1);
    g.lineTo(X(rho), Y(Ls.Wq));
    g.stroke();
    g.setLineDash([]);
    for (const [wq, color] of [[Ls.Wq, C.high], [Sh.Wq, C.accent]]) {
      g.fillStyle = color;
      g.beginPath();
      g.arc(X(rho), Y(wq), 4, 0, Math.PI * 2);
      g.fill();
    }
    g.fillStyle = C.fg;
    g.fillText('ρ = ' + rho.toFixed(2), Math.min(X(rho) + 8, x1 - 72), y0 + 14);
  } else {
    g.fillStyle = C.crit;
    g.fillText('ρ = ' + rho.toFixed(2) + '  no steady mean', x0 + 8, y0 + 18);
  }
}

function drawDist() {
  const { g, w, h } = fit($('dist'));
  g.clearRect(0, 0, w, h);
  g.font = font;
  g.globalAlpha = 1;
  const panels = [
    { title: 'between arrivals', mean: 1 / lambda, rate: lambda, samples: sim.gaps, color: C.high, dim: focus === 'mu' },
    { title: 'one checkout', mean: 1 / mu, rate: mu, samples: sim.svcs, color: C.accent, dim: focus === 'lam' },
  ];
  const gutter = 28;
  const pw = (w - gutter * 3) / 2;
  panels.forEach((p, i) => hist(g, gutter + i * (pw + gutter), 12, pw, h - 40, p));
}

function hist(g, x0, y0, pw, ph, p) {
  const bins = 18;
  const xmax = Math.max(p.mean * 4, 0.5);
  const bw = xmax / bins;
  const counts = Array(bins).fill(0);
  for (const t of p.samples) {
    const b = Math.floor(t / bw);
    if (b >= 0 && b < bins) counts[b]++;
  }
  const n = p.samples.length;
  let ymax = p.rate;
  if (n) for (const c of counts) ymax = Math.max(ymax, (c / n) / bw);
  ymax *= 1.08;
  const X = (t) => x0 + (pw * t) / xmax;
  const Y = (d) => y0 + ph - (ph * Math.min(Math.max(d, 0), ymax)) / ymax;
  g.strokeStyle = C.rule;
  g.strokeRect(x0 + 0.5, y0 + 0.5, pw, ph);
  g.globalAlpha = p.dim ? 0.35 : 1;
  g.fillStyle = p.color;
  const barW = Math.max(1, pw / bins - 1);
  for (let i = 0; i < bins; i++) {
    if (!n || !counts[i]) continue;
    const y = Y((counts[i] / n) / bw);
    g.fillRect(X(i * bw) + 1, y, barW, y0 + ph - y);
  }
  g.strokeStyle = p.color;
  g.lineWidth = 1.5;
  g.beginPath();
  for (let i = 0; i <= 48; i++) {
    const t = (xmax * i) / 48;
    const y = Y(p.rate * Math.exp(-p.rate * t));
    if (i) g.lineTo(X(t), y);
    else g.moveTo(X(t), y);
  }
  g.stroke();
  g.setLineDash([4, 4]);
  g.beginPath();
  g.moveTo(X(p.mean), y0);
  g.lineTo(X(p.mean), y0 + ph);
  g.stroke();
  g.setLineDash([]);
  g.lineWidth = 1;
  g.globalAlpha = 1;
  g.fillStyle = p.dim ? C.faint : p.color;
  g.fillText(p.title, x0, y0 + ph + 16);
  const label = 'mean ' + p.mean.toFixed(2);
  const lx = Math.min(X(p.mean) + 6, x0 + pw - 72);
  g.fillText(label, lx, y0 + 14);
}

function paint() {
  mark();
  paintReadout();
  drawStore();
  drawDist();
  drawTrace();
  drawKnee();
}

function frame() {
  timer = 0;
  const now = performance.now();
  if (frame.then != null && running) {
    const dt = Math.min(0.05, (now - frame.then) / 1000);
    advancePair(sim, sim.t + dt * speed);
  }
  frame.then = now;
  paint();
  if (running) timer = setTimeout(frame, 32);
}

function readRates(part) {
  lambda = +$('lam').value;
  mu = +$('mu').value;
  $('spdText').textContent = String(speed);
  restart();
  if (part) locked = focus = part;
  paint();
}

document.body.addEventListener('mouseover', (e) => {
  const b = e.target.closest('[data-part]');
  if (!b || focus === b.dataset.part) return;
  focus = b.dataset.part;
  paint();
});
document.body.addEventListener('mouseout', (e) => {
  const b = e.target.closest('[data-part]');
  if (!b || b.contains(e.relatedTarget)) return;
  focus = locked;
  paint();
});
document.body.addEventListener('click', (e) => {
  const b = e.target.closest('[data-part]');
  if (!b) return;
  locked = focus = b.dataset.part;
  paint();
});

$('lam').oninput = () => readRates('lam');
$('mu').oninput = () => readRates('mu');
$('spd').oninput = () => { speed = +$('spd').value; $('spdText').textContent = String(speed); };
$('counts').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  c = +b.dataset.c;
  readRates('c');
};
$('presets').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  $('lam').value = b.dataset.l;
  $('mu').value = b.dataset.m;
  c = +b.dataset.c;
  readRates('rho');
};
$('run').onclick = () => {
  running = !running;
  $('run').textContent = running ? 'Pause' : 'Run';
  $('run').classList.toggle('pri', running);
  if (running && !timer) frame();
  else paint();
};
$('reset').onclick = () => { restart(); paint(); };
$('store').onclick = (e) => {
  const r = $('store').getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  for (let i = hits.length - 1; i >= 0; i--) {
    const h = hits[i];
    if (x >= h.x && x < h.x + h.w && y >= h.y && y < h.y + h.h) {
      locked = focus = h.part;
      paint();
      return;
    }
  }
};

frame();
