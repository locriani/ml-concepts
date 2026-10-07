// Top-3 principal coordinates of n vectors. Works on the n x n Gram matrix (n is small, dimension is not),
// power iteration with deflation, deterministic start so the picture is stable between runs.

export function pca3(vecs) {
  const n = vecs.length, d = vecs[0].length;
  const mean = Array.from({ length: d }, (_, j) => vecs.reduce((s, v) => s + v[j], 0) / n);
  const Xc = vecs.map((v) => v.map((x, j) => x - mean[j]));
  const G = Xc.map((a) => Xc.map((b) => a.reduce((s, x, j) => s + x * b[j], 0)));
  const coords = vecs.map(() => [0, 0, 0]);
  for (let k = 0; k < 3; k++) {
    let u = Array.from({ length: n }, (_, i) => Math.sin(i * 1.3 + k + 1));
    let lam = 0;
    for (let it = 0; it < 300; it++) {
      const g = G.map((r) => r.reduce((s, x, j) => s + x * u[j], 0));
      lam = Math.hypot(...g) || 1;
      u = g.map((x) => x / lam);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i][j] -= lam * u[i] * u[j];
    const s = Math.sqrt(Math.max(lam, 0));
    u.forEach((x, i) => { coords[i][k] = x * s; });
  }
  return coords;
}

export const cosine = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
};
