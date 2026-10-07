// Minimal 3D scatter on a 2D canvas: yaw/pitch rotation, light perspective, zoom and pan.
// Drag rotates; shift-drag, right-drag or Shift+arrows pan; scroll, pinch or +/- zoom; arrows rotate.
// paint(s) draws one frame. World coords are [x, y(up), z(depth)].

export function createScene(canvas, paint) {
  const home = { yaw: -0.7, pitch: 0.3 };
  const rot = { ...home }, view = { zoom: 1, x: 0, y: 0 }; // view.x/y: pan in pixels
  let drag = null;

  function redraw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#0a0d10';
    g.fillRect(0, 0, w, h);
    const cy = Math.cos(rot.yaw), sy = Math.sin(rot.yaw), cp = Math.cos(rot.pitch), sp = Math.sin(rot.pitch);
    const S = Math.min(w, h) * 0.3 * view.zoom;
    const proj = ([x, y, z]) => {
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const k = 4 / (4 - z2);
      return { x: w / 2 + view.x + x1 * S * k, y: h / 2 + view.y - y2 * S * k, z: z2, k };
    };
    const line = (a, b, col, lw, dash) => {
      const pa = proj(a), pb = proj(b);
      g.beginPath(); g.moveTo(pa.x, pa.y); g.lineTo(pb.x, pb.y);
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]);
    };
    paint({ g, w, h, proj, line });
  }

  const tilt = (dx, dy) => { rot.yaw += dx; rot.pitch = Math.max(-1.4, Math.min(1.4, rot.pitch + dy)); redraw(); };
  const pan = (dx, dy) => { view.x += dx; view.y += dy; redraw(); };
  const zoomAt = (r, cx, cy) => { // keep the point under (cx, cy) fixed while scaling
    const z = Math.max(0.3, Math.min(8, view.zoom * r)), f = z / view.zoom;
    view.x = (cx - canvas.clientWidth / 2) - ((cx - canvas.clientWidth / 2) - view.x) * f;
    view.y = (cy - canvas.clientHeight / 2) - ((cy - canvas.clientHeight / 2) - view.y) * f;
    view.zoom = z; redraw();
  };
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button !== 0 }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.pan) pan(dx, dy); else tilt(dx * 0.01, dy * 0.01);
    drag = { ...drag, x: e.clientX, y: e.clientY };
  });
  const end = () => { drag = null; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('wheel', (e) => { // trackpad pinch arrives as ctrl+wheel
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002)), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  canvas.addEventListener('keydown', (e) => {
    if (e.key === '+' || e.key === '=' || e.key === '-') { e.preventDefault(); zoomAt(e.key === '-' ? 1 / 1.2 : 1.2, canvas.clientWidth / 2, canvas.clientHeight / 2); return; }
    const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!k) return;
    e.preventDefault();
    if (e.shiftKey) pan(-k[0] * 30, -k[1] * 30); else tilt(k[0] * 0.1, k[1] * 0.1);
  });
  new ResizeObserver(redraw).observe(canvas);

  return { redraw, reset: () => { Object.assign(rot, home); Object.assign(view, { zoom: 1, x: 0, y: 0 }); redraw(); } };
}
