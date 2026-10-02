// ComfyUI-Darkroom -- tone-curve maths for the freeform curve editor.
//
// Line-for-line twin of utils/tone_curve_ops.py, so the curve drawn on the node
// is the curve the backend applies (tools/test_film_tone_curve.py runs these
// functions under Node and compares them with Python). No imports on purpose:
// Node can load this file directly.

export const IDENTITY_POINTS = "0,0;1,1";
export const BUMP_HALF_WIDTH = 0.25;
export const BUMP_AMOUNT = 0.15;
export const CONTRAST_K = 0.6;

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// "x,y;x,y;..." -> sorted [[x, y], ...] clamped to [0,1]; identity on any error.
export function parsePoints(text) {
  try {
    const pts = [];
    for (const chunk of String(text).split(";")) {
      if (!chunk.trim()) continue;
      const parts = chunk.split(",");
      if (parts.length !== 2) throw new Error("bad point");
      const DEC = /^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$/;
      if (!DEC.test(parts[0]) || !DEC.test(parts[1])) throw new Error("not a number");
      const x = Number(parts[0]), y = Number(parts[1]);
      pts.push([clamp01(x), clamp01(y)]);
    }
    pts.sort((a, b) => a[0] - b[0]);
    const dedup = [];
    for (const p of pts) {
      if (dedup.length && Math.abs(p[0] - dedup[dedup.length - 1][0]) < 1e-6) dedup[dedup.length - 1] = p;
      else dedup.push(p);
    }
    if (dedup.length < 2) throw new Error("need at least two points");
    return dedup;
  } catch (_e) {
    return [[0, 0], [1, 1]];
  }
}

export function formatPoints(pts) {
  return pts.map(([x, y]) => `${+x.toFixed(4)},${+y.toFixed(4)}`).join(";");
}

// Second derivatives of the natural cubic spline (M0 = Mn-1 = 0), Thomas solve.
function secondDerivs(xs, ys) {
  const n = xs.length;
  const M = new Array(n).fill(0);
  if (n < 3) return M;
  const h = [];
  for (let i = 0; i < n - 1; i++) h.push(xs[i + 1] - xs[i]);
  const m = n - 2;
  const a = [], b = [], c = [], d = [];
  for (let r = 0; r < m; r++) {
    a.push(h[r]);
    b.push(2 * (h[r] + h[r + 1]));
    c.push(h[r + 1]);
    d.push(6 * ((ys[r + 2] - ys[r + 1]) / h[r + 1] - (ys[r + 1] - ys[r]) / h[r]));
  }
  for (let i = 1; i < m; i++) {
    const w = a[i] / b[i - 1];
    b[i] -= w * c[i - 1];
    d[i] -= w * d[i - 1];
  }
  const Min = new Array(m).fill(0);
  Min[m - 1] = d[m - 1] / b[m - 1];
  for (let i = m - 2; i >= 0; i--) Min[i] = (d[i] - c[i] * Min[i + 1]) / b[i];
  for (let i = 0; i < m; i++) M[i + 1] = Min[i];
  return M;
}

// Natural cubic spline through pts, evaluated at each x of xsEval. Flat beyond
// the end points, like Capture One.
export function naturalCubic(pts, xsEval) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const n = xs.length;
  const out = new Array(xsEval.length);
  if (n === 2) {
    const span = Math.max(xs[1] - xs[0], 1e-9);
    for (let j = 0; j < xsEval.length; j++) {
      const t = clamp01((xsEval[j] - xs[0]) / span);
      out[j] = ys[0] + t * (ys[1] - ys[0]);
    }
    return out;
  }
  const M = secondDerivs(xs, ys);
  for (let j = 0; j < xsEval.length; j++) {
    const xc = Math.min(Math.max(xsEval[j], xs[0]), xs[n - 1]);
    let k = 0;
    while (k < n - 2 && xc >= xs[k + 1]) k++;
    const hk = xs[k + 1] - xs[k];
    const t1 = xs[k + 1] - xc, t0 = xc - xs[k];
    out[j] = M[k] * t1 ** 3 / (6 * hk) + M[k + 1] * t0 ** 3 / (6 * hk)
           + (ys[k] / hk - M[k] * hk / 6) * t1 + (ys[k + 1] / hk - M[k + 1] * hk / 6) * t0;
  }
  return out;
}

function bump(y, centre) {
  const d = Math.abs(y - centre);
  return d < BUMP_HALF_WIDTH ? 0.5 * (1 + Math.cos(Math.PI * d / BUMP_HALF_WIDTH)) : 0;
}

// The full transfer curve sampled at `size` points on [0, 1].
export function compose(pts, contrast = 0, shadows = 0, midtones = 0, highlights = 0, size = 256) {
  const xs = [];
  for (let i = 0; i < size; i++) xs.push(i / (size - 1));
  const y = naturalCubic(pts, xs);
  const k = CONTRAST_K * contrast / 100;
  let run = -Infinity;
  for (let i = 0; i < size; i++) {
    let v = y[i] - k * Math.sin(2 * Math.PI * y[i]) / (2 * Math.PI);
    v = v + BUMP_AMOUNT * (shadows / 100 * bump(v, 0.25) + midtones / 100 * bump(v, 0.5)
                          + highlights / 100 * bump(v, 0.75));
    run = Math.max(run, v);
    y[i] = clamp01(run);
  }
  return y;
}

// Saved 1.28 workflows: widgets_values = [stock, strength, toe, shoulder, gamma]
// (and the unreleased dev layout with a trailing recovery bool). Returns the new
// layout [stock, strength, recovery, curve_points, contrast, shadows, midtones,
// highlights], or null if wv is already current.
export function migrateFilmStockValues(wv) {
  if (!Array.isArray(wv)) return null;
  const legacy = (wv.length === 5 || wv.length === 6) &&
    typeof wv[2] === "number" && typeof wv[3] === "number" && typeof wv[4] === "number";
  if (!legacy) return null;
  const recovery = wv.length === 6 && typeof wv[5] === "boolean" ? wv[5] : true;
  return [wv[0], wv[1], recovery, IDENTITY_POINTS, 0, 0, 0, 0];
}
