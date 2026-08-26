// ComfyUI-Darkroom -- luma-null chroma basis for CARTESIAN wheel groups.
//
// Lift Gamma Gain is the only grading node whose wheel parameters are Cartesian
// (lift_r/g/b + lift_master) rather than polar. Mapping a 2-D wheel position to
// three channel values needs a modelling choice, and that choice is derived and
// signed in docs/lgg-wheel-derivation.md (v2, signed 2026-08-26). This file is
// that document in code; do not change a constant here without amending it.
//
// THE MODEL
//   The wheel is chroma-only: it must not move luminance at all. The master bar
//   is the only luminance control for its group. For a hue theta:
//       d(theta) = normalise( c(theta) - (w . c(theta)) * (1,1,1) )
//   with w = Rec.709. That makes w . d = 0 exactly, so the wheel is
//   luma-neutral by construction.
//
//   Rec.709 rather than equal weights is NOT a taste call. Measured on a
//   mid-grey patch, a rim push toward green under equal weights raises
//   luminance by +78%; under Rec.709 every hue gives 0.00%. See derivation 3.1.
//
//   ADDITIVE groups (lift, offset; neutral 0):        p_c = A * r * d_c
//   MULTIPLICATIVE groups (gamma, gain; neutral 1):   p_c = exp(A * r * d_c)
//   The multiplicative form is required because the backend folds those masters
//   in with * , not + (nodes/lift_gamma_gain.py). Working additively there would
//   make the same drag asymmetric between lightening and darkening.
//
//   Multiplicative groups are luma-null only to FIRST ORDER: sum w_c*d_c = 0
//   does not give sum w_c*exp(A*d_c) = 1 (Jensen). At the signed A = 0.35 the
//   residual is 1.69%. Renormalising would zero it but costs 3.66 deg of hue
//   fidelity, so it is deliberately NOT done -- signed decision, derivation 4.1.

import { clamp } from "./darkroom_canvas_widget.js";

export const LUMA_W = [0.2126, 0.7152, 0.0722];

// Fully saturated RGB at hue h -- the same hexagonal path hsv2rgb walks.
function pureHue(h) {
  h = ((h % 360) + 360) % 360;
  const x = 1 - Math.abs(((h / 60) % 2) - 1);
  if (h < 60) return [1, x, 0];
  if (h < 120) return [x, 1, 0];
  if (h < 180) return [0, 1, x];
  if (h < 240) return [0, x, 1];
  if (h < 300) return [x, 0, 1];
  return [1, 0, x];
}

// d(theta): unit-length, luma-null. Normalising matters -- unnormalised the
// magnitude ripples 32% around the circle, so the same drag distance would
// push visibly harder in some directions than others.
export function lumaNullBasis(h) {
  const c = pureHue(h);
  const y = LUMA_W[0] * c[0] + LUMA_W[1] * c[1] + LUMA_W[2] * c[2];
  const d = [c[0] - y, c[1] - y, c[2] - y];
  const n = Math.hypot(d[0], d[1], d[2]);
  return n > 1e-12 ? [d[0] / n, d[1] / n, d[2] / n] : [0, 0, 0];
}

// --- the inverse: recover theta from a luma-null vector ---------------------
//
// A bare atan2 in the plane is WRONG. Because c(theta) traces a hexagon, the
// plane angle phi advances between 0.65 and 1.41 degrees per degree of hue --
// up to 41% local rate error. phi(theta) IS strictly monotonic (verified over
// 1440 samples), so the correct inverse is a monotonic LUT on phi, built once
// and binary-searched. Derivation 7.2.

const LUT_N = 1440;                 // 0.25 deg resolution
let E1 = null, E2 = null, PHI = null, PHI0 = 0;

function buildLut() {
  E1 = lumaNullBasis(0);
  const t = lumaNullBasis(120);
  const dot = t[0] * E1[0] + t[1] * E1[1] + t[2] * E1[2];
  let e2 = [t[0] - dot * E1[0], t[1] - dot * E1[1], t[2] - dot * E1[2]];
  const n = Math.hypot(e2[0], e2[1], e2[2]);
  E2 = [e2[0] / n, e2[1] / n, e2[2] / n];

  // Store phi UNWRAPPED (strictly increasing, spanning ~360 in total) rather
  // than folded into [0,360). Folding is subtly wrong at i=0: Gram-Schmidt
  // leaves b at about -1e-17, so atan2 returns a tiny NEGATIVE angle and a
  // naive `if (p < 0) p += 360` turns phi(0) into 360.00 instead of 0, which
  // destroys monotonicity at the first entry and sends every later query into
  // the wrap branch. Unwrapping sidesteps the representation entirely.
  PHI = new Float64Array(LUT_N);
  let acc = 0, prev = null;
  for (let i = 0; i < LUT_N; i++) {
    const d = lumaNullBasis((i * 360) / LUT_N);
    const a = d[0] * E1[0] + d[1] * E1[1] + d[2] * E1[2];
    const b = d[0] * E2[0] + d[1] * E2[1] + d[2] * E2[2];
    let p = (Math.atan2(b, a) * 180) / Math.PI;   // (-180, 180]
    if (prev !== null) while (p + acc < prev) acc += 360;
    PHI[i] = p + acc;
    prev = PHI[i];
  }
  PHI0 = PHI[0];
}

// theta for a luma-null vector v (not required to be unit length).
export function hueFromVector(v) {
  if (!PHI) buildLut();
  const a = v[0] * E1[0] + v[1] * E1[1] + v[2] * E1[2];
  const b = v[0] * E2[0] + v[1] * E2[1] + v[2] * E2[2];
  if (Math.abs(a) < 1e-15 && Math.abs(b) < 1e-15) return 0;

  // Bring the query into the same unwrapped window the table lives in.
  let phi = (Math.atan2(b, a) * 180) / Math.PI;
  while (phi < PHI0) phi += 360;
  while (phi >= PHI0 + 360) phi -= 360;

  const last = LUT_N - 1;
  if (phi >= PHI[last]) {
    // the wrap segment, between the final sample and the first + 360
    const span = PHI0 + 360 - PHI[last];
    const f = span > 1e-12 ? (phi - PHI[last]) / span : 0;
    return (((last + f) * 360) / LUT_N) % 360;
  }
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (PHI[mid] <= phi) lo = mid; else hi = mid;
  }
  const span = PHI[hi] - PHI[lo];
  const f = span > 1e-12 ? (phi - PHI[lo]) / span : 0;
  return (((lo + f) * 360) / LUT_N) % 360;
}

// --- forward / inverse for one group ----------------------------------------

// Signed amplitudes (derivation 6). These set what "rim" means and are the
// main taste call in the whole design -- cheap to change, nothing else depends
// on their exact values.
export const GROUP_AMP = { lift: 0.30, gamma: 0.35, gain: 0.35, offset: 0.15 };

// Write precision. NOT the slider's `step`: quantising to step costs up to 8
// degrees of hue on the round trip, worst exactly at the small radii a
// colourist works in. 4 dp costs 0.10 deg. ComfyUI validates only min/max for
// FLOAT, never step -- verified live with an out-of-range positive control.
// Derivation 7.1.
const WRITE_DP = 4;
export function quantise(v) {
  const m = Math.pow(10, WRITE_DP);
  return Math.round(v * m) / m;
}

// (hue, radius) -> three channel values.
export function wheelToChannels(hue, radius, amp, multiplicative) {
  const d = lumaNullBasis(hue);
  const k = amp * clamp(radius, 0, 1);
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = multiplicative ? Math.exp(k * d[i]) : k * d[i];
  }
  return out;
}

// three channel values -> (hue, radius). Strips any common component first, so
// hand-typed values that are not luma-null still read back sensibly: the wheel
// shows the chroma part, the master bar shows the common part.
export function channelsToWheel(vals, amp, multiplicative) {
  let v;
  if (multiplicative) {
    v = vals.map((x) => Math.log(Math.max(x, 1e-6)));
  } else {
    v = vals.slice();
  }
  const y = LUMA_W[0] * v[0] + LUMA_W[1] * v[1] + LUMA_W[2] * v[2];
  v = [v[0] - y, v[1] - y, v[2] - y];
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-9 || amp <= 0) return { hue: 0, radius: 0 };
  return { hue: hueFromVector(v), radius: clamp(n / amp, 0, 1) };
}

// --- master bar --------------------------------------------------------------
//
// Additive groups: the bar value IS the master value, linear, centre 0.
// Multiplicative groups: a LINEAR bar would put neutral 1.0 at 23% of gamma's
// [0.1,4] slider -- a bar whose middle is not "no change" is unusable. Use
// master = exp(t * ln K) for t in [-1,1]; with K = 4 the bar spans [0.25, 4],
// exactly symmetric about 1.0 and inside both sliders' ranges. Derivation 5.
export const MASTER_LOG_K = 4.0;

export function barToMaster(t, multiplicative, lo, hi) {
  if (!multiplicative) return clamp(t, lo, hi);
  return clamp(Math.exp(clamp(t, -1, 1) * Math.log(MASTER_LOG_K)), lo, hi);
}

export function masterToBar(v, multiplicative) {
  if (!multiplicative) return v;
  return clamp(Math.log(Math.max(v, 1e-6)) / Math.log(MASTER_LOG_K), -1, 1);
}
