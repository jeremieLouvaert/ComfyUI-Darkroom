// ComfyUI-Darkroom -- shared curve-strip controller.
//
// A curve editor over N control points at FIXED x positions, each bound to one
// of the node's existing float widgets and draggable only in y. Used by Tone
// Curve, Hue vs Hue, Hue vs Sat, Sat vs Sat and Lum vs Sat.
//
// See darkroom_canvas_widget.js for the attach/serialisation rules; this file
// only owns geometry and painting.
//
// The x positions are NOT evenly spaced in general -- they come from the
// backend's own band constants (HUE_CENTERS in hue_vs_hue.py / hue_vs_sat.py,
// SAT_BANDS in sat_vs_sat.py, LUM_ZONES in lum_vs_sat.py). Spacing handles
// evenly would misalign every one of them against the axis gradient beneath.

import {
  clamp, findWidget, readVal, writeVal, hsv2rgb,
} from "./darkroom_canvas_widget.js";

// --- layout -----------------------------------------------------------------

const SIDE_PAD = 12;
const TOP_PAD = 8;
const PLOT_H = 116;
const AXIS_GAP = 4;
const AXIS_H = 12;
const LABEL_GAP = 5;
const LABEL_H = 10;
const CAPTION_H = 13;
const BOTTOM_PAD = 8;

const HANDLE_R = 4.5;
// Handles at x=0 and x=1 (Tone Curve's shadows/highlights, Hue vs Hue's red)
// would be half-clipped by the plot edge. Inset the handle TRACK by a handle
// radius, and inset the axis gradient by exactly the same amount so gradient
// position and handle position stay in lockstep.
const TRACK_PAD = 6;
const HIT_R = 9;
// Release this close to the centreline -> exactly 0. Deliberately small: the
// snap must make "return to neutral" reliable without swallowing small
// deliberate values. At 3px over a 116px plot it covers ~+/-5 units on a +/-100
// axis, ~+/-2.6 on +/-50; anything finer is still reachable via the slider.
const ZERO_SNAP_PX = 3;
const FINE_SCALE = 0.25;

// --- axis gradients, cached per (kind,width) --------------------------------

const axisCache = new Map();

function getAxisStrip(kind, w, h) {
  const key = kind + ":" + (w | 0) + ":" + (h | 0);
  const hit = axisCache.get(key);
  if (hit) return hit;

  const cv = document.createElement("canvas");
  cv.width = Math.max(1, w | 0);
  cv.height = Math.max(1, h | 0);
  const c2 = cv.getContext("2d");
  const id = c2.createImageData(cv.width, cv.height);
  const d = id.data;

  for (let x = 0; x < cv.width; x++) {
    const t = cv.width === 1 ? 0 : x / (cv.width - 1);
    let rgb;
    if (kind === "hue") {
      rgb = hsv2rgb(t * 360, 1, 1);
    } else if (kind === "chroma") {
      // Increasing chroma left to right. The hue sweeps so that no single hue
      // is privileged at the saturated end; this strip is a chroma indicator,
      // NOT a hue axis (the node's x axis is pixel saturation).
      rgb = hsv2rgb(t * 360, t, 0.85);
    } else {
      rgb = [t * 255, t * 255, t * 255];   // "luma"
    }
    for (let y = 0; y < cv.height; y++) {
      const o = (y * cv.width + x) * 4;
      d[o] = rgb[0]; d[o + 1] = rgb[1]; d[o + 2] = rgb[2]; d[o + 3] = 255;
    }
  }
  c2.putImageData(id, 0, 0);
  axisCache.set(key, cv);
  return cv;
}

// --- monotone cubic through the control points ------------------------------
// Fritsch-Carlson: a smooth curve that cannot overshoot between points, so the
// drawn curve never implies a value the node will not produce.

function monotoneSlopes(xs, ys) {
  const n = xs.length;
  if (n < 2) return [0];
  const dx = [], dy = [], m = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(xs[i + 1] - xs[i]);
    dy.push(ys[i + 1] - ys[i]);
    m.push(dx[i] === 0 ? 0 : dy[i] / dx[i]);
  }
  const t = new Array(n);
  t[0] = m[0];
  t[n - 1] = m[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (m[i - 1] * m[i] <= 0) t[i] = 0;
    else {
      const w1 = 2 * dx[i] + dx[i - 1];
      const w2 = dx[i] + 2 * dx[i - 1];
      t[i] = (w1 + w2) / (w1 / m[i - 1] + w2 / m[i]);
    }
  }
  return t;
}

function evalMonotone(x, xs, ys, t) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (i < n - 2 && x > xs[i + 1]) i++;
  const h = xs[i + 1] - xs[i];
  if (h === 0) return ys[i];
  const s = (x - xs[i]) / h;
  const s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * ys[i] +
         (s3 - 2 * s2 + s) * h * t[i] +
         (-2 * s3 + 3 * s2) * ys[i + 1] +
         (s3 - s2) * h * t[i + 1];
}

// --- spec -------------------------------------------------------------------
//
// {
//   tag: "ToneCurve",
//   axis: "luma" | "hue" | "chroma",
//   range: 50,                      // symmetric: -range .. +range
//   unit: "",                       // suffix in the readout, e.g. "°"
//   minWidth: 400,
//   points: [{ x: 0.0, widget: "shadows", label: "Sh" }, ...],   // x in 0..1
//   preset: { widget, custom, caption },
// }

export function curveHeight(spec) {
  return TOP_PAD + PLOT_H + AXIS_GAP + AXIS_H + LABEL_GAP + LABEL_H +
         (spec.preset ? CAPTION_H : 0) + BOTTOM_PAD;
}

export function createCurveController(node, spec) {
  const R = spec.range;

  return {
    spec,
    geo: null,
    drag: -1,
    _v: 0,
    _lastPy: null,

    dragging() { return this.drag !== -1; },
    syncedWidgets() { return spec.points.map((p) => p.widget); },

    computeSize(width) {
      const w = width || (node.size && node.size[0]) || spec.minWidth || 400;
      return [w, curveHeight(spec)];
    },

    values(node) {
      return spec.points.map((p) => clamp(readVal(node, p.widget, 0), -R, R));
    },

    draw(ctx, node, widgetWidth, y, _h) {
      try {
        const x0 = SIDE_PAD;
        const w = Math.max(1, widgetWidth - SIDE_PAD * 2);
        const plotY = y + TOP_PAD;
        const midY = plotY + PLOT_H / 2;
        const axisY = plotY + PLOT_H + AXIS_GAP;

        const tx0 = x0 + TRACK_PAD;
        const tw = Math.max(1, w - TRACK_PAD * 2);
        this.geo = { x0, w, tx0, tw, plotY, midY, axisY };

        ctx.save();

        // plot bed
        ctx.fillStyle = "#161616";
        ctx.fillRect(x0, plotY, w, PLOT_H);

        // horizontal guides at +/- half range
        ctx.strokeStyle = "#262626";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const f of [0.25, 0.75]) {
          const gy = plotY + PLOT_H * f;
          ctx.moveTo(x0, gy + 0.5); ctx.lineTo(x0 + w, gy + 0.5);
        }
        ctx.stroke();

        // vertical guide at each control point
        ctx.strokeStyle = "#232323";
        ctx.beginPath();
        for (const p of spec.points) {
          const px = tx0 + p.x * tw;
          ctx.moveTo(px + 0.5, plotY); ctx.lineTo(px + 0.5, plotY + PLOT_H);
        }
        ctx.stroke();

        // the zero line -- brighter, it is the "no change" datum
        ctx.strokeStyle = "#4a4a4a";
        ctx.beginPath();
        ctx.moveTo(x0, midY + 0.5); ctx.lineTo(x0 + w, midY + 0.5);
        ctx.stroke();

        // --- the curve ---
        const vals = this.values(node);
        const xs = spec.points.map((p) => p.x);
        const toY = (v) => midY - (v / R) * (PLOT_H / 2);
        const slopes = monotoneSlopes(xs, vals);

        const STEP = 2;
        ctx.beginPath();
        for (let px = 0; px <= tw; px += STEP) {
          const t = px / tw;
          const py = toY(evalMonotone(t, xs, vals, slopes));
          if (px === 0) ctx.moveTo(tx0 + px, py);
          else ctx.lineTo(tx0 + px, py);
        }
        ctx.strokeStyle = "#7cc4f0";
        ctx.lineWidth = 1.6;
        ctx.stroke();

        // fill between the curve and the zero line
        ctx.lineTo(tx0 + tw, midY);
        ctx.lineTo(tx0, midY);
        ctx.closePath();
        ctx.fillStyle = "rgba(124,196,240,0.13)";
        ctx.fill();

        // --- handles ---
        const handles = [];
        for (let i = 0; i < spec.points.length; i++) {
          const p = spec.points[i];
          const hx = tx0 + p.x * tw;
          const hy = toY(vals[i]);
          handles.push({ x: hx, y: hy });

          const live = Math.abs(vals[i]) >= 0.5;
          ctx.beginPath();
          ctx.arc(hx, hy, HANDLE_R, 0, Math.PI * 2);
          ctx.fillStyle = live ? "#ffffff" : "#9a9a9a";
          ctx.fill();
          ctx.strokeStyle = "rgba(0,0,0,0.85)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        this.geo.handles = handles;

        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, plotY + 0.5, w - 1, PLOT_H - 1);

        // --- axis gradient strip ---
        ctx.drawImage(getAxisStrip(spec.axis, tw, AXIS_H), tx0, axisY, tw, AXIS_H);
        ctx.strokeStyle = "#333";
        ctx.strokeRect(tx0 + 0.5, axisY + 0.5, tw - 1, AXIS_H - 1);

        // --- labels, or the live readout for the handle being dragged ---
        const labelY = axisY + AXIS_H + LABEL_GAP + LABEL_H / 2;
        ctx.textBaseline = "middle";
        ctx.font = "9px sans-serif";
        if (this.drag !== -1) {
          const p = spec.points[this.drag];
          ctx.textAlign = "center";
          ctx.fillStyle = "#e8e8e8";
          const shown = Math.round(vals[this.drag]);
          ctx.fillText(`${p.label}  ${shown > 0 ? "+" : ""}${shown}${spec.unit || ""}`,
                       x0 + w / 2, labelY);
        } else {
          ctx.textAlign = "center";
          for (let i = 0; i < spec.points.length; i++) {
            const p = spec.points[i];
            ctx.fillStyle = Math.abs(vals[i]) >= 0.5 ? "#c8c8c8" : "#6c6c6c";
            ctx.fillText(p.label, tx0 + p.x * tw, labelY);
          }
        }

        // --- preset honesty caption ---
        if (spec.preset) {
          const pw = findWidget(node, spec.preset.widget);
          const cur = pw ? String(pw.value) : spec.preset.custom;
          if (cur && cur !== spec.preset.custom) {
            ctx.textAlign = "left";
            ctx.fillStyle = "#c9a227";
            ctx.fillText(spec.preset.caption, x0,
                         y + curveHeight(spec) - BOTTOM_PAD - CAPTION_H / 2);
          }
        }

        ctx.restore();
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " curve draw() failed:", err);
        try {
          ctx.save();
          ctx.fillStyle = "rgba(60,20,20,0.85)";
          ctx.fillRect(SIDE_PAD, y + TOP_PAD, Math.max(1, widgetWidth - SIDE_PAD * 2), 40);
          ctx.fillStyle = "#f0a0a0";
          ctx.font = "10px monospace";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText("[Darkroom] " + spec.tag + " error -- see console", SIDE_PAD + 4, y + TOP_PAD + 20);
          ctx.restore();
        } catch (_e) { /* nothing more we can safely do */ }
      }
    },

    flush(node, commit) {
      if (this.drag === -1) return;
      const p = spec.points[this.drag];
      let v = this._v;
      const g = this.geo;
      if (g) {
        const zeroY = g.midY;
        const vy = g.midY - (clamp(v, -R, R) / R) * (PLOT_H / 2);
        if (Math.abs(vy - zeroY) <= ZERO_SNAP_PX) v = 0;   // snap to "no change"
      }
      writeVal(node, p.widget, clamp(Math.round(v), -R, R), commit, spec.tag);
    },

    mouse(event, pos, node) {
      try {
        const g = this.geo;
        if (!pos || !g || !g.handles) return false;
        const px = pos[0], py = pos[1];
        const t = event.type || "";

        if (t.endsWith("down")) {
          // nearest handle within the hit radius
          let hit = -1, best = Infinity;
          for (let i = 0; i < g.handles.length; i++) {
            const h = g.handles[i];
            const d = Math.hypot(h.x - px, h.y - py);
            if (d <= HIT_R && d < best) { best = d; hit = i; }
          }
          if (hit === -1) {
            // a click on a control point's column, anywhere in the plot, grabs
            // that handle -- the handle itself is a small target
            if (px >= g.x0 && px <= g.x0 + g.w && py >= g.plotY && py <= g.plotY + PLOT_H) {
              let nearest = 0, nd = Infinity;
              for (let i = 0; i < spec.points.length; i++) {
                const d = Math.abs(g.tx0 + spec.points[i].x * g.tw - px);
                if (d < nd) { nd = d; nearest = i; }
              }
              if (nd <= g.tw / (spec.points.length * 2)) hit = nearest;
            }
          }
          if (hit === -1) return false;

          this.drag = hit;
          this._v = ((g.midY - py) / (PLOT_H / 2)) * R;
          this._lastPy = py;
          this.flush(node, false);
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (t.endsWith("move") && this.drag !== -1) {
          if (event.shiftKey && this._lastPy != null) {
            this._v += ((this._lastPy - py) / (PLOT_H / 2)) * R * FINE_SCALE;
          } else {
            this._v = ((g.midY - py) / (PLOT_H / 2)) * R;
          }
          this._v = clamp(this._v, -R, R);
          this._lastPy = py;
          this.flush(node, false);
          node.setDirtyCanvas(true, true);
          return true;
        }

        if ((t.endsWith("up") || t === "click") && this.drag !== -1) {
          this.flush(node, true);
          this.drag = -1;
          this._lastPy = null;
          node.setDirtyCanvas(true, true);
          return true;
        }

        return false;
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " curve mouse() failed:", err);
        this.drag = -1;
        this._lastPy = null;
        return false;
      }
    },
  };
}
