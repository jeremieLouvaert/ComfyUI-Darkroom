// ComfyUI-Darkroom -- freeform tone-curve editor (Capture One style).
//
// Click empty space to add a point, drag to move it, double-click a point to
// remove it (the two end points stay; drag them inward to set the black/white
// point). Shift-drag moves at 1/4 speed.
//
// RULE 1 of darkroom_canvas_widget.js holds: the canvas is a view. The curve's
// only state is the node's `curve_points` STRING widget ("x,y;x,y;..."), and the
// Contrast / Shadows / Midtones / Highlights sliders are ordinary FLOAT widgets.
// The effective curve (points + sliders) is drawn as a second, thinner line so
// the sliders are visible on the graph. Maths shared with the backend lives in
// darkroom_tone_curve_math.js.
//
// Reusable: a node opts in with registerFreeformCurve(nodeTypeName, spec).

import { app } from "../../scripts/app.js";
import { attachCanvasController, findWidget, readVal, writeVal, clamp } from "./darkroom_canvas_widget.js";
import { parsePoints, formatPoints, compose, migrateFilmStockValues } from "./darkroom_tone_curve_math.js";

const SIDE_PAD = 10, TOP_PAD = 8, BOTTOM_PAD = 10, READOUT_H = 16;
const HIT_R = 9, HANDLE_R = 5, DOUBLE_MS = 350, MIN_GAP = 0.01, FINE = 0.25;

function plotSize(width) {
  const w = Math.max(160, width - SIDE_PAD * 2);
  return Math.min(w, 300);
}

export function createFreeformCurveController(node, spec) {
  const sliders = spec.sliders || [];
  return {
    geo: null,
    drag: -1,
    pts: null,          // working copy during a drag
    lastDown: { t: 0, i: -1 },
    lastPos: null,

    dragging() { return this.drag !== -1; },
    syncedWidgets() { return [spec.pointsWidget, ...sliders]; },
    computeSize(width) { return [width, TOP_PAD + plotSize(width) + READOUT_H + BOTTOM_PAD]; },

    points(node) {
      if (this.drag !== -1 && this.pts) return this.pts;
      const w = findWidget(node, spec.pointsWidget);
      return parsePoints(w ? w.value : "0,0;1,1");
    },

    write(node, commit) {
      writeVal(node, spec.pointsWidget, formatPoints(this.pts), commit, spec.tag);
    },

    draw(ctx, node, width, y) {
      try {
        const S = plotSize(width);
        const x0 = Math.round((width - S) / 2), y0 = y + TOP_PAD;
        this.geo = { x0, y0, S };
        const X = (v) => x0 + v * S, Y = (v) => y0 + (1 - v) * S;
        const pts = this.points(node);

        ctx.save();
        ctx.fillStyle = "#161616";
        ctx.fillRect(x0, y0, S, S);
        ctx.strokeStyle = "#2a2a2a";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const f of [0.25, 0.5, 0.75]) {
          ctx.moveTo(X(f) + 0.5, y0); ctx.lineTo(X(f) + 0.5, y0 + S);
          ctx.moveTo(x0, Y(f) + 0.5); ctx.lineTo(x0 + S, Y(f) + 0.5);
        }
        ctx.stroke();
        ctx.strokeStyle = "#3a3a3a";                    // identity diagonal
        ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke();

        const N = 128;
        const xs = Array.from({ length: N }, (_, i) => i / (N - 1));
        const sv = sliders.map((n) => readVal(node, n, 0));
        if (sv.some((v) => Math.abs(v) > 1e-9)) {     // effective curve incl. sliders
          const eff = compose(pts, ...sv, N);
          ctx.strokeStyle = "rgba(255, 196, 92, 0.85)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          eff.forEach((v, i) => (i ? ctx.lineTo(X(xs[i]), Y(v)) : ctx.moveTo(X(xs[i]), Y(v))));
          ctx.stroke();
        }
        // the points' curve exactly as the backend applies it (compose() includes
        // the non-decreasing guard, so a point dragged below its neighbour draws
        // as the plateau it will really produce)
        const ys = compose(pts, 0, 0, 0, 0, N);
        ctx.strokeStyle = "#e8e8e8";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ys.forEach((v, i) => {
          const yy = Y(clamp(v, 0, 1));
          i ? ctx.lineTo(X(xs[i]), yy) : ctx.moveTo(X(xs[i]), yy);
        });
        ctx.stroke();

        this.geo.handles = pts.map(([px, py]) => ({ x: X(px), y: Y(py) }));
        pts.forEach(([px, py], i) => {
          ctx.beginPath();
          ctx.arc(X(px), Y(py), HANDLE_R, 0, Math.PI * 2);
          ctx.fillStyle = i === this.drag ? "#ffc45c" : "#ffffff";
          ctx.fill();
          ctx.strokeStyle = "#111";
          ctx.stroke();
        });

        ctx.fillStyle = "#9a9a9a";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        const msg = this.drag !== -1
          ? `Input ${(pts[this.drag][0] * 255).toFixed(0)}   Output ${(pts[this.drag][1] * 255).toFixed(0)}`
          : "click: add point   drag: move   double-click: remove";
        ctx.fillText(msg, width / 2, y0 + S + 12);
        ctx.restore();
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " freeform curve draw() failed:", err);
      }
    },

    mouse(event, pos, node) {
      try {
        const g = this.geo;
        if (!pos || !g) return false;
        const [px, py] = pos;
        const t = event.type || "";
        const toV = (vx, vy) => [clamp((vx - g.x0) / g.S, 0, 1), clamp(1 - (vy - g.y0) / g.S, 0, 1)];

        if (t.endsWith("down")) {
          const inside = px >= g.x0 - HIT_R && px <= g.x0 + g.S + HIT_R && py >= g.y0 - HIT_R && py <= g.y0 + g.S + HIT_R;
          if (!inside) return false;
          this.pts = this.points(node).map((p) => p.slice());
          let hit = -1, best = Infinity;
          (g.handles || []).forEach((h, i) => {
            const d = Math.hypot(h.x - px, h.y - py);
            if (d <= HIT_R && d < best) { best = d; hit = i; }
          });
          const now = Date.now();
          if (hit !== -1 && this.lastDown.i === hit && now - this.lastDown.t < DOUBLE_MS) {
            if (hit !== 0 && hit !== this.pts.length - 1) {   // double-click removes
              this.pts.splice(hit, 1);
              this.write(node, true);
            }
            this.lastDown = { t: 0, i: -1 };
            this.pts = null;
            node.setDirtyCanvas(true, true);
            return true;
          }
          if (hit === -1) {                                    // click adds a point
            const [vx, vy] = toV(px, py);
            let i = this.pts.findIndex((p) => p[0] > vx);
            if (i <= 0) { this.pts = null; return true; }      // outside the end points
            if (vx - this.pts[i - 1][0] < MIN_GAP || this.pts[i][0] - vx < MIN_GAP) { this.pts = null; return true; }
            this.pts.splice(i, 0, [vx, vy]);
            hit = i;
          }
          this.lastDown = { t: now, i: hit };
          this.drag = hit;
          this.lastPos = [px, py];
          this.write(node, false);
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (t.endsWith("move") && this.drag !== -1) {
          const i = this.drag, n = this.pts.length;
          let [vx, vy] = toV(px, py);
          if (event.shiftKey && this.lastPos) {                // fine adjust
            const [lx, ly] = toV(...this.lastPos);
            vx = this.pts[i][0] + (vx - lx) * FINE;
            vy = this.pts[i][1] + (vy - ly) * FINE;
          }
          let lo = i === 0 ? 0 : this.pts[i - 1][0] + MIN_GAP;
          let hi = i === n - 1 ? 1 : this.pts[i + 1][0] - MIN_GAP;
          if (lo > hi) lo = hi = this.pts[i][0];          // neighbours closer than 2 gaps: lock x
          this.pts[i] = [clamp(vx, lo, hi), clamp(vy, 0, 1)];
          this.lastPos = [px, py];
          this.write(node, false);
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (t.endsWith("up") && this.drag !== -1) {
          this.write(node, true);
          this.drag = -1;
          this.pts = null;
          this.lastPos = null;
          node.setDirtyCanvas(true, true);
          return true;
        }
        return false;
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " freeform curve mouse() failed:", err);
        this.drag = -1;
        this.pts = null;
        return false;
      }
    },
  };
}

export function registerFreeformCurve(nodeTypeName, spec) {
  app.registerExtension({
    name: "AKURATE." + nodeTypeName + ".FreeformCurve",
    async beforeRegisterNodeDef(nodeType, nodeData) {
      if (nodeData.name !== nodeTypeName) return;
      const origCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = origCreated ? origCreated.apply(this, arguments) : undefined;
        const attach = () => attachCanvasController(this, createFreeformCurveController(this, spec),
          { tag: spec.tag, minWidth: spec.minWidth || 340, requireWidget: spec.pointsWidget });
        if (!attach()) setTimeout(attach, 0);
        return r;
      };
      if (spec.migrate) {
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
          const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
          const fixed = spec.migrate(info && info.widgets_values);
          if (fixed) {
            const live = (this.widgets || []).filter((w) => w.serialize !== false);
            live.forEach((w, i) => { if (i < fixed.length) w.value = fixed[i]; });
            console.info("[Darkroom] " + spec.tag + ": upgraded a pre-1.29 workflow (the toe/shoulder/gamma "
                         + "overrides are now the tone curve; values reset to neutral)");
          }
          return r;
        };
      }
    },
  });
}

registerFreeformCurve("DarkroomFilmStockColor", {
  tag: "FilmStockCurve",
  pointsWidget: "curve_points",
  sliders: ["contrast", "shadows", "midtones", "highlights"],
  minWidth: 340,
  migrate: migrateFilmStockValues,
});
