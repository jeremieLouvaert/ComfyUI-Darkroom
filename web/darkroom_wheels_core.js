// ComfyUI-Darkroom -- shared colour-wheel controller.
//
// One DaVinci-style wheel row (N hue/saturation discs, each with an optional
// bar beneath it) driven entirely by a node's EXISTING float widgets. Used by
// darkroom_log_wheels.js and darkroom_three_way.js; any future polar grading
// node should register here rather than copy this.
//
// ARCHITECTURE, and the two rules that must not be broken:
//
// 1. The wheel is a VIEW. The float widgets are the only state. The controller
//    reads them every draw and writes back on edit (`.value` always, so the
//    slider readout tracks live; `.callback` once on release, since a bare
//    `value =` set does not fire reactivity). No INPUT_TYPES change, so stored
//    workflows and API-format prompts are untouched, and deleting the .js
//    leaves a working slider node.
//
// 2. The controller is NOT a LiteGraph widget and must never enter
//    node.widgets. LiteGraph saves widgets_values BY INDEX skipping
//    serialize===false widgets, but loads it SEQUENTIALLY with the same skip:
//      save: for (const [i,w] of widgets.entries()) { if (w.serialize===false) continue; vals[i]=w.value }
//      load: let t=0; for (const w of widgets) if (w.serialize!==false) w.value=vals[t++]
//    Those agree only when the non-serialised widget is LAST, so a widget
//    anywhere else silently shifts every later value by one slot on load.
//    (`options.serialize` is not consulted by configure() at all.) Measured on
//    frontend 1.48.7; it is what the FieldGradient v0.6.1 fix was about.
//    Space is reserved with `widgets_start_y`, which only works at the TOP of
//    the widget stack -- a canvas cannot be injected mid-stack.
//
// Mapping is a stated convention matching the nodes' own tooltips:
//   hue 0 (red) at 3 o'clock, increasing counter-clockwise, atan2(-dy, dx)
//   radius 0..1 -> saturation/intensity 0..satMax, both quantised to integers

import { app } from "../../scripts/app.js";

// --- layout -----------------------------------------------------------------

const SIDE_PAD = 10;
const TOP_PAD = 6;
const WHEEL_GAP = 12;
const MAX_D = 130;
const MIN_D = 56;
const BAR_GAP = 9;
const BAR_H = 10;
const LABEL_GAP = 6;
const LABEL_H = 11;
const CAPTION_H = 13;
const BOTTOM_PAD = 6;

// --- look -------------------------------------------------------------------

const RIM_START = 0.88;          // hue ring inner edge as a fraction of radius
const RIM_GAP = 0.045;           // dark separator just inside the ring
const INTERIOR_DIM = 0.30;       // interior sits this much darker than the rim
const INTERIOR_SAT_GAMMA = 1.7;  // >1 holds the centre neutral longer
const BODY_GREY = 22;            // neutral centre value the interior eases into
const DOT_R = 4.5;

// --- interaction ------------------------------------------------------------

const HIT_SLOP = 5;
const CENTER_SNAP_PX = 4;   // drop this close to centre -> value exactly 0
const BAR_SNAP_PX = 4;      // drop this close to the middle -> bar exactly 0
const FINE_SCALE = 0.25;    // shift-drag multiplier

// --- helpers ----------------------------------------------------------------

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function findWidget(node, name) {
  return node.widgets ? node.widgets.find((w) => w.name === name) : undefined;
}

export function readVal(node, name, dflt) {
  const w = findWidget(node, name);
  const v = w ? Number(w.value) : NaN;
  return Number.isFinite(v) ? v : dflt;
}

export function writeVal(node, name, v, commit, tag) {
  const w = findWidget(node, name);
  if (!w) return;
  w.value = v;
  if (commit && typeof w.callback === "function") {
    try {
      w.callback(v, app.canvas, node);
    } catch (e) {
      console.warn("[Darkroom] " + (tag || "wheels") + ": callback threw for " + name, e);
    }
  }
}

export function hsv2rgb(h, s, v) {
  const c = v * s;
  const hp = ((((h % 360) + 360) % 360)) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) { r = c; g = x; }
  else if (hp < 2) { r = x; g = c; }
  else if (hp < 3) { g = c; b = x; }
  else if (hp < 4) { g = x; b = c; }
  else if (hp < 5) { r = x; b = c; }
  else { r = c; b = x; }
  const m = v - c;
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

export function wheelDiameter(widgetWidth, n) {
  const count = n || 3;
  const avail = Math.max(1, widgetWidth - SIDE_PAD * 2 - WHEEL_GAP * (count - 1));
  return Math.max(MIN_D, Math.min(MAX_D, Math.floor(avail / count)));
}

// onDrawForeground draws from the node BODY origin, which is also where
// LiteGraph lays out the socket rows. Modern ComfyUI lists EVERY widget in
// node.inputs as well, so count only inputs without a `.widget` back-reference
// or a 14-widget node reserves 14 rows and opens a ~290px void.
export function topY(node) {
  const socketIns = (node.inputs || []).filter((i) => !i.widget).length;
  const rows = Math.max(socketIns, node.outputs ? node.outputs.length : 0);
  const slotH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_SLOT_HEIGHT) || 20;
  return rows * slotH + 6;
}

// --- the hue/saturation disc, pre-rendered once per pixel diameter ----------
// A per-pixel disc recomputed every frame tanks canvas FPS with several nodes
// open. Every wheel on every node shares this cache.

const discCache = new Map();

export function getDisc(d) {
  const key = d | 0;
  const hit = discCache.get(key);
  if (hit) return hit;

  const cv = document.createElement("canvas");
  cv.width = key;
  cv.height = key;
  const c2 = cv.getContext("2d");
  const id = c2.createImageData(key, key);
  const data = id.data;
  const R = key / 2;

  for (let py = 0; py < key; py++) {
    for (let px = 0; px < key; px++) {
      const dx = px + 0.5 - R;
      const dy = py + 0.5 - R;
      const rr = Math.hypot(dx, dy) / R;
      const o = (py * key + px) * 4;
      if (rr > 1) { data[o + 3] = 0; continue; }

      const hue = (Math.atan2(-dy, dx) * 180) / Math.PI;
      let rgb;
      if (rr >= RIM_START) {
        // thin fully-saturated hue ring at the rim
        rgb = hsv2rgb(hue, 1, 1);
      } else if (rr >= RIM_START - RIM_GAP) {
        // dark separator so the ring reads as a ring, not a gradient edge
        rgb = [14, 14, 14];
      } else {
        const t = rr / (RIM_START - RIM_GAP);
        rgb = hsv2rgb(hue, Math.pow(t, INTERIOR_SAT_GAMMA), 1);
        rgb = [rgb[0] * INTERIOR_DIM, rgb[1] * INTERIOR_DIM, rgb[2] * INTERIOR_DIM];
        const k = Math.min(1, t * 2.2);   // ease into the neutral body
        rgb = [
          BODY_GREY + (rgb[0] - BODY_GREY) * k,
          BODY_GREY + (rgb[1] - BODY_GREY) * k,
          BODY_GREY + (rgb[2] - BODY_GREY) * k,
        ];
      }

      data[o] = rgb[0];
      data[o + 1] = rgb[1];
      data[o + 2] = rgb[2];
      const edge = (1 - rr) * R;          // 1px antialiased outer edge
      data[o + 3] = edge >= 1 ? 255 : Math.max(0, Math.round(edge * 255));
    }
  }

  c2.putImageData(id, 0, 0);
  discCache.set(key, cv);
  return cv;
}

// --- spec -------------------------------------------------------------------
//
// {
//   tag: "LogWheels",
//   satMax: 100,
//   zones: [{ key, label, hue, sat, bar?, barMin?, barMax? }, ...],
//   preset: { widget: "preset", custom: "Custom (manual)", caption: "..." },
// }

export function widgetHeight(spec, widgetWidth) {
  const d = wheelDiameter(widgetWidth, spec.zones.length);
  const anyBar = spec.zones.some((z) => z.bar);
  return TOP_PAD + d + (anyBar ? BAR_GAP + BAR_H : 0) +
         LABEL_GAP + LABEL_H + (spec.preset ? CAPTION_H : 0) + BOTTOM_PAD;
}

export function createWheelController(node, spec) {
  const satMax = spec.satMax || 100;

  return {
    spec,

    // layout captured each draw() so mouse() hit-tests the same geometry
    lastDrawY: 0,
    lastDrawW: 0,
    geo: [],

    // interaction state
    drag: null,   // { kind: "disc" | "bar", idx }
    _nx: 0,       // live disc position in normalised -1..1 coords
    _ny: 0,
    _bar: 0,      // live bar value during a bar drag
    _hue: 0,      // held so a centre-snap does not lose the hue direction
    _lastPx: null,
    _lastPy: null,

    computeSize(width) {
      const w = width || (node.size && node.size[0]) || 420;
      return [w, widgetHeight(spec, w)];
    },

    draw(ctx, node, widgetWidth, y, _h) {
      try {
        this.lastDrawY = y;
        this.lastDrawW = widgetWidth;

        const zones = spec.zones;
        const d = wheelDiameter(widgetWidth, zones.length);
        const r = d / 2;
        const totalW = d * zones.length + WHEEL_GAP * (zones.length - 1);
        const x0 = Math.max(SIDE_PAD, (widgetWidth - totalW) / 2);
        const discImg = getDisc(d);
        const anyBar = zones.some((z) => z.bar);

        ctx.save();
        this.geo = [];

        for (let i = 0; i < zones.length; i++) {
          const zone = zones[i];
          const left = x0 + i * (d + WHEEL_GAP);
          const discY = y + TOP_PAD;
          const cx = left + r;
          const cy = discY + r;

          ctx.drawImage(discImg, left, discY, d, d);

          ctx.beginPath();
          ctx.arc(cx, cy, r - 0.5, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(0,0,0,0.7)";
          ctx.lineWidth = 1;
          ctx.stroke();

          // centre crosshair -- the "zone off" target
          ctx.strokeStyle = "rgba(255,255,255,0.32)";
          ctx.beginPath();
          ctx.moveTo(cx - 3.5, cy); ctx.lineTo(cx + 3.5, cy);
          ctx.moveTo(cx, cy - 3.5); ctx.lineTo(cx, cy + 3.5);
          ctx.stroke();

          const hue = readVal(node, zone.hue, 0);
          const sat = readVal(node, zone.sat, 0);

          const rad = clamp(sat / satMax, 0, 1) * r;
          const a = (hue * Math.PI) / 180;
          const dotX = cx + Math.cos(a) * rad;
          const dotY = cy - Math.sin(a) * rad;

          if (rad > 1) {
            ctx.strokeStyle = "rgba(255,255,255,0.55)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(dotX, dotY);
            ctx.stroke();
          }

          ctx.beginPath();
          ctx.arc(dotX, dotY, DOT_R, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
          ctx.strokeStyle = "rgba(0,0,0,0.85)";
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // --- optional bar ---
          const barX = left;
          const barY = discY + d + BAR_GAP;
          const barW = d;
          let barVal = 0;

          if (zone.bar) {
            const lo = zone.barMin != null ? zone.barMin : -100;
            const hi = zone.barMax != null ? zone.barMax : 100;
            barVal = readVal(node, zone.bar, 0);
            const midX = barX + barW * ((0 - lo) / (hi - lo));

            ctx.fillStyle = "#1b1b1b";
            ctx.fillRect(barX, barY, barW, BAR_H);

            const t = (clamp(barVal, lo, hi) - lo) / (hi - lo);
            const handleX = barX + t * barW;

            if (Math.abs(handleX - midX) > 0.5) {
              ctx.fillStyle = barVal >= 0 ? "rgba(124,196,240,0.75)" : "rgba(240,170,110,0.75)";
              ctx.fillRect(Math.min(midX, handleX), barY, Math.abs(handleX - midX), BAR_H);
            }

            ctx.strokeStyle = "#555";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(midX + 0.5, barY);
            ctx.lineTo(midX + 0.5, barY + BAR_H);
            ctx.stroke();

            ctx.strokeStyle = "#3a3a3a";
            ctx.strokeRect(barX + 0.5, barY + 0.5, barW - 1, BAR_H - 1);

            ctx.fillStyle = "#ffffff";
            ctx.fillRect(handleX - 1.5, barY - 1, 3, BAR_H + 2);
            ctx.strokeStyle = "rgba(0,0,0,0.85)";
            ctx.strokeRect(handleX - 1.5, barY - 1, 3, BAR_H + 2);
          }

          // --- label ---
          const labelY = (anyBar ? barY + BAR_H : discY + d) + LABEL_GAP + LABEL_H / 2;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.font = "9px sans-serif";
          const live = sat >= 0.5 || (zone.bar && Math.abs(barVal) >= 0.5);
          ctx.fillStyle = live ? "#c8c8c8" : "#6c6c6c";
          ctx.fillText(zone.label, cx, labelY);

          this.geo.push({ cx, cy, r, barX, barY, barW, hasBar: !!zone.bar });
        }

        // --- preset honesty caption -------------------------------------
        // These nodes ADD preset values on top of the manual ones, so with a
        // preset active the dots are not the applied grade. Say so.
        if (spec.preset) {
          const pw = findWidget(node, spec.preset.widget);
          const cur = pw ? String(pw.value) : spec.preset.custom;
          if (cur && cur !== spec.preset.custom) {
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.font = "9px sans-serif";
            ctx.fillStyle = "#c9a227";
            const capY = y + widgetHeight(spec, widgetWidth) - BOTTOM_PAD - CAPTION_H / 2;
            ctx.fillText(spec.preset.caption, SIDE_PAD, capY);
          }
        }

        ctx.restore();
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " draw() failed (rendering fallback):", err);
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
        } catch (_e2) {
          /* nothing more we can safely do */
        }
      }
    },

    // Commit the live drag state into the float widgets.
    flush(node, commit) {
      const drag = this.drag;
      if (!drag) return;
      const zone = spec.zones[drag.idx];

      if (drag.kind === "disc") {
        const mag = Math.hypot(this._nx, this._ny);
        const g = this.geo[drag.idx];
        const px = mag * (g ? g.r : 1);
        let sat, hue;
        if (px <= CENTER_SNAP_PX) {
          sat = 0;              // snap the zone fully off (below the >=0.5 backend gate)
          hue = this._hue;      // hold direction so leaving centre does not jump
        } else {
          sat = clamp(mag, 0, 1) * satMax;
          hue = (Math.atan2(-this._ny, this._nx) * 180) / Math.PI;
          this._hue = hue;
        }
        hue = ((Math.round(hue) % 360) + 360) % 360;
        writeVal(node, zone.hue, hue, commit, spec.tag);
        writeVal(node, zone.sat, clamp(Math.round(sat), 0, satMax), commit, spec.tag);
      } else {
        const lo = zone.barMin != null ? zone.barMin : -100;
        const hi = zone.barMax != null ? zone.barMax : 100;
        let v = this._bar;
        const g = this.geo[drag.idx];
        if (g) {
          const midX = g.barX + g.barW * ((0 - lo) / (hi - lo));
          const handleX = g.barX + ((clamp(v, lo, hi) - lo) / (hi - lo)) * g.barW;
          if (Math.abs(handleX - midX) <= BAR_SNAP_PX) v = 0;
        }
        writeVal(node, zone.bar, clamp(Math.round(v), lo, hi), commit, spec.tag);
      }
    },

    mouse(event, pos, node) {
      try {
        if (!pos || !this.geo.length) return false;
        const px = pos[0];
        const py = pos[1];

        const t = event.type || "";
        const isDown = t.endsWith("down");
        const isMove = t.endsWith("move");
        const isUp = t.endsWith("up") || t === "click";

        if (isDown) {
          for (let i = 0; i < this.geo.length; i++) {
            const g = this.geo[i];
            const zone = spec.zones[i];

            // bar first -- it sits below the disc, no overlap
            if (
              g.hasBar &&
              px >= g.barX - HIT_SLOP && px <= g.barX + g.barW + HIT_SLOP &&
              py >= g.barY - HIT_SLOP && py <= g.barY + BAR_H + HIT_SLOP
            ) {
              const lo = zone.barMin != null ? zone.barMin : -100;
              const hi = zone.barMax != null ? zone.barMax : 100;
              this.drag = { kind: "bar", idx: i };
              this._bar = lo + ((px - g.barX) / g.barW) * (hi - lo);
              this._lastPx = px;
              this._lastPy = py;
              this.flush(node, false);
              node.setDirtyCanvas(true, true);
              return true;
            }

            // disc
            if (Math.hypot(px - g.cx, py - g.cy) <= g.r + HIT_SLOP) {
              this.drag = { kind: "disc", idx: i };
              this._hue = readVal(node, zone.hue, 0);
              this._nx = clamp((px - g.cx) / g.r, -1, 1);
              this._ny = clamp((py - g.cy) / g.r, -1, 1);
              this._lastPx = px;
              this._lastPy = py;
              this.flush(node, false);
              node.setDirtyCanvas(true, true);
              return true;
            }
          }
          return false;
        }

        if (isMove && this.drag) {
          const g = this.geo[this.drag.idx];
          const zone = spec.zones[this.drag.idx];
          if (!g) return true;

          if (this.drag.kind === "disc") {
            let nx, ny;
            if (event.shiftKey && this._lastPx != null) {
              nx = this._nx + ((px - this._lastPx) / g.r) * FINE_SCALE;
              ny = this._ny + ((py - this._lastPy) / g.r) * FINE_SCALE;
            } else {
              nx = (px - g.cx) / g.r;
              ny = (py - g.cy) / g.r;
            }
            const mag = Math.hypot(nx, ny);
            if (mag > 1) { nx /= mag; ny /= mag; }   // clamp to the rim
            this._nx = nx;
            this._ny = ny;
          } else {
            const lo = zone.barMin != null ? zone.barMin : -100;
            const hi = zone.barMax != null ? zone.barMax : 100;
            if (event.shiftKey && this._lastPx != null) {
              this._bar += ((px - this._lastPx) / g.barW) * (hi - lo) * FINE_SCALE;
            } else {
              this._bar = lo + ((px - g.barX) / g.barW) * (hi - lo);
            }
            this._bar = clamp(this._bar, lo, hi);
          }

          this._lastPx = px;
          this._lastPy = py;
          this.flush(node, false);      // live readout, no callback storm
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (isUp && this.drag) {
          this.flush(node, true);       // the committed edit
          this.drag = null;
          this._lastPx = null;
          this._lastPy = null;
          node.setDirtyCanvas(true, true);
          return true;
        }

        return false;
      } catch (err) {
        console.error("[Darkroom] " + spec.tag + " mouse() failed:", err);
        this.drag = null;
        this._lastPx = null;
        this._lastPy = null;
        return false;
      }
    },
  };
}

// --- attach -----------------------------------------------------------------

export function attachWheels(node, spec) {
  if (node._darkroomWheelsAttached) return true;

  const anchor = findWidget(node, spec.zones[0].hue);
  if (!anchor) return false;   // widgets not built yet -- caller retries

  node._darkroomWheelsAttached = true;

  const ctrl = createWheelController(node, spec);
  node._darkroomWheels = ctrl;

  // NOT added to node.widgets -- see the header. Space comes from
  // widgets_start_y, so the wheels sit above the whole slider stack.
  const minW = spec.minWidth || 420;
  const reserve = () => {
    node.widgets_start_y = topY(node) + widgetHeight(spec, Math.max(node.size[0] || minW, minW));
  };
  reserve();

  const origDraw = node.onDrawForeground;
  node.onDrawForeground = function (ctx, canvas) {
    const r = origDraw ? origDraw.apply(this, arguments) : undefined;
    if (this.flags && this.flags.collapsed) return r;
    reserve();
    ctrl.draw(ctx, this, this.size[0], topY(this), 0);
    return r;
  };

  const origDown = node.onMouseDown;
  node.onMouseDown = function (e, pos, canvas) {
    if (ctrl.mouse({ type: "pointerdown", shiftKey: !!(e && e.shiftKey) }, pos, this)) {
      if (typeof this.captureInput === "function") this.captureInput(true);
      return true;
    }
    return origDown ? origDown.apply(this, arguments) : false;
  };

  const origMove = node.onMouseMove;
  node.onMouseMove = function (e, pos, canvas) {
    if (ctrl.drag) {
      ctrl.mouse({ type: "pointermove", shiftKey: !!(e && e.shiftKey) }, pos, this);
      return true;
    }
    return origMove ? origMove.apply(this, arguments) : undefined;
  };

  const origUp = node.onMouseUp;
  node.onMouseUp = function (e, pos, canvas) {
    if (ctrl.drag) {
      ctrl.mouse({ type: "pointerup", shiftKey: !!(e && e.shiftKey) }, pos, this);
      if (typeof this.captureInput === "function") this.captureInput(false);
      return true;
    }
    return origUp ? origUp.apply(this, arguments) : undefined;
  };

  // Reverse sync: editing a slider by hand must move the dot. draw() already
  // reads the float widgets every frame, so this only forces a repaint (while
  // preserving the original callback's return value).
  for (const zone of spec.zones) {
    for (const name of [zone.hue, zone.sat, zone.bar]) {
      if (!name) continue;
      const w = findWidget(node, name);
      if (!w || w._darkroomWheelHooked) continue;
      w._darkroomWheelHooked = true;
      const orig = w.callback;
      w.callback = function (value, ...rest) {
        const out = orig ? orig.apply(this, [value, ...rest]) : undefined;
        node.setDirtyCanvas(true, true);
        return out;
      };
    }
  }

  // Widen enough for the wheel row, without shrinking a saved size.
  try {
    const need = node.computeSize();
    node.setSize([
      Math.max(node.size[0] || 0, minW),
      Math.max(node.size[1] || 0, need[1]),
    ]);
  } catch (e) {
    console.warn("[Darkroom] " + spec.tag + ": resize failed", e);
  }

  node.setDirtyCanvas(true, true);
  return true;
}

// Register a node type against a spec. Handles the one-tick-late widget build.
export function registerWheelNode(nodeTypeName, spec) {
  app.registerExtension({
    name: "AKURATE.Darkroom" + spec.tag,
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
      if (nodeData.name !== nodeTypeName) return;
      const orig = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = orig ? orig.apply(this, arguments) : undefined;
        if (!attachWheels(this, spec)) setTimeout(() => attachWheels(this, spec), 0);
        return r;
      };
    },
  });
}
