// ComfyUI-Darkroom -- Log Wheels DaVinci-style colour wheel widget (SPIKE).
//
// The node's nine FLOAT widgets (<zone>_hue / <zone>_saturation / <zone>_density
// for zone in shadow|midtone|highlight) remain the ONLY state. This file adds a
// purely-visual custom widget above them: three DaVinci-style log wheels in a
// row, each a hue/saturation disc with a density bar beneath it. The widget
// carries `serialize: false` and owns no saved value -- on every edit it writes
// straight into the float widgets' `.value` and fires their `callback`, so
// ComfyUI's normal serialize/queue path picks the change up exactly as if the
// slider had been dragged. Delete this file and the node is unchanged.
//
// Mapping (stated convention, not a derived model -- matches the node's own
// tooltips: 0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta):
//   angle  hue 0 (red) at 3 o'clock, increasing counter-clockwise
//          screen-space hue = atan2(-dy, dx)
//   radius disc radius 0..1 maps linearly to saturation 0..100
//   both quantised to integers, matching the widgets' step: 1.0
//
// House pattern: WEB_DIRECTORY = "./web" + beforeRegisterNodeDef wrapping
// onNodeCreated for one node type by name. Canvas draw/mouse skeleton lifted
// from ComfyUI-Field/web/field_gradient_ramp.js.

import { app } from "../../scripts/app.js";

console.log("[Darkroom] Log Wheels canvas widget v1 (spike)");

const NODE_TYPE = "DarkroomLogWheels";

const ZONES = [
  { key: "shadow", label: "SHADOW" },
  { key: "midtone", label: "MIDTONE" },
  { key: "highlight", label: "HIGHLIGHT" },
];

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

const RIM_START = 0.88;      // hue ring inner edge as a fraction of radius
const RIM_GAP = 0.045;       // dark separator just inside the ring
const INTERIOR_DIM = 0.30;   // interior sits this much darker than the rim
const INTERIOR_SAT_GAMMA = 1.7;  // >1 holds the centre neutral longer
const BODY_GREY = 22;        // neutral centre value the interior eases into
const DOT_R = 4.5;

// --- interaction ------------------------------------------------------------

const HIT_SLOP = 5;
const CENTER_SNAP_PX = 4;    // drop this close to centre -> saturation exactly 0
const DEN_SNAP_PX = 4;       // drop this close to the middle -> density exactly 0
const FINE_SCALE = 0.25;     // shift-drag multiplier
const SAT_MAX = 100.0;
const DEN_MAX = 100.0;

const CUSTOM_PRESET = "Custom (manual)";

// --- helpers ----------------------------------------------------------------

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function findWidget(node, name) {
  return node.widgets ? node.widgets.find((w) => w.name === name) : undefined;
}

function readVal(node, name, dflt) {
  const w = findWidget(node, name);
  const v = w ? Number(w.value) : NaN;
  return Number.isFinite(v) ? v : dflt;
}

// Canonical write: set `.value` always (so the slider readout tracks live), and
// fire `.callback` only on a committed edit. A bare `value =` set does not fire
// reactivity, and committing on release rather than per-frame keeps downstream
// callback-wrappers (ReferenceCopy's mirror) cheap.
function writeVal(node, name, v, commit) {
  const w = findWidget(node, name);
  if (!w) return;
  w.value = v;
  if (commit && typeof w.callback === "function") {
    try {
      w.callback(v, app.canvas, node);
    } catch (e) {
      console.warn("[Darkroom] Log Wheels: callback threw for " + name, e);
    }
  }
}

function hsv2rgb(h, s, v) {
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

function wheelDiameter(widgetWidth) {
  const avail = Math.max(1, widgetWidth - SIDE_PAD * 2 - WHEEL_GAP * 2);
  return Math.max(MIN_D, Math.min(MAX_D, Math.floor(avail / 3)));
}

function widgetHeight(widgetWidth) {
  const d = wheelDiameter(widgetWidth);
  return TOP_PAD + d + BAR_GAP + BAR_H + LABEL_GAP + LABEL_H + CAPTION_H + BOTTOM_PAD;
}

// --- the hue/saturation disc, pre-rendered once per pixel diameter ----------
// A per-pixel disc recomputed every frame tanks canvas FPS with several nodes
// open. All three wheels share one cached image.

const discCache = new Map();

function getDisc(d) {
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
        // ease into the neutral body near the centre
        const k = Math.min(1, t * 2.2);
        rgb = [
          BODY_GREY + (rgb[0] - BODY_GREY) * k,
          BODY_GREY + (rgb[1] - BODY_GREY) * k,
          BODY_GREY + (rgb[2] - BODY_GREY) * k,
        ];
      }

      data[o] = rgb[0];
      data[o + 1] = rgb[1];
      data[o + 2] = rgb[2];
      const edge = (1 - rr) * R;              // 1px antialiased outer edge
      data[o + 3] = edge >= 1 ? 255 : Math.max(0, Math.round(edge * 255));
    }
  }

  c2.putImageData(id, 0, 0);
  discCache.set(key, cv);
  return cv;
}

// --- the widget -------------------------------------------------------------

function createWheelsWidget(node) {
  const widget = {
    name: "log_wheels_canvas",
    type: "custom",
    value: "",
    // This widget must NEVER add an entry to widgets_values. That array is
    // POSITIONAL: one stray entry shifts every following value by one, so a
    // workflow saved with this file present would load corrupted without it.
    // Frontends disagree on which flag they honour, so set all three.
    // Deliberately NOT a LiteGraph widget and NOT in node.widgets -- see
    // attachWheels() for why that is the only safe option here.

    // layout captured each draw() so mouse() hit-tests the same geometry
    lastDrawY: 0,
    lastDrawW: 0,
    geo: [],

    // interaction state
    drag: null,        // { kind: "disc" | "bar", idx }
    _nx: 0,            // live disc position in normalised -1..1 coords
    _ny: 0,
    _den: 0,           // live density during a bar drag
    _hue: 0,           // held so a centre-snap does not lose the hue direction
    _lastPx: null,
    _lastPy: null,

    computeSize(width) {
      const w = width || (node.size && node.size[0]) || 420;
      return [w, widgetHeight(w)];
    },

    draw(ctx, node, widgetWidth, y, _widgetHeight) {
      try {
        this.lastDrawY = y;
        this.lastDrawW = widgetWidth;

        const d = wheelDiameter(widgetWidth);
        const r = d / 2;
        const totalW = d * 3 + WHEEL_GAP * 2;
        const x0 = Math.max(SIDE_PAD, (widgetWidth - totalW) / 2);
        const discImg = getDisc(d);

        ctx.save();
        this.geo = [];

        for (let i = 0; i < ZONES.length; i++) {
          const zone = ZONES[i];
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

          const hue = readVal(node, zone.key + "_hue", 0);
          const sat = readVal(node, zone.key + "_saturation", 0);
          const den = readVal(node, zone.key + "_density", 0);

          const rad = clamp(sat / SAT_MAX, 0, 1) * r;
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

          // --- density bar ---
          const barX = left;
          const barY = discY + d + BAR_GAP;
          const barW = d;
          const midX = barX + barW / 2;

          ctx.fillStyle = "#1b1b1b";
          ctx.fillRect(barX, barY, barW, BAR_H);

          const t = (clamp(den, -DEN_MAX, DEN_MAX) + DEN_MAX) / (DEN_MAX * 2);
          const handleX = barX + t * barW;

          if (Math.abs(handleX - midX) > 0.5) {
            ctx.fillStyle = den >= 0 ? "rgba(124,196,240,0.75)" : "rgba(240,170,110,0.75)";
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

          // --- label ---
          const labelY = barY + BAR_H + LABEL_GAP + LABEL_H / 2;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.font = "9px sans-serif";
          ctx.fillStyle = sat >= 0.5 || Math.abs(den) >= 0.5 ? "#c8c8c8" : "#6c6c6c";
          ctx.fillText(zone.label, cx, labelY);

          this.geo.push({ cx, cy, r, barX, barY, barW });
        }

        // --- preset honesty caption -------------------------------------
        // log_wheels.py adds preset values ON TOP of these manual ones, so
        // whenever a preset is active the dots are not the applied grade.
        const presetW = findWidget(node, "preset");
        const preset = presetW ? String(presetW.value) : CUSTOM_PRESET;
        if (preset && preset !== CUSTOM_PRESET) {
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.font = "9px sans-serif";
          ctx.fillStyle = "#c9a227";
          const capY = y + widgetHeight(widgetWidth) - BOTTOM_PAD - CAPTION_H / 2;
          ctx.fillText("preset active, wheels show the manual offset only", SIDE_PAD, capY);
        }

        ctx.restore();
      } catch (err) {
        console.error("[Darkroom] Log Wheels draw() failed (rendering fallback):", err);
        try {
          ctx.save();
          ctx.fillStyle = "rgba(60,20,20,0.85)";
          ctx.fillRect(SIDE_PAD, y + TOP_PAD, Math.max(1, widgetWidth - SIDE_PAD * 2), 40);
          ctx.fillStyle = "#f0a0a0";
          ctx.font = "10px monospace";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText("[Darkroom] log wheels error -- see console", SIDE_PAD + 4, y + TOP_PAD + 20);
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
      const zone = ZONES[drag.idx];

      if (drag.kind === "disc") {
        const mag = Math.hypot(this._nx, this._ny);
        const g = this.geo[drag.idx];
        const px = mag * (g ? g.r : 1);
        let sat, hue;
        if (px <= CENTER_SNAP_PX) {
          sat = 0;                       // snap the zone fully off
          hue = this._hue;               // hold direction so leaving centre does not jump
        } else {
          sat = clamp(mag, 0, 1) * SAT_MAX;
          hue = (Math.atan2(-this._ny, this._nx) * 180) / Math.PI;
          this._hue = hue;
        }
        hue = ((Math.round(hue) % 360) + 360) % 360;
        writeVal(node, zone.key + "_hue", hue, commit);
        writeVal(node, zone.key + "_saturation", clamp(Math.round(sat), 0, SAT_MAX), commit);
      } else {
        let den = this._den;
        const g = this.geo[drag.idx];
        if (g) {
          const midX = g.barX + g.barW / 2;
          const handleX = g.barX + ((clamp(den, -DEN_MAX, DEN_MAX) + DEN_MAX) / (DEN_MAX * 2)) * g.barW;
          if (Math.abs(handleX - midX) <= DEN_SNAP_PX) den = 0;
        }
        writeVal(node, zone.key + "_density", clamp(Math.round(den), -DEN_MAX, DEN_MAX), commit);
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

            // density bar first -- it sits below the disc, no overlap
            if (
              px >= g.barX - HIT_SLOP && px <= g.barX + g.barW + HIT_SLOP &&
              py >= g.barY - HIT_SLOP && py <= g.barY + BAR_H + HIT_SLOP
            ) {
              this.drag = { kind: "bar", idx: i };
              this._den = ((px - g.barX) / g.barW) * (DEN_MAX * 2) - DEN_MAX;
              this._lastPx = px;
              this._lastPy = py;
              this.flush(node, false);
              node.setDirtyCanvas(true, true);
              return true;
            }

            // disc
            if (Math.hypot(px - g.cx, py - g.cy) <= g.r + HIT_SLOP) {
              this.drag = { kind: "disc", idx: i };
              this._hue = readVal(node, ZONES[i].key + "_hue", 0);
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
            if (event.shiftKey && this._lastPx != null) {
              this._den += ((px - this._lastPx) / g.barW) * (DEN_MAX * 2) * FINE_SCALE;
            } else {
              this._den = ((px - g.barX) / g.barW) * (DEN_MAX * 2) - DEN_MAX;
            }
            this._den = clamp(this._den, -DEN_MAX, DEN_MAX);
          }

          this._lastPx = px;
          this._lastPy = py;
          this.flush(node, false);          // live readout, no callback storm
          node.setDirtyCanvas(true, true);
          return true;
        }

        if (isUp && this.drag) {
          this.flush(node, true);           // the committed edit
          this.drag = null;
          this._lastPx = null;
          this._lastPy = null;
          node.setDirtyCanvas(true, true);
          return true;
        }

        return false;
      } catch (err) {
        console.error("[Darkroom] Log Wheels mouse() failed:", err);
        this.drag = null;
        this._lastPx = null;
        this._lastPy = null;
        return false;
      }
    },
  };

  return widget;
}

// --- attach to the node -----------------------------------------------------

// onDrawForeground draws from the node BODY origin, which is also where
// LiteGraph lays out the input/output slot rows. Start below them or the
// wheels sit on top of the `image` slots.
function topY(node) {
  // Modern ComfyUI lists EVERY widget in node.inputs as well, so
  // `node.inputs.length` here is 14, not 1. Only inputs without a `.widget`
  // back-reference are drawn as actual sockets and take a slot row.
  const socketIns = (node.inputs || []).filter((i) => !i.widget).length;
  const rows = Math.max(socketIns, node.outputs ? node.outputs.length : 0);
  const slotH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_SLOT_HEIGHT) || 20;
  return rows * slotH + 6;
}

function attachWheels(node) {
  if (node._darkroomLogWheelsAttached) return true;

  const anchor = findWidget(node, "shadow_hue");
  if (!anchor) return false;   // widgets not built yet -- caller retries

  node._darkroomLogWheelsAttached = true;

  const ctrl = createWheelsWidget(node);
  node._darkroomWheels = ctrl;

  // The controller is deliberately kept OUT of node.widgets.
  //
  // LiteGraph's save path writes widgets_values BY INDEX while skipping
  // serialize:false widgets:
  //   for (const [i, w] of widgets.entries()) { if (w.serialize === false) continue; vals[i] = w.value }
  // and its load path reads SEQUENTIALLY with the same skip:
  //   for (const w of widgets) if (w.serialize !== false) w.value = vals[t++]
  // Those two only agree when the non-serialised widget is LAST. Anywhere else
  // the save leaves a null hole and every later value shifts by one slot on
  // load, silently corrupting existing workflows. Measured on frontend 1.48.7,
  // both directions, before this was rewritten.
  //
  // Staying out of the array sidesteps the bug entirely: widgets_values comes
  // out byte-identical to a stock save, and the wheels can still sit ABOVE the
  // sliders because `widgets_start_y` reserves space at the top of the stack.
  const reserve = () => {
    node.widgets_start_y = topY(node) + widgetHeight(Math.max(node.size[0] || 420, 420));
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
  // reads the float widgets every frame, so all this needs to do is force a
  // repaint (and preserve the original callback's return value).
  for (const zone of ZONES) {
    for (const suffix of ["_hue", "_saturation", "_density"]) {
      const w = findWidget(node, zone.key + suffix);
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

  // Widen enough for three wheels in a row, without shrinking a saved size.
  try {
    const need = node.computeSize();
    node.setSize([
      Math.max(node.size[0] || 0, 420),
      Math.max(node.size[1] || 0, need[1]),
    ]);
  } catch (e) {
    console.warn("[Darkroom] Log Wheels: resize failed", e);
  }

  node.setDirtyCanvas(true, true);
  return true;
}

app.registerExtension({
  name: "AKURATE.DarkroomLogWheels",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE) return;
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      // Widgets are occasionally built a tick after onNodeCreated returns.
      if (!attachWheels(this)) setTimeout(() => attachWheels(this), 0);
      return r;
    };
  },
});
