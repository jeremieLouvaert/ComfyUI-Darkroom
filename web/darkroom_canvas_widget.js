// ComfyUI-Darkroom -- shared canvas-controller plumbing.
//
// Every Darkroom canvas editor (colour wheels, curve strips, anything later)
// attaches through here. The two rules below are the whole reason this file
// exists, and they must not be re-derived per component.
//
// RULE 1 -- the canvas is a VIEW. A node's existing float widgets are the only
// state. The controller reads them every draw and writes back on edit: set
// `.value` on every move so the slider readout tracks live, and fire
// `.callback` once on release as the committed edit (a bare `value =` set does
// not fire reactivity, and firing per-frame makes downstream callback wrappers
// like ReferenceCopy's mirror expensive). No INPUT_TYPES change, so stored
// workflows and API-format prompts are untouched, and deleting the .js leaves
// a working slider node.
//
// RULE 2 -- the controller is NOT a LiteGraph widget and must never enter
// node.widgets. LiteGraph saves widgets_values BY INDEX skipping
// serialize===false widgets, but loads it SEQUENTIALLY with the same skip:
//   save: for (const [i,w] of widgets.entries()) { if (w.serialize===false) continue; vals[i]=w.value }
//   load: let t=0; for (const w of widgets) if (w.serialize!==false) w.value=vals[t++]
// Those agree only when the non-serialised widget is LAST, so a widget
// anywhere else silently shifts every later value by one slot on load.
// (`options.serialize` is not consulted by configure() at all; the top-level
// `serialize` is, and `serializeValue` affects only the save side.) Measured on
// frontend 1.48.7 -- it is what the ComfyUI-Field v0.6.1 fix was about.
//
// Consequence of rule 2: space is reserved with `widgets_start_y`, which only
// works at the TOP of the widget stack. A canvas cannot be injected between
// two widgets; there is no supported way to do that.

import { app } from "../../scripts/app.js";

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

// See RULE 1. `commit` false during a drag, true on release.
export function writeVal(node, name, v, commit, tag) {
  const w = findWidget(node, name);
  if (!w) return;
  w.value = v;
  if (commit && typeof w.callback === "function") {
    try {
      w.callback(v, app.canvas, node);
    } catch (e) {
      console.warn("[Darkroom] " + (tag || "canvas") + ": callback threw for " + name, e);
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

// onDrawForeground draws from the node BODY origin, which is also where
// LiteGraph lays out the socket rows. Modern ComfyUI lists EVERY widget in
// node.inputs as well, so count only inputs without a `.widget` back-reference
// -- using node.inputs.length on a 14-widget node reserves 14 rows and opens a
// ~290px void at the top.
export function topY(node) {
  const socketIns = (node.inputs || []).filter((i) => !i.widget).length;
  const rows = Math.max(socketIns, node.outputs ? node.outputs.length : 0);
  const slotH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_SLOT_HEIGHT) || 20;
  return rows * slotH + 6;
}

// A controller must expose:
//   draw(ctx, node, width, y, h)   paint, and cache geometry for hit-testing
//   mouse(event, pos, node)        true to claim the gesture
//   computeSize(width) -> [w, h]   how much vertical space to reserve
//   dragging() -> boolean          whether a gesture is in progress
//   syncedWidgets() -> [names]     float widgets to hook for reverse sync
export function attachCanvasController(node, ctrl, opts) {
  const o = opts || {};
  const tag = o.tag || "canvas";
  const minW = o.minWidth || 420;

  if (node._darkroomCanvasAttached) return true;
  if (o.requireWidget && !findWidget(node, o.requireWidget)) return false;
  node._darkroomCanvasAttached = true;
  node._darkroomCanvas = ctrl;

  const reserve = () => {
    node.widgets_start_y = topY(node) + ctrl.computeSize(Math.max(node.size[0] || minW, minW))[1];
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
    if (ctrl.dragging()) {
      ctrl.mouse({ type: "pointermove", shiftKey: !!(e && e.shiftKey) }, pos, this);
      return true;
    }
    return origMove ? origMove.apply(this, arguments) : undefined;
  };

  const origUp = node.onMouseUp;
  node.onMouseUp = function (e, pos, canvas) {
    if (ctrl.dragging()) {
      ctrl.mouse({ type: "pointerup", shiftKey: !!(e && e.shiftKey) }, pos, this);
      if (typeof this.captureInput === "function") this.captureInput(false);
      return true;
    }
    return origUp ? origUp.apply(this, arguments) : undefined;
  };

  // Reverse sync: editing a slider by hand must move the handle. draw() already
  // reads the float widgets every frame, so this only forces a repaint (while
  // preserving the original callback's return value).
  for (const name of ctrl.syncedWidgets()) {
    const w = findWidget(node, name);
    if (!w || w._darkroomCanvasHooked) continue;
    w._darkroomCanvasHooked = true;
    const orig = w.callback;
    w.callback = function (value, ...rest) {
      const out = orig ? orig.apply(this, [value, ...rest]) : undefined;
      node.setDirtyCanvas(true, true);
      return out;
    };
  }

  try {
    const need = node.computeSize();
    node.setSize([
      Math.max(node.size[0] || 0, minW),
      Math.max(node.size[1] || 0, need[1]),
    ]);
  } catch (e) {
    console.warn("[Darkroom] " + tag + ": resize failed", e);
  }

  node.setDirtyCanvas(true, true);
  return true;
}

// Register a node type against a controller factory. Handles the case where the
// widgets are built a tick after onNodeCreated returns.
export function registerCanvasNode(nodeTypeName, extensionName, makeController, opts) {
  app.registerExtension({
    name: extensionName,
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
      if (nodeData.name !== nodeTypeName) return;
      const orig = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = orig ? orig.apply(this, arguments) : undefined;
        const attach = () => attachCanvasController(this, makeController(this), opts);
        if (!attach()) setTimeout(attach, 0);
        return r;
      };
    },
  });
}
