// ComfyUI-Darkroom -- Tone Curve curve editor.
//
// Five control points at fixed luminance positions (0/25/50/75/100%),
// each bound to the node's existing FLOAT widget. The per-channel
// shadow/highlight offsets stay as plain sliders below -- they are a
// different axis (per-channel), not points on this curve.
//
// Machinery lives in darkroom_curve_core.js (geometry/painting) and
// darkroom_canvas_widget.js (attach + the serialisation rules). Control-point
// x positions come from the backend's own band constants, NOT guessed spacing.

import { registerCanvasNode } from "./darkroom_canvas_widget.js";
import { createCurveController } from "./darkroom_curve_core.js";

const SPEC = {
  tag: "ToneCurve",
  axis: "luma",
  range: 50,
  unit: "",
  minWidth: 400,
  // tone_curve.py tooltips: curve points at 0/25/50/75/100% luminance
  points: [
    { x: 0.00, widget: "shadows",    label: "Shadows" },
    { x: 0.25, widget: "darks",      label: "Darks" },
    { x: 0.50, widget: "midtones",   label: "Mids" },
    { x: 0.75, widget: "lights",     label: "Lights" },
    { x: 1.00, widget: "highlights", label: "Highs" },
  ],
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, curve shows the manual offset only",
  },
};

registerCanvasNode("DarkroomToneCurve", "AKURATE.DarkroomToneCurve",
  (node) => createCurveController(node, SPEC),
  { tag: SPEC.tag, minWidth: SPEC.minWidth, requireWidget: SPEC.points[0].widget });
