// ComfyUI-Darkroom -- Hue vs Hue curve editor.
//
// Eight hue bands, each shifting its band's hue by +/-60 degrees.
// x positions are HUE_CENTERS/360 from hue_vs_hue.py -- the bands are
// NOT evenly spaced (0/30/60/120/180/240/270/330), so even spacing
// would misalign every handle against the spectrum strip beneath.
//
// Machinery lives in darkroom_curve_core.js (geometry/painting) and
// darkroom_canvas_widget.js (attach + the serialisation rules). Control-point
// x positions come from the backend's own band constants, NOT guessed spacing.

import { registerCanvasNode } from "./darkroom_canvas_widget.js";
import { createCurveController } from "./darkroom_curve_core.js";

const SPEC = {
  tag: "HueVsHue",
  axis: "hue",
  range: 60,
  unit: "\u00b0",
  minWidth: 420,
  // HUE_CENTERS in nodes/hue_vs_hue.py, divided by 360
  points: [
    { x: 0.0000,  widget: "red_shift", label: "Red" },
    { x: 0.0833,  widget: "orange_shift", label: "Ora" },
    { x: 0.1667,  widget: "yellow_shift", label: "Yel" },
    { x: 0.3333,  widget: "green_shift", label: "Gre" },
    { x: 0.5000,  widget: "aqua_shift", label: "Aqu" },
    { x: 0.6667,  widget: "blue_shift", label: "Blu" },
    { x: 0.7500,  widget: "purple_shift", label: "Pur" },
    { x: 0.9167,  widget: "magenta_shift", label: "Mag" },
  ],
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, curve shows the manual offset only",
  },
};

registerCanvasNode("DarkroomHueVsHue", "AKURATE.DarkroomHueVsHue",
  (node) => createCurveController(node, SPEC),
  { tag: SPEC.tag, minWidth: SPEC.minWidth, requireWidget: SPEC.points[0].widget });
