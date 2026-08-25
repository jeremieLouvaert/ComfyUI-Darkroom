// ComfyUI-Darkroom -- Lum vs Sat curve editor.
//
// Five luminance zones adjusting saturation. x positions are the zone
// centres from LUM_ZONES in lum_vs_sat.py (0.05/0.20/0.50/0.80/0.95)
// -- deliberately NOT evenly spaced.
//
// Machinery lives in darkroom_curve_core.js (geometry/painting) and
// darkroom_canvas_widget.js (attach + the serialisation rules). Control-point
// x positions come from the backend's own band constants, NOT guessed spacing.

import { registerCanvasNode } from "./darkroom_canvas_widget.js";
import { createCurveController } from "./darkroom_curve_core.js";

const SPEC = {
  tag: "LumVsSat",
  axis: "luma",
  range: 100,
  unit: "",
  minWidth: 400,
  // LUM_ZONES centres in nodes/lum_vs_sat.py
  points: [
    { x: 0.05, widget: "blacks_saturation",     label: "Blacks" },
    { x: 0.20, widget: "shadows_saturation",    label: "Shad" },
    { x: 0.50, widget: "midtones_saturation",   label: "Mids" },
    { x: 0.80, widget: "highlights_saturation", label: "Highs" },
    { x: 0.95, widget: "whites_saturation",     label: "Whites" },
  ],
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, curve shows the manual offset only",
  },
};

registerCanvasNode("DarkroomLumVsSat", "AKURATE.DarkroomLumVsSat",
  (node) => createCurveController(node, SPEC),
  { tag: SPEC.tag, minWidth: SPEC.minWidth, requireWidget: SPEC.points[0].widget });
