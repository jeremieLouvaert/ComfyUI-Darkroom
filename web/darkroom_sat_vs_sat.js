// ComfyUI-Darkroom -- Sat vs Sat curve editor.
//
// Four saturation bands. x positions are the band centres from
// sat_vs_sat.py (0.125/0.375/0.625/0.875). The axis strip below is a
// CHROMA indicator (grey to colourful), not a hue axis.
//
// Machinery lives in darkroom_curve_core.js (geometry/painting) and
// darkroom_canvas_widget.js (attach + the serialisation rules). Control-point
// x positions come from the backend's own band constants, NOT guessed spacing.

import { registerCanvasNode } from "./darkroom_canvas_widget.js";
import { createCurveController } from "./darkroom_curve_core.js";

const SPEC = {
  tag: "SatVsSat",
  axis: "chroma",
  range: 100,
  unit: "",
  minWidth: 400,
  // band centres in nodes/sat_vs_sat.py
  points: [
    { x: 0.125, widget: "low_sat_adjust",      label: "Low" },
    { x: 0.375, widget: "mid_low_sat_adjust",  label: "Mid-" },
    { x: 0.625, widget: "mid_high_sat_adjust", label: "Mid+" },
    { x: 0.875, widget: "high_sat_adjust",     label: "High" },
  ],
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, curve shows the manual offset only",
  },
};

registerCanvasNode("DarkroomSatVsSat", "AKURATE.DarkroomSatVsSat",
  (node) => createCurveController(node, SPEC),
  { tag: SPEC.tag, minWidth: SPEC.minWidth, requireWidget: SPEC.points[0].widget });
