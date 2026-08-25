// ComfyUI-Darkroom -- Hue vs Sat curve editor.
//
// Eight hue bands, each adjusting its band's saturation by +/-100.
// x positions are HUE_CENTERS/360 from hue_vs_sat.py, same non-even
// spacing as Hue vs Hue.
//
// Machinery lives in darkroom_curve_core.js (geometry/painting) and
// darkroom_canvas_widget.js (attach + the serialisation rules). Control-point
// x positions come from the backend's own band constants, NOT guessed spacing.

import { registerCanvasNode } from "./darkroom_canvas_widget.js";
import { createCurveController } from "./darkroom_curve_core.js";

const SPEC = {
  tag: "HueVsSat",
  axis: "hue",
  range: 100,
  unit: "",
  minWidth: 420,
  // HUE_CENTERS in nodes/hue_vs_sat.py, divided by 360
  points: [
    { x: 0.0000,  widget: "red_saturation", label: "Red" },
    { x: 0.0833,  widget: "orange_saturation", label: "Ora" },
    { x: 0.1667,  widget: "yellow_saturation", label: "Yel" },
    { x: 0.3333,  widget: "green_saturation", label: "Gre" },
    { x: 0.5000,  widget: "aqua_saturation", label: "Aqu" },
    { x: 0.6667,  widget: "blue_saturation", label: "Blu" },
    { x: 0.7500,  widget: "purple_saturation", label: "Pur" },
    { x: 0.9167,  widget: "magenta_saturation", label: "Mag" },
  ],
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, curve shows the manual offset only",
  },
};

registerCanvasNode("DarkroomHueVsSat", "AKURATE.DarkroomHueVsSat",
  (node) => createCurveController(node, SPEC),
  { tag: SPEC.tag, minWidth: SPEC.minWidth, requireWidget: SPEC.points[0].widget });
