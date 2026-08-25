// ComfyUI-Darkroom -- Log Wheels colour wheels.
//
// Three DaVinci-style log wheels (shadow / midtone / highlight), each a
// hue/saturation disc with a density bar beneath it, driven by the node's
// existing nine FLOAT widgets. All the machinery lives in
// darkroom_wheels_core.js -- see its header for the architecture and for why
// the controller must never enter node.widgets.
//
// Backend: nodes/log_wheels.py. Every zone is gated on `sat >= 0.5`, so the
// centre snap (drop within a few px of the crosshair -> saturation exactly 0)
// genuinely turns that zone off.

import { registerWheelNode } from "./darkroom_wheels_core.js";

registerWheelNode("DarkroomLogWheels", {
  tag: "LogWheels",
  satMax: 100,
  minWidth: 420,
  zones: [
    { key: "shadow",    label: "SHADOW",    hue: "shadow_hue",
      sat: "shadow_saturation",    bar: "shadow_density",    barMin: -100, barMax: 100 },
    { key: "midtone",   label: "MIDTONE",   hue: "midtone_hue",
      sat: "midtone_saturation",   bar: "midtone_density",   barMin: -100, barMax: 100 },
    { key: "highlight", label: "HIGHLIGHT", hue: "highlight_hue",
      sat: "highlight_saturation", bar: "highlight_density", barMin: -100, barMax: 100 },
  ],
  // log_wheels.py:106-121 ADDS preset values on top of the manual ones, so with
  // a preset active the dots are not the applied grade.
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, wheels show the manual offset only",
  },
});
