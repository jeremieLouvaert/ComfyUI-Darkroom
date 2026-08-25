// ComfyUI-Darkroom -- 3-Way Color Balance colour wheels.
//
// Three hue/intensity discs (shadow / midtone / highlight) driven by the node's
// existing six FLOAT widgets. All the machinery lives in
// darkroom_wheels_core.js -- see its header for the architecture and for why
// the controller must never enter node.widgets.
//
// Unlike Log Wheels this node has NO per-zone density, so the zones declare no
// bar and the core drops that row from the layout. `preserve_luminance`,
// `master_saturation` and `strength` stay as plain sliders below: they are
// node-wide, not per-zone, so they do not belong on a wheel.
//
// Backend: nodes/three_way_color_balance.py. Every zone is gated on
// `intensity >= 0.5`, exactly like Log Wheels, so the centre snap genuinely
// turns that zone off.

import { registerWheelNode } from "./darkroom_wheels_core.js";

registerWheelNode("DarkroomThreeWayColorBalance", {
  tag: "ThreeWay",
  satMax: 100,
  minWidth: 420,
  zones: [
    { key: "shadow",    label: "SHADOW",    hue: "shadow_hue",    sat: "shadow_intensity" },
    { key: "midtone",   label: "MIDTONE",   hue: "midtone_hue",   sat: "midtone_intensity" },
    { key: "highlight", label: "HIGHLIGHT", hue: "highlight_hue", sat: "highlight_intensity" },
  ],
  // three_way_color_balance.py ADDS preset values on top of the manual ones.
  preset: {
    widget: "preset",
    custom: "Custom (manual)",
    caption: "preset active, wheels show the manual offset only",
  },
});
