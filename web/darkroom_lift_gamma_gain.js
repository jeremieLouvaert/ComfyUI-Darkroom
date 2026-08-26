// ComfyUI-Darkroom -- Lift Gamma Gain colour wheels.
//
// Four DaVinci-style primaries wheels (Lift / Gamma / Gain / Offset), each a
// chroma-only disc over the node's existing three channel widgets, with its
// master beneath as a bar.
//
// This node is the ONE whose wheel mapping needed a derivation rather than a
// convention: its parameters are Cartesian (lift_r/g/b + lift_master), not
// polar. The model, the measurements behind it, and every constant here are in
// docs/lgg-wheel-derivation.md (v2, signed 2026-08-26). Do not change amp,
// the luma weights, or the write precision without amending that document.
//
// Two facts drive the shape of this spec:
//   - lift/offset combine with their master ADDITIVELY, gamma/gain
//     MULTIPLICATIVELY (nodes/lift_gamma_gain.py), so the two kinds need
//     different forward maps and different master bars.
//   - a LINEAR master bar would put neutral 1.0 at 23% of gamma's [0.1,4]
//     slider, so gamma/gain use a log bar (barLog) that is exactly symmetric
//     about 1.0.

import { registerWheelNode } from "./darkroom_wheels_core.js";
import { GROUP_AMP } from "./darkroom_lumanull.js";

registerWheelNode("DarkroomLiftGammaGain", {
  tag: "LiftGammaGain",
  satMax: 100,
  minWidth: 560,          // four discs in a row, Resolve's primaries layout
  zones: [
    {
      key: "lift", label: "LIFT", cartesian: true, mul: false,
      amp: GROUP_AMP.lift,
      channels: ["lift_r", "lift_g", "lift_b"],
      bar: "lift_master", barMin: -1, barMax: 1,
    },
    {
      key: "gamma", label: "GAMMA", cartesian: true, mul: true,
      amp: GROUP_AMP.gamma,
      channels: ["gamma_r", "gamma_g", "gamma_b"],
      bar: "gamma_master", barLog: true, barWidgetMin: 0.1, barWidgetMax: 4.0,
    },
    {
      key: "gain", label: "GAIN", cartesian: true, mul: true,
      amp: GROUP_AMP.gain,
      channels: ["gain_r", "gain_g", "gain_b"],
      bar: "gain_master", barLog: true, barWidgetMin: 0.0, barWidgetMax: 4.0,
    },
    {
      key: "offset", label: "OFFSET", cartesian: true, mul: false,
      amp: GROUP_AMP.offset,
      channels: ["offset_r", "offset_g", "offset_b"],
      bar: "offset_master", barMin: -0.5, barMax: 0.5,
    },
  ],
  // Lift Gamma Gain has no preset widget, so no honesty caption is needed.
});
