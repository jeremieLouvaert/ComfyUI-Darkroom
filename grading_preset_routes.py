"""
Serves the grading preset tables to the canvas widgets, so a wheel or curve can
draw the EFFECTIVE grade (manual + preset) as a ghost instead of apologising for
not knowing it.

Why this exists: every grading preset is applied in PYTHON at execute time and
its numbers never reach the browser, so a view-only canvas widget can only ever
draw the manual offset. That is why the shipped nodes carry the caption
"preset active, curve shows the manual offset only". This route closes that gap
without moving any state into the frontend.

Two rules, not one, and the second is the reason a naive "add the preset" ghost
would be wrong:

  add            effective = manual + value            (every curve node, and
                                                        the wheels' sat/density)
  set            effective = value                     (Log Wheels zone ranges)
  set_if_below   effective = value  IF  manual[gate] < threshold
                 else manual                           (wheel HUE -- log_wheels.py
                                                        and three_way both keep the
                                                        MANUAL hue once that zone's
                                                        saturation/intensity is
                                                        turned up)

and one whole-node case:

  bypass         the preset returns the image untouched (Log Wheels'
                 "Neutral - reset all"), so the ghost is plain identity.

The ops are emitted rather than the raw dataclasses because the field names do
not match the widget names (`ToneCurvePreset.master` is a 5-tuple; HueVsHue's
`red` is the widget `red_shift`), and that mapping is exactly the thing that
would silently drift. `tools/test_preset_ops.py` pins it: for every node, every
preset and a manual-value set that straddles each gate, it runs the node with
the preset against the node with `Custom (manual)` and the ops applied, and
requires the two images to be BITWISE identical.
"""

from aiohttp import web
from server import PromptServer

from .data.grading_presets import (
    COLOR_BALANCE_PRESETS,
    HUE_VS_HUE_PRESETS,
    HUE_VS_SAT_PRESETS,
    LOG_WHEELS_PRESETS,
    LUM_VS_SAT_PRESETS,
    SAT_VS_SAT_PRESETS,
    TONE_CURVE_PRESETS,
)

CUSTOM = "Custom (manual)"
NEUTRAL = "Neutral — reset all"

# Zone gate threshold, from log_wheels.py:109 and three_way_color_balance.py:89.
GATE = 0.5

TONE_MASTER = ["shadows", "darks", "midtones", "lights", "highlights"]
TONE_CHANNEL = ["red_shadows", "red_highlights", "green_shadows",
                "green_highlights", "blue_shadows", "blue_highlights"]

ZONES = ["shadow", "midtone", "highlight"]

SAT_VS_SAT_MAP = {"low": "low_sat_adjust", "mid_low": "mid_low_sat_adjust",
                  "mid_high": "mid_high_sat_adjust", "high": "high_sat_adjust"}


def _add(widget, value):
    return {"w": widget, "op": "add", "v": float(value)}


def _set(widget, value):
    return {"w": widget, "op": "set", "v": float(value)}


def _set_if_below(widget, value, gate):
    return {"w": widget, "op": "set_if_below", "v": float(value),
            "gate": gate, "thr": GATE}


def _tone_curve_ops(p):
    ops = [_add(TONE_MASTER[i], p.master[i]) for i in range(5)]
    ops += [_add(f, getattr(p, f)) for f in TONE_CHANNEL]
    return ops


def _log_wheels_ops(p):
    ops = []
    for z in ZONES:
        # hue is a conditional REPLACE gated on that zone's own saturation
        ops.append(_set_if_below("%s_hue" % z, getattr(p, "%s_hue" % z),
                                 "%s_saturation" % z))
        ops.append(_add("%s_saturation" % z, getattr(p, "%s_saturation" % z)))
        ops.append(_add("%s_density" % z, getattr(p, "%s_density" % z)))
    ops.append(_set("shadow_range", p.shadow_range))
    ops.append(_set("highlight_range", p.highlight_range))
    return ops


def _color_balance_ops(p):
    ops = []
    for z in ZONES:
        ops.append(_set_if_below("%s_hue" % z, getattr(p, "%s_hue" % z),
                                 "%s_intensity" % z))
        ops.append(_add("%s_intensity" % z, getattr(p, "%s_intensity" % z)))
    ops.append(_add("master_saturation", p.master_saturation))
    return ops


def _suffix_ops(fields, suffix):
    def build(p):
        return [_add(f + suffix, getattr(p, f)) for f in fields]
    return build


HUE_BANDS = ["red", "orange", "yellow", "green", "aqua", "blue",
             "purple", "magenta"]
LUM_BANDS = ["blacks", "shadows", "midtones", "highlights", "whites"]

# node type -> (preset table, ops builder)
NODES = {
    "DarkroomToneCurve": (TONE_CURVE_PRESETS, _tone_curve_ops),
    "DarkroomLogWheels": (LOG_WHEELS_PRESETS, _log_wheels_ops),
    "DarkroomThreeWayColorBalance": (COLOR_BALANCE_PRESETS, _color_balance_ops),
    "DarkroomHueVsHue": (HUE_VS_HUE_PRESETS, _suffix_ops(HUE_BANDS, "_shift")),
    "DarkroomHueVsSat": (HUE_VS_SAT_PRESETS, _suffix_ops(HUE_BANDS, "_saturation")),
    "DarkroomLumVsSat": (LUM_VS_SAT_PRESETS, _suffix_ops(LUM_BANDS, "_saturation")),
    "DarkroomSatVsSat": (SAT_VS_SAT_PRESETS,
                         lambda p: [_add(SAT_VS_SAT_MAP[f], getattr(p, f))
                                    for f in SAT_VS_SAT_MAP]),
}

# Presets that bypass the node entirely rather than changing values.
BYPASS = {"DarkroomLogWheels": [NEUTRAL]}


def build_payload():
    out = {"custom": CUSTOM, "nodes": {}}
    for node_type, (table, builder) in NODES.items():
        presets = {}
        for name in BYPASS.get(node_type, []):
            presets[name] = {"bypass": True}
        for name, p in table.items():
            ops = [o for o in builder(p)
                   if o["op"] != "add" or o["v"] != 0.0]   # drop no-op adds
            presets[name] = {"ops": ops}
        out["nodes"][node_type] = {"presets": presets}
    return out


@PromptServer.instance.routes.get("/darkroom/grading_presets")
async def darkroom_grading_presets(_request):
    return web.json_response(build_payload(),
                             headers={"Cache-Control": "no-store"})
