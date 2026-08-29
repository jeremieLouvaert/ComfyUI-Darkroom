"""
Pins the preset ops served by /darkroom/grading_presets to what the nodes
ACTUALLY do.

The ghost overlay draws `effective = apply(manual, ops)`. If those ops ever drift
from the Python -- a renamed widget, a preset field added, an `add` that should
have been a conditional replace -- the ghost silently lies, and a lying ghost is
worse than the caption it replaced.

The check has no opinion of its own: for every node, every preset, and a set of
manual values chosen to straddle each gate, it runs

    node(image, preset=P,        **manual)
    node(image, preset="Custom", **apply_ops(manual, ops_for_P))

and requires the two rendered images to be BITWISE identical. Anything the ops
get wrong changes pixels.

Run: python.exe tools/test_preset_ops.py
"""

import itertools
import os
import sys
import types

import numpy as np
import torch
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = r"F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
for p in (COMFY, os.path.dirname(PACK)):
    if p not in sys.path:
        sys.path.insert(0, p)

import server as _server_mod  # noqa: E402
if getattr(_server_mod.PromptServer, "instance", None) is None:
    _server_mod.PromptServer.instance = types.SimpleNamespace(
        routes=web.RouteTableDef())

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "darkroom_pack", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["darkroom_pack"] = _pack
_spec.loader.exec_module(_pack)

from darkroom_pack.grading_preset_routes import (  # noqa: E402
    CUSTOM, build_payload,
)

NCM = _pack.NODE_CLASS_MAPPINGS

PASS = 0
FAIL = 0
CONTROLS_FIRED = 0
CONTROLS_TOTAL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("  FAIL %s%s" % (name, (" | " + detail) if detail else ""))
    return ok


def make_image(seed=3):
    rng = np.random.default_rng(seed)
    a = rng.random((1, 48, 64, 3)).astype(np.float32)
    # guarantee shadows, mids and highlights are all populated, so a preset that
    # only touches one zone still moves pixels
    a[0, :16] *= 0.25
    a[0, 32:] = 0.75 + a[0, 32:] * 0.25
    return torch.from_numpy(a)


IMG = make_image()


def node_defaults(cls):
    """Every declared widget at its python default, so a call is well-formed."""
    spec = cls.INPUT_TYPES()
    out = {}
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            if not isinstance(decl, (tuple, list)) or not decl:
                continue
            t = decl[0]
            opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            if t == "IMAGE" or t == "MASK":
                continue
            if isinstance(t, (list, tuple)):
                out[name] = opts.get("default", t[0] if t else None)
            elif "default" in opts:
                out[name] = opts["default"]
    return out


def apply_ops(manual, ops):
    """The JS's rule, in Python. Gates read the MANUAL values, never the running
    result -- matching the nodes, which test the incoming argument."""
    eff = dict(manual)
    for o in ops:
        w, op, v = o["w"], o["op"], o["v"]
        if w not in manual:
            continue
        if op == "add":
            eff[w] = manual[w] + v
        elif op == "set":
            eff[w] = v
        elif op == "set_if_below":
            if manual.get(o["gate"], 0.0) < o["thr"]:
                eff[w] = v
        else:
            raise AssertionError("unknown op %r" % op)
    return eff


def render(cls, kwargs):
    node = cls()
    fn = getattr(cls, cls.FUNCTION)
    fn = getattr(fn, "__wrapped__", fn)
    out = fn(node, image=IMG.clone(), **kwargs)
    if isinstance(out, dict):
        out = out.get("result")
    return out[0]


# Manual value sets. The gated ones matter most: with a zone's saturation or
# intensity at or above 0.5 the node keeps the MANUAL hue, so an ops table that
# used a plain `add`/`set` there passes at zero and fails here.
GATE_WIDGETS = ["shadow_saturation", "midtone_saturation", "highlight_saturation",
                "shadow_intensity", "midtone_intensity", "highlight_intensity"]


def manual_sets(defaults):
    sets = [("defaults", {})]
    gates = [g for g in GATE_WIDGETS if g in defaults]
    if gates:
        sets.append(("gates_open", {g: 30.0 for g in gates}))
        sets.append(("gates_mixed", {g: (30.0 if i == 0 else 0.0)
                                     for i, g in enumerate(gates)}))
        sets.append(("gates_just_under", {g: 0.49 for g in gates}))
        sets.append(("gates_just_over", {g: 0.5 for g in gates}))
    # a non-zero manual offset on a few ordinary widgets, so `add` is exercised
    numeric = [k for k, v in defaults.items()
               if isinstance(v, float) and k not in gates and k != "strength"]
    if numeric:
        sets.append(("manual_offsets",
                     {k: (7.0 if i % 2 == 0 else -5.0)
                      for i, k in enumerate(numeric[:6])}))
    return sets


def run_node(node_type, entry, mutate=None, label=""):
    """Returns (checked, mismatches). mutate lets a negative control corrupt ops."""
    global CONTROLS_FIRED
    cls = NCM[node_type]
    defaults = node_defaults(cls)
    checked = 0
    mismatches = 0

    for preset_name, meta in entry["presets"].items():
        for set_name, overrides in manual_sets(defaults):
            manual = dict(defaults)
            manual.update({k: v for k, v in overrides.items() if k in defaults})

            with_preset = dict(manual)
            with_preset["preset"] = preset_name
            ref = render(cls, with_preset)

            if meta.get("bypass"):
                # `bypass` does NOT mean "the widgets read zero" -- log_wheels.py
                # returns the INPUT IMAGE for "Neutral - reset all", discarding
                # the manual values entirely. So the node is disabled, and the
                # ghost's honest reading is "no grade at all", not "manual grade".
                got = IMG
            else:
                ops = meta["ops"]
                if mutate is not None:
                    ops = mutate(ops)
                eff = apply_ops(manual, ops)
                eff["preset"] = CUSTOM
                got = render(cls, eff)

            checked += 1
            if not torch.equal(ref, got):
                mismatches += 1
                if mutate is None:
                    d = float((ref - got).abs().max())
                    print("     %s / %r / %s -> max |diff| %.6g"
                          % (node_type, preset_name, set_name, d))
    return checked, mismatches


def main():
    global CONTROLS_TOTAL, CONTROLS_FIRED
    payload = build_payload()

    print("=" * 72)
    print("PRESET OPS <-> NODE EQUIVALENCE")
    print("=" * 72)

    total = 0
    for node_type, entry in payload["nodes"].items():
        if node_type not in NCM:
            check("%s registered" % node_type, False, "not in NODE_CLASS_MAPPINGS")
            continue
        checked, bad = run_node(node_type, entry)
        total += checked
        ok = check("%s ops reproduce the node" % node_type, bad == 0,
                   "%d of %d combinations differ" % (bad, checked))
        print("  %-30s %4d combinations  %s"
              % (node_type, checked, "OK" if ok else "MISMATCH"))

    print("\n%d preset x manual-set combinations compared" % total)

    # ---------------------------------------------------------------- controls
    print("\n" + "=" * 72)
    print("NEGATIVE CONTROLS (each must FIRE)")
    print("=" * 72)

    def nc(name, node_type, mutate, applies=lambda e: True):
        global CONTROLS_TOTAL, CONTROLS_FIRED
        entry = payload["nodes"][node_type]
        if not applies(entry):
            return
        CONTROLS_TOTAL += 1
        _, bad = run_node(node_type, entry, mutate=mutate)
        fired = bad > 0
        if fired:
            CONTROLS_FIRED += 1
        print("  %-52s %s (%d mismatches)"
              % (name, "FIRED" if fired else "*** DEAD ***", bad))

    def to_add(ops):
        return [dict(o, op="add") if o["op"] == "set_if_below" else o for o in ops]

    def to_set(ops):
        return [dict(o, op="set") if o["op"] == "set_if_below" else o for o in ops]

    def flip_gate(ops):
        return [dict(o, thr=-1.0) if o["op"] == "set_if_below" else o for o in ops]

    def rename(ops):
        return [dict(o, w=o["w"] + "_XX") for o in ops]

    def negate(ops):
        return [dict(o, v=-o["v"]) if o["v"] != 0 else o for o in ops]

    def drop_first(ops):
        return ops[1:] if len(ops) > 1 else ops

    has_gate = lambda e: any(
        o["op"] == "set_if_below"
        for m in e["presets"].values() for o in m.get("ops", []))

    nc("wheel hue as a plain add instead of a gated replace",
       "DarkroomLogWheels", to_add, has_gate)
    nc("wheel hue as an unconditional set (gate ignored)",
       "DarkroomLogWheels", to_set, has_gate)
    nc("gate threshold moved so the replace never fires",
       "DarkroomThreeWayColorBalance", flip_gate, has_gate)
    nc("every widget name misspelled (ops silently skipped)",
       "DarkroomToneCurve", rename)
    nc("preset values negated", "DarkroomHueVsHue", negate)
    nc("first op dropped", "DarkroomLumVsSat", drop_first)

    # `got = IMG` for a bypass preset is only a real assertion if the same manual
    # values WOULD have changed the image without it. This proves that directly:
    # reading bypass as "keep the manual grade" must produce a mismatch.
    CONTROLS_TOTAL += 1
    lw = payload["nodes"]["DarkroomLogWheels"]
    cls = NCM["DarkroomLogWheels"]
    defaults = node_defaults(cls)
    bad = 0
    for name, meta in lw["presets"].items():
        if not meta.get("bypass"):
            continue
        for set_name, overrides in manual_sets(defaults):
            manual = dict(defaults)
            manual.update({k: v for k, v in overrides.items() if k in defaults})
            as_manual = dict(manual, preset=CUSTOM)
            if not torch.equal(IMG, render(cls, as_manual)):
                bad += 1
    if bad:
        CONTROLS_FIRED += 1
    print("  %-52s %s (%d mismatches)"
          % ("bypass read as 'keep the manual grade'",
             "FIRED" if bad else "*** DEAD ***", bad))

    print("\n" + "=" * 72)
    print("%d passed, %d failed, %d of %d negative controls fired"
          % (PASS, FAIL, CONTROLS_FIRED, CONTROLS_TOTAL))
    print("=" * 72)
    return 1 if (FAIL or CONTROLS_FIRED != CONTROLS_TOTAL) else 0


if __name__ == "__main__":
    sys.exit(main())
