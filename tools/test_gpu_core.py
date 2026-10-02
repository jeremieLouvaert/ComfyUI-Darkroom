"""
Golden gate for the GPU colour core (1.28.0).

The torch port must reproduce the numpy pipeline it replaces to within
0.5/255 per channel (decisions.md 2026-10-02: invisible in 8-bit). This file
captures the numpy outputs ONCE, from the pre-port code, and then gates every
later run against them:

    python.exe tools/test_gpu_core.py --capture   # on the pre-port commit
    python.exe tools/test_gpu_core.py             # after: compare

Cases per node: defaults, every float/int/bool moved on its own, every combo
value on its own, and everything moved at once, each on three inputs (seeded
noise, a ramp that hits every transfer-function threshold exactly, a real
photo crop). A case that raised on the old code must raise on the new code.
Also checked: no NaN/inf, shape and dtype, and a batch of two equals the two
singles.

Goldens are local (tools/golden/ is git-ignored) and tagged with the commit
they were captured from.

Run: python.exe tools/test_gpu_core.py [--capture] [--only NodeKey,...]
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
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

import folder_paths  # noqa: E402
import server as _server_mod  # noqa: E402
if getattr(_server_mod.PromptServer, "instance", None) is None:
    _server_mod.PromptServer.instance = types.SimpleNamespace(routes=web.RouteTableDef())

# LUT Apply reads its .cube through the allowed-folder check: give it a sandbox.
SANDBOX = os.path.realpath(tempfile.mkdtemp(prefix="darkroom_gpu_core_"))
for k in ("input", "output", "temp", "models", "user"):
    os.makedirs(os.path.join(SANDBOX, k), exist_ok=True)
folder_paths.base_path = SANDBOX
folder_paths.input_directory = os.path.join(SANDBOX, "input")
folder_paths.output_directory = os.path.join(SANDBOX, "output")
folder_paths.temp_directory = os.path.join(SANDBOX, "temp")
folder_paths.models_dir = os.path.join(SANDBOX, "models")
folder_paths.user_directory = os.path.join(SANDBOX, "user")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "darkroom_pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["darkroom_pack"] = _pack
_spec.loader.exec_module(_pack)
from darkroom_pack.utils.lut import write_cube_file  # noqa: E402

NCM = _pack.NODE_CLASS_MAPPINGS
TOL = 0.5 / 255
GOLDEN_DIR = os.path.join(HERE, "golden")
GOLDEN = os.path.join(GOLDEN_DIR, "gpu_core.npz")
META = os.path.join(GOLDEN_DIR, "gpu_core.json")
SIZE = (64, 64)

NODES = [
    "DarkroomFilmStockColor", "DarkroomPrintStock", "DarkroomCrossProcess", "DarkroomFilmStockBW",
    "DarkroomHSLSelective", "DarkroomHueVsHue", "DarkroomHueVsSat", "DarkroomColorQualifier",
    "DarkroomColorWarper", "DarkroomSkinToneUniformity",
    "DarkroomToneCurve", "DarkroomLiftGammaGain", "DarkroomLogWheels", "DarkroomThreeWayColorBalance",
    "DarkroomVibrance", "DarkroomLumVsSat", "DarkroomSatVsSat", "DarkroomOkLabColor",
    "DarkroomExposureTone", "DarkroomWhiteBalance", "DarkroomACESTonemap", "DarkroomColorSpaceTransform",
    "DarkroomLUTApply", "DarkroomSpectralFilmStock",
]

# A non-identity 17^3 LUT (gamma + channel twist) so trilinear is exercised.
CUBE = os.path.join(folder_paths.input_directory, "golden.cube")
_g = np.linspace(0, 1, 17, dtype=np.float32)
_r, _gg, _b = np.meshgrid(_g, _g, _g, indexing="ij")
write_cube_file(CUBE, np.stack([_r ** 0.8, 0.7 * _gg + 0.3 * _b, np.sqrt(_b) * 0.9 + 0.05 * _r],
                               axis=-1).astype(np.float32), 17, title="golden")


# --- inputs ---------------------------------------------------------------------
def _inputs():
    h, w = SIZE
    rng = np.random.default_rng(1234)
    noise = rng.random((h, w, 3), dtype=np.float32)

    # every transfer-function threshold hit exactly, plus 0/1 and a dense ramp,
    # spread across hues so HSL branches all see them
    special = np.array([0.0, 1.0, 0.04045, 0.0031308, 0.18, 0.5, 1e-6, 0.999999], np.float32)
    ramp = np.linspace(0, 1, h * w, dtype=np.float32)
    ramp[: special.size] = special
    ramp = ramp.reshape(h, w)
    hue = np.linspace(0, 6, w, endpoint=False, dtype=np.float32)[None, :].repeat(h, 0)
    c = np.stack([np.clip(np.abs(((hue + o) % 6) - 3) - 1, 0, 1) for o in (0, 4, 2)], -1)
    ramp_rgb = (ramp[..., None] * (0.35 + 0.65 * c)).astype(np.float32)
    ramp_rgb[0, : special.size] = special[:, None]  # neutral thresholds, row 0

    photo = None
    jpg = os.path.join(PACK, "test_data", "test.jpg")
    if os.path.isfile(jpg):
        from PIL import Image
        im = Image.open(jpg).convert("RGB")
        s = min(im.size)
        im = im.crop((0, 0, s, s)).resize((w, h), Image.BILINEAR)
        photo = np.asarray(im, dtype=np.float32) / 255.0
    else:
        photo = rng.random((h, w, 3), dtype=np.float32) ** 2.2
    return {"noise": noise, "ramp": ramp_rgb, "photo": photo}


INPUTS = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in _inputs().items()}


# --- parameter cases ----------------------------------------------------------------
def _cases(key):
    it = NCM[key].INPUT_TYPES()
    spec = {**it.get("required", {}), **it.get("optional", {})}
    base, moves = {}, []
    image_arg = None
    for name, s in spec.items():
        kind, opts = s[0], (s[1] if len(s) > 1 else {})
        if kind == "IMAGE":
            image_arg = name
            continue
        if isinstance(kind, (list, tuple)):
            base[name] = opts.get("default", kind[0])
            for v in kind:
                if v != base[name]:
                    moves.append((f"{name}={v}", {name: v}))
        elif kind in ("FLOAT", "INT"):
            d = opts.get("default", 0)
            lo, hi = opts.get("min", d - 1), opts.get("max", d + 1)
            base[name] = d
            v = d + 0.3 * (hi - d) if hi > d else d - 0.3 * (d - lo)
            if kind == "INT":
                v = int(round(v))
            if v != d:
                moves.append((f"{name}={v:.4g}", {name: v}))
        elif kind == "BOOLEAN":
            base[name] = opts.get("default", False)
            moves.append((f"{name}={not base[name]}", {name: not base[name]}))
        elif kind == "STRING":
            base[name] = CUBE if name == "lut_file" else opts.get("default", "")
        else:
            raise RuntimeError(f"{key}: unhandled input type {kind!r} for {name}")
    # everything numeric/boolean moved at once (combos stay at their default)
    together = {k: v for _label, m in moves for k, v in m.items()
                if not isinstance(spec[k][0], (list, tuple))}
    cases = [("defaults", {})] + moves + ([("all_floats_moved", together)] if together else [])
    return image_arg, base, cases


def _run(key, image_arg, kwargs, image):
    """Every tensor the node returns (image plus any matte/preview outputs)."""
    node = NCM[key]()
    out = getattr(node, NCM[key].FUNCTION)(**{image_arg: image, **kwargs})
    return [o for o in out if torch.is_tensor(o)]


def _okey(cid, k):
    return cid if k == 0 else f"{cid}#{k}"


def _commit():
    try:
        return subprocess.run(["git", "-C", PACK, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def capture(only):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    arrays, meta = {}, {"commit": _commit(), "tol": TOL, "cases": {}}
    if os.path.isfile(GOLDEN) and only:
        old = np.load(GOLDEN)
        arrays = {k: old[k] for k in old.files}
        meta = json.load(open(META, encoding="utf-8"))
    for key in only or NODES:
        image_arg, base, cases = _cases(key)
        for label, change in cases:
            for iname, img in INPUTS.items():
                cid = f"{key}|{label}|{iname}"
                try:
                    outs = _run(key, image_arg, {**base, **change}, img)
                    for k, o in enumerate(outs):
                        arrays[_okey(cid, k)] = o.numpy()
                    meta["cases"][cid] = "ok"
                    meta.setdefault("outputs", {})[cid] = len(outs)
                except Exception as e:
                    meta["cases"][cid] = f"raised {type(e).__name__}"
        print(f"captured {key}: {len(cases)} param cases x {len(INPUTS)} inputs")
    np.savez_compressed(GOLDEN, **arrays)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"golden written from commit {meta['commit']}: {len(meta['cases'])} cases")


def compare(only):
    gold = np.load(GOLDEN)
    meta = json.load(open(META, encoding="utf-8"))
    print(f"golden from commit {meta['commit']}, gate max|diff| <= {TOL:.6f}")
    fails, total, worst = 0, 0, {}
    for key in only or NODES:
        image_arg, base, cases = _cases(key)
        node_worst = 0.0
        for label, change in cases:
            for iname, img in INPUTS.items():
                cid = f"{key}|{label}|{iname}"
                if cid not in meta["cases"]:
                    continue
                total += 1
                expect = meta["cases"][cid]
                try:
                    outs = _run(key, image_arg, {**base, **change}, img)
                except Exception as e:
                    if expect == "ok":
                        fails += 1
                        print(f"  FAIL {cid}: raised {type(e).__name__}: {e}")
                    continue
                if expect != "ok":
                    fails += 1
                    print(f"  FAIL {cid}: golden {expect}, new code returned")
                    continue
                n_out = meta.get("outputs", {}).get(cid, 1)
                if len(outs) != n_out:
                    fails += 1
                    print(f"  FAIL {cid}: {len(outs)} tensor outputs, golden has {n_out}")
                    continue
                for k, out in enumerate(outs):
                    g = gold[_okey(cid, k)]
                    o = out.detach().cpu().numpy()
                    tag = _okey(cid, k)
                    if o.shape != g.shape or out.dtype != torch.float32 or out.device.type != "cpu":
                        fails += 1
                        print(f"  FAIL {tag}: shape/dtype/device {o.shape}/{out.dtype}/{out.device} vs {g.shape}")
                        continue
                    if not np.isfinite(o).all():
                        fails += 1
                        print(f"  FAIL {tag}: NaN/inf")
                        continue
                    d = float(np.abs(o - g).max())
                    node_worst = max(node_worst, d)
                    if d > TOL:
                        fails += 1
                        print(f"  FAIL {tag}: max|diff| {d:.6f} ({d * 255:.2f}/255)")
        # batch of two == two singles
        try:
            pair = torch.cat([INPUTS["noise"], INPUTS["photo"]], 0)
            b = _run(key, image_arg, base, pair)[0]
            s = torch.cat([_run(key, image_arg, base, INPUTS["noise"])[0],
                           _run(key, image_arg, base, INPUTS["photo"])[0]], 0)
            total += 1
            if not torch.allclose(b, s, atol=1e-6, rtol=0):
                fails += 1
                print(f"  FAIL {key}: batch of 2 != two singles ({(b - s).abs().max():.2e})")
        except Exception as e:
            fails += 1
            print(f"  FAIL {key}: batch run raised {type(e).__name__}: {e}")
        worst[key] = node_worst
    for key, d in worst.items():
        print(f"  {key:34s} worst {d * 255:.3f}/255")
    print(f"\n{total - fails} passed, {fails} failed")
    return fails


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    try:
        if "--capture" in sys.argv:
            capture(only)
            rc = 0
        else:
            rc = 1 if compare(only) else 0
    finally:
        shutil.rmtree(SANDBOX, ignore_errors=True)
    sys.exit(rc)
