"""
Film Stock (Color) tone curve: backend maths, JS/Python parity, node wiring and
the pre-1.29 workflow migration.

  * natural cubic spline == scipy CubicSpline(bc_type="natural") inside the
    points, flat outside them;
  * each slider moves only its zone, in its direction, with black and white
    pinned; extreme combinations stay monotonic and in [0, 1];
  * bad point strings fall back to identity;
  * identity curve + zero sliders leaves the film look bitwise unchanged, and a
    non-identity curve changes it;
  * legacy override_* kwargs (1.28 API prompts) are accepted;
  * web/darkroom_tone_curve_math.js, run under Node, gives the same curve as
    Python (so what the editor draws is what is applied), and its migration
    maps a 1.28 widgets_values layout onto the new widgets.

Run: python.exe tools/test_film_tone_curve.py
"""

import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
from aiohttp import web
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = r"F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
for p in (COMFY, os.path.dirname(PACK)):
    if p not in sys.path:
        sys.path.insert(0, p)
import server as _server_mod  # noqa: E402
if getattr(_server_mod.PromptServer, "instance", None) is None:
    _server_mod.PromptServer.instance = types.SimpleNamespace(routes=web.RouteTableDef())
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("darkroom_pack", os.path.join(PACK, "__init__.py"),
                                               submodule_search_locations=[PACK])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["darkroom_pack"] = _pack
_spec.loader.exec_module(_pack)
from darkroom_pack.utils import tone_curve_ops as T  # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


rng = np.random.default_rng(3)
CASES = [
    ("0,0;1,1", 0, 0, 0, 0),
    ("0,0.05;0.25,0.3;0.6,0.7;1,0.95", 0, 0, 0, 0),
    ("0.1,0;0.4,0.55;0.9,1", 40, -30, 20, 60),
    ("0,0;0.5,0.5;1,1", -100, 100, -100, 100),
    ("0,0;0.2,0.35;0.5,0.5;0.8,0.65;1,1", 100, -100, 100, -100),
]
for _ in range(15):
    n = rng.integers(2, 7)
    xs = np.sort(np.concatenate([[0, 1], rng.uniform(0.05, 0.95, n - 2)]))
    ys = np.clip(xs + rng.normal(0, 0.12, n), 0, 1)
    CASES.append((";".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys)),
                  *[float(v) for v in rng.uniform(-100, 100, 4)]))


# malformed point strings (first 9 must fall back to identity) + two valid oddities
BAD_TEXTS = ["", "garbage", "0,0", "0.5,nan;1,1", "1;2", "0.2,;1,1", "0,0;0.5,Infinity;1,1",
             "1_0,0;1,1", "0x1,0;1,1", " 0.5 , 0.6 ;0,0;1,1", "1e-1,2E-1;1,1"]


def backend():
    # spline vs scipy
    for text, *_ in CASES[:8]:
        pts = T.parse_points(text)
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        q = np.linspace(xs[0], xs[-1], 501)
        ref = CubicSpline(xs, ys, bc_type="natural")(q)
        check(f"spline == scipy natural ({len(pts)} pts)", np.abs(T.natural_cubic(pts, q) - ref).max() < 1e-9)
    pts = T.parse_points("0.2,0.1;0.8,0.9")
    check("flat below the first point", abs(float(T.natural_cubic(pts, np.array([0.0]))[0]) - 0.1) < 1e-12)
    check("flat above the last point", abs(float(T.natural_cubic(pts, np.array([1.0]))[0]) - 0.9) < 1e-12)

    ident = T.compose(T.parse_points("0,0;1,1"))
    x = np.linspace(0, 1, T.TABLE_SIZE)
    check("identity compose == x", np.abs(ident - x).max() < 1e-12)
    for name, kw, zone in (("shadows", "shadows", (0.0, 0.5)), ("midtones", "midtones", (0.25, 0.75)),
                           ("highlights", "highlights", (0.5, 1.0))):
        up = T.compose(T.parse_points("0,0;1,1"), **{kw: 60}) - x
        dn = T.compose(T.parse_points("0,0;1,1"), **{kw: -60}) - x
        inside = (x > zone[0] + 0.02) & (x < zone[1] - 0.02)
        outside = (x < zone[0] - 0.02) | (x > zone[1] + 0.02)
        check(f"{name} +60 raises its zone", up[inside].min() > 0)
        check(f"{name} -60 lowers its zone", dn[inside].max() < 0)
        check(f"{name} leaves other tones alone", np.abs(up[outside]).max() < 1e-12 and np.abs(dn[outside]).max() < 1e-12)
        check(f"{name}: black and white pinned", abs(up[0]) < 1e-12 and abs(up[-1]) < 1e-12)
    c = T.compose(T.parse_points("0,0;1,1"), contrast=80) - x
    check("contrast + darkens below mid, brightens above", c[(x > 0.1) & (x < 0.4)].max() < 0 < c[(x > 0.6) & (x < 0.9)].min())
    for text, *s in CASES:
        y = T.compose(T.parse_points(text), *s)
        check("monotonic, finite, in [0,1]", np.isfinite(y).all() and (np.diff(y) >= 0).all() and y.min() >= 0 and y.max() <= 1)
    for bad in BAD_TEXTS[:9]:
        check(f"bad points {bad!r} -> identity", T.parse_points(bad) == [(0.0, 0.0), (1.0, 1.0)])


def node():
    N = _pack.NODE_CLASS_MAPPINGS["DarkroomFilmStockColor"]()
    img = torch.rand(1, 32, 48, 3)
    for stock in ("Neg / Kodak Portra 400", "Neg / Cinestill 800T"):     # baked C1 + hand-authored
        base = N.execute(image=img, film_stock=stock, strength=1.0)[0]
        same = N.execute(image=img, film_stock=stock, strength=1.0, curve_points="0,0;1,1",
                         contrast=0.0, shadows=0.0, midtones=0.0, highlights=0.0)[0]
        check(f"{stock}: identity curve leaves the look bitwise", torch.equal(base, same))
        curved = N.execute(image=img, film_stock=stock, strength=1.0, curve_points="0,0;0.5,0.65;1,1")[0]
        check(f"{stock}: a curve changes the look", (curved - base).abs().max() > 0.01)
        legacy = N.execute(image=img, film_stock=stock, strength=1.0,
                           override_toe=1.2, override_shoulder=-1.0, override_gamma=-1.0)[0]
        # ComfyUI drops unknown widget inputs before execute(); only linked ones can arrive
        check(f"{stock}: linked legacy override inputs do not raise", torch.equal(legacy, base))
        expect = T.apply_table(base, T.compose(T.parse_points("0,0;0.5,0.65;1,1")))
        check(f"{stock}: curve applied after the look", (curved - expect).abs().max() < 1e-6)
        # review #1: tiny / thin images must not crash the recovery blur
        for shape in ((1, 3, 400, 3), (1, 32, 4096, 3), (1, 1, 1, 3), (1, 7, 5, 3)):
            try:
                o = N.execute(image=torch.rand(*shape), film_stock=stock, strength=1.0)[0]
                ok = tuple(o.shape) == shape and bool(torch.isfinite(o).all())
            except Exception:
                ok = False
            check(f"{stock}: image {shape[1]}x{shape[2]} runs", ok)
        # review #8: alpha survives
        rgba = torch.rand(1, 16, 16, 4)
        o = N.execute(image=rgba, film_stock=stock, strength=0.5)[0]
        check(f"{stock}: RGBA keeps 4 channels and alpha",
              o.shape == rgba.shape and torch.equal(o[..., 3], rgba[..., 3]))


def js_parity():
    mjs = os.path.join(PACK, "web", "darkroom_tone_curve_math.js").replace("\\", "/")
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "run.mjs")
        open(script, "w", encoding="utf-8").write(
            f'import * as M from "file:///{mjs}";\n'
            'const cases = JSON.parse(process.argv[2]);\n'
            'const out = cases.map(([t, c, s, m, h]) => M.compose(M.parsePoints(t), c, s, m, h, 1025));\n'
            'const mig = [M.migrateFilmStockValues(["Neg / Kodak Portra 400", 0.8, -1, 2.0, -1]),\n'
            '             M.migrateFilmStockValues(["Neg / Kodak Portra 400", 1, true, "0,0;1,1", 0, 0, 0, 0])];\n'
            'const bad = JSON.parse(process.argv[3]).map((t) => M.parsePoints(t));\n'
            'console.log(JSON.stringify({out, mig, bad}));\n')
        r = subprocess.run(["node", script, json.dumps(CASES), json.dumps(BAD_TEXTS)],
                           capture_output=True, text=True)
        if not check("node ran the JS maths", r.returncode == 0, r.stderr[-400:]):
            return
        res = json.loads(r.stdout)
    worst = max(float(np.abs(np.array(js) - T.compose(T.parse_points(c[0]), *c[1:], size=1025)).max())
                for js, c in zip(res["out"], CASES))
    check(f"JS curve == Python curve on {len(CASES)} cases", worst < 1e-9, f"max diff {worst:.2e}")
    check("migration maps the 1.28 layout",
          res["mig"][0] == ["Neg / Kodak Portra 400", 0.8, True, "0,0;1,1", 0, 0, 0, 0], str(res["mig"][0]))
    check("migration leaves the current layout alone", res["mig"][1] is None)
    for t, js in zip(BAD_TEXTS, res["bad"]):
        py = [list(p) for p in T.parse_points(t)]
        check(f"parse parity {t!r}", np.allclose(py, js), f"py {py} js {js}")
    # review #2: the editor draws compose(pts, 0,0,0,0) (see darkroom_freeform_curve.js),
    # so a point dragged below its neighbour draws as the plateau that is applied
    y = T.compose(T.parse_points("0,0;0.1,0.4;0.2,0.1;1,1"))
    check("non-monotone points apply as a guarded plateau", bool((np.diff(y) >= 0).all()) and y[1024] > 0.39)


if __name__ == "__main__":
    for n, f in (("backend", backend), ("node", node), ("js parity + migration", js_parity)):
        print(f"[{n}]")
        f()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
