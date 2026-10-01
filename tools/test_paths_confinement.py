"""
Pins Darkroom's file-access confinement (1.27.0 security fix).

Every path a client can send -- node widgets through /prompt and the folder
picker routes -- must stay inside the allowed roots (utils/paths.py). The test
builds a sandbox ComfyUI layout (input/output/temp/models/user) next to an
"outside" folder holding a secret, points folder_paths at it, and attacks the
real route handlers and node classes:

  traversal, absolute paths, the output2 prefix trick, case variants, a
  junction out of output/, NUL bytes, the extras file (and one planted under
  user/, where ComfyUI's /userdata API can write), filename traversal in both
  export nodes, out-of-root pins, folder creation outside the roots, and EXIF
  make/model steering the DCP lookup.

The route and node checks are behavioural only, so the same file run against
the pre-1.27 code is the negative control: it must FAIL there.

Run: python.exe tools/test_paths_confinement.py
"""

import asyncio
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
    _server_mod.PromptServer.instance = types.SimpleNamespace(
        routes=web.RouteTableDef())

# --- sandbox layout, before the pack is imported -------------------------------
SANDBOX = os.path.realpath(tempfile.mkdtemp(prefix="darkroom_confine_"))
BASE = os.path.join(SANDBOX, "ComfyUI")
DIRS = {k: os.path.join(BASE, k) for k in ("input", "output", "temp", "models", "user")}
OUTSIDE = os.path.join(SANDBOX, "outside")
OUTPUT2 = os.path.join(BASE, "output2")          # prefix-trick sibling of output
for d in [*DIRS.values(), OUTSIDE, OUTPUT2, os.path.join(DIRS["user"], "default")]:
    os.makedirs(d, exist_ok=True)

folder_paths.base_path = BASE
folder_paths.input_directory = DIRS["input"]
folder_paths.output_directory = DIRS["output"]
folder_paths.temp_directory = DIRS["temp"]
folder_paths.models_dir = DIRS["models"]
folder_paths.user_directory = DIRS["user"]

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "darkroom_pack", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
_pack = importlib.util.module_from_spec(_spec)
sys.modules["darkroom_pack"] = _pack
_spec.loader.exec_module(_pack)

import darkroom_pack.server_routes as routes  # noqa: E402
from darkroom_pack.utils.lut import write_cube_file  # noqa: E402
from darkroom_pack.utils.dcp import _body_folder  # noqa: E402
try:
    from darkroom_pack.utils import paths as P  # noqa: E402
except ImportError:
    P = None  # pre-1.27 code: unit section skipped, behavioural checks still run

NCM = _pack.NODE_CLASS_MAPPINGS

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("  FAIL %s%s" % (name, (" | " + detail) if detail else ""))
    return ok


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def call_route(handler, query=None, body=None):
    req = types.SimpleNamespace(query=query or {})

    async def _json():
        return body or {}
    req.json = _json
    resp = asyncio.run(handler(req))
    return resp.status, json.loads(resp.text)


def write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def identity_cube(path):
    g = np.linspace(0, 1, 2, dtype=np.float32)
    lut = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)  # (r,g,b,3)
    write_cube_file(path, lut, 2, title="t")


SECRET = os.path.join(OUTSIDE, "secret.cube")
identity_cube(SECRET)
write(os.path.join(OUTSIDE, "sub", "keep.txt"))
INSIDE_CUBE = os.path.join(DIRS["input"], "ok.cube")
identity_cube(INSIDE_CUBE)
EXTRAS = os.path.join(BASE, "darkroom_allowed_folders.json")
IMG = torch.rand(1, 8, 8, 3)


# --- 1. utils.paths units (skipped on pre-1.27 code) ---------------------------
def test_paths_module():
    if P is None:
        print("  (utils.paths absent: pre-1.27 code, unit section skipped)")
        return
    ra = P.resolve_allowed
    check("read: file in input", ra(INSIDE_CUBE, "read") == os.path.realpath(INSIDE_CUBE))
    check("read: relative -> input", ra("ok.cube", "read") == os.path.realpath(INSIDE_CUBE))
    check("write: relative -> output",
          ra("luts", "write") == os.path.realpath(os.path.join(DIRS["output"], "luts")))
    check("read: traversal out of output refused",
          raises(lambda: ra(os.path.join(DIRS["output"], "..", "..", "outside", "secret.cube"), "read"),
                 P.PathNotAllowed))
    check("read: absolute outside refused", raises(lambda: ra(SECRET, "read"), P.PathNotAllowed))
    check("read: system file refused", raises(lambda: ra(r"C:\Windows\win.ini", "read"), P.PathNotAllowed))
    check("read: output2 prefix trick refused", raises(lambda: ra(OUTPUT2, "read"), P.PathNotAllowed))
    check("read: case variant of output allowed", P.is_allowed(DIRS["output"].upper(), "read"))
    check("read: user dir refused", raises(lambda: ra(DIRS["user"], "read"), P.PathNotAllowed))
    check("read: ComfyUI root refused", raises(lambda: ra(BASE, "read"), P.PathNotAllowed))
    check("read: NUL byte refused", raises(lambda: ra(INSIDE_CUBE + "\x00.txt", "read"), P.PathNotAllowed))
    check("write: input refused", raises(lambda: ra(DIRS["input"], "write"), P.PathNotAllowed))
    check("write: models refused", raises(lambda: ra(DIRS["models"], "write"), P.PathNotAllowed))

    # junction inside output pointing outside (no admin needed for /J)
    jn = os.path.join(DIRS["output"], "jn")
    r = subprocess.run(["cmd", "/c", "mklink", "/J", jn, OUTSIDE], capture_output=True, text=True)
    if check("junction created", r.returncode == 0, r.stderr.strip()):
        check("read: through junction refused",
              raises(lambda: ra(os.path.join(jn, "secret.cube"), "read"), P.PathNotAllowed))
        check("write: into junction refused", raises(lambda: ra(jn, "write"), P.PathNotAllowed))
        os.rmdir(jn)

    # extras file adds a root; removing it takes the root away again
    with open(EXTRAS, "w", encoding="utf-8") as f:
        json.dump([OUTSIDE], f)
    check("extras: outside allowed when listed", P.is_allowed(SECRET, "read"))
    check("extras: also a write root", P.is_allowed(OUTSIDE, "write"))
    os.remove(EXTRAS)
    check("extras: refused again after removal", not P.is_allowed(SECRET, "read"))
    with open(EXTRAS, "w", encoding="utf-8") as f:
        f.write("{not json")
    check("extras: malformed file allows nothing extra", not P.is_allowed(SECRET, "read"))
    os.remove(EXTRAS)
    planted = os.path.join(DIRS["user"], "default", "darkroom_allowed_folders.json")
    with open(planted, "w", encoding="utf-8") as f:
        json.dump([OUTSIDE], f)
    check("extras: copy planted under user/ ignored", not P.is_allowed(SECRET, "read"))
    os.remove(planted)

    # Network and device paths are refused WITHOUT being opened: before the
    # lexical-first check, realpath() contacted \\host\share over SMB (leaking
    # the NTLM hash) and stalled ~21 s on an unreachable host.
    import time
    for unc in [r"\\10.255.255.1\share\x.cube", "//10.255.255.2/s/x.cube",
                r"\\.\pipe\darkroom_probe", r"\\?\UNC\10.255.255.3\s\x"]:
        t0 = time.perf_counter()
        refused = not P.is_allowed(unc, "read")
        dt = time.perf_counter() - t0
        check(f"network/device path refused fast: {unc}", refused and dt < 1.0, f"{dt:.2f}s")

    # Mapped network drive as a root: realpath(Z:\x) is \\host\share\x, and the
    # UNC form the server hands back must be accepted again, while a sibling
    # share on the same host stays refused. Simulated by patching realpath.
    share = "\\\\localhost\\C$\\" + OUTSIDE[3:]
    orig_realpath = P.os.path.realpath

    def fake_realpath(p, *a, **k):
        if os.path.normcase(p).startswith(os.path.normcase("Z:\\out")):
            return share + p[len("Z:\\out"):]
        return orig_realpath(p, *a, **k)
    P.os.path.realpath = fake_realpath
    P._roots_cache.clear()
    with open(EXTRAS, "w", encoding="utf-8") as f:
        json.dump(["Z:\\out"], f)
    try:
        got = P.resolve_allowed("Z:\\out\\luts", "write")
        check("mapped drive: Z: path resolves to its share", got == share + "\\luts", got)
        check("mapped drive: share form handed back is accepted",
              P.is_allowed(share + "\\luts\\g.cube", "write"))
        check("mapped drive: other path on the same host refused",
              not P.is_allowed("\\\\localhost\\C$\\Windows", "read"))
    finally:
        P.os.path.realpath = orig_realpath
        os.remove(EXTRAS)
        P._roots_cache.clear()

    sf = P.safe_filename
    for raw, want in [("../x", "x"), (r"C:\evil\x", "x"), ("a:b", "a_b"), ("..", "d"),
                      ("", "d"), ("  ", "d"), ("CON", "_CON"), ("nul.cube", "_nul.cube"),
                      ("ok_name", "ok_name"), ("x. ", "x"), ("CONOUT$", "_CONOUT$"),
                      ("COM¹.cube", "_COM¹.cube")]:
        check(f"safe_filename({raw!r})", sf(raw, "d") == want, repr(sf(raw, "d")))


# --- 2. routes ------------------------------------------------------------------
def test_routes():
    st, d = call_route(routes.darkroom_list_dir, {"path": OUTSIDE})
    check("list_dir outside: no listing", not d.get("subdirs") and not d.get("files"), str(d)[:200])
    check("list_dir outside: error returned", bool(d.get("error")))
    _st, d_missing = call_route(routes.darkroom_list_dir, {"path": os.path.join(OUTSIDE, "nope")})
    check("list_dir outside: existing and missing paths answer the same",
          d.get("error") == d_missing.get("error") and d.get("path") == d_missing.get("path"))
    _st, d = call_route(routes.darkroom_list_dir,
                        {"path": os.path.join(DIRS["output"], "..", "..", "outside")})
    check("list_dir traversal: no listing", not d.get("subdirs"))
    _st, d = call_route(routes.darkroom_list_dir, {"path": SECRET, "extensions": ".cube"})
    check("list_dir on outside file: parent not listed", not d.get("files"))

    _st, d = call_route(routes.darkroom_list_dir, {"path": DIRS["output"]})
    roots = [r["path"] for r in d.get("roots", [])]
    check("list_dir roots: no drives or home",
          not any(len(r.rstrip("/")) <= 2 or r.rstrip("/").lower() == os.path.expanduser("~").replace("\\", "/").lower()
                  for r in roots), str(roots))
    check("list_dir at root: Up disabled", d.get("parent", "x") == "", repr(d.get("parent")))
    _st, d = call_route(routes.darkroom_list_dir, {"path": DIRS["input"], "scope": "write"})
    check("list_dir write scope: input refused", bool(d.get("error")) and not d.get("subdirs"))

    st, _d = call_route(routes.darkroom_mkdir, body={"path": os.path.join(OUTSIDE, "made")})
    check("mkdir outside refused", not os.path.exists(os.path.join(OUTSIDE, "made")) and st >= 400)
    st, _d = call_route(routes.darkroom_mkdir, body={"path": os.path.join(DIRS["input"], "made")})
    check("mkdir in input refused", not os.path.exists(os.path.join(DIRS["input"], "made")))
    st, _d = call_route(routes.darkroom_mkdir, body={"path": os.path.join(DIRS["output"], "a", "b", "c")})
    check("mkdir nested missing parents refused", not os.path.exists(os.path.join(DIRS["output"], "a")))
    st, d = call_route(routes.darkroom_mkdir, body={"path": os.path.join(DIRS["output"], "made")})
    check("mkdir in output allowed", st == 200 and os.path.isdir(os.path.join(DIRS["output"], "made")), str(d))

    st, _d = call_route(routes.darkroom_pins_add, body={"path": OUTSIDE})
    check("pin outside refused", st >= 400)
    pins_file = routes._pins_file()
    os.makedirs(os.path.dirname(pins_file), exist_ok=True)
    with open(pins_file, "w", encoding="utf-8") as f:
        json.dump([{"name": "planted", "path": OUTSIDE.replace("\\", "/")},
                   {"name": "ok", "path": DIRS["output"].replace("\\", "/")}], f)
    _st, d = call_route(routes.darkroom_pins_get)
    names = [p["name"] for p in d.get("pins", [])]
    check("pins GET hides out-of-root pins", names == ["ok"], str(names))


# --- 3. nodes --------------------------------------------------------------------
def test_nodes():
    lut_apply = NCM["DarkroomLUTApply"]()
    check("LUT Apply: outside .cube refused",
          raises(lambda: lut_apply.execute(IMG, SECRET, 1.0), ValueError))
    check("LUT Apply: inside .cube works",
          not raises(lambda: lut_apply.execute(IMG, INSIDE_CUBE, 1.0)))

    export = NCM["DarkroomLUTExport"]()
    lattice = torch.rand(1, 4, 2, 3)  # size 2 -> 4x2 lattice
    check("LUT Export: outside directory refused",
          raises(lambda: export.execute(lattice, 2, "x", "t", OUTSIDE), ValueError)
          and not os.path.exists(os.path.join(OUTSIDE, "x.cube")))
    try:
        (out,) = export.execute(lattice, 2, os.path.join("..", "..", "..", "outside", "evil"), "t", "")
    except Exception as e:
        out = f"raised {e}"
    check("LUT Export: filename traversal stays in output",
          not os.path.exists(os.path.join(OUTSIDE, "evil.cube"))
          and os.path.realpath(str(out)).startswith(os.path.realpath(DIRS["output"])), str(out))

    raw = NCM["DarkroomRAWLoad"]()
    secret_raw = os.path.join(OUTSIDE, "x.raf")
    write(secret_raw)
    check("RAW Load: outside path refused before reading",
          raises(lambda: raw.execute(secret_raw, "Auto", "Linear sRGB", "As shot",
                                     "Rebuild (default)", True, "sRGB display"), ValueError))

    cmyk = NCM["DarkroomCMYKExportTIFF"]
    profiles = cmyk.INPUT_TYPES()["required"]["target_profile"][0]
    if profiles and not profiles[0].startswith("("):
        node = cmyk()
        check("CMYK Export: outside directory refused",
              raises(lambda: node.execute(IMG, profiles[0], "perceptual", "p", OUTSIDE, 300), ValueError)
              and not any(f.endswith(".tif") for f in os.listdir(OUTSIDE)))
        try:
            (out,) = node.execute(IMG, profiles[0], "perceptual",
                                  os.path.join("..", "..", "..", "outside", "evil"), "", 300)
        except Exception as e:
            out = f"raised {e}"
        check("CMYK Export: prefix traversal stays in output",
              not any(f.endswith(".tif") for f in os.listdir(OUTSIDE))
              and os.path.realpath(str(out)).startswith(os.path.realpath(DIRS["output"])), str(out))
    else:
        print("  (no CMYK profile on this machine: CMYK checks skipped)")

    for make, model in [("..", "../../Windows"), ("Evil\\..\\..", "x"), ("C:", "\\Windows")]:
        b = _body_folder(make, model)
        check(f"DCP body folder sanitised {make!r}/{model!r}",
              not any(t in b for t in ("..", "/", "\\", ":")), repr(b))
    check("DCP body folder: real body unchanged",
          _body_folder("FUJIFILM", "X-T5") == "FUJIFILM X-T5")


if __name__ == "__main__":
    try:
        for name, fn in [("utils.paths", test_paths_module), ("routes", test_routes),
                         ("nodes", test_nodes)]:
            print(f"[{name}]")
            fn()
    finally:
        shutil.rmtree(SANDBOX, ignore_errors=True)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
