"""
Offline validation for the Spectral B&W (Ortho/Pan) node — "teeth before trust".
Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per check
and exits nonzero if any check fails.

Loads the node via a synthetic package shim so the node's `from ..utils...`
relative imports resolve WITHOUT triggering the top-level package __init__
(which imports server_routes / ComfyUI runtime bits).

Teeth (per docs/spectral-bw-derivation.md):
  1. ORTHO SIGNATURE (headline)  + NEGATIVE CONTROL (flipped wr<->wb).
  2. TYPE ORDERING               — red-patch gray non-decreasing across types.
  3. GRAYSCALE OUT               — R==G==B everywhere.
  4. NEUTRAL PRESERVED           — grey 0.5 in -> ~grey 0.5 out.
  5. WEIGHTS SANE                — >=0, sum~1, ortho wr<0.05, Pan+ wr is max.
  6. strength                    — 0 -> identity; 0.5 -> partial desaturation.
"""

import os
import sys
import types
import importlib.util

import numpy as np
import torch

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic package shim (mirrors test_halftone.py).
# ---------------------------------------------------------------------------
PKG = "dr_pkg"

parent = types.ModuleType(PKG)
parent.__path__ = [PACK_ROOT]
sys.modules[PKG] = parent

nodes_pkg = types.ModuleType(PKG + ".nodes")
nodes_pkg.__path__ = [os.path.join(PACK_ROOT, "nodes")]
sys.modules[PKG + ".nodes"] = nodes_pkg

utils_pkg = types.ModuleType(PKG + ".utils")
utils_pkg.__path__ = [os.path.join(PACK_ROOT, "utils")]
sys.modules[PKG + ".utils"] = utils_pkg

spec = importlib.util.spec_from_file_location(
    PKG + ".nodes.spectral_bw", os.path.join(PACK_ROOT, "nodes", "spectral_bw.py")
)
spectral_bw = importlib.util.module_from_spec(spec)
sys.modules[PKG + ".nodes.spectral_bw"] = spectral_bw
spec.loader.exec_module(spectral_bw)

SpectralBW = spectral_bw.SpectralBW
_WEIGHTS = spectral_bw._WEIGHTS

# Import the color utils via the shim so the flipped-weight negative control can
# reproduce the runtime math exactly.
color = importlib.import_module(PKG + ".utils.color")
srgb_to_linear = color.srgb_to_linear
linear_to_srgb = color.linear_to_srgb
blend = color.blend

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}, torch = {torch.__version__}")

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}  {detail}")


def to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


def from_tensor(t):
    return t[0].cpu().numpy().astype(np.float32)


def run(img_np, **kw):
    node = SpectralBW()
    out = node.execute(to_tensor(img_np), **kw)[0]
    return from_tensor(out)


def patch(rgb):
    """8x8 solid sRGB patch."""
    a = np.zeros((8, 8, 3), dtype=np.float32)
    a[..., 0], a[..., 1], a[..., 2] = rgb
    return a


def gray_value(out):
    """Mean gray (output is grayscale; use R channel mean)."""
    return float(out[..., 0].mean())


def convert_with_weights(img_np, w, strength=1.0):
    """Reproduce the node's runtime math with an arbitrary weight triple."""
    original = img_np.copy()
    lin = srgb_to_linear(img_np)
    gray = w[0] * lin[..., 0] + w[1] * lin[..., 1] + w[2] * lin[..., 2]
    gray = np.clip(gray, 0.0, None)
    out = np.stack([gray, gray, gray], axis=-1)
    out = linear_to_srgb(np.clip(out, 0.0, 1.0))
    return blend(original, out, strength)


RED = patch((1.0, 0.0, 0.0))
BLUE = patch((0.0, 0.0, 1.0))

# ---------------------------------------------------------------------------
# 1. ORTHO SIGNATURE (headline) + NEGATIVE CONTROL
# ---------------------------------------------------------------------------
print("\n[1] ORTHO SIGNATURE: Orthochromatic renders red DARKER than blue")
red_ortho = gray_value(run(RED, sensitivity="Orthochromatic", strength=1.0))
blue_ortho = gray_value(run(BLUE, sensitivity="Orthochromatic", strength=1.0))
check("ortho: red gray < blue gray (red darker)", red_ortho < blue_ortho,
      f"red={red_ortho:.4f}  blue={blue_ortho:.4f}")

print("    negative control: flipped weights (swap wr<->wb) must make red >= blue -> test FAILS on flip")
wr, wg, wb = _WEIGHTS["Orthochromatic"]
flipped = (wb, wg, wr)
red_flip = gray_value(convert_with_weights(RED, flipped))
blue_flip = gray_value(convert_with_weights(BLUE, flipped))
neg_fired = not (red_flip < blue_flip)   # the signature test FAILS under the flip
check("NEGATIVE CONTROL fired (flipped weights: red >= blue, signature breaks)", neg_fired,
      f"red_flip={red_flip:.4f}  blue_flip={blue_flip:.4f}")

# ---------------------------------------------------------------------------
# 2. TYPE ORDERING — red-patch gray non-decreasing as red sensitivity grows
# ---------------------------------------------------------------------------
print("\n[2] TYPE ORDERING: red-patch gray non-decreasing across "
      "[Blue, Ortho, Orthopan, Pan, Pan+]")
order = ["Blue-sensitive", "Orthochromatic", "Orthopanchromatic", "Panchromatic", "Panchromatic+"]
reds = [gray_value(run(RED, sensitivity=t, strength=1.0)) for t in order]
print(f"    red grays: " + ", ".join(f"{t}={v:.4f}" for t, v in zip(order, reds)))
# Non-decreasing across the red-sensitivity series ortho->pan+ (blue-sensitive sits
# low but is not part of the strictly-increasing red ramp; assert the ramp itself).
ramp = reds[1:]   # ortho, orthopan, pan, pan+
nondec = all(ramp[i + 1] >= ramp[i] - 1e-6 for i in range(len(ramp) - 1))
check("red gray non-decreasing across ortho->pan+", nondec,
      f"ramp={tuple(round(v,4) for v in ramp)}")
margin = reds[order.index("Panchromatic+")] - reds[order.index("Orthochromatic")]
check("Pan+ red gray > Ortho red gray with clear margin (>0.10)", margin > 0.10,
      f"pan+={reds[-1]:.4f}  ortho={reds[1]:.4f}  margin={margin:.4f}")

# ---------------------------------------------------------------------------
# 3. GRAYSCALE OUT — R==G==B everywhere
# ---------------------------------------------------------------------------
print("\n[3] GRAYSCALE OUT: output R==G==B everywhere")
scene = np.random.rand(32, 32, 3).astype(np.float32)
out = run(scene, sensitivity="Panchromatic", strength=1.0)
dmax = float(max(np.abs(out[..., 0] - out[..., 1]).max(),
                 np.abs(out[..., 1] - out[..., 2]).max()))
check("R==G==B within 1e-6", dmax < 1e-6, f"max channel diff={dmax:.2e}")

# ---------------------------------------------------------------------------
# 4. NEUTRAL PRESERVED — grey 0.5 in -> ~grey 0.5 out
# ---------------------------------------------------------------------------
print("\n[4] NEUTRAL PRESERVED: neutral grey 0.5 in -> ~0.5 out (weights sum to 1)")
grey = np.full((8, 8, 3), 0.5, dtype=np.float32)
for t in order:
    g_out = gray_value(run(grey, sensitivity=t, strength=1.0))
    # weights sum to 1 -> linear-light value preserved exactly; the sRGB round-trip
    # (srgb->linear->srgb of a constant) is identity, so out ~ 0.5 for every type.
    check(f"{t}: grey 0.5 -> ~0.5 (within 0.02)", abs(g_out - 0.5) < 0.02,
          f"out={g_out:.4f}")

# ---------------------------------------------------------------------------
# 5. WEIGHTS SANE
# ---------------------------------------------------------------------------
print("\n[5] WEIGHTS SANE: >=0, sum~1, ortho wr<0.05, Pan+ wr is max")
print("    triples:")
for t in order:
    w = _WEIGHTS[t]
    print(f"      {t:18s} wr={w[0]:.6f} wg={w[1]:.6f} wb={w[2]:.6f} sum={sum(w):.6f}")
for t in order:
    w = _WEIGHTS[t]
    check(f"{t}: all >=0 and sum~1", all(x >= 0 for x in w) and abs(sum(w) - 1.0) < 1e-6,
          f"sum={sum(w):.6f}")
ortho_wr = _WEIGHTS["Orthochromatic"][0]
check("ortho wr < 0.05", ortho_wr < 0.05, f"ortho wr={ortho_wr:.6f}")
all_wr = {t: _WEIGHTS[t][0] for t in order}
panplus_wr = _WEIGHTS["Panchromatic+"][0]
check("Pan+ wr is the max wr across types", panplus_wr >= max(all_wr.values()) - 1e-12,
      f"pan+ wr={panplus_wr:.6f}  max={max(all_wr.values()):.6f}")

# ---------------------------------------------------------------------------
# 6. strength: 0 -> identity; 0.5 -> partial desaturation
# ---------------------------------------------------------------------------
print("\n[6] strength: 0 -> identity; 0.5 -> partial desaturation")
colored = patch((0.8, 0.2, 0.2))   # saturated red-ish


def chroma(img):
    """Max-min channel spread = a simple saturation proxy."""
    return float((img.max(axis=-1) - img.min(axis=-1)).mean())


out0 = run(colored, sensitivity="Panchromatic", strength=0.0)
check("strength=0 -> identity (out == input)",
      np.abs(out0 - colored).max() < 1e-6, f"max diff={np.abs(out0 - colored).max():.2e}")

full = run(colored, sensitivity="Panchromatic", strength=1.0)
half = run(colored, sensitivity="Panchromatic", strength=0.5)
c_in, c_full, c_half = chroma(colored), chroma(full), chroma(half)
partial = (c_full < c_half < c_in)
check("strength=0.5 -> chroma between full-B&W and full-color", partial,
      f"chroma in={c_in:.4f}  half={c_half:.4f}  full={c_full:.4f}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
