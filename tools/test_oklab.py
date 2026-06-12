"""
Offline validation for the OkLab Color node — "teeth before trust".
Run with the ComfyUI embedded python. No pytest; prints PASS/FAIL per check
and exits nonzero if any check fails.

Checks (per docs/oklab-color-derivation.md):
  1. ORACLE      — agree with colour-science XYZ_to_Oklab(RGB_to_XYZ(lin)).
  2. ROUND-TRIP  — oklab forward/inverse + full sRGB pipeline round-trip.
  3. REFERENCE   — white/black/18%-grey reference points.
  4. NEG-CONTROL — a sign-flipped M2 must FAIL the oracle check (proves teeth).
  5. INVARIANTS  — chroma scale holds L,h; hue rotate holds L,C; tone holds C,h.
"""

import os
import sys

# Pack root on sys.path so `from utils.colorspace import ...` resolves.
PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK_ROOT)

import numpy as np
import colour

from utils.colorspace import (
    linear_srgb_to_oklab,
    oklab_to_linear_srgb,
    oklab_to_oklch,
    oklch_to_oklab,
    _OKLAB_M1,
    _OKLAB_M2,
    _apply_matrix,
)
from utils.color import srgb_to_linear, linear_to_srgb

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


rng = np.random.default_rng(42)

# Random linear sRGB samples + sRGB primaries/secondaries (as linear).
N = 400
rand = rng.random((N, 3)).astype(np.float32)
prim = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],          # primaries
    [1, 1, 0], [0, 1, 1], [1, 0, 1],          # secondaries
    [1, 1, 1], [0, 0, 0], [0.18, 0.18, 0.18],  # white/black/grey
], dtype=np.float32)
lin = np.concatenate([rand, prim], axis=0)        # (N+9, 3)
lin_img = lin.reshape(1, -1, 3)                    # (H,W,3) for our funcs


def oracle_oklab(lin_samples):
    """colour-science oracle: linear sRGB -> XYZ -> Oklab."""
    xyz = colour.RGB_to_XYZ(lin_samples.astype(np.float64), 'sRGB',
                            apply_cctf_decoding=False)
    return colour.XYZ_to_Oklab(xyz)


# ---------------------------------------------------------------------------
print("\n[1] ORACLE agreement vs colour-science 0.4.7")
print("    call: colour.XYZ_to_Oklab(colour.RGB_to_XYZ(lin, 'sRGB', apply_cctf_decoding=False))")
ours = linear_srgb_to_oklab(lin_img).reshape(-1, 3).astype(np.float64)
ref = oracle_oklab(lin)
diff1 = np.max(np.abs(ours - ref))
# Tolerance is 2e-4, not 1e-4. The residual (~1.14e-4) is genuine matrix-rounding
# slack, not a bug: Ottosson published a pre-baked, 10-digit-rounded sRGB-linear->LMS
# matrix (our M1), while colour-science composes sRGB->XYZ->LMS at full precision.
# Their effective sRGB->LMS matrices differ by up to ~2.8e-4 per element. We use the
# exact frozen Ottosson constants by spec; this is the "~1e-4 matrix-rounding slack"
# the derivation doc anticipated, just slightly larger than 1e-4. The negative control
# (test 4) fires at ~3.96, so the check retains full teeth.
check("oracle max abs diff < 2e-4 (Ottosson-rounding slack)", diff1 < 2e-4,
      f"max|d|={diff1:.3e}")


# ---------------------------------------------------------------------------
print("\n[2] ROUND-TRIP identity")
back = oklab_to_linear_srgb(linear_srgb_to_oklab(lin_img))
rt_lin = np.max(np.abs(back - lin_img))
check("linear -> oklab -> linear  max abs err < 1e-5", rt_lin < 1e-5,
      f"max|d|={rt_lin:.3e}")

srgb = rng.random((1, 500, 3)).astype(np.float32)
rt_srgb = linear_to_srgb(
    np.clip(oklab_to_linear_srgb(linear_srgb_to_oklab(srgb_to_linear(srgb))), 0, 1)
)
rt_srgb_err = np.max(np.abs(rt_srgb - srgb))
check("sRGB -> lin -> oklab -> lin -> sRGB  max abs err < 1e-4", rt_srgb_err < 1e-4,
      f"max|d|={rt_srgb_err:.3e}")


# ---------------------------------------------------------------------------
print("\n[3] REFERENCE points")
white = linear_srgb_to_oklab(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))[0, 0]
check("linear white -> L~1.0", abs(white[0] - 1.0) < 1e-3, f"L={white[0]:.6f}")
check("linear white -> |a|,|b| < 1e-3", abs(white[1]) < 1e-3 and abs(white[2]) < 1e-3,
      f"a={white[1]:.3e}, b={white[2]:.3e}")

black = linear_srgb_to_oklab(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))[0, 0]
check("linear black -> ~0", np.all(np.abs(black) < 1e-3),
      f"L={black[0]:.3e}, a={black[1]:.3e}, b={black[2]:.3e}")

grey = linear_srgb_to_oklab(np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32))[0, 0]
check("18% grey -> a~0,b~0", abs(grey[1]) < 1e-3 and abs(grey[2]) < 1e-3,
      f"a={grey[1]:.3e}, b={grey[2]:.3e}")
check("18% grey -> L~0.565 (+-0.005)", abs(grey[0] - 0.565) < 0.005, f"L={grey[0]:.6f}")


# ---------------------------------------------------------------------------
print("\n[4] NEGATIVE CONTROL (flip one M2 sign -> oracle must FAIL)")
M2_bad = _OKLAB_M2.copy()
M2_bad[1, 0] = -M2_bad[1, 0]   # flip the a-row l_ coefficient (1.9779984951)


def forward_bad(img):
    lms_ = np.cbrt(_apply_matrix(img.astype(np.float32), _OKLAB_M1)).astype(np.float32)
    return _apply_matrix(lms_, M2_bad)


bad = forward_bad(lin_img).reshape(-1, 3).astype(np.float64)
diff_bad = np.max(np.abs(bad - ref))
neg_fails = diff_bad >= 1e-4   # the corrupted transform should NOT match the oracle
check("sign-flipped M2 FAILS oracle (test has teeth)", neg_fails,
      f"max|d|={diff_bad:.3e} (>=1e-4 means teeth present)")


# ---------------------------------------------------------------------------
print("\n[5] INVARIANTS (the product claim)")
# Random saturated-ish colors; work in oklab directly.
lab0 = linear_srgb_to_oklab(lin_img)
lch0 = oklab_to_oklch(lab0)
C0 = lch0[..., 1]
neutral = C0 < 1e-3   # skip near-neutral pixels where hue is undefined

# (a) chroma scale C*=1.5 leaves L and h unchanged
lch_a = lch0.copy()
lch_a[..., 1] = lch_a[..., 1] * 1.5
lch_a_re = oklab_to_oklch(oklch_to_oklab(lch_a))
dL_a = np.max(np.abs(lch_a_re[..., 0] - lch0[..., 0]))
dh_a = np.max(np.abs(lch_a_re[..., 2][~neutral] - lch0[..., 2][~neutral]))
check("chroma scale holds L (<1e-5)", dL_a < 1e-5, f"max|dL|={dL_a:.3e}")
check("chroma scale holds h (<1e-5)", dh_a < 1e-5, f"max|dh|={dh_a:.3e}")

# (b) hue rotate h+=30deg leaves L and C unchanged
lch_b = lch0.copy()
lch_b[..., 2] = lch_b[..., 2] + np.radians(30.0).astype(np.float32)
lch_b_re = oklab_to_oklch(oklch_to_oklab(lch_b))
dL_b = np.max(np.abs(lch_b_re[..., 0] - lch0[..., 0]))
dC_b = np.max(np.abs(lch_b_re[..., 1] - lch0[..., 1]))
check("hue rotate holds L (<1e-5)", dL_b < 1e-5, f"max|dL|={dL_b:.3e}")
check("hue rotate holds C (<1e-5)", dC_b < 1e-5, f"max|dC|={dC_b:.3e}")

# (c) lightness*=1.3 then contrast slope leaves C and h unchanged
lab_c = lab0.copy()
lab_c[..., 0] = lab_c[..., 0] * 1.3
lab_c[..., 0] = 0.5 + (lab_c[..., 0] - 0.5) * (2.0 ** 0.4)  # contrast=+0.4 slope
lch_c = oklab_to_oklch(lab_c)
dC_c = np.max(np.abs(lch_c[..., 1] - lch0[..., 1]))
dh_c = np.max(np.abs(lch_c[..., 2][~neutral] - lch0[..., 2][~neutral]))
check("lightness/contrast holds C (<1e-5)", dC_c < 1e-5, f"max|dC|={dC_c:.3e}")
check("lightness/contrast holds h (<1e-5)", dh_c < 1e-5, f"max|dh|={dh_c:.3e}")


# ---------------------------------------------------------------------------
print()
if PASS:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
