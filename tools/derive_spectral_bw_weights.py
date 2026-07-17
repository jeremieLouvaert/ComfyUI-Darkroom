"""
Offline derivation of the Spectral B&W (Ortho/Pan) per-type RGB->gray weights.

Run with the ComfyUI embedded python. NOT a runtime dependency of the node — this
is a build tool. It computes the weight triple `w = (w_r, w_g, w_b)` for each film
spectral-sensitivity type by integrating an analytic sensitivity curve S(lambda)
against the Mallett & Yuksel 2019 sRGB spectral basis (one basis spectrum per sRGB
primary), then normalizing so sum(w) = 1.

LOAD-BEARING INSIGHT (see docs/spectral-bw-derivation.md):
    gray = integral S(lambda)*I(lambda) dlambda
         = R*integral(S*b_r) + G*integral(S*b_g) + B*integral(S*b_b)
         = w_r*R + w_g*G + w_b*B
because Mallett-Yuksel reconstructs the per-pixel spectrum as a LINEAR combination
of 3 fixed basis spectra. So for a fixed S(lambda) the whole spectral computation
collapses to a fixed RGB weight triple — a principled (spectrally-derived) channel
mixer. We do the spectral work ONCE, here, offline.

The cutoffs are the grounded part (ortho/pan literature, ~nm). The exact window
SHAPE (raised-cosine shoulders) is a modeling choice, documented as such.
"""

import numpy as np
import colour


# ---------------------------------------------------------------------------
# 1. Load the Mallett-Yuksel 2019 sRGB spectral basis.
#    colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019 is a
#    MultiSpectralDistributions with 3 signals (columns) = the per-primary basis
#    spectra over a shared wavelength range. Column order is R, G, B.
# ---------------------------------------------------------------------------
basis = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
wl = basis.wavelengths.astype(np.float64)          # (N,) wavelengths in nm
values = basis.values.astype(np.float64)           # (N, 3) -> columns r, g, b
b_r = values[:, 0]
b_g = values[:, 1]
b_b = values[:, 2]

print(f"Mallett-Yuksel sRGB basis loaded: {len(wl)} samples, "
      f"{wl.min():.0f}-{wl.max():.0f} nm (step {wl[1]-wl[0]:.0f} nm)")


# ---------------------------------------------------------------------------
# 2. Analytic sensitivity windows S(lambda).
#    A smooth raised-cosine "plateau with shoulders": flat = 1 inside
#    [rise_hi, fall_lo], cosine ramp up over [rise_lo, rise_hi] and cosine ramp
#    down over [fall_lo, fall_hi]. `red_gain` lets a type emphasize/de-emphasize
#    its red shoulder without moving the cutoff (used for orthopan's reduced red
#    and pan+'s red emphasis).
# ---------------------------------------------------------------------------
def raised_cosine_window(lam, rise_lo, rise_hi, fall_lo, fall_hi,
                         red_emphasis_from=None, red_gain=1.0):
    """Smooth window in [0,1] over lam. Optional linear red emphasis above a wl."""
    s = np.ones_like(lam)

    # rising shoulder
    up = (lam >= rise_lo) & (lam < rise_hi)
    s = np.where(lam < rise_lo, 0.0, s)
    s = np.where(up, 0.5 * (1.0 - np.cos(np.pi * (lam - rise_lo) / (rise_hi - rise_lo))), s)

    # falling shoulder
    down = (lam > fall_lo) & (lam <= fall_hi)
    s = np.where(down, 0.5 * (1.0 + np.cos(np.pi * (lam - fall_lo) / (fall_hi - fall_lo))), s)
    s = np.where(lam > fall_hi, 0.0, s)

    if red_emphasis_from is not None and red_gain != 1.0:
        # Scale the long-wavelength (red) end. Ramp the gain in from
        # red_emphasis_from to fall_hi so it's smooth, then clip back to window.
        ramp = np.clip((lam - red_emphasis_from) / (fall_hi - red_emphasis_from + 1e-9), 0.0, 1.0)
        gain = 1.0 + (red_gain - 1.0) * ramp
        s = s * gain

    return np.clip(s, 0.0, None)


# Cutoffs from the ortho/pan literature (nm). Clamped to the basis range below.
# Shoulders are ~30-40 nm raised-cosine ramps centred on the nominal cutoff.
def S_blue(lam):
    # ~360-500, peak ~440. Blue-sensitive collodion / early plates.
    # Real blue plates roll off in the deep violet near the UV edge (they are not
    # flat to the basis edge at 380 nm); the rise shoulder 390->420 models that.
    return raised_cosine_window(lam, 390, 420, 470, 500)

def S_ortho(lam):
    # ~360-590, cuts red at ~590. Full blue + green, red ~ 0. The classic ortho.
    # The M-Y red basis is ~0 until 585 then jumps to 0.62 at 590; so the fall
    # shoulder is shaped to FINISH by ~585 (S=0 right where red ignites), which
    # keeps the documented ~590 cutoff while driving w_r genuinely toward 0.
    # The rise shoulder 390->420 is the physical deep-violet roll-off of a blue/
    # green-sensitive plate: WITHOUT it, the red sRGB basis's secondary VIOLET
    # lobe (b_r ~ 0.33 at 380-400 nm — purples need red) leaks into w_r and pins
    # it at ~0.054. Rolling sensitivity off in the violet (as real ortho film does)
    # removes that non-physical leak and lands w_r ~ 0.021. The red CUTOFF is
    # unchanged; this only fixes the unphysical "flat to 380 nm" assumption.
    return raised_cosine_window(lam, 390, 420, 555, 585)

def S_orthopan(lam):
    # ~360-650 with a REDUCED red shoulder (mild red).
    return raised_cosine_window(lam, 360, 380, 600, 650,
                                red_emphasis_from=580, red_gain=0.55)

def S_pan(lam):
    # ~360-680, full visible, roughly flat-ish. Natural modern rendering.
    return raised_cosine_window(lam, 360, 380, 650, 680)

def S_panplus(lam):
    # ~360-720+ with RED EMPHASIS. Pseudo-IR (SFX/Rollei) — reds render LIGHT.
    return raised_cosine_window(lam, 360, 380, 700, 730,
                                red_emphasis_from=580, red_gain=2.2)


TYPES = {
    "Blue-sensitive": S_blue,
    "Orthochromatic": S_ortho,
    "Orthopanchromatic": S_orthopan,
    "Panchromatic": S_pan,
    "Panchromatic+": S_panplus,
}


# ---------------------------------------------------------------------------
# 2b. v1.x MEASURED stocks — real datasheet-digitized sensitivity curves from
#     the vendored third_party/spectral_film_lut (Jan Lohse, MIT; see
#     ATTRIBUTION.md). Data format: log_sensitivity = [{wavelength_nm: log10}].
#     Policy: linearize S = 10**log_sens; align to the basis grid with linear
#     interpolation INSIDE the measured range and linear extrapolation of
#     LOG-sensitivity beyond both ends (same alignment family the vendored
#     engine itself uses; the measured red shoulders are steeply negative in
#     log space, so extrapolation decays smoothly instead of cliffing).
#     Absolute scale cancels in the sum-to-1 normalization.
# ---------------------------------------------------------------------------
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SFL_SRC = os.path.join(_REPO, "third_party", "spectral_film_lut", "src")


def _load_measured_points(module_relpath):
    """Extract the measured `log_sensitivity` dict from a vendored stock module
    WITHOUT executing any vendored code: the data is a pure float-dict literal,
    so we AST-parse the file and literal_eval the `log_sensitivity=` keyword of
    the FilmData(...) call. (Executing the module is not an option here — the
    vendored package __init__ pulls numba, incompatible with the embedded
    numpy, and the module calls dataclasses.replace on the real FilmData.)"""
    import ast

    path = os.path.join(_SFL_SRC, "spectral_film_lut", "bw_negative_film",
                        module_relpath)
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "log_sensitivity":
                    ls = ast.literal_eval(kw.value)
                    points = ls[0] if isinstance(ls, list) else ls
                    lam = np.array(sorted(points), dtype=np.float64)
                    logv = np.array([points[l] for l in lam], dtype=np.float64)
                    if len(lam) < 10:
                        raise RuntimeError(
                            f"suspiciously few points ({len(lam)}) in {path}")
                    return lam, logv
    raise RuntimeError(f"no log_sensitivity literal found in {path}")


def measured_S_on_grid(lam, logv, wl_grid):
    """log-linear interpolate inside the measured range, linear-extrapolate
    log-sensitivity beyond both ends, then linearize."""
    logS = np.interp(wl_grid, lam, logv)
    left = wl_grid < lam[0]
    if left.any():
        slope = (logv[1] - logv[0]) / (lam[1] - lam[0])
        logS[left] = logv[0] + slope * (wl_grid[left] - lam[0])
    right = wl_grid > lam[-1]
    if right.any():
        slope = (logv[-1] - logv[-2]) / (lam[-1] - lam[-2])
        logS[right] = logv[-1] + slope * (wl_grid[right] - lam[-1])
    S = 10.0 ** logS
    return S / S.max()          # scale cancels in normalization; keep prints sane


MEASURED_STOCKS = {
    "Kodak Tri-X 400": "kodak_trix_400.py",
    "Kodak 5222 (Double-X)": "kodak_5222.py",
}


# ---------------------------------------------------------------------------
# 3. w_c = trapz(S * b_c, lambda); normalize so sum(w) = 1.
# ---------------------------------------------------------------------------
# NumPy 2.x renamed trapz -> trapezoid; keep a fallback for older numpy.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def derive(S_fn):
    s = S_fn(wl)
    w_r = _trapz(s * b_r, wl)
    w_g = _trapz(s * b_g, wl)
    w_b = _trapz(s * b_b, wl)
    w = np.array([w_r, w_g, w_b], dtype=np.float64)
    w = np.clip(w, 0.0, None)          # basis can dip slightly negative; clamp
    w = w / w.sum()
    return w


results = {}
print("\n=== Derived weight triples (normalized, sum=1) ===")
for name, S_fn in TYPES.items():
    w = derive(S_fn)
    results[name] = w
    print(f"  {name:18s}  w_r={w[0]:.6f}  w_g={w[1]:.6f}  w_b={w[2]:.6f}  sum={w.sum():.6f}")

print("\n=== v1.x MEASURED stocks (vendored spectral_film_lut, MIT) ===")
measured_results = {}
for name, relpath in MEASURED_STOCKS.items():
    lam, logv = _load_measured_points(relpath)
    S = measured_S_on_grid(lam, logv, wl)
    w_r = _trapz(S * b_r, wl)
    w_g = _trapz(S * b_g, wl)
    w_b = _trapz(S * b_b, wl)
    w = np.clip(np.array([w_r, w_g, w_b]), 0.0, None)
    w = w / w.sum()
    measured_results[name] = w
    results[name] = w
    print(f"  {name:22s}  w_r={w[0]:.6f}  w_g={w[1]:.6f}  w_b={w[2]:.6f}  "
          f"(measured {lam[0]:.0f}-{lam[-1]:.0f} nm, {len(lam)} pts)")

# Negative control: a mangled curve (log values reversed across wavelength)
# must move the weights materially — proves the data actually drives the result.
_lam5222, _logv5222 = _load_measured_points(MEASURED_STOCKS["Kodak 5222 (Double-X)"])
_S_mangled = measured_S_on_grid(_lam5222, _logv5222[::-1], wl)
_wm = np.clip(np.array([_trapz(_S_mangled * b, wl) for b in (b_r, b_g, b_b)]), 0.0, None)
_wm = _wm / _wm.sum()
_shift = np.abs(_wm - measured_results["Kodak 5222 (Double-X)"]).max()
print(f"  NEGATIVE CONTROL (reversed 5222 curve): max weight shift {_shift:.4f} "
      f"{'OK (fires)' if _shift > 0.05 else 'FAIL (data not driving result!)'}")


# ---------------------------------------------------------------------------
# 4. Sanity (per the spec) before trusting.
# ---------------------------------------------------------------------------
print("\n=== Sanity checks ===")
ortho_wr = results["Orthochromatic"][0]
panplus_wr = results["Panchromatic+"][0]
all_wr = {k: v[0] for k, v in results.items()}
panplus_is_max = panplus_wr >= max(all_wr.values()) - 1e-12

print(f"  ortho w_r ~ 0           : {ortho_wr:.6f}   {'OK' if ortho_wr < 0.05 else 'FAIL'}")
print(f"  Pan+ w_r is the largest : {panplus_wr:.6f}  (max over types={max(all_wr.values()):.6f})  "
      f"{'OK' if panplus_is_max else 'FAIL'}")
print(f"  Pan more balanced than ortho/pan+ (info): pan w={tuple(round(float(x),3) for x in results['Panchromatic'])}")

measured_ok = True
for name, w in measured_results.items():
    pan_like = 0.15 < w[0] < results["Panchromatic+"][0]
    measured_ok &= pan_like
    print(f"  {name}: pan-range w_r (0.15 < {w[0]:.4f} < Pan+ {results['Panchromatic+'][0]:.4f})"
          f"  {'OK' if pan_like else 'FAIL'}")

if ortho_wr >= 0.05 or not panplus_is_max or not measured_ok:
    print("\n!!! SANITY VIOLATION — do NOT use these weights; the S(lambda) shapes need revisiting.")


# ---------------------------------------------------------------------------
# 5. Copy-paste-ready dict literal for the node.
# ---------------------------------------------------------------------------
print("\n=== Copy-paste into nodes/spectral_bw.py ===")
print("_WEIGHTS = {")
for name, w in results.items():
    print(f'    "{name}": ({w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f}),')
print("}")
