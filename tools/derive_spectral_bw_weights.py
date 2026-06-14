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

if ortho_wr >= 0.05 or not panplus_is_max:
    print("\n!!! SANITY VIOLATION — do NOT use these weights; the S(lambda) shapes need revisiting.")


# ---------------------------------------------------------------------------
# 5. Copy-paste-ready dict literal for the node.
# ---------------------------------------------------------------------------
print("\n=== Copy-paste into nodes/spectral_bw.py ===")
print("_WEIGHTS = {")
for name, w in results.items():
    print(f'    "{name}": ({w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f}),')
print("}")
