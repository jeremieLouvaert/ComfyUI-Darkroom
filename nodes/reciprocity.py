"""
Reciprocity Failure node for ComfyUI-Darkroom.
Simulates film reciprocity failure (Schwarzschild) at long exposures: the
per-channel long-exposure COLOR CAST + shadow crush, in linear light.

Datasheet-character-grounded (Kodak E-31 + Fuji datasheets), pointwise per
channel, pure numpy. See docs/reciprocity-derivation.md for the model and the
LOAD-BEARING color-shift DIRECTION rule (film cast = inverse of the recommended
CC correcting filter).
"""

import numpy as np

from ..utils.color import srgb_to_linear, linear_to_srgb, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


# Rec.709 luma weights (for cast renormalization — keep it a color shift, not
# a brightness change).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Shadow-crush strength. lin ** (1 + K_CRUSH * stop_loss): a toe-weighted power
# curve pinned at 0 and 1, darkening shadows in proportion to the stops lost.
# K_CRUSH = 0.15 chosen so a few stops (E100 @ 100s ≈ 2 stops) gives a visible
# but not extreme shadow drop: mid-grey 0.18 -> ~0.15, deep shadow 0.1 -> ~0.075,
# highlights ~untouched. See docs derivation (the crush curve is our defensible
# interpretation of "shadows lose speed faster", not a published curve).
K_CRUSH = 0.15

# ---------------------------------------------------------------------------
# Film data table.
#
# Each film -> list of (time_s, stop_loss, cc) points, monotonic in time.
#   stop_loss : stops of speed lost at that exposure (drives shadow crush).
#   cc        : None, or list of (color_letter, density) — the recommended
#               CC (colour-compensating) CORRECTING filter at that time. The
#               film's NATIVE cast is the INVERSE of this filter (see _cast_gain).
#
# Numbers from docs/reciprocity-derivation.md "Film presets" (E-31 + Fuji
# datasheets). Datasheet-character-grounded, not per-roll calibrated.
# ---------------------------------------------------------------------------
FILM_TABLE = {
    "B&W (general)": [
        (1.0, 1.0, None),
        (10.0, 2.0, None),
        (100.0, 3.0, None),
    ],
    "Kodak T-Max": [
        (1.0, 1.0 / 3.0, None),
        (10.0, 0.5, None),
        (100.0, 1.0, None),
    ],
    "Kodak Portra 400": [
        # ~none to 1s, very mild beyond. Minimal cast.
        (1.0, 0.0, None),
        (10.0, 1.0 / 3.0, None),
        (100.0, 2.0 / 3.0, None),
    ],
    "Kodak Ektachrome E100": [
        (1.0, 1.0 / 3.0, [("R", 0.025)]),
        (100.0, 2.0, [("Y", 0.10), ("R", 0.025)]),
    ],
    "Fuji Provia 100F": [
        # none to 128s; 240s -> (+1/3, CC2.5G). Cast = MAGENTA.
        (128.0, 0.0, None),
        (240.0, 1.0 / 3.0, [("G", 0.025)]),
    ],
    "Fuji Velvia 50": [
        # Strong time-only loss (4s->5s, 30s->66s, 60s->150s, 120s->290s).
        # stop_loss = log2(corrected / metered). Documented long-exposure GREEN
        # cast -> simulate via a magenta-correct CC (CC code 'M' -> film GREEN).
        (4.0, np.log2(5.0 / 4.0), None),
        (30.0, np.log2(66.0 / 30.0), [("M", 0.05)]),
        (60.0, np.log2(150.0 / 60.0), [("M", 0.075)]),
        (120.0, np.log2(290.0 / 120.0), [("M", 0.10)]),
    ],
}

FILM_NAMES = list(FILM_TABLE.keys())


def _interp_log_time(points, t):
    """
    Interpolate (stop_loss, cc_density_dict) at exposure time t over the film's
    table points, linear in log10(time).

    - Below the first point's time: clamp to ~0 effect (reciprocity onset — the
      daylight region). Returns stop_loss=0, empty cast.
    - Above the last point's time: clamp to the last point's values.
    - Between points: linear interp in log10(t) of stop_loss AND of each CC
      density (densities default to 0 where a point has no entry for a letter).
    """
    times = [p[0] for p in points]
    lt = np.log10(max(t, 1e-6))

    # Below onset -> no effect.
    if t <= times[0]:
        return 0.0, {}

    # At/above last point -> clamp to last.
    if t >= times[-1]:
        s = points[-1][1]
        return s, _cc_to_dict(points[-1][2])

    # Find bracketing segment.
    for i in range(len(points) - 1):
        t0, s0, cc0 = points[i]
        t1, s1, cc1 = points[i + 1]
        if t0 <= t <= t1:
            lt0, lt1 = np.log10(t0), np.log10(t1)
            f = (lt - lt0) / (lt1 - lt0) if lt1 > lt0 else 0.0
            stop_loss = s0 + f * (s1 - s0)
            d0 = _cc_to_dict(cc0)
            d1 = _cc_to_dict(cc1)
            letters = set(d0) | set(d1)
            cast = {}
            for L in letters:
                v = d0.get(L, 0.0) + f * (d1.get(L, 0.0) - d0.get(L, 0.0))
                if v != 0.0:
                    cast[L] = v
            return float(stop_loss), cast

    # Fallback (shouldn't reach).
    return float(points[-1][1]), _cc_to_dict(points[-1][2])


def _cc_to_dict(cc):
    """[(letter, density), ...] or None -> {letter: density}."""
    if not cc:
        return {}
    d = {}
    for letter, dens in cc:
        d[letter] = d.get(letter, 0.0) + dens
    return d


def _cast_gain(cast, flip=False):
    """
    Build the per-channel linear gain [gr, gg, gb] that SIMULATES the film's
    native cast, given the recommended CC correcting-filter densities.

    LOAD-BEARING DIRECTION (docs/reciprocity-derivation.md): the CC code is the
    CORRECTOR; the film's native cast is the INVERSE. We boost the channel(s)
    the corrector would attenuate:
      'M' (corrects GREEN cast) -> film GREEN  -> boost G by 10^(+d)
      'Y' (corrects BLUE cast)  -> film BLUE   -> boost B by 10^(+d)
      'C' (corrects RED cast)   -> film RED    -> boost R by 10^(+d)
      'R' (corrects CYAN cast)  -> film CYAN   -> boost G,B by 10^(+d/2) each
      'G' (corrects MAGENTA)    -> film MAGENTA-> boost R,B by 10^(+d/2) each
      'B' (corrects YELLOW)     -> film YELLOW -> boost R,G by 10^(+d/2) each

    Then renormalize by Rec.709 luma so a neutral grey keeps its brightness
    (cast, not exposure change).

    flip=True negates the densities (the NEGATIVE-CONTROL direction — must make
    the cast-direction test FAIL).
    """
    # Per-channel additive log10-gain (exponent) accumulator.
    exp = np.zeros(3, dtype=np.float64)  # [R, G, B]
    for letter, dens in cast.items():
        d = -dens if flip else dens
        if letter == "M":
            exp[1] += d
        elif letter == "Y":
            exp[2] += d
        elif letter == "C":
            exp[0] += d
        elif letter == "R":
            exp[1] += d / 2.0
            exp[2] += d / 2.0
        elif letter == "G":
            exp[0] += d / 2.0
            exp[2] += d / 2.0
        elif letter == "B":
            exp[0] += d / 2.0
            exp[1] += d / 2.0

    gains = np.power(10.0, exp)

    # Renormalize so Rec.709 luma of neutral grey is preserved.
    luma_of_gain = float(np.dot(gains, _LUMA))
    if luma_of_gain > 0.0:
        gains = gains / luma_of_gain

    return gains.astype(np.float64)


def _render(linear, gains, stop_loss):
    """
    Apply cast gains (per channel) then the shadow-crush power curve to a linear
    (H, W, 3) image. Returns linear (caller encodes to sRGB).
    """
    out = linear.copy()
    out[..., 0] *= gains[0]
    out[..., 1] *= gains[1]
    out[..., 2] *= gains[2]
    out = np.clip(out, 0.0, 1.0)
    if stop_loss > 0.0:
        # Toe-weighted power crush: pinned at 0 and 1, darkens shadows.
        out = np.power(out, 1.0 + K_CRUSH * stop_loss)
    return out


class Reciprocity:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "film": (FILM_NAMES, {
                    "default": "Kodak Portra 400",
                    "tooltip": "Film stock — sets the reciprocity character "
                               "(speed loss + long-exposure color cast)"
                }),
                "exposure_time": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 600.0, "step": 0.5,
                    "tooltip": "Metered exposure time in seconds. Drives the "
                               "magnitude via the film's log-time table. Below "
                               "the film's reciprocity onset = no effect"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and simulated (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, film="Kodak Portra 400", exposure_time=1.0, strength=1.0):
        points = FILM_TABLE.get(film, FILM_TABLE["Kodak Portra 400"])
        stop_loss, cast = _interp_log_time(points, exposure_time)
        gains = _cast_gain(cast)

        has_cast = bool(cast)
        # Early-exit: nothing to do (no blend), or below reciprocity onset.
        if strength <= 0.0 or (stop_loss <= 1e-4 and not has_cast):
            return (image,)

        print(f"[Darkroom] Reciprocity Failure: film={film}, t={exposure_time}s, "
              f"stops={stop_loss:.3f}, "
              f"cast=[{gains[0]:.4f},{gains[1]:.4f},{gains[2]:.4f}], "
              f"strength={strength}")

        gains_f = gains.astype(np.float32)
        images = tensor_to_numpy_batch(image)
        results = []
        for img in images:
            original = img.copy()
            linear = srgb_to_linear(img)
            rendered = _render(linear, gains_f, stop_loss)
            result = linear_to_srgb(np.clip(rendered, 0.0, 1.0))
            results.append(blend(original, result, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomReciprocity": Reciprocity}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomReciprocity": "Reciprocity Failure"}
