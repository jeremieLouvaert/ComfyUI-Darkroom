"""
Halation node for ComfyUI-Darkroom.

Physically-derived annular halation (docs/halation-derivation.md, signed off
2026-07-23; spike ALL CHECKS PASSED + Jeremie eyeball pass, ../_halation_spike/).
Supersedes the v1 threshold->blur->tint->screen fake in place (same node key
`DarkroomHalation`).

Real film halation: light from a bright image point transmits through the
film base, totally-internally-reflects off the rear base-air surface, and
re-exposes the emulsion in a RING standing off the source at a predictable
radius r_c -- not a centered glow. The TIR cutoff concentrates energy at and
beyond r_c (P_TIR(r) ~ Fresnel-jump / (1 + (r/2d)^2)^2); a diffuse
multiple-scattering tail (exp(-r/L)) fills in beyond that. Red-sensitive
emulsion sits deepest in the color-negative stack, so the round-trip through
the layers favors red -- the derived (non-tunable) ordering w_R > w_G > w_B
that makes halation red-orange, CineStill's signature.

NOT per-film calibrated -- the ring geometry and Fresnel/TIR shape are
first-principles (n=1.5 fixed film-base index); the per-channel weights are
honest presets (dye-spectra magnitudes not vendored, see derivation doc
"Honest residuals"). Composite is additive in linear light (halation ADDS
colored exposure -- the hue shift IS the look), deliberately NOT the house
hue-preserving ratio-recombine (LOAD-BEARING CALL 2).
"""

import numpy as np
from scipy.signal import fftconvolve

from ..utils.color import srgb_to_linear, linear_to_srgb
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor, halation_psf


# Preset channel weights w_c (docs/halation-derivation.md sec 4). Ordering
# w_R > w_G > w_B is derived; magnitudes are honest presets.
PRESET_WEIGHTS = {
    "CineStill 800T": (1.00, 0.32, 0.10),
    "Vision3 subtle": (1.00, 0.45, 0.18),
    "B&W plate": (1.00, 1.00, 1.00),
}
# Vision3 subtle ships with an anti-halation undercoat present in real stock;
# doc sec 4 note: "AH present -> pair with ah_strength 0.7". Floor, not clamp.
PRESET_AH_FLOOR = {
    "Vision3 subtle": 0.7,
}
PRESET_NAMES = ["CineStill 800T", "Vision3 subtle", "B&W plate", "Custom"]

# Kernel cache keyed on the (rounded) physical params + clamp size the
# kernel is built from, so repeated batch frames / repeated runs at the same
# widget settings don't rebuild the FFT kernel from scratch.
_KERNEL_CACHE = {}


def _get_kernel(ring_px, tail_px, balance, max_ksize):
    key = (round(ring_px, 4), round(tail_px, 4), round(balance, 4), int(max_ksize))
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = halation_psf(ring_px, tail_px, balance, tir_on=True, max_ksize=max_ksize)
        _KERNEL_CACHE[key] = k
    return k


class Halation:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "preset": (PRESET_NAMES, {
                    "default": "CineStill 800T",
                    "tooltip": "Halo channel weights (w_R/w_G/w_B). 'Custom' uses halo_r/g/b below"
                }),
                "strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Halation intensity, maps 0-1 UI to 0-0.25 physical energy fraction. 0 = early-exit passthrough"
                }),
                "ring_radius": ("FLOAT", {
                    "default": 8.0, "min": 3.0, "max": 60.0, "step": 0.5,
                    "tooltip": "TIR ring radius r_c, ref-px @1024 long edge (film-base-thickness stand-in, no-microns rule)"
                }),
                "tail_length": ("FLOAT", {
                    "default": 40.0, "min": 5.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Diffuse multiple-scattering tail spread L, ref-px @1024 long edge"
                }),
                "ring_tail_balance": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mix between the TIR ring (1.0) and the diffuse tail (0.0)"
                }),
                "ah_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Anti-halation layer. Halo energy scales by (1-ah)^2. 0 = remjet removed (CineStill), 1 = modern stock, halo dead"
                }),
                "halo_r": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Red halo channel weight (preset=Custom only)"
                }),
                "halo_g": ("FLOAT", {
                    "default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Green halo channel weight (preset=Custom only)"
                }),
                "halo_b": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blue halo channel weight (preset=Custom only)"
                }),
                "highlight_knee": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Isolate bright sources before halating (taste). 0 = fully physical, all light halates"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, preset="CineStill 800T", strength=0.35,
                ring_radius=8.0, tail_length=40.0, ring_tail_balance=0.65,
                ah_strength=0.0, halo_r=1.0, halo_g=0.32, halo_b=0.10,
                highlight_knee=0.0):

        ah = ah_strength
        floor = PRESET_AH_FLOOR.get(preset)
        if floor is not None:
            ah = max(ah, floor)

        if strength <= 0.0 or (1.0 - ah) <= 0.0:
            return (image,)

        w = PRESET_WEIGHTS.get(preset, (halo_r, halo_g, halo_b))
        knee = min(max(highlight_knee, 0.0), 0.95)
        frac = strength * 0.25 * (1.0 - ah) ** 2

        print(f"[Darkroom] Halation: preset={preset}, strength={strength}, "
              f"ring_radius={ring_radius}, tail_length={tail_length}, "
              f"balance={ring_tail_balance}, ah_strength={ah}, "
              f"w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}), knee={knee}")

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            h, w_px = img.shape[:2]
            long_edge = max(h, w_px)
            scale = long_edge / 1024.0
            ring_px = ring_radius * scale
            tail_px = tail_length * scale
            max_ksize = max(min(h, w_px) - 1, 1)

            kernel = _get_kernel(ring_px, tail_px, ring_tail_balance, max_ksize)

            lin = srgb_to_linear(img)
            E = np.clip(lin - knee, 0.0, None) / (1.0 - knee)

            H = np.empty_like(lin)
            for c in range(3):
                H[..., c] = fftconvolve(E[..., c], kernel, mode="same") * w[c]

            out = np.clip(lin + frac * H, 0.0, 1.0)
            results.append(linear_to_srgb(out))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {
    "DarkroomHalation": Halation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomHalation": "Halation",
}
