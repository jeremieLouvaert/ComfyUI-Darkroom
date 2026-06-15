"""
Sabattier (Solarization) node for ComfyUI-Darkroom.

INSPIRED/COMPOSED tier. The direction and non-monotonic mechanism are
physically grounded (silver shielding + density-adds-log-reflectance); the
"printing" renormalization, the shield exponent, and the Mackie composition
are tuned to the look, NOT per-developer-calibrated chemistry. Delivers the
Sabattier character (highlight reversal + Mackie lines), not a named-process
simulation.

Physics:
  Develop partially -> re-expose the emulsion to a uniform fogging light
  mid-development -> continue developing. Areas with high first-development
  silver (already-dark parts of the positive) shield the underlying halide
  from the fogging light -> receive little added density. Areas with low
  silver (bright parts) are unshielded -> gain density and darken.

  LOAD-BEARING DIRECTION: the reversal darkens the HIGHLIGHTS; the deepest
  shadows are preserved. (Man-Ray look: bright background goes gray/dark with
  a light Mackie line outlining the subject.)

  Model (linear luminance x, 0=black, 1=white):
    D0(x)   = -log10(x)            first-development density
    Dfog(x) = beta * x^k           fog density that penetrates existing silver
    x_out   = 10^(-D0-Dfog) = x * 10^(-beta * x^k)
    Normalized to peak=1 ("printing" step, required for punch).

  Mackie lines composed from the Eberhard CSF adjacency kernel, derived from
  the ORIGINAL luminance (not the tone-folded output).

See docs/sabattier-derivation.md for the full derivation and spike findings.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..utils.color import srgb_to_linear, linear_to_srgb, luminance_rec709, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor

_ASYM = 6.0  # fixed internal asymmetry (overshoot/undershoot ratio)


def solarize_lut(x, beta, k, eps=1e-4, normalize=True):
    """
    Non-monotonic tone LUT: x * 10^(-beta * x^k), optionally peak-renormalized.

    Parameters
    ----------
    x : np.ndarray
        Linear luminance values (any shape, float32).
    beta : float
        re_exposure — depth of the tonal reversal.
    k : float
        shield — sharpness/confinement of the reversal to highlights.
    eps : float
        Clip floor (avoids log10(0)).
    normalize : bool
        If True, apply the "printing" step: divide by the peak of the LUT so
        the brightest surviving tone maps to 1.0. Without this the effect is
        globally dark and muddy (spike finding).

    Returns
    -------
    np.ndarray
        Same shape as x, float32.
    """
    x = np.clip(x, eps, 1.0)
    y = x * 10.0 ** (-beta * x ** k)
    if normalize:
        # 256-sample sweep to find the LUT peak (cheap, once per call)
        t = np.linspace(eps, 1.0, 256)
        ymax = (t * 10.0 ** (-beta * t ** k)).max()
        y = y / max(ymax, 1e-6)
    return y.astype(np.float32)


def mackie_delta(L_orig, intensity, asym, sigma):
    """
    Eberhard CSF edge field derived from the ORIGINAL (full-contrast) luminance.

    Mackie lines form from the original density gradient, so derive the
    high-pass field from L_orig, not the tone-folded output. Returns a signed
    delta to ADD to the folded luminance.

    Parameters
    ----------
    L_orig : np.ndarray (H, W)
        Original linear luminance (before solarization).
    intensity : float
        Mackie-line strength (scales both positive and negative lobes).
    asym : float
        Overshoot/undershoot ratio; higher = more filmic (fixed at 6.0 in v1).
    sigma : float
        Gaussian sigma in pixels (resolution-scaled).
    """
    m = gaussian_filter(L_orig, sigma, mode="nearest")
    hp = L_orig - m
    gpos = intensity
    gneg = intensity / max(asym, 1e-6)
    return np.where(hp > 0, gpos * hp, gneg * hp).astype(np.float32)


class Sabattier:
    """Sabattier / Solarization effect (inspired/composed tier — see module docstring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "re_exposure": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": (
                        "beta — depth of the tonal reversal. 0 = no solarization "
                        "(identity LUT). Higher values fold the highlights darker "
                        "and intensify the reversal."
                    ),
                }),
                "shield": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": (
                        "k — sharpness/confinement of the reversal to the highlights. "
                        "Low k spreads the fold into the midtones; high k confines it "
                        "to only the brightest tones (threshold-like)."
                    ),
                }),
                "mackie_intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "Mackie-line strength. Bright contour halos tracing density "
                        "edges, composing the Eberhard CSF adjacency kernel. "
                        "0 = no Mackie lines."
                    ),
                }),
                "edge_width": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 8.0, "step": 0.1,
                    "tooltip": (
                        "Mackie-line spatial scale: gaussian sigma in pixels at 1024 "
                        "long edge, scales proportionally with image resolution "
                        "(same formula as Eberhard)."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and fully processed (1).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, re_exposure=1.0, shield=1.0, mackie_intensity=0.5,
                edge_width=2.0, strength=1.0):
        # Early-return identity for degenerate inputs
        if strength <= 0.0 or (re_exposure <= 0.0 and mackie_intensity <= 0.0):
            return (image,)

        print(
            f"[Darkroom] Sabattier: re_exposure={re_exposure}, shield={shield}, "
            f"mackie_intensity={mackie_intensity}, edge_width={edge_width}, "
            f"strength={strength}"
        )

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            h, w = img.shape[:2]
            long_edge = max(h, w)

            # --- linearize and extract luminance ---
            lin = srgb_to_linear(img)
            L = luminance_rec709(lin)

            # --- solarization LUT (with printing renormalization) ---
            if re_exposure > 0.0:
                x_out = solarize_lut(L, beta=re_exposure, k=shield, normalize=True)
            else:
                x_out = L.copy()

            # --- Mackie lines from the ORIGINAL luminance ---
            if mackie_intensity > 0.0:
                sigma = edge_width * (long_edge / 1024.0)
                delta = mackie_delta(L, mackie_intensity, _ASYM, sigma)
                L_m = np.clip(x_out + delta, 0.0, None)
            else:
                L_m = np.clip(x_out, 0.0, None)

            # --- ratio recombine (hue-preserving, same as Eberhard) ---
            ratio = L_m / np.maximum(L, 1e-6)
            out = np.clip(lin * ratio[..., None], 0.0, 1.0)
            out = linear_to_srgb(out)

            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomSabattier": Sabattier}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomSabattier": "Sabattier"}
