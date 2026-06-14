"""
Auto White Balance (color constancy) node for ComfyUI-Darkroom.
Estimates the scene illuminant from image content and divides out the color
cast in linear light (von Kries diagonal). Classical, public-domain estimators
(Gray World / White Patch / Shades of Gray / Gray Edge) unified by the
van de Weijer edge-based color-constancy framework. Pure numpy, no weights.
"""

import numpy as np

from ..utils.color import srgb_to_linear, linear_to_srgb, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


# White Patch robustness percentile (fixed internally, not exposed — keep UI lean).
_WHITE_PATCH_PCT = 97.0


def _minkowski(values, p):
    """Minkowski-norm mean of a 2D (H,W) channel array: (mean(values**p))**(1/p).

    p=1 -> arithmetic mean (Gray World), p->inf -> max. Used by Shades of Gray
    (on intensities) and Gray Edge (on gradient magnitudes).
    """
    v = np.abs(values).astype(np.float64)
    return float(np.mean(v ** p) ** (1.0 / p))


def _gradient_magnitude(channel):
    """Per-channel gradient magnitude |grad f_c| = sqrt(gx^2 + gy^2) via central
    differences (np.gradient handles the edges). Returns a (H,W) array.
    """
    gy, gx = np.gradient(channel.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def _estimate_illuminant(linear, method, minkowski_p):
    """Estimate e = (e_r, e_g, e_b) on the LINEAR image per the selected method.
    Each e_c clipped to >= 1e-6 (div-zero guard for flat / black images).
    """
    e = np.empty(3, dtype=np.float64)
    for c in range(3):
        ch = linear[..., c]
        if method == "Gray World":
            e[c] = float(np.mean(ch))
        elif method == "White Patch":
            e[c] = float(np.percentile(ch, _WHITE_PATCH_PCT))
        elif method == "Shades of Gray":
            e[c] = _minkowski(ch, minkowski_p)
        elif method == "Gray Edge":
            e[c] = _minkowski(_gradient_magnitude(ch), minkowski_p)
        else:
            e[c] = float(np.mean(ch))
    return np.clip(e, 1e-6, None)


class AutoWhiteBalance:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "method": (
                    ["Gray World", "White Patch", "Shades of Gray", "Gray Edge"],
                    {
                        "default": "Shades of Gray",
                        "tooltip": "Illuminant estimator. Gray World=avg-is-gray; "
                                   "White Patch=brightest-is-white (97th pct); "
                                   "Shades of Gray=Minkowski generalization (robust default); "
                                   "Gray Edge=Minkowski on gradients (often best classical)."
                    }
                ),
                "minkowski_p": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 16.0, "step": 0.5,
                    "tooltip": "Minkowski norm p. Applies to Shades of Gray and Gray Edge "
                               "only (ignored by Gray World and White Patch). ~6 is robust."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and corrected (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, method="Shades of Gray", minkowski_p=6.0, strength=1.0):
        if strength <= 0.0:
            return (image,)

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            linear = srgb_to_linear(img)

            # Per-image illuminant estimate (illuminant is per-image).
            e = _estimate_illuminant(linear, method, minkowski_p)

            # von Kries diagonal: scale each channel toward neutral while
            # preserving mean brightness via the mean(e) normalization.
            gains = (np.mean(e) / e).astype(np.float32)

            corrected = np.clip(linear * gains[np.newaxis, np.newaxis, :], 0.0, 1.0)
            out = linear_to_srgb(corrected)
            results.append(blend(original, out, strength))

            g = [round(float(x), 3) for x in gains]
            print(f"[Darkroom] Auto White Balance: method={method}, "
                  f"gains=[{g[0]}, {g[1]}, {g[2]}], strength={strength}")

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomAutoWhiteBalance": AutoWhiteBalance}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomAutoWhiteBalance": "Auto White Balance"}
