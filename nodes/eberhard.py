"""
Eberhard / Adjacency Acutance node for ComfyUI-Darkroom.

Physically-derived development edge-effect (Mackie lines / acutance) — the
"organic 3D sharpness" of large-format film, distinct from unsharp mask.
Production model = asymmetric Chemical-Spread-Function (CSF) convolution
(signed off 2026-06-14; see docs/eberhard-derivation.md, PRODUCTION MODEL).

Real adjacency effects come from developer exhaustion + lateral diffusion at a
bright/dark boundary: the bright side gains a density overshoot (Mackie line),
the dark side a density undershoot. The asymmetric, saturating, edge-localized
profile reads as organic rather than crunchy.

NOT per-film calibrated — reproduces the correct edge-profile SHAPE and length
scale, not a named emulsion. The bromide-drag advection direction/strength is
TUNED, not literature-derived; the field it advects (the dark-side inhibitor
proxy) is physically real.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

from ..utils.color import srgb_to_linear, linear_to_srgb, luminance_rec709, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


class Eberhard:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "edge_width": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 8.0, "step": 0.1,
                    "tooltip": "Mackie-line spatial scale (gaussian sigma, px @ 1024; scales with long edge)"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Edge-effect strength (scales both overshoot and undershoot gains)"
                }),
                "asymmetry": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Overshoot/undershoot ratio. 1 = symmetric (unsharp-like), high = filmic"
                }),
                "drag_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Bromide drag — density-minus streaks downstream of dark edges. 0 = off (pure adjacency)"
                }),
                "drag_angle": ("FLOAT", {
                    "default": 90.0, "min": 0.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Drag direction in degrees. 90 = downward / gravity"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and adjusted (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, edge_width=2.0, intensity=0.5, asymmetry=6.0,
                drag_amount=0.0, drag_angle=90.0, strength=1.0):
        if strength <= 0.0 or (intensity <= 0.0 and drag_amount <= 0.0):
            return (image,)

        print(f"[Darkroom] Adjacency Acutance: edge_width={edge_width}, "
              f"intensity={intensity}, asymmetry={asymmetry}, "
              f"drag={drag_amount}, strength={strength}")

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            h, w = img.shape[:2]
            long_edge = max(h, w)

            lin = srgb_to_linear(img)
            L = luminance_rec709(lin)

            sigma = edge_width * (long_edge / 1024.0)
            m = gaussian_filter(L, sigma, mode="nearest")
            hp = L - m  # high-pass detail

            # Asymmetric gain — the saturation asymmetry that makes it NOT unsharp.
            gpos = intensity                          # bright-side overshoot gain
            gneg = intensity / max(asymmetry, 1e-6)   # dark-side undershoot gain
            delta = np.where(hp > 0, gpos * hp, gneg * hp)
            L_enh = np.clip(L + delta, 0.0, None)

            # Optional bromide drag (inhibitor proxy, elevated on the dark side
            # of edges; advected along the gravity vector and subtracted).
            if drag_amount > 0.0:
                P = gaussian_filter(np.clip(-hp, 0.0, None), sigma * 0.5, mode="nearest")
                # D_px = drag travel scale. Chosen as 3% of the long edge so the
                # streak length is a fixed fraction of frame (resolution-clean,
                # same principle as sigma). At drag_amount=1 the streak reaches
                # ~3% of the long edge.
                D_px = 0.03 * long_edge
                mag = drag_amount * D_px
                rad = np.deg2rad(drag_angle)
                # Image y-axis points DOWN (row index increases downward), so
                # dy = +sin gives angle=90 -> downward (gravity). dx = +cos.
                dx = np.cos(rad) * mag
                dy = np.sin(rad) * mag
                P_shifted = ndshift(P, (dy, dx), order=1, mode="nearest")
                L_enh = np.clip(L_enh - drag_amount * P_shifted, 0.0, None)

            # Apply to RGB by RATIO -> preserves hue (no color fringing in v1).
            ratio = L_enh / np.maximum(L, 1e-6)
            out = np.clip(lin * ratio[..., None], 0.0, 1.0)
            out = linear_to_srgb(out)

            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomEberhard": Eberhard}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomEberhard": "Adjacency Acutance"}
