"""
Film Stock (B&W) node for ComfyUI-Darkroom.
Black & white conversion using real spectral sensitivity coefficients per film stock.
"""

import numpy as np
import torch

from ..data.bw_stocks import BW_STOCKS, BW_STOCK_NAMES, COLOR_FILTERS, FILTER_NAMES
from ..utils.gpu_color import (
    run_on_device, srgb_to_linear, linear_to_srgb, characteristic_curve, blend
)


class FilmStockBW:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "film_stock": (BW_STOCK_NAMES, {
                    "default": "Ilford HP5 Plus 400",
                    "tooltip": "B&W film stock. Each has unique spectral sensitivity"
                }),
                "color_filter": (FILTER_NAMES, {
                    "default": "None",
                    "tooltip": "Simulate a colored lens filter on B&W film"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original color (0) and B&W (1)"
                }),
            },
            "optional": {
                "contrast": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Additional contrast adjustment. 0 = stock default"
                }),
                "exposure_shift": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Exposure compensation in stops (Zone System)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, film_stock, color_filter, strength,
                contrast=0.0, exposure_shift=0.0):

        if strength <= 0.0:
            return (image,)

        stock = BW_STOCKS[film_stock]
        filter_mult = COLOR_FILTERS[color_filter]

        print(f"[Darkroom] Film Stock B&W: {film_stock}, filter={color_filter}, strength={strength}")

        # Compute effective channel weights: spectral sensitivity × filter transmission
        weights = np.array([
            stock.red_weight * filter_mult[0],
            stock.green_weight * filter_mult[1],
            stock.blue_weight * filter_mult[2],
        ], dtype=np.float32)

        # Renormalize so weights sum to 1.0
        total = weights.sum()
        if total > 0:
            weights /= total

        # Adjust contrast curve if user override is nonzero
        curve = stock.contrast_curve
        if abs(contrast) > 0.01:
            # Positive contrast = steeper slope + more toe/shoulder compression
            slope = curve.slope + contrast * 0.3
            toe = curve.toe_power + contrast * 0.2
            shoulder = curve.shoulder_power - contrast * 0.15
            curve_params = (max(toe, 0.5), max(shoulder, 0.5), max(slope, 0.3),
                           curve.pivot_x, curve.pivot_y)
        else:
            curve_params = (curve.toe_power, curve.shoulder_power, curve.slope,
                           curve.pivot_x, curve.pivot_y)

        w0, w1, w2 = (float(w) for w in weights)

        def pipeline(original):
            # Linearize
            linear = srgb_to_linear(original)

            # Weighted sum to monochrome using film spectral sensitivity
            bw = (linear[..., 0] * w0 +
                  linear[..., 1] * w1 +
                  linear[..., 2] * w2)

            # Exposure shift (Zone System: each stop doubles/halves light)
            if abs(exposure_shift) > 0.01:
                bw = bw * (2.0 ** exposure_shift)
                bw = bw.clamp(0.0, 1.0)

            # Apply stock contrast curve
            bw = characteristic_curve(bw, *curve_params)

            # Add base fog (lifts the black point — real film never reaches pure black)
            if stock.base_fog > 0:
                bw = bw * (1.0 - stock.base_fog) + stock.base_fog

            # Back to display gamma
            bw = linear_to_srgb(bw)

            # Stack to 3-channel (ComfyUI requires RGB)
            result = torch.stack([bw, bw, bw], dim=-1)

            # Blend with original color image
            return blend(original, result, strength)

        return (run_on_device(pipeline, image),)


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmStockBW": FilmStockBW,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmStockBW": "Film Stock (B&W)",
}
