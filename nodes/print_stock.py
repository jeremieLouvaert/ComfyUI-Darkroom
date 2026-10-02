"""
Print Stock node for ComfyUI-Darkroom.
Models the negative-to-print chain — the second characteristic curve
that gives cinema its distinctive look.

Chain after Film Stock (Color) for the full photochemical pipeline:
  Image → Film Stock (Color) [negative] → Print Stock [print] → Output
"""

import torch

from ..data.print_stocks import PRINT_STOCKS, PRINT_STOCK_NAMES
from ..utils.gpu_color import (run_on_device,
    srgb_to_linear, linear_to_srgb, apply_per_channel_curves,
    adjust_saturation, blend
)


class PrintStock:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "print_stock": (PRINT_STOCK_NAMES, {
                    "default": "Kodak 2383",
                    "tooltip": "Cinema print stock. Chain after Film Stock (Color) for full pipeline"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and full print effect (1)"
                }),
            },
            "optional": {
                "contrast_boost": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Additional contrast on top of the print curve"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, print_stock, strength, contrast_boost=0.0):
        if strength <= 0.0:
            return (image,)

        stock = PRINT_STOCKS[print_stock]
        print(f"[Darkroom] Print Stock: {print_stock}, strength={strength}")

        # Build curve params
        r_params = (stock.r_curve.toe_power, stock.r_curve.shoulder_power,
                    stock.r_curve.slope, stock.r_curve.pivot_x, stock.r_curve.pivot_y)
        g_params = (stock.g_curve.toe_power, stock.g_curve.shoulder_power,
                    stock.g_curve.slope, stock.g_curve.pivot_x, stock.g_curve.pivot_y)
        b_params = (stock.b_curve.toe_power, stock.b_curve.shoulder_power,
                    stock.b_curve.slope, stock.b_curve.pivot_x, stock.b_curve.pivot_y)

        def pipeline(original):
            # Linearize
            linear = srgb_to_linear(original)

            # Apply print stock characteristic curves
            printed = apply_per_channel_curves(linear, r_params, g_params, b_params)

            # Apply print saturation
            if abs(stock.saturation - 1.0) > 0.001:
                printed = adjust_saturation(printed, stock.saturation)

            # Apply black density floor (real print film never reaches pure black)
            if stock.black_density > 0:
                printed = printed * (1.0 - stock.black_density) + stock.black_density

            # Optional contrast boost: mild S-curve via power adjustment
            if contrast_boost > 0.01:
                midpoint = 0.18
                # Steepen the curve around the midpoint
                above = printed > midpoint
                boost = 1.0 + contrast_boost * 0.5
                printed = torch.where(
                    above,
                    midpoint + (printed - midpoint) * boost,
                    midpoint - (midpoint - printed) * boost
                )
                printed = printed.clamp(0.0, 1.0)

            # Back to display gamma
            result = linear_to_srgb(printed)

            # Blend with original
            return blend(original, result, strength)

        return (run_on_device(pipeline, image),)


NODE_CLASS_MAPPINGS = {
    "DarkroomPrintStock": PrintStock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomPrintStock": "Print Stock",
}
