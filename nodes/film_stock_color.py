"""
Film Stock (Color) node for ComfyUI-Darkroom.
Applies per-channel characteristic curves for real film stock emulation.
"""

from ..data.color_stocks import COLOR_STOCKS, COLOR_STOCK_NAMES, CurveParams
from ..utils.gpu_color import (run_on_device,
    srgb_to_linear, linear_to_srgb, apply_per_channel_curves,
    adjust_saturation, split_tone, blend
)


class FilmStockColor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "film_stock": (COLOR_STOCK_NAMES, {
                    "default": "Neg / Kodak Portra 400",
                    "tooltip": "Select a film stock to emulate"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and full effect (1)"
                }),
            },
            "optional": {
                "override_toe": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Override shadow compression. -1 = use preset value"
                }),
                "override_shoulder": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Override highlight rolloff. -1 = use preset value"
                }),
                "override_gamma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Override midtone contrast. -1 = use preset value"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def _apply_overrides(self, curve, override_toe, override_shoulder, override_gamma):
        """Apply user overrides to curve parameters. -1 means use preset."""
        toe = override_toe if override_toe > -0.5 else curve.toe_power
        shoulder = override_shoulder if override_shoulder > -0.5 else curve.shoulder_power
        slope = override_gamma if override_gamma > -0.5 else curve.slope
        return (toe, shoulder, slope, curve.pivot_x, curve.pivot_y)

    def execute(self, image, film_stock, strength,
                override_toe=-1.0, override_shoulder=-1.0, override_gamma=-1.0):

        if strength <= 0.0:
            return (image,)

        stock = COLOR_STOCKS[film_stock]
        print(f"[Darkroom] Film Stock Color: {film_stock}, strength={strength}")

        # Build curve params with overrides
        r_params = self._apply_overrides(stock.r_curve, override_toe, override_shoulder, override_gamma)
        g_params = self._apply_overrides(stock.g_curve, override_toe, override_shoulder, override_gamma)
        b_params = self._apply_overrides(stock.b_curve, override_toe, override_shoulder, override_gamma)

        has_shadow_tint = any(abs(v) > 0.001 for v in stock.shadow_tint)
        has_highlight_tint = any(abs(v) > 0.001 for v in stock.highlight_tint)

        def pipeline(original):
            # Linearize (remove sRGB gamma)
            linear = srgb_to_linear(original)

            # Apply per-channel characteristic curves
            curved = apply_per_channel_curves(linear, r_params, g_params, b_params)

            # Apply stock-specific saturation
            if abs(stock.saturation - 1.0) > 0.001:
                curved = adjust_saturation(curved, stock.saturation)

            # Apply shadow/highlight tinting
            if has_shadow_tint or has_highlight_tint:
                curved = split_tone(curved, stock.shadow_tint, stock.highlight_tint)

            # Back to display gamma
            result = linear_to_srgb(curved)

            # Blend with original in sRGB space (perceptually correct)
            return blend(original, result, strength)

        return (run_on_device(pipeline, image),)


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmStockColor": FilmStockColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmStockColor": "Film Stock (Color)",
}
