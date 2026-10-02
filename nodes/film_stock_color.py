"""
Film Stock (Color) node for ComfyUI-Darkroom.

Two kinds of stock:
  * stocks from the Capture One Film Styles pack: the style itself, baked by
    Capture One into a 3D LUT (utils/film_luts.py, decisions.md 2026-10-02);
  * hand-authored stocks (Cinestill, Kodachrome, Vision3, Fuji sims, Instant,
    Aged): per-channel characteristic curves, saturation and split tone.

Either way, a freeform tone curve with Contrast / Shadows / Midtones /
Highlights can be applied on top (utils/tone_curve_ops.py; the canvas editor
is web/darkroom_freeform_curve.js). It is identity by default.
"""

import torch

from ..data.color_stocks import COLOR_STOCKS, COLOR_STOCK_NAMES
from ..utils.film_luts import has_baked, apply_baked
from ..utils.gpu_color import (run_on_device,
    srgb_to_linear, linear_to_srgb, apply_per_channel_curves,
    adjust_saturation, split_tone, blend
)
from ..utils.tone_curve_ops import IDENTITY_POINTS, parse_points, compose, is_identity, apply_table

_SLIDER = {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}


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
                "recovery": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Capture One stocks: include the style's shadow/highlight recovery "
                               "(an approximation of C1's local tool). Off = curves and colour only."
                }),
                "curve_points": ("STRING", {
                    "default": IDENTITY_POINTS,
                    "tooltip": "Tone curve applied on top of the film look, as x,y;x,y;... "
                               "Edit it on the curve above: click to add a point, drag, double-click to remove."
                }),
                "contrast": ("FLOAT", {**_SLIDER, "tooltip": "S-curve contrast on top of the film look"}),
                "shadows": ("FLOAT", {**_SLIDER, "tooltip": "Lift (+) or deepen (-) the shadows; black stays black"}),
                "midtones": ("FLOAT", {**_SLIDER, "tooltip": "Brighten (+) or darken (-) the midtones"}),
                "highlights": ("FLOAT", {**_SLIDER, "tooltip": "Brighten (+) or pull down (-) the highlights; white stays white"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, film_stock, strength, recovery=True, curve_points=IDENTITY_POINTS,
                contrast=0.0, shadows=0.0, midtones=0.0, highlights=0.0, **legacy):
        # override_toe/shoulder/gamma (1.28) no longer exist. ComfyUI drops unknown widget
        # inputs before calling us, so this only catches them when they arrive as links.
        if any(float(legacy.get(k, -1.0)) > -0.5 for k in ("override_toe", "override_shoulder", "override_gamma")):
            print("[Darkroom] Film Stock Color: override_toe/shoulder/gamma were replaced by the "
                  "tone curve in 1.29 and are ignored")

        if strength <= 0.0:
            return (image,)

        points = parse_points(curve_points)
        table = None
        if not is_identity(points, contrast, shadows, midtones, highlights):
            table = compose(points, contrast, shadows, midtones, highlights)

        if has_baked(film_stock):
            print(f"[Darkroom] Film Stock Color: {film_stock} (baked C1 look), "
                  f"strength={strength}, recovery={recovery}, curve={'on' if table is not None else 'off'}")

            def look(x):
                return apply_baked(x, film_stock, recovery)
        else:
            stock = COLOR_STOCKS[film_stock]
            print(f"[Darkroom] Film Stock Color: {film_stock}, strength={strength}, "
                  f"curve={'on' if table is not None else 'off'}")
            curves = [(c.toe_power, c.shoulder_power, c.slope, c.pivot_x, c.pivot_y)
                      for c in (stock.r_curve, stock.g_curve, stock.b_curve)]
            has_tint = any(abs(v) > 0.001 for v in (*stock.shadow_tint, *stock.highlight_tint))

            def look(x):
                curved = apply_per_channel_curves(srgb_to_linear(x), *curves)
                if abs(stock.saturation - 1.0) > 0.001:
                    curved = adjust_saturation(curved, stock.saturation)
                if has_tint:
                    curved = split_tone(curved, stock.shadow_tint, stock.highlight_tint)
                return linear_to_srgb(curved)

        def pipeline(original):
            rgb = original[..., :3]
            result = look(rgb)
            if table is not None:
                result = apply_table(result, table)
            result = blend(rgb, result, strength)
            if original.shape[-1] > 3:             # keep alpha / extra channels as they were
                result = torch.cat([result, original[..., 3:]], dim=-1)
            return result

        return (run_on_device(pipeline, image),)


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmStockColor": FilmStockColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmStockColor": "Film Stock (Color)",
}
