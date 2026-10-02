"""
Tone Curve node for ComfyUI-Darkroom.
5-point cubic spline curves per master + per-channel with presets.
"""

import numpy as np
import torch

from ..utils import gpu_color as G
from ..data.grading_presets import TONE_CURVE_PRESETS, TONE_CURVE_PRESET_NAMES


# Fixed x-positions for the 5 control points
CONTROL_X = [0.0, 0.25, 0.5, 0.75, 1.0]


class ToneCurve:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (TONE_CURVE_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a tone curve preset or use Custom for manual control"
                }),
            },
            "optional": {
                # Master RGB curve — 5 control points
                "shadows": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Curve point at 0% luminance (black point)"
                }),
                "darks": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Curve point at 25% luminance"
                }),
                "midtones": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Curve point at 50% luminance"
                }),
                "lights": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Curve point at 75% luminance"
                }),
                "highlights": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Curve point at 100% luminance (white point)"
                }),
                # Per-channel shadow/highlight offsets
                "red_shadows": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Red channel shadow offset"
                }),
                "red_highlights": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Red channel highlight offset"
                }),
                "green_shadows": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Green channel shadow offset"
                }),
                "green_highlights": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Green channel highlight offset"
                }),
                "blue_shadows": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Blue channel shadow offset"
                }),
                "blue_highlights": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Blue channel highlight offset"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and curved (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def _build_points(self, master_offsets, channel_shadow=0.0, channel_highlight=0.0):
        """
        Build control points for one channel.
        master_offsets: 5 values for shadows/darks/midtones/lights/highlights
        channel_shadow/highlight: per-channel offsets at positions 0 and 4
        Returns list of (x, y) tuples.
        """
        points = []
        for i, x in enumerate(CONTROL_X):
            y = x + master_offsets[i] / 100.0
            # Add per-channel offsets at shadows (i=0) and highlights (i=4)
            if i == 0:
                y += channel_shadow / 100.0
            elif i == 4:
                y += channel_highlight / 100.0
            # Interpolate per-channel offset for intermediate points
            elif channel_shadow != 0.0 or channel_highlight != 0.0:
                t = i / 4.0
                y += ((1.0 - t) * channel_shadow + t * channel_highlight) / 100.0
            points.append((x, np.clip(y, 0.0, 1.0)))
        return points

    def execute(self, image, preset="Custom (manual)",
                shadows=0.0, darks=0.0, midtones=0.0, lights=0.0, highlights=0.0,
                red_shadows=0.0, red_highlights=0.0,
                green_shadows=0.0, green_highlights=0.0,
                blue_shadows=0.0, blue_highlights=0.0,
                strength=1.0):

        if strength <= 0.0:
            return (image,)

        # Build master offsets from preset + manual sliders
        master = [shadows, darks, midtones, lights, highlights]

        if preset != "Custom (manual)" and preset in TONE_CURVE_PRESETS:
            p = TONE_CURVE_PRESETS[preset]
            master = [master[i] + p.master[i] for i in range(5)]
            red_shadows += p.red_shadows
            red_highlights += p.red_highlights
            green_shadows += p.green_shadows
            green_highlights += p.green_highlights
            blue_shadows += p.blue_shadows
            blue_highlights += p.blue_highlights

        # Check if anything changed from identity
        all_zero = (
            all(abs(m) < 0.5 for m in master) and
            abs(red_shadows) < 0.5 and abs(red_highlights) < 0.5 and
            abs(green_shadows) < 0.5 and abs(green_highlights) < 0.5 and
            abs(blue_shadows) < 0.5 and abs(blue_highlights) < 0.5
        )
        if all_zero:
            return (image,)

        # Build per-channel control points
        r_points = self._build_points(master, red_shadows, red_highlights)
        g_points = self._build_points(master, green_shadows, green_highlights)
        b_points = self._build_points(master, blue_shadows, blue_highlights)

        print(f"[Darkroom] Tone Curve: preset={preset}, strength={strength}")

        def _pipeline(x):
            linear = G.srgb_to_linear(x)
            result = torch.stack([
                G.cubic_spline_curve(linear[..., 0], r_points),
                G.cubic_spline_curve(linear[..., 1], g_points),
                G.cubic_spline_curve(linear[..., 2], b_points),
            ], dim=-1)
            result = G.linear_to_srgb(result)
            return G.blend(x, result, strength)

        return (G.run_on_device(_pipeline, image),)


NODE_CLASS_MAPPINGS = {"DarkroomToneCurve": ToneCurve}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomToneCurve": "Tone Curve"}
