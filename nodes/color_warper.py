"""
Color Warper node for ComfyUI-Darkroom.
2D hue+saturation region warping with multi-region presets.

Runs on ComfyUI's device via utils/gpu_color (one upload, one download).
"""

from ..utils.gpu_color import (
    srgb_to_linear, linear_to_srgb, rgb_to_hsl, hsl_to_rgb,
    hue_range_mask, blend, run_on_device,
)
from ..utils.torch_ops import sat_range_weight
from ..data.grading_presets import COLOR_WARPER_PRESETS, COLOR_WARPER_PRESET_NAMES


class ColorWarper:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (COLOR_WARPER_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a color warping preset or use Custom for single-region manual control"
                }),
            },
            "optional": {
                # Source region selection
                "source_hue": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Center hue of the region to warp"
                }),
                "source_hue_width": ("FLOAT", {
                    "default": 60.0, "min": 10.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Width of hue range to affect (degrees)"
                }),
                "source_sat_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum saturation to affect"
                }),
                "source_sat_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Maximum saturation to affect"
                }),
                # Target shift
                "hue_shift": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Shift hue by this many degrees"
                }),
                "sat_shift": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Adjust saturation within the selected region"
                }),
                # Softness
                "feather": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Edge softness of the warped region"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and warped (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def _apply_region(self, h, s, l, src_hue, src_hue_width, src_sat_min, src_sat_max,
                      hue_shift, sat_shift, feather):
        """Apply a single warping region to HSL data (torch tensors)."""
        h_mask = hue_range_mask(h, src_hue, width=src_hue_width, softness=feather)
        s_mask = sat_range_weight(s, src_sat_min, src_sat_max, softness=0.05 + feather * 0.1)
        combined_mask = h_mask * s_mask

        if abs(hue_shift) > 0.1:
            h = (h + combined_mask * hue_shift) % 360.0

        if abs(sat_shift) > 0.5:
            s = (s * (1.0 + combined_mask * (sat_shift / 100.0))).clamp(0.0, 1.0)

        return h, s

    def execute(self, image, preset="Custom (manual)",
                source_hue=0.0, source_hue_width=60.0,
                source_sat_min=0.0, source_sat_max=1.0,
                hue_shift=0.0, sat_shift=0.0,
                feather=0.5, strength=1.0):

        if strength <= 0.0:
            return (image,)

        use_preset = preset != "Custom (manual)" and preset in COLOR_WARPER_PRESETS

        if not use_preset:
            if abs(hue_shift) < 0.1 and abs(sat_shift) < 0.5:
                return (image,)

        print(f"[Darkroom] Color Warper: preset={preset}, strength={strength}")

        def warp(img):
            linear = srgb_to_linear(img)
            h, s, l = rgb_to_hsl(linear)

            if use_preset:
                p = COLOR_WARPER_PRESETS[preset]
                for region in p.regions:
                    h, s = self._apply_region(
                        h, s, l,
                        region.src_hue, region.src_hue_width,
                        region.src_sat_min, region.src_sat_max,
                        region.hue_shift, region.sat_shift,
                        feather
                    )

                if abs(hue_shift) > 0.1 or abs(sat_shift) > 0.5:
                    h, s = self._apply_region(
                        h, s, l,
                        source_hue, source_hue_width,
                        source_sat_min, source_sat_max,
                        hue_shift, sat_shift,
                        feather
                    )
            else:
                h, s = self._apply_region(
                    h, s, l,
                    source_hue, source_hue_width,
                    source_sat_min, source_sat_max,
                    hue_shift, sat_shift,
                    feather
                )

            result = hsl_to_rgb(h, s, l)
            result = linear_to_srgb(result)
            return blend(img, result, strength)

        return (run_on_device(warp, image),)


NODE_CLASS_MAPPINGS = {"DarkroomColorWarper": ColorWarper}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomColorWarper": "Color Warper"}
