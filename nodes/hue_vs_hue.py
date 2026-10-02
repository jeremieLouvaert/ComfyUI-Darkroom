"""
Hue vs Hue node for ComfyUI-Darkroom.
Remap specific hue ranges to different hues with presets and per-band control.
"""

from ..utils.gpu_color import (
    srgb_to_linear, linear_to_srgb, blend, rgb_to_hsl, hsl_to_rgb,
    hue_range_mask, run_on_device,
)
from ..data.grading_presets import HUE_VS_HUE_PRESETS, HUE_VS_HUE_PRESET_NAMES


HUE_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 270.0,
    "magenta": 330.0,
}


class HueVsHue:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (HUE_VS_HUE_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a hue remapping preset or use Custom for manual control"
                }),
            },
            "optional": {
                "red_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift red hues (degrees)"
                }),
                "orange_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift orange hues (degrees)"
                }),
                "yellow_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift yellow hues (degrees)"
                }),
                "green_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift green hues (degrees)"
                }),
                "aqua_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift aqua/cyan hues (degrees)"
                }),
                "blue_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift blue hues (degrees)"
                }),
                "purple_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift purple hues (degrees)"
                }),
                "magenta_shift": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Shift magenta hues (degrees)"
                }),
                "feather": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How smoothly shifts blend into neighboring hues"
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
    CATEGORY = "AKURATE/Darkroom/Grading"

    def execute(self, image, preset="Custom (manual)", red_shift=0.0,
                orange_shift=0.0, yellow_shift=0.0, green_shift=0.0,
                aqua_shift=0.0, blue_shift=0.0, purple_shift=0.0,
                magenta_shift=0.0, feather=0.5, strength=1.0):

        if strength <= 0.0:
            return (image,)

        adjustments = {
            "red": red_shift,
            "orange": orange_shift,
            "yellow": yellow_shift,
            "green": green_shift,
            "aqua": aqua_shift,
            "blue": blue_shift,
            "purple": purple_shift,
            "magenta": magenta_shift,
        }

        if preset != "Custom (manual)" and preset in HUE_VS_HUE_PRESETS:
            p = HUE_VS_HUE_PRESETS[preset]
            adjustments["red"] += p.red
            adjustments["orange"] += p.orange
            adjustments["yellow"] += p.yellow
            adjustments["green"] += p.green
            adjustments["aqua"] += p.aqua
            adjustments["blue"] += p.blue
            adjustments["purple"] += p.purple
            adjustments["magenta"] += p.magenta

        active = [(name, val) for name, val in adjustments.items() if abs(val) > 0.1]
        if not active:
            return (image,)

        print(f"[Darkroom] Hue vs Hue: preset={preset}, {len(active)} active bands, strength={strength}")

        def grade(img):
            linear = srgb_to_linear(img)
            h, s, l = rgb_to_hsl(linear)

            for hue_name, shift_value in active:
                center = HUE_CENTERS[hue_name]
                mask = hue_range_mask(h, center, width=45.0, softness=feather)
                h = h + mask * shift_value
                h = h % 360.0

            result = hsl_to_rgb(h, s, l)
            result = linear_to_srgb(result)
            return blend(img, result, strength)

        return (run_on_device(grade, image),)


NODE_CLASS_MAPPINGS = {"DarkroomHueVsHue": HueVsHue}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHueVsHue": "Hue vs Hue"}
