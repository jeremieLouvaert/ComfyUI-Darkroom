"""
Hue vs Sat node for ComfyUI-Darkroom.
Adjust saturation per hue range with presets and per-band manual control.
"""

from ..utils.gpu_color import (
    srgb_to_linear, linear_to_srgb, blend, rgb_to_hsl, hsl_to_rgb,
    hue_range_mask, run_on_device,
)
from ..data.grading_presets import HUE_VS_SAT_PRESETS, HUE_VS_SAT_PRESET_NAMES


# Same 8 hue centers as HSL Selective for consistency
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


class HueVsSat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (HUE_VS_SAT_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a saturation-per-hue preset or use Custom for manual control"
                }),
            },
            "optional": {
                "red_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Red hue saturation adjustment"
                }),
                "orange_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Orange hue saturation adjustment"
                }),
                "yellow_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Yellow hue saturation adjustment"
                }),
                "green_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Green hue saturation adjustment"
                }),
                "aqua_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Aqua/Cyan hue saturation adjustment"
                }),
                "blue_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Blue hue saturation adjustment"
                }),
                "purple_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Purple hue saturation adjustment"
                }),
                "magenta_saturation": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Magenta hue saturation adjustment"
                }),
                "feather": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How smoothly adjustments blend into neighboring hues"
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

    def execute(self, image, preset="Custom (manual)", red_saturation=0.0,
                orange_saturation=0.0, yellow_saturation=0.0, green_saturation=0.0,
                aqua_saturation=0.0, blue_saturation=0.0, purple_saturation=0.0,
                magenta_saturation=0.0, feather=0.5, strength=1.0):

        if strength <= 0.0:
            return (image,)

        # Build adjustments from preset + manual sliders
        adjustments = {
            "red": red_saturation,
            "orange": orange_saturation,
            "yellow": yellow_saturation,
            "green": green_saturation,
            "aqua": aqua_saturation,
            "blue": blue_saturation,
            "purple": purple_saturation,
            "magenta": magenta_saturation,
        }

        # Apply preset values (manual sliders add on top)
        if preset != "Custom (manual)" and preset in HUE_VS_SAT_PRESETS:
            p = HUE_VS_SAT_PRESETS[preset]
            adjustments["red"] += p.red
            adjustments["orange"] += p.orange
            adjustments["yellow"] += p.yellow
            adjustments["green"] += p.green
            adjustments["aqua"] += p.aqua
            adjustments["blue"] += p.blue
            adjustments["purple"] += p.purple
            adjustments["magenta"] += p.magenta

        # Check if any adjustments are active
        active = [(name, val) for name, val in adjustments.items() if abs(val) > 0.5]
        if not active:
            return (image,)

        print(f"[Darkroom] Hue vs Sat: preset={preset}, {len(active)} active bands, strength={strength}")

        def grade(img):
            linear = srgb_to_linear(img)
            h, s, l = rgb_to_hsl(linear)

            for hue_name, adj_value in active:
                center = HUE_CENTERS[hue_name]
                mask = hue_range_mask(h, center, width=45.0, softness=feather)
                # Multiplicative saturation adjustment
                s = s * (1.0 + mask * (adj_value / 100.0))
                s = s.clamp(0.0, 1.0)

            result = hsl_to_rgb(h, s, l)
            result = linear_to_srgb(result)
            return blend(img, result, strength)

        return (run_on_device(grade, image),)


NODE_CLASS_MAPPINGS = {"DarkroomHueVsSat": HueVsSat}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHueVsSat": "Hue vs Sat"}
