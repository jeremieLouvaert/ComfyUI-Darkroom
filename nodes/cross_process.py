"""
Cross Process node for ComfyUI-Darkroom.
Simulates developing film in the wrong chemistry.
"""

from ..data.cross_process_curves import CROSS_PROCESS_PROFILES, CROSS_PROCESS_NAMES
from ..utils.gpu_color import (run_on_device,
    srgb_to_linear, linear_to_srgb, apply_per_channel_curves,
    adjust_saturation, blend
)


class CrossProcess:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "process_type": (CROSS_PROCESS_NAMES, {
                    "default": CROSS_PROCESS_NAMES[0],
                    "tooltip": "Cross-processing method"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and full cross-process (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, process_type, strength):
        if strength <= 0.0:
            return (image,)

        profile = CROSS_PROCESS_PROFILES[process_type]
        print(f"[Darkroom] Cross Process: {profile.name}, strength={strength}")

        r_params = (profile.r_curve.toe_power, profile.r_curve.shoulder_power,
                    profile.r_curve.slope, profile.r_curve.pivot_x, profile.r_curve.pivot_y)
        g_params = (profile.g_curve.toe_power, profile.g_curve.shoulder_power,
                    profile.g_curve.slope, profile.g_curve.pivot_x, profile.g_curve.pivot_y)
        b_params = (profile.b_curve.toe_power, profile.b_curve.shoulder_power,
                    profile.b_curve.slope, profile.b_curve.pivot_x, profile.b_curve.pivot_y)

        def pipeline(original):
            # Linearize
            linear = srgb_to_linear(original)

            # Apply cross-process curves
            xpro = apply_per_channel_curves(linear, r_params, g_params, b_params)

            # Apply saturation modifier
            if abs(profile.saturation - 1.0) > 0.001:
                xpro = adjust_saturation(xpro, profile.saturation)

            # Back to display gamma
            result = linear_to_srgb(xpro)

            # Blend with original
            return blend(original, result, strength)

        return (run_on_device(pipeline, image),)


NODE_CLASS_MAPPINGS = {
    "DarkroomCrossProcess": CrossProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomCrossProcess": "Cross Process",
}
