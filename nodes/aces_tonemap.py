"""
ACES Tonemap node for ComfyUI-Darkroom.
Apply industry-standard tonemapping curves to get the "ACES look" and other
cinematic/filmic tone responses. Input can be scene-referred (HDR) or standard range.
"""

from ..utils.colorspace import TONEMAP_NAMES, _SRGB_TO_ACES, _ACES_TO_SRGB
from ..utils.gpu_color import (
    run_on_device, blend, apply_matrix, srgb_decode, srgb_encode,
    TONEMAP_CURVES, reinhard_extended,
)


class ACESTonemap:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve": (TONEMAP_NAMES, {
                    "default": "ACES Filmic (Narkowicz)",
                    "tooltip": "Tonemapping curve. ACES Filmic and Hill are the standard Academy look. "
                               "AgX is the modern Blender default. Filmic (Uncharted 2) is the classic game look"
                }),
            },
            "optional": {
                "input_is_linear": (["No (sRGB input)", "Yes (linear/HDR input)"], {
                    "default": "No (sRGB input)",
                    "tooltip": "Is the input already in linear light? If No, sRGB gamma is removed first. "
                               "Use Yes if feeding from Color Space Transform or HDR sources"
                }),
                "exposure_bias": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in EV stops before tonemapping. "
                               "Positive = brighter, negative = darker"
                }),
                "tonemap_in_aces": (["Yes", "No"], {
                    "default": "Yes",
                    "tooltip": "Convert to ACEScg before tonemapping (recommended for ACES curves). "
                               "Set to No if input is already in the expected color space"
                }),
                "white_point": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5,
                    "tooltip": "White point for Reinhard Extended curve (ignored by other curves)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and tonemapped (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, image, curve="ACES Filmic (Narkowicz)",
                input_is_linear="No (sRGB input)", exposure_bias=0.0,
                tonemap_in_aces="Yes", white_point=4.0, strength=1.0):

        if strength <= 0.0:
            return (image,)

        is_linear = input_is_linear.startswith("Yes")
        use_aces_gamut = tonemap_in_aces == "Yes"
        tonemap_fn = TONEMAP_CURVES[curve]

        print(f"[Darkroom] ACES Tonemap: {curve}, exposure={exposure_bias:+.1f}EV, "
              f"ACES gamut={'yes' if use_aces_gamut else 'no'}")

        def process(img):
            original = img

            # Step 1: get to linear light
            linear = img if is_linear else srgb_decode(img)

            # Step 2: exposure adjustment (in linear light, before tonemapping)
            if abs(exposure_bias) > 0.01:
                linear = linear * (2.0 ** exposure_bias)

            # Step 3: convert to ACEScg if requested
            if use_aces_gamut:
                linear = apply_matrix(linear, _SRGB_TO_ACES)

            # Step 4: apply tonemapping curve
            # Reinhard Extended needs the white_point parameter
            if curve == "Reinhard Extended":
                tonemapped = reinhard_extended(linear.clamp(min=0.0), white_point)
            else:
                tonemapped = tonemap_fn(linear.clamp(min=0.0))

            # Step 5: convert back from ACEScg to sRGB linear
            if use_aces_gamut:
                tonemapped = apply_matrix(tonemapped, _ACES_TO_SRGB).clamp(0.0, 1.0)

            # Step 6: back to sRGB display
            result = srgb_encode(tonemapped)

            return blend(original, result, strength)

        return (run_on_device(process, image),)


NODE_CLASS_MAPPINGS = {"DarkroomACESTonemap": ACESTonemap}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomACESTonemap": "ACES Tonemap"}
