"""
Lift Gamma Gain node for ComfyUI-Darkroom.
DaVinci Resolve-style primary color correction with per-channel control.
"""

import numpy as np

from ..utils import gpu_color as G


class LiftGammaGain:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Lift (shadows) — additive offset in dark regions
                "lift_r": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Red channel lift (shadow color offset)"
                }),
                "lift_g": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Green channel lift (shadow color offset)"
                }),
                "lift_b": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blue channel lift (shadow color offset)"
                }),
                "lift_master": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Master lift — applies to all channels"
                }),
                # Gamma (midtones) — power function
                "gamma_r": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "Red channel gamma (midtone brightness). >1 = brighter mids"
                }),
                "gamma_g": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "Green channel gamma (midtone brightness)"
                }),
                "gamma_b": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "Blue channel gamma (midtone brightness)"
                }),
                "gamma_master": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "Master gamma — multiplies into all channels"
                }),
                # Gain (highlights) — multiplier
                "gain_r": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Red channel gain (highlight multiplier)"
                }),
                "gain_g": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Green channel gain (highlight multiplier)"
                }),
                "gain_b": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Blue channel gain (highlight multiplier)"
                }),
                "gain_master": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Master gain — multiplies into all channels"
                }),
                # Offset (global) — flat shift
                "offset_r": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                    "tooltip": "Red channel offset (flat global shift)"
                }),
                "offset_g": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                    "tooltip": "Green channel offset (flat global shift)"
                }),
                "offset_b": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                    "tooltip": "Blue channel offset (flat global shift)"
                }),
                "offset_master": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                    "tooltip": "Master offset — adds to all channels"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and graded (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def execute(self, image, lift_r=0.0, lift_g=0.0, lift_b=0.0, lift_master=0.0,
                gamma_r=1.0, gamma_g=1.0, gamma_b=1.0, gamma_master=1.0,
                gain_r=1.0, gain_g=1.0, gain_b=1.0, gain_master=1.0,
                offset_r=0.0, offset_g=0.0, offset_b=0.0, offset_master=0.0,
                strength=1.0):

        if strength <= 0.0:
            return (image,)

        # Combine master into per-channel values
        lift_rgb = np.array([lift_r + lift_master, lift_g + lift_master, lift_b + lift_master], dtype=np.float32)
        gamma_rgb = np.array([gamma_r * gamma_master, gamma_g * gamma_master, gamma_b * gamma_master], dtype=np.float32)
        gain_rgb = np.array([gain_r * gain_master, gain_g * gain_master, gain_b * gain_master], dtype=np.float32)
        offset_rgb = np.array([offset_r + offset_master, offset_g + offset_master, offset_b + offset_master], dtype=np.float32)

        # Check if anything changed from identity
        is_identity = (
            np.allclose(lift_rgb, 0.0, atol=0.005) and
            np.allclose(gamma_rgb, 1.0, atol=0.005) and
            np.allclose(gain_rgb, 1.0, atol=0.005) and
            np.allclose(offset_rgb, 0.0, atol=0.002)
        )
        if is_identity:
            return (image,)

        print(f"[Darkroom] Lift Gamma Gain: L={lift_rgb}, G={gamma_rgb}, Gn={gain_rgb}, O={offset_rgb}, strength={strength}")

        def _pipeline(x):
            linear = G.srgb_to_linear(x)
            graded = G.apply_lgg(linear, lift_rgb, gamma_rgb, gain_rgb, offset_rgb)
            result = G.linear_to_srgb(graded)
            return G.blend(x, result, strength)

        return (G.run_on_device(_pipeline, image),)


NODE_CLASS_MAPPINGS = {"DarkroomLiftGammaGain": LiftGammaGain}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLiftGammaGain": "Lift Gamma Gain"}
