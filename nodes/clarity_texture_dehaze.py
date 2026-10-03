"""
Clarity / Texture / Dehaze node for ComfyUI-Darkroom.
Three mid-frequency contrast tools in one node.
"""

import numpy as np
import torch

from ..utils.gpu_color import run_on_device, luminance_rec709, blend
from ..utils.gpu_filters import gaussian_filter, minimum_filter
from ..data.ai_mitigation_presets import AI_MITIGATION_CTD, CTD_PRESET_NAMES


class ClarityTextureDehaze:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clarity": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Large-scale local contrast (midtone punch). Negative = soften"
                }),
                "texture": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Fine detail enhancement without affecting edges. Negative = smooth"
                }),
                "dehaze": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Remove atmospheric haze. Negative = add haze effect"
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and adjusted (1)"
                }),
                "preset": (CTD_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "AI Mitigation presets stack with HSL Selective preset of the same tier. Manual sliders add on top"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, clarity=0.0, texture=0.0, dehaze=0.0, strength=1.0,
                preset="Custom (manual)"):
        if preset != "Custom (manual)" and preset in AI_MITIGATION_CTD:
            p = AI_MITIGATION_CTD[preset]
            clarity = clarity + p.clarity
            texture = texture + p.texture
            dehaze = dehaze + p.dehaze

        if strength <= 0.0 or (abs(clarity) < 0.5 and abs(texture) < 0.5 and abs(dehaze) < 0.5):
            return (image,)

        def one(img):
            """One image (H, W, 3) on the device: same steps as the numpy original."""
            result = img.clone()
            h, w = img.shape[:2]

            # All three tools operate via luminance to avoid color shifts
            lum = luminance_rec709(img)

            # --- Clarity: large-scale local contrast ---
            if abs(clarity) > 0.5:
                blur = gaussian_filter(lum, max(h, w) * 0.04)
                detail = lum - blur
                amount = clarity / 100.0 * 0.5
                lum_safe = lum + 1e-6        # applied per channel, preserves colour ratios
                result = result + (detail * amount)[..., None] * (result / lum_safe[..., None])

            # --- Texture: band-pass fine detail ---
            if abs(texture) > 0.5:
                texture_detail = gaussian_filter(lum, 1.0) - gaussian_filter(lum, max(h, w) * 0.01)
                amount = texture / 100.0 * 0.5
                lum_safe = luminance_rec709(result) + 1e-6
                result = result + (texture_detail * amount)[..., None] * (result / lum_safe[..., None])

            # --- Dehaze: dark channel prior (simplified) ---
            if abs(dehaze) > 0.5:
                dehaze_amount = dehaze / 100.0
                if dehaze_amount > 0:
                    min_rgb = result.amin(dim=-1)
                    dark_channel = minimum_filter(min_rgb, max(15, max(h, w) // 50))
                    # atmospheric light: mean of the brightest 0.1% of the dark channel
                    n_bright = max(1, int(h * w * 0.001))
                    # numpy's argpartition on the CPU: with tied values (flat areas) the
                    # chosen pixels must be the same ones the original picked
                    flat_dark = dark_channel.reshape(-1).cpu().numpy()
                    bright = torch.from_numpy(np.argpartition(flat_dark, -n_bright)[-n_bright:]).to(result.device)
                    atmos = result.reshape(-1, 3)[bright].mean(dim=0).clamp(min=0.1)
                    transmission = (1.0 - dehaze_amount * (dark_channel / (atmos.max() + 1e-6))).clamp(0.1, 1.0)
                    result = (result - atmos) / transmission[..., None] + atmos
                else:
                    # negative dehaze: add haze (blend toward mean brightness)
                    haze_amount = abs(dehaze_amount)
                    result = result * (1.0 - haze_amount * 0.5) + result.mean() * haze_amount * 0.5

            return blend(img, result.clamp(0.0, 1.0), strength)

        # per image: the dehaze statistics (brightest pixels, mean) are per image
        return (run_on_device(lambda x: torch.stack([one(i) for i in x], 0), image),)


NODE_CLASS_MAPPINGS = {"DarkroomClarityTextureDehaze": ClarityTextureDehaze}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomClarityTextureDehaze": "Clarity / Texture / Dehaze"}
