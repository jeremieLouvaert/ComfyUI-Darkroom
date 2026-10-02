"""
Skin Tone Uniformity node for ComfyUI-Darkroom.
Capture One-style skin tone evening: smooths color variations in skin
while fully preserving luminance texture and fine detail.

The key insight: local chrominance blur barely works because skin pixels
already have similar hues. Instead, we PULL each skin pixel toward the
AREA-WEIGHTED MEAN skin color. This evens out redness, sallowness, and
uneven tan while preserving luminance detail completely.

Runs on ComfyUI's device via utils/gpu_color (one upload, one download).
"""

import torch

from ..utils.gpu_color import rgb_to_hsl, hsl_to_rgb, blend, run_on_device
from ..utils.torch_ops import skin_mask as _skin_mask_torch, gaussian_blur_2d


# Preset definitions: (hue_center, hue_width, sat_min, sat_max, lum_min, lum_max)
SKIN_PRESETS = {
    "Universal — all skin tones": (25.0, 45.0, 0.08, 0.85, 0.10, 0.92),
    "Light / Fair skin": (20.0, 30.0, 0.10, 0.65, 0.40, 0.92),
    "Medium / Olive skin": (25.0, 35.0, 0.12, 0.75, 0.25, 0.80),
    "Dark / Deep skin": (22.0, 40.0, 0.10, 0.80, 0.08, 0.55),
    "Warm / Golden skin": (30.0, 35.0, 0.15, 0.80, 0.20, 0.85),
    "Cool / Pink skin": (12.0, 30.0, 0.10, 0.70, 0.30, 0.90),
    "Custom": None,
}

PRESET_LIST = list(SKIN_PRESETS.keys())


class SkinToneUniformity:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (PRESET_LIST, {
                    "default": "Universal — all skin tones",
                    "tooltip": (
                        "Skin tone range preset. Each covers different skin types. "
                        "Select 'Custom' to set HSL ranges manually"
                    )
                }),
                "amount": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "How much to even out skin color (0 = off, 100 = fully uniform)"
                }),
            },
            "optional": {
                "smoothing_radius": ("FLOAT", {
                    "default": 60.0, "min": 10.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Color averaging radius. Lower = local patches only, Higher = global average"
                }),
                "hue_center": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "(Custom only) Center hue for skin detection"
                }),
                "hue_width": ("FLOAT", {
                    "default": 45.0, "min": 10.0, "max": 90.0, "step": 1.0,
                    "tooltip": "(Custom only) Hue range width"
                }),
                "saturation_min": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "(Custom only) Minimum saturation"
                }),
                "saturation_max": ("FLOAT", {
                    "default": 0.85, "min": 0.3, "max": 1.0, "step": 0.01,
                    "tooltip": "(Custom only) Maximum saturation"
                }),
                "luminance_min": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "(Custom only) Minimum luminance"
                }),
                "luminance_max": ("FLOAT", {
                    "default": 0.92, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "(Custom only) Maximum luminance"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and corrected (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "mask_preview",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, preset="Universal — all skin tones", amount=60.0,
                smoothing_radius=60.0, hue_center=25.0, hue_width=45.0,
                saturation_min=0.08, saturation_max=0.85,
                luminance_min=0.10, luminance_max=0.92,
                strength=1.0):

        # Apply preset (override manual sliders unless Custom)
        p = SKIN_PRESETS.get(preset)
        if p is not None:
            hue_center, hue_width, saturation_min, saturation_max, luminance_min, luminance_max = p

        if strength <= 0.0 or amount < 0.5:
            return (image, image)

        def even_out(batch):
            outs = []
            for i in range(batch.shape[0]):
                img = batch[i]  # (H, W, C); the blurs are 2D, so one image at a time
                h_img, w_img = img.shape[0], img.shape[1]

                h, s, l = rgb_to_hsl(img)

                # Build soft skin mask
                mask = _skin_mask_torch(h, s, l, hue_center, hue_width,
                                        saturation_min, saturation_max,
                                        luminance_min, luminance_max)

                # Smooth mask edges
                mask_sigma = max(h_img, w_img) * 0.015
                mask_smooth = gaussian_blur_2d(mask, sigma=mask_sigma).clamp(0.0, 1.0)

                # --- STRATEGY: Pull toward weighted-average skin color ---
                ref_size = 1024.0
                scale = max(h_img, w_img) / ref_size
                target_sigma = (smoothing_radius / 100.0) * 40.0 * scale

                # Hue averaging in circular space (handles wraparound)
                h_rad = h * (3.141592653589793 / 180.0)
                h_sin = torch.sin(h_rad)
                h_cos = torch.cos(h_rad)

                # Weight by mask so non-skin doesn't pollute the average
                weighted_sin = h_sin * mask_smooth
                weighted_cos = h_cos * mask_smooth

                # Large blur to compute local weighted average
                avg_sin = gaussian_blur_2d(weighted_sin, sigma=target_sigma)
                avg_cos = gaussian_blur_2d(weighted_cos, sigma=target_sigma)
                avg_weight = gaussian_blur_2d(mask_smooth, sigma=target_sigma) + 1e-10

                # Normalized weighted average hue
                target_h = (torch.atan2(avg_sin / avg_weight, avg_cos / avg_weight)
                            * (180.0 / 3.141592653589793)) % 360.0

                # Weighted average saturation
                weighted_s = s * mask_smooth
                avg_s = gaussian_blur_2d(weighted_s, sigma=target_sigma)
                target_s = (avg_s / avg_weight).clamp(0.0, 1.0)

                # Amount controls how far we pull toward the target
                pull = amount / 100.0

                # Compute hue difference (circular, shortest path)
                h_diff = target_h - h
                h_diff = torch.where(h_diff > 180, h_diff - 360, h_diff)
                h_diff = torch.where(h_diff < -180, h_diff + 360, h_diff)

                # Apply: pull hue and saturation toward target, weighted by mask
                h_new = (h + mask_smooth * pull * h_diff) % 360.0
                s_new = (s + mask_smooth * pull * (target_s - s)).clamp(0.0, 1.0)

                # Reconstruct with ORIGINAL luminance (preserves all texture/detail)
                result = hsl_to_rgb(h_new, s_new, l).clamp(0.0, 1.0)
                out = blend(img, result, strength)

                mask_vis = mask.unsqueeze(-1).expand(-1, -1, 3)
                outs.append(torch.cat([out, mask_vis], dim=-1))
            return torch.stack(outs, dim=0)

        both = run_on_device(even_out, image)
        return (both[..., :3].contiguous(), both[..., 3:].contiguous())


NODE_CLASS_MAPPINGS = {"DarkroomSkinToneUniformity": SkinToneUniformity}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomSkinToneUniformity": "Skin Tone Uniformity"}
