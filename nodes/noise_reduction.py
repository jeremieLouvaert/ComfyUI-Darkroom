"""
Noise Reduction node for ComfyUI-Darkroom.
Lightroom-style luminance + color noise reduction with detail preservation.
Uses guided filter for luminance and gaussian chrominance smoothing.
"""

import torch

from ..utils.gpu_color import run_on_device, blend
from ..utils.gpu_filters import gaussian_filter, uniform_filter

GUIDE_DTYPE = torch.float32   # the box-filter statistics; see the golden gate


def _guided_filter(guide, target, radius, eps):
    """
    Fast guided filter (box-filter implementation).
    Edge-preserving smooth of 'target' using 'guide' as structure reference.
    Both inputs: (H, W) float32.
    """
    size = 2 * radius + 1
    g, t = guide.to(GUIDE_DTYPE), target.to(GUIDE_DTYPE)
    mean_g = uniform_filter(g, size)
    mean_t = uniform_filter(t, size)
    corr_gt = uniform_filter(g * t, size)
    var_g = uniform_filter(g * g, size) - mean_g * mean_g

    a = (corr_gt - mean_g * mean_t) / (var_g + eps)
    b = mean_t - a * mean_g

    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)

    return (mean_a * g + mean_b).to(torch.float32)


def _rgb_to_ycbcr(img):
    """Convert RGB (0-1) to Y, Cb, Cr channels."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / (2.0 * (1.0 - 0.0722) + 1e-10)
    cr = (r - y) / (2.0 * (1.0 - 0.2126) + 1e-10)
    return y, cb, cr


def _ycbcr_to_rgb(y, cb, cr):
    """Convert Y, Cb, Cr back to RGB (0-1)."""
    r = y + cr * 2.0 * (1.0 - 0.2126)
    g = y - cb * 2.0 * (1.0 - 0.0722) * 0.0722 / 0.7152 - cr * 2.0 * (1.0 - 0.2126) * 0.2126 / 0.7152
    b = y + cb * 2.0 * (1.0 - 0.0722)
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)


# Presets: (luminance_amount, luminance_detail, luminance_contrast, color_amount, color_detail)
NR_PRESETS = {
    "Custom": None,
    "Light — subtle cleanup": (30, 60, 60, 40, 50),
    "Medium — general purpose": (55, 45, 50, 55, 40),
    "Heavy — noisy image rescue": (80, 30, 40, 75, 30),
    "Color only — fix chroma noise": (0, 50, 50, 70, 35),
    "Luminance only — preserve color": (60, 40, 50, 0, 50),
    "Portrait — smooth skin, keep detail": (45, 65, 55, 50, 50),
    "High ISO — aggressive denoise": (90, 20, 35, 85, 25),
}

PRESET_LIST = list(NR_PRESETS.keys())


class NoiseReduction:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (PRESET_LIST, {
                    "default": "Medium — general purpose",
                    "tooltip": "Quick presets. Select 'Custom' to use manual sliders"
                }),
            },
            "optional": {
                "luminance_amount": ("FLOAT", {
                    "default": 55.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Luminance noise reduction strength (only used with Custom preset)"
                }),
                "luminance_detail": ("FLOAT", {
                    "default": 45.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Detail preservation. Higher = keep more fine detail"
                }),
                "luminance_contrast": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Local contrast preservation. Higher = keep more tonal transitions"
                }),
                "color_amount": ("FLOAT", {
                    "default": 55.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Color noise reduction (chrominance smoothing)"
                }),
                "color_detail": ("FLOAT", {
                    "default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Color detail preservation. Higher = keep more chromatic edges"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and denoised (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Raw"

    def execute(self, image, preset="Medium — general purpose",
                luminance_amount=55.0, luminance_detail=45.0,
                luminance_contrast=50.0, color_amount=55.0, color_detail=40.0,
                strength=1.0):

        # Apply preset values (override sliders unless Custom)
        p = NR_PRESETS.get(preset)
        if p is not None:
            luminance_amount, luminance_detail, luminance_contrast, color_amount, color_detail = p

        if strength <= 0.0 or (luminance_amount < 0.5 and color_amount < 0.5):
            return (image,)

        def one(img):
            original = img
            h, w = img.shape[:2]
            result = img.clone()

            # Scale radius relative to image size (reference: 1024px long edge)
            ref_size = 1024.0
            scale = max(h, w) / ref_size

            # --- Luminance noise reduction ---
            if luminance_amount > 0.5:
                y, cb, cr = _rgb_to_ycbcr(result)

                # Guided filter radius — much more aggressive range
                # amount 50 → radius ~15, amount 100 → radius ~30 at 1024px
                radius = max(3, int(luminance_amount / 100.0 * 30.0 * scale))

                # eps controls edge sensitivity:
                # Low detail = tiny eps = aggressive smoothing
                # High detail = larger eps = preserve more
                # Inverted: detail slider controls PRESERVATION, not smoothing
                detail_factor = luminance_detail / 100.0
                eps = 0.00005 + detail_factor * detail_factor * 0.04

                # Multi-pass for stronger effect at high amounts
                passes = 1 if luminance_amount < 50 else (2 if luminance_amount < 80 else 3)

                y_filtered = y.clone()
                for _ in range(passes):
                    y_filtered = _guided_filter(y, y_filtered, radius, eps)

                # Contrast preservation: blend back local contrast
                contrast_factor = luminance_contrast / 100.0
                if contrast_factor > 0.1:
                    large_radius = max(radius * 3, int(25 * scale))
                    y_coarse = _guided_filter(y, y_filtered, large_radius, eps * 4)
                    local_contrast = y_filtered - y_coarse
                    y_filtered = y_filtered + local_contrast * contrast_factor * 0.7

                result = _ycbcr_to_rgb(y_filtered, cb, cr)

            # --- Color noise reduction ---
            if color_amount > 0.5:
                y, cb, cr = _rgb_to_ycbcr(result)

                # Much more aggressive chrominance blur
                # amount 50 → sigma ~8, amount 100 → sigma ~16 at 1024px
                sigma = color_amount / 100.0 * 16.0 * scale

                detail_factor = color_detail / 100.0
                if detail_factor > 0.3:
                    # Edge-aware: guided filter on chrominance
                    chroma_radius = max(3, int(sigma * 2.0))
                    chroma_eps = 0.0005 + detail_factor * 0.015
                    cb_smooth = _guided_filter(y, cb, chroma_radius, chroma_eps)
                    cr_smooth = _guided_filter(y, cr, chroma_radius, chroma_eps)
                else:
                    # Low detail = pure gaussian (strongest smoothing)
                    cb_smooth = gaussian_filter(cb, sigma)
                    cr_smooth = gaussian_filter(cr, sigma)

                result = _ycbcr_to_rgb(y, cb_smooth, cr_smooth)

            return blend(original, result.clamp(0.0, 1.0), strength)

        return (run_on_device(lambda x: torch.stack([one(i) for i in x], 0), image),)


NODE_CLASS_MAPPINGS = {"DarkroomNoiseReduction": NoiseReduction}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomNoiseReduction": "Noise Reduction"}
