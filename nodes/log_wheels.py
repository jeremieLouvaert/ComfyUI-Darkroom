"""
Log Wheels node for ComfyUI-Darkroom.
DaVinci Resolve Log-mode color grading with soft zone masks in log2 space.
"""

from ..utils import gpu_color as G
from ..data.grading_presets import LOG_WHEELS_PRESETS, LOG_WHEELS_PRESET_NAMES


class LogWheels:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (LOG_WHEELS_PRESET_NAMES, {
                    "default": "Custom (manual)",
                    "tooltip": "Select a log grading preset or use Custom"
                }),
            },
            "optional": {
                # Shadow zone
                "shadow_hue": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Shadow tint direction (0=red, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta)"
                }),
                "shadow_saturation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Shadow tint color intensity"
                }),
                "shadow_density": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Shadow brightness shift (negative = darker)"
                }),
                # Midtone zone
                "midtone_hue": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Midtone tint direction"
                }),
                "midtone_saturation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Midtone tint color intensity"
                }),
                "midtone_density": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Midtone brightness shift"
                }),
                # Highlight zone
                "highlight_hue": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Highlight tint direction"
                }),
                "highlight_saturation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Highlight tint color intensity"
                }),
                "highlight_density": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Highlight brightness shift"
                }),
                # Zone boundaries
                "shadow_range": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 0.40, "step": 0.01,
                    "tooltip": "Upper boundary of shadow zone in linear light"
                }),
                "highlight_range": ("FLOAT", {
                    "default": 0.85, "min": 0.60, "max": 0.95, "step": 0.01,
                    "tooltip": "Lower boundary of highlight zone in linear light"
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

    def execute(self, image, preset="Custom (manual)",
                shadow_hue=0.0, shadow_saturation=0.0, shadow_density=0.0,
                midtone_hue=0.0, midtone_saturation=0.0, midtone_density=0.0,
                highlight_hue=0.0, highlight_saturation=0.0, highlight_density=0.0,
                shadow_range=0.15, highlight_range=0.85, strength=1.0):

        if strength <= 0.0:
            return (image,)

        # Build values from preset + manual
        s_hue, s_sat, s_den = shadow_hue, shadow_saturation, shadow_density
        m_hue, m_sat, m_den = midtone_hue, midtone_saturation, midtone_density
        h_hue, h_sat, h_den = highlight_hue, highlight_saturation, highlight_density
        s_range, h_range = shadow_range, highlight_range

        if preset == "Neutral \u2014 reset all":
            # Reset everything to zero
            return (image,)

        if preset != "Custom (manual)" and preset in LOG_WHEELS_PRESETS:
            p = LOG_WHEELS_PRESETS[preset]
            if shadow_saturation < 0.5:
                s_hue = p.shadow_hue
            s_sat += p.shadow_saturation
            s_den += p.shadow_density
            if midtone_saturation < 0.5:
                m_hue = p.midtone_hue
            m_sat += p.midtone_saturation
            m_den += p.midtone_density
            if highlight_saturation < 0.5:
                h_hue = p.highlight_hue
            h_sat += p.highlight_saturation
            h_den += p.highlight_density
            s_range = p.shadow_range
            h_range = p.highlight_range

        # Check if anything is active
        has_tint = s_sat >= 0.5 or m_sat >= 0.5 or h_sat >= 0.5
        has_density = abs(s_den) >= 0.5 or abs(m_den) >= 0.5 or abs(h_den) >= 0.5
        if not has_tint and not has_density:
            return (image,)

        print(f"[Darkroom] Log Wheels: preset={preset}, shadow_range={s_range}, "
              f"highlight_range={h_range}, strength={strength}")

        def _pipeline(x):
            linear = G.srgb_to_linear(x)
            lum = G.luminance_rec709(linear)

            # Build soft log-space zone masks
            shadow_mask, midtone_mask, highlight_mask = G.log_zone_masks(lum, s_range, h_range)

            result = linear

            # Apply color tints per zone
            if s_sat >= 0.5:
                result = G.apply_color_tint_to_zone(result, shadow_mask, s_hue, s_sat)
            if m_sat >= 0.5:
                result = G.apply_color_tint_to_zone(result, midtone_mask, m_hue, m_sat)
            if h_sat >= 0.5:
                result = G.apply_color_tint_to_zone(result, highlight_mask, h_hue, h_sat)

            # Apply density (brightness) shifts per zone
            if abs(s_den) >= 0.5:
                density_factor = 1.0 + s_den / 100.0 * 0.5
                result = result * (1.0 + shadow_mask[..., None] * (density_factor - 1.0))

            if abs(m_den) >= 0.5:
                density_factor = 1.0 + m_den / 100.0 * 0.3
                result = result * (1.0 + midtone_mask[..., None] * (density_factor - 1.0))

            if abs(h_den) >= 0.5:
                density_factor = 1.0 + h_den / 100.0 * 0.5
                result = result * (1.0 + highlight_mask[..., None] * (density_factor - 1.0))

            result = result.clamp(0.0, 1.0)
            result = G.linear_to_srgb(result)
            return G.blend(x, result, strength)

        return (G.run_on_device(_pipeline, image),)


NODE_CLASS_MAPPINGS = {"DarkroomLogWheels": LogWheels}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLogWheels": "Log Wheels"}
