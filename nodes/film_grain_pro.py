"""
Film Grain Pro node for ComfyUI-Darkroom.

Physically-derived, resolution-independent film grain using the Newson et al.
(IPOL 2017) stochastic Boolean-model renderer. Ships alongside the fast
heuristic Film Grain node as the quality tier. See:
  utils/grain_newson.py
  docs/film-grain-pro-derivation.md
"""

import torch

from ..utils.grain_newson import render_film_grain


class FilmGrainPro:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_size": ("FLOAT", {
                    "default": 1.2, "min": 0.7, "max": 4.0, "step": 0.05,
                    "tooltip": "Grain radius in pixels at a 1024px reference. "
                               "Scaled with image size so grain stays a fixed "
                               "fraction of the frame at any resolution."
                }),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Grain intensity. Blends the grainy render with "
                               "the original (tone is preserved either way)."
                }),
                "radius_variation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Grain size variability (sigma/mean). 0 = uniform "
                               "grains, higher = mixed sizes (log-normal)."
                }),
                "color_grain": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = monochrome (luminance) grain, 1 = independent "
                               "per-channel grain (chroma grain, like color film)."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for a reproducible grain pattern."
                }),
            },
            "optional": {
                "monte_carlo_samples": ("INT", {
                    "default": 64, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Render quality. Higher = cleaner estimate (grain "
                               "is preserved), but slower. Cost is linear in this."
                }),
                "filter_sigma": ("FLOAT", {
                    "default": 0.8, "min": 0.4, "max": 2.0, "step": 0.1,
                    "tooltip": "Grain softness (Gaussian filter sigma in output "
                               "pixels). 0.8 is the physically-motivated default."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, grain_size, strength, radius_variation, color_grain,
                seed, monte_carlo_samples=64, filter_sigma=0.8):
        if strength <= 0.0:
            return (image,)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Darkroom] Film Grain Pro (Newson): size={grain_size} "
              f"strength={strength} var={radius_variation} color={color_grain} "
              f"N={monte_carlo_samples} on {device}")

        out = []
        for idx in range(image.shape[0]):
            res = render_film_grain(
                image[idx],
                grain_size=grain_size,
                radius_variation=radius_variation,
                strength=strength,
                color_grain=color_grain,
                n_samples=int(monte_carlo_samples),
                filter_sigma=filter_sigma,
                seed=int(seed) + idx * 7919,
                device=device,
            )
            out.append(res.unsqueeze(0).cpu())

        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmGrainPro": FilmGrainPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmGrainPro": "Film Grain Pro",
}
