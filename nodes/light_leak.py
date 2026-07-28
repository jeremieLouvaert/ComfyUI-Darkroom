"""
Light Leak node for ComfyUI-Darkroom.

Sprocket-trap, gradient and pinhole leaks derived from ONE mechanism: how far
the light travelled through the film base before it reached the emulsion.
Full derivation: docs/light-leak-derivation.md (signed off 2026-07-28; spike
33/33 checks with 6 negative controls firing; Jeremie eyeball PASS).

Why this is not a tinted gradient:
  - additive in LINEAR light, because a leak ADDS exposure. It therefore adds a
    constant absolute luminance regardless of what is underneath; an sRGB screen
    blend (what an overlay pack does) cannot hold that.
  - all three channels start equal at the entry point and blue is absorbed
    fastest, so a leak is a near-white HOT CORE that reddens outward. Every
    canned plate bakes a uniform orange and flattens that structure.
  - a pinhole has no lateral path through the base, so it stays the colour of
    its source -- neutral, not orange. The anti-convention case, predicted by
    the same equation rather than asserted.

Procedural only. No scan plates: the commercial leak packs forbid redistribution
"whether modified or not", and procedural is the house default anyway.
"""

import numpy as np
import torch

from ..utils.color import srgb_to_linear, linear_to_srgb
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.light_leak import leak_field, composite, WARP_SCALE_FLOOR


LEAK_TYPES = [
    "sprocket trap (perforation modulated)",
    "gradient (seal / bellows / backing)",
    "pinhole (neutral, front-side)",
]
TYPE_KEY = {
    LEAK_TYPES[0]: "sprocket",
    LEAK_TYPES[1]: "gradient",
    LEAK_TYPES[2]: "pinhole",
}

FROM = ["top", "bottom", "left", "right",
        "top-left", "top-right", "bottom-left", "bottom-right"]

COLOUR_SOURCES = [
    "base path (derived red-shift)",
    "backing paper (120 roll, preset tint)",
    "neutral (source colour)",
]
COLOUR_KEY = {
    COLOUR_SOURCES[0]: "base path",
    COLOUR_SOURCES[1]: "backing paper",
    COLOUR_SOURCES[2]: "neutral",
}


def _resolve_origin(mode, leak_from):
    """
    Returns (edge, corner). Sprocket must enter along a real perforation row, so
    a corner selection collapses to its edge component; gradient honours corners.
    """
    is_corner = "-" in leak_from
    if mode == "sprocket":
        return (leak_from.split("-")[0] if is_corner else leak_from), None
    if is_corner:
        return None, leak_from
    return leak_from, None


class DarkroomLightLeak:
    """See module docstring. Frozen spec: docs/light-leak-derivation.md."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "leak_type": (LEAK_TYPES, {
                    "default": LEAK_TYPES[0],
                    "tooltip": "Sprocket = edge falloff modulated by the KS-1870 perforation lattice. Gradient = seal/bellows wedge. Pinhole = camera-obscura spots, neutral by physics"
                }),
                "leak_from": (FROM, {
                    "default": "top",
                    "tooltip": "Which edge or corner the light enters from. Sprocket collapses a corner to its edge, since perforations run along the film's long edge"
                }),
                "strength": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Leak exposure added, in linear light. 0 = passthrough"
                }),
                "falloff": ("FLOAT", {
                    "default": 200.0, "min": 20.0, "max": 1200.0, "step": 10.0,
                    "tooltip": "Red-channel penetration depth, ref-px @1024 long edge. Green and blue follow at 0.62 and 0.38 of it, which is what reddens the leak with distance"
                }),
                "colour_source": (COLOUR_SOURCES, {
                    "default": COLOUR_SOURCES[0],
                    "tooltip": "Base path = the derived white-core-to-red gradient. Backing paper = 120 roll-paper dye, an honest preset. Neutral = source colour, no lateral path"
                }),
                "diffusion": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Sprocket only: lateral spreading of light past a perforation. Above 0 the comb widens and softens with depth; at 0 the teeth keep straight sides and read as a bar chart"
                }),
                "pinhole_count": ("INT", {
                    "default": 3, "min": 1, "max": 40,
                    "tooltip": "Pinhole only: number of holes in the body/bellows"
                }),
                "pinhole_diameter": ("FLOAT", {
                    "default": 0.30, "min": 0.02, "max": 3.0, "step": 0.02,
                    "tooltip": "Pinhole only: hole diameter in mm. Sets the penumbra (the 'a' term in spot size = a + D*theta)"
                }),
                "flange_distance": ("FLOAT", {
                    "default": 50.0, "min": 10.0, "max": 300.0, "step": 5.0,
                    "tooltip": "Pinhole only: hole-to-film distance in mm. Spot size = a + D*theta, so a deeper body projects a larger spot"
                }),
                "displacement": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Warps the PATH LENGTH so the leak boundary wanders instead of running parallel to the frame edge. Because colour follows path length, the red fringe warps with it. 0 = clean analytic field"
                }),
                "displacement_scale": ("FLOAT", {
                    "default": 380.0, "min": 142.0, "max": 1500.0, "step": 20.0,
                    "tooltip": f"Warp feature size, ref-px @1024. Clamped at {WARP_SCALE_FLOOR:.0f} (5mm of seam): anything finer stops being a mechanical gap variation and reads as noise running through the effect"
                }),
                "vary_per_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Batch of N gets N different leaks. Off keeps one leak across the batch, which is usually what you want for a sequence"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seeds the perforation phase, the pinhole placement and the displacement warp"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "leak_mask")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, leak_type=LEAK_TYPES[0], leak_from="top",
                strength=0.45, falloff=200.0, colour_source=COLOUR_SOURCES[0],
                diffusion=0.35, pinhole_count=3, pinhole_diameter=0.30,
                flange_distance=50.0, displacement=100.0,
                displacement_scale=380.0, vary_per_frame=False, seed=42):

        mode = TYPE_KEY.get(leak_type, "sprocket")
        edge, corner = _resolve_origin(mode, leak_from)
        csrc = COLOUR_KEY.get(colour_source, "base path")

        images = tensor_to_numpy_batch(image)

        if strength <= 0.0:
            print("[Darkroom] Light Leak: strength 0, passthrough")
            empty = [np.zeros(im.shape[:2], dtype=np.float32) for im in images]
            return (image, torch.cat([torch.from_numpy(m).unsqueeze(0)
                                      for m in empty], dim=0))

        print(f"[Darkroom] Light Leak: type={mode}, from={leak_from}, "
              f"strength={strength}, falloff={falloff}, colour={csrc}, "
              f"displacement={displacement}@{displacement_scale}, seed={seed}")

        out_images, out_masks = [], []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            frame_seed = (int(seed) + i * 7919) if vary_per_frame else int(seed)

            G = leak_field(
                h, w, mode, edge=edge, corner=corner,
                lam_ref=falloff, mod_ratio=diffusion, seed=frame_seed,
                color_source=csrc, pinhole_count=pinhole_count,
                hole_mm=pinhole_diameter, flange_mm=flange_distance,
                displacement=displacement, displacement_scale=displacement_scale,
            )

            lin = srgb_to_linear(img.astype(np.float64))
            out = composite(lin, G, strength)
            out_images.append(linear_to_srgb(out).astype(np.float32))
            m = np.clip(strength * G.max(axis=2), 0.0, 1.0).astype(np.float32)
            out_masks.append(m)

        return (numpy_batch_to_tensor(out_images),
                torch.cat([torch.from_numpy(m).unsqueeze(0) for m in out_masks], dim=0))


NODE_CLASS_MAPPINGS = {
    "DarkroomLightLeak": DarkroomLightLeak,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomLightLeak": "Light Leak",
}
