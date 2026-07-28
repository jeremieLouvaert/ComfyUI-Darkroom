"""
Film Damage node for ComfyUI-Darkroom.

Dust, dirt, hairs and scratches in one node, with the defect's SIGN and COLOUR
derived from where in the imaging chain it sits rather than chosen as paint.
Full derivation: docs/film-damage-derivation.md (signed off 2026-07-27; spike
23/23 checks PASS with 3 negative controls firing; Jeremie eyeball PASS).

The model, in one line:

    L_out,c = L_in,c * tau_c ** k     k = +1 (positive-plane) / -gamma_p (negative-plane)

so dust on a negative prints WHITE while dust on a scanned positive reads DARK,
and on the SAME negative a base-side scratch prints white while a full-depth
emulsion gouge prints black. Scratch colour walks yellow -> red -> black on
colour negative and blue -> cyan -> white on reversal, from one depth control.

Procedural only. No scan plates, no PNG assets (commercial overlay licences
forbid redistribution "whether modified or not"; procedural is both the legal
path and the house default). Morphology follows the Ivanova et al. CGF 2023
gamma-distribution METHOD with our own parameters, plus Kokaram/Joyeux scratch
models. Compositing is ours: that literature writes ML training masks and
contains no polarity or blending logic at all.
"""

import numpy as np
import torch

from ..utils.color import srgb_to_linear, linear_to_srgb
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.film_damage import build_tau, composite, defect_mask


ORIGINS = ["negative (dust prints white)", "positive / scan-side (dust reads dark)"]
ORIGIN_KEY = {
    ORIGINS[0]: "negative",
    ORIGINS[1]: "positive",
}

FILM_TYPES = ["Color neg (C-41)", "B&W", "Reversal / slide"]
FILM_KEY = {
    "Color neg (C-41)": "c41",
    "B&W": "bw",
    "Reversal / slide": "reversal",
}

SCRATCH_SIDES = ["base (refractive, neutral)", "emulsion (dye loss, coloured)"]
SCRATCH_SIDE_KEY = {
    SCRATCH_SIDES[0]: "base",
    SCRATCH_SIDES[1]: "emulsion",
}

TRANSPORT = ["auto (along long edge)", "horizontal", "vertical"]
TRANSPORT_KEY = {
    TRANSPORT[0]: "auto",
    TRANSPORT[1]: "h",
    TRANSPORT[2]: "v",
}


def _mask_batch_to_tensor(arrays):
    """List of (H, W) float32 -> ComfyUI MASK tensor (B, H, W)."""
    return torch.cat([torch.from_numpy(a).unsqueeze(0) for a in arrays], dim=0)


class DarkroomFilmDamage:
    """See module docstring. Frozen spec: docs/film-damage-derivation.md."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "defect_origin": (ORIGINS, {
                    "default": ORIGINS[0],
                    "tooltip": "Where the defect sits in the chain. This alone flips the sign: on a negative it blocks PRINTING light so the print goes lighter; on a positive it just blocks the image light"
                }),
                "film_type": (FILM_TYPES, {
                    "default": FILM_TYPES[0],
                    "tooltip": "Drives emulsion-scratch colour via dye-layer order. B&W has no dye layers, so scratches stay neutral"
                }),
                "print_gamma": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Paper grade / scanner-inversion contrast. Negative-plane only: it is the exponent magnitude, so it sets how hard dust prints out"
                }),
                "density": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Master defect count multiplier across all classes. 0 = passthrough. Raise this, not the sizes, for a dirtier frame"
                }),
                "dust_amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Dust count multiplier (gamma-distributed count and size)"
                }),
                "dust_size": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 4.0, "step": 0.05,
                    "tooltip": "Dust mean radius, ref-px @1024 long edge. Default ~22-41um at real scale; larger reads as confetti, not dust"
                }),
                "dirt_amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Dirt/lint count multiplier. Larger, softer and lower opacity than dust, more strongly clustered"
                }),
                "dirt_size": ("FLOAT", {
                    "default": 1.1, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Dirt mean radius, ref-px @1024 long edge"
                }),
                "hair_amount": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Hair/fibre count multiplier. Short and long strands share one generator"
                }),
                "hair_length": ("FLOAT", {
                    "default": 140.0, "min": 20.0, "max": 600.0, "step": 10.0,
                    "tooltip": "Long-strand length, ref-px @1024 long edge (short strands are 0.35x this)"
                }),
                "scratch_count": ("INT", {
                    "default": 3, "min": 0, "max": 40,
                    "tooltip": "Number of scratches. They run along the transport axis with bounded lateral wander"
                }),
                "scratch_width": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 12.0, "step": 0.1,
                    "tooltip": "Scratch core width, ref-px @1024. The literature 3-10px band describes heavily damaged archive film; stills want far less"
                }),
                "scratch_side": (SCRATCH_SIDES, {
                    "default": SCRATCH_SIDES[0],
                    "tooltip": "Which FACE of the film. Base = a refractive groove (removes no dye, wet-gate cancels it) so it prints white. Emulsion = material loss, colour set by depth"
                }),
                "scratch_depth": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Emulsion side only. Removes dye layers top-down: C-41 walks yellow -> red -> black, reversal walks blue -> cyan -> white"
                }),
                "layer_density": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 2.5, "step": 0.05,
                    "tooltip": "Optical density of dye removed per layer. Real maxima run 2-3, which drives straight to clear base and clips"
                }),
                "transport_axis": (TRANSPORT, {
                    "default": TRANSPORT[0],
                    "tooltip": "Scratches follow the film transport. Auto = along the long edge (still 35mm). Cine runs vertically through the gate, hence vertical tramlines"
                }),
                "softness": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Optical blur of the defect, ref-px. 0 = auto (0.6 negative-plane, 1.6 positive-plane: scanner-glass dirt sits off the image plane so it is more defocused)"
                }),
                "base_scratch_cast": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "TASTE ONLY, off by default. Green/cyan tint on base-side scratches. This is an inference from wet-gate physics plus orange-mask channel gain, not a sourced causal link"
                }),
                "vary_per_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Batch of N gets N different defect fields (as a real roll does). Off = the identical field on every frame"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seeds every defect class: counts, sizes, positions, paths and scratch envelopes"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "defect_mask")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, defect_origin=ORIGINS[0], film_type=FILM_TYPES[0],
                print_gamma=2.0, density=0.5,
                dust_amount=1.0, dust_size=0.35,
                dirt_amount=1.0, dirt_size=1.1,
                hair_amount=1.0, hair_length=140.0,
                scratch_count=3, scratch_width=0.8,
                scratch_side=SCRATCH_SIDES[0], scratch_depth=0.5, layer_density=0.7,
                transport_axis=TRANSPORT[0], softness=0.0, base_scratch_cast=0.0,
                vary_per_frame=True, seed=42):

        origin = ORIGIN_KEY.get(defect_origin, "negative")
        film_key = FILM_KEY.get(film_type, "c41")
        side = SCRATCH_SIDE_KEY.get(scratch_side, "base")
        axis = TRANSPORT_KEY.get(transport_axis, "auto")

        images = tensor_to_numpy_batch(image)

        # Nothing to do: return the input untouched and an empty mask.
        no_classes = (dust_amount <= 0.0 and dirt_amount <= 0.0
                      and hair_amount <= 0.0 and scratch_count <= 0)
        if density <= 0.0 or no_classes:
            print("[Darkroom] Film Damage: no defects requested, passthrough")
            empty = [np.zeros(img.shape[:2], dtype=np.float32) for img in images]
            return (image, _mask_batch_to_tensor(empty))

        print(f"[Darkroom] Film Damage: origin={origin}, film={film_key}, "
              f"density={density}, print_gamma={print_gamma}, "
              f"scratches={scratch_count}@{side} depth={scratch_depth}, "
              f"axis={axis}, seed={seed}")

        out_images, out_masks = [], []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            frame_seed = (int(seed) + i * 7919) if vary_per_frame else int(seed)

            tau = build_tau(
                h, w, frame_seed,
                density=density,
                dust_amount=dust_amount, dirt_amount=dirt_amount,
                hair_amount=hair_amount, scratch_count=scratch_count,
                dust_size=dust_size, dirt_size=dirt_size,
                hair_length=hair_length, scratch_width=scratch_width,
                scratch_side=side, scratch_depth=scratch_depth,
                layer_density=layer_density, film_key=film_key,
                transport_axis=axis, softness=softness, origin=origin,
                base_scratch_cast=base_scratch_cast,
            )

            lin = srgb_to_linear(img.astype(np.float64))
            out = composite(lin, tau, origin, print_gamma)
            out_images.append(linear_to_srgb(out).astype(np.float32))
            out_masks.append(defect_mask(tau))

        return (numpy_batch_to_tensor(out_images), _mask_batch_to_tensor(out_masks))


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmDamage": DarkroomFilmDamage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmDamage": "Film Damage",
}
