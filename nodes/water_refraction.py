"""
Water Refraction node for ComfyUI-Darkroom.

Refracts the image through a SIMULATED water surface poured onto the screen.
Full derivation: docs/water-refraction-derivation.md.

Why this is not a displacement filter:
  - the warp is a consequence of a depth-averaged FLIP/PIC fluid simulation, not
    a noise texture. The optics read only h(x,y) from it.
  - refraction is exact Snell, so displacement is hard-bounded at 0.881 * depth
    because theta_t saturates at the 48.6deg critical angle. That bound is why
    `field_width_mm` exists and why framing wide correctly does almost nothing.
  - the map FOLDS where the surface is steep enough, producing multiple images
    of one feature. A smooth displacement filter cannot do that.
  - a finite aperture blends those folded branches instead of seaming them,
    which is what removes the chrome / liquify look.

Costly by ComfyUI standards: the fluid step is ~40ms and a default run is a few
hundred steps, so expect tens of seconds at 1024. `sim_resolution` is the
speed dial and it trades structure, not correctness.
"""

import numpy as np
import torch

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.water_refraction import (simulate, settle, to_image_res, render,
                                      jacobian_det, grain_deficit, restore_grain,
                                      CAPILLARY_MM, DELTA_MAX_RATIO)


class WaterRefraction:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "field_width_mm": ("FLOAT", {
                    "default": 40.0, "min": 8.0, "max": 250.0, "step": 0.5,
                    "tooltip": "How wide a patch of screen the frame covers, in "
                               "millimetres. THE most important control. Refraction "
                               "displaces at most 0.881x the water depth, so framing "
                               "a whole tablet (250mm) correctly gives almost "
                               "nothing. The reference look lives at 20-50mm."
                }),
                "water_ml": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "How much water is poured. The intensity dial: more "
                               "water is deeper water is more displacement. Raising "
                               "it also shrinks the undistorted area, which is what "
                               "keeps the subject readable."
                }),
                "pour_sweep": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How far the stream is swept while pouring. 0 is a "
                               "stationary source, which makes a perfectly circular "
                               "spreading disc and reads as a lens rather than as "
                               "water. Keep above ~0.5 unless you want that."
                }),
                "sweep_angle": ("FLOAT", {
                    "default": 45.0, "min": 0.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Direction the stream is swept across the frame."
                }),
                "sample_ms": ("FLOAT", {
                    "default": 80.0, "min": 20.0, "max": 300.0, "step": 5.0,
                    "tooltip": "When the photograph is taken, measured from the "
                               "start of the pour. Below the pour duration (100ms) "
                               "you catch the live event; well above it the water "
                               "has spread thin and flattened."
                }),
                "settle_ms": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 5.0,
                    "tooltip": "Extra time for the surface to calm before the shot. "
                               "Viscous damping kills short ripples far faster than "
                               "long ones (a 2.4mm ripple dies in ~72ms, a 12mm one "
                               "in ~1.8s), so this removes fine chop and keeps the "
                               "large forms. Costs folding if pushed far."
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05,
                    "tooltip": "Multiplies the water depth. The honest artistic "
                               "knob: anyone wanting more than physics allows is "
                               "really asking for deeper water, so this scales h "
                               "and the optics stay exact."
                }),
                "aperture": ("FLOAT", {
                    "default": 0.020, "min": 0.0, "max": 0.060, "step": 0.002,
                    "tooltip": "Lens aperture over camera distance (A/L). A 100mm "
                               "macro at f/8 from 300mm is about 0.021. This is "
                               "required physics, not softening: it blends the "
                               "multiple images at a fold instead of seaming them. "
                               "0 gives a pinhole and a hard chrome look."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Pour seed. Same seed gives the same water."
                }),
            },
            "optional": {
                "sim_resolution": ("INT", {
                    "default": 112, "min": 64, "max": 160, "step": 8,
                    "tooltip": "Fluid grid width. The speed dial. 112 resolves the "
                               "2.727mm capillary length across 7.6 cells; below ~80 "
                               "the surface loses the fine structure that folds. "
                               "Cost scales roughly with the square."
                }),
                "aperture_samples": ("INT", {
                    "default": 32, "min": 8, "max": 64, "step": 4,
                    "tooltip": "Rays per pixel across the lens. Lower is faster and "
                               "noisier at folds."
                }),
                "env_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Brightness of the environment reflected off the "
                               "water by the Fresnel term, strongest at grazing "
                               "angles on the steep fold lines."
                }),
                "dispersion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Refract R/G/B at their own refractive indices. Real "
                               "but subtle, and it triples the optics cost."
                }),
                "grain_restore": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Put back the fine detail the warp destroyed. The "
                               "refraction genuinely eats film grain — some of it is "
                               "real physics (the aperture) and some was the "
                               "resampler — so this restores it, weighted by a "
                               "MEASURED deficit map and calibrated so 1.0 lands at "
                               "the source's own grain level rather than adding on "
                               "top. 0 leaves the image alone and you can chain the "
                               "grain_deficit mask into a grain node instead, but "
                               "then the calibration is yours to find."
                }),
                "vary_per_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-pour with a different seed for each batch frame. "
                               "Off means one surface across the batch."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "grain_deficit")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, field_width_mm, water_ml, pour_sweep, sweep_angle,
                sample_ms, settle_ms, depth_scale, aperture, seed,
                sim_resolution=112, aperture_samples=32, env_strength=1.0,
                grain_restore=1.0, dispersion=False, vary_per_frame=False):
        frames = tensor_to_numpy_batch(image)
        H, W = frames[0].shape[:2]
        field_h_mm = field_width_mm * H / W

        print(f"[Darkroom] Water Refraction: {W}x{H}, field {field_width_mm:.1f}x"
              f"{field_h_mm:.1f}mm, {water_ml:.1f}ml, sweep {pour_sweep:.2f}, "
              f"sample {sample_ms:.0f}ms + settle {settle_ms:.0f}ms, sim {sim_resolution}")

        out_imgs, out_masks = [], []
        cached = None
        for i, frame in enumerate(frames):
            s = int(seed) + (i * 7919 if vary_per_frame else 0)
            if cached is None or vary_per_frame:
                h_sim, sim = simulate(
                    field_width_mm, field_h_mm, volume_ml=water_ml,
                    nx=int(sim_resolution), seed=s, sample_ms=sample_ms,
                    sweep=pour_sweep, sweep_angle_deg=sweep_angle)
                h_sim = settle(h_sim, sim.dx, settle_ms / 1000.0)
                cached = to_image_res(h_sim, H, W)
                fold = float((jacobian_det(cached, field_width_mm) < 0).mean())
                dmax = float(np.abs(cached).max()) * DELTA_MAX_RATIO
                print(f"[Darkroom]   surface: max depth {cached.max():.2f}mm "
                      f"(bound {dmax:.2f}mm displacement), {100*fold:.1f}% of frame "
                      f"folded, capillary length {CAPILLARY_MM:.3f}mm")
            h_mm = cached

            img = frame.astype(np.float64)
            if img.shape[2] > 3:
                img = img[..., :3]
            out = render(img, h_mm, field_width_mm, aperture_ratio=aperture,
                         samples=int(aperture_samples), depth_scale=depth_scale,
                         env_strength=env_strength, dispersion=bool(dispersion),
                         seed=s)
            deficit = grain_deficit(img.shape, h_mm, field_width_mm,
                                    aperture_ratio=aperture, seed=s + 21)
            if grain_restore > 0.0:
                out = restore_grain(out, img, deficit,
                                    amount=float(grain_restore), seed=s + 5)
            out_imgs.append(out.astype(np.float32))
            out_masks.append(torch.from_numpy(deficit).unsqueeze(0))

        return (numpy_batch_to_tensor(out_imgs), torch.cat(out_masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "DarkroomWaterRefraction": WaterRefraction,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomWaterRefraction": "Water Refraction",
}
