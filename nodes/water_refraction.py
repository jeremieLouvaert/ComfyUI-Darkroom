"""
Water Refraction node for ComfyUI-Darkroom.

Refracts the image through a SIMULATED water surface poured onto the screen.
Full derivation: docs/water-refraction-derivation.md.

Why this is not a displacement filter:
  - the warp is a consequence of a depth-averaged fluid simulation, not
    a noise texture. The optics read only h(x,y) from it.
  - refraction is exact Snell, so displacement is hard-bounded at 0.881 * depth
    because theta_t saturates at the 48.6deg critical angle. That bound is why
    `field_width_mm` exists and why framing wide correctly does almost nothing.
  - the map FOLDS where the surface is steep enough, producing multiple images
    of one feature. A smooth displacement filter cannot do that.
  - a finite aperture blends those folded branches instead of seaming them,
    which is what removes the chrome / liquify look.

Costly by ComfyUI standards: expect ~45s at 1024. `sim_resolution` is left on
AUTO by default because it is not merely a speed dial -- it sets whether the
capillary ripple is resolved, and that ripple is what makes a pool fold.
"""

import numpy as np
import torch

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.water_refraction import (simulate, settle, to_image_res, render_auto,
                                      jacobian_det, grain_deficit, restore_grain,
                                      CAPILLARY_MM, DELTA_MAX_RATIO)


SURFACES = [
    "standing pool (heavier, whole frame)",
    "dry screen (readable, large clear areas)",
]

# Cell size the solver needs, in mm. The capillary length is 2.727mm, so this is
# ~10.9 cells across it -- the resolution at which a pool actually folds. Measured
# on one pour: 7.6 cells gives 2.0% folding, 10.9 gives 20.2%.
TARGET_DX_MM = 0.25
AUTO_NX_MAX = 224            # past this the solver dominates the runtime entirely


def auto_sim_resolution(field_width_mm):
    """
    Grid width that keeps the CELL SIZE physical, rather than keeping a number fixed.

    dx = field_width / nx, and what decides whether the surface folds is dx against
    the capillary length -- not the image resolution, which is measurably irrelevant
    here (folding 33.3% / 31.3% / 33.3% at 512 / 762 / 1024 px on identical physics).

    Holding nx fixed while the field width changes is a silent trap: at nx = 160,
    folding runs 57.7% at a 20mm field, 31.3% at 40mm and 2.4% at 80mm, purely
    because dx coarsens from 0.125mm to 0.5mm. Nothing tells the user their folds
    left. Deriving nx from the field width removes that.
    """
    nx = int(round(float(field_width_mm) / TARGET_DX_MM))
    return max(64, min(nx, AUTO_NX_MAX))


# mm of water already present. The pool depth is the EXP-10 value Jeremie picked
# as the favourite; a free puddle caps near 4mm, so this implies a tray or bezel.
SURFACE_FILM_MM = {SURFACES[0]: 12.0, SURFACES[1]: 0.0}


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
                "surface": (SURFACES, {
                    "default": SURFACES[0],
                    "tooltip": "What the water is poured onto. THE biggest look "
                               "lever, and the two options are different looks "
                               "rather than two ends of a dial.\n\n"
                               "Standing pool: pouring INTO water that is already "
                               "there. The disturbance reaches the whole frame, "
                               "which gives the heavier, more liquid result.\n\n"
                               "Dry screen: the water only exists where it has been "
                               "poured, so large areas stay completely undistorted "
                               "and the subject stays very readable.\n\n"
                               "(A free puddle cannot hold more than ~4mm on its "
                               "own, so the pool option implies a tray or a bezel.)"
                }),
                "water_ml": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "How much water is poured. The intensity dial: more "
                               "water is deeper water is more displacement. Raising "
                               "it also shrinks the undistorted area, which is what "
                               "keeps the subject readable."
                }),
                "pour_sweep": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How far the stream is swept while pouring, which is "
                               "really a CONCENTRATION control. The same water drawn "
                               "along a longer stroke is shallower, so it displaces "
                               "and folds less: measured, folding runs 39% at 0, 30% "
                               "at 0.5 and 24% at 1.0. Low values pile the pour into "
                               "one deep mound for the heaviest distortion; high "
                               "values spread it into a longer, gentler disturbance."
                }),
                "sweep_angle": ("FLOAT", {
                    "default": 45.0, "min": 0.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Direction the stream is swept across the frame."
                }),
                "sample_ms": ("FLOAT", {
                    "default": 150.0, "min": 20.0, "max": 300.0, "step": 5.0,
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
                    "default": 0, "min": 0, "max": 256, "step": 8,
                    "tooltip": "0 = AUTO, which is what you want. Auto sizes the "
                               "fluid grid so each cell is 0.25mm whatever the field "
                               "width, i.e. about 11 cells across the 2.727mm "
                               "capillary length. That matters because the ripple at "
                               "that scale is what supplies the slope that makes a "
                               "pool FOLD: measured, 7.6 cells gives 2% folding and "
                               "10.9 gives 20%. Pinning this to a fixed number is a "
                               "trap, because dx = field_width / this, so widening "
                               "the framing silently coarsens the grid and the folds "
                               "quietly disappear (57.7% at 20mm, 31.3% at 40mm, "
                               "2.4% at 80mm, all at 160). Set it by hand only to "
                               "trade quality for speed deliberately. Cost scales "
                               "roughly with the square."
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

    def execute(self, image, field_width_mm, surface, water_ml, pour_sweep,
                sweep_angle, sample_ms, settle_ms, depth_scale, aperture, seed,
                sim_resolution=0, aperture_samples=32, env_strength=1.0,
                grain_restore=1.0, dispersion=False, vary_per_frame=False):
        frames = tensor_to_numpy_batch(image)
        H, W = frames[0].shape[:2]
        field_h_mm = field_width_mm * H / W

        print(f"[Darkroom] Water Refraction: {W}x{H}, field {field_width_mm:.1f}x"
              f"{field_h_mm:.1f}mm, {water_ml:.1f}ml onto {surface.split(chr(32))[0]}, "
              f"sweep {pour_sweep:.2f}, "
              f"sample {sample_ms:.0f}ms + settle {settle_ms:.0f}ms, sim {sim_resolution}")

        nx_used = (auto_sim_resolution(field_width_mm) if int(sim_resolution) <= 0
                   else int(sim_resolution))
        dx_mm = field_width_mm / max(nx_used, 1)
        note = ""
        if int(sim_resolution) <= 0 and dx_mm > TARGET_DX_MM * 1.05:
            note = (f"  [capped at {AUTO_NX_MAX}: cells are {dx_mm:.3f}mm, coarser "
                    f"than the {TARGET_DX_MM}mm the capillary ripple wants, so "
                    f"expect reduced folding at this field width]")
        print(f"[Darkroom]   grid {nx_used} ({'auto' if int(sim_resolution) <= 0 else 'manual'}), "
              f"dx {dx_mm:.3f}mm = {CAPILLARY_MM/dx_mm:.1f} cells per capillary length{note}")

        out_imgs, out_masks = [], []
        cached = None
        for i, frame in enumerate(frames):
            s = int(seed) + (i * 7919 if vary_per_frame else 0)
            if cached is None or vary_per_frame:
                h_sim, sim = simulate(
                    field_width_mm, field_h_mm, volume_ml=water_ml,
                    nx=nx_used, seed=s, sample_ms=sample_ms,
                    sweep=pour_sweep, sweep_angle_deg=sweep_angle,
                    initial_film_mm=SURFACE_FILM_MM.get(surface, 12.0))
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
            out = render_auto(img, h_mm, field_width_mm, aperture_ratio=aperture,
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
