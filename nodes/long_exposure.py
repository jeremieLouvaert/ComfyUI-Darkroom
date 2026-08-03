"""
Long Exposure node for ComfyUI-Darkroom.

Handheld long exposure: ND filter, slow shutter, the small erratic movement of a
hand holding a camera. Full derivation: docs/long-exposure-derivation.md.

Why this is not a motion-blur filter:
  - it INTEGRATES the image over a sequence of camera poses, which is what a long
    exposure physically is, rather than smearing along one angle.
  - the camera path is a synthesised HUMAN hand tremor: 1/f drift plus a
    physiological tremor peak near 9Hz. At the default it reproduces the published
    band shares (~90% of displacement at 1-3.5Hz, ~24% at 7.5-12.5Hz).
  - because the path wanders back over its own track, edges pick up GHOSTED,
    doubled copies. A single-angle smear cannot produce those, and they are the
    most recognisable thing in a real handheld long exposure.
  - roll makes corners smear differently from the centre, since they are further
    from the axis of rotation.
  - it gathers rather than pushes, so nothing tears and no holes appear.

Every existing ComfyUI motion-blur node applies one angle to the whole frame.
"""

import numpy as np
import torch

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.long_exposure import (shake_path, integrate_auto, band_shares,
                                   TREMOR_HZ, DRIFT_BAND, TREMOR_BAND)


class LongExposure:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure_px": ("FLOAT", {
                    "default": 45.0, "min": 0.0, "max": 300.0, "step": 1.0,
                    "tooltip": "How far the camera wanders during the exposure, in "
                               "pixels. The intensity dial: it is the shutter speed "
                               "and the steadiness of the hand rolled into one. 0 "
                               "returns the image untouched, exactly."
                }),
                "hand_steadiness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Character of the shake, not its size. 1.0 is a real "
                               "human hand -- the default reproduces the measured "
                               "tremor spectrum (about 90% of the movement below "
                               "3.5Hz, 24% in the 7.5-12.5Hz tremor band). 0 is a "
                               "perfectly smooth drift, like a slow tripod pan. "
                               "Above 1 the hand gets jittery and the smear starts "
                               "to read as digital rather than photographic."
                }),
                "variety": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "How much the movement CHANGES during the exposure. "
                               "This is intentional camera movement, so the hand "
                               "does not travel at one steady rate: it dwells, then "
                               "swoops, or whips and then settles. 0 gives a uniform "
                               "drift (every part of the exposure moving at the same "
                               "speed, which reads mechanical). 1.0 gives roughly a "
                               "4x spread between the slowest and fastest stretch; "
                               "2.0 about 6x. Dwelling is what leaves a readable "
                               "core with the smear trailing off it."
                }),
                "roll_deg": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 15.0, "step": 0.1,
                    "tooltip": "How much the camera twists about the lens axis. This "
                               "is the only part of the shake that varies ACROSS the "
                               "frame: corners are further from the axis, so they "
                               "travel further than the centre, which is exactly "
                               "zero. Set 0 for a pure shift."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Which shake. Same seed gives the same hand."
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "Where the movement applies, 0 to 1. This is how a "
                               "SUBJECT moves separately from the camera -- the one "
                               "thing a single camera path cannot express. Mask the "
                               "subject to smear it against a still background, or "
                               "invert it to hold the subject sharp while the world "
                               "moves around it."
                }),
                "poses": ("INT", {
                    "default": 96, "min": 8, "max": 256, "step": 8,
                    "tooltip": "How many camera positions the exposure is sampled "
                               "at. Too few and the smear breaks into visible "
                               "separate copies (which is a real look, just a "
                               "different one -- a strobe rather than a smear). "
                               "Cost is linear in this."
                }),
                "vary_per_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-roll the shake for each frame of a batch. Off "
                               "means one exposure applied to every frame."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, exposure_px, hand_steadiness, variety, roll_deg, seed,
                subject_mask=None, poses=96, vary_per_frame=False):
        frames = tensor_to_numpy_batch(image)
        H, W = frames[0].shape[:2]

        lo, hi = band_shares(int(poses), hand_steadiness)
        print(f"[Darkroom] Long Exposure: {W}x{H}, {exposure_px:.0f}px over "
              f"{poses} poses, steadiness {hand_steadiness:.2f}, variety {variety:.2f}, "
              f"roll {roll_deg:.1f}deg")
        print(f"[Darkroom]   shake spectrum: {100*lo:.0f}% of movement in "
              f"{DRIFT_BAND[0]}-{DRIFT_BAND[1]}Hz drift, {100*hi:.0f}% in "
              f"{TREMOR_BAND[0]}-{TREMOR_BAND[1]}Hz tremor "
              f"(human hand measures ~90% / ~24%)")

        masks = None
        if subject_mask is not None:
            m = subject_mask
            if m.dim() == 2:
                m = m.unsqueeze(0)
            masks = [np.clip(m[min(i, m.shape[0] - 1)].cpu().numpy().astype(np.float64),
                             0.0, 1.0) for i in range(len(frames))]
            if masks[0].shape != (H, W):
                from scipy.ndimage import zoom
                masks = [zoom(mm, (H / mm.shape[0], W / mm.shape[1]), order=1)
                         for mm in masks]
                masks = [np.clip(mm[:H, :W], 0.0, 1.0) for mm in masks]

        out_imgs = []
        for i, frame in enumerate(frames):
            s = int(seed) + (i * 7919 if vary_per_frame else 0)
            img = frame.astype(np.float64)
            if img.shape[2] > 3:
                img = img[..., :3]
            if exposure_px <= 0.0 and roll_deg <= 0.0:
                out_imgs.append(img.astype(np.float32))       # exact identity
                continue
            tx, ty, roll = shake_path(int(poses), seed=s, steadiness=hand_steadiness,
                                      variety=variety)
            out = integrate_auto(img, tx, ty, roll, amp_px=float(exposure_px),
                                 roll_deg=float(roll_deg),
                                 mask=None if masks is None else masks[i])
            out_imgs.append(np.clip(out, 0.0, 1.0).astype(np.float32))

        return (numpy_batch_to_tensor(out_imgs),)


NODE_CLASS_MAPPINGS = {
    "DarkroomLongExposure": LongExposure,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomLongExposure": "Long Exposure",
}
