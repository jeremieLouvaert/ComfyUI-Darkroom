"""
LUT Apply node for ComfyUI-Darkroom.
Loads and applies a .cube 3D LUT to any image with trilinear interpolation.
Import looks from DaVinci Resolve, Premiere, Photoshop, or use Darkroom-exported LUTs.
"""

import os
import numpy as np

from ..utils.color import blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.lut import parse_cube_file, apply_lut_trilinear
from ..utils.paths import resolve_allowed, PathNotAllowed


class LUTApply:

    # Cache parsed LUTs to avoid re-reading on every execution
    _lut_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a .cube LUT file, e.g. a LUT Export output. Must be "
                               "inside ComfyUI's input/output/temp/models folders or a folder "
                               "listed in darkroom_allowed_folders.json. Relative paths start in input/."
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and LUT-graded (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def _get_lut(self, filepath):
        """Load and cache a .cube LUT file."""
        # Use file path + modification time as cache key
        mtime = os.path.getmtime(filepath)
        cache_key = (filepath, mtime)

        if cache_key not in LUTApply._lut_cache:
            lut_3d, size = parse_cube_file(filepath)
            LUTApply._lut_cache[cache_key] = (lut_3d, size)
            print(f"[Darkroom] LUT Apply: loaded {size}^3 LUT from {filepath}")

        return LUTApply._lut_cache[cache_key]

    def execute(self, image, lut_file, strength=1.0):
        if strength <= 0.0:
            return (image,)

        filepath = lut_file.strip()
        if not filepath:
            print("[Darkroom] LUT Apply: no file specified, passing through")
            return (image,)

        try:
            filepath = resolve_allowed(filepath, "read")
        except PathNotAllowed as e:
            raise ValueError(f"[Darkroom] LUT Apply: {e}") from None
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[Darkroom] LUT Apply: file not found — {filepath}"
            )

        lut_3d, lut_size = self._get_lut(filepath)

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            graded = apply_lut_trilinear(img, lut_3d, lut_size)
            results.append(blend(original, graded, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomLUTApply": LUTApply}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLUTApply": "LUT Apply (.cube)"}
