"""
LUT Export node for ComfyUI-Darkroom.
Takes a processed identity lattice and bakes it into a .cube 3D LUT file.
The .cube format is universal — works in DaVinci Resolve, Premiere Pro, Photoshop,
Capture One, FCPX, and any tool that supports 3D LUTs.
"""

import os
import numpy as np

from ..utils.lut import image_to_lut_3d, write_cube_file
from ..utils.paths import resolve_allowed, resolve_output_file, safe_filename, PathNotAllowed


class LUTExport:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_lattice": ("IMAGE",),
                "lut_size": ("INT", {
                    "default": 33, "min": 2, "max": 129,
                    "tooltip": "Must match the LUT Identity Generator size"
                }),
                "filename": ("STRING", {
                    "default": "darkroom_grade",
                    "tooltip": "Output filename (without .cube extension)"
                }),
            },
            "optional": {
                "title": ("STRING", {
                    "default": "Darkroom Grade",
                    "tooltip": "Title embedded in the .cube file metadata"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Output directory. Leave empty for ComfyUI output/luts/. Must be "
                               "inside ComfyUI's output folder or a folder listed in "
                               "darkroom_allowed_folders.json. Relative paths start in output/."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, processed_lattice, lut_size, filename="darkroom_grade",
                title="Darkroom Grade", output_directory=""):

        size = int(lut_size)

        # Output directory: blank = ComfyUI output/luts/, anything else must
        # resolve inside a write root (relative paths start in output/).
        try:
            output_dir = resolve_allowed((output_directory or "").strip() or "luts", "write")
        except PathNotAllowed as e:
            raise ValueError(f"[Darkroom] LUT Export: {e}") from None

        # A bare file name only: no directories, drive or stream colons.
        safe_name = safe_filename(filename.replace(" ", "_"), "darkroom_grade")
        if safe_name.lower().endswith(".cube"):
            safe_name = safe_name[:-5] or "darkroom_grade"
        try:
            filepath = resolve_output_file(output_dir, f"{safe_name}.cube")
        except PathNotAllowed as e:
            raise ValueError(f"[Darkroom] LUT Export: {e}") from None

        # Convert tensor to numpy
        img = processed_lattice[0].cpu().numpy().astype(np.float32)

        # Validate dimensions
        expected_h = size * size
        expected_w = size
        if img.shape[0] != expected_h or img.shape[1] != expected_w:
            raise ValueError(
                f"[Darkroom] LUT Export: lattice image is {img.shape[1]}x{img.shape[0]}, "
                f"expected {expected_w}x{expected_h} for LUT size {size}. "
                f"Make sure lut_size matches the LUT Identity Generator."
            )

        # Convert processed image to 3D LUT array
        lut_3d = image_to_lut_3d(img, size)

        # Write .cube file
        write_cube_file(filepath, lut_3d, size, title=title)

        print(f"[Darkroom] LUT Export: saved {size}^3 ({size ** 3:,} entries) → {filepath}")

        return (filepath,)


NODE_CLASS_MAPPINGS = {"DarkroomLUTExport": LUTExport}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLUTExport": "LUT Export (.cube)"}
