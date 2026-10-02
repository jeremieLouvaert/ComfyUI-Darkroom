"""
Spectral Film Stock node for ComfyUI-Darkroom.

Applies pre-baked .cube LUTs derived from spectral datasheet simulation of
negative x print stock pairs. Source: JanLohse/spectral_film_lut (MIT), baked
offline via tools/bake_spectral_luts.py.

The LUTs encode a full neg->print chain: scene light -> negative spectral
sensitivity -> log exposure -> H&D density curve -> dye spectral density ->
printer light -> print density -> viewing illuminant -> sRGB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..utils.gpu_color import run_on_device, blend, lut_to_device, apply_lut_trilinear
from ..utils.lut import parse_cube_file

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "spectral_luts"
_MANIFEST_PATH = _DATA_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if not _MANIFEST_PATH.is_file():
        print(
            f"[Darkroom] Spectral Film Stock: manifest not found at {_MANIFEST_PATH}. "
            "Run tools/bake_spectral_luts.py --all to generate LUTs."
        )
        return []
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Darkroom] Spectral Film Stock: manifest parse failed: {e}")
        return []
    return data.get("presets", [])


def _build_labels(presets: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Build human-readable dropdown labels and a label->filename map."""
    grouped: dict[str, list[dict]] = {}
    for p in presets:
        grouped.setdefault(p.get("category", "Other"), []).append(p)

    labels: list[str] = []
    label_to_file: dict[str, str] = {}
    category_order = ["C41 Still", "Cinema", "Reversal", "Instant", "Niche", "B&W", "Other"]
    seen = set()

    for cat in category_order + sorted(grouped.keys()):
        if cat in seen or cat not in grouped:
            continue
        seen.add(cat)
        for p in sorted(grouped[cat], key=lambda x: x["name"]):
            label = f"{cat} / {p['negative']} -> {p['print']}"
            labels.append(label)
            label_to_file[label] = p["file"]

    return labels, label_to_file


_PRESETS = _load_manifest()
_LABELS, _LABEL_TO_FILE = _build_labels(_PRESETS)


class SpectralFilmStock:

    _lut_cache: dict = {}
    # Device copies of the parsed LUTs, uploaded once per (path, mtime, device)
    _lut_device_cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        choices = _LABELS if _LABELS else ["(no spectral LUTs found - run tools/bake_spectral_luts.py)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (choices, {
                    "default": choices[0],
                    "tooltip": "Negative film x print paper spectral simulation"
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and spectral look (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def _resolve(self, preset_label: str) -> Path:
        filename = _LABEL_TO_FILE.get(preset_label)
        if filename is None:
            raise ValueError(
                f"[Darkroom] Spectral Film Stock: unknown preset {preset_label!r}. "
                "Rebake with tools/bake_spectral_luts.py --all."
            )
        path = _DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"[Darkroom] Spectral Film Stock: LUT missing for {preset_label!r}: {path}"
            )
        return path

    def _get_lut(self, path: Path):
        mtime = os.path.getmtime(path)
        key = (str(path), mtime)
        if key not in SpectralFilmStock._lut_cache:
            lut_3d, size = parse_cube_file(str(path))
            SpectralFilmStock._lut_cache[key] = (lut_3d, size)
            print(f"[Darkroom] Spectral Film Stock: loaded {size}^3 LUT from {path.name}")
        return SpectralFilmStock._lut_cache[key]

    def _get_lut_device(self, path: Path, lut_3d, dev):
        key = (str(path), os.path.getmtime(path), str(dev))
        if key not in SpectralFilmStock._lut_device_cache:
            SpectralFilmStock._lut_device_cache[key] = lut_to_device(lut_3d, dev)
        return SpectralFilmStock._lut_device_cache[key]

    def execute(self, image, preset, strength=1.0):
        if strength <= 0.0 or not _LABELS:
            return (image,)

        path = self._resolve(preset)
        lut_3d, lut_size = self._get_lut(path)

        def process(img):
            lut_flat = self._get_lut_device(path, lut_3d, img.device)
            graded = apply_lut_trilinear(img, lut_flat, lut_size)
            return blend(img, graded, strength)

        out = run_on_device(process, image)
        print(f"[Darkroom] Spectral Film Stock: {preset} (strength={strength:.2f})")
        return (out,)


NODE_CLASS_MAPPINGS = {"DarkroomSpectralFilmStock": SpectralFilmStock}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomSpectralFilmStock": "Spectral Film Stock"}
