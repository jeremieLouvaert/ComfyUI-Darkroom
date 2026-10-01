"""
CMYK print-workflow utilities.

Profile discovery + PIL.ImageCms transforms for soft-proofing, gamut warning,
TAC check, and CMYK TIFF export. sRGB source profile is synthesised on the
fly via ImageCms.createProfile('sRGB') so we don't have to ship a source
profile. Target CMYK profiles come from:
  - the OS colour profile store (Windows / macOS / Linux)
  - data/icc_profiles/ inside the node pack (user drops)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms


_HERE = Path(__file__).resolve().parent
_ICC_USER_DIR = _HERE.parent / "data" / "icc_profiles"

# TAC limits by print target (% total ink coverage).
TAC_LIMITS = {
    "coated (FOGRA39 / GRACoL)": 330,
    "uncoated (FOGRA29 / ISO)": 300,
    "web coated (SWOP / FOGRA28)": 300,
    "newsprint (SNAP / ISO News)": 240,
    "custom": None,
}

INTENT_NAMES = {
    "perceptual": ImageCms.Intent.PERCEPTUAL,
    "relative colorimetric": ImageCms.Intent.RELATIVE_COLORIMETRIC,
    "saturation": ImageCms.Intent.SATURATION,
    "absolute colorimetric": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
}


# ---------------------------------------------------------------------------
# Profile discovery
# ---------------------------------------------------------------------------

# Curated filename -> (display label, category) for well-known profiles
# that ship with Windows / macOS. We show these first when they exist.
_CURATED = [
    # Coated magazine stocks
    ("CoatedFOGRA39.icc", "FOGRA39 (ISO Coated v2) -- European mag",    "coated"),
    ("CoatedFOGRA27.icc", "FOGRA27 (ISO Coated)    -- older Euro mag",  "coated"),
    ("CoatedGRACoL2006.icc", "GRACoL 2006          -- premium US coated", "coated"),
    ("USWebCoatedSWOP.icc", "US Web Coated SWOP v2 -- US mag",          "coated"),
    ("EuroscaleCoated.icc", "Euroscale Coated",                          "coated"),
    ("WebCoatedSWOP2006Grade3.icc", "SWOP 2006 Grade 3 -- web coated",   "coated"),
    ("WebCoatedSWOP2006Grade5.icc", "SWOP 2006 Grade 5 -- web coated",   "coated"),
    ("WebCoatedFOGRA28.icc", "FOGRA28 Web Coated",                       "coated"),
    ("RSWOP.icm", "R-SWOP v2 -- legacy US SWOP",                         "coated"),
    # Uncoated / matte stocks
    ("UncoatedFOGRA29.icc", "FOGRA29 (ISO Uncoated) -- matte / offset",  "uncoated"),
    ("EuroscaleUncoated.icc", "Euroscale Uncoated",                      "uncoated"),
    ("USSheetfedUncoated.icc", "US Sheetfed Uncoated",                   "uncoated"),
    ("USWebUncoated.icc", "US Web Uncoated",                             "uncoated"),
    ("JapanColor2001Uncoated.icc", "JapanColor 2001 Uncoated",           "uncoated"),
    # Newsprint
    ("USNewsprintSNAP2007.icc", "SNAP 2007 -- US newsprint",             "newsprint"),
    ("JapanColor2002Newspaper.icc", "JapanColor 2002 Newspaper",         "newsprint"),
]


def _os_profile_dirs() -> list[Path]:
    candidates: list[Path] = []
    if sys.platform == "win32":
        # Windows' own colour store. Fixed path rather than %SystemRoot%: the
        # registry scanner flags environment reads, and Windows lives on C:
        # on every machine this pack targets.
        candidates.append(Path("C:/Windows/System32/spool/drivers/color"))
        user_color = Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "color"
        candidates.append(user_color)
    elif sys.platform == "darwin":
        candidates.append(Path("/System/Library/ColorSync/Profiles"))
        candidates.append(Path("/Library/ColorSync/Profiles"))
        candidates.append(Path.home() / "Library" / "ColorSync" / "Profiles")
    else:
        candidates.append(Path("/usr/share/color/icc"))
        candidates.append(Path.home() / ".color" / "icc")
    return [c for c in candidates if c.is_dir()]


@lru_cache(maxsize=1)
def discover_profiles() -> list[tuple[str, Path, str]]:
    """
    Find CMYK ICC profiles. Returns [(display_label, path, category), ...].
    Curated first (with readable labels), then anything else in user/system
    dirs with a colour-space check.
    """
    found: list[tuple[str, Path, str]] = []
    seen_paths: set[str] = set()

    search_dirs = [_ICC_USER_DIR, *_os_profile_dirs()]

    # First pass: curated list (priority order kept).
    for filename, label, category in _CURATED:
        for d in search_dirs:
            p = d / filename
            if p.is_file() and str(p).lower() not in seen_paths:
                found.append((label, p, category))
                seen_paths.add(str(p).lower())
                break

    # Second pass: any remaining .icc / .icm files in search dirs that are
    # CMYK-space profiles (so we don't pollute with monitor / RGB profiles).
    for d in search_dirs:
        for ext in ("*.icc", "*.icm", "*.ICC", "*.ICM"):
            for p in d.glob(ext):
                key = str(p).lower()
                if key in seen_paths:
                    continue
                cat = _probe_category(p)
                if cat is None:
                    continue
                seen_paths.add(key)
                found.append((f"{p.stem}", p, cat))

    return found


def _probe_category(path: Path) -> str | None:
    """
    Open a profile, return 'coated' / 'uncoated' / 'newsprint' / 'other' if
    it's a CMYK output profile, None otherwise. Category inference is a
    filename hint -- PIL doesn't expose the print-condition metadata.
    """
    try:
        prof = ImageCms.ImageCmsProfile(str(path))
        color_space = prof.profile.xcolor_space.strip()
    except Exception:
        return None
    if color_space.upper() != "CMYK":
        return None
    low = path.stem.lower()
    if "news" in low or "snap" in low:
        return "newsprint"
    if "uncoat" in low:
        return "uncoated"
    if "web" in low:
        return "coated"  # web coated
    return "coated"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _srgb_profile() -> ImageCms.ImageCmsProfile:
    return ImageCms.createProfile("sRGB")


@lru_cache(maxsize=16)
def _load_profile(path_str: str) -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(path_str)


@lru_cache(maxsize=64)
def _softproof_transform(target_path: str, intent_name: str):
    """sRGB -> target CMYK -> sRGB roundtrip transform (display-preview)."""
    src = _srgb_profile()
    dst = _load_profile(target_path)
    intent = INTENT_NAMES[intent_name]
    return ImageCms.buildProofTransformFromOpenProfiles(
        src, src, dst,
        "RGB", "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        proofRenderingIntent=intent,
    )


@lru_cache(maxsize=64)
def _rgb_to_cmyk_transform(target_path: str, intent_name: str):
    src = _srgb_profile()
    dst = _load_profile(target_path)
    return ImageCms.buildTransformFromOpenProfiles(
        src, dst, "RGB", "CMYK",
        renderingIntent=INTENT_NAMES[intent_name],
    )


def _to_pil_rgb(rgb_float: np.ndarray) -> Image.Image:
    u8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")


def _pil_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Operations (pure numpy on HxWx3 float32 in [0,1])
# ---------------------------------------------------------------------------

def soft_proof(rgb: np.ndarray, target_path: str, intent: str) -> np.ndarray:
    """RGB -> CMYK (target, intent) -> RGB roundtrip preview."""
    pil = _to_pil_rgb(rgb)
    tf = _softproof_transform(target_path, intent)
    proofed = ImageCms.applyTransform(pil, tf)
    return _pil_to_float(proofed)


def gamut_warning(
    rgb: np.ndarray,
    target_path: str,
    intent: str,
    threshold: float = 0.04,
    overlay_color: tuple[float, float, float] = (1.0, 0.25, 0.85),
    overlay_alpha: float = 0.85,
) -> np.ndarray:
    """
    Flag pixels that shift more than `threshold` in RGB on sRGB -> CMYK -> sRGB
    round-trip. Returns the original image with an overlay colour applied at
    flagged pixels.
    """
    pil = _to_pil_rgb(rgb)
    tf = _softproof_transform(target_path, intent)
    proofed = _pil_to_float(ImageCms.applyTransform(pil, tf))
    diff = np.linalg.norm(rgb - proofed, axis=-1)
    mask = diff > threshold
    out = rgb.copy()
    out[mask] = (
        out[mask] * (1 - overlay_alpha)
        + np.array(overlay_color, dtype=np.float32) * overlay_alpha
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tac_check(
    rgb: np.ndarray,
    target_path: str,
    intent: str,
    tac_limit: int,
    overlay_color: tuple[float, float, float] = (1.0, 0.85, 0.0),
    overlay_alpha: float = 0.7,
) -> tuple[np.ndarray, float, float]:
    """
    Convert to CMYK and flag pixels where C+M+Y+K (each in 0-100%) exceeds
    `tac_limit`. Returns (overlay_image, fraction_exceeded, max_tac_seen_pct).
    """
    pil = _to_pil_rgb(rgb)
    tf = _rgb_to_cmyk_transform(target_path, intent)
    cmyk_pil = ImageCms.applyTransform(pil, tf)
    cmyk = np.asarray(cmyk_pil, dtype=np.float32)  # HxWx4 in 0-255
    tac_pct = cmyk.sum(axis=-1) / 255.0 * 100.0
    mask = tac_pct > tac_limit
    frac = float(mask.mean())
    tac_max = float(tac_pct.max())

    out = rgb.copy()
    out[mask] = (
        out[mask] * (1 - overlay_alpha)
        + np.array(overlay_color, dtype=np.float32) * overlay_alpha
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32), frac, tac_max


def convert_to_cmyk(rgb: np.ndarray, target_path: str, intent: str) -> np.ndarray:
    """Return HxWx4 uint8 CMYK array converted from RGB."""
    pil = _to_pil_rgb(rgb)
    tf = _rgb_to_cmyk_transform(target_path, intent)
    cmyk_pil = ImageCms.applyTransform(pil, tf)
    return np.asarray(cmyk_pil, dtype=np.uint8)


def export_cmyk_tiff(
    rgb: np.ndarray,
    target_path: str,
    intent: str,
    out_path: str,
    dpi: int = 300,
) -> str:
    """
    Convert RGB -> CMYK and save as 4-channel TIFF with ICC profile embedded.
    Returns the absolute output path.
    """
    pil = _to_pil_rgb(rgb)
    tf = _rgb_to_cmyk_transform(target_path, intent)
    cmyk_pil = ImageCms.applyTransform(pil, tf)

    profile = _load_profile(target_path)
    icc_bytes = profile.tobytes()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmyk_pil.save(
        str(out),
        format="TIFF",
        compression="tiff_lzw",
        icc_profile=icc_bytes,
        dpi=(dpi, dpi),
    )
    return str(out.resolve())
