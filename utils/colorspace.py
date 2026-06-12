"""
Color space conversion utilities for ComfyUI-Darkroom.
Pure numpy — no OpenColorIO dependency.

Supports: sRGB, Linear sRGB, ACEScg, ACEScct, Rec.2020, DCI-P3 (Display P3).
All matrices include D65↔D60 chromatic adaptation where needed (Bradford).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 3x3 conversion matrices (row-vector convention: result = input @ M.T)
# All derived from CIE XYZ with Bradford chromatic adaptation D65↔D60.
# ---------------------------------------------------------------------------

# sRGB linear (D65) → ACEScg / AP1 (D60)
_SRGB_TO_ACES = np.array([
    [0.6131324,  0.3395380,  0.0473296],
    [0.0701996,  0.9163554,  0.0134450],
    [0.0206155,  0.1095698,  0.8698147],
], dtype=np.float32)

# ACEScg / AP1 (D60) → sRGB linear (D65)
_ACES_TO_SRGB = np.array([
    [ 1.7050510, -0.6217921, -0.0832590],
    [-0.1302564,  1.1408048, -0.0105484],
    [-0.0240034, -0.1289690,  1.1529724],
], dtype=np.float32)

# sRGB linear (D65) → Rec.2020 linear (D65)
_SRGB_TO_REC2020 = np.array([
    [0.6274040,  0.3292820,  0.0433140],
    [0.0690970,  0.9195400,  0.0113630],
    [0.0163916,  0.0880132,  0.8955952],
], dtype=np.float32)

# Rec.2020 linear (D65) → sRGB linear (D65)
_REC2020_TO_SRGB = np.array([
    [ 1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297],
], dtype=np.float32)

# sRGB linear (D65) → Display P3 linear (D65)
_SRGB_TO_P3 = np.array([
    [0.8224622,  0.1775378,  0.0000000],
    [0.0331942,  0.9668058,  0.0000000],
    [0.0170804,  0.0723897,  0.9105299],
], dtype=np.float32)

# Display P3 linear (D65) → sRGB linear (D65)
_P3_TO_SRGB = np.array([
    [ 1.2249401, -0.2249402,  0.0000000],
    [-0.0420569,  1.0420571,  0.0000000],
    [-0.0196376, -0.0786361,  1.0982735],
], dtype=np.float32)

# ACES AP0 (D60) — for reference / RRT input
_SRGB_TO_AP0 = np.array([
    [0.4397010,  0.3829780,  0.1773210],
    [0.0897923,  0.8134230,  0.0967847],
    [0.0175440,  0.1115440,  0.8709120],
], dtype=np.float32)

_AP0_TO_SRGB = np.array([
    [ 2.5216494, -1.1368885, -0.3847609],
    [-0.2752135,  1.3697052, -0.0944917],
    [-0.0159271, -0.1478048,  1.1637319],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# OkLab (Björn Ottosson, 2020 — canonical published constants).
# Perceptually-uniform Lab built on a cube-root LMS cone response.
# Forward:  linear sRGB -> LMS (M1) -> cbrt -> Lab (M2)
# Inverse:  Lab -> LMS' (M2_inv) -> cube -> linear sRGB (M1_inv)
# Note row-vector convention (result = img @ M.T) per _apply_matrix.
# ---------------------------------------------------------------------------

# M1: linear sRGB (D65) -> LMS
_OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

# M2: LMS' (cube-rooted) -> Lab
_OKLAB_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)

# inverse M2: Lab -> LMS'
_OKLAB_M2_INV = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], dtype=np.float32)

# inverse M1: LMS -> linear sRGB (D65)
_OKLAB_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
], dtype=np.float32)


def linear_srgb_to_oklab(img):
    """Linear sRGB (D65, H,W,3) -> OkLab (L, a, b). Sign-preserving cbrt."""
    lms = _apply_matrix(img.astype(np.float32), _OKLAB_M1)
    lms_ = np.cbrt(lms).astype(np.float32)
    return _apply_matrix(lms_, _OKLAB_M2)


def oklab_to_linear_srgb(lab):
    """OkLab (L, a, b) -> linear sRGB (D65). Inverse of linear_srgb_to_oklab."""
    lms_ = _apply_matrix(lab.astype(np.float32), _OKLAB_M2_INV)
    lms = (lms_ ** 3).astype(np.float32)
    return _apply_matrix(lms, _OKLAB_M1_INV)


def oklab_to_oklch(lab):
    """OkLab (L, a, b) -> OkLch (L, C, h). h in radians via atan2(b, a)."""
    lab = lab.astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.hypot(a, b).astype(np.float32)
    h = np.arctan2(b, a).astype(np.float32)
    return np.stack([L, C, h], axis=-1).astype(np.float32)


def oklch_to_oklab(lch):
    """OkLch (L, C, h) -> OkLab (L, a, b). a = C*cos(h), b = C*sin(h)."""
    lch = lch.astype(np.float32)
    L = lch[..., 0]
    C = lch[..., 1]
    h = lch[..., 2]
    a = (C * np.cos(h)).astype(np.float32)
    b = (C * np.sin(h)).astype(np.float32)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# sRGB gamma (shared with color.py but repeated here to avoid circular import)
# ---------------------------------------------------------------------------

def srgb_decode(img):
    """sRGB → linear (remove display gamma)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return np.where(img <= 0.04045, img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb_encode(img):
    """Linear → sRGB (apply display gamma)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return np.where(img <= 0.0031308, img * 12.92,
                    1.055 * np.power(img, 1.0 / 2.4) - 0.055).astype(np.float32)


# ---------------------------------------------------------------------------
# ACEScct log encoding (S-2016-001)
# ---------------------------------------------------------------------------

_ACESCCT_CUT_LIN = 0.0078125       # 2^(-7)
_ACESCCT_CUT_LOG = 0.155251141552511
_ACESCCT_A = 10.5402377416545
_ACESCCT_B = 0.0729055341958355


def linear_to_acescct(x):
    """ACEScg linear → ACEScct (log-encoded for grading)."""
    x = np.maximum(x, 0.0).astype(np.float32)
    return np.where(
        x <= _ACESCCT_CUT_LIN,
        _ACESCCT_A * x + _ACESCCT_B,
        (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52
    ).astype(np.float32)


def acescct_to_linear(x):
    """ACEScct (log-encoded) → ACEScg linear."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(
        x <= _ACESCCT_CUT_LOG,
        (x - _ACESCCT_B) / _ACESCCT_A,
        np.power(2.0, x * 17.52 - 9.72)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Matrix application
# ---------------------------------------------------------------------------

def _apply_matrix(img, matrix):
    """Apply a 3x3 color matrix to an (H, W, 3) image. result = img @ matrix.T"""
    return (img @ matrix.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Public conversion functions
# ---------------------------------------------------------------------------

# Map of (source, target) → (decode_gamma, matrix_chain, encode_gamma)
# "linear" spaces skip gamma. "log" spaces use ACEScct encoding.

_SPACES = {
    "sRGB":         {"gamma": "srgb"},
    "Linear sRGB":  {"gamma": None},
    "ACEScg":       {"gamma": None,    "from_srgb_linear": _SRGB_TO_ACES,  "to_srgb_linear": _ACES_TO_SRGB},
    "ACEScct":      {"gamma": "acescct", "from_srgb_linear": _SRGB_TO_ACES, "to_srgb_linear": _ACES_TO_SRGB},
    "Rec.2020":     {"gamma": None,    "from_srgb_linear": _SRGB_TO_REC2020, "to_srgb_linear": _REC2020_TO_SRGB},
    "DCI-P3":       {"gamma": None,    "from_srgb_linear": _SRGB_TO_P3,    "to_srgb_linear": _P3_TO_SRGB},
}

SPACE_NAMES = list(_SPACES.keys())


def convert_colorspace(img, source, target):
    """
    Convert image between color spaces.

    img: (H, W, 3) float32, values in the source space's native encoding.
    source, target: one of SPACE_NAMES.
    Returns: (H, W, 3) float32 in the target space's native encoding.
    """
    if source == target:
        return img.copy()

    src = _SPACES[source]
    tgt = _SPACES[target]

    # Step 1: decode source to sRGB linear
    linear = img.copy()
    if src["gamma"] == "srgb":
        linear = srgb_decode(linear)
    elif src["gamma"] == "acescct":
        linear = acescct_to_linear(linear)
        linear = _apply_matrix(linear, _ACES_TO_SRGB)

    # If source has a gamut matrix, convert from source gamut to sRGB linear
    if "to_srgb_linear" in src and src["gamma"] != "acescct":
        linear = _apply_matrix(linear, src["to_srgb_linear"])

    # Step 2: from sRGB linear to target gamut
    if "from_srgb_linear" in tgt:
        linear = _apply_matrix(linear, tgt["from_srgb_linear"])

    # Step 3: encode target gamma
    if tgt["gamma"] == "srgb":
        result = srgb_encode(np.clip(linear, 0.0, 1.0))
    elif tgt["gamma"] == "acescct":
        result = linear_to_acescct(linear)
    else:
        result = linear

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Tonemapping curves
# ---------------------------------------------------------------------------

def aces_narkowicz(x):
    """
    ACES filmic tonemapping (Krzysztof Narkowicz fit).
    Input: linear scene-referred values (can exceed 1.0).
    Output: display-referred [0, 1].
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0).astype(np.float32)


def aces_hill(x):
    """
    ACES fitted tonemapping (Stephen Hill / Academy fit).
    More accurate S-curve than Narkowicz. Slightly more contrast in shadows.
    Input: linear (ACEScg or scene-referred). Output: [0, 1].
    """
    # RRT + ODT fitted curve (sRGB output)
    a = x * (x + 0.0245786) - 0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    return np.clip(a / b, 0.0, 1.0).astype(np.float32)


def reinhard(x):
    """Simple Reinhard tonemapping. x / (1 + x)."""
    return np.clip(x / (1.0 + x), 0.0, 1.0).astype(np.float32)


def reinhard_extended(x, white_point=4.0):
    """Extended Reinhard with configurable white point."""
    numer = x * (1.0 + x / (white_point * white_point))
    denom = 1.0 + x
    return np.clip(numer / denom, 0.0, 1.0).astype(np.float32)


def agx_base(x):
    """
    AgX tonemapping base curve (Blender/Filament style).
    Attempt at a punchy, neutral look with good highlight rolloff.
    Input: linear (ideally in AgX log-encoded space). Output: [0, 1].
    """
    # AgX log encoding: compress [0, ~16.3] into [0, 1]
    x = np.maximum(x, 1e-10)
    x_log = np.clip(np.log2(x) / 16.0 + 0.5, 0.0, 1.0)

    # Polynomial fit for the AgX sigmoid
    # Approximation of the Blender AgX base contrast curve
    x2 = x_log * x_log
    x4 = x2 * x2
    result = (
        15.5 * x4 * x2
        - 40.14 * x4 * x_log
        + 31.96 * x4
        - 6.868 * x2 * x_log
        + 0.4298 * x2
        + 0.1191 * x_log
        - 0.00232
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def uncharted2(x):
    """
    Uncharted 2 / Hejl-Burgess-Dawson filmic tonemapping.
    Warm, filmic rolloff with strong shoulder. Popular in games.
    """
    # Operator parameters (John Hable)
    A = 0.15   # shoulder strength
    B = 0.50   # linear strength
    C = 0.10   # linear angle
    D = 0.20   # toe strength
    E = 0.02   # toe numerator
    F = 0.30   # toe denominator

    def tonemap(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    white = 11.2
    result = tonemap(x) / tonemap(white)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


TONEMAP_CURVES = {
    "ACES Filmic (Narkowicz)":  aces_narkowicz,
    "ACES Fitted (Hill)":       aces_hill,
    "AgX (Blender)":            agx_base,
    "Reinhard":                 reinhard,
    "Reinhard Extended":        reinhard_extended,
    "Filmic (Uncharted 2)":     uncharted2,
}

TONEMAP_NAMES = list(TONEMAP_CURVES.keys())
