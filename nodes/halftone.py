"""
Halftone node for ComfyUI-Darkroom.
Professional AM (clustered-dot) halftone screening — the newsprint / comic look.
Reproduces continuous tone as a grid of ink dots whose size modulates with tone.

Screens in DISPLAY (sRGB) space, NOT linear (design call 1): halftone reproduces
tone as seen / as printed. Round dot, angled rosette, supersample AA. torch-on-CUDA
for the screen eval with a numpy fallback (mirrors utils/color.linear_to_srgb).

This is a STYLIZE effect, not a calibrated prepress proof (use cmyk_softproof for
proofing). The CMYK separation here is the naive stylize separation, no ICC.
"""

import numpy as np

from ..utils.color import luminance_rec709, blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


# Ellipse (chain dot) geometry. The field is the elliptical distance from the cell
# centre, NORMALIZED so T reaches 1 only at the cell corners (a measure-zero point
# set) — NOT hard-clipped, which would leave a 2D plateau at T=1 that ink can never
# reach (coverage > 1 is impossible), capping ink at the inscribed-ellipse area and
# making solid blacks unreachable. Normalizing lets the dots grow, merge, and fill
# the inter-dot gaps last, spanning the full tonal range like round/square.
_ELLIPSE_ASPECT = 1.6
_ELLIPSE_NORM = float(np.hypot(0.5, _ELLIPSE_ASPECT * 0.5))  # corner distance


def _threshold_field_np(u, v, dot_shape):
    """
    Per-pixel AM threshold field T ∈ [0, 1] for the given dot shape (numpy).
    u, v are the rotated, scaled screen coords (already · 2π/p). Ink where
    coverage > T; T=0 at cell center so dots grow outward and monotonicity
    holds for every shape (see docs/halftone-derivation.md v1.x).
    """
    if dot_shape == "round":
        # UNCHANGED v1 cos+cos field (proper dot<->hole transition). Verbatim.
        D = (np.cos(u) + np.cos(v)) * 0.5
        return (1.0 - D) * 0.5
    if dot_shape == "line":
        D = np.cos(u)
        return (1.0 - D) * 0.5
    # square / ellipse use cell phase pu, pv ∈ [-0.5, 0.5)
    pu = np.mod(u / (2.0 * np.pi), 1.0) - 0.5
    pv = np.mod(v / (2.0 * np.pi), 1.0) - 0.5
    if dot_shape == "square":
        return np.clip(2.0 * np.maximum(np.abs(pu), np.abs(pv)), 0.0, 1.0)
    if dot_shape == "ellipse":
        return np.clip(np.hypot(pu, _ELLIPSE_ASPECT * pv) / _ELLIPSE_NORM, 0.0, 1.0)
    # default safety: round
    D = (np.cos(u) + np.cos(v)) * 0.5
    return (1.0 - D) * 0.5


def _threshold_field_torch(u, v, dot_shape, torch):
    """torch counterpart of _threshold_field_np."""
    two_pi = 2.0 * np.pi
    if dot_shape == "round":
        D = (torch.cos(u) + torch.cos(v)) * 0.5
        return (1.0 - D) * 0.5
    if dot_shape == "line":
        D = torch.cos(u)
        return (1.0 - D) * 0.5
    pu = torch.remainder(u / two_pi, 1.0) - 0.5
    pv = torch.remainder(v / two_pi, 1.0) - 0.5
    if dot_shape == "square":
        return torch.clamp(2.0 * torch.maximum(torch.abs(pu), torch.abs(pv)), 0.0, 1.0)
    if dot_shape == "ellipse":
        return torch.clamp(torch.hypot(pu, _ELLIPSE_ASPECT * pv) / _ELLIPSE_NORM, 0.0, 1.0)
    D = (torch.cos(u) + torch.cos(v)) * 0.5
    return (1.0 - D) * 0.5


def _build_tone_warp(dot_shape, n_grid=512, n_q=257):
    """
    Tone-linearization LUT for a dot shape (see docs/halftone-derivation.md v1.x).
    A shape's ink area as a function of input coverage c is A(c)=CDF_T(c); for
    square that is c² (washed-out midtones). To keep dot_shape a PURE-GEOMETRY
    control (no tone shift), we pre-warp coverage by warp(c)=CDF_T⁻¹(c)=quantile_T(c)
    so ink area = CDF_T(warp(c)) = c for every shape. Returns (q, warp_vals) for
    np.interp. (Numerically this yields exactly sqrt(c) for square.)
    """
    lin = (np.arange(n_grid, dtype=np.float64) + 0.5) / n_grid - 0.5  # pu,pv ∈ [-0.5,0.5)
    pu, pv = np.meshgrid(lin, lin)
    if dot_shape == "square":
        T = np.clip(2.0 * np.maximum(np.abs(pu), np.abs(pv)), 0.0, 1.0)
    elif dot_shape == "ellipse":
        T = np.clip(np.hypot(pu, _ELLIPSE_ASPECT * pv) / _ELLIPSE_NORM, 0.0, 1.0)
    else:
        return None
    q = np.linspace(0.0, 1.0, n_q, dtype=np.float64)
    warp_vals = np.quantile(T.ravel(), q)
    return q.astype(np.float32), warp_vals.astype(np.float32)


# Precomputed once at import. round/line are already ~tone-linear → no warp (and
# round stays the verbatim shipped look). square/ellipse get tone-linearized.
_TONE_WARP = {s: _build_tone_warp(s) for s in ("square", "ellipse")}


def _apply_tone_warp(coverage, dot_shape):
    """Pre-warp coverage so ink area ≈ coverage for this shape (tone-preserving)."""
    lut = _TONE_WARP.get(dot_shape)
    if lut is None:
        return coverage  # round / line — already tone-linear, left verbatim
    q, warp_vals = lut
    return np.interp(coverage, q, warp_vals).astype(np.float32)


def _screen(coverage, angle_deg, lines, supersample, long_edge, flip=False, dot_shape="round"):
    """
    AM clustered-dot screen of a single ink coverage channel.

    coverage   : (H, W) float32 in [0, 1], 1 = full ink.
    angle_deg  : screen angle in degrees.
    lines      : halftone lines across the long edge (resolution-independent).
    supersample: ss; the binary ink is evaluated on an ss× finer grid and
                 box-averaged back per output pixel for anti-aliasing.
    long_edge  : max(H, W) of the image (pitch reference).
    flip       : NEGATIVE-CONTROL ONLY — inverts the ink comparison (c < T).
                 Production path is always flip=False.
    dot_shape  : "round" | "line" | "square" | "ellipse" — AM spot function.

    Returns ink (H, W) float32 in [0, 1].
    """
    # Tone-linearize coverage so dot_shape changes geometry only, not tone
    # (no-op for round/line). Applied once here so both torch and numpy paths
    # below see the warped coverage.
    coverage = _apply_tone_warp(coverage, dot_shape)

    h, w = coverage.shape
    ss = int(supersample)

    # Dot pitch in px. Floor at 3px so dots never go sub-pixel (visibility floor).
    p = max(float(long_edge) / float(lines), 3.0)
    two_pi_over_p = (2.0 * np.pi) / p

    theta = np.deg2rad(float(angle_deg))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # ss× finer pixel coordinate grid. Subpixel centers map back to output pixels
    # in contiguous ss×ss blocks, so a simple reshape+mean is the box average.
    hs, ws = h * ss, w * ss
    # coordinate in source-pixel units (so pitch p stays in image px regardless of ss)
    inv_ss = 1.0 / ss

    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            cov_t = torch.from_numpy(np.ascontiguousarray(coverage, dtype=np.float32)).to(dev)
            # Upsample coverage to the ss× grid by nearest (repeat on both axes).
            cov_up = cov_t.repeat_interleave(ss, dim=0).repeat_interleave(ss, dim=1)

            ys = (torch.arange(hs, device=dev, dtype=torch.float32) + 0.5) * inv_ss
            xs = (torch.arange(ws, device=dev, dtype=torch.float32) + 0.5) * inv_ss
            yy = ys.view(hs, 1)
            xx = xs.view(1, ws)

            u = (xx * cos_t + yy * sin_t) * two_pi_over_p
            v = (-xx * sin_t + yy * cos_t) * two_pi_over_p
            T = _threshold_field_torch(u, v, dot_shape, torch)   # broadcasts to (hs, ws)

            if flip:
                ink_hi = (cov_up < T).to(torch.float32)
            else:
                ink_hi = (cov_up > T).to(torch.float32)

            # Box-average each ss×ss block back to (h, w).
            if ss > 1:
                ink = ink_hi.view(h, ss, w, ss).mean(dim=(1, 3))
            else:
                ink = ink_hi
            return ink.cpu().numpy().astype(np.float32)
    except Exception:
        pass

    # ---- numpy fallback ----
    cov_up = np.repeat(np.repeat(coverage, ss, axis=0), ss, axis=1)
    ys = (np.arange(hs, dtype=np.float32) + 0.5) * inv_ss
    xs = (np.arange(ws, dtype=np.float32) + 0.5) * inv_ss
    yy = ys[:, None]
    xx = xs[None, :]

    u = (xx * cos_t + yy * sin_t) * two_pi_over_p
    v = (-xx * sin_t + yy * cos_t) * two_pi_over_p
    T = _threshold_field_np(u, v, dot_shape)

    if flip:
        ink_hi = (cov_up < T).astype(np.float32)
    else:
        ink_hi = (cov_up > T).astype(np.float32)

    if ss > 1:
        ink = ink_hi.reshape(h, ss, w, ss).mean(axis=(1, 3))
    else:
        ink = ink_hi
    return ink.astype(np.float32)


def _bayer_matrix(n):
    """
    n×n Bayer ordered-dither matrix (n a power of 2), built recursively.
    Returns integer matrix with values 0..n²-1. Asset-free, fully vectorized.
    """
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    half = _bayer_matrix(n // 2)
    return np.block([
        [4.0 * half + 0.0, 4.0 * half + 2.0],
        [4.0 * half + 3.0, 4.0 * half + 1.0],
    ]).astype(np.float32)


# Normalized 8×8 Bayer threshold field thr = (bayer + 0.5)/n² ∈ (0, 1).
_BAYER_N = 8
_BAYER_THR = (_bayer_matrix(_BAYER_N) + 0.5) / (_BAYER_N * _BAYER_N)


def _screen_fm(coverage, offset=(0, 0)):
    """
    FM (dispersed / Bayer) screen of a single ink coverage channel.

    Tiles the normalized Bayer threshold field over the image at native
    resolution (no supersample, no angle, no dot_shape — FM is shapeless).
    `offset` is an integer pixel shift of the tile origin, used per CMYK
    channel to decorrelate the plates.

    Returns ink (H, W) float32 in [0, 1] (0/1, dispersed dots).
    """
    h, w = coverage.shape
    oy, ox = int(offset[0]), int(offset[1])
    rows = (np.arange(h, dtype=np.int64) + oy) % _BAYER_N
    cols = (np.arange(w, dtype=np.int64) + ox) % _BAYER_N
    tiled = _BAYER_THR[np.ix_(rows, cols)]
    return (coverage > tiled).astype(np.float32)


# Per-channel Bayer tile offsets for CMYK FM — distinct shifts decorrelate the
# plates so inks don't all land on the same pixels (muddy otherwise).
_FM_OFFSETS = {"c": (0, 0), "m": (3, 1), "y": (5, 4), "k": (7, 2)}


# Standard CMYK rosette angles (30° separation that avoids moiré). FIXED.
_CMYK_ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}


class Halftone:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "color_mode": (["mono (black)", "color (CMYK)"], {
                    "default": "mono (black)",
                    "tooltip": "mono = newsprint black-on-white; color = naive CMYK rosette"
                }),
                "method": (["AM (clustered dot)", "FM (dispersed / Bayer)"], {
                    "default": "AM (clustered dot)",
                    "tooltip": "AM = clustered dots (size-modulated). FM = dispersed Bayer dither (stochastic-style). FM ignores dot_shape, angle, and supersample."
                }),
                "dot_shape": (["round", "ellipse", "line", "square"], {
                    "default": "round",
                    "tooltip": "AM dot shape. round = classic; line = engraving/linocut screen; square = retro; ellipse = chain dot. AM only."
                }),
                "lines": ("INT", {
                    "default": 100, "min": 20, "max": 400, "step": 1,
                    "tooltip": "Halftone lines across the long edge (resolution-independent screen frequency)"
                }),
                "angle": ("FLOAT", {
                    "default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Mono screen angle in degrees (classic newspaper = 45°). Ignored in CMYK (fixed rosette)."
                }),
                "black_generation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "CMYK only: 1 = full GCR (K plate from min CMY), 0 = CMY only"
                }),
                "supersample": ("INT", {
                    "default": 2, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Dot-edge anti-aliasing. Higher = smoother edges + slower."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend original (0) ↔ halftone (1). <1 = subtle screen overlay."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Print"

    def execute(self, image, color_mode="mono (black)",
                method="AM (clustered dot)", dot_shape="round",
                lines=100, angle=45.0,
                black_generation=1.0, supersample=2, strength=1.0):

        if strength <= 0.0:
            return (image,)

        is_fm = method == "FM (dispersed / Bayer)"

        print(f"[Darkroom] Halftone: mode={color_mode}, method={method}, "
              f"dot_shape={dot_shape}, lines={lines}, "
              f"angle={angle}, black_gen={black_generation}, ss={supersample}, "
              f"strength={strength}")

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            h, w = img.shape[0], img.shape[1]
            long_edge = max(h, w)

            if color_mode == "color (CMYK)":
                r = img[..., 0]
                g = img[..., 1]
                b = img[..., 2]
                c = 1.0 - r
                m = 1.0 - g
                y = 1.0 - b
                k = float(black_generation) * np.minimum(np.minimum(c, m), y)
                c = np.clip(c - k, 0.0, 1.0)
                m = np.clip(m - k, 0.0, 1.0)
                y = np.clip(y - k, 0.0, 1.0)

                if is_fm:
                    c_ink = _screen_fm(c, _FM_OFFSETS["c"])
                    m_ink = _screen_fm(m, _FM_OFFSETS["m"])
                    y_ink = _screen_fm(y, _FM_OFFSETS["y"])
                    k_ink = _screen_fm(k, _FM_OFFSETS["k"])
                else:
                    c_ink = _screen(c, _CMYK_ANGLES["c"], lines, supersample, long_edge, dot_shape=dot_shape)
                    m_ink = _screen(m, _CMYK_ANGLES["m"], lines, supersample, long_edge, dot_shape=dot_shape)
                    y_ink = _screen(y, _CMYK_ANGLES["y"], lines, supersample, long_edge, dot_shape=dot_shape)
                    k_ink = _screen(k, _CMYK_ANGLES["k"], lines, supersample, long_edge, dot_shape=dot_shape)

                out_r = (1.0 - c_ink) * (1.0 - k_ink)
                out_g = (1.0 - m_ink) * (1.0 - k_ink)
                out_b = (1.0 - y_ink) * (1.0 - k_ink)
                out = np.stack([out_r, out_g, out_b], axis=-1).astype(np.float32)
            else:
                # mono (black) — newsprint identity
                coverage = 1.0 - luminance_rec709(img)
                if is_fm:
                    ink = _screen_fm(coverage, _FM_OFFSETS["k"])
                else:
                    ink = _screen(coverage, angle, lines, supersample, long_edge, dot_shape=dot_shape)
                tone = 1.0 - ink
                out = np.stack([tone, tone, tone], axis=-1).astype(np.float32)

            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomHalftone": Halftone}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHalftone": "Halftone"}
