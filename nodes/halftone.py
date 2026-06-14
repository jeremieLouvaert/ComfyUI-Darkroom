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


def _screen(coverage, angle_deg, lines, supersample, long_edge, flip=False):
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

    Returns ink (H, W) float32 in [0, 1].
    """
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
            D = (torch.cos(u) + torch.cos(v)) * 0.5      # broadcasts to (hs, ws)
            T = (1.0 - D) * 0.5

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
    D = (np.cos(u) + np.cos(v)) * 0.5
    T = (1.0 - D) * 0.5

    if flip:
        ink_hi = (cov_up < T).astype(np.float32)
    else:
        ink_hi = (cov_up > T).astype(np.float32)

    if ss > 1:
        ink = ink_hi.reshape(h, ss, w, ss).mean(axis=(1, 3))
    else:
        ink = ink_hi
    return ink.astype(np.float32)


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

    def execute(self, image, color_mode="mono (black)", lines=100, angle=45.0,
                black_generation=1.0, supersample=2, strength=1.0):

        if strength <= 0.0:
            return (image,)

        print(f"[Darkroom] Halftone: mode={color_mode}, lines={lines}, "
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

                c_ink = _screen(c, _CMYK_ANGLES["c"], lines, supersample, long_edge)
                m_ink = _screen(m, _CMYK_ANGLES["m"], lines, supersample, long_edge)
                y_ink = _screen(y, _CMYK_ANGLES["y"], lines, supersample, long_edge)
                k_ink = _screen(k, _CMYK_ANGLES["k"], lines, supersample, long_edge)

                out_r = (1.0 - c_ink) * (1.0 - k_ink)
                out_g = (1.0 - m_ink) * (1.0 - k_ink)
                out_b = (1.0 - y_ink) * (1.0 - k_ink)
                out = np.stack([out_r, out_g, out_b], axis=-1).astype(np.float32)
            else:
                # mono (black) — newsprint identity
                coverage = 1.0 - luminance_rec709(img)
                ink = _screen(coverage, angle, lines, supersample, long_edge)
                tone = 1.0 - ink
                out = np.stack([tone, tone, tone], axis=-1).astype(np.float32)

            results.append(blend(original, out, strength))

        return (numpy_batch_to_tensor(results),)


NODE_CLASS_MAPPINGS = {"DarkroomHalftone": Halftone}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHalftone": "Halftone"}
