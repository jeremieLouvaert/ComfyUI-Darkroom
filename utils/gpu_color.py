"""
Torch colour core for ComfyUI-Darkroom.

Faithful torch ports of the numpy helpers in utils/color.py, utils/grading.py,
utils/raw.py, utils/colorspace.py and utils/lut.py, so a node can run its whole
pipeline on one device: one upload, one download, no per-image Python loop.

Rules every function here keeps:
  * same maths, same clamps, same epsilon placement as the numpy original
    (tools/test_gpu_core.py gates node outputs at 0.5/255 against goldens
    captured from the numpy code; tools/test_gpu_color_ops.py gates each op);
  * tensors are (..., 3) or (...) float32; the leading dims can be (B, H, W),
    so a whole IMAGE batch goes through in one call;
  * 3x3 colour matrices are applied as explicit per-channel sums, never with
    matmul: ComfyUI may enable TF32 matmul on Ampere GPUs, which would cost
    ~1e-3 relative error;
  * torch.where evaluates both branches, so values feeding a pow in the
    branch that is NOT taken are clamped first (no NaN reaches the output).

The numpy helpers are untouched: they are the reference these ports are tested
against, and still serve the nodes that have not moved to torch.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device and the node wrapper
# ---------------------------------------------------------------------------

def device():
    """ComfyUI's compute device (honours --cpu and device selection)."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run(fn, image, dev):
    with torch.no_grad():
        out = fn(image.to(dev, torch.float32))
        return out.to("cpu", torch.float32).contiguous()


def run_on_device(fn, image):
    """
    Run fn on an IMAGE batch (B, H, W, 3) on ComfyUI's device and return a CPU
    float32 tensor, which is what ComfyUI expects downstream. On GPU
    out-of-memory: retry one image at a time, then on the CPU.
    """
    dev = device()
    try:
        return _run(fn, image, dev)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    print("[Darkroom] GPU out of memory, retrying one image at a time")
    try:
        return torch.cat([_run(fn, image[i:i + 1], dev) for i in range(image.shape[0])], 0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    print("[Darkroom] GPU out of memory, falling back to CPU")
    return _run(fn, image, torch.device("cpu"))


def const(values, like):
    """A float32 constant on the same device as `like`."""
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=like.device)


# ---------------------------------------------------------------------------
# utils/color.py
# ---------------------------------------------------------------------------

def srgb_to_linear(img):
    img = img.clamp(0.0, 1.0)
    return torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    img = img.clamp(0.0, 1.0)
    return torch.where(img <= 0.0031308, img * 12.92, 1.055 * torch.pow(img, 1.0 / 2.4) - 0.055)


def luminance_rec709(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def characteristic_curve(x, toe_power, shoulder_power, slope, pivot_x=0.18, pivot_y=0.18):
    x = x.clamp(0.0, 1.0)
    t_toe = (x / (pivot_x + 1e-10)).clamp(min=0.0)
    toe = pivot_y * torch.pow(t_toe, toe_power) * slope
    t_sh = (x - pivot_x) / (1.0 - pivot_x + 1e-10)
    shoulder = pivot_y * slope + (1.0 - pivot_y * slope) * (
        1.0 - torch.pow((1.0 - t_sh).clamp(min=0.0), shoulder_power))
    return torch.where(x <= pivot_x, toe, shoulder).clamp(0.0, 1.0)


def apply_per_channel_curves(img, r_params, g_params, b_params):
    return torch.stack([characteristic_curve(img[..., 0], *r_params),
                        characteristic_curve(img[..., 1], *g_params),
                        characteristic_curve(img[..., 2], *b_params)], dim=-1)


def adjust_saturation(img, factor):
    if abs(factor - 1.0) < 0.001:
        return img
    lum = luminance_rec709(img)[..., None]
    return (lum + factor * (img - lum)).clamp(0.0, 1.0)


def split_tone(img, shadow_tint, highlight_tint, balance=0.5):
    lum = luminance_rec709(img)
    sw = (1.0 - lum / (balance + 1e-10)).clamp(0.0, 1.0)
    hw = ((lum - balance) / (1.0 - balance + 1e-10)).clamp(0.0, 1.0)
    result = img + sw[..., None] * const(shadow_tint, img)
    result = result + hw[..., None] * const(highlight_tint, img)
    return result.clamp(0.0, 1.0)


def blend(original, processed, strength):
    if strength >= 1.0:
        return processed
    if strength <= 0.0:
        return original
    return original * (1.0 - strength) + processed * strength


# ---------------------------------------------------------------------------
# utils/grading.py
# ---------------------------------------------------------------------------

def _pchip_pieces(control_points):
    """Breakpoints and cubic coefficients of scipy's PCHIP (extrapolate=True),
    computed once on the CPU in float64 exactly as the numpy helper does."""
    from scipy.interpolate import PchipInterpolator
    pts = sorted(control_points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    pp = PchipInterpolator(xs, ys, extrapolate=True)
    return pp.x, pp.c  # x: (n,), c: (4, n-1), highest power first


def cubic_spline_curve(x, control_points):
    """Evaluate the same piecewise cubic as grading.cubic_spline_curve."""
    xb, c = _pchip_pieces(control_points)
    breaks = const(xb, x)
    coef = const(c, x)
    n = breaks.numel()
    # interval i with breaks[i] <= x < breaks[i+1]; ends extrapolate with the
    # first/last piece, like scipy's PPoly
    i = torch.searchsorted(breaks[1:-1].contiguous(), x.contiguous(), right=True) if n > 2 \
        else torch.zeros_like(x, dtype=torch.long)
    dx = x - breaks[i]
    y = ((coef[0][i] * dx + coef[1][i]) * dx + coef[2][i]) * dx + coef[3][i]
    return y.clamp(0.0, 1.0)


def lgg_zone_masks(luminance):
    lum = luminance.clamp(0.0, 1.0)
    lift = torch.pow(1.0 - lum, 1.5)
    gain = torch.pow(lum, 1.5)
    gamma = 1.0 - torch.abs(2.0 * lum - 1.0) ** 1.5
    return lift, gamma, gain


def log_zone_masks(luminance, shadow_range=0.15, highlight_range=0.85):
    eps = 1e-6
    log_lum = torch.log2(luminance.clamp(eps, 1.0))
    log_min = float(np.log2(eps))
    log_norm = ((log_lum - log_min) / (0.0 - log_min)).clamp(0.0, 1.0)
    log_shadow = (float(np.log2(max(shadow_range, eps))) - log_min) / (0.0 - log_min)
    log_highlight = (float(np.log2(max(highlight_range, eps))) - log_min) / (0.0 - log_min)
    log_mid = (log_shadow + log_highlight) * 0.5
    shadow_sigma = max(log_shadow * 0.6, 0.05)
    highlight_sigma = max((1.0 - log_highlight) * 0.6, 0.05)
    mid_sigma = (log_highlight - log_shadow) * 0.4
    shadow = torch.exp(-0.5 * ((log_norm - log_shadow * 0.5) / shadow_sigma) ** 2)
    mid = torch.exp(-0.5 * ((log_norm - log_mid) / mid_sigma) ** 2)
    high = torch.exp(-0.5 * ((log_norm - (1.0 + log_highlight) * 0.5) / highlight_sigma) ** 2)
    return shadow, mid, high


def hue_range_mask(hue, center, width=45.0, softness=0.5):
    diff = torch.abs(hue - center)
    diff = torch.minimum(diff, 360.0 - diff)
    eff = width * (0.5 + softness * 0.5)
    weight = ((1.0 + torch.cos(np.pi * diff / max(eff, 1.0))) * 0.5).clamp(0.0, 1.0)
    return torch.where(diff > eff, torch.zeros_like(weight), weight)


def apply_lgg(img, lift_rgb, gamma_rgb, gain_rgb, offset_rgb):
    chans = []
    for c in range(3):
        x = img[..., c]
        g = max(float(gamma_rgb[c]), 0.01)
        lifted = (float(lift_rgb[c]) * (1.0 - x) + x).clamp(min=0.0)
        chans.append(float(gain_rgb[c]) * torch.pow(lifted, 1.0 / g) + float(offset_rgb[c]))
    return torch.stack(chans, dim=-1).clamp(0.0, 1.0)


def apply_color_tint_to_zone(img, zone_mask, hue_angle, intensity):
    if intensity < 0.5:
        return img
    h_rad = np.radians(hue_angle)
    tint = np.array([np.cos(h_rad) * 0.5 + 0.5,
                     np.cos(h_rad - 2.094395) * 0.5 + 0.5,
                     np.cos(h_rad + 2.094395) * 0.5 + 0.5], dtype=np.float32)
    tint = tint / (tint.sum() + 1e-10)
    tint_scaled = tint * (intensity / 100.0 * 0.3)
    return (img + zone_mask[..., None] * const(tint_scaled, img)).clamp(0.0, 1.0)


def sat_from_rgb(img):
    cmax = img.amax(dim=-1)
    cmin = img.amin(dim=-1)
    return torch.where(cmax > 1e-7, (cmax - cmin) / (cmax + 1e-10), torch.zeros_like(cmax))


def sat_range_mask(saturation, center, width=0.15):
    return torch.exp(-0.5 * ((saturation - center) / (width + 1e-10)) ** 2)


# ---------------------------------------------------------------------------
# utils/raw.py (HSL)
# ---------------------------------------------------------------------------

def rgb_to_hsl(img):
    """Same epsilons and branch order as raw.rgb_to_hsl."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = torch.maximum(torch.maximum(r, g), b)
    cmin = torch.minimum(torch.minimum(r, g), b)
    delta = cmax - cmin
    l = (cmax + cmin) * 0.5

    mask = delta > 1e-7
    zero = torch.zeros_like(l)
    s = torch.where(mask & (l <= 0.5), delta / (cmax + cmin + 1e-10), zero)
    s = torch.where(mask & (l > 0.5), delta / (2.0 - cmax - cmin + 1e-10), s)

    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g) & ~mask_r
    mask_b = mask & ~mask_r & ~mask_g
    d = delta + 1e-10
    h = torch.where(mask_r, 60.0 * (((g - b) / d) % 6), zero)
    h = torch.where(mask_g, 60.0 * (((b - r) / d) + 2), h)
    h = torch.where(mask_b, 60.0 * (((r - g) / d) + 4), h)
    return h % 360.0, s, l


def hsl_to_rgb(h, s, l):
    c = (1.0 - torch.abs(2.0 * l - 1.0)) * s
    hp = (h / 60.0) % 6.0
    x = c * (1.0 - torch.abs(hp % 2.0 - 1.0))
    m = l - c * 0.5
    zero = torch.zeros_like(h)
    r, g, b = zero, zero, zero
    for k, (rv, gv, bv) in enumerate([(c, x, None), (x, c, None), (None, c, x),
                                      (None, x, c), (x, None, c), (c, None, x)]):
        sel = (hp >= k) & (hp < k + 1)
        if rv is not None:
            r = torch.where(sel, rv, r)
        if gv is not None:
            g = torch.where(sel, gv, g)
        if bv is not None:
            b = torch.where(sel, bv, b)
    return torch.stack([r + m, g + m, b + m], dim=-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# utils/colorspace.py
# ---------------------------------------------------------------------------

def apply_matrix(img, matrix):
    """img @ matrix.T as explicit per-channel sums (TF32-proof)."""
    m = np.asarray(matrix, dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return torch.stack([float(m[i, 0]) * r + float(m[i, 1]) * g + float(m[i, 2]) * b
                        for i in range(3)], dim=-1)


srgb_decode = srgb_to_linear
srgb_encode = linear_to_srgb


def _cbrt(x):
    return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)


def linear_srgb_to_oklab(img):
    from .colorspace import _OKLAB_M1, _OKLAB_M2
    return apply_matrix(_cbrt(apply_matrix(img, _OKLAB_M1)), _OKLAB_M2)


def oklab_to_linear_srgb(lab):
    from .colorspace import _OKLAB_M1_INV, _OKLAB_M2_INV
    return apply_matrix(apply_matrix(lab, _OKLAB_M2_INV) ** 3, _OKLAB_M1_INV)


def oklab_to_oklch(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return torch.stack([L, torch.hypot(a, b), torch.atan2(b, a)], dim=-1)


def oklch_to_oklab(lch):
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    return torch.stack([L, C * torch.cos(h), C * torch.sin(h)], dim=-1)


def linear_to_acescct(x):
    from .colorspace import _ACESCCT_A, _ACESCCT_B, _ACESCCT_CUT_LIN
    x = x.clamp(min=0.0)
    return torch.where(x <= _ACESCCT_CUT_LIN, _ACESCCT_A * x + _ACESCCT_B,
                       (torch.log2(x.clamp(min=1e-10)) + 9.72) / 17.52)


def acescct_to_linear(x):
    from .colorspace import _ACESCCT_A, _ACESCCT_B, _ACESCCT_CUT_LOG
    return torch.where(x <= _ACESCCT_CUT_LOG, (x - _ACESCCT_B) / _ACESCCT_A,
                       torch.pow(2.0, x * 17.52 - 9.72))


def convert_colorspace(img, source, target):
    from .colorspace import _SPACES, _ACES_TO_SRGB
    if source == target:
        return img.clone()
    src, tgt = _SPACES[source], _SPACES[target]
    linear = img
    if src["gamma"] == "srgb":
        linear = srgb_decode(linear)
    elif src["gamma"] == "acescct":
        linear = apply_matrix(acescct_to_linear(linear), _ACES_TO_SRGB)
    if "to_srgb_linear" in src and src["gamma"] != "acescct":
        linear = apply_matrix(linear, src["to_srgb_linear"])
    if "from_srgb_linear" in tgt:
        linear = apply_matrix(linear, tgt["from_srgb_linear"])
    if tgt["gamma"] == "srgb":
        return srgb_encode(linear.clamp(0.0, 1.0))
    if tgt["gamma"] == "acescct":
        return linear_to_acescct(linear)
    return linear


def aces_narkowicz(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)


def aces_hill(x):
    a = x * (x + 0.0245786) - 0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    return (a / b).clamp(0.0, 1.0)


def reinhard(x):
    return (x / (1.0 + x)).clamp(0.0, 1.0)


def reinhard_extended(x, white_point=4.0):
    return ((x * (1.0 + x / (white_point * white_point))) / (1.0 + x)).clamp(0.0, 1.0)


def agx_base(x):
    x = x.clamp(min=1e-10)
    xl = (torch.log2(x) / 16.0 + 0.5).clamp(0.0, 1.0)
    x2 = xl * xl
    x4 = x2 * x2
    y = (15.5 * x4 * x2 - 40.14 * x4 * xl + 31.96 * x4 - 6.868 * x2 * xl
         + 0.4298 * x2 + 0.1191 * xl - 0.00232)
    return y.clamp(0.0, 1.0)


def uncharted2(x):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

    def tm(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
    return (tm(x) / tm(11.2)).clamp(0.0, 1.0)


TONEMAP_CURVES = {
    "ACES Filmic (Narkowicz)": aces_narkowicz,
    "ACES Fitted (Hill)": aces_hill,
    "AgX (Blender)": agx_base,
    "Reinhard": reinhard,
    "Reinhard Extended": reinhard_extended,
    "Filmic (Uncharted 2)": uncharted2,
}


# ---------------------------------------------------------------------------
# utils/lut.py
# ---------------------------------------------------------------------------

def lut_to_device(lut_3d, dev):
    """(S, S, S, 3) numpy LUT -> flat (S^3, 3) float32 tensor on dev."""
    return torch.as_tensor(np.ascontiguousarray(lut_3d, dtype=np.float32), device=dev).reshape(-1, 3)


def apply_lut_trilinear(image, lut_flat, lut_size):
    """Same 8-corner trilinear as lut.apply_lut_trilinear; lut_flat from lut_to_device."""
    s = int(lut_size)
    img = image.clamp(0.0, 1.0)
    coords = img * float(s - 1)
    f = torch.floor(coords).to(torch.int64).clamp(0, s - 2)
    frac = coords - f.to(torch.float32)
    r0, g0, b0 = f[..., 0], f[..., 1], f[..., 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]

    def at(r, g, b):
        return lut_flat[(r * s + g) * s + b]

    c00 = at(r0, g0, b0) * (1.0 - fr) + at(r1, g0, b0) * fr
    c10 = at(r0, g1, b0) * (1.0 - fr) + at(r1, g1, b0) * fr
    c01 = at(r0, g0, b1) * (1.0 - fr) + at(r1, g0, b1) * fr
    c11 = at(r0, g1, b1) * (1.0 - fr) + at(r1, g1, b1) * fr
    c0 = c00 * (1.0 - fg) + c10 * fg
    c1 = c01 * (1.0 - fg) + c11 * fg
    return (c0 * (1.0 - fb) + c1 * fb).clamp(0.0, 1.0)
