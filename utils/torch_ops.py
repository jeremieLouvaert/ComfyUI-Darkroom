"""
GPU-accelerated torch equivalents of numpy color math for ComfyUI-Darkroom.
All functions operate on torch tensors and stay on device (no CPU roundtrip).

Tensor format convention:
    ComfyUI IMAGE tensor: (B, H, W, C) float32
    Internal per-image:   (H, W, C) or (H, W) float32
"""

import torch
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# sRGB <-> Linear
# ---------------------------------------------------------------------------

def srgb_to_linear(img):
    """Remove sRGB gamma. Input/output: (*, 3) or (H, W, C) float32 tensor, 0-1."""
    img = img.clamp(0.0, 1.0)
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(img):
    """Apply sRGB gamma. Input/output: (*, 3) or (H, W, C) float32 tensor, 0-1."""
    img = img.clamp(0.0, 1.0)
    return torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * img.pow(1.0 / 2.4) - 0.055
    )


# ---------------------------------------------------------------------------
# Luminance
# ---------------------------------------------------------------------------

def luminance_rec709(img):
    """Rec.709 luminance from RGB. img: (H, W, 3). Returns: (H, W)."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


# ---------------------------------------------------------------------------
# RGB <-> HSL (fully vectorized, no loops, no masks-with-indexing)
# ---------------------------------------------------------------------------

def rgb_to_hsl(img):
    """
    Convert RGB image to HSL. Fully vectorized torch.
    img: (H, W, 3) float32 tensor, 0-1
    Returns: (h, s, l) each (H, W) float32 tensor.
             h in [0, 360), s and l in [0, 1].
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax, _ = img.max(dim=-1)
    cmin, _ = img.min(dim=-1)
    delta = cmax - cmin

    # Luminance
    l = (cmax + cmin) * 0.5

    # Saturation
    s = torch.zeros_like(l)
    mask = delta > 1e-7
    low = mask & (l <= 0.5)
    high = mask & (l > 0.5)
    s = torch.where(low, delta / (cmax + cmin + 1e-10), s)
    s = torch.where(high, delta / (2.0 - cmax - cmin + 1e-10), s)

    # Hue
    h = torch.zeros_like(l)
    safe_delta = delta + (~mask).float() * 1e-10  # avoid div by zero

    is_r = mask & (cmax == r)
    is_g = mask & (cmax == g) & ~is_r
    is_b = mask & ~is_r & ~is_g

    h = torch.where(is_r, 60.0 * (((g - b) / safe_delta) % 6.0), h)
    h = torch.where(is_g, 60.0 * (((b - r) / safe_delta) + 2.0), h)
    h = torch.where(is_b, 60.0 * (((r - g) / safe_delta) + 4.0), h)
    h = h % 360.0

    return h, s, l


def hsl_to_rgb(h, s, l):
    """
    Convert HSL back to RGB. Fully vectorized torch.
    h: (H, W) [0, 360), s: (H, W) [0, 1], l: (H, W) [0, 1]
    Returns: (H, W, 3) float32 tensor, 0-1.
    """
    c = (1.0 - (2.0 * l - 1.0).abs()) * s
    h_prime = (h / 60.0) % 6.0
    x = c * (1.0 - (h_prime % 2.0 - 1.0).abs())
    m = l - c * 0.5

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    # Sector 0: [0, 60)
    m0 = (h_prime >= 0) & (h_prime < 1)
    r = torch.where(m0, c, r)
    g = torch.where(m0, x, g)
    # Sector 1: [60, 120)
    m1 = (h_prime >= 1) & (h_prime < 2)
    r = torch.where(m1, x, r)
    g = torch.where(m1, c, g)
    # Sector 2: [120, 180)
    m2 = (h_prime >= 2) & (h_prime < 3)
    g = torch.where(m2, c, g)
    b = torch.where(m2, x, b)
    # Sector 3: [180, 240)
    m3 = (h_prime >= 3) & (h_prime < 4)
    g = torch.where(m3, x, g)
    b = torch.where(m3, c, b)
    # Sector 4: [240, 300)
    m4 = (h_prime >= 4) & (h_prime < 5)
    r = torch.where(m4, x, r)
    b = torch.where(m4, c, b)
    # Sector 5: [300, 360)
    m5 = (h_prime >= 5) & (h_prime < 6)
    r = torch.where(m5, c, r)
    b = torch.where(m5, x, b)

    return torch.stack([r + m, g + m, b + m], dim=-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Gaussian blur (separable, GPU)
# ---------------------------------------------------------------------------

def gaussian_blur_2d(tensor_2d, sigma):
    """
    Apply Gaussian blur to a 2D (H, W) tensor on device.
    Uses separable 1D convolutions for efficiency.
    """
    if sigma < 0.5:
        return tensor_2d

    # Kernel radius: 3 sigma, at least 1
    radius = max(int(math.ceil(sigma * 3.0)), 1)
    size = 2 * radius + 1

    # Build 1D Gaussian kernel
    x = torch.arange(size, dtype=tensor_2d.dtype, device=tensor_2d.device) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Reshape for conv: (1, 1, size) for horizontal, (1, 1, size, 1) implicit via transpose
    inp = tensor_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Horizontal pass
    k_h = kernel_1d.view(1, 1, 1, size)
    out = F.pad(inp, (radius, radius, 0, 0), mode='reflect')
    out = F.conv2d(out, k_h)

    # Vertical pass
    k_v = kernel_1d.view(1, 1, size, 1)
    out = F.pad(out, (0, 0, radius, radius), mode='reflect')
    out = F.conv2d(out, k_v)

    return out.squeeze(0).squeeze(0)


def gaussian_blur_hwc(img, sigma):
    """
    Apply Gaussian blur to an (H, W, C) tensor, per-channel.
    """
    if sigma < 0.5:
        return img
    channels = []
    for c in range(img.shape[-1]):
        channels.append(gaussian_blur_2d(img[..., c], sigma))
    return torch.stack(channels, dim=-1)


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def blend(original, processed, strength):
    """Linear interpolation between original and processed."""
    if strength >= 1.0:
        return processed
    if strength <= 0.0:
        return original
    return original * (1.0 - strength) + processed * strength


# ---------------------------------------------------------------------------
# Hue range mask
# ---------------------------------------------------------------------------

def hue_range_mask(hue, center, width=45.0, softness=0.5):
    """
    Raised-cosine feathered hue selection with wraparound at 0/360.
    hue: (H, W) tensor [0, 360). Returns: (H, W) tensor [0, 1].
    """
    diff = (hue - center).abs()
    diff = torch.min(diff, 360.0 - diff)

    effective_width = width * (0.5 + softness * 0.5)
    mask = ((1.0 + torch.cos(math.pi * diff / max(effective_width, 1.0))) * 0.5).clamp(0.0, 1.0)
    mask = torch.where(diff > effective_width, torch.zeros_like(mask), mask)
    return mask


# ---------------------------------------------------------------------------
# Skin mask
# ---------------------------------------------------------------------------

def skin_mask(h, s, l, hue_center, hue_width, sat_min, sat_max, lum_min, lum_max):
    """
    Build a soft skin-tone mask from HSL channels. Torch equivalent.
    All inputs (H, W) tensors. Returns (H, W) tensor.
    """
    diff = (h - hue_center).abs()
    diff = torch.min(diff, 360.0 - diff)
    hue_weight = ((1.0 + torch.cos(math.pi * diff / hue_width)) * 0.5).clamp(0.0, 1.0)
    hue_weight = torch.where(diff > hue_width, torch.zeros_like(hue_weight), hue_weight)

    sat_feather = 0.08
    sat_weight = ((s - sat_min) / (sat_feather + 1e-10)).clamp(0.0, 1.0)
    sat_weight = sat_weight * ((sat_max - s) / (sat_feather + 1e-10)).clamp(0.0, 1.0)

    lum_feather = 0.10
    lum_weight = ((l - lum_min) / (lum_feather + 1e-10)).clamp(0.0, 1.0)
    lum_weight = lum_weight * ((lum_max - l) / (lum_feather + 1e-10)).clamp(0.0, 1.0)

    return hue_weight * sat_weight * lum_weight


# ---------------------------------------------------------------------------
# Saturation range mask (for Color Warper)
# ---------------------------------------------------------------------------

def sat_range_weight(sat, sat_min, sat_max, softness=0.1):
    """
    Smooth saturation range mask. 1.0 inside [sat_min, sat_max], smooth falloff.
    """
    soft = max(softness, 0.01)
    low_weight = ((sat - sat_min + soft) / (2.0 * soft)).clamp(0.0, 1.0)
    high_weight = ((sat_max + soft - sat) / (2.0 * soft)).clamp(0.0, 1.0)
    return low_weight * high_weight


# ---------------------------------------------------------------------------
# Grid sampling (replaces scipy.ndimage.map_coordinates for lens ops)
# ---------------------------------------------------------------------------

def pixel_to_grid_coords(src_y, src_x, h, w):
    """
    Convert source pixel coordinates to F.grid_sample normalized [-1, 1] coords.
    src_y, src_x: (H, W) tensors of source pixel positions.
    Returns: (H, W, 2) grid tensor (x, y order as grid_sample expects).
    With align_corners=True: -1 maps to pixel 0, +1 maps to pixel (size-1).
    """
    gx = 2.0 * src_x / (w - 1) - 1.0
    gy = 2.0 * src_y / (h - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def grid_sample_channel(channel, grid, padding_mode='reflection'):
    """
    Grid-sample a single (H, W) channel using a (H, W, 2) normalized grid.
    Uses bicubic interpolation (matches scipy map_coordinates order=3).
    Returns: (H, W) tensor.
    """
    inp = channel.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    g = grid.unsqueeze(0)                      # (1, H, W, 2)
    out = F.grid_sample(inp, g, mode='bicubic', padding_mode=padding_mode,
                        align_corners=True)
    return out.squeeze(0).squeeze(0)
