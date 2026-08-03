"""
Camera Raw processing utilities for ComfyUI-Darkroom Wave 2.
All functions operate on numpy float32 arrays.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def kelvin_to_rgb(kelvin):
    """
    Convert color temperature (Kelvin) to RGB multipliers.
    Uses Tanner Helland's approximation of the Planckian locus.
    Returns (r, g, b) tuple, normalized so max channel = 1.0.
    """
    temp = np.clip(kelvin, 1000, 40000) / 100.0

    # Red
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = np.clip(r, 0, 255)

    # Green
    if temp <= 66:
        g = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = np.clip(g, 0, 255)

    # Blue
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
        b = np.clip(b, 0, 255)

    rgb = np.array([r, g, b], dtype=np.float32) / 255.0
    rgb /= rgb.max()  # Normalize so brightest channel = 1.0
    return tuple(rgb)


def rgb_to_hsl(img):
    """
    Convert RGB image to HSL. Vectorized numpy implementation.
    img: (H, W, 3) float32, 0-1 range
    Returns: (H_channel, S, L) each (H, W) float32.
             H in [0, 360), S and L in [0, 1].
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Luminance
    l = (cmax + cmin) * 0.5

    # Saturation
    s = np.zeros_like(l)
    mask = delta > 1e-7
    low = mask & (l <= 0.5)
    high = mask & (l > 0.5)
    s[low] = delta[low] / (cmax[low] + cmin[low] + 1e-10)
    s[high] = delta[high] / (2.0 - cmax[high] - cmin[high] + 1e-10)

    # Hue
    h = np.zeros_like(l)
    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g) & ~mask_r
    mask_b = mask & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-10)) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-10)) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-10)) + 4)

    h = h % 360.0

    return h.astype(np.float32), s.astype(np.float32), l.astype(np.float32)


def hsl_to_rgb(h, s, l):
    """
    Convert HSL back to RGB. Vectorized numpy implementation.
    h: (H, W) float32, [0, 360)
    s: (H, W) float32, [0, 1]
    l: (H, W) float32, [0, 1]
    Returns: (H, W, 3) float32, 0-1 range.
    """
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    h_prime = (h / 60.0) % 6.0
    x = c * (1.0 - np.abs(h_prime % 2.0 - 1.0))
    m = l - c * 0.5

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Sector 0: [0, 60)
    mask = (h_prime >= 0) & (h_prime < 1)
    r[mask] = c[mask]
    g[mask] = x[mask]
    # Sector 1: [60, 120)
    mask = (h_prime >= 1) & (h_prime < 2)
    r[mask] = x[mask]
    g[mask] = c[mask]
    # Sector 2: [120, 180)
    mask = (h_prime >= 2) & (h_prime < 3)
    g[mask] = c[mask]
    b[mask] = x[mask]
    # Sector 3: [180, 240)
    mask = (h_prime >= 3) & (h_prime < 4)
    g[mask] = x[mask]
    b[mask] = c[mask]
    # Sector 4: [240, 300)
    mask = (h_prime >= 4) & (h_prime < 5)
    r[mask] = x[mask]
    b[mask] = c[mask]
    # Sector 5: [300, 360)
    mask = (h_prime >= 5) & (h_prime < 6)
    r[mask] = c[mask]
    b[mask] = x[mask]

    result = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def parametric_tone_mask(luminance, center, sigma):
    """
    Gaussian bell-curve mask in luminance space.
    Returns (H, W) float32, 0-1 range. Peaks at 'center', falls off with 'sigma'.
    """
    return np.exp(-0.5 * ((luminance - center) / (sigma + 1e-10)) ** 2).astype(np.float32)


def unsharp_mask_channel(channel, sigma, amount):
    """
    Unsharp mask on a single 2D channel.
    channel: (H, W) float32
    sigma: blur radius
    amount: sharpening strength (0-1+ range)
    Returns: sharpened (H, W) float32
    """
    blurred = gaussian_filter(channel, sigma=sigma)
    detail = channel - blurred
    return (channel + detail * amount).astype(np.float32)


def edge_mask(luminance, threshold):
    """
    Compute edge-aware mask from luminance gradients.
    threshold: 0.0 = sharpen everything, 1.0 = sharpen only strongest edges.
    Returns: (H, W) float32 mask, 0-1 range.
    """
    # Sobel-like gradient magnitude
    gy = np.zeros_like(luminance)
    gx = np.zeros_like(luminance)
    gy[1:, :] = np.abs(luminance[1:, :] - luminance[:-1, :])
    gx[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1])
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize
    grad_max = grad.max()
    if grad_max > 0:
        grad = grad / grad_max

    # Threshold: higher masking value = only sharp edges get through
    t = threshold * 0.8  # Scale so masking=100 still leaves strong edges
    mask = np.clip((grad - t) / (1.0 - t + 1e-6), 0.0, 1.0)
    return mask.astype(np.float32)
