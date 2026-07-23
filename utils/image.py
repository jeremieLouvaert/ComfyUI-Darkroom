"""
Image processing utilities for ComfyUI-Darkroom.
Handles tensor↔numpy conversion and blur operations.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


def tensor_to_numpy_batch(tensor):
    """
    Convert ComfyUI IMAGE tensor (B, H, W, C) to list of numpy (H, W, C) float32 arrays.
    """
    arrays = []
    for i in range(tensor.shape[0]):
        arr = tensor[i].cpu().numpy().astype(np.float32)
        arrays.append(arr)
    return arrays


def numpy_batch_to_tensor(arrays):
    """
    Convert list of numpy (H, W, C) float32 arrays to ComfyUI IMAGE tensor (B, H, W, C).
    """
    tensors = [torch.from_numpy(arr).unsqueeze(0) for arr in arrays]
    return torch.cat(tensors, dim=0)


def process_batch(tensor, func, **kwargs):
    """
    Apply a per-image processing function to each image in a batch tensor.
    func signature: func(img_numpy_hwc, **kwargs) -> img_numpy_hwc
    Returns: ComfyUI IMAGE tensor (B, H, W, C)
    """
    arrays = tensor_to_numpy_batch(tensor)
    processed = [func(arr, **kwargs) for arr in arrays]
    return numpy_batch_to_tensor(processed)


def disk_kernel(radius):
    """
    Generate a normalized disk (circle) blur kernel.
    Returns (2*radius+1, 2*radius+1) float32 array.
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= (radius * radius)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel


def _fresnel_unpolarized(theta, n):
    """Unpolarized Fresnel reflectance at a base(n) -> air(1) interface for
    internal incidence angle `theta` (radians). R = 1 at/beyond the critical
    angle theta_c = asin(1/n) (total internal reflection).
    """
    st = n * np.sin(theta)          # Snell: sin(theta_transmitted)
    R = np.ones_like(theta)
    sub = st < 1.0                  # below critical angle -> partial Fresnel
    tt = np.arcsin(np.clip(st[sub], 0.0, 1.0))
    ti = theta[sub]
    rs = (n * np.cos(ti) - np.cos(tt)) / (n * np.cos(ti) + np.cos(tt))
    rp = (n * np.cos(tt) - np.cos(ti)) / (n * np.cos(tt) + np.cos(ti))
    R[sub] = 0.5 * (rs ** 2 + rp ** 2)
    return R


def halation_psf(ring_radius_px, tail_length_px, balance, tir_on=True, max_ksize=None):
    """Physically-derived annular halation PSF (docs/halation-derivation.md,
    PRODUCTION MODEL secs 1-3).

    Ring term: total-internal-reflection annulus at the film-base rear
    surface, mapped from incidence angle theta to lateral offset
    r = 2d*tan(theta):

        P_TIR(r) = R(theta(r)) / (1 + t^2)^2,   t = r / (2d)

    with 2d = ring_radius_px * sqrt(n^2 - 1) and n = 1.5 (fixed internal
    refractive index of the film base, LOAD-BEARING CALL 1: ring_radius_px
    IS r_c, d itself is never instantiated). R(theta) is the unpolarized
    Fresnel reflectance (see `_fresnel_unpolarized`), which jumps to 1 at
    the critical angle -> the annulus.

    Tail term: diffuse multiple-scattering spread (James Ch. 20 empirical
    class), T(r) = exp(-r/L) / (r + 0.5), L = tail_length_px.

    Both terms are unit-normalized independently, mixed by `balance`
    (1.0 = pure ring, 0.0 = pure tail), and the mixed kernel is
    renormalized to unit sum.

    tir_on=False is the negative-control variant (used by the teeth): the
    TIR jump is removed by clamping R to its sub-critical trend
    (R <= 0.12), which kills the ring while keeping the rest of the
    geometry -- if a test still finds a ring here, the test has no teeth.

    max_ksize optionally clamps the kernel half-size (e.g. to
    min(h, w) - 1 for small images with large tail_length_px).
    """
    ring_radius_px = max(float(ring_radius_px), 1e-6)
    tail_length_px = max(float(tail_length_px), 1e-6)
    ksize = int(max(4 * ring_radius_px, 6 * tail_length_px, 16))
    if max_ksize is not None:
        ksize = max(min(ksize, int(max_ksize)), 1)

    y, x = np.mgrid[-ksize:ksize + 1, -ksize:ksize + 1].astype(np.float64)
    r = np.hypot(x, y)

    n = 1.5
    two_d = ring_radius_px * np.sqrt(n * n - 1.0)
    t = r / two_d
    theta = np.arctan(t)
    R = _fresnel_unpolarized(theta, n)
    if not tir_on:
        R = np.clip(R, 0.0, 0.12)
    ring = R / (1.0 + t * t) ** 2
    ring_sum = ring.sum()
    if ring_sum > 0:
        ring = ring / ring_sum

    tail = np.exp(-r / tail_length_px) / (r + 0.5)
    tail_sum = tail.sum()
    if tail_sum > 0:
        tail = tail / tail_sum

    k = balance * ring + (1.0 - balance) * tail
    k_sum = k.sum()
    if k_sum > 0:
        k = k / k_sum
    return k.astype(np.float32)


def apply_gaussian_blur(img, radius, sigma=None):
    """
    Apply Gaussian blur per-channel.
    img: (H, W, C) float32. radius: approximate kernel radius. sigma: defaults to radius/3.
    """
    if radius <= 0:
        return img
    if sigma is None:
        sigma = max(radius / 3.0, 0.5)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return result.astype(np.float32)


def apply_disk_blur(img, radius):
    """
    Apply disk (uniform circle) blur per-channel via FFT convolution.
    Physically accurate for halation. Slower than Gaussian.
    """
    if radius <= 0:
        return img
    kernel = disk_kernel(radius)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = fftconvolve(img[..., c], kernel, mode='same')
    return np.clip(result, 0.0, 1.0).astype(np.float32)
