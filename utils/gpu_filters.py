"""
Torch twins of the scipy.ndimage filters Darkroom uses, matching scipy's
boundary handling exactly (mode="reflect" = half-sample symmetric, scipy's
default; kernel radius int(truncate * sigma + 0.5)), so ported nodes stay within
the 0.5/255 golden gate (tools/test_gpu_core.py).

All functions take (..., H, W) float32 tensors on any device and filter the last
two axes (or one axis for the *_1d variants).
"""


import torch
import torch.nn.functional as F


def _reflect_index(n, before, after, device):
    """Indices that extend a length-n axis by `before`/`after` samples using
    scipy's 'reflect' rule (d c b a | a b c d | d c b a), for any pad size."""
    idx = torch.arange(-before, n + after, device=device) % (2 * n)
    return torch.where(idx >= n, 2 * n - 1 - idx, idx)


def _pad_axis(x, axis, before, after):
    axis = axis % x.dim()
    return x.index_select(axis, _reflect_index(x.shape[axis], before, after, x.device))


def gaussian_kernel1d(sigma, truncate=4.0, dtype=torch.float32, device="cpu"):
    """scipy.ndimage._gaussian_kernel1d (order 0), computed in float64 like scipy."""
    radius = int(truncate * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    phi = torch.exp(-0.5 / (float(sigma) ** 2) * x * x)
    return (phi / phi.sum()).to(device=device, dtype=dtype), radius


def _correlate_axis(x, kernel, radius, axis):
    """Correlate along one axis with reflect padding; FFT for long kernels."""
    axis = axis % x.dim()
    n = x.shape[axis]
    xp = _pad_axis(x, axis, radius, radius).movedim(axis, -1)          # (..., n + 2r)
    lead = xp.shape[:-1]
    flat = xp.reshape(-1, 1, xp.shape[-1])
    k = kernel.to(x.dtype)
    if radius <= 48:
        out = F.conv1d(flat, k.view(1, 1, -1))                          # (B, 1, n)
    else:
        L = flat.shape[-1] + k.numel() - 1
        nfft = 1 << (L - 1).bit_length()
        fx = torch.fft.rfft(flat.double(), nfft)
        fk = torch.fft.rfft(k.double().flip(0), nfft)                    # correlation = conv with flipped kernel
        full = torch.fft.irfft(fx * fk, nfft)[..., :L]
        out = full[..., k.numel() - 1:k.numel() - 1 + n].to(x.dtype)
    return out.reshape(*lead, n).movedim(-1, axis)


def gaussian_filter(x, sigma, truncate=4.0):
    """scipy.ndimage.gaussian_filter(x, sigma) over the last two axes, mode='reflect'."""
    if float(sigma) <= 0:
        return x
    k, r = gaussian_kernel1d(sigma, truncate, x.dtype, x.device)
    return _correlate_axis(_correlate_axis(x, k, r, -2), k, r, -1)


def gaussian_filter1d(x, sigma, axis=-1, truncate=4.0):
    if float(sigma) <= 0:
        return x
    k, r = gaussian_kernel1d(sigma, truncate, x.dtype, x.device)
    return _correlate_axis(x, k, r, axis)


def _extreme_axis(x, size, axis, take_max):
    """Sliding min/max of `size` along one axis, scipy window placement
    (offsets -(size//2) .. size-1-size//2), reflect boundary."""
    axis = axis % x.dim()
    before = size // 2
    after = size - 1 - before
    xp = _pad_axis(x, axis, before, after).movedim(axis, -1)
    lead = xp.shape[:-1]
    flat = xp.reshape(-1, 1, xp.shape[-1])
    out = F.max_pool1d(flat if take_max else -flat, size, stride=1)
    out = out if take_max else -out
    return out.reshape(*lead, x.shape[axis]).movedim(-1, axis)


def minimum_filter(x, size):
    """scipy.ndimage.minimum_filter(x, size) over the last two axes, mode='reflect'."""
    return _extreme_axis(_extreme_axis(x, size, -2, False), size, -1, False)


def maximum_filter(x, size):
    return _extreme_axis(_extreme_axis(x, size, -2, True), size, -1, True)


def uniform_filter1d(x, size, axis=-1):
    """scipy.ndimage.uniform_filter1d(x, size, axis), mode='reflect'."""
    axis = axis % x.dim()
    before = size // 2
    after = size - 1 - before
    xp = _pad_axis(x, axis, before, after).movedim(axis, -1)
    lead = xp.shape[:-1]
    flat = xp.reshape(-1, 1, xp.shape[-1])
    out = F.avg_pool1d(flat, size, stride=1)
    return out.reshape(*lead, x.shape[axis]).movedim(-1, axis)


def uniform_filter(x, size):
    """scipy.ndimage.uniform_filter(x, size) over the last two axes."""
    return uniform_filter1d(uniform_filter1d(x, size, -2), size, -1)
