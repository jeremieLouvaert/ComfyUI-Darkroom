"""
Newson stochastic film-grain engine for ComfyUI-Darkroom (Film Grain Pro).

Filtered inhomogeneous Boolean model from Newson et al., "Realistic Film Grain
Rendering", IPOL 2017 (https://doi.org/10.5201/ipol.2017.192). Clean-room from
the paper math; pixel-wise algorithm only; s = 1 (output grid = input image).

See docs/film-grain-pro-derivation.md for the full derivation and unit model.

All work is vectorised torch and stays on device. Z is generated procedurally
from a counter-based integer hash of cell coordinates, so the grain field is one
fixed realisation with zero storage.
"""

import math
import torch


EPS = 0.1 / 255.0          # paper's ε (≈ 0.1 gray-levels), keeps ũ < 1
_Z999 = 3.090232           # 0.999 normal quantile, for the log-normal rm
_HALF = 0.5


# ---------------------------------------------------------------------------
# Counter-based integer hash -> uniform [0, 1)
# ---------------------------------------------------------------------------

def _hash_u32(x):
    """
    Integer avalanche hash on an int64 tensor, returns a 32-bit-range int64.
    Wang/Murmur-style; deterministic and device-stable.
    """
    M = 0xFFFFFFFF
    x = x & M
    x = (x ^ 61) ^ (x >> 16)
    x = (x + (x << 3)) & M
    x = x ^ (x >> 4)
    x = (x * 0x27D4EB2D) & M
    x = x ^ (x >> 15)
    return x & M


def _hash_uniform(cx, cy, stream, seed):
    """
    Deterministic uniform in [0, 1) keyed by integer cell coords (cx, cy), a
    stream id, and the global seed. cx, cy are int64 tensors; stream may be a
    python int or an int64 tensor (broadcasts against cx/cy for vectorised slots).
    """
    h = (cx * 73856093) ^ (cy * 19349663) ^ (stream * 83492791) ^ (seed * 2654435761)
    h = _hash_u32(h)
    # mix once more so adjacent streams decorrelate well
    h = _hash_u32(h + stream * 40503)
    return h.to(torch.float32) / 4294967296.0


# ---------------------------------------------------------------------------
# Inverse-CDF Poisson (small mean), vectorised over (H, W)
# ---------------------------------------------------------------------------

def _poisson_invcdf(mean, u, qmax):
    """
    Draw Poisson counts by inverse CDF from a uniform u in [0,1).
    mean, u: (H, W) tensors. Returns int64 (H, W) counts in [0, qmax].
    """
    # cumulative P(X <= k) accumulated up to qmax; count = #{k : u > CDF_k}
    cdf = torch.exp(-mean)              # P(X = 0)
    term = cdf.clone()
    count = (u > cdf).to(torch.int64)
    for k in range(1, qmax + 1):
        term = term * mean / k          # P(X = k)
        cdf = cdf + term
        count = count + (u > cdf).to(torch.int64)
    return count


# ---------------------------------------------------------------------------
# Single-channel render
# ---------------------------------------------------------------------------

def _render_channel(u, mu_r, sigma_r, n_samples, filter_sigma, seed, device):
    """
    Render the filtered Boolean model for one (H, W) field u in [0, 1].
    Returns v (H, W): the tone-preserving grainy render (E[v] = ũ).

    Cell grid = pixel grid (delta = 1). The grain field Z is hashed ONCE over the
    pixel grid; the per-sample Monte-Carlo loop then does only cheap float ops,
    reaching neighbour cells by tensor rolls (the per-sample MC offset is a
    uniform integer shift across all pixels, so a neighbour lookup is a roll).
    """
    H, W = u.shape
    u = u.to(device=device, dtype=torch.float32)
    seed_i = int(seed) & 0x7FFFFFFF

    # --- intensity map lambda(i,j); cell_area = 1 so the per-cell mean = lam ---
    u_tilde = (u / (1.0 + EPS)).clamp(0.0, 1.0 - 1e-6)
    inv_area = 1.0 / (math.pi * (mu_r * mu_r + sigma_r * sigma_r))
    lam = inv_area * torch.log(1.0 / (1.0 - u_tilde))   # (H, W)

    # --- max radius rm + log-normal params ---
    if sigma_r <= 0.0:
        rm = mu_r
        s_log = 0.0
        m_log = math.log(mu_r)
    else:
        s_log = math.sqrt(math.log(1.0 + (sigma_r * sigma_r) / (mu_r * mu_r)))
        m_log = math.log(mu_r * mu_r / math.sqrt(mu_r * mu_r + sigma_r * sigma_r))
        rm = min(math.exp(m_log + s_log * _Z999), 5.0 * mu_r)
    rm2 = rm * rm
    cell_rad = max(1, math.ceil(rm))

    # --- adaptive grain cap from the brightest cell's mean count (4-sigma) ---
    m_max = float(lam.max().item())
    qmax = int(min(24, max(2, math.ceil(m_max + 4.0 * math.sqrt(m_max) + 2.0))))

    # --- precompute the fixed grain field Z over the pixel grid (hashed once) ---
    g_idx = torch.arange(qmax, device=device, dtype=torch.int64).view(1, 1, qmax)
    cols = torch.arange(W, device=device, dtype=torch.int64).view(1, W).expand(H, W)
    rows = torch.arange(H, device=device, dtype=torch.int64).view(H, 1).expand(H, W)

    u_cnt = _hash_uniform(cols, rows, 0, seed_i)              # (H, W)
    q_grid = _poisson_invcdf(lam, u_cnt, qmax)               # (H, W) int64

    cxe, cye = cols.unsqueeze(-1), rows.unsqueeze(-1)
    ux_grid = _hash_uniform(cxe, cye, 1 + 4 * g_idx, seed_i)  # (H, W, qmax) sub-cell x
    uy_grid = _hash_uniform(cxe, cye, 2 + 4 * g_idx, seed_i)  # sub-cell y
    if sigma_r > 0.0:
        ur = _hash_uniform(cxe, cye, 3 + 4 * g_idx, seed_i)
        r_grid = torch.exp(m_log + s_log * _norm_invcdf(ur)).clamp(max=rm)
        r2_grid = r_grid * r_grid

    # --- Monte-Carlo offsets, shared across all pixels ---
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_i)
    xi = (torch.randn(n_samples, 2, generator=gen, device=device)
          * filter_sigma).cpu().tolist()                     # to python, no per-op sync

    v = torch.zeros((H, W), device=device, dtype=torch.float32)

    for xix, xiy in xi:
        sx = math.floor(_HALF + xix)
        sy = math.floor(_HALF + xiy)
        fracx = _HALF + xix - sx          # eval point sub-cell position (uniform)
        fracy = _HALF + xiy - sy

        covered = torch.zeros((H, W), device=device, dtype=torch.bool)
        for dcy in range(-cell_rad, cell_rad + 1):
            ry = sy + dcy
            offy = fracy - dcy
            for dcx in range(-cell_rad, cell_rad + 1):
                rx = sx + dcx
                uxr = torch.roll(ux_grid, shifts=(-ry, -rx), dims=(0, 1))
                uyr = torch.roll(uy_grid, shifts=(-ry, -rx), dims=(0, 1))
                qr = torch.roll(q_grid, shifts=(-ry, -rx), dims=(0, 1))
                dx = (fracx - dcx) - uxr
                dy = offy - uyr
                dist2 = dx * dx + dy * dy                     # (H, W, qmax)
                if sigma_r <= 0.0:
                    hit = dist2 < rm2
                else:
                    hit = dist2 < torch.roll(r2_grid, shifts=(-ry, -rx), dims=(0, 1))
                present = g_idx < qr.unsqueeze(-1)
                covered |= (present & hit).any(dim=-1)
        v += covered.to(torch.float32)

    return v / float(n_samples)


def _norm_invcdf(u):
    """Standard-normal inverse CDF (Acklam-style rational approx), torch (H,W)."""
    # coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    u = u.clamp(1e-6, 1.0 - 1e-6)
    plow = 0.02425
    phigh = 1.0 - plow
    out = torch.zeros_like(u)

    lo = u < plow
    hi = u > phigh
    mid = (~lo) & (~hi)

    ql = torch.sqrt(-2.0 * torch.log(u.clamp(min=1e-12)))
    num_l = ((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]
    den_l = (((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1.0
    out = torch.where(lo, num_l / den_l, out)

    qh = torch.sqrt(-2.0 * torch.log((1.0 - u).clamp(min=1e-12)))
    num_h = ((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]
    den_h = (((d[0] * qh + d[1]) * qh + d[2]) * qh + d[3]) * qh + 1.0
    out = torch.where(hi, -(num_h / den_h), out)

    qm = u - 0.5
    rm_ = qm * qm
    num_m = (((((a[0] * rm_ + a[1]) * rm_ + a[2]) * rm_ + a[3]) * rm_ + a[4]) * rm_ + a[5]) * qm
    den_m = ((((b[0] * rm_ + b[1]) * rm_ + b[2]) * rm_ + b[3]) * rm_ + b[4]) * rm_ + 1.0
    out = torch.where(mid, num_m / den_m, out)
    return out


# ---------------------------------------------------------------------------
# Public entry: render grain on an (H, W, C) image tensor
# ---------------------------------------------------------------------------

def render_film_grain(img, grain_size, radius_variation, strength,
                      color_grain, n_samples, filter_sigma, seed,
                      device=None):
    """
    Apply Newson film grain to an (H, W, C) float32 tensor in [0, 1].

    grain_size:       grain radius in px at a 1024px long-edge reference (ρ_ref).
    radius_variation: σ_r / μ_r (0 = constant radius).
    strength:         blend amount (0 = original, 1 = full grain deviation).
    color_grain:      0 = shared luminance grain, 1 = independent per-channel.
    n_samples:        Monte-Carlo samples N.
    filter_sigma:     Gaussian filter σ in output px.
    seed:             PRNG seed.
    Returns (H, W, C) float32 tensor in [0, 1].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    img = img.to(device=device, dtype=torch.float32)
    H, W, C = img.shape

    L = max(H, W)
    # input-px mean radius; floored at 0.5 (delta=1 tail-accuracy + visibility floor)
    mu_r = max(0.5, float(grain_size) * (L / 1024.0))
    sigma_r = max(0.0, float(radius_variation)) * mu_r

    if strength <= 0.0:
        return img.clamp(0.0, 1.0)

    # luminance-only fast path
    if color_grain <= 0.01 or C < 3:
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2] \
            if C >= 3 else img[..., 0]
        v = _render_channel(luma, mu_r, sigma_r, n_samples, filter_sigma, seed, device)
        dev = (v - luma).unsqueeze(-1)             # (H, W, 1) zero-mean grain
        out = img + strength * dev
        return out.clamp(0.0, 1.0)

    # per-channel render with green as the shared (luminance proxy) baseline
    devs = []
    for c in range(3):
        u_c = img[..., c]
        v_c = _render_channel(u_c, mu_r, sigma_r, n_samples, filter_sigma,
                              int(seed) + c * 101, device)
        devs.append(v_c - u_c)
    dev_shared = devs[1]                            # green channel grain
    out = img.clone()
    for c in range(3):
        dev_c = dev_shared * (1.0 - color_grain) + devs[c] * color_grain
        out[..., c] = img[..., c] + strength * dev_c
    if C > 3:
        out[..., 3:] = img[..., 3:]
    return out.clamp(0.0, 1.0)
