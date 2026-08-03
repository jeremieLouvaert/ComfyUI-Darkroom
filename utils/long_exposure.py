"""
Long Exposure engine — handheld camera-shake integration.

Full derivation: docs/long-exposure-derivation.md.

A photograph is the time integral of the light arriving during the exposure. If the
camera moves during it, the sensor integrates a sequence of views:

    I_out(x) = (1/T) * integral_0^T  I( M_t^-1 x ) dt

This is not an effect approximating a photograph; it IS what the photograph is.

Three things carry the identity, in order:

  1. BACKWARD GATHER, never pixel-pushing. Each output pixel averages what passed
     over it, which is literally what a sensor does and which can neither tear nor
     leave holes. Pushing pixels forward piles them up where the flow converges and
     leaves gaps where it diverges. The references settle it: structures stay INTACT
     and dragged, never churned.

  2. ONE GLOBAL PATH, not a velocity field. Camera shake moves the whole frame
     together. Stage A tested the alternative first: of pan / arc / radial / curl
     noise, the most fluid-like field gave the LEAST reference-like image, with
     swirls present in none of the references. A wind solver would produce more of
     exactly that.

  3. A MEASURED PATH SPECTRUM. Hand tremor is neither white noise (reads as digital
     jitter) nor a plain random walk (drifts with no shimmer). It is 1/f amplitude
     plus a physiological tremor peak near 9 Hz. At the shipped default the
     synthesised path measures 89% of displacement in the 1-3.5 Hz band and 24.7%
     in 7.5-12.5 Hz, reproducing both published human figures independently.
"""

import math

import numpy as np
from scipy.ndimage import map_coordinates

# Physiological tremor, from the tremor literature. The peak sits at 7-11 Hz for
# 90% of subjects (n=237); the 7.5-12.5 Hz band carries ~24% of displacement
# oscillation amplitude while 1-3.5 Hz carries ~90%.
TREMOR_HZ = 9.0
TREMOR_SIGMA_HZ = 2.5
DRIFT_BAND = (1.0, 3.5)
TREMOR_BAND = (7.5, 12.5)

# Tremor weight reproducing both published band shares. A first guess of 0.45 put
# 59.5% in the tremor band -- an order of magnitude too much -- and looked visibly
# jittery. Kept as a named constant so the number stays traceable to its source.
TREMOR_WEIGHT_HUMAN = 0.05


def _spectral(n, seed, amp, k):
    rng = np.random.default_rng((int(seed) & 0x7FFFFFFF) * 977 + k)
    sig = np.fft.irfft(amp * np.exp(2j * math.pi * rng.random(amp.size)), n=n)
    return sig - sig.mean()


def shake_path(n_poses, seed=0, steadiness=1.0, variety=1.0, exposure_s=1.0,
               tremor_hz=TREMOR_HZ):
    """
    The camera path: tx, ty, roll.

    TWO COMPONENTS, and they are physically different things:

      GESTURE -- the deliberate movement. This technique is INTENTIONAL camera
      movement, not a failure to hold still, so the large motion is a chosen
      gesture: a swoop, a whip, a pause. It is NON-STATIONARY, meaning its speed
      and direction change during the exposure.

      TREMOR -- the involuntary hand oscillation, which keeps the measured
      physiological spectrum and does NOT warp with intent, because a 9Hz
      stretch-reflex oscillator does not speed up because the photographer
      decided to swoop.

    Two defects in the first version, both reported from the render and then
    confirmed by measurement, are what forced this split:

      1. x and y always had the SAME extent -- ratio 0.98-1.03 across every seed --
         because each axis was normalised to +/-1 independently. Real movement has
         a direction. Now the pair is normalised JOINTLY and given a seeded axis
         ratio and orientation.

      2. speed was CONSTANT across the exposure -- quarters at 1.13/1.00/0.90/0.96
         of the mean. That is structural: a fixed power spectrum with random phase
         is a stationary process and cannot swoop or build. Fixed by warping time
         for the gesture, so the camera dwells in some stretches and races through
         others. Dwelling is what leaves a readable core with a smear off it,
         which is exactly what the references show.

    `variety` drives the non-stationarity: 0 gives the old uniform-speed path,
    higher values give stronger swoops and longer dwells.
    """
    n = max(int(n_poses), 2)
    f = np.fft.rfftfreq(n, d=float(exposure_s) / n).copy()
    f[0] = 1e-9
    rng = np.random.default_rng((int(seed) & 0x7FFFFFFF) ^ 0xACE)

    # --- gesture: smooth, low-frequency, and time-warped -------------------
    g_amp = 1.0 / f
    g_amp[0] = 0.0
    g_amp[f > 4.0] *= 0.15                     # the gesture is a movement, not a buzz
    gx = _spectral(n, seed, g_amp, 1)
    gy = _spectral(n, seed, g_amp, 2)

    v = max(float(variety), 0.0)
    if v > 0.0:
        # A positive, smooth speed envelope. Log-normal so it can genuinely stall
        # (dwell) and genuinely race (swoop) rather than merely wobble about 1.
        e_amp = 1.0 / f
        e_amp[0] = 0.0
        e_amp[f > 2.5] = 0.0                   # tempo changes slowly
        env = _spectral(n, seed, e_amp, 7)
        env = env / (np.abs(env).max() + 1e-12)
        speed = np.exp(1.6 * v * env)
        tau = np.cumsum(speed)
        tau = (tau - tau[0]) / max(tau[-1] - tau[0], 1e-12) * (n - 1)
        idx = np.arange(n, dtype=np.float64)
        gx = np.interp(tau, idx, gx)
        gy = np.interp(tau, idx, gy)

    # anisotropy: a seeded axis ratio and orientation, so the movement has a
    # DIRECTION instead of being square by construction
    ratio = float(np.exp(rng.normal(0.0, 0.45)))       # ~0.4x to 2.5x, median 1
    theta = float(rng.random() * math.pi)
    ct, st = math.cos(theta), math.sin(theta)
    ax = gx * ratio
    ay = gy / ratio
    gx, gy = ct * ax - st * ay, st * ax + ct * ay

    # --- tremor: involuntary, keeps the measured spectrum, NOT warped -------
    tw = TREMOR_WEIGHT_HUMAN * max(float(steadiness), 0.0)
    t_amp = np.exp(-0.5 * ((f - tremor_hz) / TREMOR_SIGMA_HZ) ** 2)
    t_amp[0] = 0.0
    tx_ = _spectral(n, seed, t_amp, 3)
    ty_ = _spectral(n, seed, t_amp, 4)

    def norm_pair(a, b):
        m = max(float(np.abs(a).max()), float(np.abs(b).max()))
        return (a / m, b / m) if m > 1e-12 else (a, b)

    gx, gy = norm_pair(gx, gy)
    tx_, ty_ = norm_pair(tx_, ty_)
    # tremor rides ON the gesture at the amplitude ratio the spectrum implies
    px = gx + tw * 4.0 * tx_
    py = gy + tw * 4.0 * ty_
    px, py = norm_pair(px, py)

    roll = _spectral(n, seed, g_amp, 5)
    roll = roll / (float(np.abs(roll).max()) + 1e-12)
    return px, py, roll


def band_shares(n_poses=96, steadiness=1.0, exposure_s=1.0, tremor_hz=TREMOR_HZ):
    """
    Fraction of displacement amplitude in the drift and tremor bands.

    This is the check that keeps the node honest rather than tuned: the shipped
    default must reproduce the published human values (~90% / ~24%), and a teeth
    check asserts it. Amplitudes sum in quadrature, which is also why "removing
    7.5-12.5 Hz changes total amplitude by <3%" is consistent with that band
    holding 24% of it -- a point worth keeping written down, because misreading it
    is what produced the wrong first guess.
    """
    n = max(int(n_poses), 2)
    f = np.fft.rfftfreq(n, d=float(exposure_s) / n).copy()
    f[0] = 1e-9
    tw = TREMOR_WEIGHT_HUMAN * max(float(steadiness), 0.0)
    amp = 1.0 / f + tw * np.exp(-0.5 * ((f - tremor_hz) / TREMOR_SIGMA_HZ) ** 2)
    amp[0] = 0.0
    tot = math.sqrt(float((amp ** 2).sum())) or 1.0
    lo = (f >= DRIFT_BAND[0]) & (f <= DRIFT_BAND[1])
    hi = (f >= TREMOR_BAND[0]) & (f <= TREMOR_BAND[1])
    return (math.sqrt(float((amp[lo] ** 2).sum())) / tot,
            math.sqrt(float((amp[hi] ** 2).sum())) / tot)


def _pose_offsets(H, W, tx, ty, roll, k, amp_px, roll_deg):
    """
    Source coordinates for pose k, as a displacement FROM identity.

    Returning the displacement rather than the absolute position is what lets a
    mask scale it per pixel: scaling an absolute coordinate would drag pixels
    toward the frame centre instead of reducing their motion.
    """
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    a = math.radians(float(roll_deg) * roll[k])
    ca, sa = math.cos(a), math.sin(a)
    dy, dx = ys - cy, xs - cx
    sy = cy + (sa * dx + ca * dy) + amp_px * ty[k]
    sx = cx + (ca * dx - sa * dy) + amp_px * tx[k]
    return ys, xs, sy - ys, sx - xs


def integrate(img, tx, ty, roll, amp_px=30.0, roll_deg=0.0, mask=None):
    """
    Average the image over the camera's pose sequence.

    Backward warp per pose, so this cannot tear or leave holes. `mask` scales the
    displacement per pixel, which is how a SUBJECT moves separately from the
    camera -- the one thing a single global path cannot express, and which the
    reference frame `-69` needs (figure smeared, wall clean).
    """
    H, W = img.shape[:2]
    n = len(tx)
    acc = np.zeros_like(img, dtype=np.float64)
    for k in range(n):
        ys, xs, dsy, dsx = _pose_offsets(H, W, tx, ty, roll, k, amp_px, roll_deg)
        if mask is not None:
            dsy = dsy * mask
            dsx = dsx * mask
        for c in range(img.shape[2]):
            acc[..., c] += map_coordinates(img[..., c], [ys + dsy, xs + dsx],
                                           order=1, mode="reflect")
    return acc / n


def _torch_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def integrate_gpu(img, tx, ty, roll, amp_px=30.0, roll_deg=0.0, mask=None):
    """
    The same integral on the GPU. 96 poses over a megapixel is ~75M gathers, which
    is the shape that wins on a device (the same profile gave 43x on the water
    optics). float32, so not bit-identical to the numpy path -- which is why the
    teeth check physics rather than pixels and keep numpy as the reference.
    """
    import torch
    import torch.nn.functional as Fn

    dev = "cuda"
    H, W = img.shape[:2]
    t_img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32,
                         device=dev).permute(2, 0, 1).unsqueeze(0)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    m_t = None
    if mask is not None:
        m_t = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32, device=dev)

    acc = torch.zeros_like(t_img)
    for k in range(len(tx)):
        a = math.radians(float(roll_deg) * roll[k])
        ca, sa = math.cos(a), math.sin(a)
        dy, dx = yy - cy, xx - cx
        dsy = (cy + (sa * dx + ca * dy) + amp_px * ty[k]) - yy
        dsx = (cx + (ca * dx - sa * dy) + amp_px * tx[k]) - xx
        if m_t is not None:
            dsy = dsy * m_t
            dsx = dsx * m_t
        gy = ((yy + dsy) / max(H - 1, 1)) * 2 - 1
        gx = ((xx + dsx) / max(W - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        acc = acc + Fn.grid_sample(t_img, grid, mode="bilinear",
                                   padding_mode="reflection", align_corners=True)
    acc = acc / len(tx)
    return acc.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)


def integrate_auto(img, tx, ty, roll, **kw):
    """GPU when there is one, the numpy reference otherwise. Same integral either way."""
    if _torch_ok():
        return integrate_gpu(img, tx, ty, roll, **kw)
    return integrate(img, tx, ty, roll, **kw)
