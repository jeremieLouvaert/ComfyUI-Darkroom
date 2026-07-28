"""
Film-damage engine for ComfyUI-Darkroom (Film Damage node).

Chain-polarity model. See docs/film-damage-derivation.md for the full derivation
(signed off 2026-07-27; spike 23/23 checks PASS with 3 negative controls firing;
Jeremie eyeball PASS on _film_damage_spike/ exhibits).

A defect is an optical event at a PLANE, not a mark on the picture. Working in
density, a defect that multiplies transmittance by tau at its own plane gives:

    L_out,c = L_in,c * tau_c ** k        k = +1     (defect on a positive)
                                          k = -gamma_p (defect on a negative)

Parity is carried by the SIGN of the exponent; the paper/inversion contrast
gamma_p carries the magnitude. Everything else here is a statement about tau_c.

Consequences that fall out (and that a fixed PNG overlay cannot reproduce):
  - dust on a negative prints WHITE, dust on a scanned positive reads DARK
  - a base-side scratch prints white while a full-depth emulsion gouge on the
    SAME negative prints black
  - the defect is a constant DENSITY offset, so log(L_out/L_in) is invariant to
    the substrate tone (an sRGB alpha blend cannot hold that)

All compositing is multiplicative in LINEAR light. Sizes are ref-px at a 1024px
long edge (the no-microns rule, Halation/Eberhard precedent).
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter


TAU_FLOOR = 1e-4          # -4 density; keeps tau**(-gamma) finite

# Colour-negative / reversal layer order from the exposing side:
#   blue-sensitive (YELLOW dye) -> green-sensitive (MAGENTA) -> red-sensitive (CYAN)
_THIRD = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Placement: value-noise density field, importance sampled.
# HONEST LABEL: organic clustering is artistic convention. Ivanova et al. make
# no measured-spatial-statistic claim either. Derivation sec 5.
# ---------------------------------------------------------------------------

def _density_grid(rng, gh, gw, clumpiness):
    g = rng.random((gh, gw))
    g = gaussian_filter(g, sigma=max(1.0, 0.05 * max(gh, gw)), mode="wrap")
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    return np.maximum(g, 1e-4) ** clumpiness


def _sample_positions(rng, n, H, W, clumpiness):
    """n (y, x) float positions, clustered by the density field."""
    if n <= 0:
        return np.zeros((0, 2))
    gh = gw = 48
    d = _density_grid(rng, gh, gw, clumpiness)
    p = (d / d.sum()).ravel()
    idx = rng.choice(gh * gw, size=n, p=p)
    cy, cx = idx // gw, idx % gw
    y = (cy + rng.random(n)) * (H / gh)
    x = (cx + rng.random(n)) * (W / gw)
    return np.stack([y, x], axis=1)


# ---------------------------------------------------------------------------
# Morphology (derivation sec 4). Sizes arrive already scaled to input px.
# ---------------------------------------------------------------------------

def _stamp_blob(acc, cy, cx, radius, opacity, rng, harmonics=3, rough=0.22):
    """
    Compact irregular particulate (dust / dirt): ellipse perturbed by a few
    low-order radial harmonics. Accumulated as a probabilistic union, which is
    the transmittance product for overlapping independent occluders.
    """
    H, W = acc.shape
    r_out = radius * (1.0 + rough) + 2.0
    y0, y1 = int(max(0, cy - r_out)), int(min(H, cy + r_out + 1))
    x0, x1 = int(max(0, cx - r_out)), int(min(W, cx + r_out + 1))
    if y1 <= y0 or x1 <= x0:
        return

    yy = np.arange(y0, y1)[:, None] - cy
    xx = np.arange(x0, x1)[None, :] - cx

    ar = 1.0 + rng.random() * 0.9              # real motes are not discs
    rot = rng.random() * math.pi
    cs, sn = math.cos(rot), math.sin(rot)
    yr = yy * cs - xx * sn
    xr = (yy * sn + xx * cs) / ar

    d = np.sqrt(yr * yr + xr * xr) + 1e-9
    th = np.arctan2(yr, xr)

    boundary = np.full_like(d, radius)
    for k in range(2, 2 + harmonics):
        boundary = boundary * (1.0 + (rough / harmonics) *
                               np.cos(k * th + rng.random() * 6.283))

    # A mote is imaged through the enlarger/scanner optics: its edge is never a
    # hard cut. Too stingy an edge term renders paper confetti, not dust.
    edge = max(0.5, radius * 0.45)
    a = opacity * np.clip((boundary - d) / edge + 0.5, 0.0, 1.0)

    sub = acc[y0:y1, x0:x1]
    acc[y0:y1, x0:x1] = 1.0 - (1.0 - sub) * (1.0 - a)


def _stamp_polyline(acc, pts, width, opacity, lobe=0.0):
    """
    Stamp a strand / line along pts. `opacity` may be a scalar or a per-segment
    sequence. `lobe` adds Kokaram-style diffraction side lobes: light deflected
    OUT of a refractive groove is deposited beside it, so the shoulders carry
    the opposite sign (negative opacity, i.e. tau slightly > 1).
    """
    H, W = acc.shape
    reach = width * (3.0 if lobe > 0 else 1.2) + 2.0
    for i in range(len(pts) - 1):
        (ay, ax), (by, bx) = pts[i], pts[i + 1]
        y0, y1 = int(max(0, min(ay, by) - reach)), int(min(H, max(ay, by) + reach + 1))
        x0, x1 = int(max(0, min(ax, bx) - reach)), int(min(W, max(ax, bx) + reach + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        vy, vx = by - ay, bx - ax
        L2 = vy * vy + vx * vx
        if L2 < 1e-9:
            continue
        t = np.clip(((yy - ay) * vy + (xx - ax) * vx) / L2, 0.0, 1.0)
        dy = yy - (ay + t * vy)
        dx = xx - (ax + t * vx)
        u = np.sqrt(dy * dy + dx * dx)

        w = max(width * 0.5, 0.5)
        op = opacity[i] if hasattr(opacity, "__len__") else opacity
        core = op * np.exp(-(u / w) ** 2)
        if lobe > 0.0:
            core = core - lobe * op * np.exp(-((u - 1.9 * w) / (0.8 * w)) ** 2)

        sub = acc[y0:y1, x0:x1]
        acc[y0:y1, x0:x1] = (1.0 - (1.0 - sub) * (1.0 - np.maximum(core, 0.0))
                             + np.minimum(core, 0.0))


def _wander_path(rng, H, W, axis, wander_px, n_seg=48, span=(0.0, 1.0)):
    """
    Transport-axis line with bounded lateral wander (Joyeux et al.: sinusoidal /
    cubic wander). axis 'v' runs top-to-bottom, 'h' runs left-to-right.
    """
    t = np.linspace(span[0], span[1], n_seg)
    off = np.zeros_like(t)
    for _ in range(3):
        f = 0.5 + rng.random() * 2.5
        off += (rng.random() - 0.5) * np.sin(2 * math.pi * f * t + rng.random() * 6.283)
    off = off / (np.abs(off).max() + 1e-9) * wander_px

    if axis == "v":
        base = rng.random() * W
        return list(zip(t * H, np.clip(base + off, 0, W - 1)))
    base = rng.random() * H
    return list(zip(np.clip(base + off, 0, H - 1), t * W))


def _hair_path(rng, H, W, length_px, n_seg=48, max_turn=1.4):
    """
    Fibre strand. Curvature is a BOUNDED RANDOM WALK with a capped cumulative
    turn: a constant curvature increment integrates to a perfect circle and
    renders visibly fake rings. Real fibres are arcs and S-bends, never loops.
    """
    y, x = rng.random() * H, rng.random() * W
    head = rng.random() * 6.283
    kappa = (rng.random() - 0.5) * 0.02
    step = length_px / n_seg
    total = 0.0
    pts = [(y, x)]
    for _ in range(n_seg):
        kappa += (rng.random() - 0.5) * 0.012
        kappa = max(-0.05, min(0.05, kappa))
        if abs(total + kappa) > max_turn:
            kappa = -0.5 * kappa
        total += kappa
        head += kappa
        y += step * math.sin(head)
        x += step * math.cos(head)
        pts.append((y, x))
    return pts


def _scratch_envelope(rng, n):
    """
    Along-length intensity envelope. Contact pressure varies, so a real scratch
    fades and breaks up; a dead-uniform edge-to-edge line reads as a drawn
    artifact rather than a tramline.
    """
    tt = np.linspace(0.0, 1.0, n)
    env = np.ones(n)
    for _ in range(2):
        f = 0.6 + rng.random() * 2.2
        env = env * (0.55 + 0.45 * np.sin(2 * math.pi * f * tt + rng.random() * 6.283))
    return np.clip(np.abs(env), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dye-layer model (derivation sec 3c)
# Removing a dye layer RAISES transmittance in the band that dye absorbed.
# In the native unit: a removed layer carrying optical density D gives 10**D.
# ---------------------------------------------------------------------------

def dye_tau_weights(depth, film_key, layer_density=0.7):
    """
    Per-channel transmittance multiplier (>= 1) for a full-strength emulsion
    gouge at `depth` in [0, 1]. film_key in {'bw', 'c41', 'reversal'}.
    """
    depth = float(np.clip(depth, 0.0, 1.0))
    if film_key == "bw":
        v = 10.0 ** (layer_density * depth)     # silver loss, spectrally flat
        return (v, v, v)
    f_yellow = np.clip(depth / _THIRD, 0.0, 1.0)                    # top layer
    f_magenta = np.clip((depth - _THIRD) / _THIRD, 0.0, 1.0)
    f_cyan = np.clip((depth - 2 * _THIRD) / _THIRD, 0.0, 1.0)       # bottom
    return (10.0 ** (layer_density * f_cyan),      # cyan absorbs RED
            10.0 ** (layer_density * f_magenta),   # magenta absorbs GREEN
            10.0 ** (layer_density * f_yellow))    # yellow absorbs BLUE


# ---------------------------------------------------------------------------
# Field builder + composite
# ---------------------------------------------------------------------------

def build_tau(H, W, seed, *, density=0.5,
              dust_amount=1.0, dirt_amount=1.0, hair_amount=1.0, scratch_count=3,
              dust_size=0.35, dirt_size=1.1, hair_length=140.0, scratch_width=0.8,
              scratch_side="base", scratch_depth=0.5, layer_density=0.7,
              film_key="c41", transport_axis="auto",
              softness=0.0, origin="negative", base_scratch_cast=0.0):
    """
    Build the per-channel transmittance field tau (H, W, 3).

    Sizes are ref-px @1024 long edge and are scaled here. Defect COUNT is a
    per-frame physical quantity and deliberately does NOT scale with pixel area:
    a frame carries the dust it carries whatever resolution you scan it at.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    L = max(H, W)
    s = L / 1024.0

    if softness <= 0.0:
        # derivation sec 6: a positive-plane defect sits off the image plane
        softness = 0.6 if origin == "negative" else 1.6
    softness *= s

    occl = np.zeros((H, W), dtype=np.float64)    # neutral occluders
    gouge = np.zeros((H, W), dtype=np.float64)   # emulsion dye removal, 0..1

    # --- dust ---------------------------------------------------------------
    n = int(rng.gamma(9.0, max(0.0, 55.0 * density * dust_amount) / 9.0)) if dust_amount > 0 else 0
    if n > 0:
        pos = _sample_positions(rng, n, H, W, clumpiness=1.6)
        radii = np.minimum(rng.gamma(4.0, (dust_size * s) / 4.0, size=n),
                           2.0 * dust_size * s)
        for (cy, cx), r in zip(pos, radii):
            _stamp_blob(occl, cy, cx, max(r, 0.35), 0.45 + 0.5 * rng.random(),
                        rng, harmonics=3, rough=0.22)

    # --- dirt ---------------------------------------------------------------
    n = int(rng.gamma(6.0, max(0.0, 14.0 * density * dirt_amount) / 6.0)) if dirt_amount > 0 else 0
    if n > 0:
        pos = _sample_positions(rng, n, H, W, clumpiness=2.6)
        radii = np.minimum(rng.gamma(3.0, (dirt_size * s) / 3.0, size=n),
                           2.5 * dirt_size * s)
        for (cy, cx), r in zip(pos, radii):
            _stamp_blob(occl, cy, cx, max(r, 0.8), 0.14 + 0.30 * rng.random(),
                        rng, harmonics=6, rough=0.40)

    # --- hairs / fibres (two length regimes, one code path) -----------------
    n = int(rng.gamma(3.0, max(0.0, 4.0 * density * hair_amount) / 3.0)) if hair_amount > 0 else 0
    for _ in range(n):
        long_hair = rng.random() < 0.45
        ln = hair_length * s * (1.0 if long_hair else 0.35) * (0.6 + rng.random())
        _stamp_polyline(occl, _hair_path(rng, H, W, ln),
                        max(0.6 * s, 0.55), 0.55 + 0.4 * rng.random())

    # --- scratches ----------------------------------------------------------
    if scratch_count > 0:
        axis = transport_axis
        if axis == "auto":
            # still 35mm: transport runs along the frame's LONG edge.
            # (cine runs vertically through the gate -> the vertical tramlines)
            axis = "h" if W >= H else "v"
        emulsion = (scratch_side == "emulsion")
        target = gouge if emulsion else occl
        for _ in range(int(scratch_count)):
            span = (0.0, 1.0) if rng.random() < 0.6 else \
                tuple(sorted((rng.random() * 0.5, 0.5 + rng.random() * 0.5)))
            pts = _wander_path(rng, H, W, axis, wander_px=6.0 * s, span=span)
            w = max(scratch_width * s * (0.7 + 0.8 * rng.random()), 0.7)
            amp = (0.45 + 0.5 * rng.random()) * _scratch_envelope(rng, len(pts) - 1)
            # base-side grooves are refractive -> real diffraction shoulders;
            # emulsion gouges are material loss -> no shoulder
            _stamp_polyline(target, pts, w, amp, lobe=0.0 if emulsion else 0.35)

    if softness > 0.05:
        occl = gaussian_filter(occl, softness)
        gouge = gaussian_filter(gouge, softness)

    # --- assemble tau -------------------------------------------------------
    tau = np.empty((H, W, 3), dtype=np.float64)
    occ = np.clip(occl, -0.6, 1.0)               # >0 blocks, <0 = lobe overshoot
    dye = dye_tau_weights(scratch_depth, film_key, layer_density)
    g = np.clip(gouge, 0.0, 1.0)

    # Optional base-side colour cast. HONEST STATUS: the green/cyan claim for
    # base-side scratches on colour negative is an INFERENCE chained from
    # wet-gate refractive physics + orange-mask channel gain; no single source
    # states the causal link. Off by default, exposed as taste only.
    cast = (1.0, 1.0 - 0.35 * base_scratch_cast, 1.0 - 0.12 * base_scratch_cast)

    for c in range(3):
        tau[..., c] = (1.0 - occ * cast[c]) * (1.0 + (dye[c] - 1.0) * g)
    return tau


def composite(lin, tau, origin, print_gamma):
    """
    THE model (derivation sec 1, LOAD-BEARING CALL 1).
    lin: (H, W, 3) LINEAR light. origin: 'negative' or 'positive'.
    """
    k = -float(print_gamma) if origin == "negative" else 1.0
    return np.clip(lin * (np.clip(tau, TAU_FLOOR, None) ** k), 0.0, 1.0)


def defect_mask(tau):
    """(H, W) coverage mask in [0, 1] from the strongest per-channel departure."""
    return np.clip(np.abs(1.0 - tau).max(axis=2), 0.0, 1.0).astype(np.float32)
