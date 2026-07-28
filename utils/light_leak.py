"""
Light-leak engine for ComfyUI-Darkroom (Light Leak node).

Path-length model. See docs/light-leak-derivation.md for the full derivation
(signed off 2026-07-28; spike 33/33 checks PASS with 6 negative controls firing;
Jeremie eyeball PASS, displacement added and constrained at his note).

A leak is stray light reaching the film outside the lens path, i.e. added
EXPOSURE, so the composite is additive in LINEAR light -- Halation's `lin + frac*H`,
reused because the same reasoning applies:

    L_out,c = L_in,c + s * G_c(x, y)

ONE mechanism sets both the falloff and the colour. Light entering the film EDGE
travels laterally through the base and dye layers, which absorb short wavelengths
harder (the round-trip argument behind Halation's derived w_R > w_G > w_B):

    E_c(d) = E_0 * exp(-d / lambda_c),   lambda_R > lambda_G > lambda_B

All channels start equal at d=0 and blue dies fastest, so a leak is a near-white
HOT CORE that reddens outward -- not the uniform orange every canned plate bakes.
The exception falls out of the same mechanism: a pinhole strikes the emulsion from
the FRONT with no lateral path, so d ~ 0 and it stays the colour of its source.
That is why the convention "light leaks are orange" is really a statement about
one geometry, and why this node can predict the case that is not.

Sizes are ref-px @1024 long edge (no-microns rule). Perforation geometry is Film
Rebate's shipped KS-1870 spec, not re-invented.
"""

import math

import numpy as np


# --- film geometry, straight from the shipped Film Rebate node --------------
# KS-1870: hole 1.98mm along transport x 2.79mm across, pitch 4.75mm, 8 per
# 38mm frame advance of which 36mm is the image -> 36/4.75 = 7.579 pitches
# across the long edge.
PERF_PITCH_MM = 4.75
PERF_HOLE_MM = 1.98            # extent ALONG the transport direction
PERF_CORNER_MM = 0.5           # corner radius -> the comb's soft shoulder
APERTURE_MM = 36.0

# The image area does NOT begin at the perforations. On 135, film is 35mm tall,
# the 24mm aperture is centred (so it spans 5.5-29.5mm) and perf hole centres sit
# 2.0mm from the film edge with a 2.79mm hole, i.e. the hole's inner edge is at
# 3.4mm. Light therefore travels ~2.1mm through film BEFORE it reaches the first
# image row, and arrives already diffused. Without this offset the comb hits the
# frame edge at full contrast and renders as a row of stage lights.
PERF_TO_APERTURE_MM = 2.1
PITCHES_ACROSS = APERTURE_MM / PERF_PITCH_MM      # 7.5789...

# Derived channel ordering (Halation precedent): ordering DERIVED, ratios PRESET
LAMBDA_RATIO = (1.00, 0.62, 0.38)                 # R, G, B

BACKING_PAPER_TINT = (1.00, 0.72, 0.55)           # honest preset, 120 roll paper
DAYLIGHT_TINT = (0.94, 0.97, 1.00)                # pinhole: source colour, ~neutral

# Displacement is a MECHANICAL variation: the gap of a felt trap or a failing
# door seal varies over millimetres of seam, and film contact against the
# pressure plate varies just as slowly. At 36mm across the frame, 1mm = 28.4
# ref-px @1024, so a 5mm seam feature is ~145 ref-px. Anything finer is not the
# physical thing -- it reads as noise running through the effect. The floor is
# therefore derived from the mechanism, not a taste guard.
WARP_SCALE_MIN_MM = 5.0
WARP_SCALE_FLOOR = WARP_SCALE_MIN_MM * (1024.0 / APERTURE_MM)   # ~142 ref-px


def _edge_fields(H, W, edge):
    """Return (d, u) in px: distance from `edge`, and position along it."""
    ys = np.arange(H)[:, None] * np.ones((1, W))
    xs = np.ones((H, 1)) * np.arange(W)[None, :]
    if edge == "top":
        return ys, xs
    if edge == "bottom":
        return (H - 1) - ys, xs
    if edge == "left":
        return xs, ys
    if edge == "right":
        return (W - 1) - xs, ys
    raise ValueError(edge)


def _warp_noise(H, W, scale_px, seed, octaves=3):
    """
    Smooth zero-mean multi-octave value noise in [-0.5, 0.5], used to displace
    the PATH-LENGTH field.

    Physical basis: the entry gap of a felt trap or a failing door seal is not
    uniform along its length, and the film's contact with the pressure plate
    varies, so the light does not travel the same distance everywhere. The
    mechanism is real; the noise field standing in for it is convention, the
    same honest status as the Perlin clustering in Film Damage.
    """
    from scipy.ndimage import zoom

    rng = np.random.default_rng((int(seed) * 7919 + 13) & 0x7FFFFFFF)
    acc = np.zeros((H, W))
    # Octave weights fall off FAST (1, 0.3, 0.09): the warp must stay a smooth
    # swell. Equal-ish octaves put real energy at scale/4, which is the "noise
    # running through it" look regardless of how large the base scale is.
    amp, total = 1.0, 0.0
    for o in range(max(1, int(octaves))):
        sc = max(scale_px / (2 ** o), 2.0)
        gh, gw = max(2, int(H / sc) + 2), max(2, int(W / sc) + 2)
        g = rng.random((gh, gw))
        up = zoom(g, (H / gh + 1e-9, W / gw + 1e-9), order=3, mode="nearest")
        up = up[:H, :W]
        if up.shape != (H, W):                       # zoom rounding guard
            pad = np.zeros((H, W))
            pad[:up.shape[0], :up.shape[1]] = up
            up = pad
        acc += amp * up
        total += amp
        amp *= 0.30
    acc /= total
    return acc - acc.mean()


def _corner_field(H, W, corner):
    ys = np.arange(H)[:, None] * np.ones((1, W))
    xs = np.ones((H, 1)) * np.arange(W)[None, :]
    cy = 0.0 if "top" in corner else (H - 1)
    cx = 0.0 if "left" in corner else (W - 1)
    return np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)


DUTY = PERF_HOLE_MM / PERF_PITCH_MM        # 0.4168


def _perf_comb_diffused(u, d, pitch_px, phase, sigma0, spread):
    """
    Perforation profile along the transport direction, WITH lateral diffusion.

    Light that passes a perforation spreads sideways as it travels into the
    film, so the comb does not merely fade with depth -- each tooth widens and
    softens into a blob. Modelling only the amplitude decay leaves teeth with
    parallel vertical sides, which renders as a bar chart rather than light
    (caught at the spike eyeball).

    Diffusive transport in a scattering slab gives sigma ~ sqrt(path):

        sigma(d) = sigma0 + spread * sqrt(d * pitch)

    A Gaussian-blurred rectangle is a difference of error functions, so this is
    evaluated in closed form -- exact, vectorised, no convolution.

    Normalised by the duty cycle so the far field tends to 1: the same total
    light, redistributed. Near the edge the peaks therefore run above the
    envelope (light concentrates through the holes), which is the real look.
    """
    from scipy.special import erf

    half = 0.5 * DUTY * pitch_px
    sigma = np.maximum(sigma0 + spread * np.sqrt(np.maximum(d, 0.0) * pitch_px), 1e-6)
    ph = np.mod(u - phase, pitch_px) - 0.5 * pitch_px      # signed dist to centre

    acc = np.zeros_like(ph)
    # neighbouring holes contribute once sigma grows past a pitch
    for k in (-2, -1, 0, 1, 2):
        c = k * pitch_px
        a = (ph - (c - half)) / (sigma * math.sqrt(2.0))
        b = (ph - (c + half)) / (sigma * math.sqrt(2.0))
        acc += 0.5 * (erf(a) - erf(b))
    return acc / DUTY


def leak_field(H, W, mode, *, edge="top", corner="top-left",
               intensity=1.0, lam_ref=200.0, mod_ratio=0.35, seed=42,   # mod_ratio = lateral diffusion coefficient
               color_source="base path", pinhole_count=3,
               hole_mm=0.30, flange_mm=50.0, source_angle=0.0093,
               displacement=0.0, displacement_scale=380.0, displacement_octaves=2,
               lam_ratio=LAMBDA_RATIO):
    """
    Per-channel leak field G_c, shape (H, W, 3), in LINEAR exposure units.

    lam_ref : red-channel falloff length, ref-px @1024 long edge.
    displacement : amplitude of the path-length warp, ref-px @1024. 0 = the
                clean analytic field. LOAD-BEARING that this displaces `d`
                (path length) and NOT the finished field: because colour is a
                function of d, warping d carries the falloff, the colour fringe
                and (for sprocket) the comb diffusion together, coherently.
                Warping the output instead would slide the colours off the
                geometry that produced them.
    displacement_scale : warp feature size, ref-px @1024. CLAMPED at
                WARP_SCALE_FLOOR (~142, i.e. 5mm of seam): finer features stop
                being a mechanical gap variation and read as noise running
                through the effect. Octaves are capped at 2 with a fast 0.30
                falloff for the same reason.
    mod_ratio : lateral diffusion coefficient (sigma = sigma0 + c*sqrt(d*pitch)).
                LOAD-BEARING that it is > 0 -- light
                spreads laterally as it travels, so the perforation comb blurs
                out faster than the overall glow persists.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    L = max(H, W)
    s = L / 1024.0
    lam_R = max(lam_ref * s, 1e-6)
    lam = [lam_R * (r / lam_ratio[0]) for r in lam_ratio]

    G = np.zeros((H, W, 3), dtype=np.float64)

    if mode == "pinhole":
        # camera obscura: spot diameter ~ a + D*theta (mm), converted to px via
        # the 36mm aperture. No lateral path through the base -> NEUTRAL.
        px_per_mm = L / APERTURE_MM
        core_r = 0.5 * flange_mm * source_angle * px_per_mm
        blur = max(0.5 * hole_mm * px_per_mm, 0.75)
        ys = np.arange(H)[:, None]
        xs = np.arange(W)[None, :]
        acc = np.zeros((H, W))
        for _ in range(int(pinhole_count)):
            cy, cx = rng.random() * H, rng.random() * W
            amp = 0.55 + 0.65 * rng.random()
            d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
            acc = acc + amp * np.clip((core_r - d) / blur + 0.5, 0.0, 1.0)
        tint = DAYLIGHT_TINT if color_source == "neutral" else (1.0, 1.0, 1.0)
        for c in range(3):
            G[..., c] = acc * tint[c]
        return intensity * G

    warp = None
    if displacement > 0.0:
        # clamp to the mechanical floor -- see WARP_SCALE_FLOOR
        scale_ref = max(float(displacement_scale), WARP_SCALE_FLOOR)
        warp = (2.0 * displacement * s) * _warp_noise(
            H, W, scale_ref * s, seed, min(int(displacement_octaves), 2))

    if mode == "gradient":
        d = _corner_field(H, W, corner) if corner else _edge_fields(H, W, edge)[0]
        if edge and not corner:
            d = _edge_fields(H, W, edge)[0]
        if warp is not None:
            d = np.maximum(d + warp, 0.0)
        mod = 1.0
    elif mode == "sprocket":
        d, u = _edge_fields(H, W, edge)
        if warp is not None:
            d = np.maximum(d + warp, 0.0)
        pitch_px = (PERF_PITCH_MM / APERTURE_MM) * L
        sigma0 = (PERF_CORNER_MM / APERTURE_MM) * L    # the hole's own corner radius
        phase = rng.random() * pitch_px
        # Light has already crossed the rebate band before the image starts, so
        # the comb arrives pre-diffused (see PERF_TO_APERTURE_MM).
        d_perf = d + (PERF_TO_APERTURE_MM / APERTURE_MM) * L
        # ONE mechanism now: lateral diffusion both softens the tooth shape and
        # kills its contrast with depth, so lambda_mod is no longer a free knob.
        mod = _perf_comb_diffused(u, d_perf, pitch_px, phase, sigma0, mod_ratio)
    else:
        raise ValueError(mode)

    for c in range(3):
        G[..., c] = np.exp(-d / lam[c]) * mod

    if color_source == "backing paper":
        for c in range(3):
            G[..., c] *= BACKING_PAPER_TINT[c]

    return intensity * G


def composite(lin, G, strength=1.0):
    """Additive in LINEAR light -- the leak ADDS exposure (Halation's composite)."""
    return np.clip(lin + strength * G, 0.0, 1.0)
