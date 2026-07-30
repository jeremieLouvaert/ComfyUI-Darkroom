"""
Water Refraction engine — exact Snell through a simulated free surface.

Full derivation: docs/water-refraction-derivation.md. What makes this not a
displacement filter, in order of how much each carries:

  1. The warp is a CONSEQUENCE of a simulated water surface, not a texture. A
     depth-averaged grid solver pours real water onto the screen; the optics
     then read only h(x,y) from it. Nothing here is a noise field dressed up.

  2. Refraction is EXACT Snell, not the small-angle form. Because theta_t
     saturates at the critical angle 48.6deg, displacement is HARD-BOUNDED:

         Delta = h * tan(theta_i - theta_t) <= 0.881 * h

     no matter how violent the surface gets. That bound is why the effect is
     inherently MACRO and why `field_width_mm` is a required control rather than
     a nicety: frame a whole tablet and you correctly get almost nothing.

  3. Displacement is UPHILL, along +grad h. The transmitted ray bends toward the
     INWARD normal, whose horizontal component points up-slope. The first draft
     of the derivation had this backwards and every self-consistent check passed
     either way; only an independent vector ray trace caught it. That trace is
     kept as `trace_reference` and is a live test. Sanity anchor: a water drop on
     text MAGNIFIES.

  4. Finite APERTURE, which is required physics rather than softening. A camera
     is not a pinhole; for a pixel focused on screen point p the lens collects a
     cone crossing the surface over a disc of radius rho = (A/L)*h. Every point
     of that footprint sees a different slope, so flat water stays sharp, curved
     water blurs, and at a FOLD the several images of one feature BLEND instead
     of butting together along a seam. That is what removes the chrome/liquify
     look.

UNITS: millimetres and seconds throughout — a deliberate departure from the house
ref-px rule (derivation sec 7), because the look depends on ABSOLUTE depth.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

# --- water at 20 C, in mm/s units -------------------------------------------
G_MM = 9810.0                       # gravity, mm/s^2
NU_MM = 1.004                       # kinematic viscosity, mm^2/s
SIGMA_OVER_RHO = 7.294e4            # sigma/rho, mm^3/s^2 (72.8 mN/m over 998 kg/m^3)
CAPILLARY_MM = math.sqrt(SIGMA_OVER_RHO / G_MM)          # 2.727 mm, derived not typed

N_WATER = 1.333
CRITICAL_ANGLE = math.asin(1.0 / N_WATER)                # 48.6 deg
DELTA_MAX_RATIO = math.tan(math.pi / 2 - CRITICAL_ANGLE)  # 0.881
R0 = ((1.0 - N_WATER) / (1.0 + N_WATER)) ** 2            # Fresnel at normal incidence
IOR_RGB = (1.3311, 1.3330, 1.3435)                       # ~700 / ~550 / ~400 nm

# How far a full sweep drags the stream across the frame. Was 0.778, matching the
# spike path exactly, but at that length the stroke plus its spawn disc fills the
# frame and leaves the seed no room to move the pour without pushing it off the
# plate. 0.48 leaves +/-0.16 of frame to jitter in.
#
# The original note here justified the reduction by claiming the sweep had no
# measurable effect. That was WRONG -- it came from the particle solver read through
# a metric later shown to be measuring frame aspect ratio. Re-measured on the grid
# solver with placement pinned, sweeping moves the surface 2.24mm mean and folding
# runs 39.0% / 30.4% / 24.3% across the range. So the value stands, but for the
# honest reason: sweep length and seed placement BOTH matter and 0.48 balances them.
SWEEP_SPAN = 0.48


# ===========================================================================
# Fluid solver — depth-averaged, on a grid
# ===========================================================================

class GridShallowWater:
    """
    Depth-averaged (shallow water) solver on a cell-centred grid:

        dh/dt + d(hu)/dx + d(hv)/dy = 0
        du/dt + (u.grad)u = -g grad(h+b) - 3 nu u / h^2 + nu lap(u)
                            + (sigma/rho) grad(lap h)

    There is NO incompressibility constraint and therefore no pressure
    projection: in shallow water the hydrostatic term -g grad(h) IS the pressure
    gradient and h is free to vary. That is what makes this affordable.

    The bed drag -3 nu u / h^2 is the laminar film result, not an ad-hoc pin. It
    diverges as h -> 0, so thin films pin themselves and the contact line emerges
    from the equations instead of being thresholded in.

    WHY A GRID AND NOT PARTICLES. This replaced a FLIP/PIC particle solver that
    estimated h by counting particles per cell. Because the dominant force is
    -g*grad(h), density sampling noise became force noise, which moved particles,
    which changed the density -- a feedback loop rather than a sampling error.
    Measured: a flat 6.00 mm film with no dynamics at all reconstructed as
    1.62..9.81 mm (12.7% noise), and it got WORSE with 3x the particles (14.4%)
    instead of falling as 1/sqrt(N). Since refraction differentiates h, that noise
    was the dominant structure the optics saw.

    Solving the continuity equation on the grid removes the feedback path
    entirely: the same flat film now reconstructs at 6.00..6.00 mm, 0.00% noise.
    Folding is unaffected (15.6% vs 15.1% on the same pour) and it runs 4x faster,
    because the particle scatter/gather it replaces was 99% of a step.

    The physics is unchanged -- same equations, same constants -- so the surface
    is still the solution of a simulated free surface, and now more literally so:
    it is the actual solution of the continuity equation rather than an estimate
    from counting particles.
    """

    def __init__(self, field_w_mm, field_h_mm, nx=112, seed=0,
                 tilt_deg=0.0, tilt_dir_deg=90.0, friction=1.0, viscosity=1.0,
                 surface_tension=1.0, edge="wall"):
        self.W = float(field_w_mm)
        self.H = float(field_h_mm)
        self.nx = int(nx)
        self.ny = max(2, int(round(nx * self.H / self.W)))
        self.dx = self.W / self.nx
        self.cell_area = self.dx * self.dx
        self.rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

        self.friction = float(max(friction, 0.0))
        self.viscosity = float(max(viscosity, 0.0))
        self.surface_tension = float(max(surface_tension, 0.0))
        self.edge = edge

        t = math.radians(float(tilt_deg))
        d = math.radians(float(tilt_dir_deg))
        gp = G_MM * math.sin(t)
        self.gx_plane = gp * math.cos(d)
        self.gy_plane = gp * math.sin(d)
        self.g_normal = G_MM * math.cos(t)

        self.h = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))
        self.t = 0.0
        self._yy, self._xx = np.mgrid[0:self.ny, 0:self.nx].astype(np.float64)

    # -- sources -------------------------------------------------------------

    def prefill(self, depth_mm):
        """Standing water of a given depth, everywhere."""
        if depth_mm > 0:
            self.h[:] = float(depth_mm)

    def pour(self, cx_mm, cy_mm, radius_mm, rate_mm3_s, speed_mm_s, dt):
        """
        Inject a stream at (cx, cy). Water arriving from above converts its fall
        speed into radial outflow at the impact point (the hydraulic-jump
        behaviour), which produces the spreading bore and its raised rim.

        Momentum is blended by the MASS FRACTION being added, so injecting into
        deep water barely deflects it while injecting onto dry screen sets the
        velocity outright. Adding velocity directly instead would let a trickle
        overwrite the momentum of a whole pool.
        """
        if rate_mm3_s <= 0 or dt <= 0:
            return 0.0
        r = np.hypot(self._xx * self.dx - cx_mm, self._yy * self.dx - cy_mm)
        disc = np.exp(-0.5 * (r / max(radius_mm, 1e-6)) ** 2)
        tot = float(disc.sum()) * self.cell_area
        if tot <= 1e-12:
            return 0.0
        add = disc * (rate_mm3_s * dt / tot)
        newh = self.h + add
        ang = np.arctan2(self._yy * self.dx - cy_mm, self._xx * self.dx - cx_mm)
        frac = np.divide(add, np.maximum(newh, 1e-9))
        self.u = self.u * (1.0 - frac) + speed_mm_s * np.cos(ang) * frac
        self.v = self.v * (1.0 - frac) + speed_mm_s * np.sin(ang) * frac
        self.h = newh
        return float(add.sum()) * self.cell_area

    # -- one step ------------------------------------------------------------

    def cfl_dt(self, safety=0.35, dt_max=2e-3):
        wave = math.sqrt(self.g_normal * max(float(self.h.max()), 1e-6))
        vel = float(np.hypot(self.u, self.v).max())
        return min(dt_max, safety * self.dx / max(wave + vel, 1e-6))

    def _advect(self, f, dt):
        """Semi-Lagrangian: trace back along the velocity and sample."""
        by = np.clip(self._yy - dt * self.v / self.dx, 0.0, self.ny - 1.001)
        bx = np.clip(self._xx - dt * self.u / self.dx, 0.0, self.nx - 1.001)
        return map_coordinates(f, [by, bx], order=1, mode="nearest")

    def _continuity(self, dt):
        """
        Conservative, positivity-preserving update of h.

        Upwind fluxes on cell FACES, so whatever leaves one cell enters its
        neighbour exactly and total volume can only change through the domain
        boundary. A first draft used central differences on d(hu)/dx and clamped
        h at zero, which manufactured water: 8000 mm3 poured came back as 8848,
        a 10.6% gain. Invariant I1 exists for precisely this.

        Outgoing fluxes are additionally limited so a cell can never give away
        more than it holds, which keeps h >= 0 without the clamp that broke
        conservation in the first place.
        """
        h, u, v = self.h, self.u, self.v
        # face-centred velocities and upwinded depths
        uf = 0.5 * (u[:, :-1] + u[:, 1:])
        hf = np.where(uf >= 0.0, h[:, :-1], h[:, 1:])
        fx = np.zeros((self.ny, self.nx + 1))
        fx[:, 1:-1] = uf * hf                       # walls: zero flux at 0 and nx

        vf = 0.5 * (v[:-1, :] + v[1:, :])
        hg = np.where(vf >= 0.0, h[:-1, :], h[1:, :])
        fy = np.zeros((self.ny + 1, self.nx))
        fy[1:-1, :] = vf * hg

        if self.edge != "wall":                     # open: let it run off the edge
            fx[:, 0] = np.minimum(uf[:, 0] * h[:, 0], 0.0)
            fx[:, -1] = np.maximum(uf[:, -1] * h[:, -1], 0.0)
            fy[0, :] = np.minimum(vf[0, :] * h[0, :], 0.0)
            fy[-1, :] = np.maximum(vf[-1, :] * h[-1, :], 0.0)

        k = dt / self.dx
        out = (np.maximum(fx[:, 1:], 0.0) + np.maximum(-fx[:, :-1], 0.0)
               + np.maximum(fy[1:, :], 0.0) + np.maximum(-fy[:-1, :], 0.0)) * k
        scale = np.ones_like(h)
        busy = out > 1e-12
        scale[busy] = np.minimum(1.0, h[busy] / out[busy])

        # a face's outgoing side is limited by whichever cell is donating
        sx = np.ones_like(fx)
        sx[:, 1:-1] = np.where(fx[:, 1:-1] >= 0.0, scale[:, :-1], scale[:, 1:])
        sy = np.ones_like(fy)
        sy[1:-1, :] = np.where(fy[1:-1, :] >= 0.0, scale[:-1, :], scale[1:, :])
        fx = fx * sx
        fy = fy * sy

        self.h = np.maximum(h - k * ((fx[:, 1:] - fx[:, :-1])
                                     + (fy[1:, :] - fy[:-1, :])), 0.0)

    def step(self, dt=None):
        if dt is None:
            dt = self.cfl_dt()
        h, u, v = self.h, self.u, self.v

        hs = gaussian_filter(h, 0.8)
        hy, hx = np.gradient(hs, self.dx)
        ax = -self.g_normal * hx + self.gx_plane
        ay = -self.g_normal * hy + self.gy_plane

        # surface tension sets the capillary scale, which is what makes beads and
        # rivulets instead of a flat film. Not optional at these depths.
        if self.surface_tension > 0.0:
            lap = (np.roll(hs, 1, 0) + np.roll(hs, -1, 0) + np.roll(hs, 1, 1)
                   + np.roll(hs, -1, 1) - 4.0 * hs) / (self.dx * self.dx)
            lap = gaussian_filter(lap, 0.8)
            ly, lx = np.gradient(lap, self.dx)
            k = SIGMA_OVER_RHO * self.surface_tension
            ax += k * lx
            ay += k * ly

        hd = np.maximum(h, 0.05)
        drag = 3.0 * NU_MM * self.viscosity * self.friction / (hd * hd)
        drag = np.minimum(drag, 0.9 / max(dt, 1e-9))
        ax -= drag * u
        ay -= drag * v

        if self.viscosity > 0.0:
            for arr, is_x in ((u, True), (v, False)):
                lp = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) + np.roll(arr, 1, 1)
                      + np.roll(arr, -1, 1) - 4.0 * arr) / (self.dx * self.dx)
                if is_x:
                    ax += NU_MM * self.viscosity * lp
                else:
                    ay += NU_MM * self.viscosity * lp

        dry = h < 1e-4
        ax[dry] = 0.0
        ay[dry] = 0.0

        self.u = self._advect(u, dt) + dt * ax
        self.v = self._advect(v, dt) + dt * ay
        if self.edge == "wall":
            self.u[:, 0] = self.u[:, -1] = 0.0
            self.v[0, :] = self.v[-1, :] = 0.0

        self._continuity(dt)
        self.t += dt
        return dt

    # -- output --------------------------------------------------------------

    def height(self, smooth_cells=1.8):
        """
        The rendered free surface.

        Smoothing here is only to take the corner off the discretisation, and is
        measured in CELLS. An earlier particle version smoothed by a full
        capillary length on the reasoning that surface tension cannot sustain
        finer curvature; that conflated two things and erased the real structure
        (slope 2.28 -> 0.52, folding 52.9% -> 0.0% on the SAME sim). Surface
        tension is already in the momentum equation and damps curvature
        dynamically; it must not be re-imposed geometrically on the output.
        """
        return np.maximum(gaussian_filter(self.h, max(float(smooth_cells), 0.4)), 0.0)

    def volume_mm3(self):
        return float(self.h.sum()) * self.cell_area


def sweep_path(s, w_mm, h_mm, sweep, angle_deg=45.0, wander=0.10, centre=(0.5, 0.5)):
    """
    Where the stream lands at normalised time s in [0,1].

    sweep = 0 is a stationary source, which generates an axisymmetric spreading
    disc. HONEST NOTE, measured after this shipped: sweeping does NOT change the
    surface's morphology in any way the metrics can find (elongation 1.39 swept
    against 1.42 stationary), so it is retained as a placement control rather
    than the anti-circularity fix it was originally built to be.

    Angles are taken in NORMALISED frame coordinates, not millimetres, so the
    path is the same fraction of the picture whatever the aspect ratio.

    `centre` is where the stroke is centred, as a fraction of the frame, and it
    is driven by the SEED via stroke_centre(). It used to be hardcoded to
    (0.5, 0.5), which made the seed cosmetic: it reached only the pour's own
    jitter, so re-rolling changed the water's texture and never where it landed.
    """
    a = math.radians(angle_deg)
    cx, cy = centre[0] * w_mm, centre[1] * h_mm
    span = sweep * SWEEP_SPAN
    ox = (s - 0.5) * span * math.cos(a) * w_mm
    oy = (s - 0.5) * span * math.sin(a) * h_mm
    if wander > 0.0 and sweep > 0.0:
        ox += wander * sweep * w_mm * 0.18 * math.sin(7.0 * s)
        oy += wander * sweep * h_mm * 0.12 * math.sin(4.5 * s + 1.0)
    return cx + ox, cy + oy


def stroke_centre(seed, sweep):
    """
    Where the seed puts the pour, as a fraction of the frame.

    Two bugs live here, both found from the live node rather than from the teeth,
    which is why both now have regression checks:

      THE SEED MUST MOVE THE POUR. This was hardcoded to (0.5, 0.5), so the seed
      only ever reached the particle jitter inside the spawn disc: re-rolling
      changed the water's texture and never where it landed. Measured centroid
      spread across four seeds was 0.101 mm on a 40 mm field.

      THE STROKE MUST STAY ON THE PLATE. The sweep reaches +/-0.275 of the frame
      either side of the centre, so the first fix's flat +/-0.30 jitter could put
      the pour at 107% of the frame -- off the plate for 4 of 6 seeds, dumping the
      water against the boundary. The room left for jitter therefore has to shrink
      as `sweep` grows, which is what this computes.
    """
    prng = np.random.default_rng((int(seed) & 0x7FFFFFFF) ^ 0x5EED)
    half = 0.5 * float(sweep) * SWEEP_SPAN + 0.06   # stroke half-span + the spawn disc
    room = max(0.5 - half - 0.04, 0.0)              # keep 4% clear of the wall
    return (0.5 + room * (prng.random() - 0.5) * 2.0,
            0.5 + room * (prng.random() - 0.5) * 2.0)


def simulate(field_w_mm, field_h_mm, *, volume_ml=16.0, nx=112, seed=0,
             sample_ms=80.0, pour_ms=100.0, pour_radius_mm=3.0,
             pour_speed_mm_s=700.0, sweep=1.0, sweep_angle_deg=45.0,
             initial_film_mm=0.0, tilt_deg=0.0,
             surface_tension=1.0, viscosity=1.0,
             edge="wall", progress=None):
    """
    Pour water onto the screen and return the surface, in mm, on the sim grid.

    Sampled DURING the pour by default (sample < pour duration): the reference
    look lives in the live event, and letting it finish and spread costs both the
    depth that drives displacement and the structure that drives folding.
    """
    # THE SEED MUST MOVE THE POUR. A stroke centred at (0.5,0.5) every time made the
    # seed cosmetic: it reached only the particle jitter inside the spawn disc, so
    # re-rolling changed the water's texture and never where it landed. The stroke is
    # kept inside the middle 60% of the frame so a re-roll cannot dump the whole pour
    # off-picture, and the sweep DIRECTION stays the user's `sweep_angle`.
    centre = stroke_centre(seed, sweep)

    sim = GridShallowWater(field_w_mm, field_h_mm, nx=nx, seed=seed,
                           tilt_deg=tilt_deg, surface_tension=surface_tension,
                           viscosity=viscosity, edge=edge)
    sim.prefill(initial_film_mm)

    pour_s = max(pour_ms, 1e-3) / 1000.0
    end_s = max(sample_ms, 1e-3) / 1000.0
    rate_mm3_s = float(volume_ml) * 1000.0 / pour_s

    dt, steps = 1e-4, 0
    while sim.t < end_s:
        dt = sim.cfl_dt()
        if sim.t < pour_s:
            cx, cy = sweep_path(min(sim.t / pour_s, 1.0), field_w_mm, field_h_mm,
                                sweep, sweep_angle_deg, centre=centre)
            sim.pour(cx, cy, pour_radius_mm, rate_mm3_s, pour_speed_mm_s, dt)
        sim.step(dt)
        steps += 1
        if progress is not None and steps % 100 == 0:
            progress(sim.t / end_s)
        if steps > 40000:
            break
    return sim.height(), sim


def settle(h_mm, dx_mm, seconds):
    """
    Viscous decay of the disturbance, gamma(k) = 2 nu k^2 (Lamb, weak-viscosity
    free surface). Quadratic in k, so it removes fine chop and leaves the large
    structure — a 2.4mm ripple dies in ~72ms while a 12mm one lives ~1.8s.

    This is operator splitting, not a shortcut around the physics: once the pour
    transient is over the remaining evolution IS linear wave decay, and the
    solver's own dissipation produces the same law at ~1300 CFL-limited steps
    instead of one FFT. Mean depth is conserved exactly.
    """
    if seconds <= 0:
        return h_mm.copy()
    hh, ww = h_mm.shape
    ky = np.fft.fftfreq(hh, d=dx_mm)[:, None]
    kx = np.fft.fftfreq(ww, d=dx_mm)[None, :]
    k = 2.0 * math.pi * np.hypot(kx, ky)
    m = float(h_mm.mean())
    return np.maximum(m + np.real(np.fft.ifft2(
        np.fft.fft2(h_mm - m) * np.exp(-2.0 * NU_MM * k * k * seconds))), 0.0)


def to_image_res(h_mm, out_h, out_w):
    z = max(out_h / h_mm.shape[0], out_w / h_mm.shape[1])
    up = np.maximum(zoom(h_mm, (z, z), order=3), 0.0)
    return up[:out_h, :out_w]


# ===========================================================================
# Optics
# ===========================================================================

def refraction_offsets(h_mm, field_width_mm, ior=N_WATER, depth_scale=1.0):
    """
    SOURCE offset in pixels for each output pixel, plus the incidence cosine.

    Exact Snell, not small-angle: the reference regime is steep, where the
    small-angle form underestimates and misses the 0.881 saturation entirely.
    """
    h_px_w = h_mm.shape[1]
    mm_per_px = field_width_mm / h_px_w
    h = np.maximum(h_mm * depth_scale, 0.0)

    gy, gx = np.gradient(h, mm_per_px)          # dimensionless slope
    g = np.hypot(gx, gy)
    theta_i = np.arctan(g)
    theta_t = np.arcsin(np.clip(np.sin(theta_i) / ior, -1.0, 1.0))
    delta_px = (h * np.tan(theta_i - theta_t)) / mm_per_px

    inv = np.where(g > 1e-12, 1.0 / np.maximum(g, 1e-12), 0.0)
    # UPHILL, along +grad h — see the module docstring.
    return delta_px * gx * inv, delta_px * gy * inv, np.cos(theta_i)


def _sample(img, ys, xs, order=3):
    out = np.empty((ys.shape[0], ys.shape[1], img.shape[2]), dtype=np.float64)
    for c in range(img.shape[2]):
        out[..., c] = map_coordinates(img[..., c], [ys, xs], order=order, mode="reflect")
    return out


def render(img, h_mm, field_width_mm, *, aperture_ratio=0.020, samples=32,
           ior=N_WATER, depth_scale=1.0, fresnel=True,
           env_color=(0.75, 0.78, 0.82), env_strength=1.0, dispersion=False,
           order=3, seed=0, pixel_aa=True):
    """
    Refract `img` (H,W,3 in [0,1]) through the surface, with a finite aperture.

    Backward map: every output pixel has exactly one screen point, so folds and
    multiple images fall out for free.

    LOAD-BEARING: the sample lands at p + Delta(p + delta), NOT at
    p + delta + Delta(p + delta). Focusing already converges the cone onto p;
    only the refractive deflection is sampled across the footprint. Getting this
    wrong blurs flat water, which a focused camera does not do.

    `order` is the image-sampling interpolation. It defaults to 3 rather than 1
    because bilinear retains only (2/3)^2 of a white-noise variance at a random
    sub-pixel position — a flat ~33% loss of film grain on every displaced pixel
    regardless of what the water is doing. Cubic recovers most of it for free.
    """
    H, W = img.shape[:2]
    mm_per_px = field_width_mm / W
    h = np.maximum(h_mm * depth_scale, 0.0)
    ys0, xs0 = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    iors = IOR_RGB if dispersion else (ior,)
    acc = np.zeros_like(img, dtype=np.float64)
    cos_i = None
    for ci, n_w in enumerate(iors):
        dxp, dyp, ci_map = refraction_offsets(h_mm, field_width_mm, ior=n_w,
                                              depth_scale=depth_scale)
        if cos_i is None:
            cos_i = ci_map
        rho = (aperture_ratio * h) / mm_per_px

        # How wide the pixel's preimage is, in SOURCE pixels: m = sqrt(|det J|).
        # Jitter must vanish as m -> 1, because the input image has ALREADY been
        # through a sensor's own pixel integration -- averaging over a full pixel
        # again would double-count it and blur an undistorted image. That is not
        # hypothetical: applying the jitter unconditionally broke invariant I3,
        # flat water is a bit-exact identity, by 4.3e-1. Only the EXCESS over one
        # source pixel needs integrating, which is what the sqrt(1 - 1/m^2) factor
        # is: exactly 0 at m = 1, tending to 1 under strong compression.
        _sx = dxp + np.arange(W)[None, :]
        _sy = dyp + np.arange(H)[:, None]
        _sxy, _sxx = np.gradient(_sx)
        _syy, _syx = np.gradient(_sy)
        _m2 = np.abs(_sxx * _syy - _sxy * _syx)          # |det J|, = m^2
        jitter_amp = np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(_m2, 1e-9), 0.0))
        if not pixel_aa:
            jitter_amp = np.zeros_like(jitter_amp)
        chan = np.zeros_like(img, dtype=np.float64)
        for _ in range(max(1, int(samples))):
            ang = 2.0 * math.pi * rng.random()
            rad = math.sqrt(rng.random())
            # PREIMAGE JITTER, alongside the aperture offset. Where the map
            # compresses, one output pixel draws from a LARGER patch of screen, so
            # point-sampling it aliases -- measured against a supersampled truth,
            # error climbed monotonically with compression (0.039 magnified, 0.052
            # at 1-2x, 0.062 at 2-4x, 0.078 beyond 4x). Jittering the sample
            # position makes the existing aperture budget integrate that preimage
            # too, at no extra cost. The displacement is evaluated AT the jittered
            # position, because the surface varies across the pixel as well.
            py = ys0 + jitter_amp * (rng.random() - 0.5)
            px = xs0 + jitter_amp * (rng.random() - 0.5)
            qy = py + rad * math.sin(ang) * rho
            qx = px + rad * math.cos(ang) * rho
            dxs = map_coordinates(dxp, [qy, qx], order=1, mode="nearest")
            dys = map_coordinates(dyp, [qy, qx], order=1, mode="nearest")
            chan += _sample(img, py + dys, px + dxs, order=order)
        chan /= max(1, int(samples))
        if dispersion:
            acc[..., ci] = chan[..., ci]
        else:
            acc = chan

    if fresnel:
        R = R0 + (1.0 - R0) * (1.0 - np.clip(cos_i, 0.0, 1.0)) ** 5
        env = np.array(env_color, dtype=np.float64)[None, None, :] * env_strength
        acc = (1.0 - R[..., None]) * acc + R[..., None] * env
    return np.clip(acc, 0.0, 1.0)


def _torch_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def render_gpu(img, h_mm, field_width_mm, *, aperture_ratio=0.020, samples=32,
               ior=N_WATER, depth_scale=1.0, fresnel=True,
               env_color=(0.75, 0.78, 0.82), env_strength=1.0, dispersion=False,
               seed=0, pixel_aa=True):
    """
    The same optics on the GPU. Identical maths to `render`, expressed in torch.

    Worth it here and NOT worth it for the solver, which is the interesting part:
    this is 780k pixels x 32 samples = 25 million gathers, which saturates the
    device and runs 43x faster (13.4s -> 0.31s). The fluid solver was ported too
    and managed only 1.23x, because its 160x215 grid is ~34k cells and every step
    launches dozens of kernels that never fill the GPU. Big dense problem: port it.
    Small stepped problem: leave it alone.

    float32 here, so results are not bit-identical to the numpy path -- which is
    why the invariant suite is built on physics rather than pixels, and why the
    numpy `render` is kept as the reference the teeth check against.
    """
    import torch
    import torch.nn.functional as Fn

    dev = "cuda"
    H, W = img.shape[:2]
    mm_per_px = field_width_mm / W
    t_img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32,
                         device=dev).permute(2, 0, 1).unsqueeze(0)
    hh = torch.tensor(np.ascontiguousarray(h_mm), dtype=torch.float32, device=dev)
    h_s = torch.clamp(hh * depth_scale, min=0.0)

    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")

    def grad(a):
        gy = torch.zeros_like(a); gx = torch.zeros_like(a)
        gy[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2 * mm_per_px)
        gy[0, :] = (a[1, :] - a[0, :]) / mm_per_px
        gy[-1, :] = (a[-1, :] - a[-2, :]) / mm_per_px
        gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2 * mm_per_px)
        gx[:, 0] = (a[:, 1] - a[:, 0]) / mm_per_px
        gx[:, -1] = (a[:, -1] - a[:, -2]) / mm_per_px
        return gy, gx

    def offsets(n_w):
        gy_, gx_ = grad(h_s)
        g = torch.sqrt(gx_ * gx_ + gy_ * gy_)
        th_i = torch.atan(g)
        th_t = torch.asin(torch.clamp(torch.sin(th_i) / n_w, -1.0, 1.0))
        dpx = (h_s * torch.tan(th_i - th_t)) / mm_per_px
        inv = torch.where(g > 1e-12, 1.0 / torch.clamp(g, min=1e-12), torch.zeros_like(g))
        return dpx * gx_ * inv, dpx * gy_ * inv, torch.cos(th_i)

    def samp(field4, ys, xs, mode="bilinear"):
        gyy = (ys / max(H - 1, 1)) * 2 - 1
        gxx = (xs / max(W - 1, 1)) * 2 - 1
        grid = torch.stack([gxx, gyy], dim=-1).unsqueeze(0)
        return Fn.grid_sample(field4, grid, mode=mode, padding_mode="reflection",
                              align_corners=True)

    gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed) & 0x7FFFFFFF)
    rand = lambda: float(torch.rand(1, generator=gen))

    iors = IOR_RGB if dispersion else (ior,)
    acc = torch.zeros_like(t_img)
    cos_i = None
    for ci, n_w in enumerate(iors):
        dxp, dyp, ci_map = offsets(n_w)
        if cos_i is None:
            cos_i = ci_map
        # preimage jitter amplitude from |det J|, exactly as the numpy path
        sx = dxp + xx
        sy = dyp + yy
        sxy, sxx_ = torch.gradient(sx, dim=(0, 1))
        syy_, syx = torch.gradient(sy, dim=(0, 1))
        m2 = torch.abs(sxx_ * syy_ - sxy * syx)
        jamp = torch.sqrt(torch.clamp(1.0 - 1.0 / torch.clamp(m2, min=1e-9), min=0.0))
        if not pixel_aa:
            jamp = torch.zeros_like(jamp)
        rho = (aperture_ratio * h_s) / mm_per_px
        d4 = torch.stack([dxp, dyp], 0).unsqueeze(0)
        chan = torch.zeros_like(t_img)
        for _ in range(max(1, int(samples))):
            ang = 2.0 * math.pi * rand()
            rad = math.sqrt(rand())
            py = yy + jamp * (rand() - 0.5)
            px = xx + jamp * (rand() - 0.5)
            qy = py + rad * math.sin(ang) * rho
            qx = px + rad * math.cos(ang) * rho
            ds = samp(d4, qy, qx)
            chan = chan + samp(t_img, py + ds[0, 1], px + ds[0, 0])
        chan = chan / max(1, int(samples))
        if dispersion:
            acc[:, ci] = chan[:, ci]
        else:
            acc = chan

    if fresnel:
        R = R0 + (1.0 - R0) * (1.0 - torch.clamp(cos_i, 0.0, 1.0)) ** 5
        env = torch.tensor(env_color, dtype=torch.float32,
                           device=dev).view(1, 3, 1, 1) * env_strength
        acc = (1.0 - R) * acc + R * env
    return torch.clamp(acc, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)


def render_auto(img, h_mm, field_width_mm, **kw):
    """GPU when there is one, the numpy reference otherwise. Same physics either way."""
    if _torch_ok():
        return render_gpu(img, h_mm, field_width_mm, **kw)
    return render(img, h_mm, field_width_mm, **kw)


def jacobian_det(h_mm, field_width_mm, ior=N_WATER, depth_scale=1.0):
    """
    det of d(source)/d(output). NEGATIVE means the map has FOLDED there — the
    signature that produces multiple images of one feature, and the thing a
    smooth displacement filter can never do.
    """
    dxp, dyp, _ = refraction_offsets(h_mm, field_width_mm, ior=ior,
                                     depth_scale=depth_scale)
    sx = dxp + np.arange(h_mm.shape[1])[None, :]
    sy = dyp + np.arange(h_mm.shape[0])[:, None]
    sx_y, sx_x = np.gradient(sx)
    sy_y, sy_x = np.gradient(sy)
    return sx_x * sy_y - sx_y * sy_x


def trace_reference(h_mm, field_width_mm, iy, ix, ior=N_WATER, depth_scale=1.0):
    """
    INDEPENDENT vector-form ray trace for one pixel, kept as a live oracle
    against the closed form. Deliberately a different derivation path (vector
    Snell + explicit march to z=0) rather than the angle formulation — it is the
    only check that caught the original uphill/downhill sign error, because every
    self-consistent test passed either way.
    """
    mm_per_px = field_width_mm / h_mm.shape[1]
    h = np.maximum(h_mm * depth_scale, 0.0)
    gy, gx = np.gradient(h, mm_per_px)
    n = np.array([-gx[iy, ix], -gy[iy, ix], 1.0])
    n /= np.linalg.norm(n)
    d = np.array([0.0, 0.0, -1.0])
    eta = 1.0 / ior
    cosi = -np.dot(d, n)
    sin2t = eta * eta * (1.0 - cosi * cosi)
    t = eta * d + (eta * cosi - math.sqrt(max(1.0 - sin2t, 0.0))) * n
    t /= np.linalg.norm(t)
    if abs(t[2]) < 1e-12:
        return 0.0, 0.0
    march = h[iy, ix] / (-t[2])
    return (t[0] * march) / mm_per_px, (t[1] * march) / mm_per_px


# ===========================================================================
# Grain deficit
# ===========================================================================

def grain_deficit(img_shape, h_mm, field_width_mm, *, aperture_ratio=0.020,
                  samples=12, order=3, seed=21, smooth_px=6.0):
    """
    How much fine detail the warp destroyed, per pixel, in [0,1].

    MEASURED, not modelled: a unit-variance noise field is pushed through the
    IDENTICAL warp and its surviving local RMS is the retention map r(x), free of
    any contamination from picture content. The grain that has to be added back
    to restore the original level is then exactly sqrt(1 - r^2).

    HONEST LABEL: this is a RESTORATION of what our sampler and aperture removed.
    It is NOT a physical capture-grain layer and must not be described as one.
    Chain it into a grain node's strength — Film Grain Pro is the intended
    partner — rather than baking grain in here.
    """
    H, W = img_shape[:2]
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    probe = np.repeat(rng.normal(0.5, 0.05, (H, W, 1)), 3, axis=2)
    warped = render_auto(probe, h_mm, field_width_mm, aperture_ratio=aperture_ratio,
                         samples=samples, fresnel=False, seed=seed)

    def hf(a):
        g = a.mean(axis=2)
        hp = g - gaussian_filter(g, 1.2)
        return np.sqrt(np.maximum(gaussian_filter(hp * hp, 10.0), 0.0))

    # Normalise POINTWISE against the probe's own local RMS, not against its scalar
    # mean. White noise has a local RMS that fluctuates spatially, so dividing the
    # map by a single number leaves that fluctuation in r and manufactures a
    # deficit where nothing was warped at all — flat water measured a spurious 0.34
    # before this was fixed, which the "flat water destroys no grain" test caught.
    ref = hf(probe)
    r = np.clip(gaussian_filter(hf(warped) / np.maximum(ref, 1e-9), smooth_px),
                0.0, 1.0)
    return np.sqrt(np.maximum(1.0 - r * r, 0.0)).astype(np.float32)


def _hf_rms(a, sigma=1.2, win=10.0):
    g = a.mean(axis=2) if a.ndim == 3 else a
    hp = g - gaussian_filter(g, sigma)
    return np.sqrt(np.maximum(gaussian_filter(hp * hp, win), 0.0))


def restore_grain(rendered, source, deficit, amount=1.0, grain_size=1.2,
                  engine_strength=0.45, seed=5, device=None):
    """
    Put back the grain the warp destroyed, weighted by the measured deficit.

    THE CALIBRATION IS LOAD-BEARING and is why this cannot be left to the user
    chaining the mask by hand. `deficit` says what FRACTION of the source grain is
    missing; it knows nothing about how much grain the engine actually lays down at
    a given strength. Without dividing by the engine's measured contribution, a
    deficit of 0.9 means "90% of whatever the engine felt like adding" — measured
    at 1.89x the source grain when chained naively, against a target of 1.0x.

    So both quantities are measured here, in quadrature, and the blend weight is

        a(x) = deficit(x) * (source grain RMS / engine contribution RMS)

    Applied as a spatially varying STRENGTH on the engine's own output, so the
    engine's tone-dependence — real film grains peak in the midtones — survives
    rather than being flattened by scaling a texture.

    HONEST LABEL: a restoration of what our sampler and aperture removed. NOT a
    physical capture-grain layer, and it must not be described as one.
    """
    import torch

    from .grain_newson import render_film_grain

    t = torch.from_numpy(np.ascontiguousarray(rendered)).float()
    full = render_film_grain(t, grain_size=grain_size, radius_variation=0.0,
                             strength=engine_strength, color_grain=0.0,
                             n_samples=64, filter_sigma=0.8, seed=int(seed),
                             device=device).cpu().numpy().astype(np.float64)
    added = math.sqrt(max(float((_hf_rms(full) ** 2).mean()
                                - (_hf_rms(rendered) ** 2).mean()), 1e-12))
    src = float(_hf_rms(source).mean())
    a = np.clip(deficit.astype(np.float64) * (src / added) * float(amount), 0.0, 1.0)
    return np.clip(rendered + a[..., None] * (full - rendered), 0.0, 1.0)
