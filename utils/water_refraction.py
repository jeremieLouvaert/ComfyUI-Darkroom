"""
Water Refraction engine — exact Snell through a simulated free surface.

Full derivation: docs/water-refraction-derivation.md. What makes this not a
displacement filter, in order of how much each carries:

  1. The warp is a CONSEQUENCE of a simulated water surface, not a texture. A
     depth-averaged FLIP/PIC solver pours real water onto the screen; the optics
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


# ===========================================================================
# Fluid solver — depth-averaged FLIP/PIC
# ===========================================================================

class ShallowWaterFLIP:
    """
    Depth-averaged (shallow water) FLIP/PIC:

        dh/dt + div(h u) = 0
        du/dt + (u.grad)u = -g grad(h+b) - 3 nu u / h^2 + nu lap(u)
                            + (sigma/rho) grad(lap h)

    There is NO incompressibility constraint and therefore no pressure
    projection: in shallow water the hydrostatic term -g grad(h) IS the pressure
    gradient and h is free to vary. That is what makes this affordable.

    The bed drag -3 nu u / h^2 is the laminar film result, not an ad-hoc pin. It
    diverges as h -> 0, so thin films pin themselves and the contact line emerges
    from the equations instead of being thresholded in.
    """

    def __init__(self, field_w_mm, field_h_mm, nx=112, seed=0,
                 tilt_deg=0.0, tilt_dir_deg=90.0, friction=1.0, viscosity=1.0,
                 surface_tension=1.0, flip_ratio=0.95, edge="wall"):
        self.W = float(field_w_mm)
        self.H = float(field_h_mm)
        self.nx = int(nx)
        self.ny = max(2, int(round(nx * self.H / self.W)))
        self.dx = self.W / self.nx
        self.cell_area = self.dx * self.dx
        self.rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

        self.flip_ratio = float(np.clip(flip_ratio, 0.0, 1.0))
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

        self.px = np.zeros(0)
        self.py = np.zeros(0)
        self.pu = np.zeros(0)
        self.pv = np.zeros(0)
        self.vol = 1.0
        self.t = 0.0

    # -- particles ----------------------------------------------------------

    def prefill(self, depth_mm, n):
        n = int(n)
        if n <= 0 or depth_mm <= 0:
            return
        self.px = np.concatenate([self.px, self.rng.random(n) * self.W])
        self.py = np.concatenate([self.py, self.rng.random(n) * self.H])
        self.pu = np.concatenate([self.pu, np.zeros(n)])
        self.pv = np.concatenate([self.pv, np.zeros(n)])

    def spawn(self, n, cx_mm, cy_mm, radius_mm, speed_mm_s):
        """
        Inject a pour. Water arriving from above converts its fall speed into
        radial outflow at the impact point (hydraulic-jump behaviour), producing
        the spreading bore and its raised rim.
        """
        n = int(n)
        if n <= 0:
            return
        ang = self.rng.random(n) * 2.0 * math.pi
        rad = radius_mm * np.sqrt(self.rng.random(n))
        rr = np.maximum(rad, 1e-6)
        ux = speed_mm_s * (rad / rr) * np.cos(ang) + self.rng.normal(0.0, speed_mm_s * 0.25, n)
        uy = speed_mm_s * (rad / rr) * np.sin(ang) + self.rng.normal(0.0, speed_mm_s * 0.25, n)
        self.px = np.concatenate([self.px, cx_mm + rad * np.cos(ang)])
        self.py = np.concatenate([self.py, cy_mm + rad * np.sin(ang)])
        self.pu = np.concatenate([self.pu, ux])
        self.pv = np.concatenate([self.pv, uy])

    # -- transfers ----------------------------------------------------------

    def _idx(self):
        """
        Bilinear stencil, computed ONCE per step and shared by the scatter and
        all four gathers. It used to be rebuilt five times on identical
        positions; hoisting it is bit-identical and measurably faster.
        """
        gx = np.clip(self.px / self.dx, 0.0, self.nx - 1.001)
        gy = np.clip(self.py / self.dx, 0.0, self.ny - 1.001)
        i0 = gx.astype(np.int64)
        j0 = gy.astype(np.int64)
        fx = gx - i0
        fy = gy - j0
        i1 = np.minimum(i0 + 1, self.nx - 1)
        j1 = np.minimum(j0 + 1, self.ny - 1)
        return (i0, j0, i1, j1,
                (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)

    def _p2g(self, idx=None):
        i0, j0, i1, j1, w00, w10, w01, w11 = idx if idx is not None else self._idx()
        n = self.nx * self.ny
        flat = np.concatenate([j0 * self.nx + i0, j0 * self.nx + i1,
                               j1 * self.nx + i0, j1 * self.nx + i1])
        w = np.concatenate([w00, w10, w01, w11])
        mass = np.bincount(flat, weights=w, minlength=n).astype(np.float64)
        mu = np.bincount(flat, weights=w * np.tile(self.pu, 4), minlength=n)
        mv = np.bincount(flat, weights=w * np.tile(self.pv, 4), minlength=n)
        mass = mass.reshape(self.ny, self.nx)
        u = np.zeros_like(mass)
        v = np.zeros_like(mass)
        nz = mass > 1e-12
        u[nz] = mu.reshape(self.ny, self.nx)[nz] / mass[nz]
        v[nz] = mv.reshape(self.ny, self.nx)[nz] / mass[nz]
        return mass * self.vol / self.cell_area, u, v

    def _g2p(self, arr, idx=None):
        i0, j0, i1, j1, w00, w10, w01, w11 = idx if idx is not None else self._idx()
        return (w00 * arr[j0, i0] + w10 * arr[j0, i1]
                + w01 * arr[j1, i0] + w11 * arr[j1, i1])

    # -- one step -----------------------------------------------------------

    def cfl_dt(self, h, u, v, safety=0.35, dt_max=2e-3):
        wave = math.sqrt(self.g_normal * max(float(h.max()), 1e-6))
        vel = float(np.hypot(u, v).max()) if u.size else 0.0
        return min(dt_max, safety * self.dx / max(wave + vel, 1e-6))

    def step(self, dt=None, precomputed=None):
        if self.px.size == 0:
            return 0.0
        if precomputed is not None:
            idx, h, u, v = precomputed
        else:
            idx = self._idx()
            h, u, v = self._p2g(idx)
        if dt is None:
            dt = self.cfl_dt(h, u, v)

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
        u_new = u + dt * ax
        v_new = v + dt * ay

        du = self._g2p(u_new - u, idx)
        dv = self._g2p(v_new - v, idx)
        pu_pic = self._g2p(u_new, idx)
        pv_pic = self._g2p(v_new, idx)
        r = self.flip_ratio
        self.pu = r * (self.pu + du) + (1.0 - r) * pu_pic
        self.pv = r * (self.pv + dv) + (1.0 - r) * pv_pic

        self.px += dt * self.pu
        self.py += dt * self.pv
        self._boundaries()
        self.t += dt
        return dt

    def _boundaries(self):
        if self.edge == "wall":
            for pos, vel, hi in ((self.px, self.pu, self.W), (self.py, self.pv, self.H)):
                lo_m = pos < 0.0
                hi_m = pos > hi
                pos[lo_m] = -pos[lo_m]
                vel[lo_m] *= -0.4
                pos[hi_m] = 2 * hi - pos[hi_m]
                vel[hi_m] *= -0.4
        else:
            keep = ((self.px > -2.0) & (self.px < self.W + 2.0)
                    & (self.py > -2.0) & (self.py < self.H + 2.0))
            if not keep.all():
                self.px = self.px[keep]; self.py = self.py[keep]
                self.pu = self.pu[keep]; self.pv = self.pv[keep]

    # -- output -------------------------------------------------------------

    def _density_bspline(self):
        """
        Density on a quadratic B-spline kernel (3x3 support) rather than bilinear.

        The optics take grad(h), so per-cell FLIP density noise is amplified into
        a visible cellular / soap-foam texture. A wider smooth kernel removes the
        1-cell noise while PRESERVING 3+ cell structure, which is where the real
        features live. Post-blurring instead would erase the physics too.
        """
        gx = np.clip(self.px / self.dx, 0.5, self.nx - 1.5)
        gy = np.clip(self.py / self.dx, 0.5, self.ny - 1.5)
        ib = np.round(gx).astype(np.int64)
        jb = np.round(gy).astype(np.int64)
        fx = gx - ib
        fy = gy - jb

        def wts(f):
            return (0.5 * (0.5 - f) ** 2, 0.75 - f * f, 0.5 * (0.5 + f) ** 2)

        wx, wy = wts(fx), wts(fy)
        idx_all, w_all = [], []
        for a in (-1, 0, 1):
            ii = np.clip(ib + a, 0, self.nx - 1)
            for b in (-1, 0, 1):
                jj = np.clip(jb + b, 0, self.ny - 1)
                idx_all.append(jj * self.nx + ii)
                w_all.append(wx[a + 1] * wy[b + 1])
        mass = np.bincount(np.concatenate(idx_all), weights=np.concatenate(w_all),
                           minlength=self.nx * self.ny)
        return mass.reshape(self.ny, self.nx) * self.vol / self.cell_area

    def height(self, smooth_cells=1.8):
        """
        Reconstruct the free surface.

        Smoothing here removes PARTICLE-DENSITY NOISE only and is therefore
        measured in CELLS, not in capillary lengths. An earlier version smoothed
        by a full capillary length on the reasoning that surface tension cannot
        sustain finer curvature — that conflated two things and erased the real
        structure (slope 2.28 -> 0.52, folding 52.9% -> 0.0% on the SAME sim).
        Surface tension is already in the momentum equation and damps curvature
        dynamically; it must not be re-imposed geometrically on the output.
        """
        h = self._density_bspline()
        return np.maximum(gaussian_filter(h, max(float(smooth_cells), 0.4)), 0.0)

    def volume_mm3(self):
        return self.px.size * self.vol


def sweep_path(s, w_mm, h_mm, sweep, angle_deg=45.0, wander=0.10):
    """
    Where the stream lands at normalised time s in [0,1].

    sweep = 0 is a stationary source, which generates a perfectly axisymmetric
    spreading disc — and a disc reads as a lens or a bubble, not as poured water.
    Nobody pours from a fixed point; the stream is swept across. Breaking the
    symmetry AT THE SOURCE is what works: tilting the plate was tried and does
    not, because it acts downstream and the pour's own momentum dominates over
    the interval that gets photographed.

    The span is calibrated so that sweep=1.0 at the default 45deg reproduces the
    approved spike path, which travelled 0.55 of the frame in BOTH axes. An
    earlier version of this function swept only 0.32 of the width, which
    concentrated the pour, piled it ~18mm deep and drifted back toward the very
    crater the sweep exists to prevent. Angles are taken in NORMALISED frame
    coordinates, not millimetres, so the path is the same fraction of the picture
    whatever the aspect ratio.
    """
    a = math.radians(angle_deg)
    cx, cy = 0.5 * w_mm, 0.5 * h_mm
    span = sweep * 0.778                      # 0.778 * cos(45deg) = 0.55 of frame
    ox = (s - 0.5) * span * math.cos(a) * w_mm
    oy = (s - 0.5) * span * math.sin(a) * h_mm
    if wander > 0.0 and sweep > 0.0:
        ox += wander * sweep * w_mm * 0.18 * math.sin(7.0 * s)
        oy += wander * sweep * h_mm * 0.12 * math.sin(4.5 * s + 1.0)
    return cx + ox, cy + oy


def simulate(field_w_mm, field_h_mm, *, volume_ml=16.0, nx=112, seed=0,
             sample_ms=80.0, pour_ms=100.0, pour_radius_mm=3.0,
             pour_speed_mm_s=700.0, sweep=1.0, sweep_angle_deg=45.0,
             initial_film_mm=0.0, tilt_deg=0.0, particles=110000,
             surface_tension=1.0, viscosity=1.0, flip_ratio=0.95,
             edge="wall", progress=None):
    """
    Pour water onto the screen and return the surface, in mm, on the sim grid.

    Sampled DURING the pour by default (sample < pour duration): the reference
    look lives in the live event, and letting it finish and spread costs both the
    depth that drives displacement and the structure that drives folding.
    """
    sim = ShallowWaterFLIP(field_w_mm, field_h_mm, nx=nx, seed=seed,
                           tilt_deg=tilt_deg, surface_tension=surface_tension,
                           viscosity=viscosity, flip_ratio=flip_ratio, edge=edge)
    film_mm3 = float(initial_film_mm) * field_w_mm * field_h_mm
    pour_mm3 = float(volume_ml) * 1000.0
    total = max(film_mm3 + pour_mm3, 1e-9)
    sim.vol = total / max(int(particles), 1)
    n_pre = int(film_mm3 / sim.vol)
    sim.prefill(initial_film_mm, n_pre)

    pour_s = max(pour_ms, 1e-3) / 1000.0
    end_s = max(sample_ms, 1e-3) / 1000.0
    rate = (int(particles) - n_pre) / pour_s

    carry, dt, steps = 0.0, 1e-4, 0
    while sim.t < end_s:
        if sim.t < pour_s:
            carry += rate * dt
            n = int(carry)
            carry -= n
            cx, cy = sweep_path(min(sim.t / pour_s, 1.0), field_w_mm, field_h_mm,
                                sweep, sweep_angle_deg)
            sim.spawn(n, cx, cy, pour_radius_mm, pour_speed_mm_s)
        if sim.px.size:
            idx = sim._idx()
            h, u, v = sim._p2g(idx)
            dt = sim.cfl_dt(h, u, v)
            sim.step(dt, precomputed=(idx, h, u, v))
        else:
            dt = 1e-4
            sim.t += dt
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
           order=3, seed=0):
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
        chan = np.zeros_like(img, dtype=np.float64)
        for _ in range(max(1, int(samples))):
            ang = 2.0 * math.pi * rng.random()
            rad = math.sqrt(rng.random())
            qy = ys0 + rad * math.sin(ang) * rho
            qx = xs0 + rad * math.cos(ang) * rho
            dxs = map_coordinates(dxp, [qy, qx], order=1, mode="nearest")
            dys = map_coordinates(dyp, [qy, qx], order=1, mode="nearest")
            chan += _sample(img, ys0 + dys, xs0 + dxs, order=order)
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
    warped = render(probe, h_mm, field_width_mm, aperture_ratio=aperture_ratio,
                    samples=samples, fresnel=False, order=order, seed=seed)

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
