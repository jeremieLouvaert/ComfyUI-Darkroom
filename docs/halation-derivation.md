# Halation — physically derived annular model (supersedes the v1 threshold-blur node)

Status: FROZEN 2026-07-23. Math signed off (Jeremie, "good to go") + spike ALL CHECKS PASSED on embedded python (ring peak exact at r_c, rim/center 7.46 in the predicted 3-8x band, negative control fires) + spike eyeball pass (Jeremie, "looks correct at first sight", `_halation_spike/EYE_*`). Build implements this doc with zero deviations.

## What this is / is NOT

IS: a physical model of film-base halation — light from bright image regions
transmitting through the film base, totally-internally-reflecting off the rear
surface, and re-exposing the emulsion in an ANNULUS around each source. The
CineStill look, from the actual geometry that causes it.

IS NOT: a glow. The old `DarkroomHalation` (threshold → blur → tint → screen in
sRGB) is a fake of the class this model replaces. The structural difference: a
Gaussian glow has its maximum AT the source; real halation has a rim standing
OFF the source at a predictable radius. This node is also NOT the parked
aging/fog family — it is an in-camera exposure effect, not a degradation look.

## PRODUCTION MODEL

### 1. Ring kernel P_TIR — exact derivation

Geometry: a point exposure at the emulsion plane scatters light diffusely into
the base (thickness d, refractive index n, internal). Treat the emulsion as a
Lambertian diffuser into the downward hemisphere. A ray at polar angle θ (from
the normal) reaches the rear base surface, reflects with reflectance R(θ), and
returns to the emulsion plane at lateral radius

    r(θ) = 2 d tanθ.

Power emitted into the annulus [θ, θ+dθ] (Lambertian: radiance constant,
projected-solid-angle weighting):

    dP ∝ cosθ · sinθ dθ.

Returned energy per unit radius:

    dP_ret/dr = R(θ) · cosθ sinθ / (dr/dθ),   dr/dθ = 2 d sec²θ,

so  dP_ret/dr ∝ R(θ) · sinθ cos³θ / (2d).

Energy per unit AREA (divide by the annulus circumference 2πr, r = 2d tanθ):

    PSF(r) ∝ R(θ) · sinθ cos³θ / r
           = R(θ) · [t/(1+t²)²] · 1/(2dt)        with t = tanθ = r/(2d)

    ┌─────────────────────────────────────────────┐
    │  P_TIR(r) ∝ R(θ(r)) / (1 + (r/2d)²)²        │
    └─────────────────────────────────────────────┘

One clean closed form. Its structure produces the ring:

- R(θ) is the unpolarized Fresnel reflectance at the base→air interface
  (n → 1): R(0) = ((n−1)/(n+1))² ≈ 0.04 for n = 1.5, rising slowly, then
  **jumping to R = 1 at the critical angle θ_c = asin(1/n)** (total internal
  reflection).
- Below r_c = 2d tanθ_c = 2d/√(n²−1): the inner disk is Fresnel-suppressed
  (~4-10% leak).
- At r_c: PSF jumps by ~1/R(θ_c⁻) — a sharp bright inner edge. For n = 1.5 the
  geometric factor at the ring is 1/(1+tan²θ_c)² = cos⁴θ_c ≈ 0.31 of the
  central unattenuated value, but the center IS attenuated to ~0.04-0.1 → the
  ring rim is ~3-8× the central leak. That is the annulus.
- Beyond r_c: smooth decay, asymptotically (2d/r)⁴.

Energy converges (∫PSF·2πr dr finite); kernel is normalized numerically.

Neglected at first order (v1.x candidates): second bounce (the returning ray
partially re-reflects at the emulsion-base interface, ~4%, producing a faint
ring at 2r_c); in-base absorption shaping (folded into w_c instead).

### 2. Units — LOAD-BEARING CALL 1 (the no-microns rule, Eberhard precedent)

d is never instantiated. The user knob `ring_radius` (ref-px @ 1024, scaled by
long_edge/1024, house convention) IS r_c; internally 2d ≡ r_c·√(n²−1) with
n = 1.5 fixed. The physics contributes the profile SHAPE (Fresnel floor, TIR
jump, quartic tail), not an absolute scale.
Sanity anchor only: 125 µm triacetate → r_c ≈ 224 µm ≈ 25 px on a 4K 35mm scan
≈ 6 ref-px @1024... default `ring_radius = 8`, range 3-60.
(Correction from plan draft: 224µm/36mm ≈ 0.62% of frame width; at 1024 long
edge ≈ 6.4 px. Default 8 splits realistic-35mm and visible-taste territory.)

### 3. Full per-channel kernel

    PSF_c(r) = w_c · [ a · P_TIR(r)  +  (1−a) · T(r) ],
    T(r) = exp(−r/L) / (2πL(r+ε))     (diffuse multiple-scattering tail,
                                       James Ch. 20 empirical spread class)

- a = `ring_tail_balance` (default 0.65 — ring-dominant).
- L = `tail_length` (ref-px @1024, default 40).
- Both terms unit-normalized before mixing; the mixed kernel re-normalized so
  ∫PSF_c = w_c ≤ 1.

### 4. Channel weights w_c — derived ORDERING, preset VALUES

Returning light crosses the overlying emulsion layers twice. In color negative
stock the layer order (top→bottom: blue-sensitive, yellow filter,
green-sensitive, red-sensitive) means blue is absorbed hardest on the round
trip and red barely at all:

    w_R > w_G > w_B   — this ordering is the derived, non-negotiable physics
                        (it is WHY halation is red-orange).

Exact values need vendored dye spectra we don't have → HONEST preset territory:

| preset | w_R | w_G | w_B | notes |
|---|---|---|---|---|
| CineStill 800T | 1.00 | 0.32 | 0.10 | strong red-orange, tungsten signature |
| Vision3 subtle | 1.00 | 0.45 | 0.18 | AH present → pair with ah_strength 0.7 |
| B&W plate | 1.00 | 1.00 | 1.00 | panchromatic mono halo (old glass plates) |
| Custom | user RGB | | | free |

### 5. Anti-halation layer

AH (remjet / AH undercoat) absorbs before AND after the rear reflection:
    halo_energy ← halo_energy · (1 − ah_strength)².
ah_strength = 0 → CineStill (remjet removed). 1 → modern stock, halo dead.

### 6. Application — LOAD-BEARING CALL 2 (additive in linear, NOT ratio-recombine)

    lin   = srgb_to_linear(img)                       (utils/color.py)
    E_c   = max(lin_c − knee, 0) / (1 − knee)         (knee default 0 = physical:
                                                       ALL light halates)
    H_c   = PSF_c ⊗ E_c                               (scipy fftconvolve per
                                                       channel, utils/image.py
                                                       pattern; new halation_psf
                                                       builder next to disk_kernel)
    out   = clip(lin + strength · H, 0, 1)
    img'  = linear_to_srgb(out)

Halation ADDS colored exposure — the hue shift toward the halo tint IS the
look. The house hue-preserving ratio-recombine is deliberately NOT used here
(documented deviation; the physics is additive re-exposure).
`strength` maps UI 0-1 → halation fraction 0-0.25 internally (real halation is
single-digit-% of incident energy; 0.25 headroom for taste).

### 7. Controls table (final at spike)

| widget | type | default | range | note |
|---|---|---|---|---|
| preset | combo | CineStill 800T | 4 presets | sets w_c + ah + balance |
| strength | FLOAT | 0.35 | 0-1 | 0 → early-exit passthrough |
| ring_radius | FLOAT | 8.0 | 3-60 | ref-px @1024 = r_c |
| tail_length | FLOAT | 40.0 | 5-200 | ref-px @1024 = L |
| ring_tail_balance | FLOAT | 0.65 | 0-1 | 1 = pure ring |
| ah_strength | FLOAT | 0.0 | 0-1 | (1−x)² energy scale |
| halo_r/g/b | FLOAT ×3 | preset | 0-1 | exposed on Custom |
| highlight_knee | FLOAT | 0.0 | 0-0.95 | 0 = fully physical |

## Color-space flow
sRGB in → linear (house helpers) → convolve+add in linear → clip → sRGB out →
`blend(original, out, strength)` NOT used (strength is already the physical
fraction; double-dipping would square the control). Early-exit strength ≤ 0.

## Teeth (promoted to tools/test_halation.py at build)
As frozen in the plan: ring-peak location vs r_c prediction (+gaussian negative
control), energy conservation, w ordering (+flipped negative control),
ah=1 kill / strength=0 bit-exact passthrough, 512-vs-2048 resolution
independence (proportional synthetic, 15%), strength monotonicity, perf gate
1024² + 4K.

## LOAD-BEARING CALLS
1. No microns — ring_radius IS r_c in ref-px @1024; physics fixes shape only.
2. Additive-in-linear composite; ratio-recombine deliberately not used.
3. Supersede-in-place — same node key `DarkroomHalation`; old threshold-blur
   model deleted; widget `strength` retained, other old widgets dropped.
4. Lambertian entry assumption — real emulsion scatter is forward-biased;
   Lambertian is the parameter-free worst case and only softens the ring
   (never removes it). Ring contrast is therefore a conservative floor.

## Honest residuals
- w_c values are presets (ordering derived, magnitudes not) — dye spectra not
  vendored.
- Single bounce only; faint 2r_c ring real but deferred.
- Halation of light OUTSIDE the frame (bright sources just off-frame halate
  in) not modeled — kernel only sees in-frame pixels.
- Scanner/print veiling flare not included (different mechanism, maybe a Lens
  node someday).
