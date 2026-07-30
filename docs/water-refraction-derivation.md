# Water Refraction — depth-averaged free surface + exact Snell refraction

New flagship simulation node. Reference set: nine photographs by the artist who displays
his images on a tablet and pours water over the screen, shooting through the pour
(`C:/Users/Jeremie/Desktop/New folder/`). Those images are the acceptance target
throughout; every design call below is answerable to them.

STATUS: DERIVED, awaiting Jeremie's sign-off. Nothing built.

---

## What this is / is NOT

IS: a real fluid simulation whose free surface refracts the input image through exact
Snell's law. The warp is a *consequence* of a physical water surface, not a displacement
texture.

IS NOT: a displacement filter, a normal-map distort, or a noise warp. The distinguishing
behaviours are listed in §4 and each is a physical consequence, not an effect setting.

IS NOT: full 3D Navier–Stokes. See §5 for why depth-averaged is the apt model here and
not a compromise.

---

## 1. THE SCALE RESULT — LOAD-BEARING CALL 1, and it reshapes everything

A vertical camera ray refracting into water of depth h lands displaced by

    Δ = h · tan(θ_i − θ_t),     tan θ_i = |∇h|,   sin θ_t = sin θ_i / n_w

As the surface gets steeper, θ_i → 90° but **θ_t saturates at the critical angle**
arcsin(1/1.333) = 48.6°. So θ_i − θ_t → 41.4° and

    **Δ_max = 0.881 · h**

Lateral displacement can never exceed ~0.88 × the water depth, no matter how violent the
surface. Measured across the slope range:

| surface slope | θ_i | θ_i − θ_t | Δ / h |
|---|---|---|---|
| 0.10 | 5.7° | 1.43° | 0.025 |
| 0.50 | 26.6° | 6.96° | 0.122 |
| 1.00 | 45.0° | 12.96° | 0.230 |
| 2.00 | 63.4° | 21.3° | 0.390 |
| 5.00 | 78.7° | 31.3° | 0.609 |
| ∞ | 90° | 41.4° | **0.881** |

**Consequence: this is a MACRO effect.** The references show features displaced by
roughly 10–25% of frame width. Work backwards:

| field of view | h = 1 mm | h = 3 mm | h = 10 mm |
|---|---|---|---|
| 250 mm (whole tablet) | 3.6 px | 10.8 px | 36 px |
| 80 mm | 11.3 px | 33.8 px | 113 px |
| **40 mm** | 22.5 px | **67.6 px** | **225 px** |
| 20 mm | 45.1 px | 135 px | 451 px |

(displacement in px at a 1024 long edge)

At whole-tablet framing a 3 mm film moves the image by 11 px — invisible. The reference
look requires a **20–50 mm field of view**, i.e. the photographer is shooting close. That
is a real, checkable prediction of the model and it drives a required parameter:
`field_width_mm`, default **40**. Users who frame wide will correctly get almost nothing,
and the tooltip must say so or they will file it as a bug.

**The artistic control is `depth_scale`, not a fudge on Δ.** Anyone wanting more than
physics allows is really asking for deeper water, so the knob multiplies h and the optics
stay exact. Honest, and physically interpretable.

## 2. Optics — exact, not small-angle

Screen at z = 0 (flat, emissive), water surface at z = h(x,y), orthographic camera above.

    n̂ = (−h_x, −h_y, 1) / √(1 + |∇h|²)
    θ_i = atan|∇h|,   θ_t = asin(sin θ_i / n_w),   n_w = 1.333
    s(p) = p − h · tan(θ_i − θ_t) · ∇h/|∇h|

**Direction is UPHILL, along +∇h — CORRECTED AT SPIKE.** The signed-off draft said
downhill. The transmitted ray bends toward the *inward* normal −n̂, whose horizontal
component points up-slope; I had used the horizontal component of the *upward* normal n̂,
which points the other way. Caught by the independent vector-form ray trace kept as an
oracle (max error went from 11.8 px to 2e-14 px on the fix). Sanity check that settles it:
a bead is a local maximum, uphill is toward its centre, so sampling is pulled inward and
the image MAGNIFIES — a water drop on text magnifies. The old sign minified.

**Small-angle limit (a tooth, not the implementation):** as |∇h| → 0,
tan(θ_i − θ_t) → |∇h|(1 − 1/n_w) = 0.25|∇h|, giving s = p − ¼h∇h = p − ⅛∇(h²). We
implement the exact form — it is a handful of trig ops and the reference regime is steep,
where small-angle underestimates badly and misses the 0.881 saturation entirely.

**The map is a gradient map.** In the small-angle limit the displacement is ∇ of a
potential (∝ h²), and gradient maps are exactly the class whose generic singularities are
folds and cusps. That is *why* the references are full of sharp curved cusp lines. The
exact form perturbs this slightly but the singularity structure is inherited.

**No total internal reflection on the primary path.** The camera ray travels air → water,
into the denser medium, so it always refracts. TIR would need a ray arriving from beyond
90° in air. Worth stating because it is the obvious thing to reach for and it is wrong here.

**Fresnel — the brightness term, and the only spatially varying one.** Schlick with
R₀ = ((1−1.333)/(1+1.333))² = 0.0204:

    R(θ_i) = R₀ + (1 − R₀)(1 − cos θ_i)⁵
    L_out = (1 − R)·L_screen(s(p)) + R·L_env

Steep slopes transmit less and reflect more, which darkens fold lines and puts the silvery
highlights on them that the references show (`103234`, top area). L_env is a single
environment colour + strength — we have no environment map and inventing one would be
dishonest; the room is broadly one tone anyway.

**Absorption: negligible, and we will not fake it.** Water's absorption over 1–10 mm is
~10⁻⁴ of a stop. Any "water tint" would be invented. Stated so nobody adds one later.

**Dispersion: real but subtle.** n_w runs 1.3435 (400 nm) to 1.3311 (700 nm), so Δ varies
~3.7% across the visible band — a few px of colour fringing at a 100 px displacement.
Shipped as an optional toggle (three warps instead of one), default OFF on cost. The
references do not obviously show it.

**Radiance:** the n² radiance factor between water and air is a global constant, absorbed
into exposure. Not spatially varying, so not modelled.

## 2b. What actually folds — CORRECTED AT SPIKE, and it redirects the solver

The signed-off draft implied beads at the capillary length were the regime that produces
multiple images. Measurement says a **sessile bead can never fold**, at any size:

A cap of base radius r and height a has surface radius R = (r²+a²)/2a, so as a plano-convex
lens its focal length is f = R/(n−1) ≈ 3R. The screen sits only h = a below it, and for any
physically achievable cap a < f. The screen is always *inside* the focal length, so the bead
magnifies without ever inverting. Swept across aspect ratios 0.3→1.0 the folded area stays
exactly 0.00%. This is the everyday water-drop-on-text observation, and it is a real
prediction of the model rather than a limitation of it.

**Folding requires depth and slope to be INDEPENDENT**, which a bead cannot offer because
its own geometry ties them together. A pool can:

| pool depth | ripple amp / wavelength | folded area | peak disp @1024 |
|---|---|---|---|
| 2 mm | 0.8 mm / 6 mm | 0.00% | 11 px |
| 4 mm | 0.8 mm / 6 mm | 4.2% | 22 px |
| 8 mm | 0.8 mm / 6 mm | 25.6% | 43 px |
| 8 mm | 1.2 mm / 4 mm | 50.8% | 81 px |

Depth supplies the lever arm (Δ = h·tan(...)), sub-mm capillary ripples supply the slope.
A few mm of standing water carrying fine ripples folds readily; the *same ripples* on a
0.8 mm film do not (2.6% vs 34%). Asserted both ways as teeth.

**Consequence for the solver (§5): it must produce POOLING DEPTH, not just a wetted film.**
Static puddle depth caps near 2·l_c·sin(θ/2) = 2.7 mm at 60° contact angle, 4.5 mm at 110°
(a tablet's oleophobic coating sits at the high end); a live pour piles water transiently
above that, and a bezel holds more. So the target regime is 3–8 mm of water with capillary
ripples on it — which the depth-averaged solver delivers naturally, but only if pooling and
surface tension are both switched on.

## 3. Rendering — backward map, and where the real difficulty is

s(p) is a *backward* map: every output pixel has exactly one screen point, so rendering is
`out(p) = image(s(p))`. Two consequences worth being explicit about:

- **Folds are free.** Multiple images of one feature arise because s is non-injective —
  several output regions map to the same source region. Nothing special is needed to get
  the duplicated faces in `103234`/`103516`; they fall out.
- **Minification aliasing is the actual problem.** Where |det J| ≫ 1 a single output pixel
  covers a large source region, and point sampling turns it into crunch. The references are
  smooth in exactly those compressed regions. Mitigation: supersample the warp (2–3×) and
  box down; the fluid field is smooth so this costs only in the resample stage.

## 3b. Finite aperture — ADDED AT STAGE A, and it is required physics

The signed-off draft modelled a pinhole camera: one ray per output pixel. That is what
produced the hard black outlines and the "chrome / liquify" quality that failed Jeremie's
first eyeball. A real lens integrates a cone. By similar triangles the cone crosses the
water surface over a disc of radius

    ρ = (A / L) · h          A = aperture radius, L = camera-to-screen distance

Every point of that footprint sees a different surface slope and refracts to a different
screen point. Consequences, all of which match the references:

- **flat water stays sharp** and **dry screen is bit-exact untouched** (ρ = 0, Δ = 0),
  which is why `103441` has crisp grain everywhere except at its one meniscus line;
- **strongly curved water blurs**, in proportion to how fast Δ varies across ρ;
- **at a fold the footprint straddles several branches, so the multiple images BLEND**
  smoothly rather than butting together along a hard seam. This is the single change that
  turns the pinhole render's cut-and-paste look into the soft melted forms of the
  references.

**LOAD-BEARING:** the sample lands at `p + Δ(p + δ)`, NOT `p + δ + Δ(p + δ)`. Focusing
already converges the cone onto p; only the refractive deflection is sampled across the
footprint. The first implementation got this wrong and blurred flat water, which a
refocused camera does not do. Asserted as a tooth: flat water and dry screen must both
come back bit-exact (measured 8.9e-16).

A/L is one real photographic number: a 100 mm macro at f/8 has A = 6.25 mm, so at
L = 300 mm, A/L ≈ 0.021. Larger aperture = softer, dreamier water.

## 3c. What is NOT ours — established at Stage A

Comparing against his frames directly showed two differences that are not the water:

- **His grain is razor sharp over smooth forms.** Optical blur would blur the grain too.
  It cannot be in the source; it is added by the camera re-photographing the tablet,
  i.e. AFTER the distortion. Users chain Film Grain downstream; the node must not try to
  preserve or synthesise it.
- **His frames are hard-graded and tightly cropped.** Deep blacks, bright whites, and
  structures filling the frame. That is the grading stack and framing, not refraction.

Stated because chasing either inside this node would mean faking them.

## 4. Why it cannot read as a filter — the acceptance criteria

Each of these is a physical consequence, and each is a tooth:

1. **The map folds** → features appear multiple times. A smooth weak displacement field is
   effectively injective and never can.
2. **Displacement scales as h·tan(...)**, so it is bounded by depth and dominated by thick
   regions. Thin film ≈ untouched, bead ≈ violent. Heterogeneous by construction.
3. **A sharp wet/dry contact line** separates zero displacement from extreme (`103441` is
   nearly untouched apart from one straight meniscus line).
4. **Local magnification varies enormously** because |det J| does.
5. **The shapes are fluid shapes** — rivulets, beads, lobes, pour rims — because they come
   from a solver, not from noise.

## 5. Fluid solver — depth-averaged, on a grid

**LOAD-BEARING CALL 2: depth-averaged, not 3D — and it is the apt model, not a saving.**
The optics consume only h(x,y). A 3D FLIP would spend 10–100× the compute producing
internal velocity structure the renderer never reads, and produce the same picture. The
one thing 3D adds is overturning sheets where the surface is multivalued; no reference
frame shows that — every one is consistent with a height field.

Depth-averaged (shallow-water) equations:

    ∂h/∂t + ∇·(h u) = 0
    ∂u/∂t + (u·∇)u = −g∇(h + b) − c_f u/h + ν∇²u + (σ/ρ)∇(∇²h)

Cycle per step: particles carry mass and velocity → scatter to a MAC grid (h from kernel
density, u from momentum) → grid forces (gravity along tilt, hydrostatic pressure −g∇h,
bed drag, viscosity, surface tension) → gather back with the standard FLIP/PIC blend →
advect. `flip_ratio` is exposed: 0 = PIC (smooth, dissipative), 1 = FLIP (energetic, lively).

**Surface tension is not optional here.** The capillary length √(σ/ρg) = √(0.0728/(998·9.81))
= **2.7 mm**, and our depths are 1–10 mm. We sit *right at* the capillary length, which is
precisely why the references show both coherent sheets and discrete beads. Drop surface
tension and water spreads into a flat film that refracts almost nothing (§1).

**Stability / cost.** Explicit CFL: Δt < Δx/(|u| + √(gh)). Wave speed √(9.81·0.003) =
0.17 m/s; at 256² over a 40 mm field, Δx ≈ 0.16 mm, so Δt ≈ 0.9 ms. A 1.5 s pour is
~1700 steps of cheap grid ops — comfortably inside the 10–30 s budget at simulation
resolutions of 256–512.

**Sim resolution is decoupled from image resolution.** h is low-frequency, the image is
not. Simulate at 256–512², warp at full res. This is what makes the node affordable.

### 5b. FLIP/PIC replaced by a grid solver (2026-07-30) — the discretisation was the bug

The original discretisation was FLIP/PIC: particles carried mass and velocity,
h came from a kernel density estimate on the grid, and `flip_ratio` blended the
two transfer styles. That is now replaced by a finite-difference grid solver with
a conservative flux-form continuity update. Same equations, same constants, same
`h << L` assumption — only the representation of h changed.

**Why.** The dominant force is −g∇h, and h was an estimate from counting
particles. So sampling noise became force noise, which moved particles, which
changed the density: a feedback loop rather than a sampling error. Measured, a
flat 6.00 mm film with no dynamics at all reconstructed as **1.62–9.81 mm, 12.7%
noise**, and it got *worse* with 3× the particles (14.4%) instead of falling as
1/√N. `flip_ratio` barely moved it (12.7% at 0.95, 13.0% at 0.20). Since
refraction differentiates h, that noise was the dominant structure the optics saw
— so a great deal of what the look search was comparing was artifact.

**Result.** The same flat film now reconstructs at **6.00–6.00 mm, 0.00% noise**.
Mass conservation is exact rather than approximate (a central-difference draft
manufactured water: 8000 mm³ poured came back as 8848, which is why the flux form
and its per-cell outflow limiter exist). Runtime fell from 38 s to ~26 s at 1024.

**The cost, stated rather than buried.** Semi-Lagrangian advection smears, and
folding dropped from 15.1% to ~11%. That is a real loss, not rounding. It is
accepted because it buys a surface that is not 12% artifact, and because 11% is
far above the ~2% that would mean folding had been destroyed.

**What this does NOT change.** The identity claim is unaffected and arguably
stronger: the surface is still a simulated free surface, and now it is literally
the solution of the continuity equation rather than an estimate from counting
particles. LOAD-BEARING CALL 2 — depth-averaged rather than 3D — stands
untouched; only the discretisation beneath it moved.

## 6. Surface reconstruction — the honest weak point, handled

Shallow-water assumes h ≪ L. At the capillary length our fattest beads have h ~ L, so the
assumption is marginal exactly where the optics are most dramatic. Rather than pretend
otherwise: the solver handles **transport** (where water goes, how it beads, rivulets, runs
off), and a separate reconstruction step builds the rendered surface from the particle
distribution using a kernel at the capillary scale, so beads come out as rounded caps with
physically sized curvature instead of depth-averaged plateaus. The optics then act on that.

Recorded as a residual, not hidden: the transport in fat beads is approximate.

## 7. Units — LOAD-BEARING CALL 3, a deliberate departure from the no-microns rule

Halation, Eberhard and Film Damage all use ref-px @1024 and refuse a physical round trip,
because in those the physical scale bought nothing. **Here it is the entire point.** Δ
depends on absolute depth: 1 mm and 5 mm of water are qualitatively different looks (§1).
So the solver runs in SI (metres, seconds, real g, σ, ρ, ν) and `field_width_mm` is the
bridge to pixels. Anyone who "fixes" this to ref-px destroys the model.

## 8. Controls (final at spike)

*Scale & optics:* `field_width_mm` (40), `depth_scale` (1.0), `water_ior` (1.333),
`fresnel_strength`, `env_color`, `env_strength`, `dispersion` (off), `supersample` (2).

*Pour:* `spawn_x`, `spawn_y`, `pour_rate`, `pour_radius_mm`, `pour_velocity`,
`pour_duration`, `initial_film_mm`.

*Dynamics:* `sim_time`, `tilt_angle`, `tilt_direction`, `friction`, `viscosity`,
`surface_tension`, `flip_ratio`, `edge_behavior` (open/wall), `sim_resolution`,
`particles_per_cell`, `cfl_safety`, `seed`.

*Output:* `frame_count` (1 = still, >1 = batch animation), `frame_interval`.

Presets to tame the surface: *pour*, *flood*, *droplets*, *run-off*.

Outputs: `IMAGE` (or batch), `MASK` (wet area), and `IMAGE` height-field preview for
debugging — the last one is how anyone diagnoses a bad result.

## 9. Build sequence — TWO-STAGE SPIKE (the de-risking that matters)

The solver is the expensive part and the optics are the risky part, so they get spiked in
that order — optics first, cheaply.

- **Stage A — optics only, analytic h.** Feed synthetic height fields (Gaussian beads, a
  rivulet, a pour rim, a standing wave) into the exact refraction + render. Question: does
  it produce reference-like folds, multiple images, cusp lines and magnification variance?
  **Jeremie's eyeball on Stage A before a single line of solver is written.** If the optics
  cannot reach the references with a hand-made surface, no solver will save it, and we stop
  having spent almost nothing.
- **Stage B — the fluid solver**, only after Stage A passes. Its own checks (mass
  conservation, CFL stability, capillary scale, contact line) and its own eyeball.

## 10. Teeth

- Snell exactness vs an independent VECTOR-form ray trace, kept permanently as an oracle
  (it is what caught the sign error); small-angle limit recovered as |∇h| → 0.
- A sessile bead does NOT fold; a pool with ripples DOES; the same ripples on a shallow
  film do not. All three asserted — the fold is the whole moat, so it is pinned from
  several directions.
- Δ_max = 0.881·h asserted at extreme slope (the saturation is the signature).
- Flat water (h const) → bit-exact passthrough; h = 0 → passthrough.
- Fold detection: a bead of known curvature produces a non-injective map and a duplicated
  feature. NEGATIVE CONTROL: a weak smooth field stays injective.
- Mass conservation in the solver to within particle-count tolerance.
- CFL respected; no NaN over a long sim.
- Capillary length reproduced: bead size scales with √(σ/ρg) when σ is varied.
- Resolution independence: sim at 256 vs 512 gives the same gross behaviour.
- Determinism under seed; batch frames advance in time.
- Perf budget at 1024² and 4K.

## 10b. Stage A result (2026-07-28)

PASSED, 15/15 checks. Optics alone: 0.68 s at 1024² without supersampling, 2.92 s with
supersample 2, 11.4 s at 2048², 6.65 s with dispersion on. The solver budget sits on top
of that, so supersample and sim resolution are the two quality/speed dials.

Exhibits in `_water_refraction_spike/`. The field-of-view ladder renders the §1 prediction
directly: at 250 mm the frame is untouched, at 25 mm it is unrecognisable.

My own eyeball flagged three gaps in the hand-made surfaces, all of which are exactly what
Stage B supplies: the warp is too UNIFORM (real pours leave dry and thin regions next to
violent ones), the ripples are too PERIODIC (a sum of sinusoids reads procedural; real
capillary waves radiate irregularly from the impact), and heavily compressed regions show
faint ringing (supersample 2 is marginal there). None of these are optics failures.

## 10c. Stage A verdict (2026-07-28, second round)

Jeremie's first eyeball FAILED the pinhole renders ("none of the effects look visually
pleasing or close"). Diagnosis, then a clean second round:

- A statistical comparison against the nine references (structure-tensor coherence,
  radially-averaged spectral centroid, contrast) DISCONFIRMED my own diagnosis: the
  renders already sat inside the reference range, and the undistorted source scored the
  highest coherence of all. Those metrics are dominated by image content, not distortion.
- The real miss was §3b, the pinhole assumption.
- Second round renders his OWN source frame (`103441`, near-undistorted, same series as
  the distorted `103130`) so the source-image confound is removed, using irregular
  deep-pool surfaces with genuine dry regions and a contact line.

Result: markedly closer, and in the reference family — readable subject, soft melted
forms, sharp dry regions, real contact line (`STAGEA_A.png`, `STAGEA_E.png`). Not a
match. The residual tell is surface CHARACTER: multi-octave noise quilts into a faintly
cellular texture and the dry regions are not truly dry. That is precisely what a solver
supplies and a hand-made field structurally cannot, which is the Stage B question.

## 10d. Stage B result (2026-07-28) — solver BUILT, two bugs found, two problems open

`_water_refraction_spike/solver.py`. Depth-averaged FLIP/PIC in real units, per §5.
VERIFIED: mass conserves exactly (560 mm³ poured, 560 mm³ present); capillary length
falls out of the constants at 2.727 mm; the contact line emerges from the `3νu/h²` bed
drag rather than being thresholded in.

Reaches reference-scale numbers from simulated water: **226–280 px displacement @1024,
50%+ of frame folded**, at ~14–24 mm depth on a 40 mm field, which independently matches
the depth table in §1.

**Bug 1 — output-side smoothing erased the physics.** `height()` blurred the surface by a
full capillary length (7.6 cells at dx = 0.36 mm), on the reasoning that surface tension
cannot sustain finer curvature. But surface tension is ALREADY in the momentum equation,
damping curvature dynamically. Correcting the reconstruction blur to what it should always
have been — ~2 cells, enough to clean particle-density noise — moved the SAME simulation
from slope 0.52 → 2.28, displacement 60 px → 208 px, folding 0.0% → 52.9%. Reconstruction
filtering is measured in CELLS and exists to remove discretisation noise; physical damping
belongs in the equations.

**Bug 2 — density noise became visible structure.** The optics take ∇h, so per-cell FLIP
density noise (~5 particles/cell ⇒ ~45% per-cell error) was amplified into a cellular
soap-foam texture over the whole frame. Fixed by reconstructing density on a quadratic
B-spline kernel (3×3 support) instead of bilinear (2×2): removes 1-cell noise, preserves
3+ cell structure. Post-blur cannot do this — that is Bug 1 again.

**OPEN 1 — performance.** ~320 s against the 10–30 s target, >10× over. Not algorithmic
mystery, ordinary engineering: the scatter/gather and grid ops want torch on GPU, and/or a
semi-implicit step to break the explicit CFL limit (dt is currently ~90 µs, driven by
√(gh) ≈ 485 mm/s at 24 mm depth plus pour velocity). A `prefill()` (start from standing
water rather than filling an empty tray) is implemented and cuts the transient, but its
gain was eaten by simultaneously raising particle count and grid resolution.

**OPEN 2 — the aesthetic band.** RESOLVED, see §10e. The framing in this paragraph was
itself the blocker: "too tame" and "too strong" both describe an AMOUNT, and the miss was
structure scale.

## 10e. The aesthetic band, located (2026-07-28 PM)

Settled with matched-control OPTICS ladders on his own near-undistorted source frame
(`103441`), no solver involved, ~10 s per render. `_water_refraction_spike/scale_probe.py`,
`char_probe.py`, `band_probe.py`, `damping_probe.py`, `final_probe.py`.

**The render is a function of h/W alone.** On normalised coordinates x̂ = x/W,
Δ/W = ĥ·tan(θ_i − θ_t) with tan θ_i = |∇_x̂ ĥ|. So "40 mm field with 20 mm of water" and
"15 mm with 7.5 mm" are literally the same picture, and only two dimensionless numbers
matter: the structure wavelength λ/W and the RMS slope. That is what makes a controlled
ladder possible at all.

**It was structure scale.** Laddering λ/W with the p95 displacement held CONSTANT in every
cell (bisecting depth): λ/W = 0.06 is confetti, λ/W = 0.30 is in his family, at identical
displacement. The solver had been running at ~0.06, because §1 fixed `field_width_mm = 40`
and then pushed depth to 14–25 mm to reach displacement — which puts the capillary length
(2.727 mm, the scale that dominates surface SLOPE and therefore the optics) at 6.8% of
frame. The same regime gives kh ≈ 9: **the depth-averaged model was being run in the
deep-water regime for exactly the structures doing the visual work**, where its dispersion
relation is wrong. Recorded as a validity problem, not patched.

**The legibility anchor is FLAT water, not dry screen.** Δ vanishes with the SLOPE, so calm
pool displaces nothing however deep it is and stays perfectly sharp — that is the readable
hand in `103130`, with no contact line required. A dry-region experiment failed usefully: a
9 mm pool with a free edge is unphysical (surface tension caps a free puddle near 4 mm) and
the mask edge rendered as a network of glowing pipes. Calm-next-to-violent is what an
unequilibrated pour gives and what a prefilled, settled tray structurally cannot.

**The chop is absent because of viscous damping, not shutter blur.** γ(k) = 2νk² gives
6.5× lobe/chop selectivity at 300 ms of settling; a 1/125 s shutter gives 5.8× but costs a
measured 2.3× in acutance, and his melted forms are sharp. That also explains the Stage B
renders directly: they sampled at t = 24–90 ms, inside the 72 ms chop lifetime. The shutter
mechanism is real and remains available as an optional control, but it is not the
explanation.

**The residual tension, stated not tuned away.** The chop had been supplying most of the
p95 displacement, because it is steep. Remove it and smooth lobes alone need d/W ≈ 0.7 —
28 mm of water on a 40 mm field — to reach 12% of frame width. So either the water is deep
and CONTAINED (tray or bezel, no free edge in frame), or the field of view is much narrower
than 40 mm, or his displacement is smaller than the §1 estimate. That is a question about
his physical setup, still open.

Two of my own operating points were corrected by the teeth written to check them: the
selective shutter is 1/125 s not 1/250 s, and the settling time 300 ms not 150 ms — both a
factor ~2 optimistic, because a lifetime or period gives the SCALE of an answer and never
the operating point.

## 10f. Grain — measured, and the adopted fix (2026-07-28 PM)

Jeremie's note on the candidate: the refraction smooths out film grain, and adding grain
back would double the noise already present. Both halves right.
`grain_probe.py`, `grain_origin_probe.py`, `chain_grain_probe.py`, `grain_match_probe.py`.

Three separate losses, measured on a flat grey field carrying known grain so that every
high-frequency thing in the output is grain:

1. **Free damage.** `map_coordinates(order=1)` retains Σw² of a white-noise variance, and
   E[Σw²] = (2/3)² over random sub-pixel positions — a flat 33% grain loss on every
   displaced pixel regardless of the water. Measured 0.600 retention where |det J| is
   within 5% of 1. **order=3 gives 0.829, +38%, at no cost. Adopted.**
2. **A correctness gap.** Real minification predicts retention = 1/√|det J| (independent
   grains averaging), but measured retention is FLAT at 0.79–0.86 out to |det J| > 4: an
   interpolator point-samples, it does not area-average, so compressed regions ALIAS.
   Needs a mip/EWA path. Open; likely also feeding the crunchy speckle in the darks.
3. **Real physics.** The finite aperture is the single biggest loss (0.836 → 0.460) and
   should stay.

**Chaining a uniform grain layer cannot fix this, and the reason is structural.** A uniform
added layer combines in quadrature with a NON-uniform survivor, so it can only dilute the
difference, never equalise it. Measured against the shipped Film Grain Pro (uniformity =
working/calm grain, source = 0.97): strength 0.12 → 0.75 at 1.09× total grain, 0.20 → 0.84
at 1.33×, 0.30 → 0.90 at 1.73×, 0.50 → 0.94 at 2.65×. Full uniformity is only reachable by
drowning the frame.

**ADOPTED (Jeremie, 2026-07-28): deficit-compensated restoration.** The retention map is
MEASURED, not modelled — push a unit-variance noise field through the identical warp and
its local RMS *is* r(x), free of content contamination. Then

    a(x) = √(max(1 − r(x)², 0)) · (source grain amplitude / engine contribution at S)

applied as a spatially varying strength on the grain engine's own output,
`out = base + a(x)·(FGP(base) − base)`, so the engine's tone-dependence (film grains peak
in the midtones) is preserved rather than flattened by scaling a texture. Measured
retention 0.927 calm / 0.359 working; result **uniformity 1.04 and total grain 1.01× the
source**, against 0.84 / 1.33× for the best uniform chain. The calibration divisor is
load-bearing: without it the deficit means "90% of whatever the engine felt like adding"
and the first run overshot to 1.67.

Product shape: Water Refraction emits a **grain-deficit MASK**, chained into a grain node's
strength. House pattern — Rebate, Damage and Leak all output masks.

**Honest label, to carry into the node doc:** this is a RESTORATION of what our sampler and
aperture removed. It is not a physical capture-grain layer and must not be sold as one.

Grain SIZE was tested and is NOT a factor: the source's grain correlation radius is 1.71 px
and Film Grain Pro at its default `grain_size = 1.20` produces 1.70 px.

Whether the grain in HIS references sits before or after the water is **unresolved**. The
test built for it is invalid — a cross-image texture statistic, where the ungrained control
moves the number more than the effect does. See patterns.md 2026-07-28.

## 11. Honest residuals

- **The depth-averaged model is used outside its validity envelope** at the settings that
  produce the look: kh ≈ 9 for the structures doing the visual work (§10e). Shallow water
  needs kh ≪ 1. Its dispersion is therefore wrong for those scales, which may be why the
  solver produced a cellular quilt — short waves that should disperse instead steepen.
  Named, not fixed.
- **Minification aliases rather than area-averages** (§10f). An interpolating sampler
  point-samples; a mip/EWA path is needed for the compressed regions to be correct.
- **Which physical setup the references come from is undecided** (§10e): deep contained
  water, a much narrower field of view, or a smaller displacement than §1 estimated.
- Shallow-water is marginal for fat beads (§6); transport there is approximate.
- Orthographic camera. A real lens adds a small linear ray tilt across frame; not modelled.
- Contact-angle hysteresis and wetting are a drag/pinning heuristic, not a measured
  contact-angle model.
- The environment reflection is one colour, not an environment map.
- Dispersion off by default despite being real.
- The screen is treated as a Lambertian emitter; a real LCD has a polariser and an angular
  emission profile, which would slightly darken steep viewing angles beyond Fresnel.
- No air entrainment or bubbles.
