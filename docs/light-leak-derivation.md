# Light Leak — path-length derivation (sprocket trap, gradient, pinhole)

Film-damage wave, build 3/3 (after Film Rebate and Film Damage).
Sweep basis: `comfyui-brain/discover/2026-07-25_film_damage_sweep.md` §2, scope call
`decisions.md [2026-07-25]`.

STATUS: DERIVED, awaiting Jeremie's sign-off. Nothing built.

---

## What this is / is NOT

IS: stray light that reaches the film outside the lens path, modelled as an **exposure
event**. One mechanism — how far the light travelled through the film base before it
reached the emulsion — sets both the falloff shape and the colour.

IS NOT: a tinted gradient. IS NOT a scan-overlay library (the market standard is canned
PNGs; LayerFilter's LightLeak was verified as 32 fixed plates).

**THE MOAT, in one sentence:** every commercial leak pack tints everything orange,
because every plate was captured from the same felt-trap geometry. Path length explains
*why* that one is orange, and therefore predicts the cases that are not.

---

## PRODUCTION MODEL

### 1. A leak is added exposure, so the composite is additive in LINEAR light

Photons add. The leak delivers extra exposure E_leak on top of the image exposure, and
after development the positive carries the sum. On the final positive:

    L_out,c = L_in,c + s · G(x,y) · w_c(x,y)

with G the geometric field (§2), w_c the per-channel weight (§3), s the intensity.
This is Halation's composite (`lin + frac * H`, LOAD-BEARING CALL 2 there) and it is
reused verbatim — the same reasoning applies, so the same code path should.

**Polarity is NOT the interesting axis here** (unlike Film Damage). On a negative, extra
exposure means more density, which blocks more printing light, which prints *brighter*;
on reversal, extra exposure means less retained dye, which is also brighter. Both routes
brighten the positive, so there is no sign to choose and no polarity widget. Worth
stating explicitly so nobody later "fixes" its absence.

**The exact signature (the kill-test):** an additive-in-linear leak adds a **constant
absolute luminance**, independent of what is underneath. A screen blend in sRGB — what
an overlay pack does — cannot: its delta shrinks as the underlying tone rises. So the
test is ΔL measured in linear light against substrate tone: flat for us (to floating
point), strongly sloped for the screen-blend control.

Note the pleasing symmetry with the node next door: Film Damage is a tone-invariant
*multiplicative* offset, Light Leak is a tone-invariant *additive* one. Two different
physical situations, each exact, each with the same class of negative control.

### 2. Geometry classes — G(x,y)

#### 2a. Sprocket trap (the signature look)

Light enters at the cassette mouth / felt trap and travels along the film. Where it
meets a perforation it is unobstructed; where it meets solid film it is attenuated. The
field is therefore an edge falloff **modulated by the perforation lattice**.

Lattice geometry is not invented — it is Film Rebate's, already shipped and verified:
KS-1870, **pitch 4.75 mm, 8 perforations per 38 mm frame advance**, of which 36 mm is
the image. So across the image width there are 36 / 4.75 = **7.58 pitches**, and the
lattice phase is seeded (a frame does not begin on a perforation boundary — the
pitch × 8 ≠ 36 arithmetic that the Rebate build caught).

    G_sprocket(d, u) = exp(-d/λ) · comb(u, d) / duty

- d = distance from the leaking edge, u = position along it
- comb(u, d) = the 1.98 mm perforation profile at 4.75 mm pitch, seeded phase,
  **laterally diffused** by σ(d)
- duty = 1.98/4.75 = 0.4168, so the far field tends to 1

**CORRECTED AT SPIKE — the comb must DIFFUSE, not merely fade.** The first draft of
this section decayed only the modulation *amplitude*, `m(d) = exp(-d/λ_mod)`. That
leaves every tooth with parallel vertical sides all the way down, and it renders as a
bar chart rather than as light — killed at my own eyeball before Jeremie saw it. Light
that passes a perforation spreads sideways as it travels, so each tooth widens and
softens into a blob. Diffusive transport in a scattering slab gives σ ∝ √path:

    σ(d) = σ₀ + c·√(d · pitch)        σ₀ = the hole's own 0.5 mm corner radius

A Gaussian-blurred rectangle is a difference of error functions, so this evaluates in
closed form — exact, vectorised, no convolution needed.

**This removes a free parameter.** Contrast decay is now a *consequence* of the
diffusion rather than a second knob: a blurred square wave loses amplitude on its own,
so λ_mod no longer exists. One mechanism produces both the shape softening and the
contrast falloff. The load-bearing condition becomes simply c > 0, and its negative
control is c = 0 (no diffusion, comb never blurs, contrast cannot outpace the envelope).

**The image does not start at the perforations — SECOND SPIKE CORRECTION.** On 135, the
film is 35 mm tall, the 24 mm aperture is centred (spanning 5.5–29.5 mm), and perf hole
centres sit 2.0 mm from the film edge with a 2.79 mm hole, putting the hole's inner edge
at 3.4 mm. Light therefore crosses **≈ 2.1 mm of film before it reaches the first image
row**, and arrives already diffused. Without that offset the comb meets the frame edge at
full contrast and renders as a row of stage lights — the second thing my eyeball killed.
The number comes straight from Film Rebate's shipped geometry, not from tuning.

#### 2b. Gradient (seal decay / bellows gap / backing paper)

A broad soft wedge from a chosen edge or corner: G = exp(-d/λ) with a wide angular
extent. Per the sweep's elegance merge, seal-wedge and backing-paper glow are the SAME
geometry and differ only in colour source (§3), so they are one mode with a switch, not
two modes.

#### 2c. Pinhole — the anti-convention case

A hole in the bellows or body is a **camera obscura**: it projects an image of the source
onto the film. For a hole of diameter a at distance D from the film plane, imaging a
source of angular size θ:

    spot diameter ≈ a + D·θ

For sunlight (θ ≈ 0.0093 rad) at D = 50 mm, the projected term is 0.47 mm. Pinhole spots
are therefore **small and comparatively sharp** — a different visual object from the big
soft washes above, and the size is derived rather than dialled.

#### 2d. Path-length displacement (added at spike, Jeremie's ask)

The analytic field has iso-intensity contours running perfectly parallel to the frame
edge, which is the one thing that still read as synthetic. Real leaks wander, and there
is a mechanism for it: the gap of a felt trap or a failing door seal is not uniform
along its length, and the film's contact against the pressure plate varies, so the light
does not travel the same distance everywhere.

    d_warped = d + A · N(x, y)

**LOAD-BEARING: the warp displaces the PATH LENGTH `d`, not the finished field.** Colour
is a function of d (§3), so warping d carries the falloff, the red fringe and the comb
diffusion together, coherently — the fringe follows the wandering contour instead of
sliding off the geometry that produced it. Warping the output is what happens when you
distort a PNG plate, and it decouples exactly the thing this node derives. Asserted as a
tooth: colour must remain a strict monotone function of path length under displacement.

**Scale is floored by the mechanism, not by taste.** A seam gap varies over millimetres.
At 36 mm across the frame, 1 mm = 28.4 ref-px @1024, so a 5 mm seam feature is ~142
ref-px. Finer than that is not a mechanical gap variation at all — it reads as noise
running through the effect (Jeremie, at the spike). So `displacement_scale` is CLAMPED at
WARP_SCALE_FLOOR ≈ 142 ref-px, octaves are capped at 2, and octave amplitude falls off
fast (1, 0.30) — equal-ish octaves put real energy at scale/4 and reintroduce the noise
look no matter how large the base scale is. Default scale 380 ref-px (≈13 mm of seam).

This is asserted spectrally rather than left to opinion: the fraction of warp energy
above the mechanical cutoff must stay under 5% (measured 1.7% at defaults), with an
unclamped fine multi-octave warp as the negative control (6.0%).

### 3. Colour — one mechanism, and it predicts the exception

Light entering the film EDGE travels laterally through the base and the dye/AH layers
before reaching the emulsion. Those layers absorb short wavelengths preferentially — the
identical round-trip absorption argument that produced Halation's derived ordering
w_R > w_G > w_B. Attenuating each channel over path length d:

    E_c(d) = E_0 · exp(-d/λ_c)     with     λ_R > λ_G > λ_B

**This single expression carries the falloff AND the colour**, and it makes a prediction
the tinted-gradient approach cannot: because all three channels start equal at d = 0 and
blue dies fastest, the leak is a **near-white hot core that reddens with distance**, not
a uniform orange. That is exactly what real leaks look like — a blown white centre with
an orange fringe — and it is the structure every canned plate flattens.

Ordering is derived; the λ ratios are honest presets (proposed λ_R : λ_G : λ_B =
1.00 : 0.62 : 0.38), the identical split Halation ships (ordering derived, magnitudes
preset, no dye spectra vendored).

**And the exception falls out of the same mechanism.** A pinhole leak strikes the
emulsion from the FRONT without any lateral path through the base, so d ≈ 0, so no
differential absorption, so the light stays the colour of its source — daylight, i.e.
**neutral to slightly blue**. The convention that light leaks are orange is really a
statement about one geometry, and the model says so rather than asserting it.

Three colour sources, then:

| source | colour | status |
|---|---|---|
| base path (edge/sprocket/seal) | derived red-shift with depth | DERIVED ordering, preset λ ratios |
| backing paper (120 roll film) | the paper dye's own tint | honest PRESET, not derivable |
| neutral (pinhole / front-side) | source colour, ~daylight | derived: d ≈ 0, no differential path |

### 4. Units

Ref-px @1024 long edge throughout (no-microns rule, Halation/Eberhard/Damage precedent).
λ, spot size and lattice pitch all scale by L/1024. Resolution independence is a tooth.

### 5. Compute

No FFT. Separable exponentials, a 1-D comb and a handful of projected disks — strictly
cheaper than Halation. Budget: comfortably under Halation's 0.34s/1024².

---

## RECOMMENDED SCOPE CUT: drop "burn" from v1

The 2026-07-25 plan listed a burn sub-mode (heat gradient + orange translucent rim) as a
safe subset, with full melt parked. **I recommend cutting burn entirely, and I want this
called out rather than quietly dropped.** Three reasons:

1. **It is not an exposure event.** Every other mode in this node is stray light adding
   exposure, composited additively in linear. A burn is thermal damage to the base.
   Including it means the node no longer has one honest identity.
2. **What survives the "safe subset" filter has no derivable content.** Strip the melt
   morphology (already parked as texture-scar risk) and what remains is a smooth gradient
   with an orange edge — indistinguishable from a taste gradient. That is precisely the
   Aging & Fog failure mode: a smooth global op that reads as a grade, not an artifact.
3. **It fails the 2026-07-25 first filter.** Melted, bubbled, deformed film is an
   OBJECT-surface phenomenon. Every candidate in that class has been killed at the
   eyeball gate.

If burn is wanted later it deserves its own cycle with real reference, not a corner of
this node. **Jeremie's call** — the three geometry modes stand on their own either way.

---

## Teeth (promoted to `tools/test_light_leak.py` at build)

- **T1 additive-linear invariance (kill-test)** — ΔL in linear light inside the leak is
  independent of substrate tone (exact to floating point). NEGATIVE CONTROL: an sRGB
  screen blend at matched strength, whose ΔL must slope with tone.
- **T2 colour ordering** — along the leak's path the R:B ratio increases monotonically
  with distance. NEGATIVE CONTROL: equal λ per channel produces a flat ratio and fails.
- **T3 hot core** — at d = 0 the leak is near-neutral (channel spread below a bound) and
  its saturation rises with d. This is the claim that separates us from a uniform tint.
- **T4 sprocket lattice** — the modulation period matches 4.75 mm scaled into ref-px
  (independently re-derived from the mm spec, cf. Rebate's pitch oracle), and modulation
  contrast decays faster than the envelope. NEGATIVE CONTROL: **zero lateral diffusion
  (c = 0)**, which leaves contrast depth-independent so it cannot outpace the envelope.
  Note the earlier control (raising λ_mod) is WRONG under the diffusion model — a larger
  coefficient just blurs faster and still passes. Caught at spike.
- **T5 pinhole neutrality** — pinhole spots stay neutral while a base-path leak at the
  same settings reddens. The anti-convention differentiator, asserted.
- **T6 pinhole size** — spot diameter tracks a + D·θ under parameter change.
- **T7 resolution independence** — layout and field agree 1024 vs 2048 (correlate, and
  integrated energy stable), sizes scaling by L/1024.
- **T8 determinism, identity (intensity 0 = bit-exact passthrough), batch, perf.**
- **T9 displacement** — displacement 0 is bit-identical to the analytic field; the
  iso-intensity contour is flat undisplaced (std 0.000 px) and wanders when displaced
  (std 30.4 px); colour stays a strict function of path length; additive tone-invariance
  survives; warp energy stays low-frequency (<5% above cutoff) and a too-fine requested
  scale is clamped rather than honoured. NEGATIVE CONTROL: an unclamped fine multi-octave
  warp must measure noisy.

---

## Honest residuals

- λ ratios are presets, not measured base transmission spectra. Ordering is derived.
- Backing-paper tint is a preset; the dye is not derivable and varies by era and maker.
- The exponential lateral falloff is a first-order absorption/scatter model, not a solved
  radiative-transfer problem in a layered slab.
- The perforation comb assumes the leak enters parallel to the film edge. Real cassette
  geometry has an entry angle; that would shear the comb slightly. Not modelled.
- σ ∝ √path is the diffusive limit. Light near the entry point is partly ballistic, so
  the true spreading is somewhere between √d and d. The diffusive form is the defensible
  end and matches the look; a mixed model is not justified by anything measured.
- The 2.1 mm perf-to-aperture offset is the 135 figure. Other formats have their own
  geometry; the widget stays in ref-px so the user can compensate, but the default is
  135-specific and stated as such.
- Burn: recommended out of scope (see above). Full-melt morphology remains parked.
- The displacement noise field is convention standing in for a real mechanism (non-uniform
  seam gap and film contact). Same honest status as Film Damage's Perlin clustering: the
  mechanism is real, the field is not measured. The 5 mm floor IS derived.
- Like Film Damage, the input is always a finished positive, so this simulates the frame
  that would have resulted had the leak been present. It does not recover an exposure.
