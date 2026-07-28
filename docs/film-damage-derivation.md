# Film Damage — chain-polarity derivation (dust, dirt, hairs, scratches in one node)

Film-damage wave, build 2/3 (after Film Rebate, before Light Leak).
Sweep basis: `comfyui-brain/discover/2026-07-25_film_damage_sweep.md`, scope call
`decisions.md [2026-07-25]` (three nodes, Rebate → Damage → Leak).

STATUS: DERIVED, awaiting Jeremie's sign-off. Nothing built.

---

## What this is / is NOT

IS: a physical model of *where in the imaging chain a defect sits*, from which the
defect's SIGN (light or dark) and COLOR fall out as consequences rather than as
user-chosen paint. One composite formula covers every case.

IS NOT: a scan-overlay library. No PNG assets, no captured plates (filmlooks-class
licences forbid redistribution "whether modified or not" — procedural-only is both
the legal path and the house default).

IS NOT: per-stock calibrated. Defect morphology parameters are honest presets in the
Ivanova gamma-distribution *method*; the polarity and colour physics are derived.

THE MOAT, stated plainly: a stock overlay pack bakes one fixed polarity and one fixed
colour into a PNG. It cannot re-derive sign from chain position, it cannot make a
base-side scratch print white while an emulsion-side scratch on the same negative
prints coloured, and it cannot make white defects bite hardest in the shadows while
dark defects bite hardest in the highlights. All three fall out of the model below.

---

## PRODUCTION MODEL

### 1. The chain-parity rule — LOAD-BEARING CALL 1

A defect is an **optical event at a plane**, not a mark on the final picture. What it
does to the viewed positive depends on how many tonal inversions sit between its plane
and the viewer.

Work in density. Let the defect at its own plane multiply the light passing that plane
by a transmittance τ (τ < 1 for an occluder, τ > 1 for removed dye — see §3).

**Case A — defect on the negative (printing or scan-then-invert).**
Print exposure E ∝ 10^(−D_neg); paper is a normal positive-working emulsion
(more exposure → more density), so D_print = γ_p·log₁₀E + c. The overall inversion
comes from E ∝ 10^(−D_neg), giving D_print = −γ_p·D_neg + c — a dense (bright-scene)
negative passes little light, the paper stays light, the scene prints bright. ✓

Insert the defect: E' = E·τ ⟹ ΔD_print = γ_p·log₁₀τ. With τ < 1 that is negative:
print density DROPS, the print goes **lighter**. Dust on a negative prints WHITE.
(Matches David Mullen ASC / scantips — the sourced fact the model has to reproduce.)

In the final positive's linear luminance L ∝ 10^(−D_print):

    L_out = L_in · τ^(−γ_p)

**Case B — defect on the positive (dust on a print, a slide, or the scanner glass).**
No further inverting material stands between the defect and the eye; the defect simply
attenuates the light that carries the image:

    L_out = L_in · τ^(+1)

**Unified, and this is the whole node in one line:**

    L_out,c = L_in,c · τ_c^k        k = +1  (positive-plane defect)
                                    k = −γ_p (negative-plane defect)

Parity is carried by the exponent's sign; the paper/inversion contrast γ_p carries the
magnitude. Everything else in this document is a statement about τ_c.

**The exact behavioural signature (CORRECTED AT SPIKE — see the note below):**

Because the defect multiplies transmittance at its own plane, it applies a **constant
density offset**, so log(L_out/L_in) is *independent of the substrate tone*. Measured on
the spike: std = 2.3e-16, i.e. exact to floating point, for both polarities. An sRGB
alpha blend — what every overlay pack does — cannot hold that invariance: matched-strength
controls measure CV = 0.945 (white) and 0.160 (dark) against our ~1e-16. That is the
kill-test, and it is exact rather than statistical.

The two polarities then differ in *where they clip*, which gives measurably different
visibility profiles (spike, 12 tone bins, visibility centroid on a 0–1 tone axis):

- k = −γ_p drives L **up**, clipping at paper-base white: centroid **0.492**, and the
  top tone bin collapses to 0.055 against a 0.215 peak — a white speck vanishes in a
  blown highlight.
- k = +1 drives L **down**: centroid **0.647**, rising monotonically with tone
  (Spearman ρ = 0.965) — a dark speck vanishes in blacks.

> **Correction, made at spike:** the signed-off draft of this section claimed the
> negative-plane defect is "at its most violent in the shadows" and that the two
> polarities have "opposite" profiles. Measurement says otherwise: the negative-plane
> profile *peaks in the low midtones* and falls off at BOTH ends (it dies in true black
> too, where absolute luminance differences vanish). The honest claim is the
> shadow-shifted centroid plus the highlight collapse, both quoted above. The formula
> is unchanged; only my description of its consequence was sloppy.

Honest framing: the input to the node is always an already-finished positive. Choosing
"negative" means *simulating what this image would look like had it been printed from a
negative carrying this defect*. That is a legitimate simulation and the formula above is
the right one for it; it is not a claim that we have recovered a negative.

### 2. Units — ref-px @ 1024 long edge (no-microns rule)

Same convention as Halation and Eberhard. Every size (dust radius, scratch width, hair
length) is given in pixels at a 1024px long edge and scaled by L/1024 at execute time.
No micron round trip, no film-format assumption. Resolution independence is a teeth test.

Literature widths (Newson et al. 2014: scratches commonly 3–10px) are quoted for SD-era
scans, so they are a *shape* anchor, not a pixel constant; our default sits at the low
end of that band in ref-px and is exposed as a widget.

### 3. τ_c per defect class — where the colour comes from

#### 3a. Dust, dirt, hairs — neutral occluders
Opaque particulate. Spectrally flat, so τ_c = 1 − α for all c, with α the rasterised
opacity (§4). Nothing more is defensible and nothing more is needed: colour, for these
classes, comes entirely from the parity exponent.

#### 3b. Base-side scratches — refractive, neutral
A groove in the transparent base removes no dye. It acts as a tiny lens, deflecting
light out of the printing/scanning optical path. The proof of the mechanism is that
wet-gate (fluid) printing fills the groove with index-matched fluid and the scratch
vanishes — an absorptive defect could not be cancelled that way.

Light deflected out of the aperture is, to the print, indistinguishable from light
absorbed: τ_c = 1 − α, neutral. So a base-side scratch on a negative prints **white**,
same sign as dust.

The green/cyan-cast claim for base-side scratches on colour negative (orange-mask
channel-gain chain) is **an inference, not a sourced fact** — the sweep flagged it
plausible-but-thin and no single source states the causal link. It ships as an optional
taste knob defaulting to OFF/neutral, labelled as such in the tooltip and the README.

#### 3c. Emulsion-side scratches — dye removal, depth-graded colour
This is the part that earns the node.

A gouge in the emulsion removes dye layers **top-down**. Colour-negative layer order
from the exposing side is blue-sensitive (yellow dye) → green-sensitive (magenta) →
red-sensitive (cyan), which is standard and well sourced (US National Archives
preservation material, any sensitometry text). Removing a dye layer *increases* the
film's transmittance in the band that dye absorbed: τ > 1.

Expressed in the native unit: a removed layer that carried optical density D raises
transmittance in its band by exactly 10^D. The control is `layer_density` (density units
of dye removed per layer), default **0.7** — a normally-exposed mid-scale layer. Real
maximum densities run 2–3, which drive straight to clear base and clip; 0.7 is what
lands the derived colour sequence legibly rather than as instant black or white.

**Widget change made at spike:** the draft folded "base side" into `scratch_depth = 0`.
That was false continuity — base and emulsion are opposite *faces* of the film with
different mechanisms (refractive vs material loss). Split into `scratch_side`
(base | emulsion) plus `scratch_depth` for the emulsion case.

Run that through §1 with k = −γ_p (negative):

| scratch_depth | dye layers removed | τ_c at the plane | print result (k = −γ_p) |
|---|---|---|---|
| 0 (base side) | none, refractive | τ = 1−α, neutral | **white** |
| shallow | yellow | τ_B > 1 | blue suppressed → **yellow** |
| mid | yellow + magenta | τ_B, τ_G > 1 | blue+green suppressed → **red / orange** |
| deep | all three | τ_R,G,B > 1 | clear base → max paper exposure → **black** |

Note the result that makes the point: on the *same* negative, a base-side scratch prints
white and a full-depth emulsion scratch prints black. Dust prints white, a deep gouge
prints black. One physical parameter (depth) generates the whole family.

Now reversal / slide film, k = +1, same τ_c:

| scratch_depth | dye removed | slide result (k = +1) |
|---|---|---|
| shallow | yellow | more blue transmitted → **blue** |
| mid | yellow + magenta | blue+green → **cyan** |
| deep | all three | clear → **white** |

The reversal column is the exact complement of the negative column. That is not a
coincidence to be coded twice — it is the same τ_c read through the opposite exponent
sign, and their being complementary is a **teeth test with real teeth** (§7, T4).

B&W: no dye layers. An emulsion scratch removes silver → higher transmittance → more
paper exposure → **darker** print; a base scratch → white. Neutral in both cases. The
`film_type` switch therefore collapses B&W to two neutral cases and is not a separate
code path.

This also lines the node up with Film Rebate's three-way `bw / c41 / reversal`
vocabulary, so the pack reads coherently.

### 4. Morphology — Ivanova method, our parameters

Per-class defect **count** and **size** are drawn from gamma distributions
(Ivanova et al., CGF 2023 / arXiv:2302.10004, `daniela997/FilmDamageSimulator`, code
MIT). The method is the reusable part; their fitted constants come from a CC BY 4.0
dataset of 12,135 annotated real defects, used as a calibration *reference* only, with
attribution, not embedded. Honest split, identical in kind to Halation's "ordering
derived, magnitudes preset".

That repo writes grayscale masks for ML training and contains **zero** compositing,
blending or polarity code. Everything in §1 and §3 is ours.

**Size defaults are sanity-anchored to real particle scale** (the halation r_c precedent,
where checking real triacetate thickness moved the default from 14 to 8). A dust mote is
20–100 µm on a 36 mm frame, i.e. **0.57–2.84 ref-px in DIAMETER** at a 1024 long edge.
The first spike draft ran 8–16 px and read as torn-paper confetti. This does not breach
the no-microns rule — microns are used once, offline, to pick a default; the widget stays
in ref-px.

Final defaults, after Jeremie's spike call ("overall size of dust and scratches too big"),
sit at the FINE end of the physical band, which is where a clean well-handled negative
lives:

| class | mean radius (ref-px) | rendered diameter p50 / p90 | ≈ real size |
|---|---|---|---|
| dust | 0.35 (cap 2.0×) | 0.64 / 1.16 | ~22–41 µm |
| dirt | 1.1 (cap 2.5×) | 1.96 / 3.88 | lint and fluff |
| scratch width | 0.8 | — | well under the Newson 3–10 px SD-scan band |
| hair width | 0.6 | — | single fibre |

The literature scratch widths (Newson et al., 3–10 px on SD scans) describe heavily
damaged archive film. A still photograph wants far less, and the widget covers anyone
who wants more.

Shapes, all procedural and seeded:
- **dust** — small compact blobs; ellipse with random axis ratio, perturbed by 2–3
  low-order radial harmonics so the outline is irregular but convex-ish. The edge term
  must be a real fraction of the radius (0.45), not a token 0.18: a mote is imaged
  through the enlarger/scanner optics and never has a hard cut.
- **dirt** — larger, lower opacity, more strongly clustered, higher harmonic content.
- **hairs (short / long)** — near-constant-width strands along a curvature-bounded
  random walk. Two classes = two length regimes, one code path. **Curvature must be a
  bounded random walk with a capped cumulative turn (~1.4 rad).** A constant curvature
  increment integrates to a perfect circle: the first spike draft rendered visible fake
  rings on the wall of the test frame. Real fibres are arcs and S-bends, never loops.
- **scratches** — long, near-straight, with lateral wander. Kokaram (1993/1998) gives the
  cross-section family; Joyeux et al. (~2000) gives bounded sinusoidal/cubic wander.
  Kokaram's diffraction side-lobes are a real feature of the profile and are cheap: a
  narrow core with a low-amplitude opposite-sign shoulder either side (light deflected
  out of the groove is deposited beside it). **Intensity must be modulated along the
  length** — contact pressure varies, so a scratch fades and breaks up. A dead-uniform
  edge-to-edge line reads as a drawn artifact, not a tramline; caught at spike eyeball.

**Scratch orientation — the detail that sells it.** Scratches run along the film
*transport* direction: for still 35mm that is along the frame's long edge, for cine the
film runs vertically through the gate so projected scratches are the famous vertical
tramlines. Widget `transport_axis`: auto (long edge) / horizontal / vertical.
Ivanova likewise restricts scratches to vertical/horizontal.

### 5. Placement

Perlin/value-noise density field, importance-sampled, giving organic clustering rather
than a uniform scatter. **Labelled honestly as artistic convention** — Ivanova's own
paper does not claim it as a measured spatial statistic, and neither do we.

Explicitly NOT included: any radial or edge bias. Smooth global spatial weighting is
what made Aging & Fog read as a grade (`patterns.md 2026-07-17`); the defects carry the
non-uniformity by being discrete, and nothing else should.

### 6. Optical softening — free realism from the same model

A defect is imaged by whatever optics lie between its plane and the sensor, so the two
polarities are *not* equally sharp:
- negative-plane defect: sits in the enlarger/scanner film gate, close to the image
  plane, imaged nearly in focus → slightly soft.
- positive-plane defect: sits on the scanner glass or print surface, offset from the
  image plane → noticeably more out of focus.

One `softness` widget, defaulted differently per `defect_origin`. Physically motivated,
one Gaussian, near-zero cost, and it removes the cut-out-sticker look for free.

### 7. Teeth (promoted to `tools/test_film_damage.py` at build)

- **T1 polarity sign** — same seed, same defect field, `defect_origin` flipped: mean
  luminance inside the defect mask moves *up* for negative-plane, *down* for
  positive-plane. Negative control: a build that ignores parity fails this.
- **T2 exponent magnitude** — negative-plane defects at α→1 clip to 1.0 (paper base
  white); positive-plane at α→1 reach 0.0. Both exact, not approximate.
- **T3 tone-invariance (the kill-test)** — at fixed τ, log(L_out/L_in) is independent of
  the substrate tone (std < 1e-12; spike measured 2.3e-16). Negative controls: sRGB
  alpha blends at matched strength, which must break it (measured CV 0.945 white / 0.160
  dark). Plus the profile asymmetry: negative-plane visibility centroid at least 0.12
  below positive-plane, and negative-plane top-bin collapse below half its peak.
  NOTE the profile half is *statistical* and needs a dense measurement rig; the shipping
  default is deliberately sparse and starves the tone bins. The invariance half is exact
  and is the one that actually carries the claim.
- **T4 reversal/negative complementarity** — for equal `scratch_depth`, the c41 mark
  colour and the reversal mark colour are complementary in the derived sense.
  Negative control: hardcoding either column independently breaks it.
- **T5 resolution independence** — same seed at 1024 and 2048 gives the same defect
  layout and the same **integrated opacity mass**. Two metric choices are load-bearing
  here: (a) mass, not a thresholded pixel count — the softness blur conserves mass but
  spreads it over more pixels, so a count-based test flags correct behaviour as drift;
  (b) layout by **downsample-and-correlate** (Pearson r > 0.85, measured 0.9937), not by
  a weighted centroid — at the final fine sizes the whole frame carries only ~40 px of
  opacity, so a centroid is dominated by a handful of motes and reports sub-pixel
  rasterisation noise as drift. Defect COUNT is a per-frame physical quantity and must
  NOT scale with pixel area: a frame carries the dust it carries whatever resolution you
  scan it at.
- **T6 determinism / batch** — same seed reproduces exactly; batch frames advance or
  hold per the batch policy, both deterministic.
- **T7 identity** — density 0 or all classes disabled returns the input bit-exact.
- **T8 perf** — 1024² and 4K budgets, embedded torch, in line with Rebate/Halation.

### 8. Controls (final at spike)

Sections: origin/physics, then one block per class, then global.

| widget | default | note |
|---|---|---|
| `defect_origin` | negative (prints white) | the parity k of §1 |
| `film_type` | c41 | bw / c41 / reversal; drives §3c |
| `print_gamma` | 2.0 | γ_p, paper grade / inversion contrast |
| `density` | 0.5 | master count multiplier across classes |
| `dust_amount` / `dust_size` | on | gamma-drawn count and size |
| `dirt_amount` / `dirt_size` | on | |
| `hair_amount` / `hair_length` | on | short+long via the length regime |
| `scratch_amount` / `scratch_width` | on | ref-px @1024 |
| `scratch_side` | base | base (refractive, white) or emulsion (dye loss) |
| `scratch_depth` | 0.5 | emulsion only; walks the §3c colour table |
| `layer_density` | 0.7 | density units of dye removed per layer |
| `transport_axis` | auto | auto = along the long edge |
| `softness` | auto per origin | §6 |
| `base_scratch_cast` | 0.0 (off) | the thin inference, taste knob only |
| `seed` | 42 | |

Outputs: `IMAGE` + `MASK` of the defect field (Rebate set the two-output precedent, and
the mask makes the node compose with inpainting/restoration graphs).

---

## Colour-space flow

sRGB in → linear (`srgb_to_linear`) → all τ^k compositing in linear light → back to sRGB.
Same as Halation: these are exposure events, not sRGB paint. Multiplicative in linear is
what makes the highlight/shadow asymmetry of §1 come out right; doing it in sRGB would
flatten exactly the behaviour that is the moat.

---

## LOAD-BEARING CALLS

1. **Parity as an exponent, not a sign flip.** L_out = L_in·τ^k with k = +1 or −γ_p.
   The alternative (add a light blob / subtract a dark blob) loses the tone-visibility
   asymmetry, the clipping behaviour, and the whole §3c colour table.
2. **Multiplicative in linear light**, never additive, never in sRGB.
3. **τ > 1 is legal and is the entire colour mechanism.** Dye removal increases
   transmittance; forcing τ ≤ 1 would make emulsion scratches impossible to model.
4. **Neutral base-side scratches by default.** The colour-cast inference is real enough
   to expose and too thin to default to on.
5. **No smooth global spatial term anywhere.** The Aging & Fog scar.

---

## Honest residuals

- Gamma-distribution *parameters* are our presets in Ivanova's method, not their fitted
  constants (dataset is reference-only). If their fitted values turn out to live in the
  MIT-licensed source rather than the CC BY dataset, adopting them is licence-clean and
  is a spike task, not a build assumption.
- Perlin clustering is artistic convention, stated in the README, not a measured statistic.
- Base-side colour cast: inference, off by default.
- `print_gamma` conflates optical-paper contrast with scanner-inversion contrast. Both
  behave as the same exponent to first order; the widget is named for the honest union.
- Kokaram side-lobes are implemented as a two-term profile, not the full diffraction
  integral. Shape family, not a solved wave problem.
- No frame-to-frame persistence (Newson's ≥3-frame criterion) — this is a stills node.
  Batch behaviour is a policy choice, not a physical claim.
- **Sub-1024 sampling floor.** At the final fine sizes a dust mote is 0.35 px radius at a
  1024 long edge and genuinely sub-pixel below that, so the rasteriser's softness floor
  dominates and integrated opacity mass inflates: measured 0.000081 at 512 against
  0.000038 at 1024, converging to 0.000034 by 4096. This is a sampling limit of the
  medium, not a model defect, and it is recorded rather than tuned away. T5 verifies
  invariance over 1024→2048, where the defects are resolvable, and asserts the floor
  separately so it cannot silently drift. Users working below 1024 get slightly heavier
  dust than the widget nominally asks for.
- The τ floor is 1e-4, so a "fully opaque" positive-plane defect reaches L × 1e-4 rather
  than a true zero. That is −4 density; nothing in the display chain can tell.
