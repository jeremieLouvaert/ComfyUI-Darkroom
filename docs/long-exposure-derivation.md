# Long Exposure — handheld camera-shake integration

Reference set: five Paris frames shot by Jeremie's reference photographer on an a7CR
(`C:/Users/Jeremie/Desktop/New folder/2023+Paris,+Long+Exposure+on+a7CR-*.webp`).
Those are TEST INPUTS ONLY — never committed, never shipped (Stoyanov clean-room /
filmlooks discipline).

**Technique confirmed by Jeremie, not inferred:** ND filter, slow shutter, subtle
erratic hand movement. That single sentence rules out two models I had on the table
and is the reason this derivation is short.

STATUS: DERIVED + STAGE A PROVEN. Awaiting sign-off. Nothing built.

---

## 1. The model

A photograph is the time integral of the light arriving during the exposure. If the
camera changes pose during it, the sensor integrates a sequence of views:

    I_out(x) = (1/T) ∫₀ᵀ I( M_t⁻¹ x ) dt

`M_t` is the image-space transform induced by the camera pose at time t. This is not
an effect approximating a photograph; it **is** what the photograph is, which is what
makes the node defensible rather than decorative.

Discretely: sample N poses across the exposure, backward-warp the image by each, and
average.

## 2. LOAD-BEARING CALL 1 — backward gather, never pixel-pushing

Jeremie's opening framing was "every pixel is a particle, push it with the wind".
That is the wrong operator and it is worth stating why, because it is the difference
between the references and a mess.

- **Pushing** pixels forward makes them pile up where the flow converges and leave
  holes where it diverges. Structures tear and mix.
- **Gathering** — each output pixel averaging what passed over it — can neither tear
  nor leave holes, and it is literally what a sensor does.

The references settle it: structures stay **intact and dragged**. The umbrella in
`-146` is still an umbrella; the figure in `-69` is still a figure. Nothing is churned.

Same distinction as Water Refraction's "average renders, not surfaces".

## 3. LOAD-BEARING CALL 2 — ONE global path, not a velocity field

Camera shake moves the **whole frame together**. There is one trajectory, not a
streamline per pixel.

Stage A tested this the expensive way round first, and the evidence is unambiguous:

| velocity field | verdict |
|---|---|
| pan (straight, global) | closest to the references |
| arc (curved, global) | close |
| radial from an emission point | visible outward streaking, not in any reference |
| smooth wander (curl noise) | **swirls and curls that appear in NO reference** |

The most fluid-like field produced the least reference-like image. A wiggly *global*
path reproduces the references; a spatially varying field does not.

**Consequence: the wind/fluid solver is rejected**, and so is the per-pixel
streamline model. A Navier–Stokes air solver would produce more of exactly the
swirling structure the references lack, at days of cost.

## 4. The path spectrum — sourced, and it corrected my guess

Hand tremor is not white noise and not a plain random walk. Both are wrong visibly:
white noise reads as digital jitter, a pure random walk drifts with no shimmer at all.

Measured human hand tremor, from the tremor literature:

- the physiological tremor peak sits at **7–11 Hz** for 90% of subjects (n=237);
- the **7.5–12.5 Hz** band carries **24%** of displacement oscillation amplitude;
- the **1–3.5 Hz** band dominates: removing it cuts total amplitude by **56%**,
  which for quadrature-summed components implies a share near **90%**.

So the path is synthesised in Fourier space as **1/f amplitude** (i.e. 1/f² power)
plus a Gaussian tremor bump at 9 Hz, with no DC term — the mean pose is the frame.

**This is a real validation rather than a fit.** With a tremor weight of 0.05 the
model measures **89% in the 1–3.5 Hz band and 24.7% in 7.5–12.5 Hz**, reproducing
both published band shares independently. My own first guess, 0.45, put 59.5% in the
tremor band — an order of magnitude too much — and looked visibly jittery. The
sourced value is both more accurate and better looking.

One correction recorded so it does not propagate: "removing 7.5–12.5 Hz reduces
amplitude by <3%" is NOT that band's share. Amplitudes add in quadrature, so a 24%
share removed yields ~3% total reduction. The two published figures agree.

## 5. Roll — the only frame-varying component, and it is free

Camera yaw and pitch translate the image nearly uniformly, so a shift covers them.
**Roll** rotates about the optical axis, so corners travel further than the centre.
That is visible in the references as corners smearing differently from the middle,
and it costs nothing if the pose carries an angle:

    M_t = translate(a·tx(t), a·ty(t)) ∘ rotate(θ·roll(t), about frame centre)

Modelling the pose as a rigid transform rather than a pure shift gets it for free.
Full projective shake (Whyte et al.) is deliberately NOT modelled — it needs a focal
length and a depth map, and the extra realism is invisible at these amplitudes.

## 6. What the references need that a global path CANNOT give

In `-69` the figure smears while the wall stays clean. No global path does that —
it is **subject** motion, not camera motion. That is the one place Jeremie's
spatial-control instinct is genuinely required, and it arrives as an optional
**MASK** input scaling the path amplitude per pixel, not as a fluid solver.

## 7. Why it cannot read as a plain motion blur

Every existing ComfyUI motion-blur node applies ONE angle to the whole frame
(LayerFilter: MotionBlur, comfy-magick MotionBlur). Prior-art sweep found no node
integrating along a synthesised, temporally structured camera path.

The distinguishing behaviours, each a consequence rather than a setting:

1. the path **revisits** parts of its own track, so edges pick up ghosted, doubled
   copies — visible in `-146`. A straight smear cannot do this.
2. the smear **curves**, because the path curves.
3. **corners differ from centre** whenever roll is non-zero.
4. the spectrum is a measured human one, so it reads as a hand rather than a filter.

## 8. Controls

| control | what it is |
|---|---|
| `exposure_px` | streak length. The intensity dial. |
| `hand_steadiness` | drift↔tremor balance. Default sits at the measured human value. |
| `roll_deg` | the frame-varying component. |
| `subject_mask` (optional) | per-pixel amplitude — subject moving separately from camera |
| `seed`, `poses` | which shake, and quality/speed |

Grading stays OUT. The references are high-key and washed, and that is doing real
work, but Darkroom has eleven grading nodes and this is node-per-effect. Stage A
proved the grade is not sufficient alone: high-key with no motion looks nothing like
the references.

## 9. Teeth

1. **zero exposure is the identity**, bit-exact.
2. **energy conservation** — a uniform field integrates to itself; the operator
   averages and must not brighten or darken.
3. **no holes, no tearing** — gathering guarantees it, asserted by checking every
   output pixel draws from ≥1 valid sample.
4. **band shares** — the synthesised path reproduces ~90% at 1–3.5 Hz and ~24% at
   7.5–12.5 Hz, i.e. the published human values. This is the check that keeps the
   node honest rather than tuned.
5. **roll varies with radius** — corner displacement exceeds centre displacement,
   and is zero at the centre by construction.
6. **determinism** — same seed, same shake.
7. **negative controls**: white-noise path must FAIL the band-share check; a
   zero-roll path must show no radial variation; pushing instead of gathering must
   produce holes.

## 10. Honest residuals

- Rotation is modelled about the frame centre, not the true optical axis; they differ
  if the image has been cropped off-centre.
- No parallax. Camera translation would move near and far subjects differently; only
  rotation is modelled, which is the dominant handheld mode but not the only one.
- The tremor spectrum is a population statistic, not any individual's hand.
- Subject motion via mask is a per-pixel amplitude scale, not an independent
  trajectory — two subjects moving in different directions need two passes.
- Rolling shutter is not modelled.
