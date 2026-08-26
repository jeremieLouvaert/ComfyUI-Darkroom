# Lift Gamma Gain — colour-wheel mapping derivation

**Status:** v3 (v1 SIGNED 2026-08-26 (Jeremie: "go with the recommendations") — all four
open items taken as recommended). **v2 amends §7 with two findings from the
pre-implementation checks — see §7.1 and §7.2. Neither changes a signed
decision; both change the implementation.** **v3 adds §4.2 — the end-to-end
measurement through the shipped Python, and the gamut-clipping finding it
surfaced. No signed decision changes.**
**Date:** 2026-08-25
**Why this needs a derivation at all:** every Darkroom wheel so far (Log Wheels,
3-Way) drove *polar* parameters — `<zone>_hue` 0–360 plus `<zone>_saturation`
0–100 already **are** a wheel position, so the mapping was a stated convention.
Lift Gamma Gain is **Cartesian**: `lift_r/g/b` plus `lift_master`. A wheel there
needs a defined map from a 2-D position to three channel values, and that map is
a real modelling choice, not a convention. Per the derive-before-code rule this
is pinned on paper and signed before any implementation.

---

## 1. What the backend actually computes

From `utils/grading.py::apply_lgg`, per channel *c*, in **linear light**:

```
out_c = gain_c · ( lift_c·(1 − x) + x )^(1/gamma_c) + offset_c
```

and from `nodes/lift_gamma_gain.py`, the master folds in **differently per group**:

| group  | per-channel × master | combine        | neutral | slider range |
|--------|----------------------|----------------|---------|--------------|
| lift   | `lift_c + lift_master`     | **additive**       | 0   | ±1    |
| gamma  | `gamma_c · gamma_master`   | **multiplicative** | 1   | 0.1 … 4 |
| gain   | `gain_c · gain_master`     | **multiplicative** | 1   | 0 … 4 |
| offset | `offset_c + offset_master` | **additive**       | 0   | ±0.5  |

That additive/multiplicative split is the single most important fact in this
document. It means lift and gamma **cannot** share one mapping formula, and a
master control that is linear for lift must be logarithmic for gamma and gain.

Identity short-circuit (`lift_gamma_gain.py`): the node returns the input
untouched when lift/offset are within `0.005`/`0.002` of 0 and gamma/gain within
`0.005` of 1. The wheel's centre must land inside that dead zone so that
"centred" genuinely means "off", exactly as the centre snap does on Log Wheels.

---

## 2. The problem

A wheel gives 2 degrees of freedom (angle, radius). Each group needs 3 channel
numbers. We need one constraint to close the system, and it should be the one
that makes the master bar meaningful.

**Constraint: the wheel is chroma-only. It must not move luminance at all.**
The master bar is then the *only* luminance control for that group. Without this
constraint the wheel and the master fight each other, and the master is
redundant.

---

## 3. Basis: project the hue direction onto the luma-null plane

For a requested hue θ, take the fully saturated RGB colour `c(θ) = hsv2rgb(θ,1,1)`
and remove its luminance component:

```
d̂(θ) = c(θ) − (w · c(θ)) · (1,1,1)
d(θ)  = d̂(θ) / ‖d̂(θ)‖              ← unit length, so radius means one thing
```

Since `Σ w_c = 1`, this gives `w · d̂ = w·c − (w·c)(w·1) = 0` exactly. Verified
numerically: **max |w·d| over 120 hues = 1.1e-16** for both weight choices.

Normalising matters: unnormalised `‖d̂‖` ripples **32.1%** across the hue circle
under Rec.709 (13.4% under equal weights), so without it the same drag distance
would give a visibly different push depending on direction.

### 3.1 Choice of luma weights — decided by measurement

Two candidates: equal weights `(⅓,⅓,⅓)` (the naive "channels sum to zero"
choice) and **Rec.709** `(0.2126, 0.7152, 0.0722)`, matching
`utils/color.py::luminance_rec709` which the rest of the pack already uses.

Measured: apply a rim-strength lift to a linear mid-grey patch (x = 0.18,
amplitude scaled so the largest channel moves 0.30) and read the true Rec.709
luminance shift:

| hue     | ΔY, equal weights      | ΔY, Rec.709 |
|---------|------------------------|-------------|
| red     | −24.8 % of patch luma  | **0.00 %**  |
| yellow  | +53.5 %                | **0.00 %**  |
| green   | **+78.3 %**            | **0.00 %**  |
| blue    | −53.5 %                | **0.00 %**  |
| magenta | −78.3 %                | **0.00 %**  |

Equal weights are not a chroma control at all — a rim push toward green nearly
**doubles** the patch's brightness. **Rec.709 it is**, and this is not a taste
call; the equal-weight option is simply wrong for a wheel whose whole premise is
"colour without brightness".

The price is asymmetry, and it is worth understanding before signing:

| hue    | dR     | dG     | dB     |
|--------|--------|--------|--------|
| red    | +0.934 | −0.252 | −0.252 |
| yellow | +0.077 | +0.077 | −0.994 |
| green  | −0.681 | +0.271 | −0.681 |
| blue   | −0.077 | −0.077 | +0.994 |

A "yellow" push is almost pure −blue; a "green" push mostly *removes* red and
blue rather than adding green. That is physically correct — green carries 71.5 %
of luminance, so adding green without brightening is barely possible — and it is
what any constant-luminance colour model does.

### 3.2 The dot points where the colour goes

The acceptance question for the whole convention: if the user drags toward hue
θ, does the resulting image actually read as hue θ? Measured on a mid-grey
patch through the real lift term, over 24 directions:

> **worst hue error = 0.00°**

Exact, for the additive groups. This is the property that makes the wheel
honest, and it is the first thing the teeth must check.

---

## 4. Additive vs multiplicative groups

Neutral differs per group, so the mapping must too.

**Lift and offset (additive, neutral 0):**
```
param_c = A_group · r · d_c(θ)
```

**Gamma and gain (multiplicative, neutral 1):** the neutral element is 1 and the
group operation is ×, so the luma-null construction must live in **log space**:
```
param_c = exp( A_group · r · d_c(θ) )
```
Working additively here (`1 + A·r·d_c`) would make the same drag asymmetric
between lightening and darkening, and would not compose correctly with the
multiplicative master.

### 4.1 Honest caveat: multiplicative groups are luma-null only to first order

`Σ w_c · d_c = 0` does **not** imply `Σ w_c · exp(A·d_c) = 1` — Jensen's
inequality leaves a residual. Measured worst-case luminance drift on a grey
patch:

| amplitude A | worst luma drift |
|-------------|------------------|
| 0.15 | 0.30 % |
| 0.25 | 0.84 % |
| **0.35** | **1.69 %** |
| 0.50 | 3.58 % |

It can be removed exactly by renormalising `g ← g / (w·g)`, which drives drift
to 2e-16 — **but that costs hue fidelity**: worst hue error rises from 0.00° to
**3.66°** at A = 0.35, and 5.21° at A = 0.50.

**Recommendation: do not renormalise. Cap A at 0.35 and accept 1.69 % drift.**
A 1.7 % luminance change is invisible; a 3.7° hue error is a visible mismatch
between where the dot sits and what the image does, and §3.2 is the property
worth protecting. This is a real trade-off and is flagged for sign-off, not
buried.

### 4.2 AMENDMENT — measured end-to-end, and the clipping finding

§3.1 and §4.1 measured the *mapping* on a single mid-grey patch. Running the
real `apply_lgg` over a 64-step linear grey ramp, 24 hue directions, at rim
deflection, first reported a **19.68 % mean luminance shift** — an order of
magnitude worse than predicted, and it read as a failure of the model.

**It was not.** Splitting the two mechanisms:

| group | luma shift where NOTHING clips | clipped fraction of the ramp |
|-------|-------------------------------|------------------------------|
| lift | **0.000 %** (exact) | 23.4 % |
| offset | **0.000 %** (exact) | 17.2 % |
| gain | **1.689 %** — exactly the §4.1 Jensen bound | 26.6 % |
| gamma | 1.451 % | 0.0 % |

The mapping is luma-exact for the additive groups and lands precisely on the
predicted residual for gain. The 19.68 % was **entirely gamut clipping** at the
black end: `apply_lgg` does `np.clip(lifted, 0, None)` before the power, so once
a rim-strength lift drives the opposite channel negative the three channels no
longer cancel and the luma-null guarantee is void.

**The first version of that test was measuring the wrong thing** — a mean over a
ramp *including* clipped pixels tests the gamut, not the mapping. The invariant
being claimed is about the mapping, so the measurement must exclude pixels the
gamut destroyed, and report clipping separately. (Same family as the standing
patterns.md lesson: choose the metric to match the invariant.)

**Two honest consequences to state in the README, not bury:**

1. The wheel is exactly luma-neutral **in the unclipped region**. It cannot be
   in the clipped one, and no choice of basis would change that.
2. Clipping is inherent to the operation, not to this design: lifting shadows
   toward a colour *must* crush the opposite channel. A full-range neutral grey
   ramp is the worst case; real images clip far less. Reducing amplitude helps
   roughly linearly (lift at A=0.10 clips 7.8 % of the ramp vs 23.4 % at 0.30),
   so if the signed amplitudes ever feel too strong in practice this is the
   measurement to revisit.

Gamma is the one group that never clips — `x^(1/γ)` maps [0,1] onto [0,1] — and
its 1.451 % comes from the power non-linearity rather than Jensen: a per-channel
power is not luma-preserving even when `Σ w·log γ = 0`. It is within budget and
concentrated in the shadows.

---

## 5. Master bars

Each wheel gets one bar beneath it, exactly like Log Wheels' density bar.

- **lift, offset** — linear bar, value added to all three channels. Centre = 0.
  Ranges: lift master ±1, offset master ±0.5 (the sliders' own ranges).
- **gamma, gain** — the parameter is multiplicative, so a **linear bar puts
  neutral off-centre**: 1.0 sits at 23.1 % of gamma's [0.1, 4] slider and 25.0 %
  of gain's [0, 4]. A bar whose middle is not "no change" is unusable.
  Use a **log bar**: `master = exp(t · ln K)` for `t ∈ [−1, 1]`.
  With **K = 4** the bar spans [0.25, 4], is exactly symmetric about 1.0, and
  stays inside both sliders' ranges.

---

## 6. Amplitudes at full deflection

`A_group` sets what "rim" means. All four fit their slider ranges with headroom:

| group  | A | channel range at rim | slider range | fits |
|--------|------|----------------------|--------------|------|
| lift   | 0.30 | [−0.299, +0.299] | ±1 | yes |
| gamma  | 0.35 | [0.706, 1.417] | 0.1 … 4 | yes |
| gain   | 0.35 | [0.706, 1.417] | 0 … 4 | yes |
| offset | 0.15 | [−0.149, +0.149] | ±0.5 | yes |

These are deliberately **not** set to consume the full slider range. A wheel
that reaches ±1 lift at the rim would make every small drag enormous. The
sliders remain the way to exceed the wheel's disc, which is the same
escape-hatch relationship every other Darkroom wheel has. **These four numbers
are the main taste call in this document** — they are cheap to change and are
the thing most likely to want adjusting after a first look.

---

## 7. Invertibility — the dot is rendered *from* the sliders

The wheel owns no state (house rule): every frame the dot position is recomputed
from the current channel values. So the forward map must be invertible.

Given channel values `p`, strip the master/common component and read back:
```
v = p − (w · p)·(1,1,1)      radius = ‖v‖ / A      hue = argmax_θ  d(θ) · v̂
```
(For gamma/gain, take `log p` first.) Measured over 208 (hue, radius) pairs on
the **continuous** map:

> **worst radius error 2.2e-16, worst hue error 0.00° — exactly invertible.**

This also means the wheel degrades honestly: if a user types channel values by
hand that are *not* luma-null, the wheel shows the chroma part and the master bar
shows the common part, and together they still describe the state.

### 7.1 AMENDMENT — write 4 decimal places, not the slider's step

The claim above holds for the continuous map. The shipped wheel writes real
numbers into real widgets, and **quantising to each slider's own `step` destroys
it**:

| write precision | worst hue error (lift) | worst radius error |
|-----------------|------------------------|--------------------|
| step-aligned (0.01) | **8.00°** | 3.14 % of full |
| 3 dp | 0.70° | 0.28 % |
| **4 dp** | **0.10°** | **0.03 %** |
| 6 dp | 0.00° | 0.00 % |

The error is worst at small radius, where a 0.01 step is a large fraction of the
channel magnitude — exactly where a colourist works. Step-aligned writes would
make the dot visibly jump on release.

The polar wheels (Log Wheels, 3-Way) rounded to integers because their params
are degrees and percent with `step: 1.0` over ranges of 360 and 100, where
quantisation is negligible. That reasoning does **not** transfer to a ±1 range
with `step: 0.01`.

**Verified that this is legal, with a positive control.** ComfyUI's
`execution.py` validates only `value_smaller_than_min` / `value_bigger_than_max`
for FLOAT; nothing validates `step`. Confirmed live against the running server:

- positive control — `lift_r = 5.0` → **HTTP 400**, `value_bigger_than_max`
  (proves the endpoint really does validate)
- the actual case — `lift_r = 0.0434` → **HTTP 200**, queued and executed

**Decision: write 4 dp.** `step` remains what it always was — the increment for
dragging or arrow-keying the slider itself — not a constraint on the stored
value.

### 7.2 AMENDMENT — the hue inverse needs a lookup table, not `atan2`

Building an orthonormal 2-D frame `(e₁, e₂)` of the luma-null plane and taking
`φ = atan2(v·e₂, v·e₁)` is the obvious inverse, and it is **wrong**: φ is not θ.
Because `c(θ)` traces a hexagon rather than a circle, φ advances between
**0.65° and 1.41° per degree of hue** — up to a 41 % local rate error, which at
the extremes is several degrees of absolute hue error.

φ(θ) *is* strictly monotonic (verified over 1440 samples at 0.25° steps: every
step positive, total advance 359.75°), so the correct inverse is a monotonic
**LUT on φ built once at module load, then binary-searched** — O(log n) per
frame, not the O(360) brute-force scan used in the offline derivation scripts.

(Note: the first monotonicity check reported `False` because the test demanded
exactly one wrap-around in a non-circular diff. The sequence never wraps. The
test was wrong, not the basis — recorded here because a bad precondition test
that fails *closed* is the benign direction, and it still cost a re-check.)

---

## 8. Reachable set — what the wheel deliberately cannot do

The wheel spans a **disc** of luma-null offsets; the master bar adds the
luminance axis; together they cover a **cylinder**. The sliders cover the full
**cube**, which includes combinations that are neither pure-chroma nor
pure-luma (e.g. lift_r = +1, lift_g = −1, lift_b = 0 with an arbitrary luma
component). Those corners stay slider-only.

This is a deliberate limitation, not an oversight: it is the same relationship
the Log Wheels dot has with its hue/saturation sliders. It must be stated in the
node's README rather than discovered.

---

## 9. Layout

Four groups → four wheels. Resolve's primaries panel is exactly Lift / Gamma /
Gain / Offset in a row, which is the reference this pack has already committed
to.

- **Recommended:** four in a row, `minWidth: 560` (≈ 125 px discs), each with its
  master bar and label beneath — the existing wheel core already supports
  N zones with per-zone bars, so this is a spec, not new code.
- **Alternative if 560 px is too wide:** a 2 × 2 grid, which needs a small
  addition to the wheel core (it currently lays out one row).

---

## 10. Acceptance tests, with their negative controls

Every row must have a control that provably fires, per the standing rule.

| # | test | negative control |
|---|------|------------------|
| T1 | `w · d(θ) = 0` for 120 hues | use equal weights → §3.1 shows a 78 % luma shift |
| T2 | push toward θ yields hue θ (worst < 1°) | flip the angle convention to clockwise → error explodes |
| T3 | rim values inside every slider range | raise A 4× → gamma/gain leave [0.1, 4] |
| T4 | round-trip wheel → channels → wheel | perturb the inverse basis → radius/hue error appears |
| T5 | centre lands inside the identity dead zone | widen the snap → node stops short-circuiting |
| T6 | gamma/gain master bar centre = exactly 1.0 | linear bar → neutral lands at 23 % |
| T7 | Jensen drift ≤ 2 % at the chosen A | raise A to 0.5 → 3.58 % |
| T8 | `widgets_values` byte-identical to a stock save | the standing A/B/C suite |
| T9 | round-trip at 4 dp: hue < 0.5°, radius < 0.5 % | write step-aligned instead → 8° |
| T10 | φ-LUT inverse recovers θ | replace the LUT with a bare `atan2` → several ° of error |
| T11 | end-to-end luma shift ≤ 2 % on unclipped pixels, through the real `apply_lgg` | equal-weight basis → 21 %; 4× amplitude → 37 % |
| T12 | centred wheels leave the image bit-identical | — |

Plus the live suite every other wheel node passes: real pointer drags, old-format
load, round-trip, node does not move while dragging.

---

## 11. Signed decisions

1. **Amplitudes** (§6) — SIGNED at 0.30 lift / 0.35 gamma / 0.35 gain / 0.15 offset.
2. **Jensen trade-off** (§4.1) — SIGNED: do NOT renormalise. Accept 1.69 % luma
   drift to keep 0.00° hue fidelity. A is capped at 0.35 for this reason.
3. **Layout** (§9) — SIGNED: four wheels in a row, minWidth 560.
4. **Offset wheel** — SIGNED: included, matching Resolve's four-wheel primaries panel.

Not open: the Rec.709 weighting (§3.1) and the log-space treatment of
gamma/gain (§4) are settled by measurement and by the backend's own combine
rules respectively.
