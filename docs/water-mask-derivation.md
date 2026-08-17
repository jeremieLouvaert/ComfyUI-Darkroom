# Native MASK input for Water Refraction — derivation

**Version: v2 (post-adversarial-attack, 2026-08-16). Status: DERIVED, dry-run 47/0
(`_water_mask_dryrun/dryrun.py`) + both adversaries' independent probes
(`_water_mask_dryrun/adv_a/`, `adv_b/`), two-Opus fresh-eyes attack COMPLETE and
folded in (§10: 11 spec-fatal, ~16 material against v1). Jeremie's sign-off
PENDING (§9). Nothing built.**

Mask-audit slice 3 per `comfyui-brain/procedural-plan.md` §9d/§9f. Runs in Darkroom's
version line (1.20.0 → 1.21.0 prepared at build time). Scope: optional native MASK on
`DarkroomWaterRefraction` ONLY, plus three teeth-backed ride-along fixes the attack
surfaced in the shipped node (§7.12). The generator-modulation pair is slice 4, out of
scope.

This slice inherits the slice-2 contract (`docs/lens-mask-derivation.md` v2) and
states every departure. The §9d law binds: **the mask scales the OPTICS, never the
solver** — the water surface is physical state, and masking it would make the fluid
simulation depend on a paint stroke.

---

## 0. What the shipped node actually does (read from the code, not the README)

Per frame:

1. **Solver** (`simulate` → `settle` → `to_image_res`): produces `h_mm`, the water
   height field at image resolution. Cached across the batch unless `vary_per_frame`.
2. **Optics** (`render_auto` → `render` numpy float64 / `render_gpu` torch float32;
   **on any CUDA machine, including the target box, the GPU path is the one that
   runs**): `refraction_offsets` → `(dxp, dyp)` the displacement field in pixels per
   destination pixel, plus `cos_i`; preimage jitter from `|det J|` of the source map
   (`jamp = sqrt(max(1 − 1/|det J|, 0))`); per aperture sample, displacement sampled
   at the footprint point `q` and applied at the (jittered) destination; Fresnel
   `out = (1 − R)·acc + R·env·env_strength`; `clip(0,1)`.
3. **grain_deficit**: pushes a noise probe through the warp (`render_auto`,
   `fresnel=False`), measures surviving local RMS, returns `sqrt(1 − r²)` — the MASK
   output, smoothed at `smooth_px = 6`. **Live bug found by this slice's attack: the
   probe never receives `depth_scale`, so at `depth_scale ≠ 1` it measures a
   DIFFERENT warp than the image took** (up to 0.50 pointwise at ds 0.25) — fixed as
   ride-along §7.12a, which restores the "identical warp" premise this document
   relies on.
4. **restore_grain**: adds engine grain weighted by
   `a(x) = deficit(x)·(src RMS / engine RMS)·amount`.

Node-level: input float32 → numpy float32 → `astype(float64)` → processed →
`astype(float32)`; the roundtrip is bitwise, which is what makes node-level bitwise
claims possible. A 4-channel input is sliced to 3 channels before the optics (so
bitwise rows compare against `image[..., :3]`).

---

## 1. The model — destination-sampled mask scaling the sampled displacement

For every aperture/jitter sample of destination pixel `p`:

```
d      = D(q)                (sampled exactly as today, at the footprint point q)
sample = p' + m(p) ⊙ d       (m read at the DESTINATION pixel, applied to the
                              sampled displacement — this exact op: multiply, add)
```

plus the three gates of §2–§4. No reassociated `p + m·(src−p)` form exists here:
WR's offsets ARE displacements, so the mask is a pure multiplier. Consequence
(unlike slice 2 §1d): **`m = 1.0` multiplies by exactly 1.0 — mask=ones measured
BITWISE equal to mask-absent** across a 40-combination box (both engines, sizes to
4K, depth_scale {0.25, 3.0}, aperture {0, 0.060}, dispersion on/off, env_strength
{0, 3}, `pixel_aa` off, full node with `grain_restore=1.0`). No non-multiplicative
path for `m` exists (the where-select, the Jacobian gradient and `m·R` all preserve
exact 1.0 — adversary-verified). The tooth still carries τ₄ = 1e-5 as a fallback per
the lucky-sample scar, measured-zero recorded. **Scope pin**: bitwise holds for a
FULL-RESOLUTION ones mask; a RESIZED ones mask (either direction) arrives at
`1 − 1.1e-16` — within τ₄, not bitwise.

**Inheritance:** slice-2 §1a verbatim: *the source lookup moves `m(p)` of the way
from the pixel's own location to where the unmasked node's sample would have
looked.* Paint where you want the effect to land.

**The rejected alternative — field scaling (`D_m = m ⊙ D` before sampling).** Under
field scaling the mask is effectively read at the footprint point `q`: the effect
fades over the aperture footprint inside the painted region near every edge, and m=0
pixels pick up leakage. Rejected: (1) it departs from the signed slice-2 law with no
compensating physics — a mask is a creative gate, not a physical object; (2) crisp
adherence to the painted mask is the predictable behaviour; (3) the two forms differ
only within `rho` of a mask gradient, and `rho` is SMALL — measured mean 1.72 px,
max 3.55 px at the pinned config — so field scaling buys nothing visible anyway.

**Fold behaviour, attacked and held:** all fold branches at `p` scale by the same
`m(p)`; the fold contracts toward `p` rather than un-blending. Adversary-measured
across a full c-lattice (chart + photo): Monte-Carlo residual monotone in c, never
above both endpoints; manufactured-discontinuity fraction 0.000% at every c; no new
artefact class. **Dispersion:** three IOR passes, ONE mask for all three (the
slice-2 §2b argument; the R:B dispersion ratio measured unchanged under a uniform
mask, 0.9964/1.0198).

**`rho` and `h` stay physical** (§9d law): the mask gates what the destination pixel
takes from the optics, not the optics' inputs.

### 1b. Mask-edge artefact — measured law, replacing v1's transferred bound

v1 transferred slice 2's `f ≥ 2·max|D|` feather rule. **The attack killed it three
ways**: (a) at the pinned config that rule demands 277 px of feather on a 256 px
frame — the dry-run's own "feathered" mask had no m=0 or m=1 region left, so v1's
det-J numbers described a near-uniform ramp, not an edge; (b) `det J_m`'s range is
essentially insensitive to feather width anyway (hard, 16, 32, 64 px all give
[−299, +1872]) — and no total-det promise is meaningful when the UNMASKED warp
already folds on 45.6% of the frame (folding is this node's identity); (c) the
artefact itself is not v1's "band of width |D|": measured on a smooth image, a hard
edge produces a **one-column seam** (discontinuity 0.124 ≈ 3× the source's own worst
column jump), and feathering replaces it with a partially-warped strip exactly as
wide as the feather.

**The measured feather law** (`adv_a/p7e_feather_law.py`; criterion: edge seam ≤ the
fully-warped frame's own worst column jump):

```
f ≈ 0.25 · max|D|        (round up; use 0.5·max|D| for shallow water, ds ≤ 0.25)
max|D| = 0.881 · h_max · depth_scale / mm_per_px      (pixels; mm_per_px = field_width_mm / W)
```

Measured minima: ds 0.25 → 6 px (max|D| 12.9); ds 1.0 → 24 px (115.5); ds 1.3 →
48 px (164.5); ds 3.0 → 96 px (465.9). At pool depths on a 1024-px frame that is
**tens of pixels, not hundreds** — v1's tooltip had both the number and the
direction wrong. All |D| figures are resolution-dependent (they are in pixels) and
are labelled with their frame size in the probes. The console's printed `dmax`
gains the missing `depth_scale` factor in ride-along §7.12b so the user can actually
compute this. Jitter at the edge: see §2 (the symmetric overhang pair).

### 1c. Exact preservation at m = 0 — the select rule, four reasons

```
out     = where(m == 0, original, processed)      # once per frame, after restore_grain
deficit = where(m == 0, 0,        deficit_m)      # the MASK output, §4
```

1. **numpy — the one-pixel Jacobian overhang**: with a zeros mask the numpy
   pipeline measured bitwise WITHOUT the select at every size tried (spline
   evaluation at integer knots — recorded as expected-typical, not law). But at a
   mask EDGE, `np.gradient` carries the `D⊗∇m` term one pixel into the m=0 side:
   the boundary column is jittered and differs by up to 0.52 (exactly the boundary
   column at every size probed). The select is load-bearing on numpy for that
   column, and W3's NC must therefore use an EDGED mask (a zeros-mask NC is DEAD on
   numpy at every size — adversary-measured).
2. **GPU**: `grid_sample` acT residuals: 8.1e-5 at 509×767, 5.4e-5 at 512², 1.2e-4
   at 1024², and EXACTLY 0 at 513×513 and 257×257 — the `(size−1)`-pow-2 regime.
   NC configs must avoid `2^n+1` sizes, stated in the row.
3. **Out-of-range input (the largest reason, v1 missed it)**: ComfyUI IMAGE tensors
   are not guaranteed in [0,1]; `render`'s final `clip(0,1)` alone would clamp an
   m=0 pixel of a 1.40-valued input by 0.40. With the select: 0.0 measured. The
   select passes the ORIGINAL through unclamped (slice-2 §1c contract), so W2 holds
   for arbitrary-range inputs.
4. The Fresnel gate contributes exactly `acc` at m=0 (multiplies by exact 0/1), but
   `acc` carries reasons 1–3.

Mask NaN: `clip` does not remove NaN and `NaN == 0.0` is False, so a NaN mask pixel
would poison its output pixel — §5 pins `nan_to_num(m, nan=0.0)` (NaN = not
painted). Continuity at `m = ε`: the IMAGE is continuous (displacement `ε·D`,
jitter → 0, sheen → ε·R). **The DEFICIT output is deliberately NOT continuous at
the m=0 boundary** — §4 states why that is correct rather than a defect.

---

## 2. The hard kernel — jitter from the MASKED Jacobian

```
sx_m = m·dxp + x          sy_m = m·dyp + y          (same discrete gradient op
jamp_m = sqrt(max(1 − 1/max(|det J_m|, 1e-9), 0))    the engine already uses)
```

- **Interior m=0: `jamp_m` exactly 0** — no jitter on pristine pixels. Inheriting
  the unmasked `jamp` instead would jitter 93.2% of the m=0 zone at the pinned
  compression config (the W6 NC, measured fuel).
- **Uniform m=c — v1's "compression shrinks toward identity" is STRUCK.** With
  `A = J − I`: `det J_m(c) = 1 + c·tr(A) + c²·det(A)`, a parabola in c; wherever
  `det(A) < 0` (54.9% of the frame at the pinned config) it bulges past both
  endpoints, so **partial masks locally over-jitter relative to BOTH m=0 and m=1**
  (measured: every one of 24 configs, pointwise excess up to 0.98 at c=0.5). This
  is CORRECT — the masked map genuinely compresses more there — and it is recorded
  so nobody "fixes" it later. Only MEAN jitter is monotone (held on all 24
  configs: 0.459/0.659/0.833 at c = 0.25/0.5/1 at the pinned config); no pointwise
  tooth exists.
- **Mask edges — the overhang is a symmetric PAIR** (v1 stated half of it): the
  discrete gradient puts elevated `jamp_m` on the last m=0 column (mean 0.909,
  erased by the select) AND on the first m=1 column (mean 0.963 vs 0.853 unmasked,
  +0.11, one column). The m>0 side ships DELIBERATELY: it is inside the painted
  region, bounded (±0.5 px), and the map genuinely shears there.

---

## 3. Fresnel gate — the vignette precedent, transferred

```
out = (1 − m·R)·acc + m·R·env·env_strength        (m at the destination pixel;
                                                   R, cos_i from the physical surface)
```

One mask, one meaning — the sheen is part of the effect, and gating it makes m=0 a
FULL identity so the select corrects float noise rather than erasing a real effect
(the argument that signed slice 2's vignette gate). `cos_i` is never recomputed
from a "masked surface" — there is no masked surface (§9d). Final `clip(0,1)`
unchanged.

- **Analytic anchor**: flat water has `D = 0`, `R = R0` exactly, so
  `out = clip((1 − m·R0)·img + m·R0·env·env_strength)` is an external oracle.
  Measured error: numpy ≤ 1.22e-15 (τ₇ⁿ = 1e-12); CUDA is float32 coordinate
  quantisation and **scales with linear size** — 2.2e-5 at 192×256, 7.9e-5 at
  509×767, 1.78e-4 at 1080p, 3.59e-4 at 4K (v1's flat 1e-4 failed at 1080p+):
  **τ₇ᶜ = max(3e-5, 1.5e-7·max(H,W))**, verified against all measured points.
- **The NC has a saturation dead zone (v1's config was dead)**: at env_strength 3
  the blue channel clips for img ≥ 0.9696 (ungated) / 0.9850 (gated, m=0.5) —
  above both, gated == ungated and the NC cannot fire; it is also dead on black
  frames at env_strength 0, and marginal (2e-5) on near-black. **NC pinned at the
  measured-firing config: m=0.5, env_strength 1.0, mid-tone frame → gap 8.35e-3**,
  ≥ 15× above τ₇ᶜ even at 4K. The gap formula is
  `|clip((1−R₀m_eff)img + R₀m_eff·env) − clip(...)|`, not v1's unclipped form.
- **Dry regions tint, stated**: where `h = 0`, a partial mask still applies
  `m·R0 ≈ 0.02·m` of sheen (measured 0.0084 at m=0.5) — the unmasked node tints
  dry screen identically; the mask just makes it paintable. One tooltip sentence.

---

## 4. grain_deficit gating — evaluate through the masked warp, then select

```
deficit_m = grain_deficit(…, mask=m)     # probe through the IDENTICAL masked
                                          # pipeline (identical again once §7.12a
                                          # forwards depth_scale)
deficit   = where(m == 0, 0, deficit_m)  # then the select
```

- **Why not `m · deficit_unmasked`**: retention is strongly nonlinear in m —
  measured gap 0.468 at c=0.5 (the multiplied form UNDERSTATES). That is the whole
  justification; v1's second argument (shear-band deficit the unmasked warp "never
  touched") was REFUTED by measurement — at a hard edge both forms are zero on the
  m=0 side and the evaluated form is LOWER on the m=1 side — and is struck.
- **The deficit steps at the m=0 boundary, and that is the truth, not a bug**: the
  resampler's grain destruction is O(1) for ANY nonzero displacement (sub-pixel
  sampling costs ~30% of noise variance regardless of magnitude), so deficit(ε·D)
  jumps to ~0.6 the moment m > 0 (measured: mean 0.11→0.34 between ε=0 and 1e-3).
  The select's exact-zero gate is the only clean point, `{deficit > 0} ⊆ {m > 0}`
  is exact, and the restored IMAGE stays visually continuous because restore_grain
  compensates precisely the loss the deficit reports (measured: |out−in| = 0.005 ≈
  the source's own grain amplitude at an ε-mask). A feathered mask relocates the
  deficit cliff to the exact-zero boundary column (+0.62 in one pixel, measured);
  a downstream grain node sees a gated map, which is what "no grain where the
  refraction never happened" MEANS. Stated in the doc and nowhere promised
  otherwise.
- **NO uniform-monotonicity invariant — v1's was FALSE.** The deficit saturates at
  ≈0.93–0.94 once the warp is strong; past the knee the c-curve flattens and
  drifts (21/45 configs non-monotone on a fine lattice; 7/45 fail on v1's own
  3-point sample, e.g. 0.9291/0.9273/0.9282). It is also a Monte-Carlo measurement
  whose probe seed rides the pour seed (`s+21`): mean-deficit spread across probe
  seeds is 0.114 at c=0.25 — larger than the trend v1's row rested on. The
  2026-08-15 lattice scar, hit again in-house. W8 keeps: (a) exact zero at m=0,
  (b) evaluated-vs-multiplied NC pinned as a FLOOR (`gap > 0.10`), never a value.
- **restore_grain coupling, stated honestly (v1's "mask-independent by
  construction" was false)**: `src` is mask-independent; `added` is a functional
  of the MASKED render, so grain amplitude inside the painted region shifts by up
  to ≈2% (measured −2.02% hard-half, −1.41% const 0.5) depending on what is
  outside it. **Accepted**: the drift is an order below visibility, and the
  alternative (a second full unmasked optics render just to calibrate) buys
  nothing a user can see. Recorded so a tooth is never written on the false claim.
  (Known shipped edge, mask-independent: on a pure-noise frame `added` can floor
  at 1e-12 and `a` clips to 1 — not reachable on photographic content, out of
  scope.)

---

## 5. Mask I/O — slice-2 §3 inherited, ONE resize engine, ComfyUI-real shapes

| rule | this slice |
|---|---|
| widget | optional `MASK` named `mask` (slice-2 departure 1 carried) |
| absent | untouched original code path — **bitwise** (W1) |
| shape | **`reshape((-1, H, W))`** — ComfyUI core's own normalisation (`nodes_mask.py`), NOT v1's 2-D unsqueeze: 4-D masks `(1,1,H,W)` exist in the wild and crash the unsqueeze rule (measured `RuntimeError`) |
| batch pairing | frame `i` gets mask `min(i, M−1)`; `M = 0` treated as absent + console warn (bare indexing IndexErrors, measured) |
| all-zero mask | processed normally (output == input by the select) + one console line — `LoadImage` emits all-zero 64×64 masks for alpha-less images, the single most likely accidental wiring, and silence would read as breakage (measured: full no-op) |
| values | `nan_to_num(nan=0.0)` then clamp to [0,1] (§1c: NaN would otherwise poison its pixel) |
| device/dtype | **`.cpu().numpy().astype(np.float64)`** at node level (v1 dropped the slice-2 device row; a CUDA mask crashes a bare `.numpy()`); `render_gpu` converts to float32 on device alongside `h_mm` |
| resize | numpy only: `zoom(order=1, mode='nearest')` + `[:H,:W]` + clip (mode load-bearing, patterns.md 2026-08-16). ~980 size-pair sweep: zero shape mismatches. Exact zeros survive BOTH directions exactly (the select contract survives); resized ones arrives 1−1ulp (both directions). Honesty note: integer-ratio DOWNSCALE (2048→512) erases 1-px strokes entirely — paint at or near image resolution |
| MASK output | deficit select yields **float32** pinned (`np.where` with a float64 scalar silently upcasts — measured; trailing `.astype(np.float32)`) |

One resize engine is correct here (v1's claim held): the node is numpy-orchestrated,
the mask is prepared once and handed to whichever render engine runs, like `h_mm`.
No cross-engine resize tooth exists because no second resize path exists.
vary_per_frame × mask pairing are independent (§8 W10 pins the real comparisons).

---

## 6. What does NOT exist in this slice, derived not forgotten

- **No strength-equivalence invariant** — WR has no scalar the mask duplicates
  (`depth_scale·c` changes `h`, its gradient, the fold set, `rho`, and Fresnel;
  `m=c` scales only the sampled displacement and the sheen). Its absence is a
  statement.
- **No performance change** — solver runs full-frame regardless; optics too (the
  select needs `processed` wherever m>0, and partial-frame optics would change the
  RNG stream). Measured: masked vs unmasked render_gpu within noise (0.13 vs
  0.14 s at 1080p, 0.62 vs 0.58 s at 4K). The tooltip says a mask does not make
  the node faster.
- **No auto_crop analog, no padding caveat** — no global resample; both samplers
  reflect.
- **Console fold/depth report describes the WATER, not the applied effect** — it
  stays unmasked (but gains the missing depth_scale factor, §7.12b/c).
- **Out of scope, recorded as a named open item, NOT silently accepted**: the two
  engines sample differently (numpy bicubic order=3, chosen explicitly to preserve
  grain; GPU bilinear) — measured full-render divergence up to 0.64 pointwise,
  deficit mean 0.9585 vs 0.9350. Pre-existing shipped behaviour on every CUDA box,
  nothing to do with the mask; goes to open-questions for its own decision.

---

## 7. Implementation constraints for the builder

1. **`mask=None` executes today's arithmetic UNCHANGED** — same ops, same order,
   same RNG draw sequence (W1 bitwise by construction; adversary-verified
   achievable: a spec-verbatim build measured bitwise vs the shipped node on both
   outputs).
2. **Signatures**: `render`, `render_gpu`, `render_auto`, `grain_deficit` gain
   `mask=None`; the node prepares the mask once per frame (§5) and passes the SAME
   array to render and deficit. (Existing suite is safe: every call site in
   `tools/test_water_refraction.py` passes post-`field_width_mm` args by keyword —
   verified.)
3. **No extra OR SKIPPED RNG draws when masked.** All four draws per sample are
   unconditional today; the tempting "jamp is all zero, skip the jitter draws"
   optimisation breaks W4 silently and is PROHIBITED.
4. **Per-sample multiply at the destination pixel** (§1): numpy
   `chan += WR._sample(img, py + m·dys, px + m·dxs)`; GPU
   `chan + samp(t_img, py + t_m*ds[0,1], px + t_m*ds[0,0])` — `m` is the 2-D
   destination-resolution mask, broadcast, never resampled at `q`.
5. **Jitter from the masked Jacobian** (§2) with the engine's own gradient op;
   `pixel_aa=False` early-zero unchanged.
6. **Fresnel gate** (§3): `R_eff = m·R` at the single post-loop blend, both engines.
7. **Deficit + selects** (§4): probe render gets the mask; deficit select
   (float32-pinned) before return; IMAGE select once per frame, after
   `restore_grain`, against the pristine 3-channel float64 frame (an
   `img[..., :3]` slice is a non-contiguous view — `np.where` copies, so the
   shipped path is safe; do not return a view).
8. **Declared seams — one PER ENGINE** (v1's single engine-agnostic seam cannot
   satisfy a bitwise tooth: numpy-vs-GPU `jamp_m` differs by 1.1e-3, measured):
   - numpy: `_masked_offsets(h_mm, field_width_mm, m, ior, depth_scale) →
     (dxp_m, dyp_m, jamp_m)` — float64, the literal op sequence `render` uses;
   - GPU: `_masked_offsets_gpu(...)` → float32 tensors, the literal op sequence
     `render_gpu` uses.
   W5 runs per-engine, bitwise against its own engine only. **Adjudication
   erratum (build day)**: the tuple's first two entries are the MASKED products
   `m·dxp`, `m·dyp` — they are literal intermediates of the masked Jacobian and
   the quantities W5's tooth pins bitwise. The render itself NEVER samples them:
   it samples the unmasked field and multiplies the SAMPLED values at the
   destination pixel (§1). v2's original tuple naming (`dxp, dyp`) was ambiguous
   enough that a zero-deviation builder returned the unmasked fields; the row
   text won.
9. **Widget**: optional `MASK` named `mask`. Tooltip (house register, honest, no
   em dashes): "Where the refraction applies, 0 to 1. Scales the optical
   displacement at each output pixel, so partial values do not ghost, and gates
   the surface sheen and the grain_deficit output with it. The water is still
   simulated across the whole frame, so a mask does not make the node faster.
   Feather the mask edge by about a quarter of the local displacement, typically
   tens of pixels at pool depths, or a hard edge shows a visible seam line. A
   fully black mask leaves the image untouched."
10. **README**: the Water Refraction row gains a one-phrase mask mention;
    `NODE_CLASS_MAPPINGS` untouched; `pyproject.toml` 1.20.0 → 1.21.0 prepared
    (commit and registry publish remain Jeremie's doors).
11. **Teeth architecture**: new suite `tools/test_water_masks.py`,
    implementation-blind, written from THIS document only. **Engine-pinned W1
    oracles (v1's were unbuildable on a CUDA box)**: (a) same-engine rows — frozen
    numpy `render` + a frozen `grain_deficit` variant with the engine pinned to
    numpy, compared with `render_auto` monkeypatched to `render` (of the shipped
    four, only `render` is pure; `grain_deficit`/`restore_grain` dispatch to CUDA
    internally — measured divergence 0.64/0.024); (b) a node-level golden-run row:
    `mask=None` bitwise vs a pre-build capture of the shipped node on this
    machine. `tools/test_water_refraction.py` (39/39+6NC) must stay green (§7.12a
    keeps its I9 rows valid: default `depth_scale=1.0` is bitwise-unchanged).
    Budget measured: GPU render 512²×32 = 0.04 s, numpy 512²×32 = 4.6 s, solver
    nx=64 = 0.7 s — minutes of headroom under the 5-min cap.
12. **Ride-along fixes, teeth-backed, no other behaviour change** (the slice-2
    §5.9 pattern; all three adversary-found, live in shipped 1.20.0):
    a. `grain_deficit` gains `depth_scale=1.0` and applies it to the probe warp;
       the node forwards its `depth_scale`. Fixes the probe measuring a different
       warp than the image (pointwise error up to 0.50 at ds 0.25 → restore adds
       up to ~83% wrong grain). Behaviour change for ds≠1 users, deliberate,
       teeth-backed (W13).
    b. `nodes/water_refraction.py:246`: printed `dmax` multiplies by
       `depth_scale` (currently understates the §1b feather input 3× at ds 3).
    c. `nodes/water_refraction.py:245`: fold% passes `depth_scale=depth_scale`
       (prints 45.6% at every ds today; true 50.5%/56.1% at 1.3/3.0).

---

## 8. Invariants — every row executed BEFORE sign-off

Executed by `_water_mask_dryrun/dryrun.py` (47/0) + adversary probes `adv_a/p1–p8`,
`adv_b/p1–p5` (the v2 re-pins: τ₇ scaling, NC configs, jitter parabola, deficit
non-monotonicity, node-level rows). Knife edges: 509×767 / 512² / 513² / 257² /
1024² / 1080p / 4K; ds {0.25, 1.3, 3.0}; aperture {0, 0.023, 0.060}; dispersion
on/off; env {0, 1, 3}; pool + dry-screen; masks zeros/ones/const/hard-half/
feather/blob/noise/2.0/−0.5/NaN. NC configs FORBID `2^n+1` sizes (identity regime).

| # | invariant | check | negative control (fires at its pinned config) |
|---|---|---|---|
| W1 | mask-absent is today's node | (a) same-engine: frozen numpy oracles vs numpy-pinned build, bitwise; (b) node-level golden capture, bitwise, both outputs | const-0.5 mask ≠ oracle |
| W2 | zero mask is identity | IMAGE bitwise == `input[..., :3]` (holds for out-of-range inputs — §1c.3); deficit exactly 0 | ones differs on both outputs |
| W3 | locality, BOTH outputs | `{out ≠ in} ⊆ {m > 0}`, `{deficit > 0} ⊆ {m > 0}`, exact, node level with grain_restore on | no-select with **hard-half mask** at 509×767: numpy fires on the overhang column (509 px, 0.52); GPU zeros-mask at 509×767 fires at 8e-5. Zeros-mask numpy variant is DEAD (recorded) |
| W4 | full-res ones ≡ absent | expected bitwise (measured 0.0 across the 40-combination box incl. node level); tooth ≤ τ₄ = 1e-5 | const 0.5 differs (0.93) |
| W5 | the multiply is the spec's | per-engine seam recompute: `m·dxp`, `m·dyp`, `jamp_m` bitwise against the SAME engine | `m²` variant differs (0.985) |
| W6 | jitter from MASKED Jacobian | `jamp_m == 0` on interior m=0 (exact); MEAN jitter monotone at c 0.25/0.5/1 (0.459/0.659/0.833); NO pointwise claim (§2 parabola, counterexample recorded) | inherit-unmasked-jamp: 93.2% of m=0 zone jittered at the pinned compression config |
| W7 | Fresnel gated | flat-water oracle ≤ τ₇ⁿ = 1e-12 / τ₇ᶜ = max(3e-5, 1.5e-7·max(H,W)), sizes through 4K | ungated at m=0.5, env 1.0, MID-TONE frame: gap 8.35e-3 (saturated/black configs are dead — forbidden for the NC) |
| W8 | deficit composition | (a) exact 0 at m=0, node level; (b) evaluated ≠ multiplied, floor `gap > 0.10` at c=0.5 (measured 0.468, seed-spread 0.07) | (b) is the control; NO monotonicity row (§4) |
| W9 | resize plumbing | const 0.7 @ 64² == full-res ≤ τ₉ = 1e-9 (measured ≤ 1.31e-11, 76× margin); ones through 512²→1024×1536: min > 1−1e-12, no dead strip; exact zeros survive both directions | default-mode zoom kills an edge strip (min 0.0) |
| W10 | batch semantics | M=2/B=3, BOTH vary_per_frame settings: frame_i bitwise == a SOLO run with mask[min(i,M−1)] **and the solo seed carrying the node's own stride (s + 7919·i) when vary_per_frame is on** — otherwise the row tests pour seeds, not pairing (adjudication erratum) — AND frame0 ≠ frame1 (v1's row passed trivially — frames can match for surface-cache reasons); M=0 and all-zero → warn paths | shuffled pairing differs |
| W11 | solver isolation | seam: `h_mm` bitwise between masked and unmasked runs, same seed | a solver-side `h·m` variant differs |
| W12 | clamp + NaN | 2.0 ≡ 1.0 bitwise; −0.5 ≡ 0 bitwise; NaN-pixel mask → zero NaN in either output | 0.5 ≠ 1.0 |
| W13 | ride-along §7.12a | `grain_deficit(depth_scale=0.25)` ≠ default on a shallow pool (measured 0.50 pointwise); default arg bitwise-preserves today's I9 rows | the shipped no-depth_scale form differs from the forwarded one at ds 0.25 |

### 8b. Measurement record

v1's dry-run (47/0) plus the two adversaries' independent probe sets, all on the
embedded python (3.13.9, numpy 2.4.2, torch 2.10.0+cu130, RTX A4000), CUDA
run-twice bitwise repeatability confirmed (licenses every CUDA bitwise row).
Node-level rows W1/W2/W3/W4 were measured by adversary B against a spec-verbatim
build: all bitwise, with `grain_restore=1.0` on. Headline numbers live in the rows
above; the probe scripts are the provenance (`adv_a/`: jitter parabola, τ₇ size
law, NC dead zones, deficit non-monotonicity + seed spread, restore coupling,
feather law, plumbing sweeps; `adv_b/`: live bugs, node I/O, 4-D masks, seams,
engine oracles, timings).

Exhibits (after build, my eye first, then his): real photo, pool + dry-screen,
before / hard-half (the seam line shown at its measured size) / feathered-blob
(f = 0.25·max|D|) / full; one strip driving Film Grain Pro from the gated deficit.

---

## 9. Direction calls in this spec (for Jeremie, at sign-off)

1. **Destination-sampled mask** (slice-2 law verbatim) over field-scaling — §1.
   The two differ only within ~2–4 px of a mask edge at reference apertures.
2. **Fresnel sheen gated by the mask** (§3) — the vignette-precedent call.
3. **Deficit gated by evaluation + hard select** (§4), accepting the stated
   boundary step (the restored image stays continuous; the gate is the point).
4. **restore_grain's ≤2% inside/outside coupling accepted** rather than paying a
   second full render to remove it (§4).
5. **Three ride-along fixes ship with the slice** (§7.12): deficit depth_scale
   forwarding (behaviour change for ds≠1 users — the current output is simply
   wrong), and the two console prints.
6. **The engine sampler divergence (bicubic vs bilinear) goes to open-questions**
   as its own future decision — not silently ratified by this slice (§6).
7. Widget named `mask`; no performance shortcut under the mask.
8. **DECIDED BY JEREMIE AT THE GATE (2026-08-17)**: mask semantics are an
   intensity remap with a floor (`mask_min`, default 0.3), not a hard gate —
   §11. `mask_min = 0` recovers calls 1–3's exactness contracts.

## 11. Post-gate amendment (2026-08-17, Jeremie's eyeball-gate direction call)

At the exhibit gate Jeremie rejected the hard-gate SEMANTICS: *"it should be more
gradual, everywhere on but more intense where the mask is 1 and more subtle where
it is 0."* The mask is an INTENSITY REMAP, not a gate. Amendment:

```
m_eff = mask_min + (1 − mask_min) · m^mask_gamma
        mask_min default 0.15, mask_gamma default 2.0
```

**Second live-test refinement (same day):** his Field-driven feathered mask read
near-uniform — at a 12mm standing pool even 0.3× the displacement fully
scrambles fine detail (the W14 saturation lesson, now user-visible), and a
feathered mask is mostly mid-grey. Two levers, both his ask ("a bit more in the
white zones and less in black, just not 0 or 1"): the floor dropped to 0.15,
and `mask_gamma` (γ>1) pulls the mid-greys toward the floor while white stays
at 1 — contrast between zones without touching the endpoints. `mask_gamma = 1`
SKIPS the power op entirely (structurally bitwise linear regime); gamma applies
to the clamped mask BEFORE the floor, in exactly this op order;
everything downstream (per-sample multiply, masked Jacobian, Fresnel gate, deficit
evaluation, both selects) consumes `m_eff` unchanged. Consequences, derived:

- **Every §1–§4 law survives verbatim** — all of it was proven for arbitrary mask
  values in [0,1], and `m_eff ∈ [mask_min, 1] ⊂ [0,1]`.
- **`mask_min = 0` restores the hard gate bitwise**: `0.0 + m·1.0 = m` exactly
  (multiply by 1.0 and add 0.0 are exact for m ≥ 0). Every m==0 exactness
  contract (W2/W3/W8a, the selects) is conditional on `mask_min = 0` and the
  teeth pin it there.
- **`mask_min = 1` is ones**: `1.0 + m·0.0 = 1.0` exactly → bitwise == absent
  (W4's chain).
- **At the default 0.3 nothing is identity and the deficit has a floor
  everywhere** — intended: the effect exists across the frame, the mask pushes
  it. The all-black console warn keys on the EFFECTIVE mask (fires only at
  mask_min = 0).
- **mask-absent stays bitwise-untouched** (no remap runs).
- Teeth: `node_cfg` pins `mask_min = 0.0` for every gate-contract row; new W14
  block: (a) default is 0.3; (b) the remap law — `node(mask=m, mask_min=c)`
  bitwise == `node(mask=c+m·(1−c) [float64], mask_min=0)`, NC a wrong-constant
  remap; (c) `mask_min=1` bitwise == absent; (d) behaviour: at default, the
  black zone differs from the input (subtle effect present) and its mean
  |change| is below the white zone's (subtle vs intense — his sentence as an
  assertion).

## 10. Adversarial pass record (2026-08-16)

Two fresh-eyes Opus adversaries (math lens; ComfyUI-reality lens), everything
measured on the embedded python. Combined vs v1: **11 spec-fatal, ~16 material.**
Headlines: v1's uniform-jitter monotonicity was derivationally wrong (det parabola;
pointwise counterexamples on 24/24 configs); τ₇'s flat CUDA pin failed at 1080p/4K
(size-scaled law measured); W7's NC was dead at v1's own named configs (env-3
clip saturation); the deficit monotonicity invariant was false (saturation ceiling
+ probe-seed spread larger than the trend — the lattice scar again); W3's NC was
dead on numpy at every size with v1's stated mechanism (the real fuel is the
overhang column, needing an edged mask); v1's feather law `f ≥ 2·max|D|` was
unrealisable at its own config, measured on a degenerate mask, and 4–10× over the
measured requirement (replaced by `f ≈ 0.25·max|D|`, artefact restated as a
one-column seam); the W1 frozen-numpy oracle could never pass on a CUDA machine
(render_auto dispatch; engine-pinned + golden-run split); the unsqueeze shape rule
crashed on real 4-D masks (ComfyUI core's reshape adopted); v1's
"restore-calibration is mask-independent" and "shear band destroys detail the
unmasked warp never touched" were both measured false and struck; the single
engine-agnostic seam could not meet a bitwise tooth (split per engine). Three live
shipped bugs found (deficit depth_scale, console dmax, console fold%) → §7.12;
one shipped divergence named for its own future decision (§6).

**Held under attack** (one line each): the core destination-scaling model and its
slice-2 inheritance; mask=ones bitwise across the full 40-combination box including
node level; zeros-identity and locality at node level with grain restore on; fold
behaviour under partial masks (no new artefact class, measured across the c
lattice); one-mask-for-dispersion (ratios unchanged); the m=0 select architecture
(all four reasons measured); the M=0 guard; interior-m=0 jitter exactly zero; W9's
resize sweep (~980 size pairs, zero failures, τ₉ 76× margin); τ₇'s numpy pin; the
RNG-stream constraint's satisfiability; the 5-minute teeth budget (large headroom);
"no performance change" (measured at 1080p and 4K); signature-change safety for the
existing 39/39 suite.
