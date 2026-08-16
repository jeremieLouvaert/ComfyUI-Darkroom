# Native MASK inputs for the four geometric lens nodes — derivation

**Version: v2 (post-adversarial-attack, 2026-08-16).** v1 was attacked by two fresh-eyes Opus
adversaries per the rigor protocol; their combined kill list (11 spec-fatal, ~14 material — §6)
is folded in here. **Status: built under plow-ahead. Jeremie's sign-off on the direction calls
(§7), his eyeball gate, and his commit word are all still open doors.**

Runs in Darkroom's version line (registry 1.19.0 -> 1.20.0). Scope: mask-audit slice 2 per
`comfyui-brain/procedural-plan.md` §9d/§9e. Approved 2026-08-05 (the original three) +
2026-08-15 (Perspective Correct added). Water Refraction is slice 3, OUT of scope.

Targets, all in `ComfyUI-Darkroom/nodes/`:

| node | file | warp engine | padding | interp |
|---|---|---|---|---|
| Lens Distortion | `lens_distortion.py` | torch `grid_sample` | zeros | bicubic |
| Chromatic Aberration | `chromatic_aberration.py` | torch `grid_sample` | reflection | bicubic |
| Lens Profile | `lens_profile.py` | torch `grid_sample` | reflection | bicubic |
| Perspective Correct | `perspective_correct.py` | scipy `map_coordinates` | constant 0 | bilinear (order=1) |

Plus two adjacent one-line fixes the adversarial pass surfaced (§5.9): the shipped Long
Exposure mask-resize edge-strip bug, and Lens Profile's uninitialised 4th channel.

---

## 1. The model

All four nodes are **pull-back warps**: for every destination pixel `p` they compute a source
coordinate `W(p)` and set `out(p) = I(W(p))`. Define the **displacement field**

```
D(p) = W(p) - p        (in pixels, evaluated at the destination pixel)
```

The masked warp scales the displacement field before sampling. **The universal masked form,
all four nodes — this is the implementation law, not just math**: compute the unmasked source
coordinates EXACTLY as today's code does, then

```
D     = src − p                  (elementwise, the same tensors)
src_m = p + m ⊙ D                (this exact op order: difference, multiply, add)
```

- `m(p) = 0` → the pixel samples its own location → original pixel.
- `m(p) = 1` → the full effect.
- Intermediate values interpolate the *geometry*, not the *output*: blending outputs gives two
  ghosted copies of every feature; scaling the displacement gives one partially-displaced copy.

**Composite displacement, not per-stage.** Lens Profile compounds distortion × CA into one
`total_factor`; Perspective Correct chains rotation → vertical → horizontal. The mask
multiplies the FINAL composed displacement, never the individual stages (per-stage masking
composes the mask with itself — `m²` terms — and has no one-sentence semantics). The sentence
we can state: *the source lookup moves `m(p)` of the way from the pixel's own location to where
the unmasked node would have looked.*

### 1a. Where the mask is sampled — destination, derived not assumed

`m` is read at the **destination pixel** `p`, where `D` is evaluated. (1) The warp is defined
per destination pixel; reading `m` at the source would mean `m(W_m(p))` — a fixed-point
equation with no closed form and no uniqueness guarantee. (2) Destination sampling preserves
the contract a user can reason about: *paint where you want the effect to land in the output.*

### 1b. What a mask edge does — the shear bound, corrected and stated honestly

The Jacobian of the masked warp is `J(p) = I + m·∇D + D ⊗ ∇m`. The third term is new: it
shears the sampling geometry wherever the mask has a gradient, with magnitude `|D|·|∇m|` —
and it is **signed** along the edge normal: the local stretch factor ranges over
`[1 − |D|/f, 1 + |D|/f]` for a feather of width `f` px. Consequences, measured (§6, F5/F15):

- Where `∇m` is anti-parallel to `D` (half of any closed mask boundary), `f = max|D|` drives
  `det J` to **zero** — total collapse (measured: det J = 0.028, a 35× local compression, at
  exactly that feather). The v1 guidance recommended precisely this value. **Corrected rule:
  `f ≥ 2·max|D|` bounds the two-sided stretch to [0.5×, 2×].**
- A hard edge produces a band of width ≈ `|D|` px along the edge: **duplicated** content where
  `D` points into the mask, **deleted** content on the other side (measured: a 16-px stripe
  pattern loses a stripe at a hard seam). The deletion case is the uglier artifact and is now
  stated.
- Fold-over (`det J < 0`, measured at 6.5% of a band at `f = max|D|/2`) duplicates source
  content in a pull-back warp rather than crashing; confined to the same band.

**Maximum displacement — honest forms** (corner evaluations are NOT maxima; v1's were wrong
for mixed-sign coefficients, measured 0-predicted vs 388-px-actual at k1=2, k2=−1):

- Lens Distortion: `|D|(r) = |k1·r³ + k2·r⁵| · R` with `r ∈ [0, √2]` normalised radius,
  `R = sqrt(cx²+cy²)`; take the max over `r` (interior extrema exist whenever k1, k2 have
  opposite signs — the advertised "mustache" setting).
- Chromatic Aberration: `|D|max = max(|shift_r|, |shift_b|) · strength · min(H,W)/1024` px
  (verified exact).
- Lens Profile: the compounded `(distort·ca_factor − 1)·r·R` — note the CA sub-term runs on
  the normalised-frame radius (max √2), NOT the CA node's `r/max_r` (max 1), and the cross
  term does not vanish; evaluate the compound, do not sum the two nodes' formulas.
- Perspective Correct: **unbounded within widget range** — the horizontal stage consumes the
  post-vertical `nx`, so `1 + h·nx` can reach zero (measured: |D| → ∞ at v=0.5, h=+0.5;
  97,000 px at v=0.5, h=−0.5). Feather guidance for PC is therefore qualitative: at strong
  combined keystone the required feather approaches the frame size, i.e. the artifact must be
  accepted or the mask kept away from the singular corner.

Tooltips state the guidance in these honest terms (§5.10) — "feather by twice the displacement;
at strong distortion or keystone that can be hundreds of pixels, so soften wide or accept the
band", never a false promise.

### 1c. Exact preservation at m = 0 — the select rule

```
out = where(m == 0, original, processed)      # applied once, after ALL processing
```

Two independent reasons, both measured:

1. **Torch nodes**: `grid_sample` resampling at the identity coordinate is bitwise only when
   `(H−1)` and `(W−1)` are powers of two — Darkroom's `pixel_to_grid_coords` normalises by
   `(size−1)` under `align_corners=True`, so identity is exact at 513×513/1025×1025 and NOT at
   512×512/1024×1024 (measured: up to 6e-5; the widely-cited "H,W powers of two" rule is the
   `align_corners=False` convention's and does not apply here — patterns.md 2026-08-16).
   Without the select, m=0 pixels differ on ~24% of a 509×767 frame.
2. **Perspective**: v1 called the select "mathematically redundant" for PC. **Wrong** — at
   singular parameter combinations (v=0.5, h=0.5, within widget range) the unmasked source
   coordinate is `inf` and the D-form computes `0·inf = NaN` at m=0 pixels. The select is
   load-bearing there (measured: 1 NaN pixel without it, 0 with it). At non-singular configs
   PC's m=0 identity IS exact without the select (float64 coords, exact-zero multiply,
   order-1 weights exactly (1,0) at integer coords) — but "usually redundant, occasionally the
   only thing between the user and a NaN" is exactly what a belt is for.

Continuity: at `m = ε` the displacement is `ε·D` (sub-pixel), so the select introduces no
visible step. Note the select passes the ORIGINAL input through at m=0 pixels, unclamped even
for out-of-range inputs — consistent with the nodes' own early-outs, which return the raw
input tensor (`return (image,)`), and stated here rather than discovered.

### 1d. mask = 1 is equivalence within tolerance, NOT bitwise — and why

The D-form reassociates: `p + 1.0·(src − p) ≠ src` in IEEE in general. So: **mask-absent →
bitwise** (the None path executes today's code untouched), **mask=ones → equal within
τ₄ = 1e-5**. Measured: the difference was bitwise ZERO at every pixel of every config tried —
509×767, 512×512, 1080×1920, 2160×3840, smooth/noise/checker content, both devices, both
adversaries' independent runs (a Sterbenz-adjacent cancellation: `src − p` is exact when the
operands' exponents are close). Four-plus configs are still not a proof, so the claim stays a
tolerance (the 2c lucky-sample scar), with the measured zero recorded as expected-typical.

---

## 2. Per-node derivations

### 2a. Lens Distortion

Unmasked: `factor = 1 + k1·r² + k2·r⁴`, `src = c + d·factor`, `D = d·(factor − 1)` in pixels.
Masked: the §1 universal form on `(src_y, src_x)`.

- The early-out (`|k1|,|k2| < 0.001` after strength scaling → return input) is unchanged and
  runs BEFORE any mask handling; a connected mask on an early-out call changes nothing (the
  effect is nil regardless).
- **padding_mode='zeros' caveat, now stated**: strong distortion inside a mask near the frame
  border pulls black in from outside the frame exactly as the unmasked node does — but with a
  mask it can sit hard against a pristine m=0 region (measured: a 2549-px black wedge at the
  mask boundary on a white frame at k1=0.35). Tooltip warns; behaviour is the shipped node's
  own and is not changed.

### 2b. Chromatic Aberration — one mask, three channels

Unmasked per channel `c ∈ {R,G,B}` with shifts `(s_r, 0, s_b)`: `scale_c = 1 + (s_c/max_r)·r`.
Masked: the universal form per channel, **the same single mask tensor `m` for every channel**.

Why per-channel masks are wrong: lateral CA is a wavelength-dependent radial magnification —
the R:B displacement ratio `D_r/D_b = s_r/s_b` is a property of the glass, constant across the
frame (verified exact at every pixel). One mask preserves that ratio everywhere; independent
masks would manufacture red/blue relative displacement along mask edges — painting fringes no
lens produces. Green (`D_g = 0`) is untouched by warp or mask, bitwise, as today.

The per-channel skip (`|shift| < 0.01` after scaling → channel untouched) keys on the unmasked
scalar shift, unchanged: `m` only reduces `|m·shift|`, so a skipped channel stays skipped and
the skip remains a whole-channel decision. The `strength <= 0` early-out precedes mask handling.

### 2c. Lens Profile — geometry masked, vignette gated too (one mask, one meaning)

Geometric part per channel: `total_factor = distort · ca_factor`; masked via the universal form.

**Vignette: gated by the same mask.** `V` is **the SHIPPED factor exactly as computed today** —
Add mode's already-clamped `(1 − transition·(1−falloff)·vig·2).clamp(0,1)`, Correct mode's
unclamped correction — and the gate replaces the multiply:

```
V_m = 1 + m · (V − 1)         # both modes; the final result.clamp(0,1) is unchanged
```

A builder folding `m` inside Add mode's clamp gets a different (wrong) result wherever the
clamp is active; hence "shipped factor" is pinned (§6 F14). Why gate at all: (1) one mask means
one thing — "apply this lens's signature here"; (2) with the vignette gated, m=0 means FULL
identity, so §1c's select corrects float noise instead of erasing a real effect (ungated, it
would). The alternative (geometry-only masking) is rejected but available to users by chaining
Darkroom's own Vignette node, which is tonal and externally maskable via Field Composite.

**The saturation dead zone, stated (§6 F13)**: in Correct mode the final `clamp(0,1)` means
that wherever even the GATED value blows out (`V_m · pixel ≥ 1`), gated and ungated are
identical — the mask has no visible effect on the vignette there. Measured: at vig·strength
≥ ~1.0 on bright corners, most of the correction range is clipped. Tooltip states it. Both
early-outs (`strength < 0.01`, unknown lens name) precede mask handling and return the input;
for the unknown-lens path that is the intent (no profile, no effect, mask irrelevant).

### 2d. Perspective Correct — the shipped warp is NOT a homography, and the spec stops claiming it

v1 claimed the chain rotation → vertical → horizontal is projective and derived a tooltip
promising that constant-mask regions keep lines straight. **Both adversaries proved this false
independently**: the vertical stage `(x,y) → (x/(1+v·y), y)` has no homogeneous 3×3 form (the
y-row would need a quadratic), and the destination image of a straight line is a hyperbola.
Measured: at v=0.37, lines bow by **60 px at 767-px width, 131 px at 1920** with NO mask; a
true homography gives ~1e-12; rotation alone gives exactly 0 (it is projective). The shipped
node's own docstring ("projective (homography) transformation") is the pre-existing mislabel
v1 inherited without checking.

v2 statement: the shipped warp is a keystone-style **rational warp** that approximates a
homography at small settings and visibly bows long lines at strong ones — shipped behaviour,
out of this slice's scope to change. The mask adds a second, separate bending term at mask
gradients (`|∇m|·|D|`, §1b). The tooltip therefore promises nothing about straightness:
"partial mask values bend lines through the mask edge; strong keystone bows long lines even
without a mask." Rotation is displacement-masked like everything else.

**auto_crop is ignored when a mask is connected.** auto_crop crops the transform border and
zooms the frame back — a GLOBAL resample that moves every pixel including m=0 ones, violating
I2/I3 and reintroducing exactly the problem this slice exists to solve. Behaviour: mask
connected → auto_crop treated as False; one console line, printed once outside the batch loop
and only when the widget was actually True:
`[Darkroom] Perspective: mask connected, auto_crop ignored`. Tooltip on both widgets states it.

---

## 3. Mask I/O semantics — the Long Exposure precedent, with two stated departures

Per `long_exposure.py` (the only shipped Darkroom mask input):

| rule | Long Exposure | this slice |
|---|---|---|
| widget | optional `MASK` named `subject_mask` | optional `MASK` named `mask` (departure 1: the semantic here is "where the effect applies", not a subject; stated, not silent) |
| absent | untouched original code path | same — **bitwise** (I1) |
| 2-D mask `(H,W)` | unsqueeze to `(1,H,W)` | same |
| batch pairing | frame `i` gets mask `min(i, M−1)` | same, plus a guard: `M = 0` (empty mask batch) is treated as absent with a console warning (the precedent IndexErrors) |
| values | clipped to `[0,1]` | same (clamp) |
| dtype/device | implicit | pinned: `mask.to(device=image.device, dtype=image.dtype)` before any use (a float64 mask otherwise crashes `grid_sample`; measured) |
| resize on mismatch | `scipy.ndimage.zoom`, order=1, default mode | **departure 2, both engines fixed**: torch nodes `F.interpolate(mode='bilinear', align_corners=True)` then clamp; Perspective `zoom(order=1, mode='nearest')` then `[:H,:W]` crop and clip |

**Departure 2, derived.** Three constraints force it: (a) the precedent's default
`mode='constant'` zoom **zeroes a full frame-edge row/column at 9 of 95 realistic size pairs**
(the last output coordinate lands marginally outside the input and reads cval=0) — a user's
all-ones mask arrives with a dead strip that the m=0 select then pins to the original
(measured: I4 fails by 0.999; this is also a live bug in shipped Long Exposure, fixed in §5.9);
`mode='nearest'` removes it at every firing pair. (b) `align_corners=False` interpolation uses
half-pixel-centre mapping while `zoom` uses align-corners mapping — they select measurably
different regions from the same user mask (up to 0.66 in mask value, 4.4% of the frame's
exact-zero set); `align_corners=True` matches zoom semantics to 1.5e-7 on gradient masks AND is
the pack's own convention (`pixel_to_grid_coords`). (c) GPU nodes must not roundtrip through
scipy (their documented no-CPU-roundtrip contract). Cross-engine agreement under the fixed
conventions is a TOOTH (I13, τ₁₃ = 7.4e-4, measured max 1.84e-4 on binary-noise downscales),
and **exact-zero regions stay exactly zero under BOTH engines** (measured; the select contract
survives resize). Residual honesty: on an 8× binary-noise upscale the two engines' exact-zero
footprints differ on 89 px of 390k (interior near-boundary pixels, one engine's exact 0 vs the
other's 1e-9); the contract is per-node exact, cross-node approximate, stated.

Degenerate frames (H or W = 1) NaN in the SHIPPED unmasked nodes already (`(size−1)` division);
they are out of contract for the mask path too, unchanged.

---

## 4. Invariants — every row executed BEFORE sign-off

Executed by `_lens_mask_dryrun/dryrun.py` (v1 rows) + `dryrun_v2.py` (v2 re-pins) on the
embedded python, plus both adversaries' independent probes (`adv_a/`, `adv_b/`). Knife edges:
509×767 (odd, non-pow-2, also non-(2^n+1)), 512×512, 2160×3840, awkward constants k1=0.35/
k2=−0.12, shifts −3.7/+2.9, v=0.37/h=−0.22/rot=7.3°, the singular corner v=h=0.5, masks:
zeros/ones/const/hard-half/31-px-feather/blob/binary-noise/mask=2.0. PC rows are CPU-only
(numpy engine). Teeth run the same rows at the same pinned configs.

| # | invariant | check | negative control (fires at its pinned config) |
|---|---|---|---|
| I1 | mask-absent is today's node | `mask=None` output **bitwise ==** frozen-reference (vendored copies of today's `_apply_*`), all 4 nodes | ones-mask with strong params ≠ frozen reference |
| I2 | zero mask is identity | `mask=zeros` **bitwise ==** input (raw input passes through, unclamped — §1c) | ones differs |
| I3 | locality | `{out ≠ in} ⊆ {m > 0}`, exact | torch: no-select variant at 509×767 (fired: 93k–100k px). PC: no-select at the singular v=h=0.5 (fired: NaN at an m=0 pixel). Inert at 513×513 by design — pinned sizes only |
| I4 | full mask ≡ unmasked | `mask=ones` vs `mask=None` ≤ τ₄ = 1e-5 (measured 0.0 everywhere; stays a tolerance, §1d). **PC: `auto_crop=False` on BOTH arms** (the default-True arm measured 0.96 apart — a config trap, not a defect) | const 0.5 exceeds τ₄ (0.76) |
| I5 | displacement scales linearly | seam returns `(src, src_m, m)` per node (per channel where channels differ); tooth recomputes `p + m ⊙ (src − p)` in the spec's op order, requires **bitwise** equality — the refactor guard | `m²` variant differs |
| I6 | uniform mask ≡ strength | mask=const c vs `strength·c` run ≤ τ₆ = 4.0e-3 (measured max 1.005e-3, 4K CUDA), LD + CA. **Domain-pinned**: configs where BOTH arms sit ≥10× above every early-out/skip threshold (k·c ≥ 0.01; shift·c·scale ≥ 0.1). Outside that domain the arms legitimately diverge because the nodes' early-outs key on the scaled scalar — expected behaviour, documented, NOT a tooth | c=0.5 vs strength·0.8 (0.76) |
| I7 | CA one-mask coherence | seam recompute (I5 form) for R and B **with the same mask tensor**, green channel bitwise untouched | per-channel-mask variant breaks the R:B displacement ratio where masks differ (measured gap 0.91) |
| I8 | LP vignette gated | flat-gray, vignette-only profile, const mask 0.5, **mid-radius sample (r≈0.75) where the GATED value is strictly inside (0,1)**: measured == `gray·(1 + 0.5·(V−1))` ≤ τ₈ = 1e-6, both modes (measured 4.0e-8 / 2.8e-8). V is the shipped factor (§2c) | ungated variant gap ≥ 5e-3 (fired: 0.21 Add / 0.24 Correct) |
| I9 | resize plumbing | const 0.7 mask at 64×64 == const 0.7 full-res ≤ τ₉ = 2.7e-3 (measured 6.7e-4 at 4K; plumbing only — kernel discrimination lives in I13) | const 0.3 differs |
| I10 | batch semantics | M=1/B=3: each frame **compared bitwise** against its solo run; M=2/B=3: pairing is `min(i, M−1)` (frame 2 gets mask 1), bitwise vs explicit runs; M=0 treated as absent + warn | shuffled pairing differs |
| I11 | PC auto_crop policy | mask connected: `auto_crop=True` output bitwise == `auto_crop=False` output (build-time tooth — the policy lives in `execute()`) | mask=None: auto_crop=True ≠ False at pinned params (verified live, 0.96) |
| I12 | clamp | mask=2.0 ≡ mask=1.0 bitwise; mask=−0.5 ≡ mask=0 bitwise | mask=0.5 ≠ mask=1.0 |
| I13 | cross-engine resize | torch(acT bilinear) vs scipy(zoom nearest) on disc + binary-noise masks at 5 pinned size pairs incl. 512²→1024×1536 ≤ τ₁₃ = 7.4e-4 (measured 1.84e-4); all-zero regions exactly 0 under both | nearest-KERNEL torch variant exceeds τ₁₃ on the disc (kernel discrimination — v1's const-mask row could not see it) |
| I14 | LE resize fix | all-ones 512² `subject_mask` resized to 1024×1536 has min > 1−1e-12 | default-`mode='constant'` variant: min == 0.0 (the shipped bug, must stay visible) |

No straightness tooth exists because v2 makes no straightness claim (§2d); line bending is an
exhibit. The §1b shear band and deletion-side artifact are exhibits with the predicted band
width drawn. Real-photo before/half/full strips per node complete the exhibit set.

## 5. Implementation constraints for the builder

1. **The `mask=None` path executes today's arithmetic UNCHANGED** — same ops, same order (I1
   bitwise by construction). Masked logic lives in the `mask is not None` branch only.
2. **Early-outs run BEFORE mask handling** in all four nodes and return the input unchanged.
3. **Universal D-form** (§1) with the exact op order; **declared seams**, private, callable
   without `execute()`:
   - LD: `_masked_coords(h, w, k1, k2, m, device) -> (src_y, src_x, src_y_m, src_x_m)`
   - CA: per active channel, `{c: (src_y, src_x, src_y_m, src_x_m)}`, every channel using the
     SAME mask tensor object (skipped channels absent from the dict)
   - LP: per channel `{c: (...)}` from the compounded `total_factor`
   - PC: numpy float64 `(src_y, src_x, src_y_m, src_x_m)`
4. **Mask prep helper, one per engine**: clamp(0,1) → `.to(device=image.device,
   dtype=image.dtype)` (torch) / float64 (PC) → 2-D unsqueeze → `M==0` → treat as absent +
   console warn → per-frame `min(i, M−1)` → resize per §3 (torch: bilinear acT + clamp;
   PC: zoom order=1 `mode='nearest'` + `[:H,:W]` + clip).
5. **The §1c select** applied ONCE per frame, after all processing (incl. vignette), against
   the pristine input: `torch.where((m == 0.0).unsqueeze(-1), original, processed)` / numpy
   analog.
6. **Rename `_apply_lens_profile`'s internal local `mask`** (the vignette factor, lines 74-75)
   to `vig_factor` — mandatory; the widget name `mask` would otherwise be shadowed/rebound and
   §5.5's select would silently read the wrong tensor. No behaviour change.
7. **Vignette gating**: `V` is the shipped factor exactly as computed today (Add: the clamped
   expression; Correct: the unclamped one); gate as `V_m = 1 + m·(V − 1)` at the multiply;
   final `result.clamp(0,1)` unchanged (§2c pins which-V; I8 enforces).
8. **PC auto_crop policy** per §2d: ignored when mask connected; console note once, outside
   the batch loop, only if the widget was True.
9. **Two adjacent fixes, teeth-backed, no other behaviour change**:
   a. `long_exposure.py:133`: `zoom(..., order=1)` → `zoom(..., order=1, mode='nearest')`
      (the dead-edge-strip bug, I14; `tools/test_long_exposure.py` must stay green).
   b. `lens_profile.py:44`: `result = torch.empty_like(img)` → `result = img.clone()` — a
      4-channel IMAGE currently gets uninitialised memory in channel 3 (adversary-measured);
      clone passes alpha through instead. No change for 3-channel inputs (channels 0-2 are
      fully overwritten).
10. **Tooltips** (house register, honest, no em dashes):
    - LD `mask`: "Where the distortion applies, 0 to 1. Scales the warp itself, so partial
      values do not ghost. Feather the mask edge by about twice the local displacement (tens
      to hundreds of px at strong settings) or a smear band forms along the edge. Strong
      distortion near the frame border can pull black in from outside the frame, masked or
      not."
    - CA `mask`: "Where the fringing applies, 0 to 1. One mask drives all channels together so
      the red/blue ratio stays lens-true. A few pixels of feather is enough at typical
      shifts."
    - LP `mask`: "Where the lens character applies, 0 to 1, vignette included. Mask 0 leaves
      pixels untouched. In Correct mode, corrections that clip to white are unaffected by the
      mask where they clip."
    - PC `mask`: "Where the correction applies, 0 to 1. Partial values bend lines that cross
      the mask edge, and strong keystone bows long lines even without a mask. auto_crop is
      ignored while a mask is connected."
    - PC `auto_crop` (append): "Ignored when a mask is connected."
11. **README**: the four node rows gain a one-phrase mask mention (Long Exposure's row is the
    template); nothing else. No new nodes; `NODE_CLASS_MAPPINGS` untouched; registry manifest
    unchanged except `pyproject.toml` 1.19.0 → 1.20.0 prepared in the working tree (commit and
    registry publish remain Jeremie's doors).
12. **Teeth architecture**: new suite `tools/test_lens_masks.py`, implementation-blind, written
    from THIS document only. It vendors frozen copies of today's four `_apply_*` functions as
    I1 oracles (they are pure functions of tensors; copy them verbatim into the suite before
    the build lands). Existing suites: verified to have ZERO coverage of the four target nodes
    — the new teeth are the only net; `tools/test_long_exposure.py` must stay green after
    §5.9a.

## 6. Adversarial pass record (2026-08-15/16)

Two fresh-eyes Opus adversaries (math lens; ComfyUI-reality lens), everything measured on the
embedded python. Combined: **11 spec-fatal, ~14 material, ~10 minor** against v1. Headlines:

- The shipped Perspective warp is NOT a homography (x-only division; lines bow 60-131 px at
  constant mask) — v1's §2d premise and tooltip were false; inherited from the node's own
  docstring without checking. [both adversaries, independently]
- The precedent's `scipy.zoom` default mode zeroes frame-edge mask strips at 9/95 size pairs —
  v1 canonised a live Long Exposure bug; the v1 64×64-only resize config was the exact
  lucky-sample that cannot fire.
- v1's τ₆/τ₉ broke at 4K (up to 2×); τ₈'s Correct-mode row ignored the final clamp (15/24
  configs fail as written); the I8 corner sample sat in the saturated regime.
- The feather guidance `f ≥ max|D|` recommended the det-J-zero collapse point; corrected to
  `f ≥ 2·max|D|` with the deletion-side artifact stated.
- The |D|max "corner formulas" were not maxima (0-predicted vs 388-actual at mustache
  settings; PC unbounded within widget range).
- The select IS load-bearing for PC (0·inf = NaN at singular configs) — v1's "mathematically
  redundant" struck.
- The `grid_sample` identity rule is convention-dependent: (size−1) pow-2 under acT — v1's
  "pow-2 sanity" config was the same regime as the odd sizes (patterns.md extended).
- I6 needed domain pinning (early-outs key on scaled scalars); I7's bitwise form was
  unachievable from a coordinates-only seam (seam contract unified); I9 was kernel-blind
  (I13 added); I10's harness rows were vacuous (real comparisons pinned); the two resize
  engines disagreed by up to 0.66 in mask value under v1's conventions (aligned in v2).
- Buildability: the LP local-`mask` shadowing trap; dtype unpinned (float64 mask crashes);
  M=0 IndexError; README out of scope; existing suites provide zero regression net for these
  nodes (verified).

**Held under attack** (both adversaries, one line each): the core displacement-scaling model
and its ghosting argument; destination mask sampling; composite-not-per-stage; the m=0 select
(both engines, for different reasons than v1 gave); I4's measured-bitwise-zero (attacked at
4K/noise/checker/negative-strength, never broken — stays a tolerance on principle); CA
one-mask physics (ratio exact at every pixel); the CA skip-monotonicity argument; vignette
gating structure and its m=0 coherence argument; the auto_crop-under-mask policy; the Long
Exposure precedent reading; §5.1's None-path bitwise-by-construction; CA/LP reflection padding
raising no border issue; the CA and vertical-keystone |D| formulas; I2/I3/I5/I12 at every size
and mask thrown at them.

## 7. Direction calls in this spec (plow-ahead: adopted-with-rationale, awaiting Jeremie)

1. Vignette gated by the mask in Lens Profile (§2c) — the 9e-recommended call, adopted.
2. auto_crop ignored under a mask in Perspective (§2d).
3. Widget named `mask` (departure from `subject_mask`, semantic differs).
4. Two adjacent fixes ride along (§5.9): the Long Exposure resize bug (adversary-found, live,
   user-facing) and the Lens Profile 4-channel uninitialised memory.
5. Perspective's non-homography truth stated in doc + tooltip; the node's warp itself is NOT
   changed in this slice (shipped behaviour).
