# Spectral B&W (Ortho / Pan) — derivation & scope sign-off

> A black-and-white conversion driven by a film's SPECTRAL SENSITIVITY type:
> orthochromatic renders red dark and blue light (white skies, dark skin/lips),
> panchromatic renders naturally, extended-red lightens foliage/skin. DERIVED from
> real spectral sensitivity behavior + the Mallett & Yuksel 2019 sRGB spectral
> basis (computed OFFLINE via colour-science; no runtime dep). Signed off (Opus)
> 2026-06-14 before code. (dirty-physics #3, the last cleanly-grounded one.)

## What this is (and the honest limit)

The existing B&W film-stock node applies a fixed conversion; nothing lets you choose
the spectral SENSITIVITY (the "color of the eye" of the film), which is what makes a
B&W image read as 1900s ortho vs modern pan. This node maps RGB→gray using weights
derived from a chosen sensitivity curve.

THE HONEST LIMIT (state in the node): a B&W weight is physically `∫ S(λ)·I(λ) dλ`
(sensitivity × the scene's per-pixel spectrum). sRGB gives only 3 numbers per pixel,
NOT the spectrum `I(λ)` — you cannot recover it (metamerism). So this is an
approximation via spectral upsampling, the SAME approximation every spectral renderer
makes. We deliver the ortho/pan TONAL CHARACTER, not a claim of reproducing what that
film did to a real-world spectrum. Pseudo-IR is the most stylized (sRGB carries no IR).

## LOAD-BEARING INSIGHT — the chain is LINEAR ⇒ collapses to a per-type RGB weight

Spectral upsampling (Mallett-Yuksel) reconstructs a plausible spectrum as a LINEAR
combination of 3 fixed basis spectra (one per sRGB primary), exact-round-tripping under
D65: `I(λ) ≈ R·b_r(λ) + G·b_g(λ) + B·b_b(λ)`. Then the gray response is

```
gray = ∫ S(λ)·I(λ) dλ = R·∫S·b_r + G·∫S·b_g + B·∫S·b_b = w_r·R + w_g·G + w_b·B
```

So for a FIXED sensitivity `S(λ)`, the whole spectral computation is EXACTLY a fixed
RGB→gray weight triple `w = (∫S·b_r, ∫S·b_g, ∫S·b_b)`. This is what makes the node a
PRINCIPLED channel mixer: the weights are spectrally derived, not hand-picked. Runtime =
a single weighted sum (fast, no spectral lib); the spectral work is done ONCE offline.

## Offline derivation of the weights (build step, via colour-science)

`colour-science` IS importable in the embedded python (used for OkLab validation). Use
it as an OFFLINE TOOL (not a runtime dep) to compute the weight triples:
1. Get the Mallett-Yuksel sRGB basis spectra: `colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019`
   (3 SDs, one per sRGB primary, over a shared wavelength range).
2. Define each sensitivity type `S(λ)` over the same range (analytic curves below).
3. `w_c = trapz(S(λ)·b_c(λ), λ)` for c in {r,g,b}; then NORMALIZE so `sum(w)=1`
   (keeps a neutral grey at roughly its luminance; the relative weights carry the look).
4. Hardcode the resulting triples as the node's per-type constants; keep the derivation
   script at `tools/derive_spectral_bw_weights.py` (regenerable, documented).

### Sensitivity types S(λ) (analytic; cutoffs from the ortho/pan literature, ~nm)
- **Blue-sensitive** (collodion/early): ~360–500, peak ~440. Red≈0, green low.
- **Orthochromatic**: ~360–590 (cuts red ~590). Red≈0, full blue+green. The classic.
- **Orthopanchromatic**: ~360–650, REDUCED red shoulder. Mild red.
- **Panchromatic** (modern): ~360–680, full visible, roughly flat-ish with the film
  luminosity. Natural rendering.
- **Panchromatic+ / extended-red** (pseudo-IR, e.g. SFX/Rollei): ~360–720+ with a RED
  emphasis. Reds/foliage/skin render LIGHT. (Most stylized: sRGB has no real IR, so this
  just up-weights red strongly — label as a pseudo-IR approximation.)
Curves = smooth windows (e.g. raised-cosine / logistic shoulders) over those ranges; the
exact shape is a modeling choice, the cutoffs are the grounded part. Document.

(Optional v1.x: swap the analytic pan curve for the REAL Tri-X / 5222 sensitivity that
`third_party/spectral_film_lut/` already vendors — defer; analytic types ship first.)

## Runtime model (the node)

```
lin  = srgb_to_linear(img)
w    = _WEIGHTS[sensitivity]            # precomputed triple, sums to 1
gray = w_r·R + w_g·G + w_b·B            # linear-light weighted sum (clip ≥0)
out  = stack([gray,gray,gray])          # grayscale
out  = linear_to_srgb(clip(out,0,1))
result = blend(original, out, strength) # strength<1 = partial desaturation toward color
```

## Controls (v1)
| control | default | range | note |
|---|---|---|---|
| `sensitivity` | "Panchromatic" | [Blue-sensitive, Orthochromatic, Orthopanchromatic, Panchromatic, Panchromatic+ (extended red)] | the film's spectral "eye" |
| `strength` | 1.0 | 0–1 | blend color→B&W (1 = full B&W, <1 = partial desaturation) |

Early-exit: `strength<=0` → return input. `[Darkroom]` print. CATEGORY =
`AKURATE/Darkroom/Film`. Class `DarkroomSpectralBW`, display "Spectral B&W (Ortho/Pan)".
This IS LUT-bakeable (pointwise, fixed weights) → add to the bake-allowed list.

## Teeth before trust (`tools/test_spectral_bw.py`)
1. **ORTHO SIGNATURE (headline):** Orthochromatic on a pure-red vs pure-blue patch →
   red renders DARKER than blue (red→dark, blue→light). NEGATIVE CONTROL: a flipped
   weight (red up-weighted) makes red brighter → the test FAILS on the flip.
2. **TYPE ORDERING:** red-patch gray value increases across [Blue-sensitive ≤ Ortho ≤
   Orthopan ≤ Pan ≤ Pan+] (red gets progressively lighter as red sensitivity grows);
   blue-patch the opposite trend (or at least Pan+ red > Ortho red, clear separation).
3. **GRAYSCALE OUT:** output R==G==B everywhere.
4. **NEUTRAL PRESERVED:** a neutral grey in → ~same grey (weights sum to 1).
5. **WEIGHTS SANE:** each type's weights ≥0, sum≈1; ortho `w_r`≈0; pan more balanced;
   Pan+ `w_r` largest. (Assert on the derived constants.)
6. **strength=0 → identity; strength=0.5 → partial desaturation** (chroma reduced, not zero).

## v1.x — measured Tri-X / 5222 curves (SHIPPED 2026-07-17)

Two new `sensitivity` entries, `"Kodak Tri-X 400"` and `"Kodak 5222 (Double-X)"`, swap
the analytic S(λ) window for a REAL datasheet-digitized spectral sensitivity curve.
Same runtime model, same offline-derivation pipeline (§ "Offline derivation of the
weights" above) — only the input curve changes.

**Source + license.** The measured curves come from the vendored
`third_party/spectral_film_lut` (Jan Lohse, MIT license; see `third_party/
spectral_film_lut/ATTRIBUTION.md`). Each stock module stores its curve as a
`log_sensitivity = [{wavelength_nm: log10(S)}]` literal on a `FilmData(...)` call.

**Extraction method.** `tools/derive_spectral_bw_weights.py` § 2b (`_load_measured_points`)
AST-parses the stock module and `ast.literal_eval`s the `log_sensitivity` keyword —
it does NOT import or execute the vendored package. Reason: the vendored `__init__`
pulls in `numba`, which is incompatible with the embedded numpy in this environment,
and the stock module itself calls `dataclasses.replace` against the real `FilmData`
class at import time. Literal extraction sidesteps both without touching vendored
runtime code.

**Alignment policy** (`measured_S_on_grid`): linearize `S = 10**log_sensitivity`;
resample onto the Mallett-Yuksel wavelength grid with linear interpolation INSIDE the
measured range, and linear extrapolation of the LOG-sensitivity (not linear S) beyond
both ends — the measured curves fall off steeply in log space near the shoulders, so
log-domain extrapolation decays smoothly instead of cliffing to zero or blowing up.
The curve is peak-normalized (`S / S.max()`) before integration; this is cosmetic only
— the absolute scale cancels in the sum-to-1 normalization at the end. Integration
against the Mallett-Yuksel basis and the sum-1 normalization are IDENTICAL to the
analytic pipeline (`derive()`).

**Derived triples** (verbatim from `tools/derive_spectral_bw_weights.py`, copied into
`nodes/spectral_bw.py::_WEIGHTS`):
```
"Kodak Tri-X 400":       (0.210565, 0.357127, 0.432308)
"Kodak 5222 (Double-X)": (0.176784, 0.312235, 0.510981)
```

**Validation.**
- Pan-range check: both `w_r` values sit strictly between 0.15 and Panchromatic+'s
  `w_r` (0.382438) — plausible modern-pan territory, not ortho, not pseudo-IR.
- Red-patch ordering: both stocks render a pure-red patch strictly between the
  Orthochromatic gray (darkest) and the Panchromatic+ gray (lightest); 5222 (lower
  `w_r`) renders red darker than or equal to Tri-X, matching the curve shapes.
- Negative control (reversed-curve): re-deriving 5222 with its `log_sensitivity`
  points reversed across wavelength shifts the weight triple by up to 0.2932 (max
  abs component shift) — proof the measured data is actually driving the result, not
  an artifact of the pipeline plumbing.
- 5222-bluer-than-Tri-X: 5222's lower `w_r` / higher `w_b` vs Tri-X tracks the two
  stocks' real curve shapes (5222 is the more blue-weighted cine stock of the pair).

**The film-true story.** Both measured stocks render BLUER than the idealized analytic
"Panchromatic" curve — larger `w_b`, smaller `w_r` — because real panchromatic film
sensitivity is not flat across the visible range the way the v1 analytic window
assumes; it tapers into the red. This is exactly why photographers shot yellow (or
red) filters over pan film: to correct the film's native blue-heavy response back
toward how the eye perceives contrast (darker skies, more natural skin tones). The
measured entries make that historical filter practice visible for the first time in
this node — skies render lighter and reds/skin render darker than the idealized
"Panchromatic" preset, which is the correct, film-true direction.

**Honest limit (unchanged in kind, restated for measured data).** These weights are
still a real measured S(λ) run through the SAME sRGB spectral-upsampling
approximation as the analytic types (§ "What this is (and the honest limit)" above).
Measuring the real sensitivity curve does not remove the metamerism problem — sRGB
still gives only 3 numbers per pixel, not the scene's spectrum. So "measured" upgrades
the INPUT curve's fidelity, not the overall claim: still tonal character, not a
reproduction of what Tri-X or 5222 did to a real-world spectrum.

## Still deferred (v1.2+)
- Colored-filter control (red/yellow/green filter over B&W) — the classic darkroom
  contrast tool; a nice complement to the measured stocks above.
- True IR — needs IR data sRGB does not carry; no path without a different input.
