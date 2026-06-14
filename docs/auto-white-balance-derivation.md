# Auto White Balance (Color Constancy) — derivation & scope sign-off

> Automatic white balance: estimate the scene illuminant from the image and divide
> out the color cast. Classical, public-domain illuminant-estimation algorithms,
> pure numpy, house-style (no learned weights, MIT-clean). Surfaced by a Discover
> lateral run 2026-06-14 (a learned color_constancy repo pointed at the gap; the
> classical algorithms fill it cleanly). Signed off (Opus) 2026-06-14 before code.

## What this node is (and is NOT)

The first AUTOMATIC white balance in Darkroom. The existing White Balance node is
MANUAL (Kelvin temperature + tint, `AKURATE/Darkroom/Raw`). Auto WB estimates the
illuminant from the image content and corrects the cast with no user temperature
input — the "auto WB" button every camera/raw-developer has, that Darkroom lacks.

It is a classical statistical estimator, NOT a learned/neural method (those need
weights + a license we don't have). It will not beat a trained CNN on hard cases,
but the classical methods are robust, instant, dependency-free, and cover the
common "remove the obvious color cast" need. Document honestly.

## The unifying framework (van de Weijer et al., "Edge-Based Color Constancy")

All four methods are one formula with two knobs — derivative order `n` and Minkowski
norm `p`. The illuminant estimate per channel `c`:

```
e_c = ( ∫ |∂ⁿ f_c(x)|ᵖ dx )^(1/p)          (then normalize e to unit direction)
```

- **Gray World**   = (n=0, p=1): `e_c = mean(f_c)`  — "the average scene is gray".
- **White Patch**  = (n=0, p=∞): `e_c = max(f_c)`   — "the brightest patch is white"
  (Max-RGB / Retinex). Use a high PERCENTILE (default 97th) not the true max, for
  robustness to hot pixels / specular clipping.
- **Shades of Gray** = (n=0, p): `e_c = (mean(f_cᵖ))^(1/p)`, p≈6 (Finlayson & Trezzi).
  Generalizes Gray World (p=1) ↔ White Patch (p=∞); p=6 is the cited robust default.
- **Gray Edge**    = (n=1, p): same Minkowski norm but on the per-channel GRADIENT
  magnitude `|∇f_c| = sqrt(gx² + gy²)` (van de Weijer); usually the best classical
  method. Default p=6.

One internal `_minkowski(values, p)` does all of it; methods select (use-gradient?, p).

## Correction (von Kries diagonal) — work in LINEAR light

Illuminant estimation + the diagonal correction are physically about light, so do it
in linear sRGB (like the manual WB node): `srgb_to_linear → estimate → correct → back`.

1. `lin = srgb_to_linear(img)`
2. estimate `e = (e_r, e_g, e_b)` per the method on `lin` (clip e_c ≥ 1e-6).
3. **gain to remove the cast while preserving brightness:** `gain_c = mean(e) / e_c`.
   (A channel the scene is biased toward — larger `e_c` — gets scaled DOWN; the mean
   normalization keeps overall exposure ~constant, so this is a pure cast removal,
   not a brightness change.)
4. `corrected = clip(lin * gain, 0, 1)`  (broadcast gain per channel)
5. `out = linear_to_srgb(corrected)`
6. `result = blend(original, out, strength)`   (house `blend`)

Why `mean(e)/e_c` (gray-world-style normalization): it makes the corrected channel
statistics neutral while fixing green≈unchanged-ish and preserving mean brightness.
For Gray World specifically this makes the three corrected channel MEANS exactly equal
(the defining property — a strong teeth test).

## Controls

| control | default | range | note |
|---|---|---|---|
| `image` | — | IMAGE | |
| `method` | "Shades of Gray" | [Gray World, White Patch, Shades of Gray, Gray Edge] | robust default beats plain Gray World |
| `minkowski_p` | 6.0 | 1–16 (FLOAT) | the p for Shades of Gray + Gray Edge (ignored by Gray World p=1 and White Patch percentile) |
| `strength` | 1.0 | 0–1 | blend original↔corrected (house) |

White Patch's robustness percentile is fixed internally at 97 (not exposed — keep the
UI lean). `minkowski_p` is only meaningful for SoG/Gray Edge; document in its tooltip.

Early-exit: `strength <= 0` → return input. `[Darkroom]` print on the active path
(include method + the estimated gains, useful feedback). CATEGORY = `AKURATE/Darkroom/Raw`
(sits beside manual White Balance). Class `DarkroomAutoWhiteBalance`, display
"Auto White Balance".

## Edge cases (must hold)
- **Neutral image stays neutral:** a balanced (already-gray-world-neutral) image → all
  `e_c` equal → `gain ≈ 1` → output ≈ input. No false correction.
- **Pure gray / flat image:** gradients are ~0 for Gray Edge → guard div-by-zero
  (clip e_c ≥ eps); degrade to gain≈1 (no change), never NaN.
- **Clipped highlights** bias White Patch → the 97th-percentile (not max) mitigates.

## Teeth before trust — `tools/test_auto_white_balance.py` (offline, embedded python)

No external oracle, but color constancy has hard checkable properties:
1. **CAST RECOVERY (headline):** take a balanced test image, apply a KNOWN illuminant
   cast (e.g. warm R×1.3, B×0.7), run Auto WB, and assert the corrected image's
   channel-mean imbalance (max_c mean_c / min_c mean_c) is much closer to 1 than the
   cast image's. NEGATIVE CONTROL: `strength=0` (no correction) must LEAVE the
   imbalance (assert it does NOT recover) — proves the test detects real correction.
2. **GRAY WORLD EXACTNESS:** after Gray World, the three corrected channel MEANS are
   equal within ~1e-3 (its defining property).
3. **NEUTRAL STAYS NEUTRAL:** a balanced image → output ≈ input (mean abs diff small,
   gains ≈ 1); no cast introduced.
4. **BRIGHTNESS PRESERVED:** mean luminance of corrected ≈ cast image's (gain
   normalization), within a tolerance.
5. **ALL METHODS run + output valid [0,1]**, no NaN on a flat-gray image (div-zero guard).

Run on the real embedded python; report PASS/FAIL honestly incl. the negative control.

## Deferred (v1.x / v2)
Gray Edge 2nd-order (n=2); local/grid-based constancy (per-region illuminant); a
learned method (out of the house no-weights constraint); auto-tint beyond the diagonal.
