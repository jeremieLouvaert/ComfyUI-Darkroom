# Reciprocity Failure — derivation & scope sign-off

> Simulate film reciprocity failure (Schwarzschild): long-exposure speed loss +
> the per-channel color cast (the night-photography shift no digital sensor has) +
> shadow crush. DERIVED from real Kodak E-31 datasheet tables (extracted in the
> 2026-06-14 dirty-physics scout → `comfyui-brain/research/2026-06-14_darkroom-dirty-physics.md`)
> + Fuji datasheets. Pointwise per-channel (no spatial op → no scar risk). Pure numpy.
> Signed off (Opus) 2026-06-14 before code.

## What this is (and is NOT)

Simulates "this scene shot on film X at exposure time T." The film loses sensitivity
non-linearly at long exposures and the loss differs per emulsion layer, producing the
characteristic long-exposure COLOR CAST + crushed shadows. The existing film-stock
nodes model the daylight response; none model reciprocity (a time-dependent effect).

DATASHEET-GROUNDED CHARACTER, not per-roll calibrated. Manufacturers publish discrete
correction points (1s/10s/100s); we interpolate in log-time. The per-film casts are
encoded from the published CC-filter corrections (real data), but the shadow-crush
curve is our defensible interpretation of "shadows lose speed faster," not a published
curve. Honest label in the node. Aging/fog (qualitative-only in the literature) is
DEFERRED to a separate node, not bundled here.

## LOAD-BEARING CALL — the color-shift DIRECTION (do not invert this)

E-31 lists the CC (colour-compensating) filter you ADD to NEUTRALIZE the film's cast.
So the film's NATIVE long-exposure cast = the COMPLEMENT of the recommended filter.
The simulation applies that native cast (the inverse of the correcting filter):

- A CC filter of density `d` in colour X attenuates X's opponent channel by `10^(-d)`.
  CC magenta absorbs GREEN; CC yellow absorbs BLUE; CC cyan absorbs RED (and R/G/B
  filters attenuate their own channel's complement similarly).
- Recommended filter = magenta (CCxxM) ⇒ film cast is GREEN ⇒ SIMULATE by boosting the
  green channel by `10^(+d)` (the inverse of the corrector). Likewise:
  CCxxR (red, used to correct a CYAN cast) ⇒ simulate CYAN (boost G+B / cut R);
  CCxxY (yellow, corrects BLUE cast) ⇒ simulate BLUE (boost blue);
  CCxxG (green, corrects MAGENTA cast) ⇒ simulate MAGENTA (boost R+B / cut G).
- Mapping CC code → per-channel linear gain: density `d` (e.g. CC05M → d=0.05) →
  affected channel gain `g = 10^(+d)` to SIMULATE the cast (and/or `10^(-d)` on the
  complement to keep luminance roughly neutral). Normalize gains so mid-grey luminance
  is ~preserved (the cast is a color shift, not a brightness change — exposure is
  assumed compensated).

TEETH must assert the SIGN: e.g. an Ektachrome-class film at long exposure produces a
CYAN/BLUE shift (recommended CC was red/yellow), NOT red. A flipped-direction negative
control must fail. This is the halftone-inversion lesson applied.

## Model (pointwise, per channel, in linear light)

```
lin = srgb_to_linear(img)
# 1. interpolate the film's correction at exposure_time T (log-time):
#    stop_loss s(T), cc_gain[r,g,b](T), [optional contrast c(T)]
# 2. COLOR CAST (grounded): lin_c *= cc_gain[c]  (per channel), then renormalize so
#    Rec.709 luminance of a neutral is preserved (cast only, no net brightness change).
# 3. SHADOW CRUSH (interpretation of speed loss): a pivoted power curve that darkens
#    shadows, amount ∝ s(T). e.g. lin = lin ^ (1 + k·s)  near 0, easing to identity at 1
#    (a toe-weighted curve; preserves highlights, crushes low values). k tunes feel.
# 4. out = srgb_encode; result = blend(original, out, strength)
```

`exposure_time` picks the magnitude from the film table by log-time interpolation; below
the film's reciprocity-onset time the correction is ~0 (no effect = the daylight region).

## Film presets (from E-31 + Fuji datasheets; the data table)

Each preset = points {time_s: (stop_loss, cc_code_or_none)}, log-interpolated. Honest:
approximate, datasheet-character-grounded; B&W = no color, just crush+contrast.

- **B&W (general Kodak)**: 1s→(+1, none, −10%dev), 10s→(+2,−20%), 100s→(+3,−30%). Color: none.
- **Kodak T-Max**: 1s→(+⅓), 10s→(+½), 100s→(+1). No color, no dev adj. (Milder.)
- **Kodak Portra 400**: ~none to 1s, very mild beyond. Minimal cast.
- **Kodak Ektachrome E100**: 1s→(+⅓, CC025R), 100s→(+2, CC10Y+CC025R). Cast = CYAN/BLUE
  (inverse of red+yellow).
- **Fuji Provia 100F**: none to 128s; 240s→(+⅓, CC2.5G). Cast = MAGENTA (inverse of green).
- **Fuji Velvia 50**: strong time-only loss (4s→5s, 30s→66s, 60s→150s, 120s→290s); known
  green cast → simulate with a mild magenta-correct, i.e. GREEN cast. (Approx; the chart
  is time-only, the cast direction is the documented Velvia long-exposure green.)

(`exposure_time` range ~0.5–600s. A "None/Custom" entry = pure manual via the knobs if
we expose manual cast/crush; v1 may skip Custom.)

## Controls (v1)

| control | default | range | note |
|---|---|---|---|
| `film` | "Kodak Portra 400" | the presets above | reciprocity character |
| `exposure_time` | 1.0 | 0.5–600 (FLOAT, s) | drives magnitude via the film's log-time table |
| `strength` | 1.0 | 0–1 | house blend |

Early-exit: `strength<=0` OR (the interpolated correction at `exposure_time` is ~0, i.e.
below the film's reciprocity onset) → return input. `[Darkroom]` print (film + the
applied stop/cast). CATEGORY = `AKURATE/Darkroom/Film`. Class `DarkroomReciprocity`,
display "Reciprocity Failure".

## Teeth before trust (`tools/test_reciprocity.py`)

1. **CAST DIRECTION (headline, grounded):** Ektachrome E100 @ long exposure → output mean
   shifts CYAN/BLUE (G+B up relative to R), NOT red. Provia @ 240s → MAGENTA (R+B up vs G).
   B&W → no chroma shift. Assert the sign per film. NEGATIVE CONTROL: a flipped-direction
   build fails this.
2. **MONOTONIC WITH TIME:** the cast magnitude + shadow crush increase with exposure_time
   (1s ≤ 10s ≤ 100s), and below onset (short t) the effect ≈ 0 (≈ identity).
3. **LUMINANCE ~PRESERVED:** the color cast does not change overall brightness much (cast,
   not exposure) — mean luminance within a tolerance of input (shadow crush aside).
4. **SHADOW CRUSH:** long exposure darkens shadows more than highlights (toe-weighted) —
   a dark patch drops more than a bright patch.
5. **NEUTRAL-ISH IN, CAST OUT:** a neutral grey ramp picks up the film's cast in midtones,
   stays monotonic (no inversion), endpoints sane.
6. **strength=0 / short-time → identity** (early-exit).

## Deferred (v1.x / v2)
Film aging / base-fog growth + pepper grain (qualitative-only literature → a separate
heuristic node); per-channel Schwarzschild `p` exponents fit from the tables (vs the CC
direct-encode used here); Custom manual cast.
