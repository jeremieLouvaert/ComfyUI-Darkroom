# Halftone — derivation & scope sign-off

> Professional AM (clustered-dot) halftone screening for ComfyUI-Darkroom.
> Pure numpy + torch-on-CUDA, classic public-domain screening math (no code
> borrowed from any repo — surfaced via Discover lateral run 2026-06-12,
> Trolzie/halftone, but reimplemented house-style like Newson grain / ACES).
> Signed off (Opus) 2026-06-14 before code, per derive-before-code.

## What this node is (and is NOT)

A creative print-screening effect: it reproduces continuous tone as a grid of
ink dots whose SIZE modulates with tone (amplitude modulation), the iconic
newsprint / comic look. Fills a real gap — Darkroom has a CMYK *proofing*
workflow (soft-proof, gamut, TAC, export) and grain, but no halftone screen.

It is NOT a calibrated prepress proof (that's `cmyk_softproof`). The CMYK
separation here is the naive stylize separation, no ICC. Document as a STYLIZE
effect.

## Core algorithm — AM clustered-dot screening

For each ink channel with continuous coverage `c(x,y) ∈ [0,1]` (c=1 → full ink):

1. **Resolution-independent screen frequency.** `lines` = halftone lines across
   the long edge (resolution-independent, like Newson μ_r∝L). Dot pitch in px:
   `p = max(long_edge / lines, 3.0)` (floor at 3px so dots never go sub-pixel —
   the visibility-floor lesson from Newson grain).

2. **Rotated screen coordinates** at the channel's screen angle θ:
   ```
   u = ( x*cosθ + y*sinθ) * (2π / p)
   v = (-x*sinθ + y*cosθ) * (2π / p)
   ```

3. **Euclidean (round) dot spot function** — the classic clustered-dot field
   whose level sets are round dots near cell centers and round holes near
   corners (this dot↔hole transition is what gives the full tonal range):
   ```
   D = (cos(u) + cos(v)) / 2          # ∈ [-1, 1], MAX (=1) at cell centers
   T = (1 - D) / 2                    # ∈ [0, 1], MIN (=0) at cell centers
   ```
   `T` is the per-pixel threshold; dots grow outward from cell centers as `c↑`.

4. **Ink** = `c > T` (1 where inked). Anti-aliased by SUPERSAMPLING: evaluate
   steps 2–4 on a grid `ss×` finer (ss = `supersample`, default 2), then
   box-average each ss×ss block → `ink ∈ [0,1]` with smooth dot edges.

## Color model (forks resolved)

- **mono** (default — the newsprint identity): screen `1 - luma_rec709` (dark →
  more ink) at one angle (`angle`, default 45°, the classic newspaper angle).
  Output black ink on white: `R=G=B = 1 - ink`.
- **color (CMYK)**: naive stylize separation
  ```
  c = 1-R ; m = 1-G ; y = 1-B
  k = black_generation * min(c,m,y) ; c-=k ; m-=k ; y-=k   (clamp ≥ 0)
  ```
  Screen each at its **standard rosette angle** (C=15°, M=75°, Y=0°, K=45° — the
  30° separation that avoids moiré), then recombine subtractively on white:
  ```
  R = (1-c_ink)*(1-k_ink) ; G = (1-m_ink)*(1-k_ink) ; B = (1-y_ink)*(1-k_ink)
  ```
  CMYK angles are FIXED (the rosette is standard; exposing 4 angle sliders is
  clutter). `angle` applies to mono only.

## Load-bearing design calls (Opus sign-off)

1. **Screen in DISPLAY (sRGB) space, NOT linear.** Halftone reproduces tone as
   seen / as printed; linear-light screening would misplace dots and crush the
   tonal distribution. So this node does NOT `srgb_to_linear` — it screens the
   display-space values directly. (Deliberately different from the grading nodes,
   which work in linear; documented here so it isn't "fixed" by mistake.)
2. **AM clustered-dot only in v1.** FM / error-diffusion (Floyd-Steinberg,
   stochastic, no angle) is a different aesthetic AND is inherently serial
   (slow/unvectorizable in numpy) — deferred to v1.x. The angled rosette IS the
   iconic look; ship that, tight scope (Newson "pixel-wise only" discipline).
3. **Round dot only in v1.** square / line / ellipse spot functions = v1.x.
4. **Supersample AA (default 2), not analytic soft-threshold.** Supersampling is
   provably-correct AA and trivially GPU-parallel; the "do the real version" call.
5. **torch-on-CUDA for the screen eval, numpy fallback** (mirror `linear_to_srgb`
   in utils/color.py). The screen grid at ss× on a 4K image across 4 CMYK channels
   is large; numpy alone is likely seconds-to-slow. HARD PERF GATE after the
   prototype: ~1024² in well under a second and ~4K in a couple seconds on GPU,
   tolerable CPU fallback. If numpy-only misses it, the torch path is mandatory
   (same gate discipline as Newson grain).

## Controls (v1)

| control | default | range | note |
|---|---|---|---|
| `image` | — | IMAGE | |
| `color_mode` | "mono (black)" | ["mono (black)", "color (CMYK)"] | mono = newsprint; color = CMYK rosette |
| `lines` | 100 | 20–400 (INT) | halftone lines across the long edge (resolution-independent) |
| `angle` | 45.0 | 0–90 (FLOAT, deg) | mono screen angle; ignored in CMYK (fixed rosette) |
| `black_generation` | 1.0 | 0–1 | CMYK only; 1 = full GCR (K plate), 0 = CMY only |
| `supersample` | 2 | 1–4 (INT) | dot-edge anti-aliasing; higher = smoother + slower |
| `strength` | 1.0 | 0–1 | blend original↔halftone (house `blend`); <1 = subtle screen overlay |

## Pipeline (house flow)

```
img (B,H,W,C sRGB 0..1)
  → tensor_to_numpy_batch
  per image:
      original = img
      # DISPLAY space — no srgb_to_linear (see design call 1)
      if mono:  ink  = screen(1 - luma_rec709(img), angle, lines, ss);  out = 1 - ink (3ch)
      else:     separate→CMYK; screen each at rosette angle; recombine subtractively
      result = blend(original, out, strength)
  → numpy_batch_to_tensor
```
Early-exit: `strength <= 0` → return input. `[Darkroom]` print on active path.
CATEGORY = `AKURATE/Darkroom/Print` (match the cmyk_* nodes; composes with the
print workflow). Display name "Halftone", class key `DarkroomHalftone`.

## Teeth before trust — `tools/test_halftone.py` (offline, embedded python)

No external oracle, but halftoning has hard invariant properties. The headline
teeth directly guard the NodeForge halftone INVERSION bug class:

1. **Tonal monotonicity / NO INVERSION (headline).** Feed a horizontal black→white
   gradient; the per-column mean OUTPUT luminance must increase monotonically
   with input luminance (dark in → dark out). NEGATIVE CONTROL: flip the screen
   comparison (`c < T`) and assert this test FAILS — proves it has teeth.
2. **Tone preservation.** A heavy Gaussian blur of the halftone output must
   approximate the input image (dots carry the right average ink). Tolerance
   ~0.05–0.1 mean abs over a smooth test image.
3. **Endpoints.** Pure white in → ~white out (near-zero ink); pure black in →
   ~black out (near-full ink).
4. **Binary-ish ink @ ss=1.** Before AA, the ink mask is ~binary (dots, not a
   continuous ramp) — histogram mass near 0 and 1.
5. **CMYK angles differ.** The four channel screens use distinct angles (no
   single-angle moiré) — sanity check, not a tight assert.

Run on real embedded torch/python; report PASS/FAIL honestly incl. the negative
control and the perf-gate timing.

## Deferred (v1.x / v2)
FM / error-diffusion mode; square/line/ellipse dot shapes; per-channel CMYK angle
overrides; ICC-accurate separation (use cmyk_softproof for proofing).
