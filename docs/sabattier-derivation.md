# Sabattier Effect (Solarization) — derivation & production spec

**Status:** SIGNED OFF 2026-06-15 (Opus scope + offline LUT/look spike). Dirty-physics line, "inspired/composed" tier.
**Class:** `DarkroomSabattier` · **Display:** "Sabattier" · **Category:** `AKURATE/Darkroom/Film` · count 53 → 54.
**Recipe (research digest 2026-06-14):** *"Sabattier / solarization — COMPOSE: non-monotonic LUT ∘ silver-mask re-expose ∘ Eberhard kernel."* Tier B, after Eberhard.

This node **composes already-shipped, already-eyeballed code** (the Eberhard CSF adjacency engine for Mackie lines) with a new tone LUT. No iterative PDE, no generated stochastic texture → it sidesteps the two scars that have hit this line (PDE units-misses ×2, the Lith leopard-grain wall). The only load-bearing correctness call is the **tone-reversal direction** (halftone-inversion bug class), pinned below and guarded by a flipped negative control in the teeth.

---

## 1. The physics (why solarization reverses)

The Sabattier effect: develop partially → **re-expose the emulsion to a uniform fogging light** mid-development → continue developing.

- Areas with **high first-development silver** (the already-dark parts of the positive) physically **shield** the underlying halide from the fogging light, and locally exhaust the developer → they receive **little** added density.
- Areas with **low silver** (the bright parts of the positive) are **unshielded** → the fog exposes them → they **gain density and darken**.

So in a finished **positive** image with tone `x` (linear luminance, 0 = black, 1 = white):

- **Shadows** (x→0, already dark, high silver, shielded) → stay dark.
- **Highlights** (x→1, bright, low silver, unshielded) → get fogged → **darken / reverse**.

**LOAD-BEARING DIRECTION (pinned): the reversal darkens the HIGHLIGHTS; the deepest shadows are preserved.** This is the famous Man-Ray look: bright background goes gray/dark with a light Mackie line outlining the subject. A flipped-direction control (reversing the shadows instead) is in the teeth as a negative control.

### Why it is non-monotonic (the log is the mechanism)

Densities **add** linearly; reflectance/transmittance is **logarithmic** in density. That log is *why* the curve folds (a purely linear silver model gives only monotonic darkening — verified in the spike).

Work in **linear light** (transmittance/reflectance are linear quantities; also consistent with the Eberhard engine).

```
D0(x)   = -log10(x)            # first-development density of the positive
Dfog(x) = beta * x^k           # fog density that PENETRATES existing silver:
                               #   light reaching the halide ∝ transmittance = x^k
D(x)    = D0(x) + Dfog(x)
x_out   = 10^(-D) = x * 10^(-beta * x^k)
```

- `beta` = **re_exposure** — how much fogging light → depth of the reversal.
- `k` = **shield** — how sharply developed silver protects. Higher k confines the reversal to the very brightest tones (threshold-like); lower k spreads it into the midtones.

`x_out(x) = x · 10^(−β·x^k)` is 0 at x=0, rises to a bright ridge at an interior x*, then folds back down toward `10^(−β)` at x=1. **Fold confirmed in spike:** for β=1,k=1 → black@0=0.001, ridge@x≈0.45, highlight@1 reversed.

### 2. The "printing" step (REQUIRED — key spike finding)

`x_out ≤ x` everywhere (fog only ever *adds* density), so the raw LUT is **globally dark and muddy** (spike: peak preserved tone capped at 0.16 linear). A real Sabattier print is then **printed to use the full paper tonal range**. We replicate this by normalizing the LUT so its peak maps to 1.0:

```
ymax  = max over t∈[eps,1] of  t * 10^(-beta * t^k)     # 256-sample sweep, once per call
x_out = (x * 10^(-beta * x^k)) / ymax
```

Without this the effect reads as "dark + slightly folded" (muddy); with it the fold spans the full range and has punch. The deepest black (x≈0) still maps to ≈0; the rising-limb shadows are lifted (expected — printing stretches the achieved range). This is a grounded "printing" step, not a fudge.

## 3. Mackie lines (compose the Eberhard engine)

The other iconic Sabattier feature: bright **Mackie lines** tracing contours, from lateral developer/bromide gradients at density edges. Reuse the **shipped Eberhard CSF adjacency** verbatim.

**Composition (spike finding): derive the edge field from the ORIGINAL luminance, not the post-fold tone.** After the fold, adjacent regions collapse to similar tones (tiny `hp`) → lines vanish if computed post-fold. Physically the Mackie line forms from the *original* density gradient, so:

```
sigma = edge_width * (long_edge / 1024.0)          # resolution-clean, same as Eberhard
m     = gaussian_blur(L_orig, sigma)
hp    = L_orig - m
delta = where(hp > 0, mackie_intensity*hp, (mackie_intensity/asym)*hp)   # asym fixed = 6.0
L_m   = clip( x_out + delta , 0, None )
```

`asym` (overshoot/undershoot ratio) is **fixed internally at 6.0** (the Eberhard filmic default) to keep the control surface focused on the solarization story; exposing it is a v1.x option.

## 4. Color & recombine

Operate on luminance; reapply to RGB by **ratio** (hue-preserving), exactly like Eberhard:

```
lin   = srgb_to_linear(img);  L = luminance_rec709(lin)
... compute L_m ...
ratio = L_m / max(L, 1e-6)
out   = clip(lin * ratio[...,None], 0, 1)
out   = linear_to_srgb(out)
result = blend(original, out, strength)
```

Color/chromatic solarization (the fog shifting hue in colour materials) is **v1.x**; v1 is hue-preserving and honest about it.

## 5. Controls

| control | range | default | meaning |
|---|---|---|---|
| `re_exposure` | 0.0–3.0 | 1.0 | β — depth of the tonal reversal (0 = no solarization) |
| `shield` | 0.5–4.0 | 1.0 | k — sharpness/confinement of the reversal to highlights |
| `mackie_intensity` | 0.0–2.0 | 0.5 | Mackie-line strength (Eberhard adjacency) |
| `edge_width` | 0.5–8.0 | 2.0 | Mackie-line scale (Eberhard sigma @1024, scales with long edge) |
| `strength` | 0.0–1.0 | 1.0 | blend original→processed |

Early-return identity if `strength<=0` **or** (`re_exposure<=0` **and** `mackie_intensity<=0`).

## 6. Honesty label (node docstring + README)

INSPIRED/COMPOSED tier. The **direction and non-monotonic mechanism are physically grounded** (silver shielding + density-adds-log-reflectance); the **"printing" renormalization, the shield exponent, and the Mackie composition are tuned to the look**, NOT per-developer-calibrated chemistry. Delivers the Sabattier *character* (highlight reversal + Mackie lines), not a named-process simulation.

## 7. Teeth (the Sonnet build must pass ALL on embedded python, incl. firing negative controls)

1. **FOLD / direction (headline):** on a 0→1 luminance ramp, `argmax(x_out)` is strictly **interior** (reversal exists) AND `x_out[-1] < max(x_out)` (highlights reversed). **Negative control:** `re_exposure=0` → identity LUT → argmax at the last index → assertion FIRES.
2. **Deepest black preserved:** `x_out[0] < 0.02`.
3. **Identity:** `strength=0` → bit-identical; and `re_exposure=0 & mackie_intensity=0` → ≈identity (fp).
4. **Renormalization / not-muddy (key spike finding):** for an image whose tones include the ridge, `max(output luminance) > 0.9`. **Negative control:** with normalization disabled the same image's max stays low (<0.5) → guards the muddy-dark regression.
5. **Mackie line present:** with `mackie_intensity>0` across a step edge, a bright overshoot appears on the bright side (REE>0). **Negative control:** `mackie_intensity=0` → no overshoot.
6. **Resolution independence:** Mackie edge-field scale ∝ long edge (512 vs 2048), same test shape as Eberhard.
7. **Hue preserved:** a colored patch keeps its hue angle through the ratio recombine (Δhue ≈ 0).
8. **Perf gate:** 1024² sub-second, ~4K a couple seconds (pointwise LUT + 1 gaussian blur; same class as Eberhard).

## 8. Files

- `nodes/sabattier.py` (new) — may import the Eberhard adjacency helper or inline the same ~6 lines.
- register in `nodes/__init__.py`; bump count 53→54 in `__init__.py`.
- `tools/test_sabattier.py` (new) — the teeth above.

Spike (banked WHY): `_sabattier_spike/sabattier_spike.py` + `00_original.png` (real photo) + `01..06_*.png`.
