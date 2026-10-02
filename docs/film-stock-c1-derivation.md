# Film Stock (Color): Capture One looks, baked (1.29.0)

## Why
In 1.28, 33 of the 111 colour stocks crushed everything below roughly 40% input brightness to black. On a test photo, Portra 400NC turned 56% of the pixels black; Capture One's own Portra 400 NC turns 2.2% black. All 33 were stocks derived from the Capture One Film Styles pack (MIT, a community conversion of Lightroom film presets). The 55 hand-authored stocks were fine.

The old importer (`tools/parse_costyles.py` -> `tools/generate_stocks.py`) mistranslated the styles:
- **Curves** were reduced to a toe/shoulder/slope fit (`toe = 1.5 + 8 x deviation`, so an identity curve became 1.5/1.5) and applied in linear light around 0.18.
- **Colour balance** became additive linear tints of -0.1 to -0.2, which clamp the shadows to 0.
- **Dropped entirely:** the Advanced Color Editor (`ColorCorrections`, present in all 586 styles), Levels, and Shadow/Highlight Recovery.

## How the looks are made now
Capture One renders each style itself. `tools/c1_reference_kit.py` and `tools/c1_lut_bake.py` write the style's adjustment keys into C1 session sidecars (`CaptureOne/Settings*/<image>.cos`, same keys as a `.costyle`). One export then renders:
- **a 65^3 colour grid per style** (a near-square 585x520 mosaic; C1 silently skips processing very thin images, so a 65x4225 strip comes back untouched);
- **an unstyled grid as a control:** it round-trips to 0.42/255 max, so no hidden processing;
- **validation renders of a chart and a photo.**

Only the global tools are baked: curves, levels, colour balance, saturation, contrast and the colour editor. Grain and sharpening stay out, since Darkroom has its own. Recovery and clarity are local and would smear across a grid.

**Sidecar injection was validated against a hand-applied style.** The hand-applied version differs only by the style's grain and sharpening: the 17 px-blurred residual is 0.29/255 mean.

## Accuracy (vs Capture One's own renders)
| Part | Result |
|---|---|
| Baked LUT, 65^3 | 0.03-0.08/255 mean, p99 < 0.5/255 |
| Baked LUT, 33^3 (shipped) | 0.1-0.2/255 mean, p99 ~1/255 on a photo |
| Full look incl. recovery approximation, photo | 1.6-2.5/255 mean, p99 7-13/255 (deepest shadows) |
| Crush audit, all 111 stocks | 0 stocks > 5% black (was 33); ramps monotonic (one 0.21/255 dip inherited from C1's Provia 100F) |

## Shadow/Highlight Recovery
C1's recovery is local: on the photo it lifts shadows by +8 to +14/255, while on a chart of thin bands it barely acts. It is approximated as a luminance-dependent gain, `1 + SR*S(Lb) - HR*H(Lb)`, applied before the LUT.
- `Lb` is luminance blurred at 0.5% of the long edge.
- `S` and `H` are 9-knot curves fitted jointly over 6 styles x (chart, photo).
- Held-out (leave-one-style-out) error on the photo is 1.6-2.5/255 mean.
- It can be switched off with the node's `recovery` toggle.

## Ancillary findings (kept as cross-checks)
- **Curves.** C1 applies them in sRGB, in the order master -> R/G/B -> luma, with a natural cubic spline; the luma curve acts as a luminance ratio with Rec.601 weights. This matches C1's grey ramp to 0.15/255.
- **Levels:** output = TargetHighlight x input^(1 - Midtone), matching to 0.08/255.
- **The tone curve on the node** (`utils/tone_curve_ops.py`) uses the same natural cubic interpolation.
