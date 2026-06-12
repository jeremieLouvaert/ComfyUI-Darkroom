# OkLab Color — derivation & sign-off

> Perceptually-uniform grading for ComfyUI-Darkroom. Pure numpy, Björn Ottosson's
> published OkLab constants. Signed off (Opus) 2026-06-12 before any code, per the
> derive-before-code discipline. colour-science is used as an OFFLINE ORACLE only,
> never a runtime dependency (decisions.md 2026-04-09 pure-numpy over OCIO).

## What this node is (and is NOT)

The first perceptually-uniform grader in Darkroom. Every existing grading node works
in linear sRGB / HSL / Rec.709-luma, none perceptually uniform, so contrast twists hue
and saturation is uneven across the wheel. OkLab fixes that: lightness and contrast
that hold hue and chroma, chroma that is even across all colors ("expensive colorist"
behavior).

It does **NOT** address the Wave 6 curve-fitting wall. That wall is a *spatial*
calibration-match problem (CA / NR / demosaic — a function of the pixel neighborhood,
not of input value alone). No pointwise transform, OkLab included, can touch it. Pitch
and document this node as perceptually-uniform grading only.

## OkLab transform (Ottosson 2020, canonical constants)

### Forward: linear sRGB (D65) → OkLab

Step 1 — linear sRGB → LMS (matrix M1):

```
l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
```

Step 2 — nonlinearity (cube root, **sign-preserving** — use `np.cbrt`, NOT `**(1/3)`,
because out-of-gamut / extreme pixels can produce small negative LMS):

```
l_ = cbrt(l) ; m_ = cbrt(m) ; s_ = cbrt(s)
```

Step 3 — LMS' → Lab (matrix M2):

```
L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
```

### Inverse: OkLab → linear sRGB

```
l_ = L + 0.3963377774*a + 0.2158037573*b
m_ = L - 0.1055613458*a - 0.0638541728*b
s_ = L - 0.0894841775*a - 1.2914855480*b

l = l_**3 ; m = m_**3 ; s = s_**3

r =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s
g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
```

(`l_**3` etc. cube the real values — fine for negatives. Final linear sRGB is clipped
to [0,1] = the v1 gamut handling; see below.)

### OkLch (cylindrical, for the chroma / hue controls)

```
C = hypot(a, b)
h = atan2(b, a)          # radians
# inverse
a = C*cos(h) ; b = C*sin(h)
```

Note: at C==0 (neutral grey) `h` is undefined-but-harmless (atan2(0,0)=0); chroma scale
and hue rotate are both no-ops there, so neutrals stay neutral under chroma/hue. Tint
(a,b offset) does cast neutrals, which is intended.

## Grading operations — the design that holds hue & chroma BY CONSTRUCTION

The whole correctness claim rests on one invariant: **lightness & contrast touch only L;
chroma & hue touch only C/h; tint offsets only a,b.** Because of the OkLch separation,
this guarantees "contrast holds hue and chroma" and "chroma is even across the wheel"
mathematically, not by tuning. The offline test asserts these invariants directly (the
real teeth for the product pitch).

Controls (combined "OkLab Color" node — scope confirmed 2026-06-12):

| control | default | range | operation |
|---|---|---|---|
| `lightness` | 1.0 | 0.0–2.0 | `L *= lightness` |
| `contrast`  | 0.0 | -1.0–1.0 | slope around perceptual mid: `slope = 2**contrast`; `L = 0.5 + (L-0.5)*slope` |
| `chroma`    | 1.0 | 0.0–2.0 | `C *= chroma` (in OkLch) |
| `hue`       | 0.0 | -180–180 (deg) | `h += radians(hue)` |
| `tint_a`    | 0.0 | -0.1–0.1 | `a += tint_a` (green↔red axis) |
| `tint_b`    | 0.0 | -0.1–0.1 | `b += tint_b` (blue↔yellow axis) |
| `strength`  | 1.0 | 0.0–1.0 | blend original↔graded in sRGB (house `blend`) |

Operation order (per pixel, after forward transform to L,a,b):

```
# tone (L only)
L = L * lightness
L = 0.5 + (L - 0.5) * (2.0 ** contrast)
# color (C/h only)
C = hypot(a,b) ; h = atan2(b,a)
C = C * chroma
h = h + radians(hue)
a = C*cos(h) ; b = C*sin(h)
# tint (a,b offset — applied last, a global cast)
a = a + tint_a ; b = b + tint_b
```

Rationale on `contrast`: a linear slope in OkLab-L around the perceptual mid (0.5) IS
already a perceptually-uniform contrast — that is the entire point of doing it in L.
`slope = 2**contrast` makes it symmetric (contrast=-1 → slope 0.5 flatten, +1 → slope 2)
and identity at 0. A true sigmoid S-curve with highlight/shadow rolloff is a v1.x
refinement; the linear-in-L version is correct and hue/chroma-preserving by construction.
`tint` ranges are small because OkLab a,b sit roughly in [-0.4, 0.4] for saturated colors;
±0.1 is already a strong cast.

## Pipeline (house flow)

```
img (B,H,W,C sRGB 0..1 tensor)
  → tensor_to_numpy_batch
  per image:
      original = img
      lin  = srgb_to_linear(img)                 # utils/color.py
      L,a,b = linear_srgb_to_oklab(lin)          # new, colorspace.py
      ... grading ops above ...
      lin2 = oklab_to_linear_srgb(L,a,b)
      lin2 = clip(lin2, 0, 1)                     # v1 gamut clip (linear sRGB)
      out  = linear_to_srgb(lin2)                 # utils/color.py (torch-accelerated)
      result = blend(original, out, strength)     # utils/color.py
  → numpy_batch_to_tensor
```

Early-exit (return input unchanged): `strength <= 0`, OR identity
(`lightness==1 and contrast==0 and chroma==1 and hue==0 and tint_a==0 and tint_b==0`).
`[Darkroom]` print on the active path. CATEGORY = `AKURATE/Darkroom/Grading`.

## Gamut (v1 limit, documented)

After the inverse transform, clip linear sRGB to [0,1]. Strong chroma boosts or hue
rotations can push colors out of the sRGB gamut; v1 hard-clips (can flatten the most
saturated pixels). Ottosson's gamut compression (toward the achromatic axis at constant
L/h) is the principled fix and is a v2 note, not v1.

## Teeth before trust (offline validation — `tools/test_oklab.py`)

colour-science 0.4.7 IS importable in the embedded python, so we validate against a real
oracle, not just reference values:

1. **Oracle agreement** — compare `linear_srgb_to_oklab` against
   `colour.XYZ_to_Oklab(colour.RGB_to_XYZ(lin, 'sRGB', apply_cctf_decoding=False))` across
   random colors + the sRGB primaries/secondaries. Tolerance ~1e-4 (matrix-rounding slack).
2. **Round-trip identity** — random linear sRGB → oklab → back, max abs err < 1e-5 (and
   sRGB → oklab → sRGB).
3. **Reference points** — linear white (1,1,1) → L≈1.0, a≈0, b≈0; black → (0,0,0);
   18% grey (0.18³) → a≈0, b≈0, L≈0.565.
4. **Negative control (proves the oracle test has teeth)** — flip one M2 constant's sign
   and confirm test 1 FAILS. A pass is only trusted once the negative control fails.
5. **Invariant teeth (the product claim)** — chroma scaling leaves L & h unchanged;
   hue rotation leaves L & C unchanged; lightness/contrast leave C & h unchanged
   (all within 1e-5). This is the objective proof of "holds hue and chroma."

Run on the real embedded python; report PASS/FAIL honestly, including the negative control.
