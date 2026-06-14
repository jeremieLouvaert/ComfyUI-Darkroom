# Eberhard / Adjacency Acutance — derivation & sign-off

> ARCHITECTURE DECIDED 2026-06-14 by a CSF-vs-PDE spike (look + cost): **production =
> asymmetric Chemical-Spread-Function convolution** (PRODUCTION MODEL section below).
> The reaction-diffusion PDE (further down) is DEFERRED to an optional "physical mode";
> the rigor pass killed it as the primary (scar-prone units model + 75x slower + a
> SOFTER look). The PDE + rigor sections are kept as the decision record + deferred ref.

> Physically-derived development edge-effect node (Mackie lines / acutance), the
> "organic 3D sharpness" of large-format film, distinct from unsharp mask. Derived
> from Rajkowski & Nowak, "Modelling of edge effects taking account of the diffusion
> phenomenon," Optica Applicata XXXI(2), 2001 (paper text at
> `comfyUI-DEV-Tools/_eberhard_paper.txt`). The reusable adjacency ENGINE: its
> bromide field feeds Bromide Drag, and its edge kernel feeds Sabattier Mackie lines.
> Status: DRAFT for fresh-eyes rigor + Jeremie sign-off BEFORE any code (Newson scar).

## What this is (and is NOT)

A development simulation, not a sharpening filter. Real adjacency effects come from
developer exhaustion + lateral diffusion at a bright/dark boundary: the bright side
gains a density overshoot (Mackie line), the dark side a density undershoot. The
result is asymmetric, saturating, and edge-localized in a way unsharp mask is not,
which is why it reads as organic rather than crunchy. Pitched as physically-derived
acutance; the antidote to the over-sharpened AI look.

NOT per-film calibrated: the paper's constants are representative/textbook, not
measured for a named emulsion. We reproduce the correct edge profile SHAPE and
length scale, not "Tri-X at 20°C." Honest framing in the node docs.

## PRODUCTION MODEL — asymmetric CSF (SIGNED OFF 2026-06-14; supersedes the PDE below)

Spike (`_eberhard_spike/`) proved the CSF reproduces the PDE's correctly-signed,
asymmetric Mackie profile (overshoot 0.117 / undershoot 0.019 / asym 6.1 @ 30ms vs the
PDE's 0.022/0.007/3.2 @ 2.2s, 75x) AND looks crisper (localized Mackie line vs broad
halo). The CSF kernel width IS `r` px so it scales with resolution cleanly — Blocker 1
(the Newson units scar) is DISSOLVED, no PDE timescale coupling / stability / perf-gate
drama. This is the production spec.

**Color space + flow (house norm = linear light):**
```
img(B,H,W,3 sRGB) → tensor_to_numpy_batch → per image:
  lin   = srgb_to_linear(img)
  L     = luminance_rec709(lin)                      # operate on luminance
  sigma = edge_width_ref · (long_edge / 1024)        # resolution-independent (∝ L)
  m     = gaussian_filter(L, sigma)                  # scipy, separable (nearest edges)
  hp    = L - m                                       # high-pass detail
  # ASYMMETRIC gain = the saturation asymmetry that makes it NOT an unsharp mask:
  gpos  = intensity                                   # bright-side overshoot gain
  gneg  = intensity / asymmetry                       # dark-side undershoot gain (asym>1 ⇒ gneg<gpos)
  delta = where(hp > 0, gpos·hp, gneg·hp)
  L_enh = clip(L + delta, 0, None)
  # apply to RGB by RATIO → preserves hue (no color fringing in v1):
  ratio = L_enh / maximum(L, 1e-6)
  out   = clip(lin · ratio[...,None], 0, 1)
  # OPTIONAL bromide drag (off by default) — see below, operates before linear_to_srgb
  out   = linear_to_srgb(out)
  result = blend(original, out, strength)
```

**Bromide-drag feature (optional, drag_amount=0 ⇒ pure Eberhard):** the inhibitor proxy
`P = gaussian_filter(relu(-hp), sigma·0.5)` is the dark-side field (real, cheap). Drag =
shift P by `drag_amount·D_max` px along `drag_angle` (default 90°=down/gravity) and
SUBTRACT its inhibition downstream: `L_enh -= drag_amount·shift(P, angle)` (density-MINUS
streaks). HONEST LABEL in docs: the advection direction/strength is TUNED, not literature-
derived (no published transport model); the field it advects is the real inhibitor proxy.
Surge (density-plus) = v1.x.

**Controls:**
| control | default | range | note |
|---|---|---|---|
| `edge_width` | 2.0 | 0.5–8 (FLOAT, px@1024) | Mackie-line spatial scale (gaussian sigma, ∝ long edge) |
| `intensity` | 0.5 | 0–2 (FLOAT) | edge-effect strength (scales both gains) |
| `asymmetry` | 6.0 | 1–12 (FLOAT) | overshoot/undershoot ratio; 1 = symmetric (unsharp-like), high = filmic |
| `drag_amount` | 0.0 | 0–1 (FLOAT) | bromide drag; 0 = off (pure adjacency) |
| `drag_angle` | 90.0 | 0–360 (FLOAT, deg) | drag direction (90 = downward / gravity) |
| `strength` | 1.0 | 0–1 (FLOAT) | house blend |

Early-exit: `strength<=0` OR (`intensity<=0` and `drag_amount<=0`) → return input.
`[Darkroom]` print. CATEGORY = `AKURATE/Darkroom/Film`. Class `DarkroomEberhard`,
display "Adjacency Acutance".

**Teeth (offline, `tools/test_eberhard.py`):**
1. Mackie line on a step edge: bright-side OVERSHOOT + dark-side UNDERSHOOT, localized
   within ~sigma; REE>0. (Spike-confirmed.)
2. Asymmetry: overshoot > undershoot at asymmetry=6; at asymmetry=1 ≈ symmetric. NEGATIVE
   CONTROL: intensity=0 & drag=0 → output == input (no effect).
3. Tone preserved away from edges (flat regions unchanged within tol).
4. Resolution independence: same edge profile as a fraction of frame at 512 vs 2048 (∝L).
5. Hue preserved: a saturated colored edge keeps its hue (ratio application) within tol.
6. Drag: drag_amount>0 → a density-MINUS streak downstream of a bright block along
   drag_angle; drag_amount=0 → bit-identical to the no-drag path.
7. Perf: 1024² and ~4K timings (gaussian-bound, expect ms–sub-second).

scipy `gaussian_filter` is fine (separable, fast); torch only if 4K perf needs it.

## Continuous model (Rajkowski-Nowak) — DEFERRED "physical mode" reference (NOT v1)

Three coupled fields over the emulsion. Silver kinetics (their Eq. 1):
```
dC_Ag/dt = k · C_eff · (C_Ag∞ − C_Ag)      ,  C_eff = C · (1 − κ·P)  [inhibition]
```
Developer C (their Eq. 2), in-plane diffusion + bath replenishment − consumption:
```
∂C/∂t = D_C ∇²C  +  (D_C/h²)(C₀ − C)  −  α · dC_Ag/dt
```
Inhibitor/bromide product P (their Eq. 3), diffusion + bath removal + liberation:
```
∂P/∂t = D_P ∇²P  −  (D_P/h²) P  +  β · dC_Ag/dt
```
`C_Ag∞` = the target (fully-developed) silver, set by local exposure. `C₀` = bath
developer concentration. The `/h²` terms are the z-exchange with the developer bath
across emulsion thickness h (their "strong circulation" assumption). Paper constants
(cleaned from p.496): D_C=D_P=5.2e-10 m²/s, h=10µm, dx=dy=0.8µm, C₀=5.0e3 mol/m³,
silver saturation M=1.5 g/m² (baseline 0.5), κ(inhibition)=0.2, t≤200s. REE
(relative edge enhancement) = (M2−M2') + (M1−M1') = Mackie overshoot+undershoot.

## LOAD-BEARING CALL 1 — normalized units, NOT microns (the Newson-scar fix)

We never instantiate physical microns/seconds. Collapse {D, Δt, dx, t, N-steps} into
ONE spatial knob: **edge width `r` in pixels, resolution-independent via `r_px = r_ref·(L/1024)`** (L=long edge), exactly the grain node's μ_r∝L. The paper's CONSTANTS
survive only as dimensionless RATIOS that fix the kernel shape: D_C=D_P (equal),
κ=0.2 (inhibition → edge intensity), saturation 1.5/0.5 (asymmetry), the bath-exchange
strength. Everything is computed in dimensionless diffusion units where the total
diffusion spread equals `r_px`. So the user sets edge WIDTH (px@1024) + intensity +
strength; the physics sets the SHAPE. This removes the input-grid/output-grid/units
ambiguity that inverted Newson twice.

## LOAD-BEARING CALL 2 — 2-D isotropic, reduced-iteration explicit scheme

The paper is 1-D (∂/∂y=0, an idealized straight edge). Real images have edges in all
orientations → use the **isotropic 2-D Laplacian** (5-point stencil, separable conv).
Discrete explicit (FTCS) update per step n, all fields as HxW arrays:
```
dAg   = k · clamp(C·(1−κ·P), 0) · (C_Ag∞ − C_Ag) · dτ        # kinetics
C    += g·∇²C  + b·(C₀ − C)  − α·dAg                          # g = D_C dτ/dx² (dimensionless)
P    += g·∇²P  − b·P         + β·dAg
C_Ag += dAg
```
`g` = dimensionless in-plane diffusion per step (the CFL number), `b` = bath-exchange
per step. ∇² via a 5-point Laplacian convolution (torch on CUDA, numpy fallback —
mirror the Halftone/grain pattern). Map input image → `C_Ag∞` (density or exposure);
init C=C₀, P=0, C_Ag=0. Output = developed `C_Ag` mapped back to density.

## LOAD-BEARING CALL 3 — stability + perf gate (N ∝ r²)

Explicit FTCS is conditionally stable: 2-D requires `g ≤ 0.25`. To reach spread
radius `r_px` we need total diffusion `N·g ≈ r_px²/2`, so with g pinned at ~0.2 the
step count **N ≈ r_px²/(2g) ∝ r_px²**. That is the cost: small r (a few px) → a
handful of steps (cheap); large r → quadratic blowup. HARD PERF GATE after the torch
prototype (Newson discipline): ~1024² in a couple seconds on GPU at a default r; cap
`r_ref` (and thus N) so the worst case stays bounded; document the cap. If the gate
fails, reduce N via an implicit/multigrid solver (deferred) rather than shipping a
node nobody waits on.

## LOAD-BEARING CALL 4 — retain P as an engine output (the multi-feature payoff)

The inhibitor/bromide field `P` is kept and returned by the engine. It is:
- the dark-side undershoot driver (already in the loop),
- what **Bromide Drag** advects: directional shift of P along a gravity/agitation
  vector, subtracting its inhibition downstream (drag = density-minus streaks) or
  adding fresh-developer boost (surge = density-plus). HONEST LABEL: the advection
  direction/strength is TUNED, not literature-derived (no published transport model);
  the field it advects IS physically real (it's the engine's P).
- reused for **Sabattier** Mackie lines later (same edge physics).

## Output & controls (v1 = the Eberhard acutance node)

`image → C_Ag∞ → run N steps → C_Ag → density → blend(original, result, strength)`.
- `edge_width` (r_ref, px@1024, ~1–8, def ~2): the Mackie-line spatial scale.
- `intensity` (maps to κ / kinetics gain, 0–1, def ~0.4): edge-effect strength.
- `strength` (0–1): house blend.
- Advanced: `iterations_cap` (perf guard), maybe `developer_dilution` (C₀, dilution
  raises edge enhancement per the paper Fig.7) as a secondary character knob.
Bromide-drag feature (same node or sibling, TBD): `drag_direction`, `drag_amount`,
`surge` — operating on the retained P field.
CATEGORY likely `AKURATE/Darkroom/Film` (a development effect). Class `DarkroomEberhard`.

## Teeth before trust (offline, no external oracle)

1. **Mackie line on a step edge (headline):** a bright/dark step → output has a density
   OVERSHOOT on the bright side and UNDERSHOOT on the dark side, localized within ~r_px
   of the edge. Measure the over/undershoot amplitude (REE) > 0.
2. **Asymmetry (distinguishes from unsharp):** bright-side overshoot ≥ dark-side
   undershoot (saturation makes it asymmetric) — an unsharp mask would be symmetric.
   NEGATIVE CONTROL: with diffusion off (g=0) there is NO edge effect (REE≈0) → proves
   the effect comes from diffusion, not codegen.
3. **Tone preserved away from edges:** flat regions far from any edge are unchanged
   (within tolerance) — it's an EDGE effect, not a global tone shift.
4. **Resolution independence:** the same scene at 512 vs 2048 px gives the same edge
   profile as a fraction of the frame (r∝L holds).
5. **Perf gate:** report 1024² and ~4K timings + the step count N at default r.
6. **P field sanity (for drag):** the retained P is elevated on the dark side of edges
   (where inhibitor accumulated) — a precondition for drag to act on a real field.

## RIGOR PASS FINDINGS (fresh-eyes Opus adversary, 2026-06-14) — REWORK BEFORE SIGN-OFF

The adversary prototyped the 1-D scheme on the embedded python. **Physics is SOUND:**
correctly-signed asymmetric Mackie lines confirmed (bright overshoot +0.205, dark
undershoot +0.035, asymmetry ~5.9× from saturation, negative control g=0 → REE exactly
0.0). It is genuinely NOT an unsharp mask. BUT:
- **BLOCKER 1 — Call 1 (units→pixels) is WRONG, the Newson scar realized.** The single
  knob `r` does NOT decouple shape from scale: the reaction has its own timescale (1/k)
  and the Mackie line only exists in a window where the reaction is slow enough that
  developer diffuses *during* development. Clock N by diffusion → shapes diverge wildly
  at 2× res (asym 4.2 vs 38.9). Clock N by reaction → shapes match but edge width is
  constant in px (does NOT scale with resolution). The grain `μ_r∝L` analogy is inverted
  for a diffusion length. FIX: a 2nd fixed dimensionless constant, the reaction-extent
  (Damköhler-like) ratio `φ = kdt·N`; pin `kdt = φ·g/r²` so `kdt·N` stays invariant as
  `r` scales with L (so BOTH diffusion spread `g·N∝r²` AND development extent scale
  together). Re-derive on paper.
- **BLOCKER 2 — Call 3 perf wrong ~25×.** N is gated by `max(N_diffusion, N_reaction)`;
  the reaction needs ~250 steps to develop regardless of r, so there is NO cheap "few
  steps at small r." Floor ~250 steps (still ~1–3 s on CUDA, node works, but the cost
  model + `iterations_cap` are built on the wrong variable). FIX: gate on
  `max(r²/2g, ~5/kdt)`, state the floor.
- **SHOULD-FIX 3 — stability bounds the doc omits.** CFL `g≤0.25` correct, but the bath
  term needs `8g + b ≤ 2` (the doc let `b` be a free knob → unstable at b=0.5), and the
  reaction is stiff (blew up to NaN at k=5). FIX: bound `b`, and clamp the per-step
  development fraction `kdt·C_eff·(Ag∞−Ag) ≤ (Ag∞−Ag)` (or sub-cycle the reaction).
- **NICE-TO-KNOW:** the `C_eff` clamp is benign (never fired in stable runs); 2-D lift is
  sound (5-point stencil → minor diagonal anisotropy, cosmetic).
- **ARCHITECTURE (Call 6) — the cheaper path may be the right one.** The proven
  asymmetric over/undershoot profile is reproducible by an **asymmetric chemical-spread
  convolution** (difference of offset kernels) at ~2 convolutions vs ~500 PDE steps, AND
  a convolution kernel's width IS `r` px → it scales with resolution cleanly and SIDESTEPS
  Blocker 1's reaction-timescale coupling entirely. The PDE's only unique payoff (the P
  field for drag) is reproducible cheaply (P ≈ smoothed sign-flipped silver gradient).
  RECOMMENDATION: spike the CSF path first; if the look matches the proven PDE profile,
  ship CSF as primary (cheaper + no scar) and keep the PDE as an optional "physical"
  advanced mode or defer it.

**VERDICT: focused rework, not teardown.** Re-derive Call 1 (the `φ` invariant), rebuild
the perf gate on `max(diffusion, reaction)`, add the stability bounds — OR pivot primary
to the asymmetric-CSF convolution which dissolves Blocker 1. Decide CSF-vs-PDE first.

## Honest residuals / deferred
- Constants are textbook, not per-film → shape-faithful, not calibrated (documented).
- Bromide-drag advection direction/strength is tuned (no published transport model).
- Implicit/multigrid solver if the explicit N∝r² perf gate fails at large r (deferred).
- Sabattier (Mackie via this engine + non-monotonic LUT + silver-mask) = next node.
