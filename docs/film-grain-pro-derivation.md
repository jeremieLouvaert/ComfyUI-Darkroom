# Film Grain Pro — Newson stochastic grain, derivation and implementation notes

Implements the filtered inhomogeneous Boolean model of Newson, Faraj, Delon,
Galerne, *Realistic Film Grain Rendering*, IPOL 2017
(https://doi.org/10.5201/ipol.2017.192). Clean-room from the paper math only
(the IPOL C++ is GPL-3.0; we borrow the published algorithm, not the code).

This node is **additive**: it ships alongside the existing fast heuristic
`FilmGrain` node, which stays the default. Film Grain Pro is the
physically-derived, resolution-independent quality tier.

## The model (paper units)

- **Boolean model Z**: union of disks. Per input pixel `(i,j)` draw
  `Q ~ Poisson(λ(i,j))` grain centres, uniform in that pixel's unit square.
  Radii are i.i.d. with mean `μ_r`, std `σ_r`, **in input-pixel units**.
- **Convention**: one input pixel = one unit square. Grain radii are in input px.
- **Intensity** (tone-driven density):
  `λ(i,j) = (1 / (π (μ_r² + σ_r²))) · log(1 / (1 − ũ(i,j)))`,
  with `ũ = u / (umax + ε)`, `umax = 1.0`, `ε ≈ 0.1/255`.
  Bright pixels → dense grains, dark pixels → few.
- **Filter / output**:
  `v(y) = (1/N) Σ_k 1Z((y + ξ_k)/s)`, `ξ_k ~ N(0, σ² I)`,
  with `σ` in **output-pixel units** (default 0.8), `s` the zoom.
- By construction `E[v(y)] = ũ(y)`, so the render is **tone-preserving**; grain is
  the spatial variance of one fixed realisation of `Z`.

## The two design calls that resolve the earlier scar

The node was attempted twice before and failed on the coordinate model (two
deleted drafts). The fix is not a better zoom mapping; it is to remove the zoom.

### 1. s = 1
The paper's zoom `s` exists for *their* use case: magnify a fixed low-res photo
until individual grains are visible. A grain node's input already **is** the
target resolution, so we set `s = 1`: the output grid equals the input grid
equals the user's image (1 image px = 1 input px = 1 unit square). This collapses
the input-grid / output-grid / zoom / σ-units ambiguity that inverted twice.

Resolution independence is then recovered purely by scaling the mean radius with
image size:

```
μ_r = grain_size · (L / 1024)        # L = long edge in px
```

where `grain_size` (ρ_ref) is the grain radius defined at a 1024px reference.
Grain stays a fixed fraction of the frame at any resolution. This also gives
**better density modulation** than the paper's downsample path: `λ` is evaluated
at full image resolution, so grain density follows real tonal detail.

Visibility floor: at s=1, keep `grain_size ≳ 0.7` so `μ_r ≳ 1px` at 1024+; below
~0.7px the filter averages sub-pixel grain to zero (one of the two original
failure symptoms).

### 2. Pixel-wise only (no grain-wise, no auto-chooser)
Working the eq-9 complexity at s=1: the pixel-wise cost per output pixel is
**independent of μ_r** in our regime (`σ_r < μ_r`). As `μ_r` shrinks, `λ` grows as
`1/μ_r²` but the number of cells scanned shrinks as `μ_r²`; the product is
constant (~`2N` grain-distance checks per pixel at `σ_r = 0`). So pixel-wise is
**O(W · H · N)** for any grain size, and embarrassingly parallel over output
pixels (the paper parallelises exactly this loop). Grain-wise only wins for huge,
highly-variable grains we do not serve, and needs `N` scratch binary images plus
scatter-writes that fight GPU vectorisation. We drop it and the Figure-4
auto-chooser entirely.

## Algorithm (pixel-wise, s = 1), as implemented

For a single channel field `u` (H, W) in [0, 1]:

- Cell grid: `K = ceil(1/μ_r)`, `δ = 1/K` (each input pixel split into K×K cells).
  In the common case `μ_r ≥ 1` → `K = 1, δ = 1`, cell = input pixel.
- Max radius: `σ_r = 0` → `rm = μ_r`; else `rm = ` 0.999 quantile of a log-normal
  radius distribution with mean `μ_r`, std `σ_r`.
- Cell search radius: `cell_rad = ceil(rm / δ)`; neighbourhood = `(2·cell_rad+1)²`.
- `N` Monte-Carlo offsets `ξ_k ~ N(0, σ²I)`, generated once, shared across pixels.
- For each output pixel `(r, c)`, eval point `p = (c+0.5, r+0.5) + ξ_k`.
  Coverage of `p` is tested against grains generated procedurally per cell from a
  counter-based hash of `(cell_x, cell_y, stream, seed)` — so `Z` is one fixed
  realisation with **zero storage** and every pixel/sample querying a cell sees
  the same grains. `1Z(p) = 1` if any grain covers `p`.
- `v(y) = (1/N) Σ_k 1Z(p_k)`.

Grain count per cell is drawn by inverse-CDF Poisson with mean `λ_cell · δ²` from
the cell's hash (deterministic → fixed Z). Grains per cell are capped at `Qmax`
(adaptive from the image's brightest `λ`; `P(Q > Qmax) ≈ 0`), the standard
Worley / texton-noise vectorisation (paper ref [6]).

Note on N: `Z` is fixed per render, so larger `N` cleans the Monte-Carlo estimate
of the (grainy) filtered field — it does **not** average the grain away. Grain
lives in `Z`, not in the MC noise.

## Colour

Three independent channel renders (seeds offset per channel). `color_grain`
blends between a shared grain field (green channel as the luminance proxy, applied
to all channels → monochrome/luminance grain) and per-channel independent grain
(chroma grain, the physical multi-layer-emulsion behaviour, paper §5.2):

```
out_c = u_c + strength · lerp(dev_green, dev_c, color_grain)
```

`color_grain ≤ 0.01` shortcuts to a single luminance render.

## Controls

| Control | Range / default | Maps to |
|---|---|---|
| `grain_size` (ρ_ref, px @ 1024) | 0.7–4.0, default 1.2 | `μ_r = grain_size · L/1024` |
| `radius_variation` (σ_r/μ_r) | 0–0.5, default 0.0 | `σ_r = ratio · μ_r` |
| `strength` | 0–1, default 0.5 | `out = u + strength · grain_dev` |
| `color_grain` | 0–1, default 0.0 | shared-luma ↔ per-channel grain |
| `monte_carlo_samples` (N) | 8–256, default 64 | quality vs speed (linear cost) |
| `filter_sigma` (σ) | 0.4–2.0, default 0.8 | sampling anti-alias / grain softness |
| `seed` | int | PRNG seed for Z and ξ_k |

`filter_sigma` is fixed-by-default at 0.8 output px: it is the sampling
anti-alias filter, and resolution independence already comes from `μ_r ∝ L`.

## Limitations

- Very large + highly-variable grains would render faster grain-wise (not served).
- At s=1, `grain_size < ~0.7` on small images yields sub-pixel grain that the
  filter washes out (clamped by the control minimum).
- Full-resolution density modulation can put slightly more grain along
  high-contrast edges than the paper's downsample path (believed realistic).
- Performance scales as O(W · H · N · cells); the GPU path is the supported path,
  CPU is a slow fallback.
