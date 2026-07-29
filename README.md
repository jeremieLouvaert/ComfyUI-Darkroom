# ComfyUI-Darkroom

Professional color grading and film emulation suite for ComfyUI — 58 nodes, 161 film stocks, 35 spectral neg×print LUTs, 102 lens profiles, reference-driven Color Match, colorist scopes, full CMYK print workflow, zero API costs.

The most complete color toolset in the ComfyUI ecosystem. From physics-based film emulation to DaVinci Resolve-level color grading, Camera Raw processing, optical simulation, LUT export, ACES color management, and magazine-ready CMYK print output — everything runs locally with no external dependencies.

## Nodes

### Film Emulation (14 nodes)

| Node | Description |
|------|-------------|
| **Film Rebate** | Format-correct procedural film borders: sprockets, edge printing, notch codes, Polaroid/slide frames, filed-carrier print border. Canvas-extending (composes the input into a larger film/print canvas and outputs a MASK marking where the image landed). Seven formats: 135 full frame (KS-1870 perforations, dual frame numbering, optional stylized DX bars), 135 filed carrier (the Cartier-Bresson full-negative look, seeded jagged black border), 120 6x6/6x9, 4x5 sheet (F-3-style notch code selected by stock name), Polaroid integral (dims re-verified against Polaroid's own support figures), and 35mm slide mount. Rebate polarity (B&W neg / Color neg / Reversal) drives the film-format border color, fill/fit compose modes handle the aspect mismatch, and every mm dimension is resolution-independent by construction. |
| **Light Leak** | Stray light reaching the film outside the lens path, modelled as an exposure event and composited additively in linear light, so it adds a constant absolute luminance rather than a screen-blended tint. One mechanism drives both falloff and colour: light entering the film edge travels laterally through the base and dye layers, which absorb short wavelengths hardest, so all three channels start equal at the entry point and blue dies first. A leak is therefore a near-white hot core that reddens outward, not the uniform orange every canned overlay bakes in. The same equation predicts the exception: a pinhole strikes the emulsion from the front with no lateral path, so it stays the neutral colour of its source. Three geometries: sprocket trap (edge falloff modulated by the real KS-1870 perforation lattice, with the comb diffusing into a smooth wash as light spreads inward), gradient (seal, bellows or 120 backing-paper glow via a colour-source switch), and pinhole (camera-obscura spots sized by a + D*theta from hole diameter and flange distance). A path-length displacement warps the leak boundary so it wanders instead of running parallel to the frame edge, carrying the colour fringe coherently with it. Outputs the image plus a MASK of the leak field. |
| **Film Damage** | Dust, dirt, hairs and scratches in one node, with the sign and colour of every defect derived from where it sits in the imaging chain rather than picked as paint. A defect is an optical event at a plane, so it applies a constant density offset: set `defect_origin` to negative and dust blocks the printing light and comes out WHITE; set it to positive (scanner glass, a print, a slide) and the same field reads dark. On the same negative a base-side scratch is a refractive groove that prints white, while an emulsion-side gouge removes dye layers top-down, so one `scratch_depth` control walks yellow to red to black on C-41 and blue to cyan to white on reversal. Scratches follow the film transport axis with bounded lateral wander and Kokaram-style diffraction shoulders; defect counts and sizes are gamma-distributed after the Ivanova et al. (CGF 2023) method, with sizes anchored to real particle scale. Outputs the image plus a MASK of the defect field for restoration or inpainting graphs. Fully procedural and seeded: no scan plates, no overlay PNGs. |
| **Film Stock (Color)** | 111 color film stocks with per-channel H&D characteristic curves. Kodak Portra, Ektar, Fuji Velvia, Cinestill, Polaroid, and more. Capture One curve data integration. |
| **Film Stock (B&W)** | 50 B&W stocks with real spectral sensitivity coefficients. Ilford HP5+, Kodak Tri-X, T-MAX, with pushed variants. |
| **Film Grain** | Multi-octave luminance-dependent grain. ISO-scaled, resolution-aware, blue-channel emphasis like real film. |
| **Film Grain Pro** | Resolution-independent stochastic grain (Newson et al. 2017). Physically derived, so grain holds a fixed fraction of the frame at any resolution. Monte-Carlo rendered, tone-preserving, with grain size, radius variation, and per-channel color grain. |
| **Halation** | Physically derived annular halation: light totally-internally-reflects off the rear film-base surface and re-exposes the emulsion in a ring standing off each highlight, not a centered glow. A Fresnel/TIR ring kernel plus a diffuse scattering tail, composited additively in linear light so the red-orange hue shift is the actual physics, not a tint. CineStill 800T, Vision3 subtle, and B&W plate presets, with an anti-halation strength control and a Custom preset for manual per-channel weights. |
| **Print Stock** | Photographic paper simulation — the negative-to-print chain. |
| **Cross Process** | E-6 in C-41 and C-41 in E-6 cross-processing color shifts. |
| **Adjacency Acutance** | Film development edge effects (Mackie lines): a density overshoot on the bright side of an edge and an undershoot on the dark side, the organic 3D acutance of large-format film rather than digital over-sharpening. Asymmetric and edge-localized, with an asymmetry control (1 = symmetric, higher = filmic). Optional bromide drag adds the density-minus streaks that trail from bright areas in the direction of gravity or tank agitation. Derived from a published reaction-diffusion edge-effect model (implemented as the cheaper chemical-spread convolution). Physically inspired, not per-film calibrated. |
| **Reciprocity Failure** | Simulates film reciprocity failure (the Schwarzschild effect) at long exposures: the per-channel color cast that no digital sensor has, plus crushed shadows. Pick a film and an exposure time and the long-exposure character is applied from the manufacturer datasheet corrections (Kodak E-31 and Fuji datasheets). Six stocks: B&W general, T-Max, Portra 400, Ektachrome E100 (cyan/blue cast), Provia 100F (magenta), Velvia 50 (green). Datasheet-character-grounded, not per-roll calibrated; film aging is a separate concern. |
| **Spectral B&W (Ortho/Pan)** | Black and white conversion driven by a film's spectral sensitivity. Orthochromatic renders red dark and blue light (white skies, dark skin and lips), panchromatic is natural, and an extended-red mode lightens reds and foliage. Five sensitivity types plus two measured stocks: Kodak Tri-X 400 and Kodak 5222 (Double-X), using datasheet-digitized sensitivity curves (vendored spectral_film_lut data, MIT). Real pan emulsions are bluer than the idealized panchromatic curve, so the measured stocks render skies lighter and reds darker, the reason yellow filters were standard practice. The RGB weights are derived from spectral sensitivity curves integrated against the Mallett-Yuksel sRGB spectral basis, a principled channel mixer rather than hand-picked numbers. An sRGB image cannot carry a real spectrum, so this is a spectral approximation of the ortho/pan tonal character. |
| **Sabattier** | The Sabattier effect (partial solarization): a re-exposure mid-development fogs the unshielded highlights, folding them back dark while the deep shadows hold — the classic tonal reversal. Bright Mackie lines trace the contours, composed from the Adjacency Acutance kernel. A re-exposure depth and a shield control set how much of the highlights reverse and how sharply; the look is most dramatic on high-key images. The non-monotonic reversal is physically grounded (silver shielding, densities add while reflectance is logarithmic); the printing renormalization and Mackie composition are tuned to the look, not per-developer calibrated. |

### Camera Raw Tools (10 nodes)

| Node | Description |
|------|-------------|
| **White Balance** | Color temperature (Kelvin) and tint adjustment via Planckian locus approximation. |
| **Auto White Balance** | Automatic color-cast removal by estimating the scene illuminant from the image (no temperature input). Four classical methods in one dropdown: Gray World, White Patch (Max-RGB), Shades of Gray (Minkowski norm, the robust default), and Gray Edge (gradient-based, often best). Corrects via a von Kries diagonal in linear light and preserves overall brightness. Strength blend. Pure numpy, no learned weights. A robust "remove the obvious cast" auto-WB, not a trained neural matcher. |
| **Exposure & Tone** | EV-stop exposure, S-curve contrast, parametric shadows/highlights/whites/blacks. |
| **HSL Selective** | Per-hue adjustments to hue, saturation, and luminance across 8 color bands with smooth feathering. |
| **Clarity / Texture / Dehaze** | Local contrast enhancement, surface texture detail, and atmospheric haze removal. |
| **Vibrance** | Intelligent saturation with skin tone protection — boosts muted colors, protects already-saturated areas. |
| **Sharpening Pro** | Advanced unsharp mask with edge-aware masking and radius/amount/threshold control. |
| **Noise Reduction** | Multi-pass guided filter with 7 presets — from light denoise to heavy smoothing. |
| **Skin Tone Uniformity** | Mask-weighted mean pull for even skin tones. 6 skin type presets. Preserves texture. |
| **Color Qualifier** | Color range analysis and isolation with 19 action presets combining selection + correction. |

### Color Grading (10 nodes)

| Node | Description |
|------|-------------|
| **Tone Curve** | 5-point cubic spline per channel (PchipInterpolator — monotonic, no overshoot). 11 presets: S-curves, Faded Blacks, Matte Film, Cross-over pushes. |
| **Lift Gamma Gain** | DaVinci Resolve primary corrector. Per-channel R/G/B + Master for Lift, Gamma, Gain, and Offset — 16 precision sliders. |
| **OkLab Color** | Perceptually-uniform grading in OkLab / OkLch. Lightness and contrast that hold hue and chroma, and chroma that stays even across every hue. The orthogonal-channel separation makes that true by construction, not by tuning. Controls: lightness, contrast, chroma, hue, and a/b tint. |
| **Log Wheels** | Resolve Log-mode grading. Soft Gaussian zone masks in log2-encoded luminance space. Hue angle + saturation + density per zone. 7 presets. |
| **3-Way Color Balance** | Preset-first creative color tinting. Shadow/midtone/highlight zones with hue + intensity. 15 looks — Orange & Teal, Vintage Warm, Moonlight Blue, Bleach Bypass, and more. |
| **Hue vs Hue** | Remap specific hue ranges to different hues. 8 bands with feathering. 9 presets — skin tone correction, sky shifts, autumn warmth. |
| **Hue vs Sat** | Adjust saturation per hue range. 8 bands. 8 presets — pop blues, mute greens, teal & orange pop. |
| **Lum vs Sat** | Adjust saturation based on luminance. 5 zones. 7 presets — film look (desat highlights), punch midtones, bleach bypass. |
| **Sat vs Sat** | Adjust saturation based on existing saturation level. Compress oversaturated areas, boost muted tones. 4 zones, 6 presets. |
| **Color Warper** | 2D hue + saturation region warping with multi-region presets. 9 presets — Orange & Teal push, skin cleanup, sunset enhance. Manual mode for single-region custom work. |

### Lens & Optics (6 nodes)

| Node | Description |
|------|-------------|
| **Chromatic Aberration** | Lateral CA simulation/correction with per-channel shift. |
| **Vignette** | Optical vignette with shape, midpoint, and falloff control. |
| **Lens Distortion** | Brown-Conrady barrel/pincushion distortion model. |
| **Perspective Correct** | Keystone and trapezoid correction for architectural shots. |
| **Lens Profile** | All-in-one lens correction — distortion + CA + vignette from 102 real lens models (Canon, Nikon, Sony, Zeiss, Leica, vintage). |
| **Water Refraction** | Refracts the image through a water surface poured onto the screen, where the warp is a consequence of a depth-averaged FLIP/PIC fluid simulation rather than a displacement texture: the optics read only the free surface h(x,y) from it. Refraction is exact Snell, so displacement is hard-bounded at 0.881 x the water depth because the transmitted angle saturates at the 48.6 deg critical angle. That makes the effect inherently macro and is why `field_width_mm` is the control that matters most: frame a whole tablet and you correctly get almost nothing, frame 20-50mm of screen and the reference look appears. Where the surface is steep enough the map FOLDS, producing several images of one feature, which a smooth displacement filter cannot do; a finite aperture (rho = A/L x depth) then blends those branches instead of seaming them, which is what removes the chrome/liquify look. The pour is swept rather than dropped from a fixed point, because a stationary source makes a perfectly circular spreading disc that reads as a lens. Controls are physical throughout: water volume, sweep, when the shutter fires, settling time (viscous damping kills a 2.4mm ripple in ~72ms and a 12mm one in ~1.8s, so it removes chop and keeps the large forms), and an honest `depth_scale` for anyone wanting more than physics allows. The warp genuinely destroys fine detail, so the node measures that loss by pushing a noise probe through the identical warp and restores the grain calibrated to the source's own level, and also outputs the deficit as a MASK for manual chaining. Slower than the rest of the pack (tens of seconds at 1024); `sim_resolution` is the speed dial. |

### RAW Pipeline (2 nodes)

| Node | Description |
|------|-------------|
| **RAW Load** | Decodes camera RAW files (.cr3, .nef, .arw, .raf, .dng, .rw2, .orf, .pef, .x3f, .iiq, and more) via rawpy/LibRaw. Exposes demosaic algorithm, white balance, highlight mode, output colorspace, linear-scene vs sRGB-display output, and a **Camera Look** profile selector (see below). Outputs an IMAGE and a `RAW_METADATA` sidecar. |
| **RAW Metadata Split** | Splits `RAW_METADATA` into 15 typed primitives: camera make/model, lens make/model, ISO, aperture, shutter, focal length, datetime, sensor type, resolution, Fuji film sim, and more. Wire any primitive directly into a text node or downstream tool. |

#### Camera Look Profiles — `ComfyUI/models/camera_profiles/`

RAW Load can apply Adobe `.dcp` profiles — per-body color calibration (`Adobe Standard`) or creative looks (`Camera Standard`, `Camera Landscape`, Fuji film sims, etc.). The `camera_look` dropdown on the node is a union of every `.dcp` found across these locations, in priority order:

1. **`ComfyUI/models/camera_profiles/<Make Model>/`** — drop user-installed packs here. Respects `extra_model_paths.yaml`. Example: `ComfyUI/models/camera_profiles/Fujifilm X-T5/Fujifilm X-T5 Camera VELVIA.dcp`.
2. **`ComfyUI-Darkroom/data/dcp_looks/`** — reserved for profiles bundled with the pack.
3. **Adobe install paths** (auto-discovered if Camera Raw / Lightroom is installed on the same machine): `C:/ProgramData/Adobe/CameraRaw/CameraProfiles/Camera/` and the Lightroom Classic resources equivalent.

**What ships with Adobe:** Camera Look profiles for Canon, Nikon, Sony, Panasonic, Olympus, Pentax, and others (Camera Standard / Landscape / Portrait / Faithful / Neutral / Monochrome, etc.). If you have Camera Raw or Lightroom installed, these work with zero setup.

**Fujifilm users:** Adobe does not ship Fuji Camera Look profiles as .dcp files (Fuji's in-camera film sims are baked into the Camera Raw engine binary, not exposed on disk). To get Velvia / Provia / Astia / Classic Chrome / Eterna / Pro Neg Hi / Pro Neg Std / Reala Ace as DCPs, build them yourself from [abpy/FujifilmCameraProfiles](https://github.com/abpy/FujifilmCameraProfiles) (CC-BY-NC-SA 4.0):

```bash
# One-time: clone the abpy LookTable + ToneCurve source
git clone https://github.com/abpy/FujifilmCameraProfiles third_party/FujifilmCameraProfiles

# Build 8 sim DCPs for your body (requires the body's Adobe Standard DCP
# installed at C:/ProgramData/Adobe/CameraRaw/CameraProfiles/Adobe Standard/)
python tools/build_fuji_dcps.py \
  --body "Fujifilm X-T5" \
  --abpy third_party/FujifilmCameraProfiles \
  --out  "ComfyUI/models/camera_profiles/Fujifilm X-T5"
```

The `tools/build_fuji_dcps.py` script splices abpy's per-sim LookTable + ToneCurve onto your body's Adobe Standard base matrices and writes 8 DCPs (Provia / Velvia / Astia / Classic Chrome / Pro Neg Hi / Pro Neg Std / Eterna / Reala Ace) that drop directly into the `camera_look` dropdown. Repeat for each body you shoot with. Classic Neg / Nostalgic Neg / Bleach Bypass ship as `.cube` LUTs only in abpy and can be used via the LUT Apply node.

**B&W sims (Acros / Monochrome + R/Y/G filter variants):** abpy deliberately skips Fuji's black-and-white simulations because Fuji's B&W rendering is a channel-mix plus tone curve that DCP's HSV-based LookTable expresses awkwardly, so there's no public pack. Darkroom ships a separate builder, `tools/build_fuji_bw_dcps.py`, that synthesizes the B&W look tables from published Neopan 100 Acros channel weights (for Acros) and BT.709 luma (for Monochrome), with Wratten 8 / 11 / 25 filter transmissions pre-multiplied for the +Y / +G / +R variants. This is Darkroom's approximation, not a calibrated 1:1 match to Fuji's in-camera output. Usage mirrors the color builder:

```bash
python tools/build_fuji_bw_dcps.py \
  --body "Fujifilm X-T5" \
  --out  "ComfyUI/models/camera_profiles/Fujifilm X-T5"
```

Writes 8 DCPs per body: Acros, Acros+R, Acros+Y, Acros+G, Monochrome, Monochrome+R, Monochrome+Y, Monochrome+G. Runs without the abpy checkout (the B&W tables are synthesized, not spliced).

If the selected Camera Look isn't available for the detected body, the node silently falls back to Adobe Standard and logs a console warning.

### Spectral Film Stock (1 node, 35 presets)

| Node | Description |
|------|-------------|
| **Spectral Film Stock** | Pre-baked `.cube` LUTs derived from full negative→print spectral simulation. Each preset encodes scene-light → spectral sensitivity → log exposure → H&D density → dye spectral density → printer light → print density → sRGB. Shipped presets cover C41 still (Portra 160/400/800, Ektar 100, Gold 200, Ultramax, Fuji Pro 160C/160S/400H, Superia Reala / X-Tra 400, Natura 1600, Vericolor III on Endura / Supra / Portra Endura / Fuji Crystal Archive papers), Cinema (Vision3 50D/200T/250D/500T on 2383 / 2393), Reversal slides (Velvia 50, Provia 100F, Ektachrome 100D, Kodachrome 64 on Ilfochrome / Ektachrome Radiance III), Instant (FP-100C, Instax Color on Fujiflex), Niche (Aerocolor, Aerocolor High, Agfa Vista 100), B&W (Tri-X 400, Kodak 5222 on Polymax grades). |

The baker is vendored at `third_party/spectral_film_lut/` (MIT, JanLohse/spectral_film_lut), stripped to the headless engine. Run `python tools/bake_spectral_luts.py --all` to regenerate or extend; bake takes ~13 s for all 35 LUTs. See `tools/bake_spectral_luts.py` for the preset registry — add your own `_cat(...)` entries for extra neg×print combos and rebake.

### Scopes (2 nodes)

| Node | Description |
|------|-------------|
| **Histogram** | Per-channel R/G/B / Luma / single-channel histogram with 0/25/50/75/100% graticule and clip-warning stripes on the edges. Log-scale toggle for highlight-dominant images. Output is IMAGE — wire to PreviewImage. |
| **Vectorscope** | Rec.709 YCbCr density plot with 75% + 100% saturation rings, six primary target boxes (R/Yl/G/Cy/B/Mg), the 123° skin-tone line (I-line), configurable gain (zoom into low-sat scenes) and log-scale density. Cold-to-warm heatmap. |

### Reference-driven Color Match (1 node)

| Node | Description |
|------|-------------|
| **Color Match (Reference)** | Grade a target image toward a reference's colour distribution. Four LAB-space methods in one dropdown: **reinhard** (mean/std transfer, fast safe default), **wasserstein** (sliced optimal transport via iterative advection, handles multi-modal distributions), **forgy** (K-means palette matching with Gaussian-weighted soft assignment, sklearn), **kantorovich** (closed-form Gaussian linear transport, requires `pip install POT`). Intensity blend + per-method tuning (n_colors, n_slices, sample_size, seed). Algorithms adapted from [rajawski/gradia](https://github.com/rajawski/gradia) (MIT). |

### CMYK Print Workflow (4 nodes)

| Node | Description |
|------|-------------|
| **CMYK Soft-Proof** | RGB → target CMYK → RGB roundtrip preview. Image stays in RGB for continued editing, the colour shift you see is what will happen on press. |
| **CMYK Gamut Warning** | Overlays pixels that cannot be accurately reproduced by the chosen print condition (threshold configurable). Logs out-of-gamut percentage. |
| **CMYK TAC Check** | Converts to CMYK and flags pixels where C+M+Y+K exceeds the TAC (Total Area Coverage) limit. Presets: 330% coated / 300% uncoated / 300% web coated / 240% newsprint / custom. Prevents ink-drying and show-through problems before the file ships. |
| **CMYK Export TIFF** | Writes a 4-channel CMYK TIFF with ICC profile embedded. LZW-compressed, configurable DPI, defaults to `ComfyUI/output/cmyk/`. This is the file you send to the printer. |

**ICC profile discovery:** The CMYK nodes auto-discover profiles from (1) `ComfyUI-Darkroom/data/icc_profiles/` for user drops, and (2) the OS colour-profile store. On Windows you already have FOGRA39 (ISO Coated v2), FOGRA27, FOGRA29 (uncoated), GRACoL 2006, US Web Coated SWOP v2, SWOP 2006 Grade 3/5, Euroscale Coated/Uncoated, SNAP 2007 newsprint, JapanColor 2001/2002 — all bundled by Windows at `C:\Windows\System32\spool\drivers\color\`. Additional free profiles are available from [ECI](https://www.eci.org/doku.php?id=en:downloads) (FOGRA51 / PSO Coated v3 / PSO Uncoated v3 / ISO Newspaper 26v4).

**Rendering intents:** perceptual for photos (default), relative colorimetric for logos and corporate, saturation for charts, absolute colorimetric for pre-press proofing simulation.

### Halftone (1 node)

| Node | Description |
|------|-------------|
| **Halftone** | Halftone screening, the newsprint / comic look. Reproduces continuous tone as a grid of ink dots whose size grows with tone. Four dot shapes (round, line for an engraving screen, square, ellipse for chain dots) and two methods: AM clustered dots (angled, with the standard CMYK rosette 15/75/0/45) or FM dispersed Bayer dither. Mono (black on white) or color (naive CMYK separation). Resolution-independent screen frequency (lines across the long edge), supersampled dot edges, GCR control, strength blend for a subtle screen overlay. Dot shapes are tone-linearized, so changing the shape changes the dot geometry, not the overall tone. GPU-accelerated. This is a stylize effect, not a calibrated proof (use CMYK Soft-Proof for that). |

### Pipeline — LUT & Color Management (7 nodes)

| Node | Description |
|------|-------------|
| **LUT Identity Generator** | Outputs a neutral identity lattice image. Feed into LUT Bake Inject to grade your photo and bake a .cube at the same time. Sizes: 17, 33, 65. |
| **LUT Bake Inject** | Pairs your photo with the identity lattice as a 2-image batch. The grading chain then processes both with identical settings — no node duplication. |
| **LUT Bake Extract** | Splits the batch back out after the grading chain: graded photo to preview, graded lattice to LUT Export. |
| **LUT Export (.cube)** | Bakes any Darkroom processing chain into a standard .cube 3D LUT file. Works in DaVinci Resolve, Premiere Pro, Photoshop, Capture One, FCPX — any tool that supports 3D LUTs. |
| **LUT Apply (.cube)** | Loads and applies any .cube 3D LUT with trilinear interpolation. Import looks from DaVinci Resolve, download creative LUTs, or reuse exported Darkroom grades. Strength slider for blending. |
| **Color Space Transform** | Convert between sRGB, Linear sRGB, ACEScg, ACEScct, Rec.2020, and DCI-P3. The only ACES-aware color management in ComfyUI. Soft gamut compression option. |
| **ACES Tonemap** | Industry-standard tonemapping: ACES Filmic, ACES Fitted (Hill), AgX (Blender), Reinhard, Filmic (Uncharted 2). Exposure bias, ACES gamut conversion, white point control. |

## LUT Bake Workflow — grade your photo and export a .cube in one pass

Building a LUT in Darkroom used to mean running the grading chain twice — once on your photo, once on an identity lattice — with settings duplicated across two parallel chains. That's tedious and error-prone. The **LUT Bake Inject / Extract** pair fixes that: one chain, one set of settings, your photo and the LUT come out the other side together.

**How it works:** Inject pads your photo and the identity lattice to a shared canvas and stacks them as a 2-image batch. Every Darkroom color node iterates the batch dimension and applies the exact same transform to both images. After the chain, Extract splits the batch back into the graded photo and the processed lattice — the lattice goes to LUT Export.

**Wiring:**

```
Load Image ──► photo ─────┐
                          ├─► LUT Bake Inject ─► [grading chain] ─► LUT Bake Extract ─► graded_photo  ──► Preview
LUT Identity ─► lattice ──┤                                                        ├─► graded_lattice ─► LUT Export
                          └─► lut_size ─────────────────────────────────────────── └─► lut_size       ─┘
```

- Connect **LUT Identity Generator → identity_lattice** input of **LUT Bake Inject**.
- Connect your **Load Image → photo** input of **LUT Bake Inject**.
- Run any **color-only** Darkroom nodes between Inject and Extract — Tone Curve, Lift Gamma Gain, HSL Selective, Film Stock, Hue vs X, Color Warper, etc.
- **LUT Bake Extract** gives you three outputs: `graded_photo` (to Preview / Save Image), `graded_lattice` (to LUT Export's `processed_lattice`), and `lut_size` (to LUT Export's `lut_size`).

### Color-only rule — what can go in the chain

A 3D LUT is a per-pixel color lookup. It has no idea about neighboring pixels. So only nodes that transform each pixel independently can be baked:

**Allowed in the bake chain:**
Film Stock (Color), Film Stock (B&W), Print Stock, Cross Process, Reciprocity Failure, Spectral B&W (Ortho/Pan), White Balance, Exposure & Tone, HSL Selective, Vibrance, Tone Curve, Lift Gamma Gain, OkLab Color, Log Wheels, 3-Way Color Balance, Hue vs Hue, Hue vs Sat, Lum vs Sat, Sat vs Sat, Color Warper, Color Space Transform, ACES Tonemap, LUT Apply.

**NOT allowed in the bake chain** (they use pixel neighborhoods and will corrupt the lattice):
Film Grain, Film Grain Pro, Halftone, Adjacency Acutance, Sabattier, Halation, Film Damage, Light Leak, Clarity / Texture / Dehaze, Sharpening Pro, Noise Reduction, Skin Tone Uniformity, Color Qualifier (partial — uses local masks), Auto White Balance (content-adaptive — estimates the illuminant per image), Chromatic Aberration, Vignette, Lens Distortion, Perspective Correct, Lens Profile.

If you want spatial effects on your final image, apply them to `graded_photo` **after** Extract, not inside the bake chain.

### Example workflow

A ready-to-use example is in [`workflows/lut_bake_and_apply.json`](workflows/lut_bake_and_apply.json). Drag it into ComfyUI, load a photo, and press Queue Prompt — you'll get a graded preview and a `.cube` file in `output/luts/`.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jeremieLouvaert/ComfyUI-Darkroom.git
pip install -r ComfyUI-Darkroom/requirements.txt
```

Restart ComfyUI. All 55 nodes appear under **AKURATE/Darkroom/** with subcategories: Film (incl. Spectral), Raw, Grading (incl. Color Match), Lens, Pipeline, RAW, Scopes, Print.

### Dependencies

- **scipy** (>= 1.10.0) — spline interpolation, Gaussian filters, FFT convolution
- **opensimplex** (>= 0.4) — high-quality simplex noise for film grain

Optional:
- **POT** (`pip install POT`) — enables the Kantorovich method on Color Match. Reinhard / Wasserstein / Forgy all work without it.
- **colour-science** + **numba** — only required to *regenerate* Spectral Film Stock LUTs via `tools/bake_spectral_luts.py`. Consumers who install via Comfy Registry or git clone receive the pre-baked LUTs and never run the baker.

No API keys. No GPU required. Pure numpy/scipy computation (Histogram + Vectorscope render via PIL, CMYK nodes use PIL.ImageCms / LittleCMS).

## Architecture

All processing happens in **linear light** (sRGB gamma removed before processing, reapplied after). Every node supports:

- **Strength slider** (0-1) — non-destructive blending with original
- **Batch processing** — handles ComfyUI's multi-image batches
- **Preset + override** — presets provide instant results, sliders fine-tune

Film stock data sourced from Capture One Film Styles (586 .costyle files parsed) and published Kodak/Fuji/Ilford technical data sheets. Lens profiles measured from real optical characteristics.

## License

MIT

## Author

Jeremie Louvaert — [jeremielouvaert.com](https://jeremielouvaert.com)
