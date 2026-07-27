"""
Film Rebate node for ComfyUI-Darkroom.

Format-correct procedural film borders/rebate: sprockets, edge printing,
notch codes, Polaroid/slide frames, filed-carrier print border. Build 1 of 3
in the film-damage sweep (Rebate -> Damage -> Leak), spec-signed-off plan
`lovely-coalescing-rain.md` (2026-07-25). Zero prior art does spec-correct
geometry (the market is fixed-resolution PNG overlays) -- this is procedural,
resolution-independent geometry derived from real mm specs (KS-1870 perf,
F-3 notch vocabulary, Polaroid integral dims verified against Polaroid's own
support figures at build time).

Canvas-EXTENDING (first in the pack): output = input image composed INTO a
larger film/frame canvas, batch-uniform per run (all frames share identical
geometry/params). RETURNS (IMAGE composed, MASK image_area) -- the mask
marks exactly where the original image pixels landed, for downstream
compositing.

Scaling rule (resolution-independent by construction): the input image
occupies the format's image aperture; px-per-mm = image_long_edge_px /
aperture_long_edge_mm. Every other mm dimension (perf pitch, rebate bands,
notch geometry, mount bevel) derives from that single scale factor. The
133-filed-carrier format has no separate aperture spec (the photo IS the
aperture, unscaled) -- its mm conversions borrow the 36mm 135 reference
edge per the plan ("border_width mm-equivalent ~= ratio of 36mm").

Render pipeline: geometry computed once in final (1x) px. A DECORATIVE hi
canvas (2x supersample) is built for every non-photo element (rebate fills,
perf holes, edge-print text, notches, mount bevel) and LANCZOS-downscaled to
1x for antialiasing (opaque draws throughout, no alpha-resolve needed). The
source photo is resized ONCE, directly to its final placement rect at 1x
(best-quality single resize, never double-resampled), then pasted on top of
the downscaled decorative layer. The MASK is a clean binary rect computed
from that same placement geometry -- not derived from rendered pixels -- so
it is exact, not an AA'd alpha fringe.

Rebate polarity (135/120/4x5 film formats only; Polaroid/slide use their own
surface colors) and the edge-print lettering polarity are taste calls
documented inline where they're decided (search "TASTE CALL").
"""

import hashlib
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = os.path.join(PACK_ROOT, "fonts", "BigShouldersStencil-Bold.ttf")

SS = 2  # supersample factor for the decorative layer


def mask_batch_to_tensor(arrays):
    """list of (H,W) float32 numpy arrays -> ComfyUI MASK tensor (B,H,W).
    Mirrors ComfyUI-Schematic/nodes/schematic_overlay.py's helper."""
    stacked = np.stack([np.ascontiguousarray(a, dtype=np.float32) for a in arrays], axis=0)
    return __import__("torch").from_numpy(stacked)


# ---------------------------------------------------------------------------
# Widget vocab
# ---------------------------------------------------------------------------
FORMATS = [
    "135 full frame", "135 filed carrier", "120 6x6", "120 6x9",
    "4x5 sheet", "Polaroid", "Slide mount",
]
FILM_TYPES = ["Color neg", "B&W neg", "Reversal"]
FILM_TYPE_TO_KEY = {"Color neg": "c41", "B&W neg": "bw_neg", "Reversal": "reversal"}

# Rebate polarity table (plan sec "Rebate polarity"). sRGB approx.
REBATE_COLORS = {
    "bw_neg": (0x0a, 0x0a, 0x0a),
    "c41": (0x10, 0x14, 0x1e),
    "reversal": (0x05, 0x05, 0x05),
}

BACKGROUND_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "mid-gray": (128, 128, 128)}

# TASTE CALL (plan sec "Edge printing", "Open items"): edge-print lettering
# renders light-gray on all dark (polarity-table) rebates. No format in this
# node prints text on a light surface (Polaroid ships no text per spec
# silence; Slide's date stamp is its own red-ink mechanism, not this table).
EDGE_PRINT_LIGHT = (200, 200, 200)

# Notch vocabulary: real F-3 shape set {V, U/half-round, square}, invented
# 6-pattern sequences (2-5 notches), selected by stock_name hash -- consistent
# with the sweep's "generic invented stock names" decision.
NOTCH_PATTERNS = [
    ["V", "square"],
    ["U", "V", "U"],
    ["square", "square", "V"],
    ["V", "U", "square", "V"],
    ["U", "square"],
    ["square", "V", "U", "square", "V"],
]

MOUNT_COLORS = [
    ("white plastic", (0xf2, 0xf1, 0xec)),
    ("grey", (0x8c, 0x8c, 0x8c)),
    ("cream cardboard", (0xe8, 0xdc, 0xc0)),
]


def _hash_index(s, salt, n):
    """Deterministic (process/platform independent) hash -> index in [0,n).
    Python's built-in hash() is PYTHONHASHSEED-randomized for str, so this
    uses hashlib for stable stock_name -> pattern selection."""
    h = hashlib.md5(f"{salt}|{s}".encode("utf-8")).hexdigest()
    return int(h, 16) % n


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
_FONT_CACHE = {}


def _get_font(px_size):
    px_size = max(int(round(px_size)), 4)
    f = _FONT_CACHE.get(px_size)
    if f is None:
        f = ImageFont.truetype(FONT_PATH, px_size)
        try:
            f.set_variation_by_name("Bold")  # variable-font instance (see fonts/OFL.txt)
        except Exception:
            pass
        _FONT_CACHE[px_size] = f
    return f


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _np_to_pil(arr):
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _pil_to_np(img):
    return np.asarray(img).astype(np.float32) / 255.0


def _compose_placement(src_w, src_h, ap_w, ap_h, mode):
    """Pure geometry: where does the (resized) source land inside the
    aperture? Returns (x0, y0, w, h) in aperture-local px, mode-aware.
    'fill' always covers the full aperture (cover-crop); 'fit' letterboxes.
    """
    if mode == "fill":
        return (0, 0, ap_w, ap_h)
    scale = min(ap_w / src_w, ap_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x0 = (ap_w - new_w) // 2
    y0 = (ap_h - new_h) // 2
    return (x0, y0, new_w, new_h)


def _resize_for_placement(src_pil, ap_w, ap_h, mode):
    """Returns the resized (never double-resampled) photo PIL image at
    exactly the size `_compose_placement` predicts for this mode."""
    src_w, src_h = src_pil.size
    if mode == "fill":
        scale = max(ap_w / src_w, ap_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = src_pil.resize((new_w, new_h), Image.LANCZOS)
        x0 = (new_w - ap_w) // 2
        y0 = (new_h - ap_h) // 2
        return resized.crop((x0, y0, x0 + ap_w, y0 + ap_h))
    x0, y0, new_w, new_h = _compose_placement(src_w, src_h, ap_w, ap_h, mode)
    return src_pil.resize((new_w, new_h), Image.LANCZOS)


def _low_freq_noise(rng, length_px, n_control=6):
    """Smooth low-frequency value noise over [0, length_px), values in
    [-1, 1], built from a handful of random control points + linear
    interpolation (deterministic given rng)."""
    length_px = max(int(round(length_px)), 1)
    ctrl = rng.uniform(-1.0, 1.0, size=n_control).astype(np.float64)
    xs_ctrl = np.linspace(0, length_px - 1, n_control)
    xs = np.arange(length_px)
    return np.interp(xs, xs_ctrl, ctrl)


def _draw_notch(draw, shape, cx, depth_px, width_px, color):
    x0, x1 = cx - width_px / 2.0, cx + width_px / 2.0
    if shape == "square":
        draw.rectangle([x0, 0, x1, depth_px], fill=color)
    elif shape == "V":
        draw.polygon([(x0, 0), (x1, 0), (cx, depth_px)], fill=color)
    else:  # "U" half-round, flat side on the edge, bulging inward
        r = width_px / 2.0
        draw.pieslice([cx - r, -r, cx + r, r], start=0, end=180, fill=color)


def _draw_notch_code(draw, pattern, canvas_w_px, px_per_mm, color):
    """F-3-style notch sequence within 12mm of the top-right corner of the
    (short, top) edge, cut inward from y=0."""
    start_margin_mm, notch_w_mm, spacing_mm, depth_mm = 1.5, 1.2, 0.8, 2.0
    depth_px = depth_mm * px_per_mm
    w_px = notch_w_mm * px_per_mm
    step_px = (notch_w_mm + spacing_mm) * px_per_mm
    x_edge_px = canvas_w_px - start_margin_mm * px_per_mm
    for i, shape in enumerate(pattern):
        cx = x_edge_px - i * step_px - w_px / 2.0
        _draw_notch(draw, shape, cx, depth_px, w_px, color)


def _text_color_on(bg_rgb, target_rgb, intensity):
    intensity = max(0.0, min(1.0, intensity))
    return tuple(int(round(bg_rgb[c] * (1 - intensity) + target_rgb[c] * intensity)) for c in range(3))


# ---------------------------------------------------------------------------
# Per-format geometry + hi-res decorative render
#
# Each _geom_* function returns a dict describing everything in FINAL (1x)
# px, given px_per_mm and the source image size. Each _draw_* function draws
# the decorative hi-res (SS supersampled) layer given that geometry dict.
# Both share the same dict so canvas-size math (tested by the teeth) lives
# in exactly one place per format.
# ---------------------------------------------------------------------------

def _geom_135_full(src_w, src_h, strip_context):
    long_edge_px = max(src_w, src_h)
    px_per_mm = long_edge_px / 36.0
    inter_gap_mm = 2.0
    canvas_w_mm = strip_context + inter_gap_mm + 36.0 + inter_gap_mm + strip_context
    canvas_h_mm = 35.0
    canvas_w = max(1, int(round(canvas_w_mm * px_per_mm)))
    canvas_h = max(1, int(round(canvas_h_mm * px_per_mm)))
    ap_w = int(round(36.0 * px_per_mm))
    ap_h = int(round(24.0 * px_per_mm))
    ap_x0 = int(round((strip_context + inter_gap_mm) * px_per_mm))
    ap_y0 = int(round((canvas_h_mm - 24.0) / 2.0 * px_per_mm))
    return {
        "px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
        "ap_w": ap_w, "ap_h": ap_h, "ap_x0": ap_x0, "ap_y0": ap_y0,
        "frame_x0": ap_x0, "frame_w": ap_w,
    }


def _draw_135_full(draw, g, rebate_rgb, bg_rgb, stock_name, frame_number,
                    edge_print_intensity, dx_barcode, seed):
    S = SS
    pm = g["px_per_mm"] * S
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    draw.rectangle([0, 0, cw, ch], fill=rebate_rgb)

    # -- perforations (KS-1870): 1.98x2.79mm, r~0.5mm, pitch 4.75mm, 8/frame,
    # centered on the aperture; lattice continues into the context zones.
    hole_w, hole_h, r = 1.98 * pm, 2.79 * pm, 0.5 * pm
    pitch = 4.75 * pm
    inset = 2.0 * pm  # hole-center inset from film edge (top/bottom)
    frame_cx = (g["frame_x0"] + g["frame_w"] / 2.0) * S
    n_half = int(math.ceil((cw / pitch))) + 2
    centers = [frame_cx + (i - 3.5) * pitch for i in range(-n_half, n_half + 8)]
    centers = sorted(set(round(c, 3) for c in centers if -hole_w <= c <= cw + hole_w))
    for y_center in (inset, ch - inset):
        for cx in centers:
            x0, x1 = cx - hole_w / 2, cx + hole_w / 2
            y0, y1 = y_center - hole_h / 2, y_center + hole_h / 2
            draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=bg_rgb)

    # -- edge print: bottom rebate band between the aperture edge and the
    # perf row (stock name left, dual "{n}"/"{n}A" numbering right).
    if edge_print_intensity > 0:
        ap_y1_mm = (g["ap_y0"] / g["px_per_mm"]) + (g["ap_h"] / g["px_per_mm"])
        hole_top_mm = 35.0 - 2.0 - 2.79 / 2.0
        band_top_mm, band_bot_mm = ap_y1_mm, hole_top_mm
        band_h_px = max((band_bot_mm - band_top_mm) * pm, 4)
        font = _get_font(band_h_px * 0.72)
        color = _text_color_on(rebate_rgb, EDGE_PRINT_LIGHT, edge_print_intensity)
        ty = (band_top_mm + band_bot_mm) / 2.0 * pm
        draw.text((g["frame_x0"] * S + 4 * S, ty), stock_name, font=font, fill=color, anchor="lm")
        num_text = f"{frame_number}   {frame_number}A"
        draw.text(((g["frame_x0"] + g["frame_w"]) * S - 4 * S, ty), num_text,
                   font=font, fill=color, anchor="rm")

    # -- optional stylized (non-decodable) DX barcode in the left context zone
    if dx_barcode:
        rng = np.random.default_rng(seed + 9001)
        zone_x0, zone_x1 = 2 * S, g["frame_x0"] * S - 4 * S
        if zone_x1 > zone_x0:
            x = zone_x0
            bar_color = _text_color_on(rebate_rgb, EDGE_PRINT_LIGHT, 0.5)
            while x < zone_x1:
                bw = rng.uniform(0.6, 2.2) * S
                if rng.random() > 0.5:
                    draw.rectangle([x, ch * 0.3, x + bw, ch * 0.7], fill=bar_color)
                x += bw + rng.uniform(0.8, 2.0) * S


def _geom_120(src_w, src_h, six_nine):
    long_edge_px = max(src_w, src_h)
    ap_w_mm = 84.0 if six_nine else 56.0
    ap_h_mm = 56.0
    aperture_long_edge_mm = max(ap_w_mm, ap_h_mm)
    px_per_mm = long_edge_px / aperture_long_edge_mm
    canvas_w_mm, canvas_h_mm = ap_w_mm, 61.0
    canvas_w = max(1, int(round(canvas_w_mm * px_per_mm)))
    canvas_h = max(1, int(round(canvas_h_mm * px_per_mm)))
    ap_w = int(round(ap_w_mm * px_per_mm))
    ap_h = int(round(ap_h_mm * px_per_mm))
    ap_x0 = 0
    ap_y0 = int(round((canvas_h_mm - ap_h_mm) / 2.0 * px_per_mm))
    return {"px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
            "ap_w": ap_w, "ap_h": ap_h, "ap_x0": ap_x0, "ap_y0": ap_y0}


def _draw_120(draw, g, rebate_rgb, bg_rgb, stock_name, frame_number, edge_print_intensity):
    S = SS
    pm = g["px_per_mm"] * S
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    draw.rectangle([0, 0, cw, ch], fill=rebate_rgb)
    if edge_print_intensity > 0:
        band_h_px = max(g["ap_y0"] * S * 0.65, 4)
        font = _get_font(band_h_px)
        color = _text_color_on(rebate_rgb, EDGE_PRINT_LIGHT, edge_print_intensity)
        ty = g["ap_y0"] * S / 2.0
        text = f"{stock_name}  •  {frame_number}"  # sparser: one combined row
        draw.text((cw / 2.0, ty), text, font=font, fill=color, anchor="mm")


def _geom_4x5(src_w, src_h):
    long_edge_px = max(src_w, src_h)
    sheet_w_mm, sheet_h_mm = 101.6, 127.0
    margin_mm = 2.0
    ap_w_mm, ap_h_mm = sheet_w_mm - 2 * margin_mm, sheet_h_mm - 2 * margin_mm
    aperture_long_edge_mm = max(ap_w_mm, ap_h_mm)
    px_per_mm = long_edge_px / aperture_long_edge_mm
    canvas_w = max(1, int(round(sheet_w_mm * px_per_mm)))
    canvas_h = max(1, int(round(sheet_h_mm * px_per_mm)))
    ap_w = int(round(ap_w_mm * px_per_mm))
    ap_h = int(round(ap_h_mm * px_per_mm))
    ap_x0 = int(round(margin_mm * px_per_mm))
    ap_y0 = int(round(margin_mm * px_per_mm))
    return {"px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
            "ap_w": ap_w, "ap_h": ap_h, "ap_x0": ap_x0, "ap_y0": ap_y0}


def _draw_4x5(draw, g, rebate_rgb, bg_rgb, stock_name):
    S = SS
    pm = g["px_per_mm"] * S
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    draw.rectangle([0, 0, cw, ch], fill=rebate_rgb)
    idx = _hash_index(stock_name, "notch", len(NOTCH_PATTERNS))
    _draw_notch_code(draw, NOTCH_PATTERNS[idx], cw, pm, bg_rgb)


def _geom_polaroid(src_w, src_h):
    long_edge_px = max(src_w, src_h)
    ap_w_mm, ap_h_mm = 79.0, 77.0
    aperture_long_edge_mm = max(ap_w_mm, ap_h_mm)
    px_per_mm = long_edge_px / aperture_long_edge_mm
    print_w_mm, print_h_mm = 88.5, 107.5
    side_mm = (print_w_mm - ap_w_mm) / 2.0  # ~4.75mm, top/side borders per spec
    canvas_w = max(1, int(round(print_w_mm * px_per_mm)))
    canvas_h = max(1, int(round(print_h_mm * px_per_mm)))
    ap_w = int(round(ap_w_mm * px_per_mm))
    ap_h = int(round(ap_h_mm * px_per_mm))
    ap_x0 = int(round(side_mm * px_per_mm))
    ap_y0 = int(round(side_mm * px_per_mm))
    return {"px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
            "ap_w": ap_w, "ap_h": ap_h, "ap_x0": ap_x0, "ap_y0": ap_y0}


def _draw_polaroid(draw, g):
    S = SS
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    frame_rgb = (0xf4, 0xf2, 0xee)
    draw.rectangle([0, 0, cw, ch], fill=frame_rgb)
    # slight edge shading: a soft inner border a shade darker
    shade = tuple(max(0, c - 10) for c in frame_rgb)
    shade_w = max(2 * S, int(0.4 * g["px_per_mm"] * S))
    draw.rectangle([0, 0, cw, shade_w], fill=shade)
    draw.rectangle([0, ch - shade_w, cw, ch], fill=shade)
    draw.rectangle([0, 0, shade_w, ch], fill=shade)
    draw.rectangle([cw - shade_w, 0, cw, ch], fill=shade)


def _geom_slide(src_w, src_h):
    long_edge_px = max(src_w, src_h)
    ap_w_mm, ap_h_mm = 23.0, 35.0
    aperture_long_edge_mm = max(ap_w_mm, ap_h_mm)
    px_per_mm = long_edge_px / aperture_long_edge_mm
    mount_mm = 50.0
    canvas_w = max(1, int(round(mount_mm * px_per_mm)))
    canvas_h = max(1, int(round(mount_mm * px_per_mm)))
    ap_w = int(round(ap_w_mm * px_per_mm))
    ap_h = int(round(ap_h_mm * px_per_mm))
    ap_x0 = int(round((mount_mm - ap_w_mm) / 2.0 * px_per_mm))
    ap_y0 = int(round((mount_mm - ap_h_mm) / 2.0 * px_per_mm))
    return {"px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
            "ap_w": ap_w, "ap_h": ap_h, "ap_x0": ap_x0, "ap_y0": ap_y0}


def _draw_slide(draw, g, stock_name, date_text):
    S = SS
    pm = g["px_per_mm"] * S
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    _, mount_rgb = MOUNT_COLORS[_hash_index(stock_name, "mount", len(MOUNT_COLORS))]
    draw.rectangle([0, 0, cw, ch], fill=mount_rgb)

    # aperture opening outline with a 1mm corner radius + a thin inner bevel
    ap_x0, ap_y0 = g["ap_x0"] * S, g["ap_y0"] * S
    ap_x1, ap_y1 = ap_x0 + g["ap_w"] * S, ap_y0 + g["ap_h"] * S
    r = 1.0 * pm
    bevel_px = max(2 * S, int(round(0.3 * pm)))  # 2px min-clamp per spec
    bevel_rgb = tuple(max(0, c - 40) for c in mount_rgb)
    draw.rounded_rectangle([ap_x0 - bevel_px, ap_y0 - bevel_px, ap_x1 + bevel_px, ap_y1 + bevel_px],
                            radius=r + bevel_px, fill=bevel_rgb)

    if date_text:
        font = _get_font(0.11 * g["canvas_h"] * S)
        draw.text((cw - 3 * S, ch - 3 * S), date_text, font=font, fill=(0xa8, 0x22, 0x22), anchor="rb")


def _geom_filed_carrier(src_w, src_h, border_width, paper_margin_pct):
    px_per_mm = max(src_w, src_h) / 36.0  # borrowed 135 reference edge
    border_px = max(1, int(round(border_width * px_per_mm)))
    long_edge = max(src_w, src_h)
    paper_px = max(1, int(round(paper_margin_pct / 100.0 * long_edge)))
    canvas_w = src_w + 2 * border_px + 2 * paper_px
    canvas_h = src_h + 2 * border_px + 2 * paper_px
    ap_x0 = border_px + paper_px
    ap_y0 = border_px + paper_px
    return {"px_per_mm": px_per_mm, "canvas_w": canvas_w, "canvas_h": canvas_h,
            "ap_w": src_w, "ap_h": src_h, "ap_x0": ap_x0, "ap_y0": ap_y0,
            "border_px": border_px, "paper_px": paper_px}


def _draw_filed_carrier(draw, g, rebate_rgb, roughness, seed):
    S = SS
    pm = g["px_per_mm"] * S
    cw, ch = g["canvas_w"] * S, g["canvas_h"] * S
    paper_rgb = (0xfb, 0xf8, 0xf2)
    draw.rectangle([0, 0, cw, ch], fill=paper_rgb)

    border_px = g["border_px"] * S
    img_x0, img_y0 = g["ap_x0"] * S, g["ap_y0"] * S
    img_x1, img_y1 = img_x0 + g["ap_w"] * S, img_y0 + g["ap_h"] * S

    amp_px = roughness * 1.0 * pm  # roughness 0-1 -> 0-1.0mm amplitude
    amp_px = min(amp_px, border_px * 0.85)  # keep the ring from vanishing/inverting
    rng = np.random.default_rng(seed)
    n_top = max(int(round((img_x1 - img_x0))), 1)
    n_side = max(int(round((img_y1 - img_y0))), 1)
    noise_top = _low_freq_noise(rng, n_top) * amp_px
    noise_bot = _low_freq_noise(rng, n_top) * amp_px
    noise_left = _low_freq_noise(rng, n_side) * amp_px
    noise_right = _low_freq_noise(rng, n_side) * amp_px

    step = max(1, n_top // 200)
    top_pts = [(img_x0 + x, img_y0 - border_px - noise_top[x]) for x in range(0, n_top, step)]
    bot_pts = [(img_x0 + x, img_y1 + border_px + noise_bot[x]) for x in range(0, n_top, step)]
    step_s = max(1, n_side // 200)
    right_pts = [(img_x1 + border_px + noise_right[y], img_y0 + y) for y in range(0, n_side, step_s)]
    left_pts = [(img_x0 - border_px - noise_left[y], img_y0 + y) for y in range(0, n_side, step_s)]

    poly = (
        [(img_x0 - border_px, img_y0 - border_px)] + top_pts +
        [(img_x1 + border_px, img_y0 - border_px)] + right_pts +
        [(img_x1 + border_px, img_y1 + border_px)] + list(reversed(bot_pts)) +
        [(img_x0 - border_px, img_y1 + border_px)] + list(reversed(left_pts))
    )
    draw.polygon(poly, fill=rebate_rgb)
    return (noise_top, noise_bot, noise_left, noise_right)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class DarkroomFilmRebate:
    """See module docstring. Geometry spec: plan lovely-coalescing-rain.md."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "format": (FORMATS, {
                    "default": "135 full frame",
                    "tooltip": "Film/print format geometry. Drives every mm-derived dimension"
                }),
                "film_type": (FILM_TYPES, {
                    "default": "Color neg",
                    "tooltip": "Rebate polarity for 135/120/4x5 (Polaroid/Slide use their own surface color)"
                }),
                "compose": (["fill", "fit"], {
                    "default": "fill",
                    "tooltip": "fill = cover-crop into the aperture (default). fit = letterbox with rebate-color bars"
                }),
                "stock_name": ("STRING", {
                    "default": "AKURATE 400",
                    "tooltip": "Edge-print / notch-code / mount-color stock name (also seeds the 4x5 notch pattern and slide mount color)"
                }),
                "frame_number": ("INT", {
                    "default": 7, "min": 0, "max": 999,
                    "tooltip": "First frame's edge-print number (135/120). 135 also prints the classic dual 'nA' offset"
                }),
                "increment_per_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Batch of N -> frame_number, frame_number+1, ... +N-1 (contact-sheet workflows)"
                }),
                "strip_context": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 6.0, "step": 0.1,
                    "tooltip": "135 full frame only: mm of neighboring strip hinted on each side"
                }),
                "border_width": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "135 filed carrier only: black border thickness, mm-equivalent (ratio of 36mm)"
                }),
                "roughness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "135 filed carrier only: seeded perimeter jaggedness, 0 = straight, 1 = ~1mm-equivalent"
                }),
                "paper_margin": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "135 filed carrier only: white paper margin, % of the image's long edge"
                }),
                "edge_print_intensity": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "135/120 edge-print lettering opacity. 0 hides the lettering entirely"
                }),
                "dx_barcode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "135 full frame only: stylized DX-style bars (plausible, non-decodable)"
                }),
                "date_text": ("STRING", {
                    "default": "",
                    "tooltip": "Slide mount only: stamped date text in red-ish ink (e.g. 'MAY 74'). Empty hides it"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seeds the filed-carrier perimeter noise and the DX barcode pattern"
                }),
                "background": (["black", "white", "mid-gray"], {
                    "default": "black",
                    "tooltip": "Scanner-surround color in the small margin outside the film/print edge"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "image_area")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Film"

    def execute(self, image, format="135 full frame", film_type="Color neg", compose="fill",
                stock_name="AKURATE 400", frame_number=7, increment_per_frame=True,
                strip_context=2.0, border_width=1.2, roughness=0.5, paper_margin=6.0,
                edge_print_intensity=0.7, dx_barcode=False, date_text="", seed=42,
                background="black"):

        print(f"[Darkroom] Film Rebate: format={format}, film_type={film_type}, compose={compose}, "
              f"stock_name={stock_name!r}, frame_number={frame_number}, seed={seed}")

        images = tensor_to_numpy_batch(image)
        out_images, out_masks = [], []
        cur_frame = frame_number

        for img in images:
            canvas, mask = self._render_one(
                img, format, film_type, compose, stock_name, cur_frame,
                strip_context, border_width, roughness, paper_margin,
                edge_print_intensity, dx_barcode, date_text, seed, background,
            )
            out_images.append(canvas)
            out_masks.append(mask)
            if increment_per_frame:
                cur_frame += 1

        return (numpy_batch_to_tensor(out_images), mask_batch_to_tensor(out_masks))

    # -- per-frame render --------------------------------------------------
    def _render_one(self, img, format, film_type, compose, stock_name, frame_number,
                     strip_context, border_width, roughness, paper_margin,
                     edge_print_intensity, dx_barcode, date_text, seed, background):
        src_h, src_w = img.shape[0], img.shape[1]
        film_key = FILM_TYPE_TO_KEY.get(film_type, "c41")
        rebate_rgb = REBATE_COLORS[film_key]
        bg_rgb = BACKGROUND_COLORS.get(background, BACKGROUND_COLORS["black"])
        S = SS

        if format == "135 full frame":
            g = _geom_135_full(src_w, src_h, strip_context)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_135_full(draw, g, rebate_rgb, bg_rgb, stock_name, frame_number,
                            edge_print_intensity, dx_barcode, seed)
        elif format == "120 6x6":
            g = _geom_120(src_w, src_h, six_nine=False)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_120(draw, g, rebate_rgb, bg_rgb, stock_name, frame_number, edge_print_intensity)
        elif format == "120 6x9":
            g = _geom_120(src_w, src_h, six_nine=True)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_120(draw, g, rebate_rgb, bg_rgb, stock_name, frame_number, edge_print_intensity)
        elif format == "4x5 sheet":
            g = _geom_4x5(src_w, src_h)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_4x5(draw, g, rebate_rgb, bg_rgb, stock_name)
        elif format == "Polaroid":
            g = _geom_polaroid(src_w, src_h)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_polaroid(draw, g)
        elif format == "Slide mount":
            g = _geom_slide(src_w, src_h)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_slide(draw, g, stock_name, date_text)
        elif format == "135 filed carrier":
            g = _geom_filed_carrier(src_w, src_h, border_width, paper_margin)
            hi = Image.new("RGB", (g["canvas_w"] * S, g["canvas_h"] * S))
            draw = ImageDraw.Draw(hi)
            _draw_filed_carrier(draw, g, rebate_rgb, roughness, seed)
        else:
            raise ValueError(f"[Darkroom] Film Rebate: unknown format {format!r}")

        decorative_1x = hi.resize((g["canvas_w"], g["canvas_h"]), Image.LANCZOS)

        # -- compose the photo (single, best-quality resize) on top
        src_pil = _np_to_pil(img)
        placement = _compose_placement(src_w, src_h, g["ap_w"], g["ap_h"], compose)
        photo = _resize_for_placement(src_pil, g["ap_w"], g["ap_h"], compose)
        px0, py0, pw, ph = placement
        decorative_1x.paste(photo, (g["ap_x0"] + px0, g["ap_y0"] + py0))

        # -- background margin (scanner surround)
        margin_px = max(0, int(round(0.04 * max(g["canvas_w"], g["canvas_h"]))))
        final_w = g["canvas_w"] + 2 * margin_px
        final_h = g["canvas_h"] + 2 * margin_px
        final_img = Image.new("RGB", (final_w, final_h), bg_rgb)
        final_img.paste(decorative_1x, (margin_px, margin_px))

        canvas_np = _pil_to_np(final_img)

        mask = np.zeros((final_h, final_w), dtype=np.float32)
        mx0 = margin_px + g["ap_x0"] + px0
        my0 = margin_px + g["ap_y0"] + py0
        mask[my0:my0 + ph, mx0:mx0 + pw] = 1.0

        return canvas_np, mask


NODE_CLASS_MAPPINGS = {
    "DarkroomFilmRebate": DarkroomFilmRebate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomFilmRebate": "Film Rebate",
}
