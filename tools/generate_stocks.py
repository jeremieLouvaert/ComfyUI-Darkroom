"""
Generate refined Darkroom stock definitions from Capture One .costyle data.

Reads the parsed costyle_data.json and produces updated Python source for
color_stocks.py and bw_stocks.py with real curve parameters.
"""

import json
import os
import sys
from pathlib import Path


def load_c1_data():
    here = Path(__file__).parent
    with open(here / "costyle_data.json", "r") as f:
        return json.load(f)


# ── Mapping: C1 curve_params → our CurveParams ──────────────────────────────
#
# C1's curve_params were estimated by parse_costyles.py from control points.
# Our model: CurveParams(toe_power, shoulder_power, slope, pivot_x, pivot_y)
# pivot_x is always 0.18 (18% grey). pivot_y tracks the midtone shift.
#
# For per-channel offsets, we compute R/G/B deltas from the average curve.

def c1_to_our_params(c1_entry):
    """Convert C1 parsed data to our _stock() parameters."""
    cp = c1_entry["curve_params"]
    r, g, b = cp["red"], cp["green"], cp["blue"]

    # Average curve as base
    avg_toe = (r["toe"] + g["toe"] + b["toe"]) / 3
    avg_shoulder = (r["shoulder"] + g["shoulder"] + b["shoulder"]) / 3
    avg_slope = (r["slope"] + g["slope"] + b["slope"]) / 3

    # Per-channel offsets from average
    r_off = (
        round(r["toe"] - avg_toe, 3),
        round(r["shoulder"] - avg_shoulder, 3),
        round(r["slope"] - avg_slope, 3),
    )
    g_off = (
        round(g["toe"] - avg_toe, 3),
        round(g["shoulder"] - avg_shoulder, 3),
        round(g["slope"] - avg_slope, 3),
    )
    b_off = (
        round(b["toe"] - avg_toe, 3),
        round(b["shoulder"] - avg_shoulder, 3),
        round(b["slope"] - avg_slope, 3),
    )

    # Color balance → shadow/highlight tint
    cb = c1_entry["color_balance"]
    s_tint = cb["shadow_tint"]
    h_tint = cb["highlight_tint"]

    # C1 saturation is -100..+100 scale, ours is a multiplier centered at 1.0
    # C1 +5 ≈ our 1.05, C1 -5 ≈ our 0.95 (roughly)
    c1_sat = c1_entry["saturation"]
    our_sat = round(1.0 + c1_sat / 100.0, 3)

    # Grain mapping: C1 amount 0-100 → we'll store for reference
    grain = c1_entry["grain"]

    return {
        "toe": round(avg_toe, 2),
        "shoulder": round(avg_shoulder, 2),
        "slope": round(avg_slope, 2),
        "r_off": r_off,
        "g_off": g_off,
        "b_off": b_off,
        "sat": our_sat,
        "s_tint": (round(s_tint[0], 4), round(s_tint[1], 4), round(s_tint[2], 4)),
        "h_tint": (round(h_tint[0], 4), round(h_tint[1], 4), round(h_tint[2], 4)),
        "grain_amount": grain["amount"],
        "grain_type": grain["type"],
    }


def fmt_off(off):
    """Format offset tuple, omitting if all zeros."""
    if all(abs(v) < 0.001 for v in off):
        return None
    return f"({off[0]}, {off[1]}, {off[2]})"


def fmt_tint(tint):
    """Format tint tuple."""
    return f"({tint[0]}, {tint[1]}, {tint[2]})"


def generate_stock_line(key, display_name, desc, params, indent="    "):
    """Generate a single _stock() call."""
    parts = [
        f'{indent}"{key}": _stock(',
        f'{indent}    "{display_name}", "{desc}",',
        f'{indent}    {params["toe"]}, {params["shoulder"]}, {params["slope"]}',
    ]

    # Add offsets only if non-zero
    offsets = []
    r = fmt_off(params["r_off"])
    g = fmt_off(params["g_off"])
    b = fmt_off(params["b_off"])
    if r:
        offsets.append(f"r_off={r}")
    if g:
        offsets.append(f"g_off={g}")
    if b:
        offsets.append(f"b_off={b}")

    if offsets:
        parts[-1] += ", " + ", ".join(offsets) + ","
    else:
        parts[-1] += ","

    parts.append(f'{indent}    sat={params["sat"]}, s_tint={fmt_tint(params["s_tint"])}, h_tint={fmt_tint(params["h_tint"])}),')

    return "\n".join(parts)


# ── Stock definitions ─────────────────────────────────────────────────────────
# Maps: C1 name → (our_key, display_name, description, category)

# REFINEMENTS: stocks we already have, now with C1 data
REFINE_MAP = {
    # Negative - Kodak Portra
    "Kodak Portra 160": ("Neg / Kodak Portra 160", "Kodak Portra 160",
        "Ultra-smooth portrait stock. Muted colors, warm skin, wide latitude.", "neg"),
    "Kodak Portra 160 NC": ("Neg / Kodak Portra 160NC", "Kodak Portra 160NC",
        "Natural Color variant. Even more muted, ultimate skin fidelity.", "neg"),
    "Kodak Portra 160 VC": ("Neg / Kodak Portra 160VC", "Kodak Portra 160VC",
        "Vivid Color variant. More saturation than standard, still gentle.", "neg"),
    "Kodak Portra 400": ("Neg / Kodak Portra 400", "Kodak Portra 400",
        "The workhorse portrait stock. Warm, forgiving, beautiful skin tones.", "neg"),
    "Kodak Portra 400 NC": ("Neg / Kodak Portra 400NC", "Kodak Portra 400NC",
        "Natural Color 400. Muted pastels, clean highlights.", "neg"),
    "Kodak Portra 400 VC": ("Neg / Kodak Portra 400VC", "Kodak Portra 400VC",
        "Vivid Color 400. Richer than NC, still portrait-safe.", "neg"),
    "Kodak Portra 800": ("Neg / Kodak Portra 800", "Kodak Portra 800",
        "High-speed Portra. More grain, green-cyan shadows, gritty.", "neg"),
    # Negative - Kodak Ektar
    "Kodak Ektar 100": ("Neg / Kodak Ektar 100", "Kodak Ektar 100",
        "Vivid landscape stock. High saturation, punchy contrast, electric colors.", "neg"),
    # Negative - Kodak Gold
    "Kodak Gold 100": ("Neg / Kodak Gold 100", "Kodak Gold 100",
        "Consumer classic. Warm, slightly yellow cast, modest saturation.", "neg"),
    "Kodak Gold 200": ("Neg / Kodak Gold 200", "Kodak Gold 200",
        "Everyday film. Warm tones, reliable color, slight yellow bias.", "neg"),
    # Negative - Kodak UltraMax
    "Kodak UltraMax 400": ("Neg / Kodak UltraMax 400", "Kodak UltraMax 400",
        "Affordable all-rounder. Warm, saturated, cheerful. Student staple.", "neg"),
    # Negative - Fuji Pro
    "Fuji 160C": ("Neg / Fuji Pro 160C", "Fuji Pro 160C",
        "Cool-balanced professional. Neutral skin, cool shadows.", "neg"),
    "Fuji 160S": ("Neg / Fuji Pro 160S", "Fuji Pro 160S",
        "Standard-balanced professional. Natural, smooth, versatile.", "neg"),
    "Fuji 400H": ("Neg / Fuji Pro 400H", "Fuji Pro 400H",
        "Cool pastel tones. Blue-green shadows, lifted greens. Wedding favorite.", "neg"),
    "Fuji 800Z": ("Neg / Fuji Pro 800Z", "Fuji Pro 800Z",
        "High-speed Fuji pro. Warm midtones, slightly green shadows.", "neg"),
    # Negative - Fuji Consumer
    "Fuji Superia 400": ("Neg / Fuji Superia 400", "Fuji Superia 400",
        "Everyday Fuji 400. Green-ish shadows, decent saturation.", "neg"),
    "Fuji Superia 800": ("Neg / Fuji Superia 800", "Fuji Superia 800",
        "High-speed consumer Fuji. Grainier, green bias stronger.", "neg"),
    "Fuji Superia 1600": ("Neg / Fuji Superia 1600", "Fuji Superia 1600",
        "Ultra high speed consumer. Heavy grain, strong color shifts.", "neg"),
    # Negative - Agfa
    "Agfa Vista 400": ("Neg / Agfa Vista 400", "Agfa Vista 400",
        "Discontinued Agfa. Warm tones, moderate contrast, European color science.", "neg"),
    # Slide - Fuji
    "Fuji Velvia 50": ("Slide / Fuji Velvia 50", "Fuji Velvia 50",
        "Legendary. Extreme saturation, deep blacks, electric blues.", "slide"),
    "Fuji Velvia 100": ("Slide / Fuji Velvia 100", "Fuji Velvia 100",
        "Slightly less extreme than 50. Still vivid, finer grain.", "slide"),
    "Fuji Velvia 100F": ("Slide / Fuji Velvia 100F", "Fuji Velvia 100F",
        "Velvia with finer grain and slightly less saturation. More controlled.", "slide"),
    "Fuji Provia 100F": ("Slide / Fuji Provia 100F", "Fuji Provia 100F",
        "Neutral slide film. Accurate color, moderate saturation. All-rounder.", "slide"),
    "Fuji Provia 400X": ("Slide / Fuji Provia 400X", "Fuji Provia 400X",
        "Fast slide film. More grain, slightly less saturation than 100F.", "slide"),
    "Fuji Astia 100F": ("Slide / Fuji Astia 100F", "Fuji Astia 100F",
        "Soft slide film. Lower contrast, gentle saturation. Portrait slide.", "slide"),
    # Slide - Kodak Ektachrome
    "Kodak E100G": ("Slide / Kodak Ektachrome 100G", "Kodak Ektachrome 100G",
        "Modern Ektachrome. Clean, slightly cool, professional.", "slide"),
    "Kodak E100VS": ("Slide / Kodak Ektachrome 100VS", "Kodak Ektachrome 100VS",
        "Vivid Saturation Ektachrome. Kodak's answer to Velvia.", "slide"),
    # Slide - Agfa
    "Agfa RSX 200 II": ("Slide / Agfa RSX II 200", "Agfa RSX II 200",
        "Agfa professional slide. Cool, slightly blue, clean.", "slide"),
}

# NEW STOCKS: from C1 that we don't have yet
NEW_STOCKS = {
    # Negative - Kodak
    "Kodak Portra 100T": ("Neg / Kodak Portra 100T", "Kodak Portra 100T",
        "Tungsten-balanced Portra. Cool under daylight, neutral under tungsten.", "neg"),
    "Kodak Portra 400 UC": ("Neg / Kodak Portra 400UC", "Kodak Portra 400UC",
        "Ultra Color 400. More saturated than standard Portra, punchy.", "neg"),
    "Kodak Ektar 25": ("Neg / Kodak Ektar 25", "Kodak Ektar 25",
        "Ultrafine grain, extreme sharpness. Discontinued legendary landscape stock.", "neg"),
    "Kodak Gold 100 + Alt 2": None,  # skip variant
    "Kodak Max 800": ("Neg / Kodak Max 800", "Kodak Max 800",
        "High-speed consumer. Warm, grainy, contrasty. Party/event film.", "neg"),
    "Kodak Royal Gold 400": ("Neg / Kodak Royal Gold 400", "Kodak Royal Gold 400",
        "Premium consumer stock. Finer grain than Gold, richer colors.", "neg"),
    "Kodak UltraMax 800": ("Neg / Kodak UltraMax 800", "Kodak UltraMax 800",
        "High-speed UltraMax. More grain and warmth than 400.", "neg"),
    "Kodak E200": ("Slide / Kodak Ektachrome 200", "Kodak Ektachrome 200",
        "Fast Ektachrome. Pushable to 800. Slightly warm for a slide.", "slide"),
    "Kodak Ektachrome 64": ("Slide / Kodak Ektachrome 64", "Kodak Ektachrome 64",
        "Classic Ektachrome. Moderate saturation, fine grain, cool tones.", "slide"),
    "Kodak Ektachrome 64T": ("Slide / Kodak Ektachrome 64T", "Kodak Ektachrome 64T",
        "Tungsten-balanced Ektachrome. Neutral under artificial light.", "slide"),
    "Kodak Elite 50 II": ("Slide / Kodak Elite Chrome 50", "Kodak Elite Chrome 50",
        "Consumer slide. Fine grain, moderate saturation. Discontinued.", "slide"),
    "Kodak Elite Chrome 160T": ("Slide / Kodak Elite Chrome 160T", "Kodak Elite Chrome 160T",
        "Tungsten consumer slide. Fast for indoor use.", "slide"),
    # Negative - Fuji
    "Fuji Superia 100": ("Neg / Fuji Superia 100", "Fuji Superia 100",
        "Low-speed consumer Fuji. Fine grain, slightly cool, clean.", "neg"),
    "Fuji Sensia 100": ("Slide / Fuji Sensia 100", "Fuji Sensia 100",
        "Consumer slide film. Less saturated than Provia, affordable E-6.", "slide"),
    "Fuji Fortia SP": ("Slide / Fuji Fortia SP", "Fuji Fortia SP",
        "Ultra-vivid slide. Even more saturated than Velvia. Extreme landscape film.", "slide"),
    "Fuji T64": ("Slide / Fuji T64", "Fuji T64",
        "Tungsten-balanced Fuji slide. Studio/product photography staple.", "slide"),
    # Negative - Agfa
    "Agfa Optima 100 II": ("Neg / Agfa Optima 100", "Agfa Optima 100",
        "German consumer stock. Natural colors, moderate contrast. Discontinued.", "neg"),
    "Agfa Portrait XPS 160": ("Neg / Agfa Portrait XPS 160", "Agfa Portrait XPS 160",
        "Agfa's portrait stock. Warm skin tones, wide latitude. Professional.", "neg"),
    "Agfa Ultra 50": ("Neg / Agfa Ultra 50", "Agfa Ultra 50",
        "Ultra-saturated Agfa. Vivid colors, fine grain. European Ektar.", "neg"),
    "Agfa Ultra 100": ("Neg / Agfa Ultra 100", "Agfa Ultra 100",
        "Saturated consumer Agfa. Punchy colors, moderate grain.", "neg"),
    "Agfa Vista 100": ("Neg / Agfa Vista 100", "Agfa Vista 100",
        "Budget Agfa consumer. Warm, slightly muted. European nostalgia.", "neg"),
    "Agfa Vista 800": ("Neg / Agfa Vista 800", "Agfa Vista 800",
        "High-speed Agfa. Warm, grainy, strong character.", "neg"),
    # Instant / Polaroid
    "Polaroid 665": ("Instant / Polaroid 665", "Polaroid 665",
        "Peel-apart B&W positive/negative. Fine grain, beautiful tones. Legendary.", "instant"),
    "Polaroid 669": ("Instant / Polaroid 669", "Polaroid 669",
        "Peel-apart color. Muted, creamy, soft. Transfer art staple.", "instant"),
    "Polaroid 690": ("Instant / Polaroid 690", "Polaroid 690",
        "Professional peel-apart color. Sharper than 669, still dreamy.", "instant"),
    "Time-Zero Polaroid (Expired)": ("Instant / Polaroid Time-Zero", "Polaroid Time-Zero",
        "Expired SX-70 Time-Zero. Heavy color shifts, soft focus, ethereal.", "instant"),
    "PX-680": ("Instant / Impossible PX-680", "Impossible PX-680",
        "Impossible Project 600-type. Unpredictable colors, vintage instant.", "instant"),
    "PX-70": ("Instant / Impossible PX-70", "Impossible PX-70",
        "Impossible Project SX-70 type. Faded, dreamy, low saturation.", "instant"),
    # Fuji instant
    "Fuji FP-100c": ("Instant / Fuji FP-100c", "Fuji FP-100c",
        "Peel-apart instant color. Sharp, vivid for instant. Cult classic.", "instant"),
}


def main():
    c1 = load_c1_data()

    # ── Process refinements ──
    print("=" * 60)
    print("REFINED STOCKS (existing, now with C1 data)")
    print("=" * 60)

    refined_neg = []
    refined_slide = []

    for c1_name, (key, display, desc, cat) in sorted(REFINE_MAP.items()):
        if c1_name not in c1:
            print(f"  [SKIP] {c1_name} - not in C1 data")
            continue
        params = c1_to_our_params(c1[c1_name])
        line = generate_stock_line(key, display, desc, params)
        if cat == "neg":
            refined_neg.append(line)
        else:
            refined_slide.append(line)
        print(f"  [OK] {c1_name} -> {key}")

    # ── Process new stocks ──
    print()
    print("=" * 60)
    print("NEW STOCKS (from C1)")
    print("=" * 60)

    new_neg = []
    new_slide = []
    new_instant = []

    for c1_name, mapping in sorted(NEW_STOCKS.items()):
        if mapping is None:
            continue
        key, display, desc, cat = mapping
        if c1_name not in c1:
            print(f"  [SKIP] {c1_name} - not in C1 data")
            continue
        params = c1_to_our_params(c1[c1_name])
        line = generate_stock_line(key, display, desc, params)
        if cat == "neg":
            new_neg.append(line)
        elif cat == "slide":
            new_slide.append(line)
        elif cat == "instant":
            new_instant.append(line)
        print(f"  [OK] {c1_name} -> {key}")

    # ── Output for copy-paste ──
    # .txt, not .py: this is a bare dict BODY meant to be pasted into
    # data/color_stocks.py, so it has never been importable on its own. Naming it
    # .py made every linter that walks the package report it as broken source.
    output_path = Path(__file__).parent / "generated_stocks.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ═══════════════════════════════════════════════════════════\n")
        f.write("# REFINED NEGATIVE STOCKS (C1 data)\n")
        f.write("# ═══════════════════════════════════════════════════════════\n\n")
        for line in refined_neg:
            f.write(line + "\n")
        f.write("\n")

        f.write("# ═══════════════════════════════════════════════════════════\n")
        f.write("# REFINED SLIDE STOCKS (C1 data)\n")
        f.write("# ═══════════════════════════════════════════════════════════\n\n")
        for line in refined_slide:
            f.write(line + "\n")
        f.write("\n")

        f.write("# ═══════════════════════════════════════════════════════════\n")
        f.write("# NEW NEGATIVE STOCKS (from C1)\n")
        f.write("# ═══════════════════════════════════════════════════════════\n\n")
        for line in new_neg:
            f.write(line + "\n")
        f.write("\n")

        f.write("# ═══════════════════════════════════════════════════════════\n")
        f.write("# NEW SLIDE STOCKS (from C1)\n")
        f.write("# ═══════════════════════════════════════════════════════════\n\n")
        for line in new_slide:
            f.write(line + "\n")
        f.write("\n")

        f.write("# ═══════════════════════════════════════════════════════════\n")
        f.write("# NEW INSTANT STOCKS (from C1)\n")
        f.write("# ═══════════════════════════════════════════════════════════\n\n")
        for line in new_instant:
            f.write(line + "\n")

    print(f"\nGenerated stock code written to: {output_path}")

    # ── B&W data ──
    print()
    print("=" * 60)
    print("B&W STOCKS (C1 channel mix data)")
    print("=" * 60)

    bw_output_path = Path(__file__).parent / "generated_bw_stocks.py"
    with open(bw_output_path, "w", encoding="utf-8") as f:
        f.write("# B&W stock channel mix data from Capture One\n")
        f.write("# C1 uses: BwRed, BwGreen, BwBlue (relative adjustments from default)\n")
        f.write("# C1 also has: BwCyan, BwMagenta, BwYellow for CMY fine-tuning\n\n")

        for name, data in sorted(c1.items()):
            if not data.get("is_bw"):
                continue
            bw = data.get("bw_mix", {})
            grain = data["grain"]
            f.write(f'# {name}\n')
            f.write(f'#   Channel mix: R={bw.get("red", 0)}, G={bw.get("green", 0)}, B={bw.get("blue", 0)}\n')
            f.write(f'#   CMY: C={bw.get("cyan", 0)}, M={bw.get("magenta", 0)}, Y={bw.get("yellow", 0)}\n')
            f.write(f'#   Grain: amount={grain["amount"]}, type={grain["type"]}\n\n')
            print(f"  [OK] {name}")

    print(f"\nB&W data written to: {bw_output_path}")


if __name__ == "__main__":
    main()
