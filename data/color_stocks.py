"""
Color film stock profiles for ComfyUI-Darkroom.

Comprehensive library covering negative, slide, cinema, instant,
camera simulation, and aged/expired film looks.

Curve parameters: (toe_power, shoulder_power, slope, pivot_x, pivot_y)
- toe_power: >1 compresses shadows more (film's gentle shadow rolloff)
- shoulder_power: >1 gives softer highlight compression (film's hallmark)
- slope: midtone contrast multiplier
- pivot_x/pivot_y: typically 0.18 (18% grey, the photographic midpoint)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CurveParams:
    toe_power: float
    shoulder_power: float
    slope: float
    pivot_x: float
    pivot_y: float


@dataclass(frozen=True)
class ColorFilmStock:
    name: str
    r_curve: CurveParams
    g_curve: CurveParams
    b_curve: CurveParams
    saturation: float
    shadow_tint: Tuple[float, float, float]
    highlight_tint: Tuple[float, float, float]
    description: str


# Helper: build a stock quickly with per-channel offsets from a base curve
def _stock(name, desc, toe, shoulder, slope,
           r_off=(0, 0, 0), g_off=(0, 0, 0), b_off=(0, 0, 0),
           sat=1.0, s_tint=(0, 0, 0), h_tint=(0, 0, 0),
           pivot_x=0.18, pivot_y=0.18):
    return ColorFilmStock(
        name=name,
        r_curve=CurveParams(toe + r_off[0], shoulder + r_off[1], slope + r_off[2], pivot_x, pivot_y + r_off[2] * 0.02),
        g_curve=CurveParams(toe + g_off[0], shoulder + g_off[1], slope + g_off[2], pivot_x, pivot_y + g_off[2] * 0.02),
        b_curve=CurveParams(toe + b_off[0], shoulder + b_off[1], slope + b_off[2], pivot_x, pivot_y + b_off[2] * 0.02),
        saturation=sat, shadow_tint=s_tint, highlight_tint=h_tint, description=desc,
    )


# =============================================================================
# NEGATIVE FILMS — Print / Professional
# =============================================================================

COLOR_NEGATIVE_PRINT = {
    # --- KODAK PORTRA FAMILY ---
    "Neg / Kodak Portra 160": _stock(
        "Kodak Portra 160", "Ultra-smooth portrait stock. Muted colors, warm skin, wide latitude.",
        1.55, 1.57, 1.0, r_off=(0.011, -0.089, 0.028), g_off=(-0.068, 0.021, -0.025), b_off=(0.058, 0.068, -0.003),
        sat=1.0, s_tint=(-0.1, -0.1, -0.1), h_tint=(-0.0625, -0.0625, -0.0625)),
    "Neg / Kodak Portra 160NC": _stock(
        "Kodak Portra 160NC", "Natural Color variant. Even more muted, ultimate skin fidelity.",
        1.75, 1.33, 1.12, r_off=(0.047, -0.006, 0.015), g_off=(-0.031, 0.042, -0.022), b_off=(-0.016, -0.037, 0.007),
        sat=0.95, s_tint=(-0.1375, -0.1375, -0.1375), h_tint=(-0.12, -0.12, -0.12)),
    "Neg / Kodak Portra 160VC": _stock(
        "Kodak Portra 160VC", "Vivid Color variant. More saturation than standard, still gentle.",
        1.72, 1.33, 1.12, r_off=(0.073, -0.047, 0.034), g_off=(0.01, 0.048, -0.01), b_off=(-0.084, -0.0, -0.025),
        sat=1.0, s_tint=(-0.2, -0.2, -0.2), h_tint=(-0.12, -0.12, -0.12)),
    "Neg / Kodak Portra 400": _stock(
        "Kodak Portra 400", "The workhorse portrait stock. Warm, forgiving, beautiful skin tones.",
        1.55, 1.58, 1.0, r_off=(0.058, 0.047, 0.005), g_off=(-0.068, 0.016, -0.025), b_off=(0.011, -0.062, 0.02),
        sat=1.05, s_tint=(-0.0875, -0.0875, -0.0875), h_tint=(-0.0625, -0.0625, -0.0625)),
    "Neg / Kodak Portra 400NC": _stock(
        "Kodak Portra 400NC", "Natural Color 400. Muted pastels, clean highlights.",
        1.63, 1.43, 1.06, r_off=(0.028, -0.026, 0.014), g_off=(0.017, 0.021, 0.0), b_off=(-0.045, 0.005, -0.014),
        sat=0.94, s_tint=(-0.185, -0.185, -0.185), h_tint=(-0.125, -0.125, -0.125)),
    "Neg / Kodak Portra 400VC": _stock(
        "Kodak Portra 400VC", "Vivid Color 400. Richer than NC, still portrait-safe.",
        1.72, 1.34, 1.12, r_off=(0.087, -0.026, 0.036), g_off=(-0.007, 0.037, -0.014), b_off=(-0.08, -0.011, -0.021),
        sat=1.0, s_tint=(-0.125, -0.125, -0.125), h_tint=(-0.125, -0.125, -0.125)),
    "Neg / Kodak Portra 800": _stock(
        "Kodak Portra 800", "High-speed Portra. More grain, green-cyan shadows, gritty.",
        1.58, 1.55, 1.01, r_off=(0.011, 0.031, -0.005), g_off=(-0.005, -0.047, 0.01), b_off=(-0.005, 0.016, -0.005),
        sat=1.0, s_tint=(-0.1, -0.1, -0.1), h_tint=(-0.05, -0.05, -0.05)),
    # --- KODAK PORTRA (new from C1) ---
    "Neg / Kodak Portra 100T": _stock(
        "Kodak Portra 100T", "Tungsten-balanced Portra. Cool under daylight, neutral under tungsten.",
        1.72, 1.45, 1.07, r_off=(-0.032, 0.052, -0.07), g_off=(-0.063, -0.105, 0.007), b_off=(0.094, 0.052, 0.063),
        sat=1.0, s_tint=(-0.025, -0.025, -0.025), h_tint=(0.0127, 0.0127, 0.0127)),
    "Neg / Kodak Portra 400UC": _stock(
        "Kodak Portra 400UC", "Ultra Color 400. More saturated than standard Portra, punchy.",
        1.73, 1.21, 1.16, r_off=(0.034, -0.021, 0.019), g_off=(-0.028, 0.011, -0.01), b_off=(-0.007, 0.011, -0.009),
        sat=1.05, s_tint=(-0.135, -0.135, -0.135), h_tint=(-0.125, -0.125, -0.125)),

    # --- KODAK EKTAR ---
    "Neg / Kodak Ektar 100": _stock(
        "Kodak Ektar 100", "Vivid landscape stock. High saturation, punchy contrast, electric colors.",
        2.11, 1.0, 1.42, r_off=(0.141, 0.0, 0.058), g_off=(-0.016, 0.0, -0.013), b_off=(-0.126, 0.0, -0.044),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak Ektar 25": _stock(
        "Kodak Ektar 25", "Ultrafine grain, extreme sharpness. Discontinued legendary landscape stock.",
        1.8, 1.18, 1.16, r_off=(0.011, 0.073, -0.015), g_off=(-0.005, 0.042, -0.009), b_off=(-0.005, -0.115, 0.024),
        sat=1.0, s_tint=(-0.025, -0.025, -0.025), h_tint=(-0.0375, -0.0375, -0.0375)),

    # --- KODAK GOLD ---
    "Neg / Kodak Gold 100": _stock(
        "Kodak Gold 100", "Consumer classic. Warm, slightly yellow cast, modest saturation.",
        1.99, 1.13, 1.27, r_off=(0.041, 0.052, -0.002), g_off=(0.026, -0.026, 0.013), b_off=(-0.068, -0.026, -0.011),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak Gold 200": _stock(
        "Kodak Gold 200", "Everyday film. Warm tones, reliable color, slight yellow bias.",
        2.13, 1.0, 1.41, r_off=(0.057, 0.0, 0.028), g_off=(0.01, 0.0, -0.003), b_off=(-0.068, 0.0, -0.026),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak Gold 400": _stock(
        "Kodak Gold 400", "Higher speed Gold. Warmer, grainier, more contrast than 200.",
        1.35, 1.48, 1.05, r_off=(0, 0, 0.03), b_off=(0.08, -0.05, -0.07),
        sat=0.98, s_tint=(0.025, 0.012, -0.015), h_tint=(0.018, 0.01, -0.012)),

    # --- KODAK ULTRAMAX / COLORPLUS / MAX ---
    "Neg / Kodak UltraMax 400": _stock(
        "Kodak UltraMax 400", "Affordable all-rounder. Warm, saturated, cheerful. Student staple.",
        1.9, 1.0, 1.34, r_off=(0.068, 0.0, 0.02), g_off=(0.021, 0.0, 0.019), b_off=(-0.089, 0.0, -0.039),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak UltraMax 800": _stock(
        "Kodak UltraMax 800", "High-speed UltraMax. More grain and warmth than 400.",
        1.87, 1.01, 1.33, r_off=(0.026, 0.019, -0.017), g_off=(-0.037, -0.01, 0.017), b_off=(0.01, -0.01, -0.0),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak ColorPlus 200": _stock(
        "Kodak ColorPlus 200", "Budget film. Warm, slightly flat, nostalgic feel.",
        1.35, 1.55, 0.98, r_off=(0, 0, 0.02), b_off=(0.1, -0.05, -0.05),
        sat=0.92, s_tint=(0.02, 0.01, -0.012), h_tint=(0.012, 0.008, -0.008)),
    "Neg / Kodak Ultra Color 100UC": _stock(
        "Kodak Ultra Color 100UC", "Ultra Color saturated consumer stock. Vivid but not slide-level.",
        1.45, 1.5, 1.08, r_off=(0, 0, 0.03), b_off=(0.05, -0.05, -0.04),
        sat=1.15, s_tint=(0.01, 0.0, -0.008), h_tint=(0.008, 0.005, -0.005)),
    "Neg / Kodak Max 800": _stock(
        "Kodak Max 800", "High-speed consumer. Warm, grainy, contrasty. Party/event film.",
        1.79, 1.0, 1.26, r_off=(0.073, 0.0, 0.016), g_off=(0.026, 0.0, 0.019), b_off=(-0.1, 0.0, -0.035),
        sat=0.85, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Kodak Royal Gold 400": _stock(
        "Kodak Royal Gold 400", "Premium consumer stock. Finer grain than Gold, richer colors.",
        1.9, 1.14, 1.26, r_off=(0.058, 0.042, -0.006), g_off=(-0.021, -0.005, -0.005), b_off=(-0.036, -0.036, 0.011),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),

    # --- FUJI PROFESSIONAL ---
    "Neg / Fuji Pro 160C": _stock(
        "Fuji Pro 160C", "Cool-balanced professional. Neutral skin, cool shadows.",
        1.59, 1.63, 1.0, r_off=(0.063, 0.011, 0.015), g_off=(-0.078, -0.005, -0.015), b_off=(0.016, -0.005, 0.0),
        sat=1.0, s_tint=(-0.1, -0.1, -0.1), h_tint=(-0.1125, -0.1125, -0.1125)),
    "Neg / Fuji Pro 160S": _stock(
        "Fuji Pro 160S", "Standard-balanced professional. Natural, smooth, versatile.",
        1.79, 1.32, 1.12, r_off=(0.083, -0.042, 0.069), g_off=(-0.042, 0.021, -0.036), b_off=(-0.042, 0.021, -0.033),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(-0.0125, -0.0125, -0.0125)),
    "Neg / Fuji Pro 400H": _stock(
        "Fuji Pro 400H", "Cool pastel tones. Blue-green shadows, lifted greens. Wedding favorite.",
        1.55, 1.55, 1.02, r_off=(0.031, -0.094, 0.037), g_off=(-0.078, 0.016, -0.022), b_off=(0.047, 0.078, -0.015),
        sat=1.0, s_tint=(-0.0875, -0.0875, -0.0875), h_tint=(-0.1125, -0.1125, -0.1125)),
    "Neg / Fuji Pro 800Z": _stock(
        "Fuji Pro 800Z", "High-speed Fuji pro. Warm midtones, slightly green shadows.",
        1.56, 1.44, 1.05, r_off=(0.077, -0.031, 0.033), g_off=(-0.09, -0.062, -0.002), b_off=(0.014, 0.094, -0.03),
        sat=1.0, s_tint=(-0.0875, -0.0875, -0.0875), h_tint=(-0.05, -0.05, -0.05)),

    # --- FUJI CONSUMER ---
    "Neg / Fuji Reala 100": _stock(
        "Fuji Reala 100", "4th color layer for accurate daylight/tungsten rendering. Natural.",
        1.4, 1.65, 0.98, r_off=(0, 0, 0), b_off=(0.05, -0.05, 0),
        sat=0.95, s_tint=(0.0, 0.005, 0.005), h_tint=(0.005, 0.005, 0.003)),
    "Neg / Fuji Reala - Pushed 1 Stop": _stock(
        "Fuji Reala Pushed 1", "Reala push-processed. More contrast, slightly shifted colors.",
        1.3, 1.55, 1.08, r_off=(0, 0, 0.02), b_off=(0.05, -0.08, -0.02),
        sat=1.0, s_tint=(0.005, 0.008, 0.005), h_tint=(0.008, 0.005, 0.0)),
    "Neg / Fuji Reala - Pushed 2 Stops": _stock(
        "Fuji Reala Pushed 2", "Reala pushed hard. High contrast, color shifts emerge.",
        1.2, 1.45, 1.18, r_off=(0, 0, 0.04), b_off=(0.05, -0.1, -0.05),
        sat=1.05, s_tint=(0.01, 0.012, 0.005), h_tint=(0.012, 0.005, -0.005)),
    "Neg / Fuji Superia 100": _stock(
        "Fuji Superia 100", "Low-speed consumer Fuji. Fine grain, slightly cool, clean.",
        1.53, 1.47, 1.02, r_off=(0.063, 0.01, 0.013), g_off=(-0.031, -0.021, -0.003), b_off=(-0.031, 0.01, -0.01),
        sat=0.95, s_tint=(-0.1625, -0.1625, -0.1625), h_tint=(0.0152, 0.0152, 0.0152)),
    "Neg / Fuji Superia 200": _stock(
        "Fuji Superia 200", "Consumer Fuji. Cooler than Kodak Gold, green-tinted shadows.",
        1.38, 1.55, 1.0, g_off=(-0.03, 0, 0.01), b_off=(-0.05, 0.05, 0),
        sat=0.95, s_tint=(-0.005, 0.01, 0.008), h_tint=(0.005, 0.008, 0.003)),
    "Neg / Fuji Superia 400": _stock(
        "Fuji Superia 400", "Everyday Fuji 400. Green-ish shadows, decent saturation.",
        1.81, 1.2, 1.2, r_off=(0.052, -0.026, 0.015), g_off=(-0.058, 0.005, -0.015), b_off=(0.005, 0.021, -0.0),
        sat=0.95, s_tint=(-0.0775, -0.0775, -0.0775), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Fuji Superia 800": _stock(
        "Fuji Superia 800", "High-speed consumer Fuji. Grainier, green bias stronger.",
        1.78, 1.15, 1.18, r_off=(0.084, -0.042, 0.035), g_off=(-0.042, -0.042, -0.0), b_off=(-0.042, 0.083, -0.035),
        sat=0.95, s_tint=(-0.175, -0.175, -0.175), h_tint=(-0.05, -0.05, -0.05)),
    "Neg / Fuji Superia 1600": _stock(
        "Fuji Superia 1600", "Ultra high speed consumer. Heavy grain, strong color shifts.",
        1.7, 1.28, 1.14, r_off=(0.047, -0.01, 0.014), g_off=(-0.063, 0.005, -0.021), b_off=(0.016, 0.005, 0.006),
        sat=0.95, s_tint=(-0.1875, -0.1875, -0.1875), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Fuji C200": _stock(
        "Fuji C200", "Budget Fuji. Slight green cast, low saturation, nostalgic.",
        1.38, 1.58, 0.97, g_off=(-0.03, 0, 0.01), b_off=(-0.05, 0.05, 0),
        sat=0.88, s_tint=(-0.003, 0.01, 0.008), h_tint=(0.003, 0.008, 0.003)),

    # --- AGFA ---
    "Neg / Agfa Vista 100": _stock(
        "Agfa Vista 100", "Budget Agfa consumer. Warm, slightly muted. European nostalgia.",
        2.2, 1.03, 1.41, r_off=(0.041, -0.031, 0.012), g_off=(-0.021, 0.061, -0.018), b_off=(-0.021, -0.031, 0.006),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Agfa Vista 200": _stock(
        "Agfa Vista 200", "European consumer stock. Warm, slightly muted. Discontinued nostalgia.",
        1.38, 1.55, 0.98, r_off=(0, 0, 0.02), b_off=(0.08, -0.05, -0.04),
        sat=0.93, s_tint=(0.015, 0.008, -0.01), h_tint=(0.01, 0.008, -0.005)),
    "Neg / Agfa Vista 400": _stock(
        "Agfa Vista 400", "Discontinued Agfa. Warm tones, moderate contrast, European color science.",
        1.94, 1.0, 1.37, r_off=(0.11, 0.0, 0.029), g_off=(0.0, 0.0, 0.002), b_off=(-0.11, 0.0, -0.032),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Agfa Vista 800": _stock(
        "Agfa Vista 800", "High-speed Agfa. Warm, grainy, strong character.",
        1.91, 1.01, 1.33, r_off=(0.152, 0.005, 0.039), g_off=(-0.005, 0.005, -0.019), b_off=(-0.146, -0.009, -0.02),
        sat=0.9, s_tint=(-0.0458, -0.0286, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Neg / Agfa Optima 100": _stock(
        "Agfa Optima 100", "German consumer stock. Natural colors, moderate contrast. Discontinued.",
        1.71, 1.26, 1.15, r_off=(-0.005, -0.073, 0.022), g_off=(0.026, 0.021, 0.001), b_off=(-0.021, 0.053, -0.023),
        sat=0.95, s_tint=(-0.1, -0.1, -0.1), h_tint=(0.0256, 0.0256, 0.0256)),
    "Neg / Agfa Portrait XPS 160": _stock(
        "Agfa Portrait XPS 160", "Agfa's portrait stock. Warm skin tones, wide latitude. Professional.",
        1.82, 1.0, 1.3, r_off=(0.042, 0.0, -0.055), g_off=(-0.021, 0.0, 0.037), b_off=(-0.021, 0.0, 0.018),
        sat=1.0, s_tint=(-0.025, -0.025, -0.025), h_tint=(-0.05, -0.05, -0.05)),
    "Neg / Agfa Ultra 50": _stock(
        "Agfa Ultra 50", "Ultra-saturated Agfa. Vivid colors, fine grain. European Ektar.",
        1.77, 1.39, 1.1, r_off=(0.136, 0.084, 0.013), g_off=(0.01, 0.084, -0.018), b_off=(-0.147, -0.167, 0.005),
        sat=1.0, s_tint=(-0.1125, -0.1125, -0.1125), h_tint=(-0.0625, -0.0625, -0.0625)),
    "Neg / Agfa Ultra 100": _stock(
        "Agfa Ultra 100", "Saturated consumer Agfa. Punchy colors, moderate grain.",
        1.7, 1.35, 1.11, r_off=(0.083, -0.042, 0.046), g_off=(-0.105, 0.021, -0.054), b_off=(0.021, 0.021, 0.008),
        sat=1.05, s_tint=(-0.075, -0.075, -0.075), h_tint=(0.0, 0.0, 0.0)),

    # --- CINESTILL ---
    "Neg / Cinestill 50D": _stock(
        "Cinestill 50D", "Cinema stock for stills. Daylight balanced, clean, fine grain.",
        1.4, 1.7, 0.95, r_off=(0, 0, 0), b_off=(0.05, -0.05, -0.02),
        sat=0.95, s_tint=(0.005, 0.0, -0.005), h_tint=(0.005, 0.003, 0.0)),
    "Neg / Cinestill 800T": _stock(
        "Cinestill 800T", "Cinema stock without remjet. Tungsten-balanced, warm highlights. Pair with Halation node.",
        1.3, 1.5, 1.0, r_off=(0, 0, 0.01), b_off=(0.1, 0.05, -0.05),
        sat=0.95, s_tint=(0.0, 0.012, 0.008), h_tint=(0.02, 0.01, -0.01)),

    # --- LOMOGRAPHY ---
    "Neg / Lomography 100": _stock(
        "Lomography 100", "Vivid consumer stock. Saturated, contrasty, lomo aesthetic.",
        1.45, 1.4, 1.1, r_off=(0.05, 0, 0.03), b_off=(0.05, 0, -0.03),
        sat=1.15, s_tint=(0.01, -0.005, -0.01), h_tint=(0.01, 0.005, -0.005)),
    "Neg / Lomography 400": _stock(
        "Lomography 400", "Punchy 400 speed lomo. Strong saturation, cross-process friendly.",
        1.4, 1.38, 1.12, r_off=(0.05, 0, 0.03), b_off=(0.05, 0, -0.04),
        sat=1.18, s_tint=(0.012, -0.005, -0.01), h_tint=(0.012, 0.005, -0.008)),
    "Neg / Lomography 800": _stock(
        "Lomography 800", "High-speed lomo. Grain and color shifts add to the aesthetic.",
        1.35, 1.35, 1.15, r_off=(0.05, 0, 0.04), b_off=(0.03, 0, -0.05),
        sat=1.12, s_tint=(0.015, -0.003, -0.012), h_tint=(0.015, 0.005, -0.01)),
}

# =============================================================================
# SLIDE FILMS — Reversal / Transparency
# =============================================================================

COLOR_SLIDE = {
    # --- FUJI VELVIA ---
    "Slide / Fuji Velvia 50": _stock(
        "Fuji Velvia 50", "Legendary. Extreme saturation, deep blacks, electric blues.",
        2.22, 1.0, 1.42, r_off=(0.031, 0.0, -0.029), g_off=(-0.032, 0.0, -0.024), b_off=(0.0, 0.0, 0.053),
        sat=1.05, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Slide / Fuji Velvia 100": _stock(
        "Fuji Velvia 100", "Slightly less extreme than 50. Still vivid, finer grain.",
        2.16, 1.0, 1.39, r_off=(0.0, 0.0, -0.042), g_off=(0.031, 0.0, -0.01), b_off=(-0.032, 0.0, 0.051),
        sat=1.05, s_tint=(-0.025, -0.025, -0.025), h_tint=(0.0256, 0.0256, 0.0256)),
    "Slide / Fuji Velvia 100F": _stock(
        "Fuji Velvia 100F", "Velvia with finer grain and slightly less saturation. More controlled.",
        1.94, 1.0, 1.28, r_off=(-0.0, 0.0, -0.015), g_off=(0.032, 0.0, 0.014), b_off=(-0.031, 0.0, 0.0),
        sat=1.05, s_tint=(-0.08, -0.08, -0.08), h_tint=(0.0, 0.0, 0.0)),

    # --- FUJI PROVIA ---
    "Slide / Fuji Provia 100F": _stock(
        "Fuji Provia 100F", "Neutral slide film. Accurate color, moderate saturation. All-rounder.",
        2.05, 1.0, 1.33, r_off=(0.011, 0.0, 0.004), g_off=(-0.052, 0.0, -0.01), b_off=(0.042, 0.0, 0.006),
        sat=1.04, s_tint=(-0.1125, -0.1125, -0.1125), h_tint=(0.056, 0.0627, 0.0667)),
    "Slide / Fuji Provia 400X": _stock(
        "Fuji Provia 400X", "Fast slide film. More grain, slightly less saturation than 100F.",
        2.12, 1.0, 1.36, r_off=(0.073, 0.0, -0.001), g_off=(0.042, 0.0, 0.029), b_off=(-0.115, 0.0, -0.029),
        sat=1.0, s_tint=(-0.0125, -0.0125, -0.0125), h_tint=(0.0, 0.0, 0.0)),

    # --- FUJI ASTIA ---
    "Slide / Fuji Astia 100F": _stock(
        "Fuji Astia 100F", "Soft slide film. Lower contrast, gentle saturation. Portrait slide.",
        1.87, 1.06, 1.19, r_off=(0.01, 0.124, -0.041), g_off=(0.01, -0.062, 0.014), b_off=(-0.021, -0.062, 0.027),
        sat=1.05, s_tint=(-0.0375, -0.0375, -0.0375), h_tint=(0.0204, 0.0204, 0.0204)),

    # --- FUJI SENSIA / FORTIA / T64 ---
    "Slide / Fuji Sensia 100": _stock(
        "Fuji Sensia 100", "Consumer slide film. Less saturated than Provia, affordable E-6.",
        1.92, 1.04, 1.34, r_off=(-0.01, -0.041, 0.003), g_off=(-0.01, -0.041, 0.039), b_off=(0.021, 0.083, -0.042),
        sat=1.0, s_tint=(-0.05, -0.05, -0.05), h_tint=(0.0127, 0.0127, 0.0127)),
    "Slide / Fuji T64": _stock(
        "Fuji T64", "Tungsten-balanced Fuji slide. Studio/product photography staple.",
        1.9, 1.0, 1.24, r_off=(0.136, 0.0, 0.064), g_off=(-0.021, 0.0, 0.029), b_off=(-0.115, 0.0, -0.093),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Slide / Fuji Fortia SP": _stock(
        "Fuji Fortia SP", "Ultra-vivid slide. Even more saturated than Velvia. Extreme landscape film.",
        2.27, 1.0, 1.56, r_off=(0.10, 0.0, 0.04), g_off=(0.0, 0.0, -0.007), b_off=(-0.05, 0.0, -0.02),
        sat=1.05, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),

    # --- KODAK EKTACHROME ---
    "Slide / Kodak Ektachrome 64": _stock(
        "Kodak Ektachrome 64", "Classic Ektachrome. Moderate saturation, fine grain, cool tones.",
        1.76, 1.36, 1.11, r_off=(0.021, -0.047, 0.016), g_off=(0.021, 0.063, -0.006), b_off=(-0.041, -0.016, -0.01),
        sat=1.0, s_tint=(-0.0375, -0.0375, -0.0375), h_tint=(-0.0625, -0.0625, -0.0625)),
    "Slide / Kodak Ektachrome 64T": _stock(
        "Kodak Ektachrome 64T", "Tungsten-balanced Ektachrome. Neutral under artificial light.",
        1.59, 1.54, 1.02, r_off=(0.126, 0.178, -0.011), g_off=(-0.0, -0.073, 0.007), b_off=(-0.125, -0.105, 0.004),
        sat=1.0, s_tint=(-0.0375, -0.0375, -0.0375), h_tint=(0.0256, 0.0256, 0.0256)),
    "Slide / Kodak Ektachrome 100G": _stock(
        "Kodak Ektachrome 100G", "Modern Ektachrome. Clean, slightly cool, professional.",
        2.04, 1.0, 1.31, r_off=(0.011, 0.0, -0.014), g_off=(-0.005, 0.0, 0.017), b_off=(-0.005, 0.0, -0.003),
        sat=1.05, s_tint=(-0.05, -0.05, -0.05), h_tint=(0.0, 0.0, 0.0)),
    "Slide / Kodak Ektachrome 100GX": _stock(
        "Kodak Ektachrome 100GX", "Enhanced Ektachrome. Warmer than 100G, more saturated.",
        1.65, 1.42, 1.12, r_off=(0, 0, 0.01), b_off=(-0.03, 0.03, 0),
        sat=1.15, s_tint=(0.0, 0.0, 0.005), h_tint=(0.008, 0.005, 0.003)),
    "Slide / Kodak Ektachrome 100VS": _stock(
        "Kodak Ektachrome 100VS", "Vivid Saturation Ektachrome. Kodak's answer to Velvia.",
        2.29, 1.0, 1.47, r_off=(0.057, 0.0, -0.003), g_off=(-0.021, 0.0, 0.002), b_off=(-0.037, 0.0, 0.001),
        sat=1.1, s_tint=(-0.06, -0.06, -0.06), h_tint=(0.0256, 0.0256, 0.0256)),
    "Slide / Kodak Ektachrome 200": _stock(
        "Kodak Ektachrome 200", "Fast Ektachrome. Pushable to 800. Slightly warm for a slide.",
        2.13, 1.0, 1.3, r_off=(0.032, 0.0, 0.007), g_off=(-0.0, 0.0, 0.007), b_off=(-0.031, 0.0, -0.014),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Slide / Kodak Ektachrome E100": _stock(
        "Kodak Ektachrome E100", "Modern reformulated Ektachrome (2018+). Clean, fine grain, neutral.",
        1.63, 1.48, 1.08, r_off=(0, 0, 0), b_off=(-0.03, 0.03, 0.01),
        sat=1.08, s_tint=(-0.002, 0.002, 0.005), h_tint=(0.003, 0.003, 0.005)),
    "Slide / Kodak Ektachrome EES": _stock(
        "Kodak Ektachrome EES", "High-speed push Ektachrome. Grain, contrast, warmth.",
        1.55, 1.35, 1.15, r_off=(0.03, 0, 0.02), b_off=(0, 0, -0.03),
        sat=1.08, s_tint=(0.008, 0.003, -0.005), h_tint=(0.01, 0.005, -0.005)),
    "Slide / Kodak Ektachrome EES - Subdued": _stock(
        "Kodak Ektachrome EES Subdued", "EES with restrained processing. Faded, muted character.",
        1.5, 1.45, 1.02, r_off=(0.02, 0, 0.01), b_off=(0, 0.05, -0.02),
        sat=0.88, s_tint=(0.008, 0.005, 0.0), h_tint=(0.008, 0.005, 0.0)),
    "Slide / Kodak Ektachrome mid-1970s": _stock(
        "Kodak Ektachrome 1970s", "Vintage Ektachrome. Blue cast, faded warmth, period color science.",
        1.5, 1.5, 1.05, r_off=(0, 0, -0.02), b_off=(-0.05, 0.05, 0.05),
        sat=0.95, s_tint=(-0.005, 0.0, 0.02), h_tint=(0.005, 0.003, 0.015)),

    # --- KODAK ELITE CHROME ---
    "Slide / Kodak Elite Chrome 50": _stock(
        "Kodak Elite Chrome 50", "Consumer slide. Fine grain, moderate saturation. Discontinued.",
        1.92, 1.08, 1.24, r_off=(0.0, 0.041, 0.024), g_off=(0.047, -0.083, 0.019), b_off=(-0.048, 0.041, -0.043),
        sat=0.9, s_tint=(-0.0375, -0.0375, -0.0375), h_tint=(0.0, 0.0, 0.0)),
    "Slide / Kodak Elite Chrome 160T": _stock(
        "Kodak Elite Chrome 160T", "Tungsten consumer slide. Fast for indoor use.",
        1.77, 1.11, 1.2, r_off=(0.11, -0.099, 0.069), g_off=(0.016, -0.099, 0.028), b_off=(-0.125, 0.199, -0.098),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0127, 0.0127, 0.0127)),

    # --- KODAK KODACHROME ---
    "Slide / Kodak Kodachrome 25": _stock(
        "Kodak Kodachrome 25", "The legend. Warm reds, deep blues, unique dye transfer process.",
        1.7, 1.4, 1.15, r_off=(0.05, 0, 0.05), b_off=(-0.05, 0.05, 0.02),
        sat=1.20, s_tint=(0.01, -0.005, -0.005), h_tint=(0.015, 0.005, 0.005)),
    "Slide / Kodak Kodachrome 25 (sharp)": _stock(
        "Kodak Kodachrome 25 Sharp", "Kodachrome 25 with more midtone contrast. Punchy version.",
        1.75, 1.35, 1.22, r_off=(0.05, 0, 0.06), b_off=(-0.05, 0.05, 0.02),
        sat=1.22, s_tint=(0.012, -0.005, -0.005), h_tint=(0.018, 0.005, 0.005)),
    "Slide / Kodak Kodachrome 64": _stock(
        "Kodak Kodachrome 64", "The standard Kodachrome. Rich, warm, iconic color rendition.",
        1.68, 1.42, 1.12, r_off=(0.04, 0, 0.04), b_off=(-0.05, 0.05, 0.01),
        sat=1.18, s_tint=(0.008, -0.005, -0.003), h_tint=(0.012, 0.005, 0.005)),
    "Slide / Kodak Kodachrome 200": _stock(
        "Kodak Kodachrome 200", "Fast Kodachrome. Grainier, less saturated than 64, warm.",
        1.6, 1.45, 1.08, r_off=(0.03, 0, 0.03), b_off=(-0.03, 0.05, 0),
        sat=1.10, s_tint=(0.008, -0.003, -0.003), h_tint=(0.01, 0.005, 0.003)),

    # --- AGFA SLIDE ---
    "Slide / Agfa RSX II 200": _stock(
        "Agfa RSX II 200", "Agfa professional slide. Cool, slightly blue, clean.",
        1.53, 1.56, 0.99, r_off=(0.032, -0.063, 0.017), g_off=(-0.031, -0.063, 0.005), b_off=(-0.0, 0.125, -0.022),
        sat=1.0, s_tint=(-0.0375, -0.0375, -0.0375), h_tint=(-0.05, -0.05, -0.05)),
    "Slide / Agfachrome 1000 RS": _stock(
        "Agfachrome 1000 RS", "Ultra high-speed slide. Massive grain, wild colors, cult favorite.",
        1.45, 1.3, 1.2, r_off=(0.05, 0, 0.05), b_off=(-0.05, 0, -0.05),
        sat=1.15, s_tint=(0.01, -0.005, -0.01), h_tint=(0.015, 0.005, -0.005)),

    # --- GAF ---
    "Slide / GAF 500": _stock(
        "GAF 500", "Rare high-speed slide from GAF. Warm, grainy, contrasty.",
        1.5, 1.35, 1.15, r_off=(0.03, 0, 0.03), b_off=(0, 0, -0.03),
        sat=1.05, s_tint=(0.01, 0.005, -0.008), h_tint=(0.01, 0.005, -0.005)),
    "Slide / GAF 500 - Warm": _stock(
        "GAF 500 Warm", "GAF 500 with extra warmth. Sunset tones.",
        1.5, 1.35, 1.15, r_off=(0.05, 0, 0.05), b_off=(0, -0.05, -0.05),
        sat=1.08, s_tint=(0.015, 0.008, -0.012), h_tint=(0.018, 0.008, -0.008)),
}

# =============================================================================
# CINEMA FILMS — Motion Picture Negative
# =============================================================================

COLOR_CINEMA = {
    # --- KODAK VISION3 ---
    "Cinema / Kodak Vision3 50D": _stock(
        "Kodak Vision3 50D", "Pristine daylight cinema. Clean, neutral, wide latitude.",
        1.4, 1.7, 0.95, r_off=(0, 0, 0), b_off=(0.05, -0.05, -0.02),
        sat=0.95, s_tint=(0.005, 0.0, -0.005), h_tint=(0.005, 0.003, 0.0)),
    "Cinema / Kodak Vision3 250D": _stock(
        "Kodak Vision3 250D", "Daylight cinema workhorse. Good all-rounder.",
        1.38, 1.65, 0.97, r_off=(0, 0, 0.01), b_off=(0.05, -0.05, -0.02),
        sat=0.95, s_tint=(0.005, 0.0, -0.005), h_tint=(0.005, 0.003, 0.0)),
    "Cinema / Kodak Vision3 200T": _stock(
        "Kodak Vision3 200T", "Tungsten cinema stock. Blue shadows under daylight.",
        1.35, 1.65, 0.97, r_off=(0, 0, 0), b_off=(-0.05, 0.05, 0.02),
        sat=0.93, s_tint=(0.0, -0.003, 0.01), h_tint=(0.008, 0.005, -0.005)),
    "Cinema / Kodak Vision3 500T": _stock(
        "Kodak Vision3 500T", "Fast cinema stock. Cyan shadows, gritty low-light character.",
        1.25, 1.55, 1.02, r_off=(0, 0, 0.02), b_off=(-0.05, 0.05, -0.02),
        sat=0.94, s_tint=(-0.005, 0.008, 0.015), h_tint=(0.01, 0.005, -0.008)),

    # --- KODAK VISION2 ---
    "Cinema / Kodak Vision2 500T": _stock(
        "Kodak Vision2 500T", "Previous generation cinema stock. Warmer, less latitude than V3.",
        1.3, 1.5, 1.0, r_off=(0.03, 0, 0.02), b_off=(-0.03, 0.05, -0.03),
        sat=0.92, s_tint=(-0.003, 0.005, 0.012), h_tint=(0.012, 0.005, -0.008)),

    # --- FUJI ETERNA ---
    "Cinema / Fuji Eterna 250D": _stock(
        "Fuji Eterna 250D", "Fuji daylight cinema. Slightly cooler than Kodak, clean.",
        1.4, 1.65, 0.96, r_off=(0, 0, -0.01), b_off=(-0.03, 0.03, 0.02),
        sat=0.93, s_tint=(-0.003, 0.003, 0.008), h_tint=(0.003, 0.005, 0.005)),
    "Cinema / Fuji Eterna 500T": _stock(
        "Fuji Eterna 500T", "Fuji tungsten cinema. Cooler than Kodak V3 500T.",
        1.3, 1.55, 0.99, r_off=(0, 0, -0.01), b_off=(-0.05, 0.05, 0.03),
        sat=0.92, s_tint=(-0.005, 0.003, 0.012), h_tint=(0.005, 0.005, 0.003)),
    "Cinema / Fuji Eterna Vivid 500": _stock(
        "Fuji Eterna Vivid 500", "Vivid variant of Eterna. More saturated, punchier.",
        1.35, 1.5, 1.02, r_off=(0, 0, 0.01), b_off=(-0.03, 0.03, 0.01),
        sat=1.05, s_tint=(-0.003, 0.003, 0.008), h_tint=(0.005, 0.005, 0.003)),
}

# =============================================================================
# FUJI CAMERA SIMULATIONS — Digital film modes
# =============================================================================

FUJI_CAMERA_SIM = {
    "FujiSim / Provia (Standard)": _stock(
        "Fuji Sim Provia", "Standard Fuji look. Accurate, slightly vivid, clean.",
        1.55, 1.5, 1.05, sat=1.08,
        s_tint=(0.0, 0.0, 0.003), h_tint=(0.005, 0.003, 0.003)),
    "FujiSim / Velvia (Vivid)": _stock(
        "Fuji Sim Velvia", "Vivid mode. High saturation, rich contrast.",
        1.7, 1.35, 1.18, sat=1.35,
        s_tint=(0.005, -0.003, 0.005), h_tint=(0.008, 0.005, 0.0)),
    "FujiSim / Astia (Soft)": _stock(
        "Fuji Sim Astia", "Soft mode. Lower contrast, gentle saturation. Portrait-friendly.",
        1.45, 1.6, 0.95, sat=1.0,
        s_tint=(0.003, 0.003, 0.005), h_tint=(0.005, 0.005, 0.003)),
    "FujiSim / Classic Chrome": _stock(
        "Fuji Sim Classic Chrome", "Desaturated, documentary feel. Muted shadows, subdued highlights.",
        1.5, 1.55, 1.02, sat=0.80,
        s_tint=(0.005, 0.003, -0.003), h_tint=(0.008, 0.005, -0.003)),
    "FujiSim / Classic Neg": _stock(
        "Fuji Sim Classic Neg", "Film negative simulation. High contrast, unique color rendering.",
        1.4, 1.45, 1.1, sat=0.90,
        s_tint=(0.012, 0.005, -0.008), h_tint=(0.01, 0.005, -0.005)),
    "FujiSim / PRO Neg Hi": _stock(
        "Fuji Sim PRO Neg Hi", "Professional negative, high contrast. Studio portrait look.",
        1.5, 1.5, 1.08, sat=0.88,
        s_tint=(0.008, 0.003, -0.005), h_tint=(0.008, 0.005, -0.003)),
    "FujiSim / PRO Neg Std": _stock(
        "Fuji Sim PRO Neg Std", "Professional negative, standard. Neutral, smooth skin tones.",
        1.45, 1.55, 1.0, sat=0.85,
        s_tint=(0.005, 0.003, -0.003), h_tint=(0.005, 0.005, -0.003)),
    "FujiSim / Eterna (Cinema)": _stock(
        "Fuji Sim Eterna", "Cinema mode. Low saturation, flat, wide dynamic range.",
        1.3, 1.7, 0.88, sat=0.75,
        s_tint=(0.0, 0.003, 0.005), h_tint=(0.003, 0.003, 0.003)),
    "FujiSim / Eterna Bleach Bypass": _stock(
        "Fuji Sim Bleach Bypass", "Bleach bypass look. Desaturated, high contrast, gritty.",
        1.6, 1.3, 1.25, sat=0.55,
        s_tint=(0.005, 0.0, -0.005), h_tint=(0.008, 0.003, -0.003)),
    "FujiSim / Nostalgic Neg": _stock(
        "Fuji Sim Nostalgic Neg", "Warm nostalgia. Amber highlights, soft contrast.",
        1.35, 1.6, 0.95, sat=0.85,
        s_tint=(0.015, 0.008, -0.01), h_tint=(0.02, 0.012, -0.005)),
}

# =============================================================================
# POLAROID / INSTANT FILMS
# =============================================================================

COLOR_INSTANT = {
    "Instant / Polaroid 600": _stock(
        "Polaroid 600", "Classic instant film. Low contrast, shifted colors, warm base.",
        1.2, 1.6, 0.88, r_off=(0.03, 0, 0.02), b_off=(0.05, -0.05, -0.05),
        sat=0.82, s_tint=(0.02, 0.01, -0.015), h_tint=(0.015, 0.01, -0.005)),
    "Instant / Polaroid SX-70": _stock(
        "Polaroid SX-70", "The original SX-70. Warm, faded, dreamy color palette.",
        1.15, 1.65, 0.85, r_off=(0.03, 0, 0.02), b_off=(0.08, -0.05, -0.06),
        sat=0.78, s_tint=(0.025, 0.012, -0.015), h_tint=(0.02, 0.012, -0.008)),
    "Instant / Polaroid Spectra": _stock(
        "Polaroid Spectra", "Spectra format. Slightly cooler than SX-70, wider tonal range.",
        1.2, 1.6, 0.90, r_off=(0.02, 0, 0.01), b_off=(0.05, -0.03, -0.03),
        sat=0.80, s_tint=(0.015, 0.008, -0.008), h_tint=(0.012, 0.008, -0.003)),
    "Instant / Polaroid 665": _stock(
        "Polaroid 665", "Peel-apart B&W positive/negative. Fine grain, beautiful tones. Legendary.",
        1.5, 1.5, 1.0, r_off=(0.01, 0.0, 0.005), b_off=(-0.01, 0.0, -0.005),
        sat=1.0, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Polaroid 669": _stock(
        "Polaroid 669", "Peel-apart color. Muted, creamy, soft. Transfer art staple.",
        1.69, 1.12, 1.15, r_off=(0.11, 0.161, -0.022), g_off=(-0.031, -0.119, 0.002), b_off=(-0.078, -0.043, 0.02),
        sat=0.85, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Polaroid 690": _stock(
        "Polaroid 690", "Professional peel-apart color. Sharper than 669, still dreamy.",
        1.9, 1.15, 1.21, r_off=(0.011, 0.041, -0.011), g_off=(0.011, 0.104, -0.036), b_off=(-0.021, -0.145, 0.046),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Polaroid Time-Zero": _stock(
        "Polaroid Time-Zero", "Expired SX-70 Time-Zero. Heavy color shifts, soft focus, ethereal.",
        1.5, 1.61, 0.77, r_off=(0.0, -0.605, -0.156), g_off=(0.0, -0.325, 0.002), b_off=(0.0, 0.93, 0.154),
        sat=0.85, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Impossible PX-680": _stock(
        "Impossible PX-680", "Impossible Project 600-type. Unpredictable colors, vintage instant.",
        1.3, 1.65, 0.89, r_off=(-0.047, -0.356, 0.045), g_off=(-0.078, -0.434, 0.076), b_off=(0.126, 0.789, -0.121),
        sat=0.8, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Impossible PX-70": _stock(
        "Impossible PX-70", "Impossible Project SX-70 type. Faded, dreamy, low saturation.",
        1.28, 1.67, 0.97, r_off=(-0.026, -0.549, 0.1), g_off=(-0.199, -0.673, 0.099), b_off=(0.225, 1.223, -0.199),
        sat=0.7, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
    "Instant / Fuji Instax": _stock(
        "Fuji Instax", "Fuji instant. Brighter, more saturated than Polaroid. Cooler tones.",
        1.25, 1.55, 0.95, r_off=(0, 0, 0), b_off=(0, 0.05, 0.02),
        sat=0.92, s_tint=(0.005, 0.008, 0.01), h_tint=(0.008, 0.008, 0.005)),
    "Instant / Fuji FP-100c": _stock(
        "Fuji FP-100c", "Peel-apart instant color. Sharp, vivid for instant. Cult classic.",
        1.71, 1.0, 1.24, r_off=(-0.021, 0.0, -0.009), g_off=(0.058, 0.0, 0.019), b_off=(-0.036, 0.0, -0.01),
        sat=0.95, s_tint=(0.0, 0.0, 0.0), h_tint=(0.0, 0.0, 0.0)),
}

# =============================================================================
# AGED / EXPIRED FILMS
# =============================================================================

COLOR_AGED = {
    "Aged / Expired Kodak Gold": _stock(
        "Expired Kodak Gold", "Expired Gold. Shifted colors, orange cast, foggy shadows.",
        1.15, 1.5, 0.88, r_off=(0.05, 0, 0.04), b_off=(0.1, -0.1, -0.08),
        sat=0.78, s_tint=(0.03, 0.015, -0.02), h_tint=(0.02, 0.012, -0.01)),
    "Aged / Expired Kodak Portra": _stock(
        "Expired Kodak Portra", "Expired Portra. Faded warmth, color shifts, lifted blacks.",
        1.1, 1.55, 0.85, r_off=(0.03, 0, 0.03), b_off=(0.08, -0.08, -0.06),
        sat=0.72, s_tint=(0.025, 0.012, -0.015), h_tint=(0.015, 0.01, -0.005)),
    "Aged / Expired Fuji": _stock(
        "Expired Fuji", "Expired Fuji consumer. Green color shift, low contrast, foggy.",
        1.12, 1.55, 0.85, g_off=(-0.05, 0, 0.03), b_off=(-0.05, 0.05, -0.03),
        sat=0.75, s_tint=(0.005, 0.02, 0.01), h_tint=(0.008, 0.015, 0.005)),
    "Aged / Expired Ektachrome": _stock(
        "Expired Ektachrome", "Expired slide film. Magenta shift, faded saturation, dream-like.",
        1.3, 1.5, 0.92, r_off=(0.05, 0, 0.03), b_off=(-0.03, 0.05, 0.03),
        sat=0.80, s_tint=(0.015, -0.005, 0.01), h_tint=(0.012, 0.0, 0.01)),
    "Aged / Faded Kodachrome": _stock(
        "Faded Kodachrome", "Aged Kodachrome. Lost saturation, shifted reds, warm fog.",
        1.4, 1.5, 0.95, r_off=(0.05, 0, 0.02), b_off=(-0.03, 0.05, -0.02),
        sat=0.85, s_tint=(0.015, 0.005, -0.005), h_tint=(0.012, 0.005, 0.003)),
    "Aged / 1970s Warm": _stock(
        "1970s Warm", "Generic 1970s warm film look. Amber highlights, faded shadows.",
        1.2, 1.55, 0.92, r_off=(0.05, 0, 0.04), b_off=(0.08, -0.08, -0.06),
        sat=0.82, s_tint=(0.025, 0.012, -0.018), h_tint=(0.025, 0.015, -0.008)),
    "Aged / 1980s Cool": _stock(
        "1980s Cool", "1980s film look. Cooler, slightly cyan shadows, muted.",
        1.25, 1.5, 0.95, r_off=(0, 0, -0.02), b_off=(-0.03, 0.05, 0.03),
        sat=0.85, s_tint=(-0.005, 0.008, 0.015), h_tint=(0.005, 0.005, 0.01)),
}


# =============================================================================
# MASTER REGISTRY
# =============================================================================

COLOR_STOCKS = {}
COLOR_STOCKS.update(COLOR_NEGATIVE_PRINT)
COLOR_STOCKS.update(COLOR_SLIDE)
COLOR_STOCKS.update(COLOR_CINEMA)
COLOR_STOCKS.update(FUJI_CAMERA_SIM)
COLOR_STOCKS.update(COLOR_INSTANT)
COLOR_STOCKS.update(COLOR_AGED)

COLOR_STOCK_NAMES = sorted(COLOR_STOCKS.keys())
