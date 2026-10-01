"""
RAW Load node for ComfyUI-Darkroom.
Decodes camera RAW files into linear scene-referred IMAGE tensors via rawpy/LibRaw.
Outputs metadata as a RAW_METADATA custom type consumed by RAW Metadata Split.
"""

import os

import torch

from ..utils.dcp import list_brands_with_looks, list_all_look_variants
from ..utils.paths import resolve_allowed, PathNotAllowed
from ..utils.raw_loader import (
    decode_raw_linear,
    apply_post_processing,
    DEMOSAIC_OPTIONS,
    COLORSPACE_OPTIONS,
    HIGHLIGHT_OPTIONS,
    WHITE_BALANCE_OPTIONS,
)


OUTPUT_MODE_OPTIONS = ["sRGB display", "Linear scene"]


class DarkroomRAWLoad:

    # Two-stage cache. _decode_cache holds the expensive rawpy output keyed
    # only on parameters that affect the decode itself (11 s on a 51 MP GFX
    # 50S). _final_cache holds the post-processed image keyed on everything
    # including DCP/exposure/output_mode. Changing just camera_look or
    # baseline_exposure hits _decode_cache (cheap replay, ~2 s) instead of
    # re-running rawpy (~11 s).
    _decode_cache = {}
    _final_cache = {}
    _cache_max = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a camera RAW file (.raf .cr3 .nef .arw .dng ...). "
                               "Use the Browse button to pick one. Must be inside ComfyUI's "
                               "input/output/temp/models folders or a folder listed in "
                               "darkroom_allowed_folders.json. Relative paths start in input/."
                }),
                "demosaic": (DEMOSAIC_OPTIONS, {
                    "default": DEMOSAIC_OPTIONS[0],
                    "tooltip": "Demosaic algorithm. Auto uses DHT for Bayer sensors and "
                               "LibRaw's 3-pass Markesteijn for Fuji X-Trans."
                }),
                "output_colorspace": (COLORSPACE_OPTIONS, {
                    "default": "Linear sRGB",
                    "tooltip": "Output colorspace primaries. Linear sRGB is the standard choice."
                }),
                "white_balance": (WHITE_BALANCE_OPTIONS, {
                    "default": "As shot",
                    "tooltip": "As shot = use the camera's recorded WB (what you saw in the viewfinder). "
                               "Auto = gray-world estimate. Daylight = fixed ~5500K."
                }),
                "highlight_mode": (HIGHLIGHT_OPTIONS, {
                    "default": "Rebuild (default)",
                    "tooltip": "How to handle blown highlights. Rebuild reconstructs partial-channel "
                               "data (Adobe default). Clip discards. Blend is a middle ground."
                }),
                "half_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Half-resolution decode. Skips one demosaic step per axis. "
                               "~10x faster. Great for iteration, disable for final output."
                }),
                "output_mode": (OUTPUT_MODE_OPTIONS, {
                    "default": "sRGB display",
                    "tooltip": "sRGB display = gamma-encoded — matches Camera Raw's default open view "
                               "(flatter than JPEG, same brightness). Drop-in compatible with all "
                               "Darkroom grading nodes. "
                               "Linear scene = pure linear HDR for ACES pipelines."
                }),
                "baseline_exposure": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Exposure offset in stops, applied in linear light before any downstream "
                               "stage. In sRGB display mode: brightens or darkens from the ACR-matched "
                               "default view (default 0). In Linear scene mode: needed to lift the pure "
                               "scene-referred linear output into the range expected by downstream LUTs. "
                               "rawpy's linear decode lands typical midtones near 0.05; most Camera Raw-"
                               "calibrated LUTs (abpy's Fuji sims, Sowerby's) expect midtones near 0.18. "
                               "+2 EV is a good starting point for a daylight scene; push higher for "
                               "darker scenes."
                }),
                "camera_look": (["(not used)"] + list_all_look_variants(), {
                    "default": "(not used)",
                    "tooltip": "The specific look within the chosen brand (e.g. 'Standard', "
                               "'Landscape', 'Vivid', 'VELVIA'). Narrows automatically when "
                               "camera_brand changes. Shows '(not used)' when camera_brand is "
                               "'Adobe Standard' — the variant is ignored in that case."
                }),
                "camera_brand": (list_brands_with_looks(), {
                    "default": "Adobe Standard",
                    "tooltip": "Camera brand for the Camera Look profile. 'Adobe Standard' is the "
                               "neutral per-body calibration (hue/sat corrections only) and is the "
                               "default for all bodies. Pick a brand to unlock that brand's Camera "
                               "Looks in the camera_look dropdown below."
                               "\n\n"
                               "Drop .dcp packs into ComfyUI/models/camera_profiles/<Make Model>/ "
                               "(e.g. '.../camera_profiles/Fujifilm X-T5/Fujifilm X-T5 Camera VELVIA.dcp'). "
                               "Adobe ships Camera Looks for Canon/Nikon/Sony/Panasonic if Camera Raw "
                               "or Lightroom is installed on the same machine. For Fuji film sims, "
                               "install a third-party pack (e.g. Stuart Sowerby)."
                               "\n\n"
                               "If the selected brand + look isn't installed for the detected body, "
                               "the node falls back to Adobe Standard and logs a warning."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "RAW_METADATA")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/RAW"

    def execute(self, raw_file, demosaic, output_colorspace,
                white_balance, highlight_mode, half_size, output_mode,
                baseline_exposure=0.0, camera_look="", camera_brand="Adobe Standard"):
        raw_file = (raw_file or "").strip()
        if not raw_file:
            raise ValueError("[Darkroom RAW Load] no file path provided")
        try:
            raw_file = resolve_allowed(raw_file, "read").replace("\\", "/")
        except PathNotAllowed as e:
            raise ValueError(f"[Darkroom RAW Load] {e}") from None
        if not os.path.isfile(raw_file):
            raise FileNotFoundError(f"[Darkroom RAW Load] file not found: {raw_file}")

        if baseline_exposure is None:
            baseline_exposure = 0.0
        baseline_exposure = float(baseline_exposure)
        camera_brand = (camera_brand or "Adobe Standard").strip()
        camera_look = (camera_look or "").strip()

        # Combine brand + variant into the canonical "Brand / Look" string the
        # resolver expects. Legacy values (a bare "Brand / Look" already sitting
        # in camera_look from an older saved workflow) pass through untouched.
        if camera_brand == "Adobe Standard" or camera_look in ("", "(not used)"):
            look_spec = "Adobe Standard"
        elif " / " in camera_look:
            look_spec = camera_look
        else:
            look_spec = f"{camera_brand} / {camera_look}"

        mtime = os.path.getmtime(raw_file)
        decode_key = (raw_file, mtime, demosaic, output_colorspace,
                      white_balance, highlight_mode, bool(half_size))
        final_key = decode_key + (output_mode, round(baseline_exposure, 3), look_spec)

        cached_final = DarkroomRAWLoad._final_cache.get(final_key)
        if cached_final is not None:
            img, meta = cached_final
            print(f"[Darkroom RAW Load] final cache hit: {os.path.basename(raw_file)} "
                  f"(look={look_spec}, ev={baseline_exposure:+.1f})")
        else:
            cached_decode = DarkroomRAWLoad._decode_cache.get(decode_key)
            if cached_decode is not None:
                img_linear, meta_base = cached_decode
                print(f"[Darkroom RAW Load] decode cache hit "
                      f"({os.path.basename(raw_file)}); replaying post-processing "
                      f"(mode={output_mode}, ev={baseline_exposure:+.1f}, look={look_spec})")
            else:
                print(f"[Darkroom RAW Load] decoding {os.path.basename(raw_file)} "
                      f"(demosaic={demosaic}, wb={white_balance}, half={half_size})")
                img_linear, meta_base = decode_raw_linear(
                    raw_file,
                    demosaic=demosaic,
                    colorspace=output_colorspace,
                    white_balance=white_balance,
                    highlight_mode=highlight_mode,
                    half_size=half_size,
                )
                DarkroomRAWLoad._decode_cache[decode_key] = (img_linear, meta_base)
                _trim_cache(DarkroomRAWLoad._decode_cache, DarkroomRAWLoad._cache_max)

                sensor = meta_base.get("sensor_type", "?")
                model = meta_base.get("camera_model", "?")
                print(f"[Darkroom RAW Load] decoded: {model} "
                      f"{meta_base['image_width']}x{meta_base['image_height']} ({sensor})")
                if meta_base.get("film_simulation"):
                    print(f"[Darkroom RAW Load] film sim: {meta_base['film_simulation']}")

            # meta is mutated by apply_post_processing (dcp_profile / dcp_look),
            # so hand it a fresh copy each time — otherwise a later cache replay
            # with a different look would surface stale fields.
            meta = dict(meta_base)
            img, meta = apply_post_processing(
                img_linear, meta,
                output_mode=output_mode,
                baseline_exposure=baseline_exposure,
                camera_look=look_spec,
            )
            DarkroomRAWLoad._final_cache[final_key] = (img, meta)
            _trim_cache(DarkroomRAWLoad._final_cache, DarkroomRAWLoad._cache_max)

        tensor = torch.from_numpy(img).unsqueeze(0).contiguous()
        return (tensor, meta)


def _trim_cache(cache, max_size):
    while len(cache) > max_size:
        oldest = next(iter(cache))
        del cache[oldest]


NODE_CLASS_MAPPINGS = {"DarkroomRAWLoad": DarkroomRAWLoad}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomRAWLoad": "RAW Load"}
