"""
Lens Profile node for ComfyUI-Darkroom.

Combines distortion correction, chromatic aberration correction, and
vignetting correction in one node -- simulating what Lightroom/Capture One
do with lens profiles.

Select a real lens to apply or correct all its optical characteristics at once.

GPU-accelerated via torch. No CPU roundtrip.
"""

import math
import torch
import torch.nn.functional as F

from ..data.lens_profiles import LENS_PROFILES_FLAT, LENS_PROFILE_NAMES
from ..utils.torch_ops import pixel_to_grid_coords, grid_sample_channel


PRESET_NAMES = LENS_PROFILE_NAMES  # No "Custom" -- use individual nodes for that


def _apply_lens_profile(img, k1, k2, ca_r, ca_b, vignette_strength, vignette_mid, mode):
    """
    Apply combined lens correction on GPU: distortion + CA + vignette.
    Distortion and CA are compounded into a single grid_sample per channel.
    img: (H, W, C) tensor on device.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    scale = min(h, w) / 1024.0
    device = img.device

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx
    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    r = torch.sqrt(r2)
    max_r = math.sqrt(cx * cx + cy * cy)

    result = img.clone()

    # Per-channel: each gets distortion + its own CA shift
    ca_shifts = [ca_r * scale, 0.0, ca_b * scale]

    for c in range(3):
        distort = 1.0 + k1 * r2 + k2 * r4

        ca = ca_shifts[c]
        if abs(ca) > 0.01:
            ca_factor = 1.0 + (ca / max_r) * r
            total_factor = distort * ca_factor
        else:
            total_factor = distort

        src_x = nx * total_factor * cx + cx
        src_y = ny * total_factor * cy + cy

        grid = pixel_to_grid_coords(src_y, src_x, h, w)
        result[..., c] = grid_sample_channel(img[..., c], grid,
                                             padding_mode='reflection')

    # Vignette
    if vignette_strength > 0.01:
        cos_theta = 1.0 / torch.sqrt(1.0 + r2)
        falloff = cos_theta ** 4
        transition = ((r - vignette_mid * 0.8) / 0.4).clamp(0.0, 1.0)

        if mode == "Add Aberrations":
            # Darken edges
            vig_factor = (1.0 - transition * (1.0 - falloff) * vignette_strength * 2).clamp(0.0, 1.0)
            result = result * vig_factor.unsqueeze(-1)
        else:
            # Brighten edges (correct vignette)
            correction = 1.0 + transition * (1.0 / falloff.clamp(0.3, 1.0) - 1.0) * vignette_strength
            result = result * correction.unsqueeze(-1)

    return result.clamp(0.0, 1.0)


def _prepare_masks(mask, batch, h, w, device, dtype):
    """
    Mask prep for the torch engine (docs/lens-mask-derivation.md §5.4/§3):
    clamp(0,1) -> pin dtype/device -> 2-D unsqueeze -> M==0 guard (treated as
    absent, console warn) -> per-frame min(i, M-1) -> resize via
    F.interpolate(bilinear, align_corners=True) + clamp.
    Returns a list of (H, W) tensors (one per batch frame), or None if the
    mask batch is empty.
    """
    m = mask.clamp(0.0, 1.0).to(device=device, dtype=dtype)
    if m.dim() == 2:
        m = m.unsqueeze(0)
    M = m.shape[0]
    if M == 0:
        print("[Darkroom] Lens Profile: mask batch is empty (M=0), treating as absent")
        return None

    masks = []
    for i in range(batch):
        mi = m[min(i, M - 1)]
        if mi.shape != (h, w):
            mi = F.interpolate(mi.unsqueeze(0).unsqueeze(0), size=(h, w),
                               mode='bilinear', align_corners=True)
            mi = mi.squeeze(0).squeeze(0).clamp(0.0, 1.0)
        masks.append(mi)
    return masks


def _masked_coords_lp(h, w, k1, k2, ca_r, ca_b, m, device):
    """
    Declared seam (docs/lens-mask-derivation.md §5.3), per channel, from the
    COMPOUNDED total_factor = distort * ca_factor (never the individual
    stages, §1). Returns {c: (src_y, src_x, src_y_m, src_x_m)} for c in
    {0, 1, 2}.
    """
    cy, cx = h / 2.0, w / 2.0
    scale = min(h, w) / 1024.0

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx
    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    r = torch.sqrt(r2)
    max_r = math.sqrt(cx * cx + cy * cy)

    ca_shifts = [ca_r * scale, 0.0, ca_b * scale]

    out = {}
    for c in range(3):
        distort = 1.0 + k1 * r2 + k2 * r4

        ca = ca_shifts[c]
        if abs(ca) > 0.01:
            ca_factor = 1.0 + (ca / max_r) * r
            total_factor = distort * ca_factor
        else:
            total_factor = distort

        src_x = nx * total_factor * cx + cx
        src_y = ny * total_factor * cy + cy

        d_y = src_y - yy
        d_x = src_x - xx
        src_y_m = yy + m * d_y
        src_x_m = xx + m * d_x

        out[c] = (src_y, src_x, src_y_m, src_x_m)

    return out


def _apply_lens_profile_masked(img, k1, k2, ca_r, ca_b, vignette_strength, vignette_mid, mode, m):
    """
    Masked Lens Profile (§2c): geometric distortion+CA masked via the
    universal D-form (per-channel seam _masked_coords_lp); the vignette is
    GATED by the same mask -- V_m = 1 + m*(V - 1), where V is the SHIPPED
    factor exactly as computed today (Add mode's already-clamped
    expression, Correct mode's unclamped correction) -- so one mask means
    one thing and m=0 is full identity. The final clamp(0,1) is unchanged.
    The §1c select is applied once, at the end, against the pristine input.
    img: (H, W, C) tensor on device. m: (H, W) tensor.
    """
    h, w = img.shape[:2]
    device = img.device
    cy, cx = h / 2.0, w / 2.0

    coords = _masked_coords_lp(h, w, k1, k2, ca_r, ca_b, m, device)
    result = img.clone()

    for c, (_, _, src_y_m, src_x_m) in coords.items():
        grid = pixel_to_grid_coords(src_y_m, src_x_m, h, w)
        result[..., c] = grid_sample_channel(img[..., c], grid,
                                             padding_mode='reflection')

    # Vignette, gated by the same mask (§2c). r2 recomputed at destination.
    if vignette_strength > 0.01:
        yy = torch.arange(h, dtype=torch.float32, device=device)
        xx = torch.arange(w, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        ny = (yy - cy) / cy
        nx = (xx - cx) / cx
        r2 = nx * nx + ny * ny
        r = torch.sqrt(r2)

        cos_theta = 1.0 / torch.sqrt(1.0 + r2)
        falloff = cos_theta ** 4
        transition = ((r - vignette_mid * 0.8) / 0.4).clamp(0.0, 1.0)

        if mode == "Add Aberrations":
            vig_factor = (1.0 - transition * (1.0 - falloff) * vignette_strength * 2).clamp(0.0, 1.0)
        else:
            vig_factor = 1.0 + transition * (1.0 / falloff.clamp(0.3, 1.0) - 1.0) * vignette_strength

        vig_factor_m = 1.0 + m * (vig_factor - 1.0)
        result = result * vig_factor_m.unsqueeze(-1)

    result = result.clamp(0.0, 1.0)
    return torch.where((m == 0.0).unsqueeze(-1), img, result)


class LensProfile:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lens": (PRESET_NAMES, {
                    "default": PRESET_NAMES[0],
                    "tooltip": "Select a lens to simulate or correct its optical characteristics"
                }),
                "mode": (["Add Aberrations", "Correct Aberrations"], {
                    "default": "Add Aberrations",
                    "tooltip": "'Add' simulates lens flaws. 'Correct' removes them (inverts the profile)"
                }),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Overall profile strength. 0.5 = half effect"
                }),
                "mask": ("MASK", {
                    "tooltip": "Where the lens character applies, 0 to 1, vignette included. Mask 0 "
                               "leaves pixels untouched. In Correct mode, corrections that clip to white "
                               "are unaffected by the mask where they clip."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, lens, mode, strength=1.0, mask=None):
        if strength < 0.01 or lens not in LENS_PROFILES_FLAT:
            return (image,)

        p = LENS_PROFILES_FLAT[lens]
        sign = 1.0 if mode == "Add Aberrations" else -1.0

        k1 = p.k1 * strength * sign
        k2 = p.k2 * strength * sign
        ca_r = p.ca_r * strength * sign
        ca_b = p.ca_b * strength * sign
        vig = p.vig_strength * strength
        vig_mid = p.vig_midpoint

        print(f"[Darkroom] Lens Profile: {p.name} ({mode}), strength={strength}")

        masks = None
        if mask is not None:
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            masks = _prepare_masks(mask, b, h, w, image.device, image.dtype)

        results = []
        for i in range(image.shape[0]):
            if masks is None:
                results.append(
                    _apply_lens_profile(image[i], k1, k2, ca_r, ca_b, vig, vig_mid, mode)
                )
            else:
                results.append(
                    _apply_lens_profile_masked(image[i], k1, k2, ca_r, ca_b, vig, vig_mid,
                                               mode, masks[i])
                )

        return (torch.stack(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DarkroomLensProfile": LensProfile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomLensProfile": "Lens Profile",
}
