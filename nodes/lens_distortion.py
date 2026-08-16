"""
Lens Distortion node for ComfyUI-Darkroom.

Simulates barrel distortion (wide-angle), pincushion distortion (telephoto),
and mustache/complex distortion. Uses the Brown-Conrady distortion model.

Select a real lens profile or use Custom for manual control.

GPU-accelerated via torch. No CPU roundtrip.
"""

import torch
import torch.nn.functional as F

from ..data.lens_profiles import LENS_PROFILES_FLAT, LENS_PROFILE_NAMES
from ..utils.torch_ops import pixel_to_grid_coords


PRESET_NAMES = ["Custom"] + LENS_PROFILE_NAMES


def _apply_distortion(img, k1, k2):
    """
    Apply Brown-Conrady radial distortion on GPU.
    r_distorted = r * (1 + k1*r^2 + k2*r^4)
    img: (H, W, C) tensor on device.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    device = img.device

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')

    ny = (yy - cy) / cy
    nx = (xx - cx) / cx

    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    distort = 1.0 + k1 * r2 + k2 * r4

    src_x = nx * distort * cx + cx
    src_y = ny * distort * cy + cy

    grid = pixel_to_grid_coords(src_y, src_x, h, w)

    # Same grid for all channels -- single grid_sample call
    inp = img.permute(2, 0, 1).unsqueeze(0)       # (1, C, H, W)
    g = grid.unsqueeze(0)                           # (1, H, W, 2)
    out = F.grid_sample(inp, g, mode='bicubic', padding_mode='zeros',
                        align_corners=True)
    return out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


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
        print("[Darkroom] Lens Distortion: mask batch is empty (M=0), treating as absent")
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


def _masked_coords(h, w, k1, k2, m, device):
    """
    Declared seam (docs/lens-mask-derivation.md §5.3). Unmasked and masked
    source coordinates for Lens Distortion, computed in the spec's exact op
    order from the UNMASKED coords: D = src - p, src_m = p + m*D.
    Returns (src_y, src_x, src_y_m, src_x_m), each (H, W).
    """
    cy, cx = h / 2.0, w / 2.0

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')

    ny = (yy - cy) / cy
    nx = (xx - cx) / cx

    r2 = nx * nx + ny * ny
    r4 = r2 * r2
    distort = 1.0 + k1 * r2 + k2 * r4

    src_x = nx * distort * cx + cx
    src_y = ny * distort * cy + cy

    d_y = src_y - yy
    d_x = src_x - xx
    src_y_m = yy + m * d_y
    src_x_m = xx + m * d_x

    return src_y, src_x, src_y_m, src_x_m


def _apply_distortion_masked(img, k1, k2, m):
    """
    Masked Lens Distortion (§1): the source lookup moves m(p) of the way
    from the pixel's own location to where the unmasked warp would have
    looked. The §1c select is applied once, at the end, against the
    pristine input.
    img: (H, W, C) tensor on device. m: (H, W) tensor.
    """
    h, w = img.shape[:2]
    device = img.device

    _, _, src_y_m, src_x_m = _masked_coords(h, w, k1, k2, m, device)

    grid = pixel_to_grid_coords(src_y_m, src_x_m, h, w)

    inp = img.permute(2, 0, 1).unsqueeze(0)       # (1, C, H, W)
    g = grid.unsqueeze(0)                           # (1, H, W, 2)
    out = F.grid_sample(inp, g, mode='bicubic', padding_mode='zeros',
                        align_corners=True)
    result = out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)

    return torch.where((m == 0.0).unsqueeze(-1), img, result)


class LensDistortion:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lens": (PRESET_NAMES, {
                    "default": "Custom",
                    "tooltip": "Select a real lens or 'Custom' for manual distortion control"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Multiplier. Negative inverts (correct distortion instead of adding it)"
                }),
            },
            "optional": {
                "k1": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Primary radial coefficient. Negative=barrel, Positive=pincushion"
                }),
                "k2": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Secondary radial coefficient. Creates mustache distortion when opposite sign to k1"
                }),
                "mask": ("MASK", {
                    "tooltip": "Where the distortion applies, 0 to 1. Scales the warp itself, so partial "
                               "values do not ghost. Feather the mask edge by about twice the local "
                               "displacement (tens to hundreds of px at strong settings) or a smear band "
                               "forms along the edge. Strong distortion near the frame border can pull "
                               "black in from outside the frame, masked or not."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, lens, strength, k1=0.0, k2=0.0, mask=None):
        if lens != "Custom" and lens in LENS_PROFILES_FLAT:
            p = LENS_PROFILES_FLAT[lens]
            k1 = p.k1
            k2 = p.k2

        k1 *= strength
        k2 *= strength

        if abs(k1) < 0.001 and abs(k2) < 0.001:
            return (image,)

        print(f"[Darkroom] Lens Distortion: {lens}, k1={k1:.4f}, k2={k2:.4f}")

        masks = None
        if mask is not None:
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            masks = _prepare_masks(mask, b, h, w, image.device, image.dtype)

        results = []
        for i in range(image.shape[0]):
            if masks is None:
                results.append(_apply_distortion(image[i], k1, k2))
            else:
                results.append(_apply_distortion_masked(image[i], k1, k2, masks[i]))

        return (torch.stack(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DarkroomLensDistortion": LensDistortion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomLensDistortion": "Lens Distortion",
}
