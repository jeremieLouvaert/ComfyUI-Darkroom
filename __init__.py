from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import server_routes  # registers /darkroom/list_dir and /darkroom/mkdir

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']


def _warm_cuda():
    """
    First CUDA call in a fresh Python process can stall several minutes while
    the kernel loads. Touching the device at module import means the stall (if
    any) happens during ComfyUI startup, not inside someone's first RAW Load.
    Silent no-op when CUDA isn't available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
    except Exception:
        pass


_warm_cuda()

print("\033[34m[ComfyUI-Darkroom] \033[92mLoaded — 49 nodes (Film Emulation + Camera Raw + Color Grading + Lens & Optics + LUT Pipeline + RAW Pipeline w/ DCP + Spectral Film Stock + Scopes + Color Match + CMYK Print)\033[0m")
