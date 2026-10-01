"""
Full DCP (DNG Color Profile) parser and applier for ComfyUI-Darkroom.

Reads Adobe per-camera .dcp files and runs the HueSatMap → LookTable →
ProfileToneCurve pipeline on a linear-light RGB image. Used by the RAW
Load node to match Camera Raw's default open view for every supported
camera body, replacing the camera-agnostic stopgap curve from session 1.

Pure Python + numpy. No DNG SDK, no OCIO, no OpenImageIO.
"""

import os
import re
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from .grading import cubic_spline_curve


# --- DNG tag numbers ------------------------------------------------------

TAG_UNIQUE_CAMERA_MODEL      = 0xC614  # 50708  ASCII
TAG_COLOR_MATRIX_1           = 0xC621  # 50721  SRATIONAL x9
TAG_COLOR_MATRIX_2           = 0xC622  # 50722  SRATIONAL x9
TAG_CAMERA_CALIBRATION_1     = 0xC623  # 50723  SRATIONAL x9
TAG_CAMERA_CALIBRATION_2     = 0xC624  # 50724  SRATIONAL x9
TAG_CALIBRATION_ILLUMINANT_1 = 0xC65A  # 50778  SHORT
TAG_CALIBRATION_ILLUMINANT_2 = 0xC65B  # 50779  SHORT
TAG_PROFILE_CALIBRATION_SIG  = 0xC6F4  # 50932  ASCII
TAG_PROFILE_NAME             = 0xC6F8  # 50936  ASCII
TAG_PROFILE_HSM_DIMS         = 0xC6F9  # 50937  LONG x3  (H,S,V)
TAG_PROFILE_HSM_DATA_1       = 0xC6FA  # 50938  FLOAT
TAG_PROFILE_HSM_DATA_2       = 0xC6FB  # 50939  FLOAT
TAG_PROFILE_TONE_CURVE       = 0xC6FC  # 50940  FLOAT pairs
TAG_PROFILE_EMBED_POLICY     = 0xC6FD  # 50941  LONG
TAG_PROFILE_COPYRIGHT        = 0xC6FE  # 50942  ASCII
TAG_FORWARD_MATRIX_1         = 0xC714  # 50964  SRATIONAL x9
TAG_FORWARD_MATRIX_2         = 0xC715  # 50965  SRATIONAL x9
TAG_PROFILE_LUT_DIMS         = 0xC725  # 50981  LONG x3
TAG_PROFILE_LUT_DATA         = 0xC726  # 50982  FLOAT
TAG_PROFILE_HSM_ENCODING     = 0xC761  # 51041  LONG (0=linear, 1=sRGB)
TAG_PROFILE_LUT_ENCODING     = 0xC762  # 51042  LONG
TAG_BASELINE_EXPOSURE_OFFSET = 0xC7A5  # 51109  SRATIONAL
TAG_DEFAULT_BLACK_RENDER     = 0xC7A6  # 51110  LONG (0=auto, 1=none)

TYPE_BYTE, TYPE_ASCII, TYPE_SHORT, TYPE_LONG       = 1, 2, 3, 4
TYPE_RATIONAL, TYPE_SRATIONAL                      = 5, 10
TYPE_FLOAT, TYPE_DOUBLE                            = 11, 12

_TYPE_SIZES = {
    TYPE_BYTE: 1, TYPE_ASCII: 1, TYPE_SHORT: 2, TYPE_LONG: 4,
    TYPE_RATIONAL: 8, TYPE_SRATIONAL: 8, TYPE_FLOAT: 4, TYPE_DOUBLE: 8,
}

ADOBE_STANDARD_DIR = r"C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Adobe Standard"

# Camera Look DCP search roots. Each root contains per-body subfolders
# (e.g. "Canon EOS 5D Mark IV/") with one .dcp per look. Priority order:
#
#   1. ComfyUI/models/camera_profiles/   (ComfyUI-native, registered via
#      folder_paths so it respects extra_model_paths.yaml — users drop
#      packs here, e.g. Stuart Sowerby's Fuji sims)
#   2. ComfyUI-Darkroom/data/dcp_looks/   (bundled, for profiles we may
#      curate and ship with the pack later)
#   3. Adobe install paths                (opportunistic — picked up only
#      if Camera Raw / Lightroom is installed on the same machine, so
#      Canon/Nikon/Sony/Panasonic users get Camera Looks for free)
_PKG_DCP_LOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dcp_looks"
)
_ADOBE_LOOKS_DIRS = [
    r"C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Camera",
    r"C:\Program Files\Adobe\Adobe Lightroom Classic\Resources\CameraProfiles\Camera",
]

try:
    import folder_paths as _comfy_folder_paths
    _CAMERA_PROFILES_DEFAULT = os.path.join(
        _comfy_folder_paths.models_dir, "camera_profiles"
    )
    _comfy_folder_paths.add_model_folder_path(
        "camera_profiles", _CAMERA_PROFILES_DEFAULT, is_default=True
    )
except Exception:
    _comfy_folder_paths = None
    _CAMERA_PROFILES_DEFAULT = None


def _camera_looks_dirs():
    dirs = []
    if _comfy_folder_paths is not None:
        try:
            dirs.extend(_comfy_folder_paths.get_folder_paths("camera_profiles"))
        except Exception:
            pass
    dirs.append(_PKG_DCP_LOOKS_DIR)
    dirs.extend(_ADOBE_LOOKS_DIRS)
    return dirs


CAMERA_LOOKS_DIRS = _camera_looks_dirs()


# --- Profile dataclass ----------------------------------------------------

@dataclass
class DCPProfile:
    path: str = ""
    unique_camera_model: str = ""
    profile_name: str = ""
    calibration_illuminant_1: int = 0
    calibration_illuminant_2: int = 0
    color_matrix_1: Optional[np.ndarray] = None
    color_matrix_2: Optional[np.ndarray] = None
    forward_matrix_1: Optional[np.ndarray] = None
    forward_matrix_2: Optional[np.ndarray] = None
    hsm_dims: Optional[Tuple[int, int, int]] = None
    hsm_data_1: Optional[np.ndarray] = None           # (H,S,V,3)
    hsm_data_2: Optional[np.ndarray] = None
    hsm_encoding: int = 0
    lut_dims: Optional[Tuple[int, int, int]] = None
    lut_data: Optional[np.ndarray] = None             # (H,S,V,3)
    lut_encoding: int = 0
    tone_curve: Optional[np.ndarray] = None           # (N,2)
    baseline_exposure_offset: Optional[float] = None

    @property
    def has_hsm(self) -> bool:
        return self.hsm_data_1 is not None

    @property
    def has_lut(self) -> bool:
        return self.lut_data is not None

    @property
    def has_tone_curve(self) -> bool:
        return self.tone_curve is not None and len(self.tone_curve) >= 2

    def summary(self) -> str:
        parts = []
        if self.has_hsm:
            H, S, V = self.hsm_dims
            n_maps = 2 if self.hsm_data_2 is not None else 1
            parts.append(f"HSM {H}x{S}x{V} ({n_maps} illum)")
        if self.has_lut:
            H, S, V = self.lut_dims
            parts.append(f"LUT {H}x{S}x{V}")
        if self.has_tone_curve:
            parts.append(f"Curve {len(self.tone_curve)}pt")
        return ", ".join(parts) if parts else "identity"


# --- Low-level TIFF/IFD parsing -------------------------------------------

def _read_entries(data, ifd_offset, e):
    n = struct.unpack_from(f"{e}H", data, ifd_offset)[0]
    out = []
    start = ifd_offset + 2
    for i in range(n):
        off = start + i * 12
        tag, dtype, count = struct.unpack_from(f"{e}HHI", data, off)
        type_size = _TYPE_SIZES.get(dtype, 0)
        total = type_size * count
        if total <= 4:
            vb = data[off + 8 : off + 8 + max(total, 1)]
        else:
            val_off = struct.unpack_from(f"{e}I", data, off + 8)[0]
            vb = b"" if val_off + total > len(data) else data[val_off : val_off + total]
        out.append((tag, dtype, count, vb))
    next_ifd = struct.unpack_from(f"{e}I", data, start + n * 12)[0]
    return out, next_ifd


def _parse_ascii(vb):
    return vb.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def _parse_short_array(vb, e, count):
    if len(vb) < count * 2:
        return []
    return list(struct.unpack_from(f"{e}{count}H", vb, 0))


def _parse_long_array(vb, e, count):
    if len(vb) < count * 4:
        return []
    return list(struct.unpack_from(f"{e}{count}I", vb, 0))


def _parse_float_array(vb, e, count):
    if len(vb) < count * 4:
        return np.zeros(0, dtype=np.float32)
    dtype = "<f4" if e == "<" else ">f4"
    return np.frombuffer(vb[: count * 4], dtype=dtype).astype(np.float32).copy()


def _parse_srational_matrix(vb, e, count):
    if len(vb) < count * 8:
        return None
    vals = []
    for i in range(count):
        num, den = struct.unpack_from(f"{e}ii", vb, i * 8)
        vals.append(0.0 if den == 0 else num / den)
    if count == 9:
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    return np.array(vals, dtype=np.float64)


# --- Full DCP reader ------------------------------------------------------

def read_dcp_profile(path):
    """
    Parse a .dcp file into a DCPProfile. Returns None on I/O error or
    malformed TIFF. Missing optional tags are silently skipped.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 8:
        return None

    bo = data[:2]
    if bo == b"II":
        e = "<"
    elif bo == b"MM":
        e = ">"
    else:
        return None
    # Standard TIFF uses magic 42. Adobe .dcp files use 0x4352 ("CR"
    # = Camera Raw profile) — a DNG-specific variant. Accept both.
    magic = struct.unpack_from(f"{e}H", data, 2)[0]
    if magic not in (42, 0x4352):
        return None
    first_ifd = struct.unpack_from(f"{e}I", data, 4)[0]
    if first_ifd == 0 or first_ifd >= len(data):
        return None

    p = DCPProfile(path=path)

    ifd_offset = first_ifd
    visited = set()
    while ifd_offset and ifd_offset not in visited and ifd_offset < len(data):
        visited.add(ifd_offset)
        entries, next_ifd = _read_entries(data, ifd_offset, e)
        for tag, dtype, count, vb in entries:
            try:
                if tag == TAG_UNIQUE_CAMERA_MODEL:
                    p.unique_camera_model = _parse_ascii(vb)
                elif tag == TAG_PROFILE_NAME:
                    p.profile_name = _parse_ascii(vb)
                elif tag == TAG_COLOR_MATRIX_1:
                    p.color_matrix_1 = _parse_srational_matrix(vb, e, count)
                elif tag == TAG_COLOR_MATRIX_2:
                    p.color_matrix_2 = _parse_srational_matrix(vb, e, count)
                elif tag == TAG_FORWARD_MATRIX_1:
                    p.forward_matrix_1 = _parse_srational_matrix(vb, e, count)
                elif tag == TAG_FORWARD_MATRIX_2:
                    p.forward_matrix_2 = _parse_srational_matrix(vb, e, count)
                elif tag == TAG_CALIBRATION_ILLUMINANT_1:
                    arr = _parse_short_array(vb, e, count)
                    if arr:
                        p.calibration_illuminant_1 = int(arr[0])
                elif tag == TAG_CALIBRATION_ILLUMINANT_2:
                    arr = _parse_short_array(vb, e, count)
                    if arr:
                        p.calibration_illuminant_2 = int(arr[0])
                elif tag == TAG_PROFILE_HSM_DIMS:
                    dims = _parse_long_array(vb, e, count)
                    if len(dims) >= 3:
                        p.hsm_dims = (int(dims[0]), int(dims[1]), int(dims[2]))
                elif tag == TAG_PROFILE_HSM_DATA_1:
                    p.hsm_data_1 = _parse_float_array(vb, e, count)
                elif tag == TAG_PROFILE_HSM_DATA_2:
                    p.hsm_data_2 = _parse_float_array(vb, e, count)
                elif tag == TAG_PROFILE_HSM_ENCODING:
                    arr = _parse_long_array(vb, e, count)
                    if arr:
                        p.hsm_encoding = int(arr[0])
                elif tag == TAG_PROFILE_LUT_DIMS:
                    dims = _parse_long_array(vb, e, count)
                    if len(dims) >= 3:
                        p.lut_dims = (int(dims[0]), int(dims[1]), int(dims[2]))
                elif tag == TAG_PROFILE_LUT_DATA:
                    p.lut_data = _parse_float_array(vb, e, count)
                elif tag == TAG_PROFILE_LUT_ENCODING:
                    arr = _parse_long_array(vb, e, count)
                    if arr:
                        p.lut_encoding = int(arr[0])
                elif tag == TAG_PROFILE_TONE_CURVE:
                    arr = _parse_float_array(vb, e, count)
                    if arr.size >= 4 and arr.size % 2 == 0:
                        p.tone_curve = arr.reshape(-1, 2)
                elif tag == TAG_BASELINE_EXPOSURE_OFFSET:
                    m = _parse_srational_matrix(vb, e, count)
                    if m is not None and m.size >= 1:
                        p.baseline_exposure_offset = float(m.flatten()[0])
            except Exception as ex:
                print(f"[Darkroom DCP] tag 0x{tag:04X} parse error: {ex}")
        ifd_offset = next_ifd

    # Reshape raw float arrays now that dims are known. A size mismatch
    # invalidates the table so we never index into it with bad shapes.
    if p.hsm_dims and p.hsm_data_1 is not None:
        H, S, V = p.hsm_dims
        expected = H * S * V * 3
        if p.hsm_data_1.size == expected:
            p.hsm_data_1 = p.hsm_data_1.reshape(H, S, V, 3)
        else:
            p.hsm_data_1 = None
        if p.hsm_data_2 is not None:
            if p.hsm_data_2.size == expected:
                p.hsm_data_2 = p.hsm_data_2.reshape(H, S, V, 3)
            else:
                p.hsm_data_2 = None

    if p.lut_dims and p.lut_data is not None:
        H, S, V = p.lut_dims
        expected = H * S * V * 3
        if p.lut_data.size == expected:
            p.lut_data = p.lut_data.reshape(H, S, V, 3)
        else:
            p.lut_data = None

    return p


# --- DCP writer -----------------------------------------------------------

def _to_srational_pair(v):
    """Float -> (numerator, denominator) at 1e6 precision, signed."""
    denom = 1_000_000
    num = int(round(float(v) * denom))
    if num > 0x7FFFFFFF:
        num = 0x7FFFFFFF
    if num < -0x80000000:
        num = -0x80000000
    return num, denom


def write_dcp_profile(path: str, p: 'DCPProfile') -> None:
    """
    Serialize a DCPProfile to a .dcp file. Writes little-endian, magic 0x4352.
    Emits only the tags present on the profile (skips matrices/look tables/curves
    that are None). Sufficient for camera-look profiles built from Adobe Standard
    + abpy LookTable + ToneCurve splices.
    """
    e = "<"  # little-endian
    entries = []  # (tag, dtype, count, payload_bytes)

    def add_ascii(tag, s):
        if not s:
            return
        b = s.encode("ascii", errors="replace") + b"\x00"
        entries.append((tag, TYPE_ASCII, len(b), b))

    def add_long(tag, v):
        entries.append((tag, TYPE_LONG, 1, struct.pack(f"{e}I", int(v) & 0xFFFFFFFF)))

    def add_long_array(tag, vs):
        entries.append((tag, TYPE_LONG, len(vs),
                        b"".join(struct.pack(f"{e}I", int(v) & 0xFFFFFFFF) for v in vs)))

    def add_short(tag, v):
        entries.append((tag, TYPE_SHORT, 1, struct.pack(f"{e}H", int(v) & 0xFFFF)))

    def add_srational_matrix(tag, mat):
        if mat is None:
            return
        flat = np.asarray(mat, dtype=np.float64).reshape(-1)
        buf = b"".join(struct.pack(f"{e}ii", *_to_srational_pair(v)) for v in flat)
        entries.append((tag, TYPE_SRATIONAL, len(flat), buf))

    def add_floats(tag, arr):
        if arr is None:
            return
        flat = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1))
        entries.append((tag, TYPE_FLOAT, flat.size, flat.tobytes()))

    add_ascii(TAG_UNIQUE_CAMERA_MODEL, p.unique_camera_model)
    add_srational_matrix(TAG_COLOR_MATRIX_1, p.color_matrix_1)
    add_srational_matrix(TAG_COLOR_MATRIX_2, p.color_matrix_2)
    if p.calibration_illuminant_1:
        add_short(TAG_CALIBRATION_ILLUMINANT_1, p.calibration_illuminant_1)
    if p.calibration_illuminant_2:
        add_short(TAG_CALIBRATION_ILLUMINANT_2, p.calibration_illuminant_2)
    add_ascii(TAG_PROFILE_NAME, p.profile_name)
    if p.hsm_dims and p.hsm_data_1 is not None:
        add_long_array(TAG_PROFILE_HSM_DIMS, list(p.hsm_dims))
        add_floats(TAG_PROFILE_HSM_DATA_1, p.hsm_data_1)
        if p.hsm_data_2 is not None:
            add_floats(TAG_PROFILE_HSM_DATA_2, p.hsm_data_2)
    if p.tone_curve is not None and len(p.tone_curve) >= 2:
        add_floats(TAG_PROFILE_TONE_CURVE, p.tone_curve)
    add_long(TAG_PROFILE_EMBED_POLICY, 0)  # 0 = allow copying
    add_srational_matrix(TAG_FORWARD_MATRIX_1, p.forward_matrix_1)
    add_srational_matrix(TAG_FORWARD_MATRIX_2, p.forward_matrix_2)
    if p.lut_dims and p.lut_data is not None:
        add_long_array(TAG_PROFILE_LUT_DIMS, list(p.lut_dims))
        add_floats(TAG_PROFILE_LUT_DATA, p.lut_data)
    if p.hsm_data_1 is not None:
        add_long(TAG_PROFILE_HSM_ENCODING, int(p.hsm_encoding))
    if p.lut_data is not None:
        add_long(TAG_PROFILE_LUT_ENCODING, int(p.lut_encoding))

    # IFD entries must be sorted by tag.
    entries.sort(key=lambda x: x[0])

    n = len(entries)
    ifd_size = 2 + n * 12 + 4   # count + entries + next-IFD offset
    header_size = 8
    blob_offset = header_size + ifd_size

    blob = bytearray()
    out_entries = []
    for tag, dtype, count, payload in entries:
        if len(payload) <= 4:
            value_field = payload + b"\x00" * (4 - len(payload))
            out_entries.append((tag, dtype, count, value_field))
        else:
            offset = blob_offset + len(blob)
            blob.extend(payload)
            # Each value blob is padded to even length per TIFF convention.
            if len(blob) & 1:
                blob.append(0)
            out_entries.append((tag, dtype, count, struct.pack(f"{e}I", offset)))

    out = bytearray()
    out.extend(struct.pack(f"{e}2sHI", b"II", 0x4352, header_size))   # header
    out.extend(struct.pack(f"{e}H", n))                                # IFD count
    for tag, dtype, count, value_field in out_entries:
        out.extend(struct.pack(f"{e}HHI", tag, dtype, count))
        out.extend(value_field)
    out.extend(struct.pack(f"{e}I", 0))                                # next IFD = 0
    out.extend(blob)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)


# --- abpy XML fragment parser --------------------------------------------

def parse_abpy_xml_table(text: str):
    """
    Parse one of abpy/FujifilmCameraProfiles' xml table fragments. Returns
    (lut_data, lut_dims, tone_curve) where lut_data has shape (H, S, V, 3) of
    (HueShift, SatScale, ValScale) values and tone_curve is shape (N, 2) of
    (h, v) pairs.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(f"<root>{text}</root>")
    lut_node = root.find("LookTable")
    tc_node = root.find("ToneCurve")
    if lut_node is None or tc_node is None:
        raise ValueError("XML fragment missing <LookTable> or <ToneCurve>")

    H = int(lut_node.attrib["hueDivisions"])
    S = int(lut_node.attrib["satDivisions"])
    V = int(lut_node.attrib["valDivisions"])
    lut = np.zeros((H, S, V, 3), dtype=np.float32)
    for el in lut_node.findall("Element"):
        h = int(el.attrib["HueDiv"])
        s = int(el.attrib["SatDiv"])
        v = int(el.attrib["ValDiv"])
        lut[h, s, v, 0] = float(el.attrib["HueShift"])
        lut[h, s, v, 1] = float(el.attrib["SatScale"])
        lut[h, s, v, 2] = float(el.attrib["ValScale"])

    pts = []
    for el in tc_node.findall("Element"):
        pts.append((float(el.attrib["h"]), float(el.attrib["v"])))
    tone_curve = np.asarray(pts, dtype=np.float32)
    return lut, (H, S, V), tone_curve


def build_bw_look_table(channel_weights, dims=(72, 12, 16)):
    """
    Synthesize a ProfileLookTable that converts color input to grayscale
    using per-channel luma weights, while leaving neutral (S=0) inputs
    unchanged.

    channel_weights: (R_wt, G_wt, B_wt). Need not sum to 1; normalized here.
    dims: (hueDivisions, satDivisions, valDivisions). Default matches abpy.

    Mechanism:
      sat_scale = 0 at every bin (forces output to neutral gray).
      val_scale varies linearly with sat: 1.0 at S=0, luma-of-hue at S=1.
      luma-of-hue = Rn*r + Gn*g + Bn*b where (r,g,b) is the pure hue at
      (S=1, V=1). For a pure-hue input at arbitrary sat s and value V,
      the luma-weighted target is V * [1 - s * (1 - luma)]. Linear
      interpolation between val_scale=1 at s=0 and val_scale=luma at s=1
      reproduces exactly this factor. Desaturated inputs interpolate
      smoothly toward neutral gray. V axis is left flat because the
      channel-mix weighting is brightness-independent.

    Returns (lut_data, dims) with lut_data shape (H, S, V, 3) of
    (HueShift, SatScale, ValScale) -- the same shape write_dcp_profile
    expects for lut_data.
    """
    R, G, B = channel_weights
    total = float(R + G + B)
    if total <= 0:
        raise ValueError("channel_weights must sum to positive")
    Rn, Gn, Bn = R / total, G / total, B / total

    H, S, V = dims
    lut = np.zeros((H, S, V, 3), dtype=np.float32)

    for h_idx in range(H):
        hue_deg = (h_idx / H) * 360.0
        pure_hsv = np.array([[[hue_deg, 1.0, 1.0]]], dtype=np.float32)
        pure_rgb = hsv_to_rgb(pure_hsv)[0, 0]
        r, g, b = float(pure_rgb[0]), float(pure_rgb[1]), float(pure_rgb[2])
        luma = Rn * r + Gn * g + Bn * b

        for s_idx in range(S):
            s_frac = s_idx / (S - 1) if S > 1 else 0.0
            val_scale = (1.0 - s_frac) * 1.0 + s_frac * luma
            for v_idx in range(V):
                lut[h_idx, s_idx, v_idx] = (0.0, 0.0, float(val_scale))

    return lut, dims


def build_tone_curve_from_points(points, n=128):
    """
    PchipInterpolate a list of (h, v) control points into a 128-point
    ProfileToneCurve. Endpoints should be (0, 0) and (1, 1).
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        raise RuntimeError("scipy required for tone curve interpolation")
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[pts[:, 0].argsort()]
    pchip = PchipInterpolator(pts[:, 0], pts[:, 1])
    h = np.linspace(0.0, 1.0, n)
    v = np.clip(pchip(h), 0.0, 1.0)
    return np.stack([h.astype(np.float32), v.astype(np.float32)], axis=-1)


# --- Profile lookup -------------------------------------------------------

_profile_cache: dict = {}
_look_names_cache: list = None


def _body_folder(make, model):
    """Folder name Adobe uses for a body in Camera/: '<Make> <Model>' trimmed.

    make/model come from the RAW file's EXIF, so they are untrusted: path
    separators, colons and '..' are stripped so a crafted file cannot steer the
    profile lookup outside its root."""
    parts = [p for p in [(make or "").strip(), (model or "").strip()] if p]
    name = re.sub(r'[\\/:\x00]', " ", " ".join(parts)).replace("..", " ")
    return " ".join(name.split())


# Longest-prefix-first so multi-word brands ("OM Digital Solutions", "Phase One")
# match before their shorter single-word collisions ("OM", "Phase"). Also
# collapses brand casing variants (FUJIFILM/Fujifilm, SIGMA/Sigma) to one form.
_BRAND_PREFIXES = [
    ("OM Digital Solutions", "OM"),
    ("Phase One",            "Phase One"),
    ("FUJIFILM",             "Fujifilm"),
    ("Fujifilm",             "Fujifilm"),
    ("NIKON CORPORATION",    "Nikon"),
    ("NIKON",                "Nikon"),
    ("Nikon",                "Nikon"),
    ("SIGMA",                "Sigma"),
    ("Sigma",                "Sigma"),
    ("Canon",                "Canon"),
    ("Sony",                 "Sony"),
    ("SONY",                 "Sony"),
    ("Panasonic",            "Panasonic"),
    ("Olympus",              "Olympus"),
    ("OLYMPUS",              "Olympus"),
    ("Leica",                "Leica"),
    ("LEICA",                "Leica"),
    ("Pentax",               "Pentax"),
    ("PENTAX",               "Pentax"),
    ("Hasselblad",           "Hasselblad"),
    ("HASSELBLAD",           "Hasselblad"),
    ("Ricoh",                "Ricoh"),
    ("RICOH",                "Ricoh"),
    ("Apple",                "Apple"),
    ("Google",               "Google"),
    ("Samsung",              "Samsung"),
]


def _canonical_brand(body_or_make):
    """'Fujifilm GFX 50S' / 'FUJIFILM' / 'OM Digital Solutions OM-1' → canonical brand."""
    if not body_or_make:
        return None
    s = body_or_make.strip()
    for prefix, canonical in _BRAND_PREFIXES:
        if s == prefix or s.lower().startswith(prefix.lower() + " "):
            return canonical
    return s.split(" ", 1)[0] if s else None


def _prettify_look(look):
    """
    Normalize an Adobe Camera Look filename fragment for display.

    - Drops the leading 'Camera ' prefix (every Adobe Camera Look carries it,
      so it adds no information to the dropdown).
    - Collapses underscores to spaces (some vendor DCPs use both conventions
      for the same look, e.g. 'Camera_Standard_v2' vs 'Camera Standard v2').
    - Squeezes runs of whitespace.
    """
    s = look.replace("_", " ").strip()
    s = " ".join(s.split())
    if s.lower().startswith("camera "):
        s = s[len("camera "):].strip()
    return s


def _look_dedup_key(look):
    """Case/underscore-insensitive key for collapsing near-duplicates per brand."""
    return _prettify_look(look).lower()


# Display separator for "Brand / Look" entries. ASCII only (no em dash —
# Jeremie flagged em dashes as an AI-authorship tell in human-facing text).
_LOOK_SEP = " / "


def list_all_look_names():
    """
    Legacy flat 'Brand / Look' list. Retained for scripts / tests that predate
    the split-dropdown design. The live RAW Load node uses list_brands_with_looks
    + list_looks_for_brand instead.
    """
    global _look_names_cache
    if _look_names_cache is not None:
        return _look_names_cache
    by_brand = _cached_scan()
    entries = []
    for brand in sorted(by_brand):
        for key in sorted(by_brand[brand]):
            entries.append(f"{brand}{_LOOK_SEP}{by_brand[brand][key]}")
    out = ["Adobe Standard"] + entries
    _look_names_cache = out
    return out


def _scan_looks_by_brand():
    """
    One pass across every root × body × DCP file. Returns {brand: {dedup_key: pretty_look}}.
    Cached for the process lifetime via _look_names_cache indirection.
    """
    by_brand: dict = {}
    for root in CAMERA_LOOKS_DIRS:
        if not os.path.isdir(root):
            continue
        try:
            bodies = os.listdir(root)
        except OSError:
            continue
        for body in bodies:
            body_path = os.path.join(root, body)
            if not os.path.isdir(body_path):
                continue
            brand = _canonical_brand(body)
            if not brand:
                continue
            prefix = body + " "
            try:
                files = os.listdir(body_path)
            except OSError:
                continue
            for fn in files:
                if not fn.lower().endswith(".dcp"):
                    continue
                stem = fn[:-4]
                if not stem.startswith(prefix):
                    continue
                raw_look = stem[len(prefix):].strip()
                if not raw_look or raw_look == "Adobe Standard":
                    continue
                pretty = _prettify_look(raw_look)
                if not pretty:
                    continue
                by_brand.setdefault(brand, {}).setdefault(pretty.lower(), pretty)
    return by_brand


_scan_cache: dict = None


def _cached_scan():
    global _scan_cache
    if _scan_cache is None:
        _scan_cache = _scan_looks_by_brand()
    return _scan_cache


def list_brands_with_looks():
    """
    Brand names (canonicalized) that have at least one Camera Look installed.
    'Adobe Standard' is always the first entry — it's the universal per-body
    neutral profile and doesn't require brand selection.
    """
    brands = sorted(_cached_scan().keys())
    return ["Adobe Standard"] + brands


def list_looks_for_brand(brand):
    """
    Prettified look names for one brand. 'Adobe Standard' and unknown brands
    return an empty list.
    """
    if not brand or brand == "Adobe Standard":
        return []
    by_brand = _cached_scan()
    bucket = by_brand.get(brand)
    if not bucket:
        return []
    return [bucket[k] for k in sorted(bucket)]


def list_all_look_variants():
    """
    Deduped union of every prettified look name across every brand.
    Used as the static combo-options list for camera_look; the JS narrows
    the visible subset to the selected brand at runtime.
    """
    seen = set()
    for bucket in _cached_scan().values():
        for pretty in bucket.values():
            seen.add(pretty)
    return sorted(seen)


def parse_look_entry(entry):
    """
    Split a dropdown entry into (brand, display_look).

    'Canon / Standard' -> ('Canon', 'Standard')
    'Adobe Standard'   -> (None, 'Adobe Standard')
    Bare legacy name   -> (None, entry) — old workflows still resolve.
    """
    if not entry or entry == "Adobe Standard":
        return None, "Adobe Standard"
    if _LOOK_SEP in entry:
        brand, _, look = entry.partition(_LOOK_SEP)
        return brand.strip() or None, look.strip()
    return None, entry.strip()


def find_camera_look_dcp(make, model, look_name):
    """
    Locate a Camera Look DCP for (make, model) whose prettified suffix matches
    the given look_name. Accepts either a prettified name ('Standard',
    'VELVIA', 'Cinelike D') or a raw filename fragment ('Camera Standard',
    'Camera_VELVIA') — both resolve.

    Scans every root × body folder whose canonical brand matches the make,
    so 'Canon EOS 5D Mark IV' will match folders named exactly that under
    any of our search roots.
    """
    if not look_name:
        return None

    target_key = _look_dedup_key(look_name)
    body = _body_folder(make, model)
    if not body:
        return None

    body_lower = body.lower()
    for root in CAMERA_LOOKS_DIRS:
        body_dir = os.path.join(root, body)
        # On Windows the filesystem is case-insensitive so isdir() succeeds
        # even when EXIF make ('FUJIFILM') differs from the folder casing
        # ('Fujifilm'). The filename suffix-strip below must also tolerate
        # that mismatch, which is why we compare lowercased throughout.
        if not os.path.isdir(body_dir):
            continue
        try:
            entries = os.listdir(body_dir)
        except OSError:
            continue
        for fn in entries:
            if not fn.lower().endswith(".dcp"):
                continue
            stem = fn[:-4]
            stem_lower = stem.lower()
            if not stem_lower.startswith(body_lower + " "):
                continue
            suffix = stem[len(body):].strip()
            if not suffix:
                continue
            if _look_dedup_key(suffix) == target_key:
                return os.path.join(body_dir, fn)
    return None


def find_adobe_standard_dcp(make, model, root=ADOBE_STANDARD_DIR):
    """
    Case-insensitive directory scan for "{Make} {Model} Adobe Standard.dcp".
    Handles Adobe's inconsistent capitalisation across brands.
    """
    if not os.path.isdir(root):
        return None
    make_clean = (make or "").strip()
    model_clean = (model or "").strip()
    if not model_clean:
        return None
    try:
        entries = os.listdir(root)
    except OSError:
        return None

    target = f"{make_clean} {model_clean} adobe standard.dcp".lower()
    for fn in entries:
        if fn.lower() == target:
            return os.path.join(root, fn)

    model_lower = model_clean.lower()
    for fn in entries:
        low = fn.lower()
        if low.endswith(" adobe standard.dcp") and model_lower in low:
            return os.path.join(root, fn)

    target_no_make = f"{model_clean} adobe standard.dcp".lower()
    for fn in entries:
        if fn.lower() == target_no_make:
            return os.path.join(root, fn)

    return None


def resolve_profile(make, model, look_name="Adobe Standard"):
    """
    Find and parse a DCP for (make, model, look_name), cached.

    Accepted look_name forms:
      - "Adobe Standard"      neutral per-body calibration (ADOBE_STANDARD_DIR)
      - "Brand / Look"        brand-gated Camera Look (new canonical form)
      - "Look"                legacy bare name — resolves if a matching file
                              exists for the detected body, regardless of brand

    If the requested brand doesn't match the detected body (e.g. user picked
    'Canon / Standard' but the RAW is Fuji), or the look isn't installed for
    this body, the node falls back to Adobe Standard with a console warning.
    Returns None only when even Adobe Standard is missing.
    """
    requested_brand, bare_look = parse_look_entry(look_name or "Adobe Standard")
    body_brand = _canonical_brand(_body_folder(make, model)) or _canonical_brand(make)

    make_k = (make or "").strip().lower()
    model_k = (model or "").strip().lower()
    key = (make_k, model_k, (requested_brand or "").lower(), bare_look.lower())
    if key in _profile_cache:
        return _profile_cache[key]

    if bare_look == "Adobe Standard":
        path = find_adobe_standard_dcp(make, model)
    elif requested_brand and body_brand and requested_brand.lower() != body_brand.lower():
        print(f"[Darkroom DCP] look '{look_name}' is for {requested_brand} bodies; "
              f"this RAW is {body_brand} ({make} {model}). Falling back to Adobe Standard.")
        path = find_adobe_standard_dcp(make, model)
    else:
        path = find_camera_look_dcp(make, model, bare_look)
        if path is None:
            print(f"[Darkroom DCP] '{bare_look}' not installed for {make} {model}; "
                  f"falling back to Adobe Standard.")
            path = find_adobe_standard_dcp(make, model)

    if path is None:
        _profile_cache[key] = None
        return None
    profile = read_dcp_profile(path)
    _profile_cache[key] = profile
    return profile


# --- RGB <-> HSV (unbounded V for HDR) ------------------------------------

def rgb_to_hsv(rgb):
    """
    Linear RGB → HSV. RGB shape (..., 3), values >= 0 (V is unbounded).
    Returns H in degrees [0, 360), S in [0, 1], V = max(R,G,B).
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    v = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = v - mn

    v_safe = np.maximum(v, 1e-10)
    s = np.where(v > 1e-10, delta / v_safe, 0.0).astype(np.float32)

    delta_safe = np.maximum(delta, 1e-10)
    rc = (v - r) / delta_safe
    gc = (v - g) / delta_safe
    bc = (v - b) / delta_safe

    h = np.where(
        r >= v, bc - gc,
        np.where(g >= v, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h * 60.0) % 360.0
    h = np.where(delta > 1e-10, h, 0.0).astype(np.float32)

    return np.stack([h, s, v.astype(np.float32)], axis=-1)


def hsv_to_rgb(hsv):
    """
    HSV → RGB. H in degrees, S in [0,1], V >= 0. Output is unbounded RGB.
    """
    h = hsv[..., 0]
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = hsv[..., 2]

    h6 = (h % 360.0) / 60.0
    i = np.floor(h6).astype(np.int32) % 6
    f = (h6 - np.floor(h6)).astype(np.float32)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [v,       q,      p,      p,      t,      v],
        default=v,
    )
    g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [t,      v,      v,      q,      p,      p],
        default=v,
    )
    b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [p,      p,      t,      v,      v,      q],
        default=v,
    )
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# --- HSV 3D LUT application (HueSatMap / LookTable) -----------------------

def _srgb_encode_v(v):
    """sRGB OETF applied to a V channel for encoding=1 tables."""
    v = np.clip(v, 0.0, None)
    return np.where(
        v <= 0.0031308,
        v * 12.92,
        1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055,
    )


def apply_hsv_table(img_hsv, table, encoding=0):
    """
    Trilinear interpolation of an HSV 3D LUT with a DCP shape:
      - hue axis is periodic (wraps 0 <-> 360)
      - sat axis is clamped to [0,1]
      - value axis is clamped to [0,1] (optionally sRGB-encoded first)

    Each cell stores (hue_shift_deg, sat_scale, val_scale). Returns the
    image in HSV with those corrections applied. Uses scipy's C
    map_coordinates — roughly 5x faster than pure-numpy advanced indexing
    on a 12MP image, dominated now by the HSV conversion itself.
    """
    H_div, S_div, V_div, _ = table.shape

    h = img_hsv[..., 0]
    s = img_hsv[..., 1]
    v = img_hsv[..., 2]

    # Hue axis is periodic — pad a copy of row 0 at index H_div so the
    # interpolator can hit (H_div - 1) + eps without wrap logic.
    # map_coordinates only supports one boundary mode per call, so this
    # padding trick lets us use 'nearest' (safe for clamped sat/val).
    table_padded = np.concatenate([table, table[:1]], axis=0)  # (H+1, S, V, 3)

    hi = ((h / 360.0) % 1.0) * H_div   # stays in [0, H_div]

    if S_div <= 1:
        si = np.zeros_like(h)
    else:
        si = np.clip(s, 0.0, 1.0) * (S_div - 1)

    if V_div <= 1:
        vi = np.zeros_like(h)
    else:
        v_key = _srgb_encode_v(v) if encoding == 1 else v
        vi = np.clip(v_key, 0.0, 1.0) * (V_div - 1)

    coords = np.stack([hi.ravel(), si.ravel(), vi.ravel()], axis=0)

    result_flat = np.empty((3, h.size), dtype=np.float32)
    # Each of the three output channels (hue_shift_deg, sat_scale, val_scale)
    # is interpolated independently over the padded 3D grid.
    for ch in range(3):
        result_flat[ch] = map_coordinates(
            table_padded[..., ch],
            coords,
            order=1,
            mode="nearest",
            cval=0.0,
            prefilter=False,
        )
    result = result_flat.reshape(3, *h.shape)

    hue_shift = result[0]
    sat_scale = result[1]
    val_scale = result[2]

    new_h = (h + hue_shift) % 360.0
    new_s = np.clip(s * sat_scale, 0.0, 1.0)
    new_v = v * val_scale

    return np.stack([new_h, new_s, new_v], axis=-1)


# --- Torch GPU path (50-100x faster on CUDA) ------------------------------

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _rgb_to_hsv_torch(rgb):
    """RGB -> HSV on a (H,W,3) or (N,3) torch tensor. V unbounded."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    v, _ = rgb.max(dim=-1)
    mn, _ = rgb.min(dim=-1)
    delta = v - mn

    s = torch.where(v > 1e-10, delta / v.clamp_min(1e-10), torch.zeros_like(v))

    delta_safe = delta.clamp_min(1e-10)
    rc = (v - r) / delta_safe
    gc = (v - g) / delta_safe
    bc = (v - b) / delta_safe

    h = torch.where(
        r >= v, bc - gc,
        torch.where(g >= v, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h * 60.0) % 360.0
    h = torch.where(delta > 1e-10, h, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=-1)


def _hsv_to_rgb_torch(hsv):
    """HSV -> RGB on a torch tensor. Mirrors hsv_to_rgb numpy version."""
    h = hsv[..., 0]
    s = hsv[..., 1].clamp(0.0, 1.0)
    v = hsv[..., 2]

    h6 = (h % 360.0) / 60.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    zero = torch.zeros_like(v)
    # Use gather-style selection via stacked candidates per sextant.
    r_opts = torch.stack([v, q, p, p, t, v], dim=-1)
    g_opts = torch.stack([t, v, v, q, p, p], dim=-1)
    b_opts = torch.stack([p, p, t, v, v, q], dim=-1)
    idx = i.unsqueeze(-1)
    r = r_opts.gather(-1, idx).squeeze(-1)
    g = g_opts.gather(-1, idx).squeeze(-1)
    b = b_opts.gather(-1, idx).squeeze(-1)
    return torch.stack([r, g, b], dim=-1)


def _build_hsv_table_tensor(table_np, device):
    """
    Turn a (H, S, V, 3) numpy table into a 5D tensor shaped for grid_sample:
    (1, 3, V_out, S_out, H_out+1). Pads hue by 1 (wrap) and forces at least
    2 samples per sat/val axis so grid_sample can interpolate.
    """
    H, S, V, _ = table_np.shape
    # Pad hue by duplicating slice 0 at the end
    padded = np.concatenate([table_np, table_np[:1]], axis=0)  # (H+1, S, V, 3)
    # Ensure >=2 samples on S and V axes
    if S == 1:
        padded = np.concatenate([padded, padded[:, :1]], axis=1)
    if V == 1:
        padded = np.concatenate([padded, padded[:, :, :1]], axis=2)
    # Reorder to (3, V_eff, S_eff, H_eff) for grid_sample
    t = torch.from_numpy(padded).to(device)                    # (H+1, S_eff, V_eff, 3)
    t = t.permute(3, 2, 1, 0).contiguous().unsqueeze(0)        # (1, 3, V, S, H+1)
    return t


def _apply_hsv_table_torch(img_hsv, table_tensor, dims_original, encoding=0):
    """
    img_hsv: (H, W, 3) torch tensor
    table_tensor: (1, 3, V_eff, S_eff, H_eff+1) — output of _build_hsv_table_tensor
    dims_original: (H_div, S_div, V_div) — *pre-pad* dimensions, used for normalization
    """
    H_div, S_div, V_div = dims_original
    device = img_hsv.device

    h = img_hsv[..., 0]
    s = img_hsv[..., 1]
    v = img_hsv[..., 2]

    img_h, img_w = h.shape

    # Hue: periodic, normalized to [-1, 1] across the padded [0, H_div] range.
    hi = ((h / 360.0) % 1.0) * H_div                  # [0, H_div]
    h_grid = (hi / H_div) * 2.0 - 1.0                 # [-1, 1]

    # Sat: clamped. After padding, S_eff = max(S_div, 2). Valid si in
    # [0, S_div - 1]; the padded extra slice is never hit unless S_div == 1
    # (in which case we always land at index 0).
    if S_div <= 1:
        s_grid = torch.full_like(h, -1.0)
    else:
        si = s.clamp(0.0, 1.0) * (S_div - 1)          # [0, S_div - 1]
        S_eff = S_div                                 # no padding needed
        s_grid = (si / (S_eff - 1)) * 2.0 - 1.0

    if V_div <= 1:
        v_grid = torch.full_like(h, -1.0)
    else:
        if encoding == 1:
            v_clamped = v.clamp_min(0.0)
            v_key = torch.where(
                v_clamped <= 0.0031308,
                v_clamped * 12.92,
                1.055 * v_clamped.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
            )
        else:
            v_key = v
        vi = v_key.clamp(0.0, 1.0) * (V_div - 1)
        V_eff = V_div
        v_grid = (vi / (V_eff - 1)) * 2.0 - 1.0

    # grid_sample 5D input: (N, C, D, H, W)
    # grid: (N, D_out, H_out, W_out, 3) where 3-component is (x, y, z) = (W, H, D)
    # We use D_out=1, H_out=img_h, W_out=img_w
    # x maps to hue axis (W), y maps to sat axis (H), z maps to val axis (D)
    grid = torch.stack([h_grid, s_grid, v_grid], dim=-1)      # (H, W, 3)
    grid = grid.unsqueeze(0).unsqueeze(0)                     # (1, 1, H, W, 3)

    sampled = F.grid_sample(
        table_tensor, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )                                                         # (1, 3, 1, H, W)
    sampled = sampled.squeeze(0).squeeze(1).permute(1, 2, 0)  # (H, W, 3)

    hue_shift = sampled[..., 0]
    sat_scale = sampled[..., 1]
    val_scale = sampled[..., 2]

    new_h = (h + hue_shift) % 360.0
    new_s = (s * sat_scale).clamp(0.0, 1.0)
    new_v = v * val_scale
    return torch.stack([new_h, new_s, new_v], dim=-1)


def _apply_dcp_torch(img_linear_np, profile, illuminant_weight, device):
    """
    Torch path for apply_dcp. Runs HSM + LUT + optional ToneCurve on GPU.
    Input and output are numpy arrays; the torch detour is internal.
    """
    img = torch.from_numpy(np.clip(img_linear_np, 0.0, None).astype(np.float32)).to(device)
    hsv = _rgb_to_hsv_torch(img)

    if profile.has_hsm:
        if profile.hsm_data_2 is not None:
            w = float(np.clip(illuminant_weight, 0.0, 1.0))
            if w >= 0.999:
                hsm = profile.hsm_data_2
            elif w <= 0.001:
                hsm = profile.hsm_data_1
            else:
                hsm = (profile.hsm_data_1 * (1.0 - w) + profile.hsm_data_2 * w).astype(np.float32)
        else:
            hsm = profile.hsm_data_1
        hsm_t = _build_hsv_table_tensor(hsm, device)
        hsv = _apply_hsv_table_torch(hsv, hsm_t, profile.hsm_dims, encoding=profile.hsm_encoding)

    if profile.has_lut:
        lut_t = _build_hsv_table_tensor(profile.lut_data, device)
        hsv = _apply_hsv_table_torch(hsv, lut_t, profile.lut_dims, encoding=profile.lut_encoding)

    if profile.has_tone_curve:
        # ProfileToneCurve goes through the numpy spline — the points are few
        # and the cost is per-pixel, not per-control-point. Ship V back to CPU
        # for this one call, then straight back.
        v_np = hsv[..., 2].clamp_min(0.0).cpu().numpy()
        v_clamped = np.clip(v_np, 0.0, 1.0)
        pts = [(float(x), float(y)) for x, y in profile.tone_curve]
        v_new = cubic_spline_curve(v_clamped, pts)
        v_new_t = torch.from_numpy(v_new).to(device)
        hsv = torch.stack([hsv[..., 0], hsv[..., 1], v_new_t], dim=-1)

    rgb = _hsv_to_rgb_torch(hsv)
    return rgb.clamp_min(0.0).cpu().numpy().astype(np.float32)


# --- Top-level applier ----------------------------------------------------

def apply_dcp(img_linear, profile, illuminant_weight=1.0):
    """
    Apply a DCP profile's HueSatMap + LookTable + ProfileToneCurve to a
    linear-light image (HWC float32).

    illuminant_weight: blend between HSM illuminant 1 and 2 maps.
        0.0 = pure map 1 (typically tungsten ~2850K)
        1.0 = pure map 2 (typically daylight ~6500K)
        Default 1.0 favours daylight — good enough for most scenes until
        we wire up a color-temperature estimator from As-Shot WB gains.

    The image is assumed to be linear sRGB (what rawpy returns with our
    current postprocess settings). DCPs are formally defined in ProPhoto
    linear; using linear sRGB here is a controlled approximation — the
    HSM/LookTable corrections are small enough that the primaries error
    is dominated by the benefit of applying the profile at all.

    Returns the image unchanged if the profile is None or empty.
    """
    if profile is None:
        return img_linear
    if not (profile.has_hsm or profile.has_lut or profile.has_tone_curve):
        return img_linear

    # Prefer the torch GPU path on CUDA — ~100x faster than numpy on a
    # 50 MP image. Falls back to numpy if torch is unavailable or CUDA
    # is missing.
    if _HAS_TORCH:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                return _apply_dcp_torch(img_linear, profile, illuminant_weight, device)
        except Exception as ex:
            print(f"[Darkroom DCP] torch path failed ({ex}), falling back to numpy")

    img = np.clip(img_linear, 0.0, None).astype(np.float32)
    hsv = rgb_to_hsv(img)

    if profile.has_hsm:
        hsm = profile.hsm_data_1
        if profile.hsm_data_2 is not None:
            w = float(np.clip(illuminant_weight, 0.0, 1.0))
            if w >= 0.999:
                hsm = profile.hsm_data_2
            elif w <= 0.001:
                hsm = profile.hsm_data_1
            else:
                hsm = profile.hsm_data_1 * (1.0 - w) + profile.hsm_data_2 * w
        hsv = apply_hsv_table(hsv, hsm, encoding=profile.hsm_encoding)

    if profile.has_lut:
        hsv = apply_hsv_table(hsv, profile.lut_data, encoding=profile.lut_encoding)

    if profile.has_tone_curve:
        pts = [(float(x), float(y)) for x, y in profile.tone_curve]
        v = hsv[..., 2]
        v_clamped = np.clip(v, 0.0, 1.0)
        v_new = cubic_spline_curve(v_clamped, pts)
        hsv = np.stack([hsv[..., 0], hsv[..., 1], v_new], axis=-1)

    rgb = hsv_to_rgb(hsv)
    return np.clip(rgb, 0.0, None).astype(np.float32)
