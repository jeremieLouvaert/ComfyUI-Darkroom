"""
Allowed-folder confinement for every path Darkroom reads or writes on behalf of
a client: node widgets (RAW Load, LUT Apply, LUT Export, CMYK Export) and the
folder-picker HTTP routes.

ComfyUI's /prompt and our /darkroom/* routes are unauthenticated, so any path
that arrives from a workflow or a request is untrusted. It must resolve inside
an allowed root:

    read  : ComfyUI input, output, temp, models  + extra folders
    write : ComfyUI output                        + extra folders

Extra folders come from <ComfyUI>/darkroom_allowed_folders.json, a JSON list of
absolute paths. The file deliberately lives in the ComfyUI root, NOT under
user/: ComfyUI's own /userdata API can write files under user/, which would let
a remote caller allow themselves any folder. Nothing in ComfyUI or Darkroom can
write to the ComfyUI root over HTTP, so only someone with local file access can
widen the roots.

The check is lexical first: a path only reaches the filesystem (realpath) once
its normalised text already sits inside a root. Checking the other way round
lets a client make the server open a network share or a named pipe just by
naming it, which leaks the Windows login hash over SMB and stalls the server
while the connection times out.
"""

import json
import os
import re

try:
    import folder_paths
except ImportError:  # outside ComfyUI (offline tools): no roots, nothing allowed
    folder_paths = None


EXTRAS_FILENAME = "darkroom_allowed_folders.json"

_SUPERSCRIPT_DIGITS = ["\u00b9", "\u00b2", "\u00b3"]
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{i}" for i in [*range(1, 10), *_SUPERSCRIPT_DIGITS]),
    *(f"LPT{i}" for i in [*range(1, 10), *_SUPERSCRIPT_DIGITS]),
}
_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_MAX_NAME = 120
# UNC shares (\\host\share, //host/share), \\?\ and \\.\ device paths, \??\ NT paths
_NETWORK_OR_DEVICE = ("\\\\", "//", "\\??\\")


class PathNotAllowed(ValueError):
    """A client-supplied path resolved outside every allowed root."""


def extras_file():
    if folder_paths is None:
        return None
    return os.path.join(folder_paths.base_path, EXTRAS_FILENAME)


_extras_cache = {"mtime": None, "roots": []}


def _extra_roots():
    """Folders listed in the extras file, re-read whenever it changes."""
    p = extras_file()
    try:
        mtime = os.path.getmtime(p) if p else None
    except OSError:
        mtime = None
    if mtime is None:
        _extras_cache.update(mtime=None, roots=[])
        return []
    if mtime == _extras_cache["mtime"]:
        return _extras_cache["roots"]

    roots = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("expected a JSON list of folder paths")
        for item in data:
            if isinstance(item, str) and item.strip() and os.path.isabs(item.strip()):
                roots.append(item.strip())
            else:
                print(f"[Darkroom] {EXTRAS_FILENAME}: ignoring {item!r} (not an absolute path)")
    except Exception as e:
        print(f"[Darkroom] {EXTRAS_FILENAME}: could not read ({e}); no extra folders allowed")
        roots = []
    _extras_cache.update(mtime=mtime, roots=roots)
    return roots


def _builtin_roots(scope):
    if folder_paths is None:
        return []
    if scope == "write":
        return [("ComfyUI output", folder_paths.get_output_directory())]
    return [
        ("ComfyUI input", folder_paths.get_input_directory()),
        ("ComfyUI output", folder_paths.get_output_directory()),
        ("ComfyUI temp", folder_paths.get_temp_directory()),
        ("ComfyUI models", folder_paths.models_dir),
    ]


def _check_scope(scope):
    if scope not in ("read", "write"):
        raise ValueError(f"unknown scope {scope!r}")


def _lex(path):
    """Absolute, normalised path text. String work only: never touches the disk."""
    return os.path.normpath(os.path.abspath(path))


def _is_network_or_device(path):
    return path.startswith(_NETWORK_OR_DEVICE)


_roots_cache = {}


def _roots(scope):
    """[(label, lexical_root, real_root)] for the scope, builtin roots first.

    Cached per configuration, so a slow extras folder (an offline NAS) costs one
    realpath when the configuration changes, not one per request."""
    _check_scope(scope)
    config = (tuple(_builtin_roots(scope)), tuple(_extra_roots()))
    cached = _roots_cache.get(scope)
    if cached and cached[0] == config:
        return cached[1]
    out = [(label, _lex(path), os.path.realpath(path)) for label, path in config[0]]
    for path in config[1]:
        out.append((os.path.basename(path.rstrip("/\\")) or path, _lex(path), os.path.realpath(path)))
    _roots_cache[scope] = (config, out)
    return out


def allowed_roots(scope):
    """[(label, realpath)] for the scope, builtin roots first."""
    return [(label, real) for label, _lex_root, real in _roots(scope)]


def _inside(path, root):
    """True if `path` is `root` or below it (case-insensitive on Windows)."""
    p, r = os.path.normcase(path), os.path.normcase(root)
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:  # different drives, or mixed absolute/relative
        return False


def resolve_allowed(path, scope):
    """
    Resolve a client-supplied path and return its real absolute path if it lies
    inside an allowed root for `scope`, else raise PathNotAllowed.

    Relative paths resolve against ComfyUI input (read) or output (write).
    Symlinks and junctions are followed before the final check, so a link
    inside a root that points outside it is refused.
    """
    _check_scope(scope)
    raw = (path or "").strip()
    if not raw:
        raise PathNotAllowed("empty path")
    if "\x00" in raw:
        raise PathNotAllowed("invalid path")
    if not os.path.isabs(raw):
        if folder_paths is None:
            raise PathNotAllowed("relative paths need ComfyUI")
        base = (folder_paths.get_output_directory() if scope == "write"
                else folder_paths.get_input_directory())
        raw = os.path.join(base, raw)

    roots = _roots(scope)
    try:
        lex = _lex(raw)
    except (OSError, ValueError):
        raise PathNotAllowed("invalid path")

    # 1. Lexical containment, before any filesystem access. A network or
    #    device path only passes if it sits under a root that is itself one,
    #    as configured or once resolved (a mapped drive Z: realpaths to
    #    \\nas\share, and the picker hands that form back). Roots come from
    #    server configuration, never from the client.
    if _is_network_or_device(lex):
        ok = any((_is_network_or_device(r) and _inside(lex, r))
                 or (_is_network_or_device(real) and _inside(lex, real))
                 for _l, r, real in roots)
    else:
        ok = any(_inside(lex, r) or _inside(lex, real) for _l, r, real in roots)
    if not ok:
        raise PathNotAllowed(not_allowed_message(scope))

    # 2. Follow symlinks and junctions, then check the real path again.
    try:
        real = os.path.realpath(lex)
    except (OSError, ValueError):
        raise PathNotAllowed("invalid path")
    if any(_inside(real, root) for _l, _lex_root, root in roots):
        return real
    raise PathNotAllowed(not_allowed_message(scope))


def is_allowed(path, scope):
    try:
        resolve_allowed(path, scope)
        return True
    except PathNotAllowed:
        return False


def is_root(path, scope):
    try:
        real = os.path.normcase(resolve_allowed(path, scope))
    except PathNotAllowed:
        return False
    return any(os.path.normcase(r) == real for _l, r in allowed_roots(scope))


def resolve_output_file(directory, name):
    """
    Final path for a file a node is about to write: `name` (already passed
    through safe_filename) inside the confined `directory`. Re-checked after
    realpath so a link already sitting at that name cannot redirect the write.
    """
    target = os.path.join(directory, name)
    return resolve_allowed(target, "write")


def not_allowed_message(scope):
    where = ("ComfyUI's output folder" if scope == "write"
             else "ComfyUI's input, output, temp or models folders")
    return (f"Path is outside the folders Darkroom may {'write to' if scope == 'write' else 'read'}: "
            f"{where}. To allow another folder, add it to {EXTRAS_FILENAME} in the ComfyUI "
            f"folder (a JSON list of absolute paths).")


def safe_filename(name, default):
    """
    Reduce a client-supplied file name to a single safe path component:
    no directories, no drive or stream colons, no Windows-reserved names.
    """
    s = (name or "").replace("\\", "/").split("/")[-1]
    s = _ILLEGAL_CHARS.sub("_", s).strip().rstrip(". ")
    if s in ("", ".", ".."):
        s = default
    if s.split(".")[0].upper() in _WINDOWS_RESERVED:
        s = "_" + s
    return s[:_MAX_NAME]
