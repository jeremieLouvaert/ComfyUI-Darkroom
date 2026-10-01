"""
Backend HTTP routes for ComfyUI-Darkroom.
Registers aiohttp endpoints on ComfyUI's PromptServer for the folder picker
used by LUT Export, LUT Apply and RAW Load.

These routes are unauthenticated like the rest of ComfyUI, so every path a
client sends is confined to the allowed roots in utils/paths.py: browsing,
pinning and folder creation never reach outside them.
"""

import asyncio
import json
import os

from .utils.paths import (
    allowed_roots, resolve_allowed, is_allowed, is_root, PathNotAllowed,
    not_allowed_message,
)

from aiohttp import web
from server import PromptServer

try:
    import folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False


def _norm(path):
    return path.replace("\\", "/") if path else path


# --- Pinned folders (cross-node quick-access) -----------------------------
#
# User-added favourites, shared by every node that opens the path picker.
# Lives under ComfyUI's user/ dir so it survives ComfyUI-Darkroom updates
# and follows the user's settings-follow-dotfiles workflow if any.

_PINS_REL_PATH = os.path.join("default", "darkroom", "pinned_folders.json")


def _pins_file():
    if not _HAS_FOLDER_PATHS:
        return None
    try:
        return os.path.join(folder_paths.get_user_directory(), _PINS_REL_PATH)
    except Exception:
        return None


def _load_pins():
    p = _pins_file()
    if not p or not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("path"):
            path = _norm(str(item["path"]))
            name = str(item.get("name") or os.path.basename(path.rstrip("/")) or path)
            out.append({"name": name, "path": path})
        elif isinstance(item, str) and item:
            path = _norm(item)
            out.append({"name": os.path.basename(path.rstrip("/")) or path, "path": path})
    return out


def _visible_pins(pins):
    return [p for p in pins if is_allowed(p["path"], "read")]


def _save_pins(pins):
    p = _pins_file()
    if not p:
        return False
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=2)
    return True


def _get_roots(scope):
    return [{"name": label, "path": _norm(path)} for label, path in allowed_roots(scope)]


def _scope_of(query):
    return "write" if query.get("scope") == "write" else "read"


async def _read_json(request):
    try:
        data = await request.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


# Every handler below is a thin async shell around a synchronous body run in a
# worker thread: the bodies touch the filesystem, and a slow disk or share must
# never stall ComfyUI's event loop (prompts, websocket, UI).


def _list_entries(path, extensions=None):
    subdirs, files = [], []
    with os.scandir(path) as it:
        for entry in it:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not entry.name.startswith("."):
                        subdirs.append({
                            "name": entry.name,
                            "path": _norm(entry.path),
                        })
                elif extensions and entry.is_file(follow_symlinks=False):
                    name_lower = entry.name.lower()
                    if any(name_lower.endswith(ext) for ext in extensions):
                        files.append({
                            "name": entry.name,
                            "path": _norm(entry.path),
                        })
            except (PermissionError, OSError):
                continue
    subdirs.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())
    return subdirs, files


def _parent_of(path, scope):
    """Parent folder for the Up button, or "" at an allowed root."""
    if not path or is_root(path, scope):
        return ""
    parent = os.path.dirname(path.rstrip("/\\"))
    if not parent or not is_allowed(parent, scope):
        return ""
    return _norm(parent)


def _parse_extensions(raw):
    if not raw:
        return None
    parts = [e.strip().lower() for e in raw.split(",") if e.strip()]
    if not parts:
        return None
    return tuple(e if e.startswith(".") else "." + e for e in parts)


def _list_dir(query):
    scope = _scope_of(query)
    raw_path = query.get("path", "").strip()
    extensions = _parse_extensions(query.get("extensions", ""))
    roots = _get_roots(scope)
    empty = {"path": "", "parent": "", "subdirs": [], "files": [], "roots": roots, "writable": False}

    if raw_path:
        try:
            path = resolve_allowed(raw_path, scope)
        except PathNotAllowed:
            # Same answer whether or not the path exists: no probing.
            return web.json_response({**empty, "error": not_allowed_message(scope)})
        if os.path.isfile(path):
            path = os.path.dirname(path)
    else:
        # Start in output/luts (where LUT Export writes), else the first root.
        out = next((r["path"] for r in roots if r["name"] == "ComfyUI output"), "")
        candidates = [os.path.join(out, "luts"), out] if out else []
        candidates += [r["path"] for r in roots]
        path = next((c for c in candidates if c and os.path.isdir(c)), "")

    if not path or not os.path.isdir(path):
        return web.json_response(empty)

    path = _norm(path)
    try:
        subdirs, files = _list_entries(path, extensions)
    except OSError:
        return web.json_response({**empty, "path": path, "parent": _parent_of(path, scope),
                                  "error": "Could not read this folder"})

    return web.json_response({
        "path": path,
        "parent": _parent_of(path, scope),
        "subdirs": subdirs,
        "files": files,
        "roots": roots,
        "writable": scope == "write" and os.access(path, os.W_OK),
    })


def _mkdir(data):
    raw_path = (data.get("path") or "").strip()
    if not raw_path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)

    # Folders can only be created where Darkroom may write, and only one level
    # below an existing allowed folder.
    try:
        path = resolve_allowed(raw_path, "write")
        parent = resolve_allowed(os.path.dirname(path), "write")
    except PathNotAllowed:
        return web.json_response({"ok": False, "error": not_allowed_message("write")}, status=403)
    if not os.path.isdir(parent):
        return web.json_response({"ok": False, "error": "Parent folder does not exist"}, status=400)

    try:
        os.mkdir(path)
    except FileExistsError:
        return web.json_response({"ok": False, "error": "Folder already exists"}, status=400)
    except OSError:
        return web.json_response({"ok": False, "error": "Could not create the folder"}, status=500)

    return web.json_response({"ok": True, "path": _norm(path)})


def _pins_get():
    # Pins outside the allowed roots stay in the file (they come back if the
    # folder is allowed later) but are never shown or followed.
    return web.json_response({"ok": True, "pins": _visible_pins(_load_pins())})


def _pins_add(data):
    raw_path = (data.get("path") or "").strip()
    if not raw_path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)
    try:
        path = _norm(resolve_allowed(raw_path, "read"))
    except PathNotAllowed:
        return web.json_response({"ok": False, "error": not_allowed_message("read")}, status=403)
    if not os.path.isdir(path):
        return web.json_response({"ok": False, "error": "Path is not a directory"}, status=400)

    name = (data.get("name") or os.path.basename(path.rstrip("/")) or path).strip()
    pins = _load_pins()
    if any(p["path"].lower() == path.lower() for p in pins):
        return web.json_response({"ok": True, "pins": _visible_pins(pins), "duplicate": True})
    pins.append({"name": name, "path": path})
    _save_pins(pins)
    return web.json_response({"ok": True, "pins": _visible_pins(pins)})


def _pins_remove(query):
    raw_path = (query.get("path") or "").strip()
    if not raw_path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)
    path = _norm(raw_path).lower()
    pins = _load_pins()
    new_pins = [p for p in pins if p["path"].lower() != path]
    _save_pins(new_pins)
    return web.json_response({"ok": True, "pins": _visible_pins(new_pins)})


_BAD_JSON = {"ok": False, "error": "Invalid JSON body"}


@PromptServer.instance.routes.get("/darkroom/list_dir")
async def darkroom_list_dir(request):
    return await asyncio.to_thread(_list_dir, dict(request.query))


@PromptServer.instance.routes.post("/darkroom/mkdir")
async def darkroom_mkdir(request):
    data = await _read_json(request)
    if data is None:
        return web.json_response(_BAD_JSON, status=400)
    return await asyncio.to_thread(_mkdir, data)


@PromptServer.instance.routes.get("/darkroom/pins")
async def darkroom_pins_get(_request):
    return await asyncio.to_thread(_pins_get)


@PromptServer.instance.routes.post("/darkroom/pins")
async def darkroom_pins_add(request):
    data = await _read_json(request)
    if data is None:
        return web.json_response(_BAD_JSON, status=400)
    return await asyncio.to_thread(_pins_add, data)


@PromptServer.instance.routes.delete("/darkroom/pins")
async def darkroom_pins_remove(request):
    return await asyncio.to_thread(_pins_remove, dict(request.query))


@PromptServer.instance.routes.get("/darkroom/camera_looks")
async def darkroom_camera_looks(request):
    """Return the prettified look names for one brand, used by the RAW Load
    node's frontend to narrow the camera_look combo after the brand changes."""
    brand = (request.query.get("brand") or "").strip()
    try:
        from .utils.dcp import list_looks_for_brand
        looks = await asyncio.to_thread(list_looks_for_brand, brand)
    except Exception as e:
        print(f"[Darkroom] camera_looks failed: {type(e).__name__}: {e}")
        return web.json_response({"ok": False, "error": "Could not list camera looks"}, status=500)
    return web.json_response({"ok": True, "brand": brand, "looks": looks})


print("[ComfyUI-Darkroom] registered HTTP routes: /darkroom/list_dir, /darkroom/mkdir, "
      "/darkroom/camera_looks, /darkroom/pins (GET/POST/DELETE)")
