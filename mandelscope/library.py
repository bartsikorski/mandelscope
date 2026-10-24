"""Location library: built-in presets + the user's saved locations (with thumbnails).

User files live in   Documents/Mandelscope Files/Locations/<name>.json  (+ <name>.png thumbnail)
so they are easy to find, back up and share.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mp import View

BUILTIN_DIR = Path(__file__).parent / "locations"
BUILTIN_THUMBS = BUILTIN_DIR / "thumbs"


def user_root() -> Path:
    """Documents/Mandelscope Files (Windows) or ~/Mandelscope Files elsewhere; overridable with
    MANDELSCOPE_HOME.  (Not "Documents/Mandelscope": on Windows that is the same folder as a
    clone of the repository named "mandelscope".)  Files of the pre-release versions
    (Documents/DeepZoom) keep being used if that is all there is."""
    env = os.environ.get("MANDELSCOPE_HOME") or os.environ.get("DEEPZOOM_HOME")
    if env:
        return Path(env)
    docs = Path.home() / "Documents"
    base = docs if docs.exists() else Path.home()
    if not (base / "Mandelscope Files").exists() and (base / "DeepZoom").exists():
        return base / "DeepZoom"
    return base / "Mandelscope Files"


def user_dir() -> Path:
    return user_root() / "Locations"


def safe_name(name: str) -> str:
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    return n[:80] or "location"


@dataclass
class Entry:
    name: str
    path: Path
    builtin: bool
    thumb: Path | None
    mtime: float = 0.0
    zoom: float = 0.0
    title: str = ""
    formula: str = "mandelbrot"

    @property
    def label(self) -> str:
        return self.title or self.name

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class Library:
    def __init__(self):
        self.entries: list[Entry] = []
        self.version = 0
        self.refresh()

    def refresh(self):
        out = []
        for p in sorted(BUILTIN_DIR.glob("*.json")):
            if p.name.endswith(".seq.json"):
                continue
            t = BUILTIN_THUMBS / (p.stem + ".png")
            z, title, f = _info(p)
            out.append(Entry(p.stem, p, True, t if t.exists() else None, p.stat().st_mtime, z, title, f))
        # built-ins from shallow to deep, the Mandelbrot set first
        out.sort(key=lambda e: (e.formula != "mandelbrot", e.zoom))
        ud = user_dir()
        if ud.exists():
            for p in sorted(ud.glob("*.json"), key=lambda q: -q.stat().st_mtime):
                if p.name.endswith(".seq.json"):
                    continue
                t = p.with_suffix(".png")
                z, title, f = _info(p)
                out.append(Entry(p.stem, p, False, t if t.exists() else None, p.stat().st_mtime, z, title, f))
        self.entries = out
        self.version += 1

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def find(self, name: str) -> Entry | None:
        name_l = name.strip().strip('"').lower()
        for e in self.entries:
            if e.name.lower() == name_l:
                return e
        for e in self.entries:
            if e.title and e.title.lower() == name_l:
                return e
        p = Path(name.strip().strip('"'))
        if p.suffix.lower() == ".json" and p.exists():
            return Entry(p.stem, p, False, None)
        return None

    def view(self, name: str) -> View | None:
        e = self.find(name)
        if e is None:
            return None
        d = e.load()
        v = View.from_dict(d["view"] if "view" in d else d)
        v.formula = d.get("formula", "mandelbrot")      # (read by the script compiler)
        return v

    def save(self, name: str, location: dict, thumb_rgb: np.ndarray | None = None,
             overwrite: bool = False) -> Entry:
        ud = user_dir()
        ud.mkdir(parents=True, exist_ok=True)
        base = safe_name(name)
        p = ud / f"{base}.json"
        if p.exists() and not overwrite:
            i = 2
            while (ud / f"{base} ({i}).json").exists():
                i += 1
            p = ud / f"{base} ({i}).json"
        location = dict(location)
        location["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        p.write_text(json.dumps(location, indent=1), encoding="utf-8")
        if thumb_rgb is not None:
            save_thumbnail(thumb_rgb, p.with_suffix(".png"))
        self.refresh()
        return self.find(p.stem)

    def delete(self, name: str) -> bool:
        e = self.find(name)
        if e is None or e.builtin:
            return False
        e.path.unlink(missing_ok=True)
        if e.thumb is not None:
            e.thumb.unlink(missing_ok=True)
        self.refresh()
        return True

    def rename(self, name: str, new_name: str) -> bool:
        e = self.find(name)
        if e is None or e.builtin:
            return False
        q = e.path.with_name(safe_name(new_name) + ".json")
        if q.exists():
            return False
        e.path.rename(q)
        if e.thumb is not None and e.thumb.exists():
            e.thumb.rename(q.with_suffix(".png"))
        self.refresh()
        return True


def _info(p: Path) -> tuple[float, str, str]:
    """(zoom depth in decades, display title, formula) of a location file."""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        title = str(d.get("title", ""))
        f = str(d.get("formula", "mandelbrot"))
        v = d.get("view", d)
        if "log2r" in v:
            return (1.0 - float(v["log2r"])) * 0.30102999566398, title, f
        z = str(v.get("zoom", "1e0")).lower()
        return (float(z.split("e")[1]) if "e" in z else 0.0), title, f
    except Exception:
        return 0.0, "", "mandelbrot"


def save_thumbnail(rgb: np.ndarray, path: Path, size=(256, 144)):
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(rgb))
    w, h = im.size
    tw, th = size
    # centre crop to the thumbnail aspect, then resize
    if w / h > tw / th:
        nw = int(h * tw / th)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w * th / tw)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    im = im.resize((tw, th), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def location_from_png(path: str) -> dict | None:
    """Locations embedded in PNGs saved by Mandelscope ('mandelscope' text chunk)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            txt = (im.info.get("mandelscope") or im.info.get("deepzoom")
                   or (im.text.get("mandelscope") or im.text.get("deepzoom") if hasattr(im, "text") else None))
        return json.loads(txt) if txt else None
    except Exception:
        return None
