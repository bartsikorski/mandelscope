"""Render the thumbnails of the built-in locations (mandelscope/locations/thumbs/*.png).

    uv run python tools/make_thumbnails.py [--look Gilded] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mandelscope.cli import make_headless_context          # noqa: E402
from mandelscope.library import BUILTIN_DIR, BUILTIN_THUMBS, save_thumbnail   # noqa: E402
from mandelscope.mp import View                            # noqa: E402
from mandelscope.offline import FrameRenderer, RenderSettings   # noqa: E402
from mandelscope.renderer import ColorSettings, GPUEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default="Gilded")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    eng = GPUEngine(make_headless_context())
    fr = FrameRenderer(eng, RenderSettings(width=384, height=216, ss=2))
    BUILTIN_THUMBS.mkdir(exist_ok=True)
    for p in sorted(BUILTIN_DIR.glob("*.json")):
        if p.name.endswith(".seq.json"):
            continue
        out = BUILTIN_THUMBS / (p.stem + ".png")
        if out.exists() and not a.force:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        v = View.from_dict(d["view"])
        fr.formula = d.get("formula", "mandelbrot")
        fr.julia = tuple(d["julia"]) if d.get("julia") else None
        t0 = time.perf_counter()
        look = d.get("colors", {}).get("look", a.look)
        img = fr.render(v, ColorSettings.from_look(look))
        save_thumbnail(img, out)
        print(f"{p.stem:24s} {time.perf_counter() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
