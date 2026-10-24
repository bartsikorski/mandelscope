"""Render the animations and pictures of the README (docs/images/*.gif, *.jpg).

    uv run python tools/readme_media.py                 # all of them
    uv run python tools/readme_media.py deep_zoom ship  # some
    uv run python tools/readme_media.py --list

Clips are short scripts in the explorer's script language, rendered as videos (fast zoom
movies, 2x2 supersampled) and turned into GIFs with ffmpeg's two-pass palette.  Runs on any
OpenGL 4.3 driver, the software renderer included (minutes instead of seconds).
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mandelscope.cli import make_headless_context, load_location_full   # noqa: E402
from mandelscope.library import BUILTIN_DIR                              # noqa: E402
from mandelscope.mp import View                                          # noqa: E402
from mandelscope.offline import FrameRenderer, RenderSettings, render_sequence_iter   # noqa: E402
from mandelscope.renderer import ColorSettings, GPUEngine                # noqa: E402
from mandelscope.script import compile_script                           # noqa: E402
from mandelscope.video import find_ffmpeg                                # noqa: E402

OUT = ROOT / "docs" / "images"

GIF_WIDTH = 360            # (a zoom changes every pixel of every frame: GIFs grow fast with size)

# name: (script, size, fps, julia c or None)
CLIPS = {
    "seahorse_zoom": ("""look Nebula
start location "seahorse_valley"
zoom to location "seahorse_1e-12" over 9s spin 0.25 ease smooth
hold 1s
""", (416, 234), 12, None),
    "deep_zoom": ("""look Midnight
start home
zoom to location "spiral_1e-100" over 10s ease smooth
hold 1s
""", (416, 234), 12, None),
}

# name: (location, look, size, supersampling)
STILLS = {
    "ship_armada": ("ship_armada", "Gilded", (960, 540), 2),
    "ship_mini_1e-33": ("ship_mini_1e-33", "Copper", (960, 540), 2),
    "julia_spirals": ({"re": "0.2", "im": "0.1", "log2r": -1.5, "angle": 0.0, "max_iter": 3000}, "Gilded", (960, 540), 2),
}
JULIA = {"julia_spirals": ("-0.7445", "0.1123")}       # c of the Julia set (a point of the seahorse valley)


def resolve(name: str):
    p = BUILTIN_DIR / f"{name.strip().strip(chr(34))}.json"
    if not p.exists():
        return None
    view, _, _, formula = load_location_full(str(p))
    view.formula = formula
    return view


def gif(mp4: Path, out: Path, fps: int, width: int):
    ff = find_ffmpeg()
    flt = f"fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as td:
        pal = Path(td) / "palette.png"
        subprocess.run([ff, "-v", "error", "-y", "-i", str(mp4), "-vf", f"{flt},palettegen=stats_mode=diff",
                        str(pal)], check=True)
        subprocess.run([ff, "-v", "error", "-y", "-i", str(mp4), "-i", str(pal), "-lavfi",
                        f"{flt} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", str(out)], check=True)


def render_clip(eng, name: str, tmp: Path):
    text, (w, h), fps, julia = CLIPS[name]
    res = compile_script(text, resolve_location=resolve)
    if res.errors:
        raise SystemExit(f"{name}: {res.errors}")
    seq = res.sequence
    if julia is not None:
        seq.meta["julia"] = julia
    colors = ColorSettings.from_look(res.look) if res.look else ColorSettings()
    rs = RenderSettings(width=w, height=h, ss=2, fast=True)
    mp4 = tmp / f"{name}.mp4"
    for _ in render_sequence_iter(eng, seq, str(mp4), rs, colors, fps, "libx264", 12):
        pass
    gif(mp4, OUT / f"{name}.gif", fps, min(w, GIF_WIDTH))


def render_still(eng, name: str):
    loc, look, (w, h), ss = STILLS[name]
    if isinstance(loc, dict):
        view, colors, jc, formula = View.from_dict(loc), ColorSettings(), JULIA.get(name), "mandelbrot"
    else:
        view, colors, jc, formula = load_location_full(str(BUILTIN_DIR / f"{loc}.json"))
    if look:
        colors = ColorSettings.from_look(look)
    fr = FrameRenderer(eng, RenderSettings(width=w, height=h, ss=ss))
    fr.formula = formula
    fr.julia = tuple(jc) if jc else None
    from PIL import Image
    Image.fromarray(fr.render(view, colors)).save(OUT / f"{name}.jpg", quality=90)
    fr.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", metavar="DIR", help="keep the videos the GIFs are made from in DIR")
    a = ap.parse_args()
    if a.list:
        print("clips :", " ".join(CLIPS))
        print("stills:", " ".join(STILLS))
        return
    names = a.names or list(CLIPS) + list(STILLS)
    eng = GPUEngine(make_headless_context())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        if a.keep:
            Path(a.keep).mkdir(parents=True, exist_ok=True)
            td = a.keep
        for name in names:
            t0 = time.perf_counter()
            if name in CLIPS:
                render_clip(eng, name, Path(td))
                out = OUT / f"{name}.gif"
            elif name in STILLS:
                render_still(eng, name)
                out = OUT / f"{name}.jpg"
            else:
                raise SystemExit(f"unknown: {name} (--list)")
            print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.1f} MB  {time.perf_counter() - t0:.0f} s",
                  flush=True)


if __name__ == "__main__":
    main()
