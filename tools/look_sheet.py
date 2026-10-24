"""Contact sheet of colour looks: every look (columns) at several places (rows).

    python tools/look_sheet.py [--looks Gilded,Nebula] [--places seahorse_valley,starfish]
                               [--size 320x180] [--out looks.png]

Each place is computed once (all colouring channels: distance, slope, stripes, decomposition
angle), then coloured with every look exactly like the explorer does (same shader, 'auto'
histogram per picture), 2x2 supersampled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mandelscope.cli import make_headless_context                           # noqa: E402
from mandelscope.renderer import (GPUEngine, Field, GPURef, GPUBla, AutoContrast, ColorSettings,  # noqa: E402
                               view_params, julia_params)
from mandelscope.reference import compute_reference                         # noqa: E402
from mandelscope.bla import build_bla                                       # noqa: E402
from mandelscope.mp import View                                             # noqa: E402
from mandelscope.palettes import LOOKS                                      # noqa: E402

PLACES = ["seahorse_valley", "elephant_valley", "starfish", "lightning", "double_spiral",
          "seahorse_1e-12", "minibrot_p6006", "spiral_1e-100"]


def compute(eng, name, W, H, stripe_freq=5.0):
    d = json.loads((ROOT / "mandelscope" / "locations" / f"{name}.json").read_text())
    v = View.from_dict(d["view"] if "view" in d else d)
    dec = max(0.0, v.zoom_magnification_log10())
    v.max_iter = max(int(v.max_iter), int(1000 + 400 * dec))
    ctx = eng.ctx
    f = Field(eng, W, H, False)
    if "julia" in d:
        c = tuple(float(x) for x in d["julia"])
        fp = julia_params(v, W, H, with_de=True, with_interior=True, with_stripes=True, stripe_freq=stripe_freq)
        f.configure(fp, None, None)
        eng.setup_julia(f, v, c, W, H, fp)
    else:
        ref = compute_reference(str(v.re), str(v.im), v.needed_precision(H) + 32, v.max_iter)
        ref.ref_id = hash(name) & 0xFFFF
        fp = view_params(v, v.re, v.im, W, H, with_de=True, with_interior=True, with_stripes=True,
                         stripe_freq=stripe_freq)
        f.configure(fp, GPURef(ctx, ref, False), GPUBla(ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax), False))
    f.run_to_completion()
    return f, v


def colour(eng, f, v, look, W, H, fbo, auto):
    cs = ColorSettings.from_look(look)
    a = None
    if cs.mapping == 3:
        auto.update(f)
        a = auto
    eng.display.render(f, fbo, (0, 0, W, H), 1.0, cs, fallback=False, max_iter_show=v.max_iter, auto=a)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    return img


def main():
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser()
    ap.add_argument("--looks", default=",".join(LOOKS))
    ap.add_argument("--places", default=",".join(PLACES))
    ap.add_argument("--size", default="320x180")
    ap.add_argument("--out", default="looks.png")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    W, H = 2 * w, 2 * h
    looks = [x for x in args.looks.split(",") if x]
    places = [x for x in args.places.split(",") if x]
    eng = GPUEngine(make_headless_context())
    tex = eng.ctx.texture((W, H), 4)
    fbo = eng.ctx.framebuffer(color_attachments=[tex])
    auto = AutoContrast(eng.ctx)
    pad, top, left = 6, 26, 130
    sheet = Image.new("RGB", (left + len(looks) * (w + pad), top + len(places) * (h + pad)), (18, 18, 22))
    dr = ImageDraw.Draw(sheet)
    try:
        import imgui_bundle
        font = ImageFont.truetype(str(Path(imgui_bundle.__file__).parent / "assets/fonts/Roboto/Roboto-Regular.ttf"), 14)
    except Exception:
        font = ImageFont.load_default()
    for j, look in enumerate(looks):
        dr.text((left + j * (w + pad) + 4, 5), look, fill=(230, 230, 235), font=font)
    for i, place in enumerate(places):
        f, v = compute(eng, place, W, H)
        dr.text((6, top + i * (h + pad) + h // 2 - 8), place, fill=(200, 200, 205), font=font)
        for j, look in enumerate(looks):
            img = Image.fromarray(colour(eng, f, v, look, W, H, fbo, auto)).resize((w, h), Image.BOX)
            sheet.paste(img, (left + j * (w + pad), top + i * (h + pad)))
        f.release()
        print(place, flush=True)
    sheet.save(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
