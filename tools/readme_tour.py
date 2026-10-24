"""Record the README's interface animation (docs/images/tour.gif) from the real explorer window.

    uv run python tools/readme_tour.py              (Linux without a display: xvfb-run -a ...)

The explorer runs a short scripted tour (a wheel zoom into the seahorse valley, the Look tab
and three looks, the Julia set preview following the cursor) and every displayed frame is
kept.  While a picture refines, only its last frame is kept, so a slow GPU (or the software
renderer) gives the same animation as a fast one, just later.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
from PIL import Image                                  # noqa: E402

import mandelscope.selftest as selftest_mod            # noqa: E402
from mandelscope.app import InputSnapshot, run_app     # noqa: E402
from mandelscope.renderer import ColorSettings         # noqa: E402
from mandelscope.video import find_ffmpeg              # noqa: E402

SIZE = (960, 540)
FPS = 15
GIF_WIDTH = 640


class Tour(selftest_mod.SelfTest):
    frames_dir: Path = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec = False
        self.held = None             # (frame image) kept while a picture refines
        self.n = 0

    def before_swap(self):
        super().before_swap()
        if not (self.rec or self.held is not None):
            return
        a = self.app
        W, H = a.fb
        data = a.ctx.screen.read(viewport=(0, 0, W, H), components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]
        if self.rec:
            self._save(img)
        else:
            self.held = img

    def _save(self, img, repeat: int = 1):
        for _ in range(repeat):
            Image.fromarray(img).save(self.frames_dir / f"{self.n:05d}.png", compress_level=1)
            self.n += 1

    def steps(self):
        a = self.app
        W, H = a.fb
        st = self

        def act(fn):
            def f():
                fn()
                return True
            return f

        def rec(on):
            return act(lambda: setattr(st, "rec", on))

        def settle(hold_s=0.6, timeout=600):
            """Pause recording until the picture is complete, then show it for hold_s."""
            def f():
                if st.held is None:
                    st.rec = False
                    st.held = np.zeros((1, 1, 3), np.uint8)
                    st.t0 = time.perf_counter()
                if st.done_rendering() and a.aa_count >= a.settings.aa_passes or time.perf_counter() - st.t0 > timeout:
                    if st.held.shape[0] > 1:
                        st._save(st.held, int(hold_s * FPS))
                    st.held = None
                    st.rec = True
                    return True
                return False
            return f

        def frames(n, fn=None):
            c = {"i": 0}

            def f():
                if fn is not None:
                    fn(c["i"], n)
                c["i"] += 1
                return c["i"] >= n
            return f

        def mouse(x, y, **kw):
            st.snap = InputSnapshot(mouse=(x, y), **kw)

        def wheel_in(x, y, notches):
            return frames(notches * 3, lambda i, n: mouse(x, y, wheel=1.0 if i % 3 == 0 else 0.0))

        def look(name):
            return act(lambda: setattr(a, "colors", ColorSettings.from_look(name, a.colors)))

        def tab(name):
            return act(lambda: setattr(a.settings, "sidebar_tab", name))

        def tap(key):
            return act(lambda: setattr(st, "snap", InputSnapshot(mouse=st.snap.mouse, keys={key}, pressed={key})))

        def path(n, p0, p1):
            return frames(n, lambda i, k: mouse(p0[0] + (p1[0] - p0[0]) * i / k, p0[1] + (p1[1] - p0[1]) * i / k))

        sx, sy = W * 0.501, H * 0.457             # the seahorse valley (-0.745 + 0.113i) on the home view
        return [
            ("start", frames(10)),
            ("home", settle(1.0)),
            ("aim", path(12, (W * 0.5, H * 0.7), (sx, sy))),
            ("zoom1", wheel_in(sx, sy, 3)),
            ("ref1", settle(0.5)),
            ("zoom2", wheel_in(sx, sy, 4)),
            ("ref3", settle(0.5)),
            ("zoom3", wheel_in(sx, sy, 4)),
            ("ref2", settle(0.8)),
            ("colour_tab", tab("Colour")),
            ("previews", settle(0.8)),
            ("ember", look("Ember")),
            ("e", settle(0.8)),
            ("silk", look("Silk")),
            ("s", settle(0.8)),
            ("midnight", look("Midnight")),
            ("m", settle(0.8)),
            ("explore_tab", tab("Explore")),
            ("julia_on", tap("j")),
            ("j0", settle(0.3)),
            ("julia_move", path(45, (W * 0.25, H * 0.30), (W * 0.45, H * 0.65))),
            ("j1", settle(0.8)),
            ("julia_off", tap("j")),
            ("end", settle(0.8)),
            ("stop", rec(False)),
        ]


def main():
    out = ROOT / "docs" / "images" / "tour.gif"
    with tempfile.TemporaryDirectory() as td:
        frames = Path(td) / "frames"
        frames.mkdir()
        Tour.frames_dir = frames
        selftest_mod.SelfTest = Tour             # the explorer creates its scripted session from this name
        run_app(size=SIZE, selftest=True, selftest_dir=str(Path(td) / "session"))
        ff = find_ffmpeg()
        flt = f"scale={GIF_WIDTH}:-1:flags=lanczos"
        pal = Path(td) / "palette.png"
        src = ["-framerate", str(FPS), "-i", str(frames / "%05d.png")]
        subprocess.run([ff, "-v", "error", "-y", *src, "-vf", f"{flt},palettegen=stats_mode=diff", str(pal)],
                       check=True)
        subprocess.run([ff, "-v", "error", "-y", *src, "-i", str(pal), "-lavfi",
                        f"{flt} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", str(out)], check=True)
        n = len(list(frames.glob("*.png")))
    print(f"{out.relative_to(ROOT)}: {n} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
