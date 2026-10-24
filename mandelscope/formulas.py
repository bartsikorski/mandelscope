"""The fractals Mandelscope draws: the Mandelbrot set and the Burning Ship.

The Burning Ship (Michael Michelitsch and Otto E. Roessler, 1992) iterates
z -> (|Re z| + i |Im z|)^2 + c.  Pictures of it are almost always shown mirrored top to
bottom, so that the ship stands upright; Mandelscope draws it upright with the imaginary axis
pointing up as usual by iterating x' = x^2 - y^2 + Re c, y' = 2|xy| - Im c (the original map
at the conjugate point).  A point x + iy here is x - iy in programs that use the original
formula.
"""
from __future__ import annotations

import math

from gmpy2 import mpfr

from .mp import View

FORMULAS = ("mandelbrot", "ship")
FORMULA_NAMES = {"mandelbrot": "the Mandelbrot set", "ship": "the Burning Ship"}
FORMULA_LABELS = {"mandelbrot": "Mandelbrot set", "ship": "Burning Ship"}
_ALIASES = {"mandelbrot": "mandelbrot", "mandel": "mandelbrot", "m": "mandelbrot",
            "ship": "ship", "burningship": "ship", "burning ship": "ship", "burning_ship": "ship",
            "burning-ship": "ship"}


def parse_formula(name: str) -> str | None:
    return _ALIASES.get(" ".join(str(name).lower().replace('"', " ").split()))


def home_view(formula: str = "mandelbrot", fb=(16, 9)) -> View:
    """The whole set, fitted to the window shape."""
    W, H = fb
    if formula == "ship":
        # x from -2 to 1.04, y from -0.45 to 1.54 (upright)
        lr = math.log2(max(1.12, 1.62 * H / W)) if W > 0 and H > 0 else math.log2(1.12)
        return View(mpfr("-0.48", 128), mpfr("0.54", 128), lr, 0.0, 1000)
    v = View.home()
    if W > 0 and H > 0:
        v.log2r = math.log2(max(1.35, 1.45 * H / W))
    return v
