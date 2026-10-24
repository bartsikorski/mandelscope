"""Arbitrary precision helpers and the View (camera) model.

All coordinates of the complex plane that must survive deep zooms are kept as
gmpy2 ``mpfr`` values.  Scale is stored as ``log2r`` (log2 of the half height
of the view in complex units) so it never under/overflows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import gmpy2
from gmpy2 import mpfr

# --------------------------------------------------------------------------
# precision helpers
# --------------------------------------------------------------------------

LOG10_2 = math.log10(2.0)


def bits_for_log2(log2_h: float, extra: int = 64) -> int:
    """MPFR precision (bits) needed to resolve features of size 2**log2_h."""
    return max(64, int(math.ceil(-log2_h)) + extra)


def mpf(x, prec: int) -> mpfr:
    """Create an mpfr with an explicit precision from str/float/int/mpfr."""
    return mpfr(x, prec)


def to_str(x: mpfr, prec_bits: int | None = None) -> str:
    """Decimal string with enough digits to round-trip the given precision."""
    if prec_bits is None:
        prec_bits = x.precision
    digits = int(math.ceil(prec_bits * LOG10_2)) + 3
    return format(x, f".{digits}e")


def frexp_mp(x: mpfr) -> tuple[float, int]:
    """Return (mantissa_float, exponent) with x = m * 2**e, 0.5 <= |m| < 1."""
    if x == 0:
        return 0.0, 0
    e, m = gmpy2.frexp(x)
    return float(m), int(e)


def mp_to_fe(x: mpfr) -> tuple[float, int]:
    return frexp_mp(x)


def mp_ratio_to_float(num: mpfr, log2_den: float) -> float:
    """num / 2**log2_den as a float64, robust to huge exponent ranges."""
    if num == 0:
        return 0.0
    m, e = frexp_mp(num)
    return m * 2.0 ** (e - log2_den) if abs(e - log2_den) < 1000 else (
        0.0 if e - log2_den < 0 else math.copysign(math.inf, m))


def log2_abs_mp(x: mpfr) -> float:
    if x == 0:
        return -math.inf
    m, e = frexp_mp(x)
    return e + math.log2(abs(m))


# --------------------------------------------------------------------------
# View
# --------------------------------------------------------------------------

@dataclass
class View:
    """Camera for the complex plane.

    re, im  : centre (mpfr, arbitrary precision)
    log2r   : log2 of the half *height* of the view in complex units
    angle   : rotation in radians (counter clockwise)
    max_iter: iteration limit
    """
    re: mpfr
    im: mpfr
    log2r: float = 1.0
    angle: float = 0.0
    max_iter: int = 1000

    # ------------------------------------------------------------ factories
    @staticmethod
    def home() -> "View":
        return View(mpfr("-0.75", 128), mpfr("0", 128), math.log2(1.35), 0.0, 1000)

    @staticmethod
    def from_strings(re: str, im: str, log2r: float, angle: float = 0.0,
                     max_iter: int = 1000) -> "View":
        prec = bits_for_log2(log2r - 12)
        # make sure the decimal string itself is not truncated
        prec = max(prec, int(len(re.strip()) / LOG10_2) + 16, int(len(im.strip()) / LOG10_2) + 16)
        return View(mpfr(re.strip(), prec), mpfr(im.strip(), prec), float(log2r), float(angle), int(max_iter))

    # --------------------------------------------------------------- basics
    @property
    def precision(self) -> int:
        return max(self.re.precision, self.im.precision)

    def needed_precision(self, height_px: int = 4096) -> int:
        log2h = self.log2r + 1 - math.log2(max(height_px, 1))
        return bits_for_log2(log2h, extra=48)

    def ensure_precision(self, height_px: int = 4096) -> "View":
        p = self.needed_precision(height_px)
        if self.precision < p:
            self.re = mpfr(self.re, p)
            self.im = mpfr(self.im, p)
        return self

    def copy(self) -> "View":
        return replace(self, re=mpfr(self.re, self.re.precision), im=mpfr(self.im, self.im.precision))

    def pixel_log2(self, height_px: int) -> float:
        """log2 of the spacing between samples for a grid of the given height."""
        return self.log2r + 1.0 - math.log2(height_px)

    # ----------------------------------------------------------- transforms
    def offset(self, dx: float, dy: float, height_px: int) -> "View":
        """Move by (dx, dy) screen pixels (x right, y down) - returns new view."""
        v = self.copy().ensure_precision(height_px)
        p = v.precision
        log2h = self.pixel_log2(height_px)
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        # screen -> complex: q = (dx, -dy) rotated by angle, times h
        qx = ca * dx - sa * (-dy)
        qy = sa * dx + ca * (-dy)
        hm, he = math.frexp(2.0 ** (log2h - math.floor(log2h)))
        he += int(math.floor(log2h))
        with gmpy2.context(gmpy2.get_context(), precision=p):
            v.re = v.re + gmpy2.mul_2exp(mpfr(qx * hm, p), he)
            v.im = v.im + gmpy2.mul_2exp(mpfr(qy * hm, p), he)
        return v

    def zoom_at(self, sx: float, sy: float, factor_log2: float, width_px: int, height_px: int) -> "View":
        """Zoom by 2**factor_log2 (positive = in) keeping screen point (sx,sy) fixed."""
        dx = sx - width_px / 2.0
        dy = sy - height_px / 2.0
        s = 1.0 - 2.0 ** (-factor_log2)
        v = self.offset(dx * s, dy * s, height_px)
        v.log2r = self.log2r - factor_log2
        return v.ensure_precision(height_px)

    def rotate_by(self, dangle: float) -> "View":
        v = self.copy()
        v.angle = self.angle + dangle
        return v

    def screen_to_c(self, sx: float, sy: float, width_px: int, height_px: int) -> tuple[mpfr, mpfr]:
        v = self.offset(sx - width_px / 2.0, sy - height_px / 2.0, height_px)
        return v.re, v.im

    def delta_in_pixels(self, other: "View", height_px: int) -> tuple[float, float]:
        """(other.centre - self.centre) expressed in *self* screen pixels (x right, y down)."""
        p = max(self.precision, other.precision)
        dre = mpfr(other.re, p) - mpfr(self.re, p)
        dim = mpfr(other.im, p) - mpfr(self.im, p)
        log2h = self.pixel_log2(height_px)
        fx = mp_ratio_to_float(dre, log2h)
        fy = mp_ratio_to_float(dim, log2h)
        ca, sa = math.cos(-self.angle), math.sin(-self.angle)
        qx = ca * fx - sa * fy
        qy = sa * fx + ca * fy
        return qx, -qy

    # -------------------------------------------------------- serialisation
    def zoom_magnification_log10(self) -> float:
        """log10 of magnification relative to a view of half height 2."""
        return (1.0 - self.log2r) * LOG10_2

    def to_dict(self) -> dict:
        p = max(self.precision, self.needed_precision())
        return {
            "re": to_str(mpfr(self.re, p), p),
            "im": to_str(mpfr(self.im, p), p),
            "log2r": self.log2r,
            "angle": self.angle,
            "max_iter": self.max_iter,
            "zoom": f"1e{self.zoom_magnification_log10():.3f}",
        }

    @staticmethod
    def from_dict(d: dict) -> "View":
        if "log2r" in d:
            log2r = float(d["log2r"])
        elif "radius" in d:
            log2r = log2_abs_mp(mpfr(str(d["radius"]), 64))
        elif "zoom" in d:
            z = str(d["zoom"]).lower()
            if "e" in z:
                mant, ex = z.split("e")
                l10 = float(ex) + math.log10(float(mant) if mant else 1.0)
            else:
                l10 = math.log10(float(z))
            log2r = 1.0 - l10 / LOG10_2
        else:
            log2r = 1.0
        return View.from_strings(str(d["re"]), str(d["im"]), log2r,
                                 float(d.get("angle", 0.0)), int(d.get("max_iter", 1000)))
