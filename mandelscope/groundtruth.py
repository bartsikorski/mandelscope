"""Brute force arbitrary precision iteration - the oracle used by the tests."""
from __future__ import annotations

import math

import gmpy2
from gmpy2 import mpfr, mpc

from .mp import View


def sample_c(view: View, x: float, y: float, img_w: int, img_h: int, prec: int) -> mpc:
    """Complex coordinate of sample (x, y) (same mapping as the shader)."""
    log2h = view.pixel_log2(img_h)
    qx = x + 0.5 - img_w / 2.0
    qy = -(y + 0.5 - img_h / 2.0)
    ca, sa = math.cos(view.angle), math.sin(view.angle)
    rx = ca * qx - sa * qy
    ry = sa * qx + ca * qy
    he = int(math.floor(log2h)) + 1
    hm = 2.0 ** (log2h - he)
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        re = mpfr(view.re, prec) + gmpy2.mul_2exp(mpfr(rx * hm, prec), he)
        im = mpfr(view.im, prec) + gmpy2.mul_2exp(mpfr(ry * hm, prec), he)
        return mpc(re, im)


def iterate(c: mpc, max_iter: int, prec: int, bailout: float = 256.0):
    """Return (n, |z|^2 as float, smooth_frac) - n = 0xFFFFFFFF if no escape."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        z = mpc(0)
        B2 = mpfr(bailout * bailout)
        sq = gmpy2.square
        norm = gmpy2.norm
        for n in range(1, max_iter + 1):
            z = sq(z) + c
            r2 = norm(z)
            if r2 > B2:
                r2f = float(r2)
                frac = 1.0 - math.log2(0.5 * math.log2(r2f) / math.log2(bailout))
                return n, r2f, frac
        return 0xFFFFFFFF, float(norm(z)), 0.0


def iterate_ship(c: mpc, max_iter: int, prec: int, bailout: float = 256.0):
    """The Burning Ship as Mandelscope draws it (upright): x' = x^2 - y^2 + Re c,
    y' = 2|xy| - Im c.  Returns like iterate()."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        x = mpfr(0)
        y = mpfr(0)
        cr, ci = mpfr(c.real), mpfr(c.imag)
        B2 = mpfr(bailout * bailout)
        for n in range(1, max_iter + 1):
            x, y = x * x - y * y + cr, 2 * abs(x * y) - ci
            r2 = x * x + y * y
            if r2 > B2:
                r2f = float(r2)
                frac = 1.0 - math.log2(0.5 * math.log2(r2f) / math.log2(bailout))
                return n, r2f, frac
        return 0xFFFFFFFF, float(x * x + y * y), 0.0


def iterate_julia(z0: mpc, c, max_iter: int, prec: int, bailout: float = 256.0):
    """Julia set of c: (n, |z|^2, smooth_frac) for the orbit of z0, counted like the kernels
    (the first n >= 0 with |z_n| > bailout, reported as at least 1)."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        z = mpc(z0)
        cc = mpc(mpfr(c[0], prec), mpfr(c[1], prec)) if isinstance(c, (tuple, list)) else mpc(c)
        B2 = mpfr(bailout * bailout)
        for n in range(0, max_iter + 1):
            r2 = gmpy2.norm(z)
            if r2 > B2:
                r2f = float(r2)
                frac = 1.0 - math.log2(0.5 * math.log2(r2f) / math.log2(bailout))
                return max(n, 1), r2f, frac
            if n == max_iter:
                break
            z = gmpy2.square(z) + cc
        return 0xFFFFFFFF, float(gmpy2.norm(z)), 0.0
