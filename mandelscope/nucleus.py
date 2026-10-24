"""Minibrot (periodic nucleus) finder: ball-period detection + Newton + size.

Used for:
  * "Find minibrot" in the UI (zoom/centre on the nearest nucleus) -
    a reference orbit at a nucleus is periodic, so the reference computation
    becomes O(period) and BLA skips are maximal;
  * generating deep test locations.
"""
from __future__ import annotations

import math

import gmpy2
from gmpy2 import mpfr, mpc

from .mp import log2_abs_mp


def _ship_sq(z: mpc) -> mpc:
    """(|x| + i|y|)^2: z^2 when x and y have the same sign, else conj(z^2)."""
    w = z * z
    return w.conjugate() if (z.real < 0) != (z.imag < 0) else w


def find_period(c: mpc, radius: mpfr, max_period: int, prec: int, cancelled=None,
                formula: str = "mandelbrot") -> int:
    """Lowest period p such that the ball B(c, radius) iterated p times contains 0.  (The
    Burning Ship: the same bound, as folding into the first quadrant moves no two points
    further apart.)"""
    ship = formula == "ship"
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        z = mpc(0)
        rr = mpfr(radius, 64)
        rho = mpfr(0, 64)
        cc = c.conjugate() if ship else c
        for n in range(1, max_period + 1):
            az = abs(mpc(z, precision=64))
            rho = rho * (2 * az + rho) + rr
            z = (_ship_sq(z) if ship else z * z) + cc
            if abs(mpc(z, precision=64)) < rho:
                return n
            if gmpy2.norm(z) > 1e8:
                return 0
            if cancelled is not None and (n & 4095) == 0 and cancelled():
                return 0
    return 0


def newton_nucleus(c0: mpc, period: int, prec: int, max_steps: int = 64, progress=None,
                   cancelled=None) -> mpc | None:
    """Newton iteration for z_p(c) = 0 starting at c0."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = mpc(c0)
        eps_log2 = -(prec - 16)
        for step in range(max_steps):
            z = mpc(0)
            dz = mpc(0)
            for i in range(period):
                dz = 2 * z * dz + 1
                z = z * z + c
                if cancelled is not None and (i & 8191) == 0 and cancelled():
                    return None
            if dz == 0:
                return None
            delta = z / dz
            c = c - delta
            if progress is not None:
                progress(step / max_steps)
            ad = abs(delta)
            ac = max(abs(c), mpfr(1e-300))
            if ad == 0 or log2_abs_mp(ad) - log2_abs_mp(ac) < eps_log2:
                return c
            if not gmpy2.is_finite(c.real) or abs(c) > 4:
                return None
        return c


def newton_ship(c0: mpc, period: int, prec: int, max_steps: int = 64, progress=None,
                cancelled=None):
    """Burning Ship (as drawn: z' = (|x| + i|y|)^2 + conj c): Newton for z_p(c) = 0 with the
    real 2x2 Jacobian dz_p/dc.  Returns (nucleus, Jacobian [[a, b], [c, d]] at it) or None."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = mpc(c0)
        eps_log2 = -(prec - 16)
        J = None
        for step in range(max_steps):
            x = y = mpfr(0)
            a, b, cc_, d = mpfr(0), mpfr(0), mpfr(0), mpfr(0)   # dz/dc, rows (x, y), columns (cx, cy)
            cr, ci = c.real, c.imag
            for i in range(period):
                s = 1 if (x >= 0) == (y >= 0) else -1
                # Jacobian of (x^2 - y^2, 2|xy|): [[2x, -2y], [2sy, 2sx]]; then + diag(1, -1)
                a, b, cc_, d = (2 * x * a - 2 * y * cc_ + 1, 2 * x * b - 2 * y * d,
                                2 * s * (y * a + x * cc_), 2 * s * (y * b + x * d) - 1)
                x, y = x * x - y * y + cr, 2 * abs(x * y) - ci
                if cancelled is not None and (i & 8191) == 0 and cancelled():
                    return None
            det = a * d - b * cc_
            if det == 0:
                return None
            dx = (d * x - b * y) / det
            dy = (-cc_ * x + a * y) / det
            c = mpc(c.real - dx, c.imag - dy)
            J = ((a, b), (cc_, d))
            if progress is not None:
                progress(step / max_steps)
            ad = abs(mpc(dx, dy))
            ac = max(abs(c), mpfr(1e-300))
            if ad == 0 or log2_abs_mp(ad) - log2_abs_mp(ac) < eps_log2:
                return c, J
            if not gmpy2.is_finite(c.real) or abs(c) > 4:
                return None
        return None                     # (no convergence: the fold makes z_p only piecewise smooth)


def size_estimate_ship(c: mpc, period: int, prec: int, J) -> float:
    """Rough log2 size of a mini Burning Ship: like the Mandelbrot set's 1 / (b l^2) with the
    multiplier l = prod 2|z_i| and sqrt|det J| for |l b| (the minis are skewed)."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        z = mpc(0)
        cc = c.conjugate()
        log2l = 0.0
        for _ in range(1, period):
            z = _ship_sq(z) + cc
            az = abs(z)
            if az == 0:
                return -1e9
            log2l += 1.0 + log2_abs_mp(az)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        return -(0.5 * log2_abs_mp(abs(det)) + log2l)


def size_estimate(c: mpc, period: int, prec: int) -> mpc:
    """Complex size/orientation of the minibrot with nucleus c (Claude Heiland-Allen)."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        z = mpc(0)
        lo = 64
        l = mpc(1, precision=lo)
        b = mpc(1, precision=lo)
        for _ in range(1, period):
            z = z * z + c
            l = mpc(2 * mpc(z, precision=lo) * l, precision=lo)
            b = mpc(b + 1 / l, precision=lo)
        return 1 / (b * l * l)


def find_minibrot(re: str | mpfr, im: str | mpfr, log2_radius: float, max_period: int = 200000,
                  prec: int | None = None, cancelled=None, progress=None, formula: str = "mandelbrot"):
    """Find a nucleus near (re, im) within a ball of radius 2**log2_radius (formula "ship": a
    mini Burning Ship).

    Returns dict(re, im, period, log2_size, angle) or None."""
    if prec is None:
        prec = max(64, int(-log2_radius) + 64)
    prec_n = prec * 2 + 64
    with gmpy2.context(gmpy2.get_context(), precision=prec_n):
        c = mpc(mpfr(re, prec_n), mpfr(im, prec_n))
        radius = gmpy2.mul_2exp(mpfr(1, 64), int(math.floor(log2_radius)))
        p = find_period(c, radius, max_period, prec, cancelled, formula=formula)
        if p <= 0:
            return None
        nprec = max(prec_n, int(-log2_radius * 2) + 128)
        if formula == "ship":
            out = newton_ship(c, p, nprec, cancelled=cancelled, progress=progress)
            if out is None:
                return None
            cn, J = out
            if abs(cn - c) > 64 * radius:          # converged to a far away nucleus
                return None
            return {"re": cn.real, "im": cn.imag, "period": p,
                    "log2_size": size_estimate_ship(cn, p, nprec, J), "angle": 0.0, "prec": nprec}
        cn = newton_nucleus(c, p, nprec, cancelled=cancelled, progress=progress)
        if cn is None:
            return None
        s = size_estimate(cn, p, nprec)
        asz = abs(s)
        if asz == 0:
            return None
        return {
            "re": cn.real, "im": cn.imag, "period": p,
            "log2_size": log2_abs_mp(asz),
            "angle": math.atan2(float(s.imag / asz), float(s.real / asz)),
            "prec": nprec,
        }


def misiurewicz(c0: mpc, preperiod: int, period: int, prec: int, max_steps: int = 80) -> mpc | None:
    """Newton for a pre-periodic (Misiurewicz) point: z_{k+p}(c) = z_k(c)."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = mpc(c0)
        for _ in range(max_steps):
            z = mpc(0); dz = mpc(0)
            zk = dzk = None
            for i in range(1, preperiod + period + 1):
                dz = 2 * z * dz + 1
                z = z * z + c
                if i == preperiod:
                    zk, dzk = z, dz
            f = z - zk
            df = dz - dzk
            if df == 0:
                return None
            delta = f / df
            c = c - delta
            if delta == 0 or log2_abs_mp(abs(delta)) < -(prec - 16):
                return c
        return c
