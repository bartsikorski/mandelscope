"""Free ride: an endless, slow camera flight from minibrot to minibrot.

The ride flies to a minibrot near the view, rests there, then dives into one of the copies of
the Mandelbrot set's own minibrots embedded in it, and so on.  When the next minibrot would
need too many iterations it flies back out to the whole set and starts a new branch.

How the next target is found (the tuning of the Mandelbrot set): a minibrot with nucleus n,
period p and complex size s (nucleus.size_estimate) is close to the image of the whole set
under w -> n + s w.  The whole set's minibrot with nucleus w and period q therefore has a
copy in it near n + s w, of period p q and size about s * size(w).  Newton's method for that
period, started there, finds its nucleus; it is accepted only when it lies where the copy
should be and has about the expected size.  Which of LOCAL_MINIS is taken (and whether its
mirror image) is random, so every ride takes another way down.

The searches run in the background worker (kind "ride"), one leg ahead of the camera: while
the camera flies to a minibrot the next one is already being looked for.
"""
from __future__ import annotations

import math
import random

import gmpy2
from gmpy2 import mpc, mpfr

from .mp import log2_abs_mp, to_str
from .nucleus import find_minibrot, newton_nucleus, size_estimate

# Minibrots of the whole set (upper half plane; the mirror images are used too):
# nucleus re, im, period.  Checked with tests/test_ride.py.
LOCAL_MINIS = [
    ("-1.754877666246693", "0", 3),          # the big one on the antenna
    ("-1.940799806529", "0", 4),
    ("-1.860782522205", "0", 5),
    ("-1.625413725123", "0", 5),
    ("-0.156520166834", "1.032247108923", 4),  # above the top bulb
    ("-0.074470451380", "0.970542136899", 8),
    ("-0.616532685850", "0.612431354461", 9),
    ("-1.206410020826", "0.316422238084", 10),
    ("-1.041469216353", "0.336537138761", 11),
    ("0.450877055531", "0.383884743455", 11),
    ("-0.405854510228", "0.647679992222", 12),
    ("-0.603421923426", "0.616048562056", 13),
    ("-0.066197201731", "0.894523591233", 13),
    ("-0.655222343777", "0.488679421938", 14),
    ("-0.246007329560", "0.728202714453", 14),
    ("-0.661362520583", "0.409289248017", 15),
    ("-0.107114623902", "0.912231331794", 15),
]

MAX_PERIOD = 120_000         # deeper minibrots: fly back out (the iteration limit gets large)
MAIN_SET = {"re": "0", "im": "0", "period": 1, "log2_size": 0.0, "angle": 0.0, "prec": 64}


def _mini(c: mpc, period: int, prec: int) -> dict | None:
    s = size_estimate(c, period, prec)
    asz = abs(s)
    if asz == 0 or not gmpy2.is_finite(asz):
        return None
    return {"re": to_str(c.real, prec), "im": to_str(c.imag, prec), "period": int(period),
            "log2_size": log2_abs_mp(asz), "angle": math.atan2(float(s.imag / asz), float(s.real / asz)),
            "prec": int(prec)}


def parent_size(parent: dict) -> complex:
    """Direction of a minibrot's complex size s (s / |s|)."""
    a = float(parent.get("angle", 0.0))
    return complex(math.cos(a), math.sin(a))


def embedded(parent: dict, local: tuple, mirror: bool, cancelled=None) -> dict | None:
    """The copy of the whole set's minibrot `local` (re, im, period) inside `parent`."""
    lre, lim, q = local
    p = int(parent["period"])
    period = p * int(q)
    l2s = float(parent["log2_size"])
    # size of the local minibrot in the whole set (cheap: low period, double-ish precision)
    lprec = 128
    with gmpy2.context(gmpy2.get_context(), precision=lprec):
        w = mpc(mpfr(lre, lprec), (-1 if mirror else 1) * mpfr(lim, lprec))
        ls = size_estimate(w, int(q), lprec)
        l2_local = log2_abs_mp(abs(ls))
    l2_expect = l2s + l2_local
    prec = max(int(parent.get("prec", 64)), int(-l2_expect * 2) + 128)
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        n = mpc(mpfr(parent["re"], prec), mpfr(parent["im"], prec))
        k = int(math.floor(l2s))
        s = gmpy2.mul_2exp(mpc(parent_size(parent) * 2.0 ** (l2s - k), precision=prec), k)
        w = mpc(mpfr(lre, prec), (-1 if mirror else 1) * mpfr(lim, prec))
        guess = n + s * w
        c = newton_nucleus(guess, period, prec, cancelled=cancelled)
        if c is None:
            return None
        # where the copy should be: the map w -> n + s w is only approximate away from the
        # nucleus (off by up to ~0.2 in w at the antenna's tip); a copy of another minibrot
        # of the same period would be much further off, or have another size
        err = log2_abs_mp(abs(c - guess)) - l2s
        if err > -1.5:
            return None
    out = _mini(c, period, prec)
    if out is None or abs(out["log2_size"] - l2_expect) > 1.5:
        return None
    return out


def next_target(parent: dict, rng: random.Random, max_period: int = MAX_PERIOD, cancelled=None,
                tries: int = 10) -> dict | None:
    """A minibrot embedded in `parent` (random choice), or None (too deep / none found)."""
    p = int(parent["period"])
    options = [(m, mirror) for m in LOCAL_MINIS for mirror in ((False, True) if m[1] != "0" else (False,))
               if p * m[2] <= max_period]
    rng.shuffle(options)
    for local, mirror in options[:tries]:
        if cancelled is not None and cancelled():
            return None
        r = embedded(parent, local, mirror, cancelled)
        if r is not None:
            return r
    return None


def nearby(re: str, im: str, log2r: float, cancelled=None, max_period: int = MAX_PERIOD) -> dict | None:
    """The minibrot nearest the view (centre re, im; half height 2**log2r): the lowest period
    in a ball around the centre, a smaller ball when none is found.  Its nucleus must be on
    the screen (or close to it)."""
    if log2r >= -1.0:
        return dict(MAIN_SET)
    for k in (0.0, -3.0, -6.0):
        lr = log2r + k
        r = find_minibrot(re, im, lr, max_period=max_period, cancelled=cancelled)
        if r is None or not math.isfinite(r["log2_size"]):
            continue
        prec = r["prec"]
        with gmpy2.context(gmpy2.get_context(), precision=prec):
            d = abs(mpc(mpfr(r["re"], prec), mpfr(r["im"], prec)) - mpc(mpfr(re, prec), mpfr(im, prec)))
            if log2_abs_mp(d) > log2r + 1.0:
                continue
        r = dict(r)
        r["re"], r["im"] = to_str(r["re"], prec), to_str(r["im"], prec)
        return r
    return None


def plan(job: dict, cancelled=None) -> dict:
    """The worker's "ride" job.  job: either {"parent": mini} (the next minibrot in it) or
    {"re", "im", "log2r"} (the first one, near the view).  Returns {"target": mini or None,
    "parent": the minibrot the target lies in (or None)}."""
    rng = random.Random(job.get("seed"))
    maxp = int(job.get("max_period", MAX_PERIOD))
    parent = job.get("parent")
    if parent is None:
        near = nearby(job["re"], job["im"], float(job["log2r"]), cancelled, maxp)
        if near is None:
            return {"target": None, "parent": None}
        if near["log2_size"] + math.log2(1.35) < float(job["log2r"]) - 2.0:
            return {"target": near, "parent": None}       # small: fly there first
        parent = near                                     # the view shows it already: dive in
    return {"target": next_target(parent, rng, maxp, cancelled), "parent": parent}


# ------------------------------------------------------------------ camera timing
def leg_progress(t: float, dur: float, ramp: float) -> float:
    """Fraction u of a leg after t seconds: the zoom speed rises from rest over `ramp`
    seconds, stays constant, and falls back to rest over the last `ramp` seconds."""
    if dur <= 0:
        return 1.0
    t = min(max(t, 0.0), dur)
    r = min(ramp, 0.5 * dur)
    v = 1.0 / (dur - r)                       # top speed (u per second): area under the trapezoid = 1
    if t < r:
        return 0.5 * v * t * t / r
    if t > dur - r:
        e = dur - t
        return 1.0 - 0.5 * v * e * e / r
    return v * (t - 0.5 * r)


def leg_duration(octaves: float, pixels: float, speed: float, ramp: float) -> float:
    """Seconds for a leg of `octaves` zoom (either way) and a sideways move of `pixels`
    window heights: `speed` octaves per second at the top speed (ramps count half)."""
    oct_ = max(abs(octaves), 0.5 * abs(pixels))
    return max(4.0, oct_ / max(speed, 1e-3) + ramp)


def framing(mini: dict):
    """(re, im, log2r, angle) strings/floats of the view that shows `mini` the way the home
    view shows the whole set: centred on its body (w = -0.75), upright."""
    prec = max(64, int(mini.get("prec", 64)))
    l2s = float(mini["log2_size"])
    a = float(mini.get("angle", 0.0))
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        n = mpc(mpfr(mini["re"], prec), mpfr(mini["im"], prec))
        k = int(math.floor(l2s))
        s = gmpy2.mul_2exp(mpc(complex(math.cos(a), math.sin(a)) * 2.0 ** (l2s - k), precision=prec), k)
        c = n + s * mpfr(-0.75, prec)
        return c.real, c.imag, l2s + math.log2(1.35), a
