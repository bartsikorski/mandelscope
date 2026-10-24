"""Keyframe sequences and camera interpolation.

Between two keyframes the camera follows the one-parameter group of
similarity transforms connecting them ("log-spiral" interpolation):

    a     = (r1 / r0) * exp(i (theta1 - theta0))
    r(u)  = r0 * |a|^u,   theta(u) = theta0 + u (theta1 - theta0)
    c(u)  = c0 + (c1 - c0) (a^u - 1) / (a - 1)

so a zoom towards a target keeps the target fixed on screen and the
perceived zoom speed is constant.  The centre is evaluated relative to the
deeper keyframe, so it stays exact at any depth.

Timing: with ``smooth=True`` (default) the zoom depth log2(r) follows a
monotone cubic spline through all keyframes (no speed jumps at keyframes);
the segment parameter u is derived from it.  Segments without a zoom change
use the segment's easing curve.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import gmpy2
from gmpy2 import mpfr, mpc

from .mp import View

EASINGS = ["linear", "smooth", "ease-in", "ease-out"]


def ease(name: str, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    if name == "smooth":
        return u * u * (3.0 - 2.0 * u)
    if name == "ease-in":
        return u * u
    if name == "ease-out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    return u


@dataclass
class Keyframe:
    view: View
    time: float = 0.0
    ease: str = "linear"            # easing of the segment that starts here
    color: dict = field(default_factory=dict)   # colour parameters at this keyframe

    def to_dict(self):
        col = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.color.items()}
        return {"view": self.view.to_dict(), "time": self.time, "ease": self.ease, "color": col}

    @staticmethod
    def from_dict(d):
        col = {k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(d.get("color", {})).items()}
        return Keyframe(View.from_dict(d["view"]), float(d.get("time", 0.0)), d.get("ease", "linear"), col)


def interpolate_views(v0: View, v1: View, u: float) -> View:
    """Similarity-group interpolation between two views, u in [0, 1]."""
    dl = v1.log2r - v0.log2r
    dth = v1.angle - v0.angle
    log2r = v0.log2r + u * dl
    angle = v0.angle + u * dth
    prec = max(v0.precision, v1.precision, View(v0.re, v0.im, log2r).needed_precision())
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        dre = mpfr(v1.re, prec) - mpfr(v0.re, prec)
        dim = mpfr(v1.im, prec) - mpfr(v0.im, prec)
        delta = mpc(dre, dim)
        with gmpy2.context(gmpy2.get_context(), precision=128):
            la = mpc(mpfr(dl) * gmpy2.log(mpfr(2)), mpfr(dth))   # log a
            if abs(la) < mpfr("1e-12"):
                # pure translation: linear
                if u <= 0.5:
                    f0, base = mpc(u), 0
                else:
                    f0, base = mpc(u - 1.0), 1
            else:
                a = gmpy2.exp(la)
                au = gmpy2.exp(la * u)
                if dl <= 0:       # zooming in: relative to the end point (deeper)
                    f0, base = (au - a) / (a - 1), 1
                else:             # zooming out: relative to the start point (deeper)
                    f0, base = (au - 1) / (a - 1), 0
        f = mpc(f0, precision=prec)
        if base == 0:
            c = mpc(mpfr(v0.re, prec), mpfr(v0.im, prec)) + delta * f
        else:
            c = mpc(mpfr(v1.re, prec), mpfr(v1.im, prec)) + delta * f
        mi = int(round(math.exp((1 - u) * math.log(max(v0.max_iter, 1)) + u * math.log(max(v1.max_iter, 1)))))
        return View(c.real, c.imag, log2r, angle, mi)


def _pchip_slopes(x: list, y: list) -> list:
    """Fritsch-Carlson slopes of a monotone piecewise cubic Hermite interpolant."""
    n = len(x)
    if n < 2:
        return [0.0] * n
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] if h[i] > 0 else 0.0 for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _hermite(y0, y1, m0, m1, h, s):
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * m0 +
            (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * m1)


def _lerp_value(a, b, u):
    if isinstance(a, int) and isinstance(b, int):
        return a if u < 0.5 else b         # choices (method, mapping, pattern): no in-between
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a + (b - a) * u
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(x + (y - x) * u for x, y in zip(a, b))
    return a


class Sequence:
    def __init__(self, keyframes: list[Keyframe] | None = None, fps: float = 60.0, smooth: bool = True):
        self.keyframes: list[Keyframe] = list(keyframes or [])
        self.fps = fps
        self.smooth = smooth
        self.meta: dict = {}           # e.g. size requested by a script
        self.sort()

    def sort(self):
        self.keyframes.sort(key=lambda k: k.time)
        self._slopes = None

    def invalidate(self):
        self._slopes = None

    @property
    def duration(self) -> float:
        return self.keyframes[-1].time - self.keyframes[0].time if len(self.keyframes) > 1 else 0.0

    @property
    def start(self) -> float:
        return self.keyframes[0].time if self.keyframes else 0.0

    @property
    def end(self) -> float:
        return self.keyframes[-1].time if self.keyframes else 0.0

    def add(self, kf: Keyframe):
        self.keyframes.append(kf)
        self.sort()

    def segment(self, t: float) -> tuple[int, float]:
        ks = self.keyframes
        if t <= ks[0].time:
            return 0, 0.0
        for i in range(len(ks) - 1):
            if t <= ks[i + 1].time:
                span = ks[i + 1].time - ks[i].time
                s = (t - ks[i].time) / span if span > 0 else 1.0
                return i, s
        return len(ks) - 2, 1.0

    def _segment_u(self, i: int, s: float) -> float:
        ks = self.keyframes
        k0, k1 = ks[i], ks[i + 1]
        dl = k1.view.log2r - k0.view.log2r
        if not self.smooth or abs(dl) < 1e-9 or len(ks) < 3:
            return ease(k0.ease, s)
        if self._slopes is None:
            self._slopes = _pchip_slopes([k.time for k in ks], [k.view.log2r for k in ks])
        h = k1.time - k0.time
        if h <= 0:
            return 1.0
        m0, m1 = self._slopes[i], self._slopes[i + 1]
        # keep the eased endpoints of the whole sequence: first/last segment honour easing
        if k0.ease != "linear":
            return ease(k0.ease, s)
        l = _hermite(k0.view.log2r, k1.view.log2r, m0, m1, h, s)
        return min(1.0, max(0.0, (l - k0.view.log2r) / dl))

    def view_at(self, t: float) -> tuple[View, dict]:
        """Camera and colour parameters at time t (seconds)."""
        ks = self.keyframes
        if not ks:
            raise ValueError("empty sequence")
        if len(ks) == 1:
            return ks[0].view.copy(), dict(ks[0].color)
        i, s = self.segment(t)
        k0, k1 = ks[i], ks[i + 1]
        u = self._segment_u(i, s)
        v = interpolate_views(k0.view, k1.view, u)
        col = {}
        ce = ease(k0.ease, s)
        for key in set(k0.color) | set(k1.color):
            a = k0.color.get(key, k1.color.get(key))
            b = k1.color.get(key, a)
            col[key] = _lerp_value(a, b, ce)
        return v, col

    def deepest_in_segment(self, t: float) -> View:
        i, _ = self.segment(t)
        k0, k1 = self.keyframes[i], self.keyframes[min(i + 1, len(self.keyframes) - 1)]
        return k0.view if k0.view.log2r < k1.view.log2r else k1.view

    def frame_count(self, fps: float | None = None) -> int:
        fps = fps or self.fps
        return max(1, int(round(self.duration * fps)) + 1)

    def auto_time(self, octaves_per_second: float = 1.0, pan_seconds: float = 2.0,
                  start: float = 0.0, rotate_deg_per_second: float = 90.0):
        """Assign keyframe times for a constant zoom speed."""
        t = start
        for i, k in enumerate(self.keyframes):
            if i == 0:
                k.time = t
                continue
            prev = self.keyframes[i - 1]
            dz = abs(k.view.log2r - prev.view.log2r)
            dx, dy = prev.view.delta_in_pixels(k.view, 1000) if prev.view.log2r >= k.view.log2r \
                else k.view.delta_in_pixels(prev.view, 1000)
            pan = math.hypot(dx, dy) / 1000.0
            rot = abs(math.degrees(k.view.angle - prev.view.angle)) / max(rotate_deg_per_second, 1e-6)
            dur = max(dz / max(octaves_per_second, 1e-6), min(pan, 4.0) * pan_seconds, rot, 0.25)
            t += dur
            k.time = t
        self.invalidate()

    def retime(self, index: int, new_time: float, min_gap: float = 0.05):
        """Move one keyframe in time, kept between its neighbours."""
        ks = self.keyframes
        lo = ks[index - 1].time + min_gap if index > 0 else -1e9
        hi = ks[index + 1].time - min_gap if index + 1 < len(ks) else 1e9
        ks[index].time = min(max(new_time, lo), hi)
        self.invalidate()

    # ------------------------------------------------------------- io
    def to_dict(self):
        return {"version": 2, "fps": self.fps, "smooth": self.smooth, "meta": self.meta,
                "keyframes": [k.to_dict() for k in self.keyframes]}

    @staticmethod
    def from_dict(d) -> "Sequence":
        s = Sequence([Keyframe.from_dict(k) for k in d.get("keyframes", [])], float(d.get("fps", 60.0)),
                     bool(d.get("smooth", True)))
        s.meta = dict(d.get("meta", {}))
        return s

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @staticmethod
    def load(path) -> "Sequence":
        return Sequence.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class PathRecorder:
    """Records the live camera into a dense keyframe sequence."""

    def __init__(self, min_interval: float = 1.0 / 30.0):
        self.samples: list[tuple[float, View, dict]] = []
        self.min_interval = min_interval
        self.t0 = None

    def add(self, t: float, view: View, color: dict | None = None):
        if self.t0 is None:
            self.t0 = t
        rt = t - self.t0
        if self.samples and rt - self.samples[-1][0] < self.min_interval:
            return
        self.samples.append((rt, view.copy(), dict(color or {})))

    def to_sequence(self, fps: float = 60.0, smooth: int = 0) -> Sequence:
        ks = [Keyframe(v, t, "linear", c) for t, v, c in self.samples]
        if smooth > 0 and len(ks) > 2:
            # moving average of log2r and angle (centres are kept exact)
            l2 = [k.view.log2r for k in ks]
            an = [k.view.angle for k in ks]
            for i, k in enumerate(ks):
                a, b = max(0, i - smooth), min(len(ks), i + smooth + 1)
                k.view.log2r = sum(l2[a:b]) / (b - a)
                k.view.angle = sum(an[a:b]) / (b - a)
        return Sequence(ks, fps, smooth=False)
