"""Offline (exact, deterministic) rendering of stills and keyframe sequences."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
from gmpy2 import mpfr

from .mp import View
from .reference import ReferenceOrbit, compute_reference
from .bla import BLATable, build_bla
from .renderer import (GPUEngine, Field, GPURef, GPUBla, FieldParams, ColorSettings,
                       AutoContrast, view_params, julia_params, julia_min_log2r)
from .sequence import Sequence
from .video import FfmpegWriter, PngSaver


# Exports use the explorer's scheduling: the default per-sample budget (long-running samples
# are regrouped into compact pending lists instead of idling the other lanes of their warp)
# and dispatches of ~15 ms.  (v2 used an 8192-step budget and 250 ms dispatches; with those,
# exported frames took 2-4x longer than the same picture in the explorer on an RTX 5060 Ti.)
DISPATCH_TARGET = 0.015


def default_eps(double: bool) -> float:
    return -45.0 if double else -24.0


class ReferenceCache:
    """Current reference orbit + BLA table (+ their GPU copies) and reuse rules."""

    def __init__(self, engine: GPUEngine):
        self.engine = engine
        self.ref: ReferenceOrbit | None = None
        self.ref_re = None
        self.ref_im = None
        self.bla: BLATable | None = None
        self._gref: dict = {}
        self._gbla: dict = {}
        self.resumable = False
        self.formula = "mandelbrot"          # references of another formula are not usable

    def _wrong_formula(self, r) -> bool:
        return getattr(r, "formula", "mandelbrot") != self.formula

    # ------------------------------------------------------------ queries
    def ref_usable(self, view: View, img_w: int, img_h: int, max_iter: int | None = None) -> bool:
        r = self.ref
        if r is None or self._wrong_formula(r):
            return False
        if r.prec < view.needed_precision(img_h):
            return False
        dx, dy = view.delta_in_pixels(View(self.ref_re, self.ref_im, view.log2r, view.angle), img_h)
        if math.hypot(dx, dy) > 1.5 * max(img_w, img_h):
            return False
        mi = view.max_iter if max_iter is None else max_iter
        if not r.complete_for(mi):
            return False
        return True

    def ref_status(self, view: View, img_w: int, img_h: int) -> str:
        """'ok', 'iterations' (usable with a lower iteration cap) or 'missing'."""
        r = self.ref
        if r is None or r.prec < view.needed_precision(img_h) or self._wrong_formula(r):
            return "missing"
        dx, dy = view.delta_in_pixels(View(self.ref_re, self.ref_im, view.log2r, view.angle), img_h)
        if math.hypot(dx, dy) > 1.5 * max(img_w, img_h):
            return "missing"
        return "ok" if r.complete_for(view.max_iter) else "iterations"

    def bla_usable(self, fp: FieldParams, log2_eps: float) -> bool:
        b = self.bla
        if self.formula == "ship":
            return True                      # (the Burning Ship has no BLA tables)
        if b is None or self.ref is None:
            return False
        if b.ref_id != self.ref.ref_id or abs(b.log2_eps - log2_eps) > 1e-9:
            return False
        return fp.log2_cmax <= b.log2_cmax + 1e-9 and b.log2_cmax - fp.log2_cmax < 6.0

    # ------------------------------------------------------------ updates
    def set_reference(self, ref: ReferenceOrbit, bla: BLATable | None = None):
        self.release_gpu()
        self.ref = ref
        prec = max(ref.prec, 64)
        self.ref_re = mpfr(ref.re, prec)
        self.ref_im = mpfr(ref.im, prec)
        self.bla = bla

    def set_bla(self, bla: BLATable):
        for g in self._gbla.values():
            g.release()
        self._gbla.clear()
        self.bla = bla
        self.gpu_serial = getattr(self, "gpu_serial", 0) + 1

    def gpu(self, double: bool) -> tuple[GPURef, GPUBla]:
        if double not in self._gref:
            self._gref[double] = GPURef(self.engine.ctx, self.ref, double)
        if double not in self._gbla:
            self._gbla[double] = GPUBla(self.engine.ctx, self.bla, double)
        return self._gref[double], self._gbla[double]

    def release_gpu(self):
        for g in list(self._gref.values()) + list(self._gbla.values()):
            g.release()
        self._gref.clear()
        self._gbla.clear()
        self.gpu_serial = getattr(self, "gpu_serial", 0) + 1   # GPU objects handed out are now invalid

    def reference_request(self, view: View, img_w: int, img_h: int, anchor: View | None = None,
                          extra_prec: int = 32) -> dict:
        """Parameters for a new reference orbit for this view."""
        re, im = view.re, view.im
        prec = view.needed_precision(img_h) + extra_prec
        if anchor is not None:
            dx, dy = view.delta_in_pixels(anchor, img_h)
            if math.hypot(dx, dy) < 0.75 * max(img_w, img_h):
                re, im = anchor.re, anchor.im
                prec = max(prec, anchor.needed_precision(img_h) + extra_prec)
        p = max(prec, re.precision, im.precision)
        from .mp import to_str
        return dict(re=to_str(mpfr(re, p), p), im=to_str(mpfr(im, p), p), prec=prec,
                    max_iter=view.max_iter, formula=self.formula)


def exact_str(x) -> str:
    """Exact decimal of a float (a Julia parameter c stored as float64): the deep-zoom
    references must use the same c as the direct kernel, not a 17-digit rounding of it."""
    if isinstance(x, str):
        return x
    from decimal import Decimal
    return str(Decimal(float(x)))


JULIA_NO_C = -10000.0     # BLA tables of Julia references: no c term (log2 of the c range)


def compute_julia_references(req: dict, log2_eps: float, ref_b: ReferenceOrbit | None = None,
                             progress=None, cancelled=None):
    """Reference A (orbit of the view's point z0 under z^2 + c) and B (orbit of the critical
    point 0, i.e. the Mandelbrot-set reference of c), with their BLA tables.  `ref_b` is reused
    when given (it depends only on c and the precision)."""
    if ref_b is None:
        ref_b = compute_reference(req["c_re"], req["c_im"], req["prec"], req["max_iter"],
                                  cancelled=cancelled, progress=(lambda f: progress(0.5 * f)) if progress else None)
        if ref_b is None:
            return None
    ref_a = compute_reference(req["c_re"], req["c_im"], req["prec"], req["max_iter"], z0=(req["re"], req["im"]),
                              cancelled=cancelled, progress=(lambda f: progress(0.5 + 0.5 * f)) if progress else None)
    if ref_a is None:
        return None
    bla_a = build_bla(ref_a.zm, ref_a.ze, JULIA_NO_C, log2_eps)
    bla_b = build_bla(ref_b.zm, ref_b.ze, JULIA_NO_C, log2_eps)
    return ref_a, ref_b, bla_a, bla_b


class JuliaReferences:
    """Deep Julia sets: reference A (view's point) + B (critical orbit) and their tables, with
    the same queries as ReferenceCache (`ref` is A)."""

    def __init__(self, engine: GPUEngine):
        self.engine = engine
        self.c = None                 # (re, im) exact decimal strings
        self.ref = self.ref_b = None
        self.bla = self.bla_a = None
        self.ref_re = self.ref_im = None
        self.log2_eps = None
        self._g: dict = {}

    def select(self, c):
        """Use the Julia set of c (float pair or strings); forgets references of another c."""
        cs = (exact_str(c[0]), exact_str(c[1]))
        if cs != self.c:
            self.release_gpu()
            self.c = cs
            self.ref = self.ref_b = self.bla = self.bla_a = None
            self.ref_re = self.ref_im = None
        return cs

    def ref_status(self, view: View, img_w: int, img_h: int) -> str:
        r = self.ref
        if r is None or self.ref_b is None or r.prec < view.needed_precision(img_h):
            return "missing"
        dx, dy = view.delta_in_pixels(View(self.ref_re, self.ref_im, view.log2r, view.angle), img_h)
        if math.hypot(dx, dy) > 1.5 * max(img_w, img_h):
            return "missing"
        return "ok" if (r.complete_for(view.max_iter) and self.ref_b.complete_for(view.max_iter)) else "iterations"

    def ref_usable(self, view: View, img_w: int, img_h: int, max_iter=None) -> bool:
        if max_iter is not None and max_iter != view.max_iter:
            view = view.copy()
            view.max_iter = max_iter
        return self.ref_status(view, img_w, img_h) == "ok"

    def bla_usable(self, fp: FieldParams, log2_eps: float) -> bool:
        return self.bla is not None and self.log2_eps is not None and abs(self.log2_eps - log2_eps) < 1e-9

    def set_references(self, ref_a, ref_b, bla_a, bla_b, log2_eps: float):
        self.release_gpu()
        self.ref, self.ref_b, self.bla_a, self.bla = ref_a, ref_b, bla_a, bla_b
        prec = max(ref_a.prec, 64)
        self.ref_re = mpfr(ref_a.z0[0], prec)
        self.ref_im = mpfr(ref_a.z0[1], prec)
        self.log2_eps = log2_eps

    def set_bla(self, bla):            # (ReferenceCache interface; Julia tables come with the references)
        pass

    def gpu(self, double: bool):
        if double not in self._g:
            self._g[double] = (GPURef.pair(self.engine.ctx, self.ref, self.ref_b, double),
                               GPUBla(self.engine.ctx, self.bla, double, bla_a=self.bla_a))
        return self._g[double]

    def release_gpu(self):
        for gr, gb in self._g.values():
            gr.release()
            gb.release()
        self._g.clear()
        self.gpu_serial = getattr(self, "gpu_serial", 0) + 1

    def reference_request(self, view: View, img_w: int, img_h: int, anchor: View | None = None,
                          extra_prec: int = 32) -> dict:
        """Parameters for new references for this view (c from select())."""
        from .mp import to_str
        re, im = view.re, view.im
        prec = view.needed_precision(img_h) + extra_prec
        if anchor is not None:
            dx, dy = view.delta_in_pixels(anchor, img_h)
            if math.hypot(dx, dy) < 0.75 * max(img_w, img_h):
                re, im = anchor.re, anchor.im
                prec = max(prec, anchor.needed_precision(img_h) + extra_prec)
        p = max(prec, re.precision, im.precision)
        return dict(c_re=self.c[0], c_im=self.c[1], re=to_str(mpfr(re, p), p), im=to_str(mpfr(im, p), p),
                    prec=prec, max_iter=view.max_iter)


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    ss: int = 1                    # supersampling factor per axis
    double: bool = False
    use_bla: bool = True
    log2_eps: float | None = None
    interior: bool = True
    max_tile_samples: int = 8_400_000   # a 1080p frame with 2x2 supersampling in one piece (~0.8 GB)
    overlay: str = ""              # "" | "depth" : draw the zoom depth into the frame
    adaptive: bool = False         # supersample only pixels near edges / the set (ss > 1)
    fast: bool = False             # sequences: frames cut out of shared key pictures (keymovie.py)
    iter_mult: int = 1             # iteration limit x this (stills and videos with more detail)


class FrameRenderer:
    def __init__(self, engine: GPUEngine, settings: RenderSettings):
        self.engine = engine
        self.s = settings
        ctx = engine.ctx
        w, h = settings.width, settings.height
        self.tex = ctx.texture((w, h), 3)
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])
        self.refs = ReferenceCache(engine)
        self._fields: dict = {}
        ss = max(1, int(settings.ss))
        self.tile_rows = max(1, min(h, settings.max_tile_samples // max(1, w * ss * ss)))
        self.stats = {}
        self._ctargets: dict = {}
        self.adaptive_skip = 0          # sequences: frames to render without adaptive sampling
        self.hist_from_frames = False   # sequences: 'auto' colours from the previous frame (no pre-pass)
        self.auto = AutoContrast(ctx)
        self.auto_smooth = 0.0          # >0 for sequences: temporal smoothing of the colour histogram
        self.julia = None               # (re, im): render the Julia set of c instead
        self.jrefs = None               # deep Julia sets: JuliaReferences
        self._deep = False              # this frame of a Julia set uses perturbation
        self.formula = "mandelbrot"     # or "ship": the Burning Ship

    def _vp(self, view: View, img_w: int, img_h: int, **kw) -> FieldParams:
        if self.julia is not None and self._deep:
            return view_params(view, self.jrefs.ref_re, self.jrefs.ref_im, img_w, img_h, julia=True, **kw)
        if self.julia is not None:
            return julia_params(view, img_w, img_h, **kw)
        if self.formula == "ship":
            kw = dict(kw, use_bla=False)
        return view_params(view, self.refs.ref_re, self.refs.ref_im, img_w, img_h, formula=self.formula, **kw)

    def _configure(self, field: Field, fp: FieldParams, view: View, img_w: int, img_h: int, gref, gbla):
        field.configure(fp, gref, gbla)
        if self.julia is not None and not self._deep:
            self.engine.setup_julia(field, view, self.julia, img_w, img_h, fp)
        else:
            field.program = None

    def _prepare_refs(self, view: View, anchor, worker, img_size):
        """Generator: the references this frame needs; returns (GPURef, GPUBla) or (None, None)
        for a Julia set shallow enough for the direct kernel."""
        if self.julia is None:
            self._deep = False
            self.refs.formula = self.formula
            yield from self.ensure_reference(view, anchor, worker, img_size=img_size)
            return self.refs.gpu(self.prec)
        self._deep = view.log2r < julia_min_log2r(False, img_size[1]) + 1.0
        if not self._deep:
            return None, None
        yield from self.ensure_julia_reference(view, worker, img_size, anchor=anchor)
        return self.jrefs.gpu(self.prec)

    def ensure_julia_reference(self, view: View, worker=None, img_size=None, anchor: View | None = None):
        """Generator: references A and B for a deep Julia view (yields while waiting)."""
        s = self.s
        img_w, img_h = img_size or (s.width * s.ss, s.height * s.ss)
        if self.jrefs is None:
            self.jrefs = JuliaReferences(self.engine)
        jr = self.jrefs
        jr.select(self.julia)
        if jr.ref_usable(view, img_w, img_h) and jr.bla_usable(None, self.eps):
            return
        req = jr.reference_request(view, img_w, img_h, anchor=anchor)
        if worker is not None:
            jid = worker.submit("julia_reference", log2_eps=self.eps, **req)
            while True:
                got = None
                for msg in worker.poll():
                    if msg["id"] == jid and msg["kind"] in ("julia_reference", "error", "cancelled"):
                        got = msg
                    elif msg["id"] == jid and msg["kind"] == "progress":
                        self.stats["ref_progress"] = msg["value"]
                if got is not None:
                    if got["kind"] != "julia_reference":
                        raise RuntimeError(got.get("error", "reference cancelled"))
                    jr.set_references(got["ref_a"], got["ref_b"], got["bla_a"], got["bla_b"], self.eps)
                    return
                yield ("reference", self.stats.get("ref_progress", 0.0))
        out = compute_julia_references(req, self.eps)
        jr.set_references(*out, self.eps)

    def release(self):
        for f in self._fields.values():
            f.release()
        self._fields.clear()
        if getattr(self, "_key_field", None) is not None:
            self._key_field.release()
            self._key_field = None
        for t, fb in self._ctargets.values():
            fb.release()
            t.release()
        self._ctargets.clear()
        self.refs.release_gpu()
        self.fbo.release()
        self.tex.release()
        self.auto.release()

    def _colour_target(self, W, H):
        key = (W, H)
        if key not in self._ctargets:
            t = self.engine.ctx.texture((W, H), 4)
            self._ctargets[key] = (t, self.engine.ctx.framebuffer(color_attachments=[t]))
        return self._ctargets[key]

    def _field(self, W, H, keep=()):
        """A field of this size (cached; the least recently used ones beyond 5 are released,
        never one listed in ``keep``)."""
        key = (W, H, int(self.prec))
        f = self._fields.pop(key, None)
        if f is None:
            for k in list(self._fields):
                if len(self._fields) < 5:
                    break
                if self._fields[k] not in keep:
                    self._fields.pop(k).release()
            f = Field(self.engine, W, H, self.prec)
        self._fields[key] = f                 # most recently used last
        return f

    @property
    def eps(self) -> float:
        return self.s.log2_eps if self.s.log2_eps is not None else default_eps(self.prec)

    def _more_iterations(self, v: View | None) -> View | None:
        if v is None or self.s.iter_mult <= 1:
            return v
        v = v.copy()
        v.max_iter = int(v.max_iter * self.s.iter_mult)
        return v

    @property
    def prec(self):
        """Delta precision of the kernels: settings.double (False / True / 2 = float-float), except
        that the Burning Ship has no float-float kernel (float64 there, or float32 without it)."""
        d = self.s.double
        if int(d) == 2 and self.formula == "ship" and self.julia is None:
            return True if self.engine.has_fp64 else False
        return d

    def ensure_reference(self, view: View, anchor: View | None = None, worker=None, img_size=None):
        """Generator: makes sure a usable reference + BLA exist (yields while waiting)."""
        s = self.s
        img_w, img_h = img_size or (s.width * s.ss, s.height * s.ss)
        if not self.refs.ref_usable(view, img_w, img_h):
            req = self.refs.reference_request(view, img_w, img_h, anchor)
            fp = view_params(view, mpfr(req["re"], req["prec"]), mpfr(req["im"], req["prec"]), img_w, img_h)
            ship = req["formula"] == "ship"
            if worker is not None:
                jid = worker.submit("reference", log2_cmax=None if ship else fp.log2_cmax, log2_eps=self.eps, **req)
                while True:
                    got = None
                    for msg in worker.poll():
                        if msg["id"] == jid and msg["kind"] in ("reference", "error", "cancelled"):
                            got = msg
                        elif msg["id"] == jid and msg["kind"] == "progress":
                            self.stats["ref_progress"] = msg["value"]
                    if got is not None:
                        if got["kind"] != "reference":
                            raise RuntimeError(got.get("error", "reference cancelled"))
                        self.refs.set_reference(got["ref"], got["bla"])
                        break
                    yield ("reference", self.stats.get("ref_progress", 0.0))
            else:
                ref = compute_reference(req["re"], req["im"], req["prec"], req["max_iter"], formula=req["formula"])
                ref.ref_id = id(ref)
                self.refs.set_reference(ref, None)
        fp = view_params(view, self.refs.ref_re, self.refs.ref_im, img_w, img_h)
        if self.formula != "ship" and not self.refs.bla_usable(fp, self.eps):
            b = build_bla(self.refs.ref.zm, self.refs.ref.ze, fp.log2_cmax, self.eps, max_iter=self.refs.ref.max_iter)
            b.ref_id = self.refs.ref.ref_id
            self.refs.set_bla(b)

    def render_iter(self, view: View, colors: ColorSettings, time_offset: float = 0.0,
                    anchor: View | None = None, worker=None, time_slice: float = 0.2):
        """Generator yielding ('reference'|'compute', progress); returns RGB uint8 array."""
        t0 = time.perf_counter()
        s = self.s
        view = self._more_iterations(view)
        anchor = self._more_iterations(anchor)
        ss = max(1, int(s.ss))
        img_w, img_h = s.width * ss, s.height * ss
        gref, gbla = yield from self._prepare_refs(view, anchor, worker, (img_w, img_h))
        t_ref = time.perf_counter() - t0
        kw = dict(double=self.prec, use_bla=s.use_bla, with_de=colors.needs_de, with_interior=s.interior,
                  with_stripes=colors.needs_stripes, stripe_freq=colors.stripe_freq)
        t1 = time.perf_counter()
        steps = bla = refined = 0
        auto = None
        # adaptive supersampling pays off when a good part of the frame is smooth; when most
        # pixels needed refining, the next frames of a sequence skip the extra centre pass
        adaptive = ss > 1 and s.adaptive and self.adaptive_skip == 0
        if self.adaptive_skip > 0:
            self.adaptive_skip -= 1
        t_pre = time.perf_counter()
        syncs = 0
        # Colour histogram for the 'auto' mapping:
        #  * one tile (1080p up to 2x2 supersampling): taken from the finished samples of the frame
        #    itself, before colouring - exact and no extra pass;
        #  * several tiles in a video: the previous frame's histogram (all its samples, temporally
        #    smoothed) colours this one while this frame's is gathered for the next;
        #  * otherwise (first frame, stills in tiles, adaptive sampling): a small low-res pre-pass.
        single = self.tile_rows >= s.height and not adaptive
        from_self = colors.mapping == 3 and single
        from_frames = (colors.mapping == 3 and not single and self.hist_from_frames and self.auto.valid)
        if from_self:
            pass
        elif from_frames:
            auto = self.auto
            lim = math.log2(view.max_iter + 1.0)
            self.auto.begin_accumulate(0.0, lim)
            it_lo, it_hi = None, None
        elif colors.mapping == 3:
            # quick low-resolution pass of the same view -> colour histogram for the 'auto' mapping
            pw = min(s.width, 384)
            ph = max(8, int(round(pw * s.height / s.width)))
            pf = self._field(pw, ph)
            fp = self._vp(view, pw, ph, **kw)
            pf.reset()
            self._configure(pf, fp, view, pw, ph, gref, gbla)
            while not pf.step(time_slice, dispatch_target=DISPATCH_TARGET):
                yield ("compute", 0.0)
            if self.auto.update(pf, smooth=self.auto_smooth):
                auto = self.auto
            syncs += pf.syncs
        t_pre = time.perf_counter() - t_pre
        total_rows = s.height
        y0 = 0
        while y0 < total_rows:
            y1 = min(total_rows, y0 + self.tile_rows)
            rows = y1 - y0
            field = self._field(s.width * ss, rows * ss)
            fp = self._vp(view, img_w, img_h, tile_origin=(0, y0 * ss), **kw)
            field.reset()
            if adaptive:
                # one sample per pixel first (with distance estimates), then sub-samples only
                # where the colour changes or the set is near
                # with one row above and below the tile, so tiles decide exactly like one image
                c0, c1 = max(0, y0 - 1), min(total_rows, y1 + 1)
                cf = self._field(s.width, c1 - c0, keep=(field,))
                fpc = self._vp(view, s.width, s.height, tile_origin=(0, c0), **dict(kw, with_de=True))
                cf.reset()
                self._configure(cf, fpc, view, s.width, s.height, gref, gbla)
                while not cf.step(time_slice, dispatch_target=DISPATCH_TARGET):
                    yield ("compute", (y0 + rows * 0.25 * cf.progress) / total_rows)
                steps += cf.steps
                bla += cf.bla_skips
                syncs += cf.syncs
                ctex, cfbo = self._colour_target(s.width, c1 - c0)
                self.engine.display.render(cf, cfbo, (0, 0, s.width, c1 - c0), 1.0, colors,
                                           time_offset=time_offset, iter_base=0,
                                           max_iter_show=view.max_iter, fallback=False, auto=auto)
                field.seed_adaptive(cf, ctex, ss, row0=y0 - c0)
            self._configure(field, fp, view, img_w, img_h, gref, gbla)
            first = True
            while not field.step(time_slice, dispatch_target=DISPATCH_TARGET):
                if first and adaptive:
                    refined += field.flagged
                    first = False
                yield ("compute", (y0 + rows * field.progress) / total_rows)
            if first and adaptive:
                refined += field.flagged
            steps += field.steps
            bla += field.bla_skips
            syncs += field.syncs
            if from_self:
                if self.auto.update(field, smooth=self.auto_smooth):
                    auto = self.auto
                syncs += 1
            if from_frames:
                self.auto.accumulate(field)
                if field.min_iter is not None:
                    it_lo = field.min_iter if it_lo is None else min(it_lo, field.min_iter)
                if field.max_iter_seen is not None:
                    it_hi = field.max_iter_seen if it_hi is None else max(it_hi, field.max_iter_seen)
            self.engine.display.render(field, self.fbo, (0, s.height - y1, s.width, y1 - y0), float(ss),
                                       colors, time_offset=time_offset, iter_base=0,
                                       max_iter_show=view.max_iter, fallback=False, auto=auto)
            # the next tile reuses the same buffers: make sure this draw has consumed them
            self.engine.ctx.finish()
            y0 = y1
        if from_frames and it_lo is not None and it_hi is not None:
            self.auto.end_accumulate(math.log2(max(it_lo, 1)), math.log2(it_hi + 1.0), smooth=self.auto_smooth)
        self.engine.ctx.finish()
        t_rd = time.perf_counter()
        data = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(s.height, s.width, 3)[::-1].copy()
        t_read = time.perf_counter() - t_rd
        if s.overlay:
            img = draw_overlay(img, view, s.overlay)
        frac = refined / max(1, s.width * s.height) if adaptive else 1.0
        if adaptive and frac > 0.6:
            self.adaptive_skip = 15
        self.stats.update(t_ref=t_ref, t_compute=time.perf_counter() - t1 - t_read, t_read=t_read, steps=steps,
                          bla_skips=bla, refined=frac, adaptive=adaptive, t_prepass=t_pre, syncs=syncs)
        return img

    def key_picture_iter(self, plan, *, with_de: bool, with_stripes: bool, stripe_freq: float,
                         anchor: View | None = None, worker=None, time_slice: float = 0.2):
        """Generator: computes a key picture of a fast zoom movie (keymovie.KeyPlan), in tiles
        of equal height on one working field; returns a keymovie.KeyPicture."""
        from .keymovie import KeyPicture, KeptTile
        s = self.s
        if s.iter_mult > 1:
            plan = replace(plan, view=self._more_iterations(plan.view))
            anchor = self._more_iterations(anchor)
        Wk, Hk = plan.W, plan.H
        t0 = time.perf_counter()
        gref, gbla = yield from self._prepare_refs(plan.view, anchor, worker, (Wk, Hk))
        t_ref = time.perf_counter() - t0
        kw = dict(double=self.prec, use_bla=s.use_bla, with_de=with_de, with_interior=s.interior,
                  with_stripes=with_stripes, stripe_freq=stripe_freq)
        rows = max(1, min(Hk, s.max_tile_samples // max(Wk, 1)))
        ntiles = -(-Hk // rows)
        rows = -(-Hk // ntiles)                 # equal tiles: one working field for all of them
        rows += (-rows) % 16                    # sizes that repeat between key pictures
        ctx = self.engine.ctx
        f = getattr(self, "_key_field", None)
        if f is None or (f.W, f.H) != (Wk, rows) or f.double != self.prec:
            if f is not None:
                f.release()
            f = self._key_field = Field(self.engine, Wk, rows, self.prec)
        pic = KeyPicture(ctx, plan)
        steps = bla = syncs = 0
        t1 = time.perf_counter()
        y0 = 0
        while y0 < Hk:
            fp = self._vp(plan.view, Wk, Hk, tile_origin=(0, y0), **kw)
            f.reset()
            self._configure(f, fp, plan.view, Wk, Hk, gref, gbla)
            while not f.step(time_slice, dispatch_target=DISPATCH_TARGET):
                yield ("compute", min(1.0, (y0 + rows * f.progress) / Hk))
            pic.tiles.append(KeptTile(ctx, f, y0))
            steps += f.steps
            bla += f.bla_skips
            syncs += f.syncs
            y0 += rows
        ctx.finish()
        self.stats.update(t_ref=t_ref, t_compute=time.perf_counter() - t1, steps=steps, bla_skips=bla,
                          syncs=syncs, samples=Wk * Hk)
        return pic

    def render(self, view: View, colors: ColorSettings, time_offset: float = 0.0,
               anchor: View | None = None) -> np.ndarray:
        gen = self.render_iter(view, colors, time_offset, anchor, None, time_slice=1.0)
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value


def draw_overlay(img: np.ndarray, view: View, kind: str) -> np.ndarray:
    """Zoom-depth caption in the lower left corner (for zoom videos)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    h = img.shape[0]
    size = max(12, h // 28)
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "Roboto-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            from pathlib import Path
            import imgui_bundle
            font = ImageFont.truetype(str(Path(imgui_bundle.__file__).parent / "assets/fonts/Roboto/Roboto-Regular.ttf"), size)
        except Exception:
            font = ImageFont.load_default()
    z = view.zoom_magnification_log10()
    text = f"zoom 10^{z:.1f}" if kind == "depth" else kind
    x, y = size, h - int(size * 1.8)
    d.text((x + 2, y + 2), text, fill=(0, 0, 0), font=font)
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(im).copy()


def apply_color_overrides(colors: ColorSettings, overrides: dict) -> ColorSettings:
    if not overrides:
        return colors
    c = replace(colors)
    for k, v in overrides.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c


def julia_settings(engine: GPUEngine, settings: RenderSettings, log2r: float) -> RenderSettings:
    """Settings for a Julia set render. Deep Julia sets use perturbation since 1.1, so nothing
    changes any more (1.0 switched to the float64 direct kernel, which stopped at 10^12)."""
    return settings


def render_sequence_iter(engine: GPUEngine, seq: Sequence, path: str, settings: RenderSettings,
                         colors: ColorSettings, fps: float = 60.0, codec: str = "auto",
                         quality: int = 18, worker=None, png_dir: str | None = None,
                         frame_range: tuple | None = None, time_slice: float = 0.2, julia=None,
                         stats: dict | None = None):
    """Generator: renders a keyframe sequence to a video file (``julia`` = c: of that Julia set).

    yields dict(frame, frames, phase, progress, eta)."""
    if julia is None and seq.meta.get("julia") is not None:
        julia = seq.meta["julia"]
    if julia is not None and seq.keyframes:
        settings = julia_settings(engine, settings, min(k.view.log2r for k in seq.keyframes))
    if settings.fast:
        from .keymovie import render_sequence_fast_iter
        yield from render_sequence_fast_iter(engine, seq, path, settings, colors, fps, codec, quality,
                                             worker=worker, png_dir=png_dir, frame_range=frame_range,
                                             time_slice=time_slice, julia=julia, stats=stats)
        return
    if stats is not None:
        for k in ("t_compute", "t_read", "t_encode", "t_frames"):
            stats[k] = 0.0
    fr = FrameRenderer(engine, settings)
    fr.julia = tuple(julia) if julia is not None else None
    fr.formula = seq.meta.get("formula", "mandelbrot")
    fr.auto_smooth = 0.75
    fr.hist_from_frames = True
    n = seq.frame_count(fps)
    a, b = (0, n) if frame_range is None else frame_range
    writer = FfmpegWriter(path, settings.width, settings.height, fps, codec, quality) if path else None
    pngs = PngSaver() if png_dir else None
    t_start = time.perf_counter()
    try:
        for i in range(a, b):
            t = seq.start + i / fps
            view, over = seq.view_at(t)
            cs = apply_color_overrides(colors, over)
            anchor = seq.deepest_in_segment(t)
            t_f0 = time.perf_counter()
            gen = fr.render_iter(view, cs, time_offset=colors.cycle_speed * (t - seq.start),
                                 anchor=anchor, worker=worker, time_slice=time_slice)
            while True:
                try:
                    phase, p = next(gen)
                    done = (i - a + p) / max(b - a, 1)
                    el = time.perf_counter() - t_start
                    yield dict(frame=i, frames=n, phase=phase, progress=done,
                               eta=el / max(done, 1e-6) - el, codec=writer.codec if writer else None)
                except StopIteration as e:
                    img = e.value
                    break
            t_e0 = time.perf_counter()
            if writer is not None:
                writer.write(img)
            if pngs is not None:
                pngs.save(img, f"{png_dir}/frame_{i:06d}.png")      # compressed on worker threads
            if stats is not None:
                stats["t_encode"] += time.perf_counter() - t_e0
                stats["t_compute"] += fr.stats.get("t_compute", 0.0)
                stats["t_read"] += fr.stats.get("t_read", 0.0)
                stats["t_frames"] += time.perf_counter() - t_f0
            done = (i - a + 1) / max(b - a, 1)
            el = time.perf_counter() - t_start
            yield dict(frame=i, frames=n, phase="frame", progress=done, eta=el / done - el,
                       image=img, codec=writer.codec if writer else None)
    finally:
        if pngs is not None:
            pngs.close()
        if writer is not None:
            rc, err = writer.close()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed ({rc}): {err}")
        fr.release()
