"""Shared helpers: render a view on the GPU and compare with MPFR ground truth."""
from __future__ import annotations

import time

import numpy as np

from mandelscope.mp import View
from mandelscope.reference import compute_reference
from mandelscope.bla import build_bla
from mandelscope.renderer import Field, GPURef, GPUBla, view_params
from mandelscope import groundtruth as gt

RES_INTERIOR = 0xFFFFFFFE


def render_view(engine, view: View, W: int, H: int, double=False, use_bla=True, de=True,
                interior=True, ref_at=None, timeout=900, formula="mandelbrot", stripes=False):
    prec = view.needed_precision(H) + 32
    rr, ri = (view.re, view.im) if ref_at is None else ref_at
    t = time.perf_counter()
    ref = compute_reference(str(rr), str(ri), prec, view.max_iter, formula=formula)
    t_ref = time.perf_counter() - t
    ship = formula == "ship"
    fp = view_params(view, rr, ri, W, H, double=double, use_bla=use_bla and not ship, with_de=de,
                     with_interior=interior, formula=formula, with_stripes=stripes)
    t = time.perf_counter()
    bla = None if ship else build_bla(ref.zm, ref.ze, fp.log2_cmax, -45.0 if double else -24.0)
    t_bla = time.perf_counter() - t
    f = Field(engine, W, H, double)
    f.configure(fp, GPURef(engine.ctx, ref, double), GPUBla(engine.ctx, bla, double) if bla is not None else None)
    t = time.perf_counter()
    ok = f.run_to_completion(timeout=timeout)
    t_gpu = time.perf_counter() - t
    res = f.read_results()
    info = dict(ref_len=ref.length, ref_escaped=ref.escaped, period=ref.period, t_ref=t_ref,
                t_bla=t_bla, bla_size=bla.size if bla else 0, bla_levels=bla.levels if bla else 0,
                t_gpu=t_gpu, complete=ok,
                dispatches=f.dispatches, bla_skips=f.bla_skips, rebases=f.rebases, steps=f.steps)
    f.release()
    return res, info


def render_julia(engine, view: View, c, W: int, H: int, double=False, use_bla=True, de=True,
                 interior=True, timeout=900):
    """A Julia set by perturbation: reference A at the view centre, B = the critical orbit."""
    prec = view.needed_precision(H) + 32
    cre, cim = str(c[0]), str(c[1])
    from mandelscope.mp import to_str
    p = max(prec, view.re.precision, view.im.precision)
    ref_a = compute_reference(cre, cim, prec, view.max_iter, z0=(to_str(view.re, p), to_str(view.im, p)))
    ref_b = compute_reference(cre, cim, prec, view.max_iter)
    fp = view_params(view, view.re, view.im, W, H, double=double, use_bla=use_bla, with_de=de,
                     with_interior=interior, julia=True)
    eps = -45.0 if double else -24.0
    bla_a = build_bla(ref_a.zm, ref_a.ze, -10000.0, eps)
    bla_b = build_bla(ref_b.zm, ref_b.ze, -10000.0, eps)
    f = Field(engine, W, H, double)
    f.configure(fp, GPURef.pair(engine.ctx, ref_a, ref_b, double), GPUBla(engine.ctx, bla_b, double, bla_a=bla_a))
    ok = f.run_to_completion(timeout=timeout)
    res = f.read_results()
    info = dict(complete=ok, ref_a=ref_a.length, ref_b=ref_b.length, bla_skips=f.bla_skips,
                rebases=f.rebases, steps=f.steps)
    f.release()
    return res, info


def compare(view: View, res: np.ndarray, nsamp: int = 40, seed: int = 1, tol_px: float = 1e-3,
            prec_extra: int = 64, only_escaped=False, julia=None, formula="mandelbrot"):
    """Compare GPU iteration counts with ground truth.

    Returns dict(exact, chaotic, hard, details)."""
    H, W = res.shape[:2]
    prec = view.needed_precision(H) + prec_extra
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < nsamp and tries < nsamp * 20:
        tries += 1
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        if only_escaped and res[y, x, 0] >= RES_INTERIOR:
            continue
        pts.append((x, y))
    exact = chaotic = hard = 0
    details = []
    for x, y in pts:
        g = int(res[y, x, 0])
        def truth(xx, yy):
            p0 = gt.sample_c(view, xx, yy, W, H, prec)
            if julia is not None:
                return gt.iterate_julia(p0, julia, view.max_iter, prec)
            if formula == "ship":
                return gt.iterate_ship(p0, view.max_iter, prec)
            return gt.iterate(p0, view.max_iter, prec)
        n, r2, fr = truth(x, y)
        if g == n or (g >= RES_INTERIOR and n == 0xFFFFFFFF):
            exact += 1
            continue
        # is the true value itself unstable within tol_px?
        neigh = []
        for dx, dy in ((tol_px, 0), (-tol_px, 0), (0, tol_px), (0, -tol_px)):
            neigh.append(truth(x + dx, y + dy)[0])
        vals = [n] + neigh
        if len(set(vals)) > 1 or g in vals:
            chaotic += 1
            details.append(("chaotic", x, y, n, g, neigh))
        else:
            hard += 1
            details.append(("HARD", x, y, n, g, neigh))
    return dict(exact=exact, chaotic=chaotic, hard=hard, details=details, n=len(pts))


def stats(res):
    ns = res[..., 0]
    esc = ns[(ns > 0) & (ns < RES_INTERIOR)]
    return dict(min=int(esc.min()) if esc.size else None, max=int(esc.max()) if esc.size else None,
                interior=int(np.sum(ns == RES_INTERIOR)), maxiter=int(np.sum(ns == 0xFFFFFFFF)),
                unfinished=int(np.sum(ns == 0)))
