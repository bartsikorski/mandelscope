"""Command line interface: GUI, still images, sequence videos, benchmark, info."""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from pathlib import Path


def make_headless_context():
    import moderngl
    errors = []
    for kw in ({}, {"backend": "egl"}):
        try:
            return moderngl.create_standalone_context(require=430, **kw)
        except Exception as e:      # pragma: no cover - platform dependent
            errors.append(f"{kw}: {e}")
    raise RuntimeError("Could not create an OpenGL 4.3 context:\n" + "\n".join(errors))


def _parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def load_location(path: str):
    view, colors, _, _ = load_location_full(path)
    return view, colors


def load_location_full(path: str):
    """(view, colours, julia c or None, formula) from a location .json (or a PNG saved by
    Mandelscope)."""
    from .mp import View
    from .renderer import ColorSettings
    if str(path).lower().endswith(".png"):
        from .library import location_from_png
        d = location_from_png(str(path)) or {}
    else:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    view = View.from_dict(d["view"] if "view" in d else d)
    colors = ColorSettings.from_dict(d.get("colors", {}))
    jc = d.get("julia")
    return view, colors, (float(jc[0]), float(jc[1])) if jc is not None else None, d.get("formula", "mandelbrot")


def cmd_render(args):
    from PIL import Image
    from .renderer import GPUEngine
    from .offline import FrameRenderer, RenderSettings, julia_settings
    view, colors, julia, formula = load_location_full(args.location)
    if args.max_iter:
        view.max_iter = args.max_iter
    if getattr(args, "look", None):
        from .palettes import LOOKS
        from .renderer import ColorSettings
        if args.look not in LOOKS:
            raise SystemExit(f"unknown look {args.look!r}; looks: {', '.join(LOOKS)}")
        colors = ColorSettings.from_look(args.look, colors)
    if args.palette:
        colors.palette = args.palette
    w, h = _parse_size(args.size)
    ctx = make_headless_context()
    eng = GPUEngine(ctx)
    rs = RenderSettings(width=w, height=h, ss=args.ss, double=2 if args.ff else args.f64, use_bla=not args.no_bla,
                        adaptive=args.adaptive, iter_mult=max(1, args.iter_mult))
    if julia is not None:
        rs = julia_settings(eng, rs, view.log2r)
    fr = FrameRenderer(eng, rs)
    fr.julia = julia
    fr.formula = formula
    t0 = time.perf_counter()
    img = fr.render(view, colors)
    dt = time.perf_counter() - t0
    Image.fromarray(img).save(args.output)
    print(f"saved {args.output} ({w}x{h}, ss={args.ss}) in {dt:.2f}s  ref={fr.stats.get('t_ref', 0):.2f}s "
          f"compute={fr.stats.get('t_compute', 0):.2f}s  GPU: {eng.renderer_name}")


def cmd_video(args):
    from .renderer import GPUEngine, ColorSettings
    from .offline import RenderSettings, render_sequence_iter
    from .sequence import Sequence
    d = json.loads(Path(args.sequence).read_text(encoding="utf-8"))
    seq = Sequence.from_dict(d)
    colors = ColorSettings.from_dict(d.get("colors", {}))
    if args.palette:
        colors.palette = args.palette
    w, h = _parse_size(args.size)
    fps = args.fps or seq.fps
    ctx = make_headless_context()
    eng = GPUEngine(ctx)
    rs = RenderSettings(width=w, height=h, ss=args.ss, double=2 if args.ff else args.f64, use_bla=not args.no_bla,
                        adaptive=args.adaptive and not args.fast, fast=args.fast, iter_mult=max(1, args.iter_mult))
    last = 0
    t0 = time.perf_counter()
    for st in render_sequence_iter(eng, seq, args.output, rs, colors, fps, args.codec, args.quality,
                                   png_dir=args.png_dir):
        if st["phase"] == "frame" and (time.perf_counter() - last > 1.0 or st["frame"] == st["frames"] - 1):
            last = time.perf_counter()
            print(f"frame {st['frame'] + 1}/{st['frames']}  {100 * st['progress']:.1f}%  "
                  f"ETA {st['eta']:.0f}s  codec={st['codec']}", flush=True)
    print(f"done: {args.output} in {time.perf_counter() - t0:.1f}s")


def cmd_info(args):
    from .video import find_ffmpeg, available_codecs
    from . import software
    if software.active():
        print(software.describe())
    ctx = make_headless_context()
    print("OpenGL:", ctx.info["GL_VERSION"])
    print("GPU   :", ctx.info["GL_RENDERER"], "/", ctx.info["GL_VENDOR"])
    if sys.platform == "win32" and not software.active():
        d = software.find_mesa()
        print("CPU mode:", f"available (Mesa in {d})" if d else "Mesa not installed (run: mandelscope get-mesa)")
    print("fp64  :", "GL_ARB_gpu_shader_fp64" in ctx.extensions or ctx.version_code >= 400)
    print("ffmpeg:", find_ffmpeg())
    print("video encoders:", ", ".join(c for c, _ in available_codecs()))


def cmd_bench(args):
    """Benchmark: offline frames + the interactive operations (restart, pan, colour)."""
    from .renderer import GPUEngine, ColorSettings
    from .offline import FrameRenderer, RenderSettings
    from .mp import View
    ctx = make_headless_context()
    eng = GPUEngine(ctx)
    w, h = _parse_size(args.size)
    print(f"GPU: {eng.renderer_name}   OpenGL {ctx.info['GL_VERSION']}   size {w}x{h}")
    here = Path(__file__).parent / "locations"
    cases = [("home", View.home())]
    for name in ("seahorse_1e-12.json", "spiral_1e-100.json", "deep_1e-1000.json"):
        p = here / name
        if p.exists():
            cases.append((name.replace(".json", ""), load_location(str(p))[0]))
    cs = ColorSettings()
    precisions = [False] + ([True] if args.f64 else []) + ([2] if (args.f64 or getattr(args, "ff", False)) else [])
    label = {False: "f32", True: "f64", 2: "ff"}
    print("\n-- full frames (exact, as for video export) --")
    print(f"{'':4s}{'location':18s}{'total':>9s}{'compute':>9s}{'Gsteps/s':>10s}{'steps':>15s}{'BLA skips':>13s}"
          f"{'pre-pass':>10s}{'syncs':>7s}")
    refs = []
    for double in precisions:
        fr = FrameRenderer(eng, RenderSettings(width=w, height=h, ss=1, double=double))
        for name, v in cases:
            t0 = time.perf_counter()
            fr.render(v, cs)                      # warm up: reference, BLA, shader compile
            if not double:
                r = fr.refs.ref
                refs.append((name, fr.stats.get("t_ref", 0.0), r.length, r.prec, r.period, v.max_iter))
            t0 = time.perf_counter()
            fr.render(v, cs)
            dt = time.perf_counter() - t0
            st = fr.stats
            gs = st["steps"] / max(st["t_compute"], 1e-9) / 1e9
            print(f"{label[double]:4s}{name:18s}{dt * 1000:8.1f}ms{st['t_compute'] * 1000:8.1f}ms"
                  f"{gs:10.2f}{st['steps']:15,d}{st['bla_skips']:13,d}{st['t_prepass'] * 1000:8.1f}ms{st['syncs']:7d}",
                  flush=True)
        if double != 2:           # (the Burning Ship has no float-float kernel)
            ship_bench(eng, w, h, double, deep=bool(getattr(args, "bla", False) or getattr(args, "experiments", False)))
        # interactive operations on the last renderer's reference
        if not double:
            interactive_ops(eng, fr, cases, w, h)
            if getattr(args, "experiments", False):
                experiments(eng, fr, cases, w, h)
            if getattr(args, "experiments", False) or getattr(args, "bla", False):
                bla_bench(eng, w, h)
        fr.release()
    print("\n-- reference orbits (CPU, MPFR, one core; computed once per area) --")
    for name, t, length, prec, period, mi in refs:
        per = f", period {period}" if period else ""
        print(f"    {name:18s}{t * 1000:9.1f} ms   {length:,} iterations of {mi:,}{per}, {prec} bits"
              f"   ({1e6 * t / max(length, 1):.2f} us/iteration)", flush=True)


def ship_bench(eng, w, h, double: bool, deep: bool = False):
    """The Burning Ship: full frames (perturbation with diffabs and rebasing, no BLA)."""
    from .renderer import ColorSettings
    from .offline import FrameRenderer, RenderSettings
    here = Path(__file__).parent / "locations"
    names = ["ship_home", "ship_armada", "ship_mini_1e-11", "ship_mini_1e-33"] + (["ship_mini_1e-106"] if deep else [])
    print(f"\n-- the Burning Ship ({'f64' if double else 'f32'}; no BLA) --")
    fr = FrameRenderer(eng, RenderSettings(width=w, height=h, ss=1, double=double))
    fr.formula = "ship"
    cs = ColorSettings()
    for name in names:
        p = here / f"{name}.json"
        if not p.exists():
            continue
        v = load_location(str(p))[0]
        fr.render(v, cs)                          # warm up: reference, shader compile
        t_ref = fr.stats.get("t_ref", 0.0)
        t0 = time.perf_counter()
        fr.render(v, cs)
        dt = time.perf_counter() - t0
        st = fr.stats
        gs = st["steps"] / max(st["t_compute"], 1e-9) / 1e9
        print(f"    {name:18s}{dt * 1000:8.1f}ms{st['t_compute'] * 1000:8.1f}ms{gs:10.2f} Gsteps/s"
              f"{st['steps']:15,d} steps   reference {t_ref * 1000:.0f} ms ({fr.refs.ref.length:,} iterations)",
              flush=True)
    fr.release()


def interactive_ops(eng, fr, cases, w, h):
    """Times of the operations the explorer performs every frame while you move."""
    from .renderer import Field, view_params, ColorSettings
    print("\n-- interactive operations (float32) --")
    ctx = eng.ctx
    f = Field(eng, w, h, False)
    tex = ctx.texture((w, h), 3)
    fbo = ctx.framebuffer(color_attachments=[tex])
    cs = ColorSettings()
    for name, v in cases:
        fr.render(v, cs)
        gref, gbla = fr.refs.gpu(False)
        fp = view_params(v, fr.refs.ref_re, fr.refs.ref_im, w, h, with_de=cs.needs_de)
        times = {}
        t0 = time.perf_counter(); f.reset(); ctx.finish(); times["restart"] = time.perf_counter() - t0
        f.configure(fp, gref, gbla)
        t0 = time.perf_counter()
        frames = 0
        while not f.step(0.012, 0.006):
            frames += 1
        times["compute"] = time.perf_counter() - t0
        eng.display.render(f, fbo, (0, 0, w, h), 1.0, cs, max_iter_show=v.max_iter)   # warm-up
        ctx.finish()
        t0 = time.perf_counter()
        eng.display.render(f, fbo, (0, 0, w, h), 1.0, cs, max_iter_show=v.max_iter)
        ctx.finish(); times["colour"] = time.perf_counter() - t0
        # a pan step (17 x 9 pixels); the first one of a run also compiles the copy shader and
        # warms the pending-list path (1.0.0 timed that one: home 6.3 ms on the RTX 5060 Ti)
        for k in (1, 2):
            v2 = v.offset(17 * k, -9 * k, h)
            fp2 = view_params(v2, fr.refs.ref_re, fr.refs.ref_im, w, h, with_de=cs.needs_de)
            ctx.finish()
            t0 = time.perf_counter()
            f.shift(-17, 9); f.configure(fp2, gref, gbla)
            f.run_to_completion(0.012, 0.006)
            times["pan step"] = time.perf_counter() - t0
        print(f"    {name:18s}" + "  ".join(f"{k} {v * 1000:6.1f}ms" for k, v in times.items())
              + f"  ({frames + 1} frame slices, {f.syncs} syncs)", flush=True)
    f.release(); fbo.release(); tex.release()


def bla_bench(eng, w, h, reps: int = 3, names=None, variants=None):
    """Deep, high-iteration places: BLA order, shortest stored skip (lev0) and how skips are
    computed, one full picture each (wall time and GPU time, best of `reps`, interleaved)."""
    from .renderer import Field, GPURef, GPUBla, view_params
    from .reference import compute_reference
    from .bla import build_bla
    from dataclasses import replace
    here = Path(__file__).parent / "locations"
    names = names or ["minibrot_p153231", "minibrot_p6006", "seahorse_1e-12", "spiral_1e-100", "deep_1e-1000"]
    # (label, order, lev0, engine options); the first is the default.  lev0 None: chosen by the
    # place's iteration limit (bla.default_lev0: 2^6 below 65,536 iterations, else 2^7)
    variants = variants or [
        ("auto run32", 2, None, dict(run_length=32)),
        ("2^7 run32", 2, 7, dict(run_length=32)),
        ("2^6 run16", 2, 6, dict(run_length=16)),
        ("2^6 run32", 2, 6, dict(run_length=32)),
        ("2^7 run16", 2, 7, dict(run_length=16)),
        ("2^8 run32", 2, 8, dict(run_length=32)),
        ("2^7 run64", 2, 7, dict(run_length=64)),
        ("2^7 -pre", 2, 7, dict(run_length=32, bla_prefilter=False)),
    ]
    print(f"\n-- deep renders: second-order BLA, shortest skip 2^lev0, skip options (float32, {w}x{h}, best of {reps}) --")
    print("    auto: 2^6 below 65,536 iterations, 2^7 above (default);  runN: N plain iterations per run (default 32);"
          "  -pre: runs leave for every lookup")
    print(f"    {'place':18s}" + "".join(f"{lab:>16s}" for lab, *_ in variants)
          + "      (wall / GPU ms; below: steps / skips / failed lookups per pixel)")
    base = replace(eng.opt)
    q = eng.ctx.query(time=True)
    for name in names:
        p = here / f"{name}.json"
        if not p.exists():
            continue
        v = load_location(str(p))[0]
        ref = compute_reference(str(v.re), str(v.im), v.needed_precision(h) + 32, v.max_iter)
        ref.ref_id = 1
        gref = GPURef(eng.ctx, ref, False)
        fp = view_params(v, v.re, v.im, w, h, with_de=True)
        tabs = {}
        for _, o, lv, _ in variants:
            if (o, lv) not in tabs:
                tabs[(o, lv)] = GPUBla(eng.ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax, -24.0, lev0=lv, order=o,
                                                         max_iter=v.max_iter), False)
        best = {lab: (1e9, 1e9, 0, 0, 0) for lab, *_ in variants}
        f = Field(eng, w, h, False)
        for _ in range(reps):
            for lab, o, lv, opts in variants:
                eng.opt = replace(base, bla_order=o, **opts)
                f.reset()
                f.configure(fp, gref, tabs[(o, lv)])
                eng.ctx.finish()
                t0 = time.perf_counter()
                with q:
                    while not f.step(0.05, 0.02):
                        pass
                wall = time.perf_counter() - t0
                b = best[lab]
                best[lab] = (min(b[0], wall), min(b[1], q.elapsed * 1e-9), f.steps / f.N, f.bla_skips / f.N,
                             f.bla_fails / f.N)
                f.steps = 0
                f.bla_skips = 0
                f.bla_fails = 0
        eng.opt = base
        print(f"    {name:18s}" + "".join(f"{'%.0f / %.0f' % (best[k][0] * 1e3, best[k][1] * 1e3):>16s}"
                                         for k, *_ in variants), flush=True)
        print(f"    {'':18s}" + "".join(f"{'%s/%s/%s' % (f'{best[k][2]:,.0f}', f'{best[k][3]:,.0f}', f'{best[k][4]:,.0f}'):>16s}"
                                        for k, *_ in variants), flush=True)
        f.release()
        for gb in tabs.values():
            gb.release()
        gref.release() if hasattr(gref, "release") else None
    getattr(q, "release", lambda: None)()


def experiments(eng, fr, cases, w, h, reps: int = 3):
    """A/B tests of engine options on this GPU (the first row is the default).

    Every location keeps its own reference; all variants run on the same field, which starts
    every picture from the same state; repeats are interleaved (A B C .. A B C ..) so slow
    drifts of the GPU clock affect all variants alike."""
    from dataclasses import replace
    from .renderer import Field, view_params, ColorSettings
    from .offline import FrameRenderer, RenderSettings
    cs = ColorSettings()
    base = replace(eng.opt)
    print("\n-- diagnostics --")
    tiny = Field(eng, 8, 8, False)
    for _ in range(20):
        tiny._read_counters()
    t0 = time.perf_counter()
    for _ in range(200):
        tiny._read_counters()
    t_idle = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        tiny._word_dispatch(2)
        tiny._read_counters()
    t_rt = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(500):
        tiny._word_dispatch(2)
    tiny._read_counters()
    t_disp = (time.perf_counter() - t0) / 500
    tiny.release()
    print(f"    counter read, GPU idle        {t_idle * 1000:7.3f} ms")
    print(f"    tiny dispatch + read (a sync)  {t_rt * 1000:7.3f} ms")
    print(f"    dispatch submit cost           {t_disp * 1000:7.3f} ms")

    locs = []
    expect = {}
    for name, v in cases:                       # one reference per location, kept on the GPU
        frl = FrameRenderer(eng, RenderSettings(width=w, height=h))
        frl.render(v, cs)
        for k in list(frl._fields):
            frl._fields.pop(k).release()
        locs.append((name, v, frl))
        # what a correct picture looks like: a fresh field, as in the explorer
        gref, gbla = frl.refs.gpu(False)
        ff = Field(eng, w, h, False)
        ff.configure(view_params(v, frl.refs.ref_re, frl.refs.ref_im, w, h, with_de=cs.needs_de), gref, gbla)
        while not ff.step(0.012, 0.006):
            pass
        expect[name] = (ff.steps, ff.maxiter_count, ff.interior_count)
        ff.release()

    reuse_diagnostic(eng, locs, w, h, cs)

    print("\n-- experiments (float32, one full picture as in the explorer; best of %d, interleaved) --" % reps)
    variants = [
        ("default", {}, True),
        ("v3.0: wait after each level", dict(sync_per_level=True, chunk_start=32768), True),
        ("v2: + barrier, glFinish", dict(sync_per_level=True, chunk_start=32768,
                                         barrier_each_dispatch=True, finish_before_read=True), True),
        ("no plain-float runs", dict(runs=False), True),
        ("no BLA gate", dict(bla_gate=False), True),
        ("budget 1024", dict(base_budget=1024), True),
        ("budget 4096", dict(base_budget=4096), True),
        ("no interior detection", {}, False),
        ("default, new field each time", {"_fresh": True}, True),
        ("default, reference re-uploaded", {"_reupload": True}, True),
        ("v3.3: counters read directly", dict(staged_counters=False), True),
    ]

    def check(name, fld, interior):
        st, mx, it = expect[name]
        if not interior:
            return fld.maxiter_count + fld.interior_count == mx + it
        # steps may differ by a few % (the plain-float runs notice inside pixels a few
        # iterations later); a broken picture changes them by far more
        return (abs(fld.steps - st) <= 0.05 * st + 1000 and abs(fld.maxiter_count - mx) <= 0.001 * fld.N + 10
                and abs(fld.interior_count - it) <= 0.001 * fld.N + 10)

    f = Field(eng, w, h, False)
    best, info, bad = {}, {}, {}
    for _ in range(reps):
        for label, opts, interior in variants:
            eng.opt = replace(base, **{k: v for k, v in opts.items() if not k.startswith("_")})
            for name, v, frl in locs:
                if opts.get("_reupload"):
                    frl.refs.release_gpu()
                gref, gbla = frl.refs.gpu(False)
                fp = view_params(v, frl.refs.ref_re, frl.refs.ref_im, w, h, with_de=cs.needs_de,
                                 with_interior=interior)
                fld = Field(eng, w, h, False) if opts.get("_fresh") else f
                fld.base_budget = eng.opt.base_budget
                fld.reset()
                fld.configure(fp, gref, gbla)
                eng.ctx.finish()
                t0 = time.perf_counter()
                while not fld.step(0.012, 0.006):
                    pass
                dt = time.perf_counter() - t0
                key = (label, name)
                ok = check(name, fld, interior)
                if not ok:
                    bad[key] = (fld.steps, fld.maxiter_count, fld.interior_count)
                if dt < best.get(key, 1e9):
                    best[key] = dt
                    info[key] = (fld.dispatches, fld.syncs, fld.steps)
                if fld is not f:
                    fld.release()
    eng.opt = base
    f.release()
    names = [n for n, _ in cases]
    print(f"    {'variant':32s}" + "".join(f"{n:>17s}" for n in names))
    for label, _, _ in variants:
        print(f"    {label:32s}" + "".join(f"{best[(label, n)] * 1000:14.1f}ms{'!' if (label, n) in bad else ' '}"
                                         for n in names), flush=True)
    print(f"    {'  (dispatches/syncs, default)':32s}" + "".join(f"{'%d/%d' % info[('default', n)][:2]:>17s}" for n in names))
    print(f"    {'  (million steps, correct)':32s}" + "".join(f"{expect[n][0] / 1e6:16.1f} " for n in names))
    if bad:
        print("    ! = WRONG PICTURE in at least one repeat (steps / undecided / inside differ from a fresh render):")
        for (label, n), (st, mx, it) in list(bad.items())[:12]:
            e = expect[n]
            print(f"      {label} / {n}: steps {st / 1e6:.1f}M (correct {e[0] / 1e6:.1f}M), "
                  f"undecided {mx} ({e[1]}), inside {it} ({e[2]})")
    else:
        print("    all pictures correct (checked against a fresh render)")

    print("\n-- export 2x2 supersampling: adaptive vs every sub-sample (best of 2) --")
    rows = {}
    for adaptive in (False, True):
        cells = []
        for name, v, frl in locs:
            frl.s = replace(frl.s, ss=2, adaptive=adaptive)
            frl.tile_rows = max(1, min(h, frl.s.max_tile_samples // (w * 4)))
            frl.render(v, cs)
            tb = None
            for _ in range(2):
                frl.adaptive_skip = 0
                t0 = time.perf_counter()
                frl.render(v, cs)
                dt = time.perf_counter() - t0
                if tb is None or dt < tb:
                    tb, st = dt, dict(frl.stats)
            cells.append((tb, st))
            for k in list(frl._fields):
                frl._fields.pop(k).release()
        rows[adaptive] = cells
    for adaptive in (False, True):
        label = "adaptive" if adaptive else "every sub-sample"
        extra = ("   refined " + " ".join(f"{100 * st['refined']:.0f}%" for _, st in rows[True])) if adaptive else ""
        print(f"    {label:30s}" + "".join(f"{t * 1000:15.1f}ms" for t, _ in rows[adaptive]) + extra, flush=True)
    print(f"    {'  (colour pre-pass, syncs)':30s}" + "".join(
        f"{'%.1fms %d' % (st['t_prepass'] * 1000, st['syncs']):>17s}" for _, st in rows[False]))
    for _, _, frl in locs:
        frl.release()
    movie_bench(eng, cases, w, h)


def movie_bench(eng, cases, w, h, fps: float = 30.0, seconds: float = 1.0):
    """Zoom movie: one octave in one second, frames computed one by one vs cut out of shared
    key pictures (both 2x2 samples per pixel at least), with a breakdown per frame and a run
    with video encoding."""
    import tempfile
    import numpy as np
    from .offline import RenderSettings, render_sequence_iter
    from .renderer import ColorSettings
    from .sequence import Sequence, Keyframe
    cs = ColorSettings()
    print(f"\n-- zoom movie, 1 octave in {seconds:g} s at {fps:g} frames/s, {w}x{h}, 2x2 --")
    print("    (ms per frame.  Fast mode overlaps the GPU work of a frame with the CPU work of the one")
    print("     before; GPU = GPU time measured with timer queries, it overlaps the other parts)")
    tmp = Path(tempfile.mkdtemp(prefix="mandelscope_bench_"))
    # starting / finishing ffmpeg (process, encoder session, moving the index to the front of the
    # file) costs a fixed time per video: measured on its own, so the per-frame cost is visible
    from .video import FfmpegWriter
    t0 = time.perf_counter()
    wr = FfmpegWriter(str(tmp / "fixed.mp4"), w, h, fps, "auto", 18)
    wr.write(np.zeros((h, w, 3), np.uint8))
    rc, _ = wr.close()
    t_fixed = time.perf_counter() - t0
    for name, v in cases:
        if name == "home":
            continue
        v0 = v.copy()
        v0.log2r = v.log2r + 1.0
        seq = Sequence([Keyframe(v0, 0.0), Keyframe(v, seconds)], fps=fps)
        n = seq.frame_count(fps)
        res = {}
        for fast in (False, True):
            rs = RenderSettings(width=w, height=h, ss=2, fast=fast)
            st = {}
            for rep in range(2):             # warm up (reference, shaders), then time a full run
                t0 = time.perf_counter()
                for _ in render_sequence_iter(eng, seq, None, rs, cs, fps=fps, stats=st):
                    pass
                dt = time.perf_counter() - t0
            res[fast] = (dt, dict(st))
            # the same with video encoding (as an export)
            codec = None
            t0 = time.perf_counter()
            for d in render_sequence_iter(eng, seq, str(tmp / f"{name}_{fast}.mp4"), rs, cs, fps=fps):
                codec = d.get("codec") or codec
            res[(fast, "enc")] = (time.perf_counter() - t0, codec)
        ms = lambda t: 1000.0 * t / n
        t_n, sn = res[False]
        t_f, sf = res[True]
        other_n = t_n - sn["t_compute"] - sn["t_read"]
        print(f"    {name:16s} one by one {ms(t_n):7.1f}   compute {ms(sn['t_compute']):.1f}, "
              f"read-back {ms(sn['t_read']):.1f}, other {ms(other_n):.1f}", flush=True)
        other_f = t_f - sf["t_keys"] - sf["t_hist"] - sf["t_issue"] - sf["t_read"]
        ratio = sf.get("samples", 0) / max(sf.get("direct_samples", 1), 1)
        print(f"    {'':16s} fast       {ms(t_f):7.1f}   key pictures {ms(sf['t_keys']):.1f}, "
              f"colour histograms {ms(sf['t_hist']):.1f}, issuing {ms(sf['t_issue']):.1f}, "
              f"read-back incl. waiting {ms(sf['t_read']):.1f}, other {ms(other_f):.1f}  "
              f"(GPU: colour {ms(sf['t_colour']):.1f}, filter {ms(sf['t_filter']):.1f})   "
              f"[{sf.get('key_pictures', 0)} key pictures, {100 * ratio:.0f}% of the samples]   "
              f"{t_n / max(t_f, 1e-9):.1f}x", flush=True)
        (e_n, codec), (e_f, _) = res[(False, "enc")], res[(True, "enc")]
        en, ef = e_n - t_fixed, e_f - t_fixed
        print(f"    {'':16s} with encoding ({codec}, without the {t_fixed:.2f} s once per video): "
              f"one by one {ms(en):.1f}, fast {ms(ef):.1f}   {en / max(ef, 1e-9):.1f}x", flush=True)
    try:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    except OSError:
        pass


MOVE_HINT = "memory to "


def reuse_diagnostic(eng, locs, w, h, cs, rounds: int = 6):
    """One field used for many pictures in a row, as in the explorer (v3.3 bench: up to 50x
    slower than a new field).  Wall and GPU time per picture, and what the driver reports."""
    from dataclasses import replace
    from statistics import median
    from .renderer import Field, view_params
    from .gldebug import DebugLog, buffer_moves
    ctx = eng.ctx
    base = replace(eng.opt)
    print(f"\n-- one field reused for {rounds} x {len(locs)} pictures, as in the explorer "
          f"(median wall / GPU ms per picture) --")
    print(f"    {'':34s}" + "".join(f"{n:>17s}" for n, _, _ in locs))
    names = {}
    for _, _, frl in locs:
        gref, gbla = frl.refs.gpu(False)
        for b, what in ((gref.bf, "reference"), (gref.bm, "reference"), (gref.be, "reference"),
                        (gbla.buf, "BLA table")):
            names[b.glo] = what
    for k, (b, _) in eng._orders.items():
        names[b.glo] = "sample order"
    log = DebugLog(ctx)
    moves_all = []
    q = ctx.query(time=True)
    modes = [("counters read directly (v3.3)", dict(staged_counters=False), False),
             ("counters through a copy (new)", dict(staged_counters=True), False),
             ("new field for every picture", dict(staged_counters=True), True)]
    for label, opts, fresh in modes:
        eng.opt = replace(base, **opts)
        f = None if fresh else Field(eng, w, h, False)
        wall = {n: [] for n, _, _ in locs}
        gpu = {n: [] for n, _, _ in locs}
        log.drain()
        log.messages.clear()
        seen = dict(names)
        for _ in range(rounds):
            for name, v, frl in locs:
                gref, gbla = frl.refs.gpu(False)
                fp = view_params(v, frl.refs.ref_re, frl.refs.ref_im, w, h, with_de=cs.needs_de)
                fld = Field(eng, w, h, False) if fresh else f
                seen.update(fld.buffer_names())
                fld.reset()
                fld.configure(fp, gref, gbla)
                ctx.finish()
                t0 = time.perf_counter()
                with q:
                    while not fld.step(0.012, 0.006):
                        pass
                wall[name].append(time.perf_counter() - t0)
                gpu[name].append(q.elapsed * 1e-9)
                if fresh:
                    fld.release()
                log.drain()
        if f is not None:
            f.release()
        print(f"    {label:34s}" + "".join(
            f"{'%.1f / %.1f' % (median(wall[n]) * 1000, median(gpu[n]) * 1000):>17s}" for n, _, _ in locs), flush=True)
        moves = buffer_moves(log.messages, seen)
        perf = sorted({m for t, m in log.messages if t == "performance"})
        moves_all.append((label, moves, len(log.messages), perf, seen))
    eng.opt = base
    log.close()
    getattr(q, "release", lambda: None)()
    if not log.ok:
        print("    driver messages: not available here")
        return
    any_moves = False
    for label, moves, n, perf, _ in moves_all:
        for what, a, b, k in moves:
            any_moves = True
            print(f"    driver: '{what}' moved from {a} to {b} memory ({k}x)   [{label}]")
    if not any_moves:
        total = sum(m[2] for m in moves_all)
        print(f"    driver: no buffer moved between video and system memory ({total} other messages)")
    shown = set()
    for label, _, _, perf, names_all in moves_all:
        for m in perf:
            if m not in shown and len(shown) < 4 and not MOVE_HINT in m:
                shown.add(m)
                named = re.sub(r"buffer (\d+)", lambda g: f"buffer {g.group(1)} ({names_all.get(int(g.group(1)), '?')})", m)
                print(f"    driver performance note [{label}]: {named[:300]}")


def cmd_get_mesa(args):
    """Install Mesa's software OpenGL (llvmpipe) for CPU mode on Windows."""
    from . import software
    dest = Path(args.dest) if args.dest else software.user_mesa_dir()
    if args.archive:
        software.install_from_archive(Path(args.archive), dest)
        print(f"installed into {dest}")
    else:
        software.download_mesa(dest)
    print("Start the explorer in CPU mode with:  run.bat --software")
    return 0


def _start_software() -> int:
    """--software / --cpu / MANDELSCOPE_SOFTWARE=1: switch to llvmpipe before any OpenGL loads."""
    from . import software
    try:
        software.enable()
        return 0
    except software.MesaMissing as e:
        if sys.platform == "win32" and software.ensure_mesa_interactive(
                "CPU mode uses Mesa's software renderer, which is not installed yet."):
            software.enable()
            return 0
        print(f"{e}\nInstall it with:  run.bat get-mesa", file=sys.stderr)
        return 2


def main(argv=None):
    from . import software
    argv = software.take_flags(list(sys.argv[1:] if argv is None else argv))
    if software.requested():
        rc = _start_software()
        if rc:
            return rc
    ap = argparse.ArgumentParser(
        prog="mandelscope", description="GPU deep-zoom Mandelbrot explorer",
        epilog="--software (or --cpu) before or after the command runs everything on the processor "
               "with Mesa's software OpenGL; --gpu disables the automatic fallback.")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="interactive explorer (default)")
    g.add_argument("location", nargs="?", help="location .json to open")
    g.add_argument("--size", default="1600x900")
    g.add_argument("--selftest", action="store_true", help="scripted UI test, writes screenshots")
    g.add_argument("--selftest-dir", default="selftest_out")
    g.add_argument("--selftest-quick", action="store_true", help="short version of the scripted test")
    g.add_argument("--motion-study", action="store_true",
                   help="scripted zooms / drags / rotations, recorded with a per-frame log into footage/")
    g.add_argument("--motion-study-quick", action="store_true", help=argparse.SUPPRESS)
    r = sub.add_parser("render", help="render a still image")
    r.add_argument("location")
    r.add_argument("-o", "--output", default="render.png")
    r.add_argument("--size", default="1920x1080")
    r.add_argument("--ss", type=int, default=2)
    r.add_argument("--f64", action="store_true")
    r.add_argument("--ff", action="store_true", help="precise float-float deltas (two float32 per number)")
    r.add_argument("--iter-mult", type=int, default=1, help="iteration limit x this (more detail)")
    r.add_argument("--no-bla", action="store_true")
    r.add_argument("--adaptive", action="store_true", help="sub-samples only where needed")
    r.add_argument("--max-iter", type=int, default=0)
    r.add_argument("--palette", default=None)
    r.add_argument("--look", default=None, help="colour look (e.g. Gilded, Silk, Midnight)")
    v = sub.add_parser("video", help="render a keyframe sequence to video")
    v.add_argument("sequence")
    v.add_argument("-o", "--output", default="zoom.mp4")
    v.add_argument("--size", default="1920x1080")
    v.add_argument("--fps", type=float, default=0)
    v.add_argument("--ss", type=int, default=2)
    v.add_argument("--codec", default="auto")
    v.add_argument("--quality", type=int, default=18)
    v.add_argument("--f64", action="store_true")
    v.add_argument("--ff", action="store_true", help="precise float-float deltas (two float32 per number)")
    v.add_argument("--iter-mult", type=int, default=1, help="iteration limit x this (more detail)")
    v.add_argument("--no-bla", action="store_true")
    v.add_argument("--adaptive", action="store_true", help="sub-samples only where needed")
    v.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                   help="frames cut out of shared key pictures, >= ss x ss samples per pixel (default; "
                        "--no-fast computes every frame on its own)")
    v.add_argument("--palette", default=None)
    v.add_argument("--png-dir", default=None)
    sub.add_parser("info", help="show GPU / encoder information")
    b = sub.add_parser("bench", help="benchmark")
    b.add_argument("--size", default="1920x1080")
    b.add_argument("--f64", action="store_true", help="also float64 and float-float frames")
    b.add_argument("--ff", action="store_true", help="also float-float frames")
    b.add_argument("--experiments", action="store_true",
                   help="also A/B-test engine options (sync, BLA gate, budget, interior, adaptive ss)")
    b.add_argument("--bla", action="store_true",
                   help="deep places: first- vs second-order BLA and the shortest skip")
    m = sub.add_parser("get-mesa", help="Windows: install Mesa's software OpenGL for CPU mode")
    m.add_argument("--dest", default=None, help="folder (default: %%LOCALAPPDATA%%\\mandelscope\\mesa)")
    m.add_argument("--archive", default=None, help="install from a downloaded mesa-llvmpipe .7z / .zip")
    args = ap.parse_args(argv)
    if args.cmd in (None, "gui"):
        if software.fallback_if_needed():
            return 0                 # no OpenGL 4.3 driver: a CPU-mode Mandelscope was started
        from .app import run_app
        loc = getattr(args, "location", None)
        size = _parse_size(getattr(args, "size", "1600x900"))
        quick = getattr(args, "selftest_quick", False)
        if getattr(args, "motion_study", False) or getattr(args, "motion_study_quick", False):
            out = Path(args.selftest_dir if args.selftest_dir != "selftest_out" else
                       Path.cwd() / "footage" / time.strftime("%Y%m%d_%H%M%S"))
            print(f"motion study -> {out}", flush=True)
            rc = run_app(location=loc, size=size, selftest="study-quick" if args.motion_study_quick else "study",
                         selftest_dir=str(out))
            print(f"done: {out}\\study.mp4, frames.jsonl, meta.json" if os.name == "nt" else f"done: {out}", flush=True)
            return rc
        return run_app(location=loc, size=size, selftest="quick" if quick else getattr(args, "selftest", False),
                       selftest_dir=getattr(args, "selftest_dir", "selftest_out"))
    return {"render": cmd_render, "video": cmd_video, "info": cmd_info, "bench": cmd_bench,
            "get-mesa": cmd_get_mesa}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
