"""GPU kernels versus brute-force MPFR ground truth, plus engine features."""
import json
import math

import numpy as np
import gmpy2
import pytest
from gmpy2 import mpfr

from mandelscope.mp import View
from mandelscope.reference import compute_reference
from mandelscope.bla import build_bla
from mandelscope.renderer import Field, GPURef, GPUBla, view_params, ColorSettings
from mandelscope.offline import FrameRenderer, RenderSettings
from accuracy import render_view, compare, stats

pytestmark = pytest.mark.gpu


def misiurewicz_view(depth10, max_iter=30000, angle=0.3):
    l2 = -depth10 / 0.30103
    p = int(-l2) + 100
    re = gmpy2.mul_2exp(mpfr(0.37, p), int(l2))
    im = mpfr(1, p) + gmpy2.mul_2exp(mpfr(0.21, p), int(l2))
    return View(re, im, l2, angle, max_iter)


CASES = {
    "home": lambda: View(mpfr("-0.75", 64), mpfr(0, 64), math.log2(1.35), 0.0, 600),
    "seahorse_1e-6": lambda: View.from_strings("-0.743643887037158704752191506114774",
                                               "0.131825904205311970493132056385139", math.log2(1e-6), 0.0, 3000),
    "spiral_1e-40": lambda: misiurewicz_view(40),
    "spiral_1e-320": lambda: misiurewicz_view(320),
    "spiral_1e-1000": lambda: misiurewicz_view(1000),
}


@pytest.mark.parametrize("double", [False, True, 2], ids=["f32", "f64", "ff"])
@pytest.mark.parametrize("name", list(CASES))
def test_matches_ground_truth(engine, name, double):
    if double is True and not engine.has_fp64:
        pytest.skip("no fp64")
    v = CASES[name]()
    res, info = render_view(engine, v, 64, 48, double=double)
    assert info["complete"]
    st = stats(res)
    assert st["unfinished"] == 0
    cmp = compare(v, res, nsamp=24, tol_px=1e-3 if not double else 1e-6)
    hard = [d for d in cmp["details"] if d[0] == "HARD"]
    assert not hard, hard
    assert cmp["exact"] >= 0.6 * cmp["n"]


@pytest.mark.slow
@pytest.mark.parametrize("double", [False, True, 2], ids=["f32", "f64", "ff"])
def test_minibrot_period_6006(engine, locations, double):
    d = json.loads((locations / "minibrot_p6006.json").read_text())["view"]
    v = View.from_dict(d)
    v.max_iter = 6006 * 12
    res, info = render_view(engine, v, 48, 32, double=double)
    assert info["complete"] and info["period"] == 6006
    cmp = compare(v, res, nsamp=16, tol_px=1e-3 if not double else 1e-6)
    assert cmp["hard"] == 0, cmp["details"]


def _field(engine, v, W, H, double=False, budget=2048, use_bla=True):
    ref = compute_reference(str(v.re), str(v.im), v.needed_precision(H) + 32, v.max_iter)
    ref.ref_id = 1
    fp = view_params(v, v.re, v.im, W, H, double=double, use_bla=use_bla)
    bla = build_bla(ref.zm, ref.ze, fp.log2_cmax, -53.0 if double else -24.0)
    f = Field(engine, W, H, double)
    f.base_budget = budget
    f.reset()
    f.configure(fp, GPURef(engine.ctx, ref, double), GPUBla(engine.ctx, bla, double))
    return f, ref, bla


def test_small_budget_resumes_identically(engine):
    v = misiurewicz_view(100, 5000)
    f1, _, _ = _field(engine, v, 48, 32, budget=1 << 16)
    f1.run_to_completion()
    f2, _, _ = _field(engine, v, 48, 32, budget=3)       # forces many resumptions
    while not f2.step(0.5, dispatch_target=0.001, max_budget=3):
        pass
    assert f2.dispatches > 10
    assert np.array_equal(f1.read_results()[..., 0], f2.read_results()[..., 0])


def test_bla_on_off_agree(engine):
    v = misiurewicz_view(320, 5000)
    a, _, _ = _field(engine, v, 48, 32, use_bla=True)
    a.run_to_completion()
    b, _, _ = _field(engine, v, 48, 32, use_bla=False)
    b.run_to_completion()
    assert a.bla_skips > 0 and b.bla_skips == 0
    ra, rb = a.read_results()[..., 0].astype(np.int64), b.read_results()[..., 0].astype(np.int64)
    assert np.mean(ra == rb) > 0.97
    assert a.steps < b.steps / 3                    # BLA really skips work


VARIANTS = (("floatexp", dict(bla_fast=False)), ("fast", {}), ("backoff", dict(bla_backoff=True)))


def test_skip_arithmetic_and_backoff_agree(engine):
    """Skips in plain arithmetic (v3.11) and the failed-lookup back-off give the same picture as
    floatexp skips up to rounding (a non-chaotic place: escape counts agree)."""
    from mandelscope.renderer import GPUEngine, EngineOptions
    v = misiurewicz_view(320, 5000)
    out = {}
    for label, opts in VARIANTS:
        eng = GPUEngine(engine.ctx, EngineOptions(**opts))
        f, _, _ = _field(eng, v, 48, 32)
        f.run_to_completion()
        r = f.read_results()
        out[label] = (r[..., 0].astype(np.int64), r[..., 1].copy().view(np.float32), f.bla_skips, f.bla_fails)
    n0, fr0, sk0, fl0 = out["floatexp"]
    for label in ("fast", "backoff"):
        n1, fr1, sk1, fl1 = out[label]
        assert sk1 > 0
        assert np.mean(n0 == n1) > 0.98, label
        same = n0 == n1
        assert np.all(np.abs(fr0[same] - fr1[same]) < 1e-3), label
    assert out["backoff"][3] <= fl0                 # the back-off looks up less often


@pytest.mark.slow
def test_skip_variants_match_ground_truth_at_a_deep_minibrot(engine, locations):
    """Near the period-6006 minibrot most pixels are chaotic (even BLA on / off agree on only
    ~75 % of the escape counts): compare every variant with MPFR instead."""
    from mandelscope.renderer import GPUEngine, EngineOptions
    v = View.from_dict(json.loads((locations / "minibrot_p6006.json").read_text())["view"])
    v.max_iter = 6006 * 6
    for label, opts in VARIANTS:
        eng = GPUEngine(engine.ctx, EngineOptions(**opts))
        res, info = render_view(eng, v, 48, 32)
        assert info["complete"], label
        cmp = compare(v, res, nsamp=16, tol_px=1e-3)
        assert cmp["hard"] == 0, (label, cmp["details"])


def test_shift_reuse_equals_fresh_render(engine):
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(3e-5), 0.0, 2000)
    W, H = 64, 48
    f, ref, bla = _field(engine, v, W, H)
    f.run_to_completion()
    v2 = v.offset(5, -3, H)                         # move 5 px right, 3 px up
    fp2 = view_params(v2, v.re, v.im, W, H)
    f.shift(-5, 3)
    f.configure(fp2, f.gref, f.gbla)
    f.run_to_completion()
    fresh = Field(engine, W, H, False)
    fresh.configure(fp2, f.gref, f.gbla)
    fresh.run_to_completion()
    assert np.array_equal(f.read_results()[..., 0], fresh.read_results()[..., 0])


@pytest.mark.parametrize("order", [1, 2])
def test_reopen_after_raising_max_iter(engine, order):
    """Raising the limit continues the undecided samples.  First-order BLA gives exactly the
    fresh picture; second order splits its skips differently at the old limit (different
    truncation), which a few chaotic samples amplify - they must stay rare."""
    from mandelscope.renderer import GPUEngine, EngineOptions
    engine = GPUEngine(engine.ctx, EngineOptions(bla_order=order))
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(1e-6), 0.0, 400)
    W, H = 48, 32
    ref = compute_reference(str(v.re), str(v.im), 160, 3000)
    ref.ref_id = 3
    g = GPURef(engine.ctx, ref, False)
    fp = view_params(v, v.re, v.im, W, H, with_interior=False)
    b = GPUBla(engine.ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax, order=order), False)
    f = Field(engine, W, H, False)
    f.configure(fp, g, b)
    f.run_to_completion()
    assert np.sum(f.read_results()[..., 0] == 0xFFFFFFFF) > 0
    fp.max_iter = 3000
    f.reopen_maxiter()
    f.configure(fp, g, b)
    f.run_to_completion()
    fresh = Field(engine, W, H, False)
    fresh.configure(fp, g, b)
    fresh.run_to_completion()
    same = f.read_results()[..., 0] == fresh.read_results()[..., 0]
    if order == 1:
        assert same.all()
    else:
        assert same.mean() > 0.97, same.mean()


def test_tiled_render_equals_single_tile(engine):
    v = misiurewicz_view(100, 5000)
    # fixed colour mapping: with 'auto' a single tile takes its histogram from all its samples,
    # several tiles from a low-resolution pre-pass (slightly different colours by design)
    cs = ColorSettings(slope_strength=0.5, mapping=1)
    a = FrameRenderer(engine, RenderSettings(96, 54, ss=2, max_tile_samples=10**8)).render(v, cs)
    b = FrameRenderer(engine, RenderSettings(96, 54, ss=2, max_tile_samples=96 * 4 * 7)).render(v, cs)
    assert a.shape == (54, 96, 3)
    assert np.array_equal(a, b)


def test_image_orientation(engine):
    """Top of the image is +imaginary, right is +real."""
    v = View(mpfr("-0.75"), mpfr("0.75"), math.log2(0.6), 0.0, 300)
    cs = ColorSettings(de_strength=0.0, interior=(0.0, 0.0, 0.0))
    img = FrameRenderer(engine, RenderSettings(160, 160, ss=1)).render(v, cs).astype(int)
    inside = img.sum(axis=2) == 0
    # rows near the top (im ~ 1.3) are outside the set, rows near the bottom (im ~ 0.2) mostly inside
    assert inside[:20].mean() < 0.02
    assert inside[-40:].mean() > 0.2
    # quick check of the left/right orientation: the region right of the centre (re ~ -0.2..0) has the
    # main cardioid, left (re ~ -1.3) is mostly outside
    assert inside[100:, 120:].mean() > inside[100:, :40].mean()


def test_f32_and_f64_agree_at_depth(engine):
    v = misiurewicz_view(320, 5000)
    r32, _ = render_view(engine, v, 48, 32, double=False)
    r64, _ = render_view(engine, v, 48, 32, double=True)
    assert np.mean(r32[..., 0] == r64[..., 0]) > 0.97


def test_shift_keeps_samples_counted(engine):
    """After a GPU shift, the reused samples count as done immediately (no recompute)."""
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(3e-5), 0.0, 2000)
    W, H = 160, 90
    f, _, _ = _field(engine, v, W, H)
    f.run_to_completion()
    v2 = v.offset(7, -4, H)
    f.shift(-7, 4)
    f.configure(view_params(v2, v.re, v.im, W, H), f.gref, f.gbla)
    f.step(0.0002, 0.0001)
    assert f.done_count >= (W - 7) * (H - 4)
    assert not f.complete or f.done_count == f.N


def test_bla_table_swap_keeps_results(engine):
    """Replacing the BLA table in the middle of a field (as the app does while panning)
    gives the same result as an uninterrupted render."""
    v = misiurewicz_view(100, 5000)
    W, H = 64, 40
    a, ref, bla = _field(engine, v, W, H, budget=64)
    a.step(0.001, 0.0005)
    fp = a.params
    bla2 = build_bla(ref.zm, ref.ze, fp.log2_cmax + 2.0, -24.0)
    a.configure(fp, a.gref, GPUBla(engine.ctx, bla2, False))
    a.run_to_completion()
    b, _, _ = _field(engine, v, W, H)
    b.run_to_completion()
    ra, rb = a.read_results()[..., 0].astype(np.int64), b.read_results()[..., 0].astype(np.int64)
    assert np.mean(ra == rb) > 0.97


def test_stripes_and_looks_render(engine):
    """Every look renders a non-trivial image (stripe / DE variants included)."""
    from mandelscope.palettes import LOOKS
    from mandelscope.renderer import AutoContrast
    v = View.from_strings("-0.7453", "0.1127", math.log2(0.0065), 0.0, 2000)
    W, H = 96, 54
    ref = compute_reference(str(v.re), str(v.im), v.needed_precision(H) + 32, v.max_iter)
    ref.ref_id = 7
    fp = view_params(v, v.re, v.im, W, H, with_de=True, with_interior=True, with_stripes=True)
    f = Field(engine, W, H, False)
    f.reset()
    f.configure(fp, GPURef(engine.ctx, ref, False), GPUBla(engine.ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax), False))
    f.run_to_completion()
    res = f.read_results()
    stripe = res[..., 5].view(np.float32)
    esc = res[..., 0] != 0xFFFFFFFF
    assert esc.any() and np.isfinite(stripe[esc]).all()
    assert 0.0 <= float(stripe[esc].min()) and float(stripe[esc].max()) <= 1.0 + 1e-5
    auto = AutoContrast(engine.ctx)
    auto.update(f)
    tex = engine.ctx.texture((W, H), 3)
    fbo = engine.ctx.framebuffer(color_attachments=[tex])
    for name in LOOKS:
        cs = ColorSettings.from_look(name)
        engine.display.render(f, fbo, (0, 0, W, H), 1.0, cs, fallback=False, auto=auto, max_iter_show=v.max_iter)
        img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)
        assert img.std() > 10, name
    fbo.release()
    tex.release()


def test_reopen_then_shift_equals_fresh_render(engine):
    """Raising the iteration limit and panning in the same frame keeps every sample exact."""
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(1e-6), 0.0, 400)
    W, H = 48, 32
    ref = compute_reference(str(v.re), str(v.im), 160, 3000)
    ref.ref_id = 5
    g = GPURef(engine.ctx, ref, False)
    fp = view_params(v, v.re, v.im, W, H, with_interior=False)
    b = GPUBla(engine.ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax + 2.0), False)
    f = Field(engine, W, H, False)
    f.configure(fp, g, b)
    f.run_to_completion()
    v2 = v.offset(4, -2, H)
    v2.max_iter = 3000
    fp2 = view_params(v2, v.re, v.im, W, H, with_interior=False)
    f.reopen_maxiter()
    f.shift(-4, 2)
    f.configure(fp2, g, b)
    f.run_to_completion()
    fresh = Field(engine, W, H, False)
    fresh.configure(fp2, g, b)
    fresh.run_to_completion()
    assert f.done_count == f.N
    assert np.array_equal(f.read_results()[..., 0], fresh.read_results()[..., 0])


@pytest.mark.parametrize("octaves", [1, 2, -1], ids=["in2", "in4", "out2"])
def test_zoom_remap_equals_fresh_render(engine, octaves):
    """Zooming by a power of two keeps the samples that land on the new grid (exactly)."""
    from mandelscope.renderer import grid_map, snap_view_to_grid
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(3e-5), 0.0, 2000)
    W, H = 64, 48
    f, ref, bla = _field(engine, v, W, H)
    f.run_to_completion()
    target = v.zoom_at(W * 0.3, H * 0.7, octaves, W, H)        # zoom at a point, like the Z key
    t = snap_view_to_grid(v, target, W, H)
    dx, dy = target.delta_in_pixels(t, H)                      # in samples of the new grid
    assert math.hypot(dx, dy) <= 0.75                           # moved by less than a sample
    m = grid_map(v, t, W, H)
    assert m not in (None, "same")
    mul, div, add, de_scale = m
    fp2 = view_params(t, v.re, v.im, W, H)
    f.remap(mul, div, add, de_scale)
    f.configure(fp2, f.gref, f.gbla)
    kept = int(f._read_counters()[1])   # counted on the GPU by the remap
    expect = (W * H) // 4 ** abs(octaves) if octaves > 0 else (W * H) // 4
    assert kept >= 0.8 * expect                                 # a quarter (2x) / 16th (4x) is reused
    f.run_to_completion()
    fresh = Field(engine, W, H, False)
    fresh.configure(fp2, f.gref, f.gbla)
    fresh.run_to_completion()
    a, b = f.read_results(), fresh.read_results()
    assert np.array_equal(a[..., 0], b[..., 0])
    assert np.array_equal(a[..., 1], b[..., 1])                 # smooth iteration fraction
    da, db = a[..., 2].view(np.float32), b[..., 2].view(np.float32)
    ok = (a[..., 0] < 0xFFFFFFFE) & (db > 0)
    assert np.allclose(da[ok], db[ok], rtol=1e-4)              # distance estimates rescaled to the new pixels


def test_adaptive_supersampling_matches_full(engine, locations):
    """Extra sub-samples only where needed: (almost) the same picture as computing them all."""
    from mandelscope.mp import View as _V
    d = json.loads((locations / "seahorse_valley.json").read_text())
    v = _V.from_dict(d["view"] if "view" in d else d)
    cs = ColorSettings(mapping=1)
    full = FrameRenderer(engine, RenderSettings(160, 90, ss=2, adaptive=False))
    ad = FrameRenderer(engine, RenderSettings(160, 90, ss=2, adaptive=True, max_tile_samples=160 * 4 * 31))
    a = full.render(v, cs).astype(int)
    b = ad.render(v, cs).astype(int)
    assert ad.stats["adaptive"] and ad.stats["refined"] < 0.9   # smooth parts were filled, not computed
    diff = np.abs(a - b).max(axis=2)
    assert diff.max() <= 6, diff.max()
    assert np.mean(diff > 2) < 0.005
    full.release(); ad.release()


def test_auto_colours_from_the_frame_itself(engine, locations):
    """'auto' colours of a one-tile frame come from its own samples (no pre-pass) and are close
    to the pre-pass version."""
    v = View.from_dict(json.loads((locations / "spiral_1e-100.json").read_text())["view"])
    cs = ColorSettings()
    one = FrameRenderer(engine, RenderSettings(128, 72, ss=1))
    a = one.render(v, cs).astype(int)
    assert one.stats["t_prepass"] < 0.05 and one.stats["syncs"] > 0
    tiled = FrameRenderer(engine, RenderSettings(128, 72, ss=1, max_tile_samples=128 * 20))
    b = tiled.render(v, cs).astype(int)
    d = np.abs(a - b).max(axis=2)
    assert np.mean(d) < 6
    one.release(); tiled.release()


@pytest.mark.gpu
@pytest.mark.parametrize("double", [False, True])
def test_julia_kernel_matches_direct_iteration(engine, double):
    """Julia sets: escape counts of the GPU kernel = plain complex iteration (NumPy)."""
    import math
    from gmpy2 import mpfr
    from mandelscope.renderer import Field, julia_params
    from mandelscope.mp import View
    c = (-0.8, 0.156)
    W, H = 96, 54
    v = View(mpfr(0.1), mpfr(-0.05), 0.6, 0.3, 400)
    fp = julia_params(v, W, H, double=double, with_de=True, with_interior=True)
    f = Field(engine, W, H, double)
    f.configure(fp, None, None)
    engine.setup_julia(f, v, c, W, H, fp)
    assert f.run_to_completion()
    r = f.read_results()
    h = 2 ** v.pixel_log2(H)
    ys, xs = np.mgrid[0:H, 0:W]
    qx, qy = xs + 0.5 - W / 2, -(ys + 0.5 - H / 2)
    ca, sa = math.cos(v.angle), math.sin(v.angle)
    z = (0.1 + h * (ca * qx - sa * qy)) + 1j * (-0.05 + h * (sa * qx + ca * qy))
    n = np.zeros(z.shape, int)
    alive = np.ones(z.shape, bool)
    for i in range(400):
        esc = alive & (np.abs(z) ** 2 > fp.bail2)
        n[esc] = i
        alive &= ~esc
        z[alive] = z[alive] ** 2 + complex(*c)
    gn = r[:, :, 0].astype(np.int64)
    both = (gn < 0xFFFFFFFE) & ~alive
    assert both.sum() > 0.8 * (~alive).sum()
    assert (gn[both] == np.maximum(n[both], 1)).mean() > 0.97     # (chaotic points near the set may differ)
    f.release()


@pytest.mark.gpu
def test_julia_zoom_remap_and_still(engine):
    """Julia pictures reuse samples exactly like Mandelbrot ones (zoom x2 remap = fresh render),
    and still images / sequences render them (FrameRenderer.julia)."""
    from gmpy2 import mpfr
    from mandelscope.renderer import Field, julia_params, grid_map, snap_view_to_grid, ColorSettings
    from mandelscope.offline import FrameRenderer, RenderSettings, render_sequence_iter
    from mandelscope.sequence import Sequence, Keyframe
    from mandelscope.mp import View
    c = (-0.7453, 0.1127)
    W, H = 128, 72
    v = View(mpfr(-0.2), mpfr(0.1), -1.0, 0.0, 600)
    fp = julia_params(v, W, H, with_de=True, with_interior=True)
    f = Field(engine, W, H, False)
    f.configure(fp, None, None)
    engine.setup_julia(f, v, c, W, H, fp)
    f.run_to_completion()
    t = v.copy()
    t.log2r -= 1.0
    t = snap_view_to_grid(v, t.offset(3.3, -2.2, H), W, H)
    m = grid_map(v, t, W, H)
    mul, div, add, de_scale = m
    fp2 = julia_params(t, W, H, with_de=True, with_interior=True)
    f.remap(mul, div, add, de_scale)
    f.configure(fp2, None, None)
    engine.setup_julia(f, t, c, W, H, fp2)
    f.run_to_completion()
    fresh = Field(engine, W, H, False)
    fresh.configure(fp2, None, None)
    engine.setup_julia(fresh, t, c, W, H, fp2)
    fresh.run_to_completion()
    a, b = f.read_results(), fresh.read_results()
    # (z0 is rounded differently from the old and the new view centre: a few chaotic points
    # next to the set may take another path - equally valid samples)
    assert (a[:, :, 0] == b[:, :, 0]).mean() > 0.995
    f.release()
    fresh.release()
    cs = ColorSettings()
    fr = FrameRenderer(engine, RenderSettings(W, H, ss=2))
    fr.julia = c
    img = fr.render(v, cs)
    assert img.shape == (H, W, 3) and img.std() > 5
    fr.release()
    # a Julia timeline (sequence meta) renders as video frames, also the fast way
    v1 = v.copy()
    v1.log2r -= 2.0
    seq = Sequence([Keyframe(v, 0.0), Keyframe(v1, 1.0)], fps=6)
    seq.meta["julia"] = list(c)
    frames = [f["image"] for f in render_sequence_iter(engine, seq, None, RenderSettings(W, H, ss=1), cs, fps=6.0)
              if f["phase"] == "frame"]
    fast = [f["image"] for f in render_sequence_iter(engine, seq, None, RenderSettings(W, H, ss=1, fast=True), cs,
                                                     fps=6.0) if f["phase"] == "frame"]
    assert len(frames) == len(fast) == 7
    assert not np.array_equal(frames[0], FrameRenderer(engine, RenderSettings(W, H, ss=1)).render(v, cs))
    diffs = [np.abs(x.astype(int) - y.astype(int)).mean() for x, y in zip(frames, fast)]
    assert max(diffs) < 8, diffs


@pytest.mark.parametrize("double", [False, 2], ids=["f32", "ff"])
def test_periodic_reference_wraps_like_tiling(engine, double):
    """A periodic reference stores its cycle once and the kernel wraps the index; the pictures
    equal those of the reference tiled up to the iteration limit (as 1.0.0 stored it)."""
    import dataclasses
    from mandelscope.nucleus import newton_nucleus
    from gmpy2 import mpc
    from mandelscope.mp import to_str
    c = newton_nucleus(mpc(-1.7548776662466927, 0), 3, 300)        # the period-3 minibrot
    re, im = to_str(c.real, 300), to_str(c.imag, 300)
    v = View.from_strings(re, im, math.log2(0.01), 0.3, 20000)
    W, H = 128, 96
    ref = compute_reference(re, im, 300, v.max_iter, chunk=256)
    assert ref.period == 3 and ref.length < 4000
    need = v.max_iter + 1 - ref.length
    tail = slice(ref.length - ref.period, ref.length)
    tiled = dataclasses.replace(ref, period=0, **{k: np.concatenate([getattr(ref, k), np.tile(getattr(ref, k)[tail],
                                                                     need // ref.period + 1)[:need]])
                                                  for k in ("zf", "zm", "ze")})
    out = []
    for r in (ref, tiled):
        fp = view_params(v, v.re, v.im, W, H, double=double, with_de=True, with_interior=False)
        bla = build_bla(r.zm, r.ze, fp.log2_cmax, -45.0 if double else -24.0)
        f = Field(engine, W, H, double)
        f.configure(fp, GPURef(engine.ctx, r, double), GPUBla(engine.ctx, bla, double))
        assert f.run_to_completion()
        out.append(f.read_results()[..., 0].copy())
        f.release()
    long = (out[1] > ref.length) & (out[1] < 0xFFFFFFFE)
    assert (out[1] == 0xFFFFFFFF).sum() > 100 and long.sum() > 10      # (both cases do occur)
    assert np.mean(out[0] == out[1]) > 0.995


def _ff_lib(fma: bool) -> str:
    from mandelscope.renderer import _src
    src = _src("perturb_ff.comp")
    lib = src[src.index("// ---------------------------------------------------------------- float-float"):
              src.index("// complex floatexp with a float-float mantissa")]
    return ("#define FF_FMA 1\n" if fma else "") + lib


@pytest.mark.parametrize("fma", [False, True], ids=["dekker", "fma"])
def test_float_float_arithmetic(engine, fma):
    """The float-float operations of the precise mode keep ~48 bits on this GPU (the compiler
    must not rewrite the error-free transforms; fma must round once where it is used)."""
    from mandelscope.renderer import ff_complex
    if fma and not engine.fma_exact:
        pytest.skip("fma() rounds twice on this GPU: the kernel uses Dekker's products")
    ctx = engine.ctx
    prog = ctx.compute_shader("""#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer A { vec4 xa[]; };
layout(std430, binding = 1) readonly buffer B { vec4 xb[]; };
layout(std430, binding = 2) writeonly buffer O { vec4 o[]; };
""" + _ff_lib(fma) + """
void main() {
    uint i = gl_GlobalInvocationID.x;
    vec4 p = xa[i], q = xb[i];
    o[3u * i] = c_mul(p, q);
    o[3u * i + 1u] = c_add(p, q);
    o[3u * i + 2u] = c_sqr(p);
}
""")
    rng = np.random.default_rng(1)
    N = 4096
    fa = ff_complex((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.exp(rng.uniform(-8, 8, N)))
    fb = ff_complex((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.exp(rng.uniform(-8, 8, N)))

    def val(f):
        return (f[:, 0].astype(np.float64) + f[:, 2]) + 1j * (f[:, 1].astype(np.float64) + f[:, 3])
    za, zb = val(fa), val(fb)
    bufs = [ctx.buffer(fa.tobytes()), ctx.buffer(fb.tobytes()), ctx.buffer(reserve=N * 48)]
    for i, b in enumerate(bufs):
        b.bind_to_storage_buffer(i)
    prog.run(N // 64)
    o = np.frombuffer(bufs[2].read(), dtype=np.float32).reshape(N, 3, 4)
    for k, ref in enumerate((za * zb, za + zb, za * za)):
        err = np.abs(val(o[:, k]) - ref) / np.abs(ref)
        assert np.median(err) < 2.0 ** -46 and err.max() < 2.0 ** -43, (k, np.log2(np.median(err)), np.log2(err.max()))
    for b in bufs:
        b.release()
    prog.release()


@pytest.mark.parametrize("name", ["seahorse_1e-12", "minibrot_p6006"])
def test_float_float_agrees_with_float64(engine, locations, name):
    """Where float32 and float64 pictures differ (chaotic, high-iteration places), the float-float
    picture is the float64 one."""
    if not engine.has_fp64:
        pytest.skip("no fp64")
    v = View.from_dict(json.loads((locations / f"{name}.json").read_text())["view"])
    v.max_iter = min(v.max_iter, 20000)
    r = {d: render_view(engine, v, 96, 72, double=d)[0][..., 0] for d in (False, True, 2)}
    same32 = np.mean(r[False] == r[True])
    sameff = np.mean(r[2] == r[True])
    assert sameff > 0.98 and sameff > same32 + 0.05, (same32, sameff)


def test_float_float_falls_back_for_the_ship(engine):
    """(No float-float Burning Ship kernel: float64 there, the same picture.)"""
    from mandelscope.renderer import ColorSettings as CS
    v = SHIP_CASES["armada"]()
    imgs = []
    for d in (True, 2):
        fr = FrameRenderer(engine, RenderSettings(64, 48, ss=1, double=d))
        fr.formula = "ship"
        imgs.append(fr.render(v, CS()))
        assert fr.prec is True or not engine.has_fp64
        fr.release()
    assert np.array_equal(imgs[0], imgs[1])


def ship_tip_view(depth10, max_iter, angle=0.3):
    """Next to c = -2, the tip of the Burning Ship's needle: the reference's imaginary part is
    far below its real part for hundreds of iterations (the case where diffabs needs the exact
    product XY)."""
    l2 = -depth10 / 0.30103
    p = int(-l2) + 100
    re = mpfr(-2, p) + gmpy2.mul_2exp(mpfr(0.37, p), int(l2))
    im = gmpy2.mul_2exp(mpfr(0.21, p), int(l2))
    return View(re, im, l2, angle, max_iter)


# mini Burning Ships (nucleus finder, chained) of period 39, 228 and 1730
SHIP_MINIS = {
    39: ("-1.77832935517465558049355033", "0.02227250757112231451758436", -36.03),
    228: ("-1.7783293552065527431607898315243354394675170011191",
          "0.0222725076051085208337975420903878665658485832815", -109.98),
    1730: ("-1.77832935520655274316078983152433552944114484703896473204234323193131863090770760359210344477794781133"
           "269509398062440677814",
           "0.02227250760510852083379754209038939566931652938168130986958595886858077694815502685631561122100641801"
           "388282129894674603499", -352.48),
}


def ship_mini_view(period, zoom_out, max_iter, angle=0.2):
    re, im, l2 = SHIP_MINIS[period]
    return View.from_strings(re, im, l2 + zoom_out, angle, max_iter)


SHIP_CASES = {
    "home": lambda: View(mpfr("-0.45", 64), mpfr("0.5", 64), math.log2(1.3), 0.0, 500),
    "armada": lambda: View(mpfr("-1.762", 64), mpfr("0.028", 64), math.log2(0.04), 0.0, 1000),
    "tip_1e-300": lambda: ship_tip_view(300, 5000),
    "tip_1e-1000": lambda: ship_tip_view(1000, 9000),
    "mini39_1e-10": lambda: ship_mini_view(39, 3.0, 4000),
    "mini228_1e-32": lambda: ship_mini_view(228, 4.0, 12000),
    "mini1730_1e-105": lambda: ship_mini_view(1730, 5.0, 60000),
}


SHIP_SMOOTH = ("home", "armada", "tip_1e-300", "tip_1e-1000")


@pytest.mark.parametrize("double", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("name", list(SHIP_CASES))
def test_ship_matches_ground_truth(engine, name, double):
    """The Burning Ship by perturbation (diffabs, rebasing, floatexp) = MPFR iteration."""
    v = SHIP_CASES[name]()
    res, info = render_view(engine, v, 64, 48, double=double, formula="ship")
    assert info["complete"], info
    st = stats(res)
    assert st["unfinished"] == 0 and st["min"] is not None, st
    cmp = compare(v, res, nsamp=30, tol_px=1e-3 if not double else 1e-6, formula="ship")
    assert cmp["hard"] == 0, [d for d in cmp["details"] if d[0] == "HARD"][:3]
    # (the exterior around mini ships is chaotic: counts change within 10^-6 of a pixel, so
    # the "chaotic" verdict there - the GPU count is among the counts of nearby points or
    # the count itself is unstable - is all a correct renderer can achieve)
    assert cmp["exact"] >= (0.8 if name in SHIP_SMOOTH else 0.3) * cmp["n"], cmp


def _ship_z(c, n, prec=300):
    from gmpy2 import mpc
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        x = y = mpfr(0)
        for _ in range(n):
            x, y = x * x - y * y + c.real, 2 * abs(x * y) - c.imag
        return mpc(x, y)


@pytest.mark.parametrize("view", ["armada", "mini228_1e-32"])
def test_ship_distance_estimate(engine, view):
    """The Burning Ship's dz/dc is a real 2x2 matrix (v -> P v + Q conj v in the shader): the
    distance estimate |z|^2 log|z| / |J^T z| and the light direction J^T z match finite
    differences of MPFR orbits; the variants agree on the iteration counts."""
    from accuracy import RES_INTERIOR
    from mandelscope import groundtruth as gt
    v = SHIP_CASES[view]()
    W, H = 48, 36
    res, _ = render_view(engine, v, W, H, double=True, formula="ship")
    plain, _ = render_view(engine, v, W, H, double=True, de=False, interior=False, formula="ship")
    striped, _ = render_view(engine, v, W, H, double=True, formula="ship", stripes=True)
    n = res[..., 0]
    both = (n < RES_INTERIOR) & (plain[..., 0] < RES_INTERIOR) & (striped[..., 0] < RES_INTERIOR)
    assert (n[both] == plain[..., 0][both]).mean() > 0.99
    assert (n[both] == striped[..., 0][both]).mean() > 0.99
    de = res[..., 2].view(np.float32)
    ang = res[..., 3].view(np.float32)
    rng = np.random.default_rng(3)
    pix = 2.0 ** v.pixel_log2(H)
    checked = 0
    for y, x in rng.permutation(np.argwhere(n < RES_INTERIOR))[:40]:
        if checked >= 10:
            break
        prec = v.needed_precision(H) + 120
        c = gt.sample_c(v, x, y, W, H, prec)
        k = int(n[y, x])
        if gt.iterate_ship(c, v.max_iter, prec)[0] != k:
            continue                # (chaotic: another count, another orbit)
        z = _ship_z(c, k, prec)
        with gmpy2.context(gmpy2.get_context(), precision=prec):
            eps = gmpy2.mul_2exp(mpfr(1), int(v.pixel_log2(H) - 40))
            cx, cy = c + eps, c + gmpy2.mpc(0, eps)
        zx = _ship_z(cx, k, prec)
        zy = _ship_z(cy, k, prec)
        J = np.array([[float((zx.real - z.real) / eps), float((zy.real - z.real) / eps)],
                      [float((zx.imag - z.imag) / eps), float((zy.imag - z.imag) / eps)]])
        zf = np.array([float(z.real), float(z.imag)])
        g = J.T @ zf
        r = float(np.hypot(*zf))
        de_ref = r * r * math.log(r) / np.hypot(*g) / pix
        # (float derivatives over thousands of non-conformal steps: within a few % where the
        # estimate matters for the picture, i.e. above ~1/100 pixel)
        tol = 0.02 if de_ref > 0.01 else 0.3
        assert abs(de[y, x] / de_ref - 1) < tol, (x, y, de[y, x], de_ref)
        da = (ang[y, x] - math.atan2(g[1], g[0]) + math.pi) % (2 * math.pi) - math.pi
        assert abs(da) < tol, (x, y, ang[y, x], g)
        checked += 1
    assert checked >= 6


def julia_deep_view(c, depth10, max_iter, angle=0.4, off=0.31):
    """A view next to the repelling fixed point beta of z^2 + c, where the Julia set repeats
    itself at every scale; the centre is computed at full precision (a 53-bit centre would
    land somewhere featureless)."""
    l2 = -depth10 / 0.30103
    p = int(depth10 * 3.33) + 120
    with gmpy2.context(gmpy2.get_context(), precision=p):
        cc = gmpy2.mpc(mpfr(c[0], p), mpfr(c[1], p))
        b = (1 + gmpy2.sqrt(1 - 4 * cc)) / 2
        o = gmpy2.mul_2exp(mpfr(off, p), int(l2))
        return View(b.real + o, b.imag - o, l2, angle, max_iter)


JULIA_DEEP = {
    "basilica_1e-6": (("-0.8", "0.156"), 6, 3000),
    "basilica_1e-30": (("-0.8", "0.156"), 30, 5000),
    "basilica_1e-100": (("-0.8", "0.156"), 100, 8000),
    "rabbit_1e-40": (("-0.123", "0.745"), 40, 5000),
}


@pytest.mark.gpu
@pytest.mark.parametrize("double", [False, True, 2], ids=["f32", "f64", "ff"])
@pytest.mark.parametrize("name", list(JULIA_DEEP))
def test_deep_julia_matches_ground_truth(engine, name, double):
    """Deep Julia zooms by perturbation (reference A at the view's point, B = the critical
    orbit, rebasing to B) give the escape counts of MPFR iteration of each sample."""
    from accuracy import render_julia
    c, depth, mi = JULIA_DEEP[name]
    v = julia_deep_view(c, depth, mi)
    res, info = render_julia(engine, v, c, 64, 48, double=double)
    assert info["complete"], info
    st = stats(res)
    assert st["unfinished"] == 0 and st["min"] is not None, st
    cmp = compare(v, res, nsamp=30, tol_px=1e-3 if not double else 1e-6, julia=c)
    assert cmp["hard"] == 0, [d for d in cmp["details"] if d[0] == "HARD"][:3]
    assert cmp["exact"] >= 0.8 * cmp["n"], cmp
    if depth >= 30:
        assert info["rebases"] > 0, info
    if depth >= 100:
        # (near the repelling point deltas grow ~3x per step: only a deep view leaves room
        # for 64-step skips along reference A)
        assert info["bla_skips"] > 0, info


@pytest.mark.gpu
def test_deep_julia_stills_and_movies(engine):
    """Stills and movies of Julia sets switch to perturbation past the float32 limit: frames
    on both sides of the switch and far beyond the old float64 limit (2^-39) show detail."""
    from mandelscope.offline import render_sequence_iter
    from mandelscope.renderer import julia_min_log2r
    from mandelscope.sequence import Sequence, Keyframe
    c = ("-0.8", "0.156")
    W, H = 96, 54
    cs = ColorSettings()
    fr = FrameRenderer(engine, RenderSettings(W, H, ss=1))
    fr.julia = (float(c[0]), float(c[1]))
    v = julia_deep_view(c, 60, 6000)
    img = fr.render(v, cs)
    assert fr._deep and img.std() > 5
    assert v.log2r < julia_min_log2r(True, H) - 100
    fr.release()
    v0 = julia_deep_view(c, 3, 2000)
    v1 = julia_deep_view(c, 6, 3000)
    assert v0.log2r > julia_min_log2r(False, H) + 1.0 > v1.log2r
    seq = Sequence([Keyframe(v0, 0.0), Keyframe(v1, 1.0)], fps=4)
    seq.meta["julia"] = [float(c[0]), float(c[1])]
    for fast in (False, True):
        frames = [f["image"] for f in render_sequence_iter(engine, seq, None, RenderSettings(W, H, ss=1, fast=fast),
                                                           cs, fps=4.0) if f["phase"] == "frame"]
        assert len(frames) == 5
        assert min(f.std() for f in frames) > 5, [f.std() for f in frames]


@pytest.mark.gpu
@pytest.mark.parametrize("consistent", [True, False])
def test_refining_field_shows_no_dot_grid(engine, consistent):
    """While a field refines, its finished samples form a sparse lattice; drawn sharp between
    pixels of the older picture they make a dot grid.  With the consistent compositing the
    older (better) picture is shown everywhere until the samples are dense."""
    from mandelscope.renderer import Field, ColorSettings, RESULT_WORDS
    W, H = 64, 40
    f = Field(engine, W, H, False)
    res = np.zeros((H, W, RESULT_WORDS), np.uint32)
    yy, xx = np.mgrid[0:H, 0:W]
    lattice = (yy % 8 == 0) & (xx % 8 == 0)
    res[lattice, 0] = 40                                   # finished: escaped after 40 iterations
    f.result.write(res.tobytes())
    f.complete = False
    prev = engine.ctx.texture((W, H), 4, dtype="f2")
    prev.write(np.tile(np.array([0.2, 0.6, 0.3, 1.0], np.float16), W * H).tobytes())
    out = engine.ctx.texture((W, H), 4, dtype="f2")
    fbo = engine.ctx.framebuffer(color_attachments=[out])
    disp = engine.display
    old = disp.consistent
    disp.consistent = consistent
    try:
        disp.render(f, fbo, (0, 0, W, H), 1.0, ColorSettings(mapping=1), prev_tex=prev, prev_xform=(1.0, (0.0, 0.0)),
                    prev_size=(W, H), prev_q=0.5, field_q=1.0, fallback=True)
    finally:
        disp.consistent = old
    img = np.frombuffer(out.read(), dtype=np.float16).reshape(H, W, 4)[::-1, :, :3].astype(np.float32)
    differs = np.abs(img - np.array([0.2, 0.6, 0.3], np.float32)).max(axis=2) > 0.02
    if consistent:
        assert not differs.any()                           # one source: the older picture
    else:
        assert differs[lattice].mean() > 0.9 and not differs[~lattice].any()   # v3.6: the dot grid
    for o in (fbo, out, prev):
        o.release()
    f.release()


def test_iteration_octaves_match_the_results(engine, locations):
    """The GPU octave statistics agree with the per-sample results (inside points keep the
    iteration at which they were recognised)."""
    from mandelscope.control import iteration_advice
    d = json.loads((locations / "seahorse_valley.json").read_text())["view"]
    v = View.from_strings(d["re"], d["im"], d["log2r"], 0.0, 2000)
    v = v.zoom_at(0.62 * 1600, 0.38 * 900, 3.0, 1600, 900).zoom_at(0.40 * 1600, 0.60 * 900, -2.0, 1600, 900)
    v.max_iter = 2115
    W, H = 96, 54
    ref = compute_reference(str(v.re), str(v.im), v.needed_precision(H) + 32, 5000)
    ref.ref_id = 7
    fp = view_params(v, v.re, v.im, W, H, with_interior=True)
    f = Field(engine, W, H, False)
    f.configure(fp, GPURef(engine.ctx, ref, False), GPUBla(engine.ctx, build_bla(ref.zm, ref.ze, fp.log2_cmax), False))
    f.run_to_completion()
    st = f.iteration_octaves(v.max_iter)
    r = f.read_results()
    n = r[..., 0].astype(np.int64).ravel()
    det = r[..., 1].copy().view(np.float32).ravel()
    L = v.max_iter
    esc = n[(n > 0) & (n < 0xFFFFFFFE)]
    ins = np.maximum(det[n == 0xFFFFFFFE].astype(np.int64), 1)
    oct_ = lambda x: np.clip(np.floor(np.log2(L / x.astype(np.float64))), 0, 7).astype(int)
    assert st["esc"] == np.bincount(oct_(esc), minlength=8).tolist()
    assert st["int"] == np.bincount(oct_(ins), minlength=8).tolist()
    assert st["und"] == int(np.sum(n == 0xFFFFFFFF)) and st["done"] == W * H
    assert ins.min() > 1 and ins.max() <= L                     # detection iterations are stored
    # this view (a large piece of the main cardioid) needs no raise: the 1/3 of undecided
    # samples are inside points converging slowly (v3.7 doubled up to 128x here)
    assert iteration_advice(st, 1.0) == (0, "inside")


def test_step_stays_inside_its_time_slice(engine):
    """Field.step stops before a dispatch that would end after the time slice (v3.7 overshot
    10 ms slices by up to half a slice on new pictures)."""
    import time as _t
    from mandelscope.renderer import GPUEngine, EngineOptions

    def overshoot(predictive):
        eng = GPUEngine(engine.ctx, EngineOptions(predictive_budget=predictive))
        f, _, _ = _field(eng, misiurewicz_view(100, 20000), 768, 480)
        f.chunk = 4096
        over = []
        while True:
            t0 = _t.perf_counter()
            done = f.step(0.03, dispatch_target=0.006)
            over.append(_t.perf_counter() - t0 - 0.03)
            if done:
                break
        f.release()
        return float(np.median(over[:-1])) if len(over) > 2 else None
    new, old = overshoot(True), overshoot(False)
    if new is not None and old is not None:
        assert new < 0.004 and new < old, (new, old)


def test_earlier_gpu_work_does_not_shrink_dispatches(engine, monkeypatch):
    """GPU work queued before a time slice (the previous frame's display, panels, capture) must
    not count as the field's dispatch time: v3.8 measured it into the first batch, shrank the
    dispatches to the minimum and then allowed one per frame (pictures 4-8x slower in the
    user's study).  The backlog is simulated: whoever waits first for the GPU pays 5 ms."""
    import time as _t
    from mandelscope.renderer import GPUEngine, EngineOptions
    ctx = engine.ctx
    pending = {"t": 0.0}
    real_finish, real_read = ctx.finish, Field._read_counters

    def pay():
        if pending["t"]:
            _t.sleep(pending["t"])
            pending["t"] = 0.0

    monkeypatch.setattr(ctx, "finish", lambda: (real_finish(), pay())[0])
    monkeypatch.setattr(Field, "_read_counters", lambda self: (pay(), real_read(self))[1])

    def samples_per_slice(**opts):
        eng = GPUEngine(ctx, EngineOptions(**opts))
        f, _, _ = _field(eng, misiurewicz_view(4, 400), 1024, 640)
        prog = []
        for _ in range(40):
            pending["t"] = 0.005
            d0 = f.done_count
            if f.step(0.010, dispatch_target=0.005):
                break
            prog.append(f.done_count - d0)
        f.release()
        return float(np.median(prog[1:])) if len(prog) > 2 else float("inf")
    fixed = samples_per_slice()
    v38 = samples_per_slice(sync_before_step=False)
    assert fixed > 2.0 * v38, (fixed, v38)


