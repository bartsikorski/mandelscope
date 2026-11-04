"""CPU-side numerics: reference orbits, floatexp, BLA, views, sequences, nucleus finder."""
import math

import numpy as np
import gmpy2
import pytest
from gmpy2 import mpfr, mpc

from mandelscope.mp import View, to_str
from mandelscope.reference import compute_reference, EXP_ZERO
from mandelscope.bla import build_bla
from mandelscope.nucleus import newton_nucleus, size_estimate, find_minibrot, misiurewicz
from mandelscope.sequence import Sequence, Keyframe, PathRecorder


def direct_orbit(re, im, prec, n):
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = mpc(mpfr(re, prec), mpfr(im, prec))
        z = mpc(0)
        out = [z]
        for _ in range(n):
            z = z * z + c
            out.append(z)
        return out


def test_reference_matches_direct_iteration():
    re, im = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"
    ref = compute_reference(re, im, 200, 3000, allow_period=False)
    assert ref.length == 3001 and not ref.escaped
    orbit = direct_orbit(re, im, 200, 3000)
    for i in (0, 1, 2, 17, 500, 2999, 3000):
        z = complex(orbit[i])
        assert abs(ref.zf[i] - z) <= 1e-15 * max(1.0, abs(z))
        # floatexp representation is consistent
        assert abs(ref.zm[i] * 2.0 ** int(ref.ze[i]) - z) <= 1e-15 * max(1e-300, abs(z)) or i == 0
    assert ref.ze[0] == EXP_ZERO


def test_reference_escape_index():
    ref = compute_reference("0.4", "0.3", 128, 1000)
    orbit = direct_orbit("0.4", "0.3", 128, 60)
    n = next(i for i, z in enumerate(orbit) if abs(complex(z)) > 256.0)
    assert ref.escaped and ref.length == n + 1


def test_tiny_values_keep_their_exponent():
    # nucleus of the period 3 island: Z_3 is ~ 2^-(prec) -> far below float64 range
    c = newton_nucleus(mpc(-1.75), 3, 4000)
    re, im = to_str(c.real, 4000), to_str(c.imag, 4000)
    ref = compute_reference(re, im, 4000, 10, allow_period=False)
    orbit = direct_orbit(re, im, 4000, 10)
    z3 = orbit[3]
    assert abs(complex(z3)) == 0.0                        # underflows float64
    e = int(ref.ze[3])
    assert e < -1000
    m = ref.zm[3]
    # compare log2 magnitude with the exact value
    l2 = math.log2(abs(m)) + e
    exact = math.log2(float(abs(z3) * gmpy2.mul_2exp(mpfr(1), -e))) + e
    assert abs(l2 - exact) < 1e-9


def test_periodic_shortcut_matches_direct():
    c = newton_nucleus(mpc(-0.1225611668766536, 0.7448617666197442), 3, 300)
    re, im = to_str(c.real, 300), to_str(c.imag, 300)
    ref = compute_reference(re, im, 300, 50000, chunk=256)     # (period checks from 1024 on)
    assert ref.period == 3 and ref.complete_for(10 ** 9)
    assert ref.length < 50001                  # the cycle is stored once, not tiled to the limit
    orbit = direct_orbit(re, im, 300, 3000)
    assert ref.length < 2999
    for i in (1, 2, 1000, ref.length - 1, ref.length, ref.length + 1, 2999):
        assert abs(ref.zf[ref.index(i)] - complex(orbit[i])) < 1e-12, i


def _nested_nucleus(p, q, prec):
    """Nucleus of a period p*q minibrot nested in the period-p minibrot near -2 (at the spot of
    its own period-q minibrot near -2): "a minibrot inside a minibrot"."""
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        n1 = newton_nucleus(mpc(mpfr(-2) + mpfr(2) ** (-2 * p + 2)), p, prec, max_steps=200)
        s1 = size_estimate(n1, p, prec)
        w2 = newton_nucleus(mpc(mpfr(-2) + mpfr(2) ** (-2 * q + 2)), q, prec, max_steps=200)
        return newton_nucleus(n1 + s1 * w2, p * q, prec, max_steps=200)


@pytest.mark.parametrize("p,q", [(30, 20), (30, 8), (40, 10)])
def test_periodic_shortcut_at_nested_minibrots(p, q):
    """At a minibrot inside a minibrot the orbit shadows the outer minibrot's cycle (and some
    multiples of it) to within ~2^-57..2^-77 for a while: float64 cannot tell it from the true
    cycle, and the shortcut used to tile the wrong period (150 instead of 600, 320 instead of
    400), so every pixel was computed against a wrong reference orbit."""
    prec = 300
    c = _nested_nucleus(p, q, prec)
    re, im = to_str(c.real, prec), to_str(c.imag, prec)
    ref = compute_reference(re, im, prec, 120000)
    assert ref.period == p * q
    exact = compute_reference(re, im, prec, 120000, allow_period=False)
    idx = np.array([ref.index(k) for k in range(exact.length)])
    assert np.allclose(ref.zf[idx], exact.zf, rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("off", [0.1, 0.3])
def test_periodic_shortcut_off_nested_nucleus(off):
    """Off the nucleus the orbit only approaches the cycle (weakly attracting at 0.3: multiplier
    ~0.86): the period must still be found early.  (Waiting for the orbit to reach the cycle in
    full precision made the reference orbits many times longer and the skip tables late.)"""
    prec = 300
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        n = _nested_nucleus(30, 20, prec)
        c = n + size_estimate(n, 600, prec) * mpc(off, off / 3)
    re, im = to_str(c.real, prec), to_str(c.imag, prec)
    ref = compute_reference(re, im, prec, 400000)
    assert ref.period == 600 and ref.length <= 32769
    exact = compute_reference(re, im, prec, 120000, allow_period=False)
    idx = np.array([ref.index(k) for k in range(exact.length)])
    assert np.allclose(ref.zf[idx], exact.zf, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("off", [0.0, 0.3])
def test_periodic_shortcut_mini_ship(off):
    re, im, l2 = "-1.77832935517465558049355033", "0.02227250757112231451758436", -36.03
    prec = 200
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        c = mpc(mpfr(re), mpfr(im)) + mpfr(2) ** l2 * mpc(off, -off / 3)
    re, im = to_str(c.real, prec), to_str(c.imag, prec)
    ref = compute_reference(re, im, prec, 400000, formula="ship")
    assert ref.period % 39 == 0 and ref.length <= 32769
    exact = compute_reference(re, im, prec, 60000, allow_period=False, formula="ship")
    idx = np.array([ref.index(k) for k in range(exact.length)])
    assert np.allclose(ref.zf[idx], exact.zf, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("w", [0.1, -1.0])
def test_periodic_shortcut_inside_a_minibrot_off_its_nucleus(w):
    """A reference elsewhere inside a minibrot (the explorer picks a non-escaping pixel) keeps
    the shortcut: its orbit converges onto an attracting cycle only to float64 accuracy within
    the limit, not to the full precision (w = 0.1: the cardioid of the period-39 minibrot,
    -1.0: its period-2 bulb, cycle 78).  The first version of the nested-minibrot fix demanded
    full precision and computed such orbits to the limit: lag when zooming out of a minibrot."""
    prec = 200
    with gmpy2.context(gmpy2.get_context(), precision=prec):
        n = newton_nucleus(mpc(-0.7436423016578859, 0.13182651981259472), 39, prec)
        c = n + size_estimate(n, 39, prec) * mpc(w)
    re, im = to_str(c.real, prec), to_str(c.imag, prec)
    ref = compute_reference(re, im, prec, 60000, chunk=1024)
    assert ref.period in (39, 78) and ref.length < 10000
    exact = compute_reference(re, im, prec, 60000, allow_period=False)
    idx = np.array([ref.index(k) for k in range(exact.length)])
    assert np.allclose(ref.zf[idx], exact.zf, rtol=1e-12, atol=1e-30)


def test_nucleus_and_size_estimate():
    c = newton_nucleus(mpc(-1.75), 3, 200)
    assert abs(complex(c) - (-1.7548776662466927)) < 1e-15
    s = size_estimate(c, 3, 200)
    assert abs(abs(complex(s)) - 0.019035515913) < 1e-9


def test_find_minibrot_in_seahorse_valley():
    r = find_minibrot(mpfr("-0.7436438870371587", 200), mpfr("0.1318259042053119", 200), math.log2(1e-6))
    assert r is not None and r["period"] == 39
    assert -19.5 < r["log2_size"] < -18.0


def test_misiurewicz_point():
    c = misiurewicz(mpc(-0.10109636384562, 0.95628651080914), 4, 1, 300)
    assert abs(complex(c) - complex(-0.10109636384562216, 0.9562865108091415)) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
def test_bla_skips_match_perturbation(order):
    """A skip must reproduce its exact perturbation steps for |dz| < R (second order: its
    larger radius and the quadratic terms)."""
    with gmpy2.context(gmpy2.get_context(), precision=600):
        c0 = misiurewicz(mpc(-0.10109636384562, 0.95628651080914), 4, 1, 600)
    re, im = to_str(c0.real, 600), to_str(c0.imag, 600)
    ref = compute_reference(re, im, 600, 2000)
    log2_c = -330.0                                    # |dc| ~ 2^-330
    bla = build_bla(ref.zm, ref.ze, log2_c, -24.0, lev0=2, order=order)
    Z = [complex(v) for v in ref.zf]
    rng = np.random.default_rng(0)
    checked = 0
    for k in rng.permutation(bla.size)[:60]:
        logR = bla.logR[k]
        if not np.isfinite(logR) or logR < -1e29:
            continue
        m, ln = int(bla.start[k]), int(bla.steps[k])
        # work in units of 2^-330 to stay in float range
        scale = 2.0 ** -330
        dc = complex(0.7, -0.4)                       # |dc| < 2^-330 units = 1
        r_units = 2.0 ** (logR + 330) * 0.5
        dz0 = complex(r_units * 0.6, -r_units * 0.5)
        z = dz0
        for j in range(m, m + ln):
            z = 2 * Z[j] * z + scale * z * z + dc
        Am = complex(bla.Am[k]) * 2.0 ** int(bla.Ae[k])
        Bm = complex(bla.Bm[k]) * 2.0 ** int(bla.Be[k])
        approx = Am * dz0 + Bm * dc
        if order == 2:
            C, D, E = (complex(getattr(bla, x + "m")[k]) * 2.0 ** int(getattr(bla, x + "e")[k])
                       if int(getattr(bla, x + "e")[k]) != -(1 << 20) else 0j for x in "CDE")
            approx += scale * (C * dz0 * dz0 + D * dz0 * dc + E * dc * dc)
        assert abs(approx - z) <= 1e-5 * abs(z) + 1e-300
        checked += 1
    assert checked > 5


def _feigenbaum_reference():
    """A reference deep in nested minibrots (periods 16, 32, 128, 2560, 38400) near the
    Feigenbaum point: its orbit passes close to 0 at every multiple of each period."""
    return compute_reference("-1.401156950707496902987425727671044895039719", "2.44e-20", 128, 60000)


def test_skips_are_cut_at_near_zero_points():
    """The skip tree (bla.la_structure): no block between cuts has a near-zero point inside
    (there the orbit of a pixel next to a minibrot must rebase or take an exact step), chains
    hold skips of the same start, shortest first, with radii that only shrink, and the tables
    of the worker and the reference upload use the same structure."""
    from mandelscope.bla import LA_TAU, la_structure, structure_for, skip_at
    ref = _feigenbaum_reference()
    st = la_structure(ref.zm, ref.ze)
    with np.errstate(divide="ignore"):
        lz = np.log2(np.abs(ref.zm)) + ref.ze
    lz[0] = -np.inf
    a, b = st.a[:st.n_leaves], st.b[:st.n_leaves]
    assert np.all(b - a <= 1 << st.leaf_log2)
    for x, y in zip(a, b):
        assert np.all(lz[x + 1:y] >= LA_TAU), (x, y)
    # every near-zero point starts a segment, so skips can start right after it
    dips = np.nonzero(lz[1:-1] < LA_TAU)[0] + 1
    assert set(dips.tolist()) <= set(a.tolist())
    tab = build_bla(ref.zm, ref.ze, -60.0, -24.0, structure=st)
    assert tab.size == st.nodes and tab.signature == st.signature
    starts = np.nonzero(tab.first >= 0)[0]
    for m in starts[:2000]:
        k = int(tab.first[m])
        chain = [k]
        while not tab.last[chain[-1]]:
            chain.append(chain[-1] + 1)
        assert np.all(tab.start[chain] == m)
        assert np.all(np.diff(tab.steps[chain]) > 0)
        assert np.all(np.diff(tab.logR[chain]) <= 1e-9)
        assert np.all(m + tab.steps[chain] <= ref.length - 1)
    # the longest valid skip for a tiny delta right after the start crosses many dips
    k = skip_at(tab, 1, -200.0)
    assert k >= 0 and tab.steps[k] > 10000
    assert structure_for(ref).signature == st.signature        # (cached on the orbit)


def test_worker_tables_use_the_structure_of_their_orbit():
    """The explorer's background worker builds the tables of long references; the orbit it
    sends carries the structure (its `first` index) the kernels need for that table."""
    import pickle
    import queue
    from types import SimpleNamespace
    from mandelscope.bla import DEFAULT_LEAF_LOG2, DEFAULT_ORDER, structure_for
    from mandelscope.worker import _worker_main
    req, res = queue.Queue(), queue.Queue()
    re, im = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"
    req.put({"id": 1, "kind": "reference", "re": re, "im": im, "prec": 128, "max_iter": 5000,
             "log2_cmax": -30.0, "log2_eps": -24.0})
    req.put({"id": 2, "kind": "bla", "ref_id": 1, "log2_cmax": -28.0, "log2_eps": -24.0})
    req.put(None)
    _worker_main(req, res, SimpleNamespace(value=-1))
    got = {}
    while not res.empty():
        r = res.get()
        got.setdefault(r["kind"], r)
    ref = pickle.loads(pickle.dumps(got["reference"]["ref"]))       # (as between processes)
    sent = structure_for(ref, full=False)
    assert not sent.full and sent.first.shape[0] == ref.length
    for kind in ("reference", "bla"):
        b = got[kind]["bla"]
        assert b.order == DEFAULT_ORDER and b.lev0 == DEFAULT_LEAF_LOG2, kind
        assert b.signature == sent.signature and np.array_equal(b.first, sent.first), kind


def test_partial_and_resumed_orbits_are_prefixes_of_the_full_one():
    full = compute_reference("0", "1", 128, 60000, allow_period=False)
    parts = []
    r = compute_reference("0", "1", 128, 60000, allow_period=False, chunk=4096,
                          partial=parts.append, partial_after=0.0)
    assert np.array_equal(r.zm, full.zm) and np.array_equal(r.ze, full.ze)
    assert len(parts) >= 3 and all(p.state is None for p in parts)
    lens = [p.length for p in parts]
    assert lens == sorted(lens) and lens[-1] < full.length
    for p in parts:
        assert np.array_equal(p.zm, full.zm[:p.length]) and np.array_equal(p.ze, full.ze[:p.length])
    # cancelled with keep_cancelled: resumable, and the continuation equals the uninterrupted orbit
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] >= 3
    c = compute_reference("0", "1", 128, 60000, allow_period=False, chunk=4096, cancelled=stop,
                          keep_cancelled=True)
    assert c.cancelled and c.state is not None and c.length == 3 * 4096 + 1
    rest = compute_reference("0", "1", 128, 60000, allow_period=False, chunk=4096, resume=c)
    assert np.array_equal(rest.zm, full.zm) and np.array_equal(rest.zf, full.zf) and np.array_equal(rest.ze, full.ze)


def test_worker_streams_partial_orbits_and_resumes_cancelled_ones():
    import queue
    from types import SimpleNamespace
    from mandelscope.worker import _worker_main
    req, res = queue.Queue(), queue.Queue()
    job = {"kind": "reference", "re": "0", "im": "1", "prec": 128, "max_iter": 1_500_000,
           "log2_cmax": -60.0, "log2_eps": -24.0, "stream": True}
    flag = SimpleNamespace(value=1)                     # job 1 is cancelled at its first check
    req.put(dict(job, id=1))
    req.put(dict(job, id=2))
    req.put(None)
    _worker_main(req, res, flag)
    msgs = []
    while not res.empty():
        msgs.append(res.get())
    kinds = [(m["id"], m["kind"]) for m in msgs if m["kind"] != "progress"]
    assert kinds[0] == (1, "cancelled")
    final = [m for m in msgs if m["id"] == 2 and m["kind"] == "reference"][0]["ref"]
    parts = [m for m in msgs if m["id"] == 2 and m["kind"] == "partial"]
    assert final.resumed_from == 8193                   # continued job 1's orbit
    assert final.length == 1_500_001
    assert parts and all(p["bla"] is not None and p["ref"].length < final.length for p in parts)
    p = parts[0]["ref"]
    assert np.array_equal(p.zm, final.zm[:p.length])
    direct = compute_reference("0", "1", 128, 1_500_000)
    assert np.array_equal(direct.zm, final.zm) and np.array_equal(direct.ze, final.ze)


def test_fast_bla_levels_match_floatexp():
    """The blocks between cuts are composed in complex128; tiny orbit values (minibrot nuclei)
    and references that stay small are redone in floatexp - the tables match the all-floatexp ones."""
    from mandelscope.bla import FAST_ZE
    re_, im_ = "-0.743643887037158704752191506114774", "0.131825904205311970493132056385139"
    refs = [compute_reference(re_, im_, 200, 30000), compute_reference("0.001", "0", 128, 5000),
            compute_reference("1e-300", "1e-301", 1200, 3000)]
    # a minibrot nucleus orbit has tiny values: the floatexp redo is exercised
    with gmpy2.context(gmpy2.get_context(), precision=200):
        nuc = newton_nucleus(mpc("-1.7548776662466927"), 3, 200)
    refs.append(compute_reference(to_str(nuc.real, 200), to_str(nuc.imag, 200), 200, 3000, allow_period=False))
    assert np.any(refs[-1].ze[1:] < FAST_ZE[0])
    for ref in refs:
        for order, lev0 in ((2, 6), (1, 4), (2, 3)):
            a = build_bla(ref.zm, ref.ze, -40.0, -24.0, lev0=lev0, order=order, fast=False)
            b = build_bla(ref.zm, ref.ze, -40.0, -24.0, lev0=lev0, order=order, fast=True)
            assert a.signature == b.signature and np.array_equal(a.steps, b.steps)
            for k in ("A", "B") + (("C", "D", "E") if order == 2 else ()):
                am, ae = getattr(a, k + "m"), getattr(a, k + "e")
                bm, be = getattr(b, k + "m"), getattr(b, k + "e")
                nz = np.abs(am) > 0
                assert np.array_equal(ae[nz], be[nz]) or np.all(np.abs(ae[nz] - be[nz]) <= 1), k
                va = am * np.exp2((ae - np.maximum(ae, be)).astype(float))
                vb = bm * np.exp2((be - np.maximum(ae, be)).astype(float))
                assert np.all(np.abs(va - vb) <= 1e-12 * np.abs(va) + 1e-300), k
            fin = a.logR > -1e29
            assert np.array_equal(fin, b.logR > -1e29)
            assert np.allclose(a.logR[fin], b.logR[fin], rtol=0, atol=1e-9)


def test_view_zoom_keeps_focus_fixed():
    v = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                          math.log2(1e-20), 0.4, 1000)
    W, H = 1000, 600
    before = v.screen_to_c(731.0, 211.0, W, H)
    v2 = v.zoom_at(731.0, 211.0, 3.7, W, H)
    after = v2.screen_to_c(731.0, 211.0, W, H)
    tol = 2.0 ** (v2.pixel_log2(H) - 30)
    assert abs(float(before[0] - after[0])) < tol and abs(float(before[1] - after[1])) < tol
    # offset by pixels and back
    v3 = v2.offset(13.0, -7.0, H).offset(-13.0, 7.0, H)
    assert abs(float(v3.re - v2.re)) < tol and abs(float(v3.im - v2.im)) < tol
    dx, dy = v2.delta_in_pixels(v2.offset(5.0, 3.0, H), H)
    assert abs(dx - 5.0) < 1e-6 and abs(dy - 3.0) < 1e-6


def test_location_roundtrip_keeps_precision():
    v = View.from_strings("0.1" + "3" * 300, "-0.2" + "7" * 300, -900.0, 0.1, 5000)
    d = v.to_dict()
    w = View.from_dict(d)
    diff = abs(float(w.re - v.re))
    assert diff < 2.0 ** (-900 - 30)


def test_sequence_interpolation_exact_endpoints_and_fixed_point():
    with gmpy2.context(gmpy2.get_context(), precision=1200):
        c = misiurewicz(mpc(-0.10109636384562, 0.95628651080914), 4, 1, 1200)
    v0 = View.home()
    v1 = View(c.real, c.imag, -1000.0, 1.0, 20000)
    s = Sequence([Keyframe(v0, 0.0), Keyframe(v1, 10.0)])
    a, _ = s.view_at(0.0)
    b, _ = s.view_at(10.0)
    assert abs(float(a.re - v0.re)) < 1e-30 and a.log2r == v0.log2r
    assert abs(float(b.re - v1.re)) < 2.0 ** -1040 and abs(float(b.im - v1.im)) < 2.0 ** -1040
    # the fixed point of the similarity stays at the same screen position along the path
    H = 1000
    with gmpy2.context(gmpy2.get_context(), precision=1300):
        la = mpc(mpfr(v1.log2r - v0.log2r) * gmpy2.log(mpfr(2)), mpfr(v1.angle - v0.angle))
        av = gmpy2.exp(la)
        delta = mpc(mpfr(v1.re, 1300) - mpfr(v0.re, 1300), mpfr(v1.im, 1300) - mpfr(v0.im, 1300))
        fp = mpc(mpfr(v0.re, 1300), mpfr(v0.im, 1300)) + delta / (1 - av)
    pos = []
    for t in (0.0, 2.5, 5.0, 9.0, 9.9):
        v, _ = s.view_at(t)
        pos.append(v.delta_in_pixels(View(fp.real, fp.imag, v.log2r), H))
    for p in pos[1:]:
        assert abs(p[0] - pos[0][0]) < 1e-3 and abs(p[1] - pos[0][1]) < 1e-3
    # log-zoom is linear in time
    m, _ = s.view_at(5.0)
    assert abs(m.log2r - (v0.log2r + v1.log2r) / 2) < 1e-9


def test_sequence_pure_pan_and_io(tmp_path):
    v0 = View.from_strings("-0.5", "0.1", -3.0)
    v1 = View.from_strings("-0.4", "0.1", -3.0)
    s = Sequence([Keyframe(v0, 0.0, "smooth"), Keyframe(v1, 1.0)])
    m, _ = s.view_at(0.5)
    assert abs(float(m.re) + 0.45) < 1e-12
    p = tmp_path / "s.json"
    s.save(p)
    s2 = Sequence.load(p)
    assert len(s2.keyframes) == 2 and s2.keyframes[0].ease == "smooth"
    s.auto_time(1.0)
    assert s.keyframes[1].time > 0


def test_path_recorder():
    r = PathRecorder(min_interval=0.0)
    v = View.home()
    for i in range(20):
        r.add(i * 0.05, v.zoom_at(500, 300, 0.1 * i, 1000, 600))
    seq = r.to_sequence(30.0, smooth=2)
    assert len(seq.keyframes) == 20
    assert seq.duration == pytest.approx(0.95)
    a, _ = seq.view_at(0.5)
    assert a.log2r < v.log2r


# ------------------------------------------------------------------ the Burning Ship
def test_ship_reference_matches_direct_iteration():
    """Burning Ship orbits (z^2 or conj(z^2), plus conj c) = plain MPFR iteration, including the
    exact X*Y where the imaginary part is far below the float64 range, and the period shortcut."""
    from mandelscope import groundtruth as gt
    for re_, im_ in (("-1.762", "0.028"), ("-0.5", "0.6"), ("-1.75", "0"), ("-1.99", "1e-400")):
        r = compute_reference(re_, im_, 1500, 3000, formula="ship")
        assert r.formula == "ship" and r.xym is not None and len(r.xym) == r.length
        with gmpy2.context(gmpy2.get_context(), precision=1500):
            x = y = mpfr(0)
            cr, ci = mpfr(re_), mpfr(im_)
            for n in range(1, min(r.length, 400)):
                x, y = x * x - y * y + cr, 2 * abs(x * y) - ci
                assert abs(complex(float(x), float(y)) - r.zf[n]) <= 1e-12 * (1 + abs(r.zf[n]))
                xy = x * y
                if xy == 0:
                    assert r.xym[n] == 0
                else:
                    assert r.xym[n] * 2.0 ** (int(r.xye[n]) - int(gmpy2.frexp(xy)[0])) == \
                        pytest.approx(float(gmpy2.frexp(xy)[1]), rel=1e-12)
        n_true = gt.iterate_ship(mpc(mpfr(re_, 1500), mpfr(im_, 1500)), 3000, 1500)[0]
        if r.escaped and re_ != "-1.99":       # (on the needle the orbit loses ~1 bit per step)
            assert r.length - 1 == n_true
    # the needle: Y underflows float64 (1e-400) but X*Y keeps its exponent
    r = compute_reference("-1.99", "1e-400", 1500, 50, formula="ship")
    assert r.zf[1].imag == 0.0 and r.xym[1] != 0 and r.xye[1] < -1300


def test_find_mini_ship():
    """Ball period + Newton with the real Jacobian find the mini Burning Ships of the places."""
    r = find_minibrot("-1.7783293552", "0.022272507", -30, formula="ship")
    assert r is not None and r["period"] == 39
    assert -38 < r["log2_size"] < -34
    from mandelscope import groundtruth as gt
    p = r["prec"]
    with gmpy2.context(gmpy2.get_context(), precision=p):
        c = mpc(r["re"], r["im"])
        x = y = mpfr(0)
        for _ in range(39):
            x, y = x * x - y * y + c.real, 2 * abs(x * y) - c.imag
        assert abs(mpc(x, y)) < mpfr(2) ** -200          # the nucleus: z_p = 0
    assert gt.iterate_ship(c, 5000, p)[0] == 0xFFFFFFFF   # (inside)
