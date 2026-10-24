"""Bivariate Linear Approximation (BLA) tables.

For a reference orbit Z_m the perturbed iteration
    z_{m+1} = 2 Z_m z_m + z_m^2 + c
is approximated over l steps by
    z_{m+l} = A z_m + B c        valid while |z_m| < R.

Single step  : A = 2 Z_m, B = 1, R = eps*|Z_m| - |c|/|2 Z_m|
Merge x then y (Claude Heiland-Allen / Zhuoran):
    A = Ay Ax,  B = Ay Bx + By,
    R = max(0, min(Rx, (Ry - |Bx| |c|) / |Ax|))

Second order (v3.9, default): the approximation keeps the quadratic terms,
    z_{m+l} = A z + B c + C z^2 + D z c + E c^2,
which a single step has exactly (A = 2 Z_m, B = C = 1, D = E = 0).  Composing x then y drops
only third-order terms:
    C = Ay Cx + Cy Ax^2,   D = Ay Dx + 2 Cy Ax Bx + Dy Ax,   E = Ay Ex + Cy Bx^2 + Dy Bx + Ey,
so a step stays valid for deltas up to ~sqrt(eps) |Z| instead of ~eps |Z|.  Radii merge as
above, with the quadratic part of x's output included when checking y's radius.  Around a
deep minibrot (period 153231) this cut the GPU iteration steps per pixel 4x (9,300 -> 2,200)
with a 10x smaller error (CPU prototype against plain perturbation).

Level j holds BLAs skipping 2**j iterations starting at m = 1 + i*2**j.
Coefficients are stored as floatexp (complex mantissa + int exponent) because
|A| grows exponentially with the skip length; radii are stored as log2(R).
Everything is vectorised with numpy (O(N) work, O(N) memory).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .reference import EXP_ZERO

NEG_INF = -1e30


def _ldexp_c(m: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.ldexp(m.real, e) + 1j * np.ldexp(m.imag, e)


def fe_norm(m: np.ndarray, e: np.ndarray):
    mx = np.maximum(np.abs(m.real), np.abs(m.imag))
    _, de = np.frexp(mx)
    m = _ldexp_c(m, -de)
    e = e.astype(np.int64) + de
    e = np.where(mx == 0.0, EXP_ZERO, e)
    return m, e


def fe_mul(am, ae, bm, be):
    return fe_norm(am * bm, ae.astype(np.int64) + be)


def fe_add(am, ae, bm, be):
    e = np.maximum(ae, be).astype(np.int64)
    sa = np.clip(ae - e, -4000, 0)
    sb = np.clip(be - e, -4000, 0)
    return fe_norm(_ldexp_c(am, sa) + _ldexp_c(bm, sb), e)


def fe_log2abs(m, e):
    with np.errstate(divide="ignore"):
        r = e + np.log2(np.abs(m))
    return np.where(np.abs(m) == 0.0, -np.inf, r)


def _log2_add(a, b):
    """log2(2**a + 2**b) (-inf entries allowed)."""
    hi = np.maximum(a, b)
    lo = np.minimum(a, b)
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.where(np.isfinite(lo), np.clip(lo - hi, -200.0, 0.0), -200.0)
        r = hi + np.log2(1.0 + np.exp2(d))
    return np.where(np.isfinite(hi), r, hi)


def _log2_sub(a, b):
    """log2(2**a - 2**b) (or -inf when a <= b)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d = np.clip(b - a, -200.0, 0.0)
        r = a + np.log2(-np.expm1(d * math.log(2.0)))
    return np.where(a > b, r, -np.inf)


@dataclass
class BLATable:
    Am: np.ndarray      # complex128, concatenated stored levels
    Ae: np.ndarray      # int32
    Bm: np.ndarray
    Be: np.ndarray
    logR: np.ndarray    # float64 (NEG_INF for invalid)
    lev0: int           # lowest stored level
    levels: int         # number of levels (highest index + 1)
    offsets: list       # per level (index = level), offset into arrays (only for stored levels)
    counts: list        # per level count
    log2_cmax: float
    log2_eps: float
    ref_id: int = 0
    order: int = 1
    Cm: np.ndarray | None = None    # second order: quadratic coefficients (floatexp)
    Ce: np.ndarray | None = None
    Dm: np.ndarray | None = None
    De: np.ndarray | None = None
    Em: np.ndarray | None = None
    Ee: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.Am.shape[0])


# lowest stored level per order (shortest skip = 2**lev0 iterations).  A skip costs a table
# lookup and breaks the fast run of plain iterations, so short skips do not pay: on an RTX 5060 Ti
# (bench --bla, period-6006 / 153231 minibrots) second order took 1585 / 899 ms at 2**2,
# 957 / 781 ms at 2**4 and 801 / 686 ms at 2**6; first order 1351 / 1772 ms at 2**2 and
# 865 / 931 ms at 2**4.  The software renderer agrees.
DEFAULT_LEV0 = {1: 4, 2: 7}         # RTX 5060 Ti bench: 2^7 740 ms vs 2^6 819 ms (p6006), equal elsewhere
DEFAULT_ORDER = 2
# ... but only for high iteration limits.  The RTX bench of 1.0.0 (1920x1080, ms per picture,
# 2^6 / 2^7): spiral 10^100 (5,000 iterations) 5.0 / 7.0, 10^1000 (20,000) 5.6 / 6.9,
# seahorse 10^12 (20,000) 36.9 / 37.1, minibrots p6006 (120,120) 761 / 688 and p153231
# (3 million) 761 / 749.  Below this limit the second-order table starts at 2^6.
LEV0_DEEP_ITER = 65536


def default_lev0(order: int, max_iter: int = 0) -> int:
    """Shortest stored skip (2**lev0) for a reference computed for `max_iter` iterations
    (0: unknown, the default of the order)."""
    if order == 2 and 0 < max_iter < LEV0_DEEP_ITER:
        return 6
    return DEFAULT_LEV0.get(order, 2)


FAST_LEVELS = 6            # levels 1..6 (skips of up to 64 iterations) can be merged in complex128:
FAST_ZE = (-11, 9)         # with 2**-12 <= |Z| < 2**9 every product stays far inside float64
                           # (|A| <= 2**640, |C|, |D|, |E| <= 2**1000; terms that underflow are
                           # below the rounding of the sums they belong to)


def _merge_fe(cur, q: bool, log2_cmax: float):
    """One merge level in floatexp (x = even, y = odd neighbours)."""
    cnt = cur[0].shape[0]
    h = cnt // 2
    x = [arr[0:2 * h:2] for arr in cur]
    y = [arr[1:2 * h:2] for arr in cur]
    xA, xAe, xB, xBe, xR = x[:5]
    yA, yAe, yB, yBe, yR = y[:5]
    nA, nAe = fe_mul(yA, yAe, xA, xAe)
    pB, pBe = fe_mul(yA, yAe, xB, xBe)
    nB, nBe = fe_add(pB, pBe, yB, yBe)
    lxA, lxB = fe_log2abs(xA, xAe), fe_log2abs(xB, xBe)
    if q:
        xC, xCe, xD, xDe, xE, xEe = x[5:]
        yC, yCe, yD, yDe, yE, yEe = y[5:]
        # C = Ay Cx + Cy Ax^2
        t1 = fe_mul(yA, yAe, xC, xCe)
        ax2 = fe_mul(xA, xAe, xA, xAe)
        t2 = fe_mul(yC, yCe, *ax2)
        nC, nCe = fe_add(*t1, *t2)
        # D = Ay Dx + 2 Cy Ax Bx + Dy Ax
        t1 = fe_mul(yA, yAe, xD, xDe)
        axbx = fe_mul(xA, xAe, xB, xBe)
        t2 = fe_mul(yC, yCe + 1, *axbx)
        t3 = fe_mul(yD, yDe, xA, xAe)
        nD, nDe = fe_add(*fe_add(*t1, *t2), *t3)
        # E = Ay Ex + Cy Bx^2 + Dy Bx + Ey
        t1 = fe_mul(yA, yAe, xE, xEe)
        bx2 = fe_mul(xB, xBe, xB, xBe)
        t2 = fe_mul(yC, yCe, *bx2)
        t3 = fe_mul(yD, yDe, xB, xBe)
        nE, nEe = fe_add(*fe_add(*fe_add(*t1, *t2), *t3), yE, yEe)
        val = _radius2(yR, lxA, lxB, fe_log2abs(xC, xCe), fe_log2abs(xD, xDe), fe_log2abs(xE, xEe), log2_cmax)
    else:
        val = _radius1(yR, lxA, lxB, log2_cmax)
    nR = np.minimum(xR, val)
    return (nA, nAe, nB, nBe, nR) + ((nC, nCe, nD, nDe, nE, nEe) if q else ())


def _radius1(yR, lxA, lxB, log2_cmax):
    num = _log2_sub(yR, lxB + log2_cmax)
    with np.errstate(invalid="ignore"):
        val = num - lxA
    return np.where(np.isfinite(val) | (val == np.inf), val, -np.inf)


def _radius2(yR, lxA, lxB, lxC, lxD, lxE, log2_cmax):
    # largest r with |x(z, c)| <= Ry for |z| <= r, |c| <= cmax:
    #   |Cx| r^2 + (|Ax| + |Dx| cmax) r + |Bx| cmax + |Ex| cmax^2 <= Ry
    # (sufficient: each of the two r-terms <= half of what is left)
    c0 = _log2_add(lxB + log2_cmax, lxE + 2.0 * log2_cmax)
    left = _log2_sub(yR, c0) - 1.0
    b1 = _log2_add(lxA, lxD + log2_cmax)
    with np.errstate(invalid="ignore"):
        r1 = left - b1
        r2 = 0.5 * (left - lxC)
    val = np.minimum(r1, r2)
    return np.where(np.isfinite(val) | (val == np.inf), val, -np.inf)


def _log2abs_c(v):
    with np.errstate(divide="ignore"):
        return np.log2(np.abs(v))


def _merge_c(cur, q: bool, log2_cmax: float):
    """One merge level in plain complex128 - the same formulas, ~7x fewer numpy passes."""
    cnt = cur[0].shape[0]
    h = cnt // 2
    x = [arr[0:2 * h:2] for arr in cur]
    y = [arr[1:2 * h:2] for arr in cur]
    xA, xB, xR = x[:3]
    yA, yB, yR = y[:3]
    nA = yA * xA
    nB = yA * xB + yB
    lxA, lxB = _log2abs_c(xA), _log2abs_c(xB)
    if q:
        xC, xD, xE = x[3:]
        yC, yD, yE = y[3:]
        nC = yA * xC + yC * (xA * xA)
        nD = yA * xD + 2.0 * yC * (xA * xB) + yD * xA
        nE = yA * xE + yC * (xB * xB) + yD * xB + yE
        val = _radius2(yR, lxA, lxB, _log2abs_c(xC), _log2abs_c(xD), _log2abs_c(xE), log2_cmax)
        return (nA, nB, np.minimum(xR, val), nC, nD, nE)
    return (nA, nB, np.minimum(xR, _radius1(yR, lxA, lxB, log2_cmax)))


def _level0(zm1, ze1, log2_cmax: float, log2_eps: float, q: bool):
    """Single-step BLAs (floatexp): A = 2Z, B = 1 (C = 1, D = E = 0)."""
    n0 = zm1.shape[0]
    Am, Ae = fe_norm(2.0 * zm1, ze1)
    Bm = np.ones(n0, dtype=np.complex128) * 0.5
    Be = np.ones(n0, dtype=np.int64)          # 1 = 0.5 * 2^1
    lz = fe_log2abs(zm1, ze1)
    # first order: |z|^2 < eps |2 Z z|; second order: the dropped third-order terms are
    # ~ |z|/|Z| smaller than the kept quadratic ones -> sqrt(eps)
    a = (0.5 * log2_eps if q else log2_eps) + lz
    b = log2_cmax - 1.0 - lz                  # |c| / |2Z|
    logR = _log2_sub(a, b)
    if q:
        zero_m = np.zeros(n0, dtype=np.complex128)
        zero_e = np.full(n0, EXP_ZERO, dtype=np.int64)
        return (Am, Ae, Bm, Be, logR, Bm.copy(), Be.copy(), zero_m, zero_e, zero_m.copy(), zero_e.copy())
    return (Am, Ae, Bm, Be, logR)


def _to_fe(cur_c, q: bool):
    """complex128 level -> floatexp level."""
    A, B, R = cur_c[:3]
    out = [*fe_norm(A, np.zeros(A.shape, np.int64)), *fe_norm(B, np.zeros(B.shape, np.int64)), R]
    if q:
        for v in cur_c[3:]:
            out += [*fe_norm(v, np.zeros(v.shape, np.int64))]
    return tuple(out)


def build_bla(zm: np.ndarray, ze: np.ndarray, log2_cmax: float, log2_eps: float = -24.0,
              lev0: int | None = None, max_levels: int = 31, order: int | None = None,
              fast: bool = True, max_iter: int = 0) -> BLATable:
    """max_iter: the iteration limit the reference is for; it picks the shortest stored
    skip when `lev0` is not given (default_lev0)."""
    order = DEFAULT_ORDER if order is None else order
    lev0 = default_lev0(order, max_iter) if lev0 is None else lev0
    N = zm.shape[0]
    n0 = max(N - 2, 0)                       # level 0 entries for m = 1 .. N-2
    zm1 = zm[1:1 + n0]
    ze1 = ze[1:1 + n0].astype(np.int64)
    q = order == 2
    cur = _level0(zm1, ze1, log2_cmax, log2_eps, q)
    stored = []
    offsets, counts = [], []
    off = 0
    j = 0
    # The short levels below lev0 hold almost all the work (N + N/2 + ...) and are not stored:
    # compute them in complex128 (the values fit: |Z| <= 256, |A| <= 4**(2**j)).  Entries whose
    # Z is too small for that (the orbit passing a minibrot's nucleus) are redone in floatexp.
    nfast = min(lev0, FAST_LEVELS, max_levels - 1) if fast else 0
    if nfast > 0 and n0 >> nfast >= 1:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            zc = _ldexp_c(zm1, np.clip(ze1, -1100, 1100))
            ones = np.ones(n0, dtype=np.complex128)
            cur_c = (2.0 * zc, ones, cur[4]) + ((ones.copy(), np.zeros(n0, np.complex128),
                                                 np.zeros(n0, np.complex128)) if q else ())
        unsafe = (ze1 != EXP_ZERO) & ((ze1 < FAST_ZE[0]) | (ze1 > FAST_ZE[1]))
        lvl0_fe = cur
        for _ in range(nfast):               # levels 0 .. nfast-1 (not stored: nfast <= lev0)
            offsets.append(0)
            counts.append(cur_c[0].shape[0])
            j += 1
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                cur_c = _merge_c(cur_c, q, log2_cmax)
        cur = _to_fe(cur_c, q)
        nb = cur[0].shape[0]
        bad = np.zeros(nb, bool)
        ub = np.unique(np.nonzero(unsafe)[0] >> j)
        bad[ub[ub < nb]] = True
        for v in cur_c[:2] + cur_c[3:]:
            bad |= ~np.isfinite(v)
        bad |= np.isnan(cur_c[2])
        if np.any(bad):
            # redo the blocks with a tiny / huge Z (or anything non-finite) in floatexp
            blocks = np.nonzero(bad)[0]
            idx = (blocks[:, None] * (1 << j) + np.arange(1 << j)[None, :]).ravel()
            sub = tuple(a[idx] for a in lvl0_fe)
            for _ in range(j):
                sub = _merge_fe(sub, q, log2_cmax)
            cur = tuple(a.copy() for a in cur)
            for a, b in zip(cur, sub):
                a[blocks] = b
    while True:
        cnt = cur[0].shape[0]
        if cnt == 0:
            break
        if j >= lev0:
            stored.append(cur)
            offsets.append(off)
            counts.append(cnt)
            off += cnt
        else:
            offsets.append(0)
            counts.append(cnt)
        j += 1
        if j >= max_levels or cnt < 2:
            break
        cur = _merge_fe(cur, q, log2_cmax)
    return _finish(stored, offsets, counts, lev0, j, log2_cmax, log2_eps, order, q)


def _finish(stored, offsets, counts, lev0, levels, log2_cmax, log2_eps, order, q) -> BLATable:
    def cat(k, dtype):
        return np.concatenate([c[k] for c in stored]) if stored else np.zeros(0, dtype)
    A, Ae_, B, Be_, R = (cat(0, np.complex128), cat(1, np.int64), cat(2, np.complex128), cat(3, np.int64),
                         cat(4, np.float64))
    R = np.where(np.isfinite(R), R, NEG_INF)

    def clip32(e):
        return np.where(e == EXP_ZERO, EXP_ZERO, np.clip(e, -(1 << 30), 1 << 30)).astype(np.int32)
    t = BLATable(Am=A, Ae=clip32(Ae_), Bm=B, Be=clip32(Be_), logR=R, lev0=lev0, levels=levels,
                 offsets=offsets, counts=counts, log2_cmax=log2_cmax, log2_eps=log2_eps, order=order)
    if q:
        t.Cm, t.Ce = cat(5, np.complex128), clip32(cat(6, np.int64))
        t.Dm, t.De = cat(7, np.complex128), clip32(cat(8, np.int64))
        t.Em, t.Ee = cat(9, np.complex128), clip32(cat(10, np.int64))
    return t


def apply_reference_check(bla: BLATable, level: int, m: int):
    """Return (A, B, logR) as python complex/float values for testing."""
    i = (m - 1) >> level
    k = bla.offsets[level] + i
    A = complex(bla.Am[k]) * 2.0 ** int(bla.Ae[k])
    B = complex(bla.Bm[k]) * 2.0 ** int(bla.Be[k])
    return A, B, float(bla.logR[k])
