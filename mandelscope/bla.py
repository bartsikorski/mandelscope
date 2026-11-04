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


# ---------------------------------------------------------------------------
# structure: which skips exist (depends on the reference only)
# ---------------------------------------------------------------------------

LA_TAU = -3.0              # a near-zero point ("dip") of the reference: |Z_m| < 2^-3
LA_DELTA = 2.0             # dips are told apart by depth in steps of 2 bits
DEFAULT_LEAF_LOG2 = 5      # between dips, the shortest skips are blocks of 2^5 iterations
MIN_SKIP = 16              # shorter skips are not stored: on a GPU a skip costs about as much as
                           # 30 plain steps, so short ones only add lookups
DEFAULT_ORDER = 2
FAST_LEAF_LOG2 = 6         # blocks of up to 64 iterations can be composed in complex128:
FAST_ZE = (-11, 9)         # with 2**-12 <= |Z| < 2**9 every product stays far inside float64
                           # (|A| <= 2**640, |C|, |D|, |E| <= 2**1000; terms that underflow are
                           # below the rounding of the sums they belong to)
_TOP = 1 << 20             # depth class of the ends of the reference (boundaries at every stage)


@dataclass
class LAStructure:
    """The skips of a reference orbit Z_0 .. Z_{L-1}: a tree of segments [a, b) of iterations.

    The iteration at a is done exactly (a plain perturbation step, wherever |Z_a| is); the
    skip covers a+1 .. b-1 and lands on b, where the kernel checks escape and rebasing.
    Segments are cut at the near-zero points ("dips") of the orbit, the deepest dips first:
    nested minibrots make the orbit pass close to 0 at every multiple of each period, and a
    pixel near them must either rebase or cross such a point with an exact step.  Between
    dips, segments are halved down to blocks of 2^leaf_log2 iterations.  Sibling segments
    are paired into a binary tree.

    Stored skips are ordered by (a, length): the skips that start at the same iteration form
    a contiguous "chain", shortest first.  first[m] is the index of the shortest skip that
    starts at m (-1: none), last[k] marks the longest of its chain.  The structure depends
    on the reference only (not on the pixel spacing or the tolerance), so the kernels can
    keep `first` next to the reference values."""
    leaf_log2: int
    length: int                  # L (reference length)
    a: np.ndarray                # int64 [items]  segment start
    b: np.ndarray                # int64 [items]  segment end (exclusive)
    n_leaves: int                # items 0 .. n_leaves-1 are the blocks (no children)
    rounds: list                 # [(new, left, right)] item ids: new = left followed by right
    stored: np.ndarray           # int64 [nodes] item ids of the stored skips, in chain order
    last: np.ndarray             # bool  [nodes]
    first: np.ndarray            # int32 [L]

    min_skip: int = MIN_SKIP
    n_nodes: int = -1            # (kept when the arrays above are dropped, see __getstate__)
    full: bool = True            # False: only `first` (and the counts) - enough for the kernels

    @property
    def nodes(self) -> int:
        return int(self.stored.shape[0]) if self.full else int(self.n_nodes)

    @property
    def signature(self) -> tuple:
        """Equal for equal structures of the same orbit (tables and uploads must agree)."""
        return (self.length, self.leaf_log2, self.min_skip, self.nodes)

    def __getstate__(self):
        # sent between processes without the tree (the receiver uploads `first` with the
        # reference; a table built there makes the structure again, deterministically)
        d = dict(self.__dict__)
        d["n_nodes"] = self.nodes
        z = np.zeros(0, np.int64)
        d.update(a=z, b=z, rounds=[], stored=z, last=np.zeros(0, bool), full=False)
        return d


def la_structure(zm: np.ndarray, ze: np.ndarray, leaf_log2: int | None = None,
                 tau: float = LA_TAU, delta: float = LA_DELTA, min_skip: int = MIN_SKIP) -> LAStructure:
    """The skip tree of a reference (see LAStructure).  O(L) numpy work."""
    leaf_log2 = DEFAULT_LEAF_LOG2 if leaf_log2 is None else int(leaf_log2)
    L = int(zm.shape[0])
    E = L - 1                                  # skips land at most on the last stored value
    z64 = np.zeros(0, np.int64)
    if E < 2:
        return LAStructure(leaf_log2, L, z64, z64.copy(), 0, [], z64.copy(), np.zeros(0, bool),
                           np.full(L, -1, np.int32), min_skip=int(min_skip))
    lz = fe_log2abs(zm[:E + 1], ze[:E + 1].astype(np.int64))
    with np.errstate(invalid="ignore", over="ignore"):
        cls = np.floor((tau - lz) / delta) + 1.0
    q = np.where(lz < tau, np.minimum(cls, 4096.0), 0.0).astype(np.int64)
    q[0] = _TOP
    q[E] = _TOP
    leaf = 1 << leaf_log2
    seg_a_list, seg_b_list, rounds = [], [], []
    count = [0]

    def add(a, b):
        ids = np.arange(count[0], count[0] + a.shape[0], dtype=np.int64)
        count[0] += a.shape[0]
        seg_a_list.append(a)
        seg_b_list.append(b)
        return ids

    def group(cid, ca, cb, pid, npar):
        """Pair consecutive children of the same parent (pid sorted) until one item is left
        per parent; returns that item per parent."""
        while cid.shape[0]:
            n = cid.shape[0]
            newpar = np.ones(n, bool)
            newpar[1:] = pid[1:] != pid[:-1]
            if newpar.all():
                break
            starts = np.nonzero(newpar)[0]
            rank = np.arange(n) - np.repeat(starts, np.diff(np.append(starts, n)))
            even = (rank & 1) == 0
            nxt = np.zeros(n, bool)
            nxt[:-1] = ~newpar[1:]
            pair = even & nxt
            li = np.nonzero(pair)[0]
            ri = li + 1
            new = add(ca[li], cb[ri])
            rounds.append((new, cid[li], cid[ri]))
            keep = np.nonzero(even)[0]
            pk = pair[keep]
            cid2 = cid[keep].copy()
            cb2 = cb[keep].copy()
            cid2[pk] = new
            cb2[pk] = cb[ri]
            cid, ca, cb, pid = cid2, ca[keep], cb2, pid[keep]
        top = np.full(npar, -1, np.int64)
        top[pid] = cid
        return top

    # finest segments: between consecutive dips; blocks of `leaf` iterations inside them
    bnd = np.nonzero(q >= 1)[0]
    sa, sb = bnd[:-1], bnd[1:]
    nblk = (sb - sa + leaf - 1) // leaf
    seg = np.repeat(np.arange(sa.shape[0]), nblk)
    off = np.arange(seg.shape[0]) - np.repeat(np.cumsum(nblk) - nblk, nblk)
    ba = sa[seg] + off * leaf
    bb = np.minimum(ba + leaf, sb[seg])
    leaf_ids = add(ba, bb)
    n_leaves = int(ba.shape[0])
    top = group(leaf_ids, ba, bb, seg, sa.shape[0])
    # coarser and coarser: the segments between dips of class >= v are the children of the
    # segments between dips of the next class
    inner = q[1:E]
    classes = np.unique(inner[inner >= 1])
    cur_a, cur_b = sa, sb
    for v in list(classes[1:]) + [_TOP]:
        pb = np.nonzero(q >= v)[0]
        pid = np.searchsorted(pb, cur_a, side="right") - 1
        top = group(top, cur_a, cur_b, pid, pb.shape[0] - 1)
        cur_a, cur_b = pb[:-1], pb[1:]
    a = np.concatenate(seg_a_list)
    b = np.concatenate(seg_b_list)
    idx = np.nonzero(b - a - 1 >= max(int(min_skip), 1))[0]
    stored = idx[np.lexsort((b[idx], a[idx]))]
    s_a = a[stored]
    head = np.ones(stored.shape[0], bool)
    head[1:] = s_a[1:] != s_a[:-1]
    last = np.ones(stored.shape[0], bool)
    last[:-1] = head[1:]
    first = np.full(L, -1, np.int32)
    first[s_a[head] + 1] = np.nonzero(head)[0].astype(np.int32)
    return LAStructure(leaf_log2, L, a, b, n_leaves, rounds, stored, last, first, min_skip=int(min_skip))


def structure_for(ref, leaf_log2: int | None = None, full: bool = True) -> LAStructure:
    """The skip structure of a ReferenceOrbit (cached on it: the kernels keep its `first`
    index next to the reference values, and every table for the orbit must use it).
    leaf_log2 None: the cached one's (else the default).  full=False: a structure that came
    from another process (only `first`) will do."""
    st = getattr(ref, "la", None)
    if leaf_log2 is None:
        leaf_log2 = st.leaf_log2 if st is not None else DEFAULT_LEAF_LOG2
    leaf_log2 = int(leaf_log2)
    if (st is None or st.length != ref.length or st.leaf_log2 != leaf_log2
            or (full and not st.full)):
        st = la_structure(ref.zm, ref.ze, leaf_log2, min_skip=st.min_skip if st is not None else MIN_SKIP)
        try:
            ref.la = st
        except AttributeError:
            pass
    return st


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------

@dataclass
class BLATable:
    """Stored skips in chain order (LAStructure): skip k starts at the iteration after its
    segment start and covers steps[k] iterations: z' = A z + B c (+ C z^2 + D z c + E c^2),
    valid while |z| < 2^logR[k]."""
    Am: np.ndarray      # complex128 mantissas, int32 exponents (floatexp)
    Ae: np.ndarray
    Bm: np.ndarray
    Be: np.ndarray
    logR: np.ndarray    # float64 (NEG_INF: never valid)
    steps: np.ndarray   # int64 skip lengths
    start: np.ndarray   # int64 iteration each skip starts at (the one after its segment start)
    last: np.ndarray    # bool: the longest skip of its chain
    first: np.ndarray   # int32 [L]: shortest skip starting at iteration m (-1: none)
    lev0: int           # log2 of the block length between dips (LAStructure.leaf_log2)
    length: int         # reference length the table was made for
    signature: tuple    # LAStructure.signature of the structure it was made with
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

    @property
    def max_logR(self) -> float:
        """Largest radius: no lookup can succeed for deltas above it."""
        return float(np.max(self.logR)) if self.size else NEG_INF


def _compose_fe(x, y, q: bool, log2_cmax: float):
    """x followed by y, in floatexp (tuples as produced by _single_fe)."""
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


def _compose_c(x, y, q: bool, log2_cmax: float):
    """x followed by y in plain complex128 - the same formulas, ~7x fewer numpy passes."""
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


def _single_radius(zm, ze, log2_cmax: float, log2_eps: float, q: bool):
    # first order: |z|^2 < eps |2 Z z|; second order: the dropped third-order terms are
    # ~ |z|/|Z| smaller than the kept quadratic ones -> sqrt(eps).  Minus |c| / |2Z|.
    lz = fe_log2abs(zm, ze)
    return _log2_sub((0.5 * log2_eps if q else log2_eps) + lz, log2_cmax - 1.0 - lz)


def _single_fe(zm, ze, idx, log2_cmax, log2_eps, q):
    """One iteration at the indices idx (floatexp): A = 2Z, B = 1 (C = 1, D = E = 0)."""
    n = idx.shape[0]
    m = zm[idx]
    e = ze[idx].astype(np.int64)
    Am, Ae = fe_norm(2.0 * m, e)
    Bm = np.full(n, 0.5 + 0j)
    Be = np.ones(n, np.int64)                  # 1 = 0.5 * 2^1
    R = _single_radius(m, e, log2_cmax, log2_eps, q)
    if q:
        zero_m = np.zeros(n, np.complex128)
        zero_e = np.full(n, EXP_ZERO, np.int64)
        return (Am, Ae, Bm, Be, R, Bm.copy(), Be.copy(), zero_m, zero_e, zero_m.copy(), zero_e.copy())
    return (Am, Ae, Bm, Be, R)


def _identity_fe(n, q):
    one_m = np.full(n, 0.5 + 0j)
    one_e = np.ones(n, np.int64)
    zero_m = np.zeros(n, np.complex128)
    zero_e = np.full(n, EXP_ZERO, np.int64)
    out = (one_m, one_e, zero_m, zero_e.copy(), np.full(n, np.inf))
    if q:
        out += (zero_m.copy(), zero_e.copy(), zero_m.copy(), zero_e.copy(), zero_m.copy(), zero_e.copy())
    return out


def _c_to_fe(cur_c, q: bool):
    """complex128 composites -> floatexp."""
    A, B, R = cur_c[:3]
    z = np.zeros(A.shape, np.int64)
    out = [*fe_norm(A, z), *fe_norm(B, z), R]
    if q:
        for v in cur_c[3:]:
            out += [*fe_norm(v, z)]
    return tuple(out)


def _take(t, idx):
    return tuple(v[idx] for v in t)


def _put(t, idx, v):
    for dst, src in zip(t, v):
        dst[idx] = src


def _leaf_rests(zm, ze, la, rl, log2_cmax, log2_eps, q, fast):
    """Composites of the iterations la+1 .. la+rl (rl >= 1 each), one after another."""
    n = la.shape[0]
    ze64 = ze.astype(np.int64)
    ncomp = 11 if q else 5
    out = [np.zeros(n, np.complex128) if k in (0, 2, 5, 7, 9) else
           (np.zeros(n) if k == 4 else np.zeros(n, np.int64)) for k in range(ncomp)]
    todo = np.ones(n, bool)
    if fast and n:
        unsafe = (ze64 < FAST_ZE[0]) | (ze64 > FAST_ZE[1])
        cs = np.concatenate([[0], np.cumsum(unsafe)])
        ok = (cs[la + rl + 1] - cs[la + 1]) == 0
        ok &= rl <= (1 << FAST_LEAF_LOG2)
        sel = np.nonzero(ok)[0]
        if sel.shape[0]:
            sel = sel[np.argsort(-rl[sel], kind="stable")]       # longest first: active = prefix
            a0, r0 = la[sel], rl[sel]
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                zc = _ldexp_c(zm, np.clip(ze64, -1100, 1100))
                R = _single_radius(zm, ze64, log2_cmax, log2_eps, q)

                def single_c(pos):
                    k = pos.shape[0]
                    one = np.ones(k, np.complex128)
                    s = (2.0 * zc[pos], one, R[pos])
                    if q:
                        s += (one.copy(), np.zeros(k, np.complex128), np.zeros(k, np.complex128))
                    return s
                acc = single_c(a0 + 1)
                for t in range(1, int(r0[0])):
                    k = int(np.searchsorted(-r0, -t, side="left"))       # rl > t: a prefix
                    head = tuple(v[:k] for v in acc)
                    nxt = _compose_c(head, single_c(a0[:k] + 1 + t), q, log2_cmax)
                    for v, w in zip(acc, nxt):
                        v[:k] = w
                fe = _c_to_fe(acc, q)
            good = np.ones(sel.shape[0], bool)
            for v in acc[:2] + acc[3:]:
                good &= np.isfinite(v)
            good &= ~np.isnan(acc[2])
            _put(out, sel[good], _take(fe, np.nonzero(good)[0]))
            todo[sel[good]] = False
    sel = np.nonzero(todo)[0]
    if sel.shape[0]:
        sel = sel[np.argsort(-rl[sel], kind="stable")]
        a0, r0 = la[sel], rl[sel]
        acc = _single_fe(zm, ze, a0 + 1, log2_cmax, log2_eps, q)
        acc = tuple(np.array(v) for v in acc)
        for t in range(1, int(r0[0])):
            k = int(np.searchsorted(-r0, -t, side="left"))
            head = tuple(v[:k] for v in acc)
            nxt = _compose_fe(head, _single_fe(zm, ze, a0[:k] + 1 + t, log2_cmax, log2_eps, q), q, log2_cmax)
            for v, w in zip(acc, nxt):
                v[:k] = w
        _put(out, sel, acc)
    return tuple(out)


def build_bla(zm: np.ndarray, ze: np.ndarray, log2_cmax: float, log2_eps: float = -24.0,
              lev0: int | None = None, order: int | None = None, fast: bool = True,
              max_iter: int = 0, structure: LAStructure | None = None) -> BLATable:
    """Skip table of a reference for pixel offsets |c| <= 2^log2_cmax and tolerance 2^log2_eps.

    structure: the reference's LAStructure (made here when not given; lev0 is then its
    block length, log2).  max_iter is accepted for compatibility (unused)."""
    order = DEFAULT_ORDER if order is None else int(order)
    q = order == 2
    st = structure if structure is not None else la_structure(zm, ze, lev0)
    L = int(zm.shape[0])
    if st.length != L or not st.full:
        st = la_structure(zm, ze, st.leaf_log2, min_skip=st.min_skip)
    log2_cmax = float(log2_cmax)
    log2_eps = float(log2_eps)
    nitems = st.a.shape[0]
    empty_c = np.zeros(0, np.complex128)
    empty_i = np.zeros(0, np.int32)
    if st.nodes == 0:
        t = BLATable(Am=empty_c, Ae=empty_i, Bm=empty_c.copy(), Be=empty_i.copy(), logR=np.zeros(0),
                     steps=np.zeros(0, np.int64), start=np.zeros(0, np.int64), last=np.zeros(0, bool),
                     first=st.first, lev0=st.leaf_log2,
                     length=L, signature=st.signature, log2_cmax=log2_cmax, log2_eps=log2_eps, order=order)
        if q:
            t.Cm, t.Ce, t.Dm, t.De, t.Em, t.Ee = empty_c, empty_i, empty_c, empty_i, empty_c, empty_i
        return t
    # per item: `rest` (iterations a+1 .. b-1) and `full` (a .. b-1)
    rest = _identity_fe(nitems, q)
    rest = tuple(np.array(v) for v in rest)
    has_rest = np.zeros(nitems, bool)
    full = tuple(np.array(v) for v in _identity_fe(nitems, q))
    nl = st.n_leaves
    la, lb = st.a[:nl], st.b[:nl]
    rl = lb - la - 1
    sel = np.nonzero(rl >= 1)[0]
    if sel.shape[0]:
        _put(rest, sel, _leaf_rests(zm, ze, la[sel], rl[sel], log2_cmax, log2_eps, q, fast))
        has_rest[sel] = True
    ids = np.arange(nl)
    s0 = _single_fe(zm, ze, la, log2_cmax, log2_eps, q)
    f = _compose_fe(s0, _take(rest, ids), q, log2_cmax)
    nr = ~has_rest[:nl]
    if np.any(nr):
        f = tuple(np.where(nr, a, b) for a, b in zip(s0, f))
    _put(full, ids, f)
    for new, left, right in st.rounds:
        x = _take(rest, left)
        y = _take(full, right)
        r = _compose_fe(x, y, q, log2_cmax)
        nx = ~has_rest[left]
        if np.any(nx):
            r = tuple(np.where(nx, b, a) for a, b in zip(r, y))
        _put(rest, new, r)
        has_rest[new] = True
        _put(full, new, _compose_fe(_take(full, left), y, q, log2_cmax))
    k = st.stored
    R = rest[4][k]
    R = np.where(np.isfinite(R) | (R == np.inf), R, NEG_INF)
    R = np.minimum(R, 1e30)

    def clip32(e):
        return np.where(e == EXP_ZERO, EXP_ZERO, np.clip(e, -(1 << 30), 1 << 30)).astype(np.int32)
    t = BLATable(Am=rest[0][k], Ae=clip32(rest[1][k]), Bm=rest[2][k], Be=clip32(rest[3][k]), logR=R,
                 steps=(st.b[k] - st.a[k] - 1).astype(np.int64), start=(st.a[k] + 1).astype(np.int64),
                 last=st.last, first=st.first,
                 lev0=st.leaf_log2, length=L, signature=st.signature, log2_cmax=log2_cmax,
                 log2_eps=log2_eps, order=order)
    if q:
        t.Cm, t.Ce = rest[5][k], clip32(rest[6][k])
        t.Dm, t.De = rest[7][k], clip32(rest[8][k])
        t.Em, t.Ee = rest[9][k], clip32(rest[10][k])
    return t


def skip_at(bla: BLATable, m: int, logw: float, budget: int | None = None):
    """The longest stored skip starting at iteration m that is valid for log2|z| = logw
    (and at most `budget` iterations long): its index, or -1 (the kernels' lookup)."""
    k = int(bla.first[m]) if 0 <= m < bla.first.shape[0] else -1
    best = -1
    while k >= 0:
        if not (logw < bla.logR[k]) or (budget is not None and bla.steps[k] > budget):
            break
        best = k
        if bla.last[k]:
            break
        k += 1
    return best
