"""GPU engine: compute shaders, sample fields, progressive scheduling, colouring.

Terminology
  field   - a grid of samples (one per output pixel, or ss*ss per pixel when
            supersampling) with persistent per-sample iteration state, so work
            can be resumed across many short dispatches.
  order   - coarse-to-fine progressive ordering of the samples (1/64, 3/64,
            12/64, 48/64) so a usable preview appears almost immediately.
  pending - compacted list of samples that ran out of per-dispatch budget.

Performance rules (v2): large buffers never travel between CPU and GPU in the
interactive loop.  Clearing, shifting (panning) and counting all run in small
compute shaders; the CPU only reads a 48-byte counter block, once per batch of
dispatches.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import moderngl
from gmpy2 import mpfr

from .mp import View, mp_ratio_to_float
from .palettes import make_palette
from .reference import ReferenceOrbit
from .bla import BLATable

SHADER_DIR = Path(__file__).parent / "shaders"
RES_MAXITER = 0xFFFFFFFF
RES_INTERIOR = 0xFFFFFFFE
# Delta precision ("double" in the code): False / 0 float32, True / 1 float64, 2 float-float
# (perturb_ff.comp: two float32 per number, about 48 bits, no float64 instructions)
PREC_FF = 2
PRECISION_NAMES = {0: "float32", 1: "float64", 2: "float-float"}
STATE_STRIDE = {False: 56, True: 64, PREC_FF: 64}       # bytes per sample (std430 Pix)
STATE_STRIDE_SHIP = {False: 64, True: 80, PREC_FF: 80}  # (the Burning Ship's Pix has one more vec2)
STATUS_WORD = {False: 11, True: 13, PREC_FF: 13}         # index of 'status' in uint words
RESULT_WORDS = 8                            # uint words per sample (std430 Result)
SMALL_EXP = {False: -60, True: -480, PREC_FF: -60}       # reference values below 2^this use floatexp maths


def ff_split(x) -> tuple[np.ndarray, np.ndarray]:
    """float64 values as float-float pairs (hi, lo): hi = x rounded to float32, lo = the rest."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        hi = x.astype(np.float32)
        lo = (x - hi.astype(np.float64)).astype(np.float32)
    lo[~np.isfinite(hi)] = 0.0
    return hi, lo


def ff_complex(z) -> np.ndarray:
    """complex128 values as float32 [N, 4] = (re.hi, im.hi, re.lo, im.lo)."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    rh, rl = ff_split(z.real)
    ih, il = ff_split(z.imag)
    return np.stack([rh, ih, rl, il], axis=1)
MAX_GROUPS = 65535
N_COUNTERS = 12


def _src(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


def _set(prog, name, value):
    try:
        prog[name].value = value
    except KeyError:
        pass


_UCACHE: dict = {}


def _setc(prog, name, value):
    """Set a uniform only if it changed since the last call (cheap per-dispatch updates).

    All uniform updates of a program must go through this function, so the cache
    always mirrors the GL state."""
    cache = _UCACHE.setdefault(prog.glo, {})
    if cache.get(name) == value:
        return
    cache[name] = value
    try:
        prog[name].value = value
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# GPU copies of reference orbit and BLA table
# ---------------------------------------------------------------------------

class GPURef:
    """Reference orbit on the GPU.

    refF: vec4 per iteration (Z.re, Z.im, small-flag, 0) - one 16-byte load per step;
    refM/refE: floatexp copy used only where the reference is tiny."""

    def __init__(self, ctx: moderngl.Context, ref: ReferenceOrbit, double: bool):
        n = ref.length
        self.length = n
        self.double = double
        self.ref = ref
        self.base_b, self.len_a, self.len_b = 0, 0, n      # (Julia: see pair())
        self.per_a, self.per_b = 0, int(getattr(ref, "period", 0) or 0)
        small = (ref.ze.astype(np.int64) <= SMALL_EXP[double]).astype(np.float64)
        f = np.zeros((n, 4), dtype=np.float64)
        f[:, 0] = ref.zf.real
        f[:, 1] = ref.zf.imag
        f[:, 2] = small
        if double == PREC_FF:
            # RefQ: (X.hi, Y.hi, X.lo, Y.lo) and (small, 0, 0, 0); mantissas as float-float
            zf = np.zeros((n, 8), dtype=np.float32)
            zf[:, 0:4] = ff_complex(ref.zf)
            zf[:, 4] = small
            zm = ff_complex(ref.zm)
        elif double:
            zf = f
            zm = np.ascontiguousarray(ref.zm.astype(np.complex128)).view(np.float64)
        else:
            with np.errstate(over="ignore", under="ignore"):
                zf = f.astype(np.float32)
                zm = np.ascontiguousarray(ref.zm.astype(np.complex64)).view(np.float32)
        ze = np.ascontiguousarray(ref.ze.astype(np.int32))
        self.bf = ctx.buffer(np.ascontiguousarray(zf).tobytes())
        self.bm = ctx.buffer(zm.tobytes())
        self.be = ctx.buffer(ze.tobytes())
        self.nbytes = zf.nbytes + zm.nbytes + ze.nbytes
        self.bxy = None
        if getattr(ref, "xym", None) is not None:
            # Burning Ship: X*Y as (mantissa, exponent) pairs in the delta precision
            xy = np.stack([ref.xym, ref.xye.astype(np.float64)], axis=1)
            xy = np.ascontiguousarray(xy if double else xy.astype(np.float32))
            self.bxy = ctx.buffer(xy.tobytes())
            self.nbytes += xy.nbytes

    def release(self):
        for b in (self.bf, self.bm, self.be, self.bxy):
            if b is not None:
                b.release()

    @classmethod
    def pair(cls, ctx: moderngl.Context, ref_a: ReferenceOrbit, ref_b: ReferenceOrbit, double: bool) -> "GPURef":
        """Julia sets: reference A (the orbit of the view's point) followed by reference B (the
        orbit of the critical point 0) in the same buffers."""
        both = ReferenceOrbit(re=ref_a.re, im=ref_a.im, prec=ref_a.prec,
                              zf=np.concatenate([ref_a.zf, ref_b.zf]), zm=np.concatenate([ref_a.zm, ref_b.zm]),
                              ze=np.concatenate([ref_a.ze, ref_b.ze]), escaped=ref_a.escaped)
        g = cls(ctx, both, double)
        g.ref = ref_a
        g.ref_b = ref_b
        g.base_b, g.len_a, g.len_b = ref_a.length, ref_a.length, ref_b.length
        g.per_a, g.per_b = int(ref_a.period or 0), int(ref_b.period or 0)
        return g


BLA_DTYPE = {
    False: np.dtype([("Am", "<f4", 2), ("Bm", "<f4", 2), ("Ae", "<i4"), ("Be", "<i4"),
                     ("logR", "<f4"), ("pad", "<f4")]),
    True: np.dtype([("Am", "<f8", 2), ("Bm", "<f8", 2), ("Ae", "<i4"), ("Be", "<i4"),
                    ("logR", "<f4"), ("pad", "<f4")]),
    PREC_FF: np.dtype([("Am", "<f4", 4), ("Bm", "<f4", 4), ("Ae", "<i4"), ("Be", "<i4"),
                       ("logR", "<f4"), ("pad", "<f4")]),
}


# second-order BLA: the quadratic coefficients C, D, E (separate buffer, same indexing)
BLAQ_DTYPE = {
    False: np.dtype([("Cm", "<f4", 2), ("Dm", "<f4", 2), ("Em", "<f4", 2), ("Ce", "<i4"), ("De", "<i4"),
                     ("Ee", "<i4"), ("pad", "<i4")]),
    True: np.dtype([("Cm", "<f8", 2), ("Dm", "<f8", 2), ("Em", "<f8", 2), ("Ce", "<i4"), ("De", "<i4"),
                    ("Ee", "<i4"), ("pad", "<i4")]),
    PREC_FF: np.dtype([("Cm", "<f4", 4), ("Dm", "<f4", 4), ("Em", "<f4", 4), ("Ce", "<i4"), ("De", "<i4"),
                       ("Ee", "<i4"), ("pad", "<i4")]),
}


def _pack_bla(bla: BLATable, double: bool):
    """Table entries as the kernel reads them: (BLA records, BLAQ records or None)."""
    arr = np.zeros(bla.size, dtype=BLA_DTYPE[double])
    ff = double == PREC_FF

    def put(rec, key, m):
        if ff:
            rec[key][:] = ff_complex(m)
        else:
            rec[key][:, 0] = m.real
            rec[key][:, 1] = m.imag

    put(arr, "Am", bla.Am)
    put(arr, "Bm", bla.Bm)
    arr["Ae"] = bla.Ae
    arr["Be"] = bla.Be
    arr["logR"] = np.clip(bla.logR, -1e30, 1e30).astype(np.float32)
    qa = None
    if getattr(bla, "order", 1) == 2 and bla.Cm is not None:
        qa = np.zeros(bla.size, dtype=BLAQ_DTYPE[double])
        for k in ("C", "D", "E"):
            put(qa, k + "m", getattr(bla, k + "m"))
            qa[k + "e"] = getattr(bla, k + "e")
    return arr, qa


def _levels(bla: BLATable, arr, base: int = 0):
    """(levels, offsets[32], counts[32], largest radius of the shortest stored skips)."""
    levels = min(bla.levels, 32)
    offsets = [int(o) + base for o in (list(bla.offsets) + [0] * 32)[:32]]
    counts = [int(c) for c in (list(bla.counts) + [0] * 32)[:32]]
    for j in range(bla.lev0):
        counts[j] = 0
    max_logR = -1e30
    # radii only shrink when entries are merged: the lowest stored level holds the largest
    # one -> the kernel skips table lookups for deltas above it
    if levels > bla.lev0 and counts[bla.lev0] > 0:
        o = offsets[bla.lev0] - base
        lev = arr["logR"][o:o + counts[bla.lev0]]
        max_logR = float(np.max(lev)) if lev.size else -1e30
    return levels, offsets, counts, max_logR


class GPUBla:
    def __init__(self, ctx: moderngl.Context, bla: BLATable | None, double: bool, bla_a: BLATable | None = None):
        """bla_a (Julia sets): the table of reference A, stored in front of `bla` (reference B)."""
        dt = BLA_DTYPE[double]
        self.table = bla
        self.order = 1
        self.qbuf = None
        self.levels_a, self.offsets_a, self.counts_a, self.max_logR_a = 0, [0] * 32, [0] * 32, -1e30
        parts, qparts = [], []
        base = 0
        if bla_a is not None and bla_a.size > 0:
            arr_a, qa_a = _pack_bla(bla_a, double)
            self.levels_a, self.offsets_a, self.counts_a, self.max_logR_a = _levels(bla_a, arr_a)
            parts.append(arr_a)
            qparts.append(qa_a)
            base = bla_a.size
        if bla is None or bla.size == 0:
            parts.append(np.zeros(1, dtype=dt))
            qparts.append(None)
            self.levels, self.lev0, self.offsets, self.counts, self.max_logR = 0, 0, [0] * 32, [0] * 32, -1e30
            if bla_a is not None and bla_a.size > 0:
                self.lev0 = bla_a.lev0
        else:
            arr, qa = _pack_bla(bla, double)
            self.lev0 = bla.lev0
            self.levels, self.offsets, self.counts, self.max_logR = _levels(bla, arr, base)
            parts.append(arr)
            qparts.append(qa)
        arr = np.concatenate(parts)
        self.buf = ctx.buffer(arr.tobytes())
        self.nbytes = arr.nbytes
        if all(q is not None for q in qparts if q is not None) and any(q is not None for q in qparts):
            qd = BLAQ_DTYPE[double]
            qs = [q if q is not None else np.zeros(len(a), dtype=qd) for q, a in zip(qparts, parts)]
            qa = np.concatenate(qs)
            self.qbuf = ctx.buffer(qa.tobytes())
            self.nbytes += qa.nbytes
            self.order = 2

    def release(self):
        self.buf.release()
        if self.qbuf is not None:
            self.qbuf.release()


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass
class FieldParams:
    off: complex = 0j            # (view centre - C_ref) / h
    rot: tuple = (1.0, 0.0)
    hm: float = 0.5
    he: int = 0
    img_w: int = 1               # full image grid (samples)
    img_h: int = 1
    tile_origin: tuple = (0, 0)
    jitter: tuple = (0.0, 0.0)
    max_iter: int = 1000
    bail2: float = 65536.0
    use_bla: bool = True
    with_de: bool = True
    with_interior: bool = True
    with_stripes: bool = False
    stripe_freq: float = 5.0
    stripe_decay: float = 0.08
    log2_interior_thr: float = math.log2(1e-3)
    double: bool = False
    log2_cmax: float = 0.0
    log2h: float = 0.0
    julia: bool = False          # Julia set by perturbation (two references, see perturb.comp)
    formula: str = "mandelbrot"  # "mandelbrot" | "ship" (the Burning Ship)

    def variant(self):
        return (self.double, self.with_de, self.with_interior, self.with_stripes, self.formula)


def view_params(view: View, ref_re: mpfr | str, ref_im: mpfr | str, img_w: int, img_h: int,
                **kw) -> FieldParams:
    log2h = view.pixel_log2(img_h)
    he = int(math.floor(log2h)) + 1
    hm = 2.0 ** (log2h - he)
    p = max(view.precision, 64)
    rr = mpfr(ref_re, p) if isinstance(ref_re, str) else ref_re
    ri = mpfr(ref_im, p) if isinstance(ref_im, str) else ref_im
    fx = mp_ratio_to_float(mpfr(view.re, p) - mpfr(rr, p), log2h)
    fy = mp_ratio_to_float(mpfr(view.im, p) - mpfr(ri, p), log2h)
    off = complex(fx, fy)
    cmax_px = abs(off) + 0.5 * math.hypot(img_w, img_h) + 2.0
    fp = FieldParams(off=off, rot=(math.cos(view.angle), math.sin(view.angle)), hm=hm, he=he,
                     img_w=img_w, img_h=img_h, max_iter=view.max_iter,
                     log2_cmax=log2h + math.log2(cmax_px), log2h=log2h)
    for k, v in kw.items():
        setattr(fp, k, v)
    return fp


def julia_uniforms(view: View, c, img_w: int, img_h: int) -> dict:
    """Uniforms of the Julia kernel: c, the view centre and the sample spacing."""
    log2h = view.pixel_log2(img_h)
    return {"uJC": (float(c[0]), float(c[1])), "uJCentre": (float(view.re), float(view.im)),
            "uJH": float(2.0 ** log2h), "uJLog2H": float(log2h)}


def julia_min_log2r(double: bool, img_h: int) -> float:
    """Deepest Julia view (log2 of the half height) the direct iteration resolves: samples
    must stay ~8 units in the last place of |z| ~ 2 apart."""
    ulp = 2.0 ** (-52 if double else -23)
    return math.log2(16.0 * ulp * img_h / 2.0)


def julia_params(view: View, img_w: int, img_h: int, **kw) -> "FieldParams":
    """FieldParams for a Julia view (no reference: offsets relative to 0)."""
    return view_params(view, mpfr(0), mpfr(0), img_w, img_h, **dict(kw, use_bla=False))


def grid_map(fv: View, tv: View, Wf: int, Hf: int, max_octaves: int = 2):
    """How the sample grid of view `tv` relates to the grid of `fv` (both Wf x Hf samples).

    Returns 'same', None (no exact relation), or (mul, div, add, de_scale) for Field.remap:
    target sample t coincides with source sample (t * mul - add) / div whenever that is an
    integer (pans: mul = div = 1; zooming by 2^k: one of them is 2^|k|)."""
    if abs(math.remainder(tv.angle - fv.angle, 2 * math.pi)) > 1e-12:
        return None
    dl = tv.log2r - fv.log2r
    k = int(round(dl))
    if abs(dl - k) > 1e-9 or abs(k) > max_octaves:
        return None
    sx, sy = fv.delta_in_pixels(tv, Hf)          # new centre in old samples (x right, y down)
    if k == 0:
        if abs(sx) < 1e-3 and abs(sy) < 1e-3:
            return "same"
        if (abs(sx - round(sx)) < 1e-3 and abs(sy - round(sy)) < 1e-3
                and abs(sx) < Wf * 0.9 and abs(sy) < Hf * 0.9):
            return ((1, 1), (1, 1), (-int(round(sx)), -int(round(sy))), 1.0)
        return None
    # new sample X sits at old sample  2^k X + C,  C = s + 2^k (1/2 - N/2) - 1/2 + N/2
    M, D = 2 ** max(k, 0), 2 ** max(-k, 0)
    add = []
    for s_, n in ((sx, Wf), (sy, Hf)):
        cd = (s_ + 2.0 ** k * (0.5 - n / 2.0) - 0.5 + n / 2.0) * D
        if abs(cd - round(cd)) > 1e-3:
            return None
        add.append(-int(round(cd)))
    return ((M, M), (D, D), tuple(add), 2.0 ** (-k))


def snap_view_to_grid(fv: View, tv: View, Wf: int, Hf: int, max_octaves: int = 2) -> View:
    """`tv` moved by less than half a sample so that grid_map(fv, tv) is an exact zoom, when
    `tv` is `fv` zoomed by a power of two (otherwise `tv` unchanged)."""
    dl = tv.log2r - fv.log2r
    k = int(round(dl))
    if (k == 0 or abs(k) > max_octaves or abs(dl - k) > 1e-6
            or abs(math.remainder(tv.angle - fv.angle, 2 * math.pi)) > 1e-9):
        return tv
    t = tv.copy()
    t.log2r = fv.log2r + k
    t.angle = fv.angle
    sx, sy = fv.delta_in_pixels(t, Hf)
    D = 2 ** max(-k, 0)
    fix = []
    for s_, n in ((sx, Wf), (sy, Hf)):
        cd = (s_ + 2.0 ** k * (0.5 - n / 2.0) - 0.5 + n / 2.0) * D
        fix.append((round(cd) - cd) / D * 2.0 ** (-k))       # in new samples
    if fix[0] or fix[1]:
        t = t.offset(fix[0], fix[1], Hf)
        t.log2r = fv.log2r + k
    t.max_iter = tv.max_iter
    return t


def progressive_order(W: int, H: int, with_segments: bool = False):
    """Sample order coarse -> fine: every 8th sample, then 4th, 2nd, rest."""
    yy, xx = np.mgrid[0:H, 0:W]
    lvl = np.zeros((H, W), np.int8)
    lvl[(yy % 2 == 0) & (xx % 2 == 0)] = 1
    lvl[(yy % 4 == 0) & (xx % 4 == 0)] = 2
    lvl[(yy % 8 == 0) & (xx % 8 == 0)] = 3
    flat = lvl.ravel()
    parts = [np.flatnonzero(flat == L) for L in (3, 2, 1, 0)]
    order = np.concatenate(parts).astype(np.uint32)
    if not with_segments:
        return order
    bounds, a = [], 0
    for p in parts:
        bounds.append((a, a + len(p)))
        a += len(p)
    return order, bounds


# ---------------------------------------------------------------------------
# small helper shaders
# ---------------------------------------------------------------------------

FULLSCREEN_VS = """
#version 430
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

BLIT_FS = """
#version 430
uniform sampler2D uTex;
uniform float uScale;      // 1 / accumulated count (resolve) or 1
uniform ivec2 uSrcOffset;
out vec4 fragColor;
void main() {
    vec3 c = texelFetch(uTex, ivec2(gl_FragCoord.xy) + uSrcOffset, 0).rgb * uScale;
    fragColor = vec4(c, 1.0);
}
"""

# shift (pan reuse, with GPU-side counting), reopen max-iteration samples, clear
WORD_SHADERS = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer S0 { uint s0[]; };
layout(std430, binding = 1) buffer R0 { uvec4 r0[]; };
layout(std430, binding = 2) buffer S1 { uint s1[]; };
layout(std430, binding = 3) buffer R1 { uvec4 r1[]; };
layout(std430, binding = 4) buffer Cnt { uint cnt[]; };
uniform uvec2 uField;
uniform ivec2 uMul;       // mode 0: target sample t takes source (t * uMul - uAdd) / uDiv when exact
uniform ivec2 uDiv;       //   (pan: 1, 1, shift;  zoom in by 2^k: 1, 2^k, a;  zoom out: 2^k, 1, a)
uniform ivec2 uAdd;
uniform float uDEScale;   // distance estimates are in pixels: rescale them when the pixel size changes
uniform uint uStride;     // words per state entry
uniform uint uStatusWord;
uniform uint uMode;       // 0 = remap copy (0->1) + count, 1 = reopen max-iter samples, 2 = clear
shared uint sDone, sMaxIt, sInt, sLo, sHi;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = uField.x * uField.y;
    if (uMode == 2u) {
        if (i < N) {
            s0[i * uStride + uStatusWord] = 0u;
            r0[2u * i] = uvec4(0u);
            r0[2u * i + 1u] = uvec4(0u);
        }
        return;
    }
    if (uMode == 1u) {
        if (i < N && r0[2u * i].x == 0xFFFFFFFFu) {
            r0[2u * i] = uvec4(0u);
            r0[2u * i + 1u] = uvec4(0u);
            s0[i * uStride + uStatusWord] = 1u;      // continue from the saved state
        }
        return;
    }
    if (gl_LocalInvocationIndex == 0u) { sDone = 0u; sMaxIt = 0u; sInt = 0u; sLo = 0xFFFFFFFFu; sHi = 0u; }
    barrier();
    if (i < N) {
        int x = int(i % uField.x), y = int(i / uField.x);
        int nx = x * uMul.x - uAdd.x, ny = y * uMul.y - uAdd.y;
        int sx = nx / uDiv.x, sy = ny / uDiv.y;
        bool exact = nx >= 0 && ny >= 0 && (nx % uDiv.x) == 0 && (ny % uDiv.y) == 0;
        uint n = 0u;
        if (exact && sx < int(uField.x) && sy < int(uField.y)) {
            uint j = uint(sy) * uField.x + uint(sx);
            for (uint w = 0u; w < uStride; ++w) s1[i * uStride + w] = s0[j * uStride + w];
            uvec4 a = r0[2u * j];
            if (uDEScale != 1.0) a.z = floatBitsToUint(uintBitsToFloat(a.z) * uDEScale);
            r1[2u * i] = a;
            r1[2u * i + 1u] = r0[2u * j + 1u];
            n = a.x;
        } else {
            s1[i * uStride + uStatusWord] = 0u;
            r1[2u * i] = uvec4(0u);
            r1[2u * i + 1u] = uvec4(0u);
        }
        if (n != 0u) {
            atomicAdd(sDone, 1u);
            if (n == 0xFFFFFFFFu) atomicAdd(sMaxIt, 1u);
            else if (n == 0xFFFFFFFEu) atomicAdd(sInt, 1u);
            else { atomicMin(sLo, n); atomicMax(sHi, n); }
        }
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        if (sDone > 0u) atomicAdd(cnt[1], sDone);
        if (sLo != 0xFFFFFFFFu) atomicMin(cnt[2], sLo);
        if (sHi != 0u) atomicMax(cnt[3], sHi);
        if (sMaxIt > 0u) atomicAdd(cnt[8], sMaxIt);
        if (sInt > 0u) atomicAdd(cnt[9], sInt);
    }
}
"""

# per 8x8 block: best reference candidate (non-escaping first, then highest iteration count)
REDUCE_CS = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 1) readonly buffer R0 { uvec4 r0[]; };
layout(std430, binding = 2) writeonly buffer Out { uvec2 outv[]; };
uniform uvec2 uField;
uniform uvec2 uBlocks;
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= uBlocks.x * uBlocks.y) return;
    uint bx = (b % uBlocks.x) * 8u, by = (b / uBlocks.x) * 8u;
    uint best = 0u, bi = 0xFFFFFFFFu;
    for (uint y = by; y < min(by + 8u, uField.y); ++y)
        for (uint x = bx; x < min(bx + 8u, uField.x); ++x) {
            uint i = y * uField.x + x;
            uint n = r0[2u * i].x;
            if (n > best) { best = n; bi = i; }
        }
    outv[b] = uvec2(best, bi);
}
"""

# How the samples of a finished picture ended, by octave of the iteration limit L:
#   out[k]      escaped at n in (L/2^(k+1), L/2^k]          k = 0..7 (k = 7: everything below)
#   out[8 + k]  recognised as inside at that iteration (the result keeps the iteration in .y)
#   out[16]     undecided (stopped at the limit, or escaped later than a lowered limit)
#   out[17]     finished samples
# The automatic iteration limit decides from it whether doubling the limit would reveal outside
# detail or only confirm slowly converging inside points (app.iteration_advice).
OCTAVE_CS = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer R0 { uvec4 res[]; };
layout(std430, binding = 2) buffer Out { uint outv[18]; };
uniform uint uN;
uniform uint uLimit;
shared uint sh[18];
void main() {
    uint lid = gl_LocalInvocationIndex;
    if (lid < 18u) sh[lid] = 0u;
    barrier();
    uint total = gl_NumWorkGroups.x * 256u;
    float L = float(uLimit);
    for (uint i = gl_GlobalInvocationID.x; i < uN; i += total) {
        uvec4 r = res[2u * i];
        if (r.x == 0u) continue;
        atomicAdd(sh[17], 1u);
        uint n;
        uint base;
        if (r.x == 0xFFFFFFFFu) { atomicAdd(sh[16], 1u); continue; }
        if (r.x == 0xFFFFFFFEu) { n = uint(max(uintBitsToFloat(r.y), 1.0)); base = 8u; }
        else { n = r.x; base = 0u; }
        if (n > uLimit) {
            if (base == 0u) { atomicAdd(sh[16], 1u); continue; }
            n = uLimit;
        }
        uint k = uint(clamp(floor(log2(L / float(max(n, 1u)))), 0.0, 7.0));
        atomicAdd(sh[base + k], 1u);
    }
    barrier();
    if (lid < 18u && sh[lid] != 0u) atomicAdd(outv[lid], sh[lid]);
}
"""

HIST_CS = """
#version 430
// Histogram of log2(iterations) for the 'auto' colour mapping.  Each work group counts into
// shared memory first and adds its non-empty bins to the global histogram once: a direct
// global atomicAdd per sample piles hundreds of thousands of updates onto a few bins, which
// NVIDIA serialises (~40 ms per 1080p frame on an RTX 5060 Ti).
layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer ResultBuf { uvec4 res[]; };
layout(std430, binding = 2) buffer Hist { uint hist[]; };
uniform uint uN;
uniform uint uStride;
uniform vec2 uRange;
uniform uint uW;             // field width (for the optional rectangle)
uniform uvec4 uRect;         // x0, y0, x1, y1 (samples): only count inside
uniform int uXf;             // 1: also only inside a (rotated) frame: d = InvA * (pos + Off)
uniform vec2 uXfInvA;
uniform vec2 uXfOff;
uniform vec2 uXfHalf;
shared uint sh[1024];
void main() {
    uint lid = gl_LocalInvocationIndex;
    for (uint b = lid; b < 1024u; b += 256u) sh[b] = 0u;
    barrier();
    uint cnt = (uN + uStride - 1u) / uStride;
    uint total = gl_NumWorkGroups.x * 256u;
    float inv = 1.0 / max(uRange.y - uRange.x, 1e-6);
    for (uint k = gl_GlobalInvocationID.x; k < cnt; k += total) {
        uint i = k * uStride;
        uint px = i % uW, py = i / uW;
        if (px < uRect.x || py < uRect.y || px >= uRect.z || py >= uRect.w) continue;
        if (uXf != 0) {
            vec2 q = vec2(float(px) + 0.5, float(py) + 0.5) + uXfOff;
            vec2 d = vec2(uXfInvA.x * q.x - uXfInvA.y * q.y, uXfInvA.y * q.x + uXfInvA.x * q.y);
            if (abs(d.x) > uXfHalf.x || abs(d.y) > uXfHalf.y) continue;
        }
        uvec4 r = res[2u * i];
        if (r.x == 0u || r.x >= 0xFFFFFFFEu) continue;
        float x = (log2(float(r.x) + uintBitsToFloat(r.y)) - uRange.x) * inv;
        atomicAdd(sh[uint(clamp(x, 0.0, 0.99999) * 1024.0)], 1u);
    }
    barrier();
    for (uint b = lid; b < 1024u; b += 256u)
        if (sh[b] != 0u) atomicAdd(hist[b], sh[b]);
}
"""


SYNC_BARRIER = moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT
RUN_LENGTH = 16                 # plain-float iterations per run (derivatives stay far from overflow)


@dataclass
class EngineOptions:
    """Switches for A/B measurements (``mandelscope bench --experiments``).

    The defaults are the fast settings; the alternatives reproduce the v2 behaviour."""
    barrier_each_dispatch: bool = False   # v2: full memory barrier after every dispatch
    finish_before_read: bool = False      # v2: glFinish before reading the counters
    bla_gate: bool = True                 # skip BLA lookups above the largest radius
    base_budget: int = 2048               # steps per sample per dispatch (first sweep)
    sync_per_level: bool = False          # v3.0: wait for the counters after every progressive level
    chunk_start: int = 65536              # samples per dispatch at the start of every sweep
    chunk_min: int = 8192                 # never smaller (v3.0 went down to 64 and kept it)
    runs: bool = True                     # plain-float iteration runs without per-step bookkeeping
    run_length: int = 32                  # iterations per run (derivatives grow <= 4x per iteration;
                                          # RTX bench: 32 beat 16 by 0-6 %)
    staged_counters: bool = True          # CPU reads/writes small GPU buffers through a copy
    predictive_budget: bool = True        # stop before a dispatch that would overrun the time slice
    sync_before_step: bool = True         # time slices measure only the field's own dispatches
    bla_order: int = 2                    # 2: second-order BLA (larger skips); 1: v3.8
    bla_fast: bool = True                 # skips in plain arithmetic when the terms fit (v3.11)
    bla_backoff: bool = False             # after a failed lookup wait 2x, 4x ... longer (experiment)
    bla_prefilter: bool = True            # runs check the shortest skip's radius before a lookup (v3.11)


# Adaptive supersampling: seed a supersampled field from a field with one sample per pixel.
# Pixels whose colour matches all 8 neighbours (and that are not within uDEThr pixels of the
# set) get their centre sample copied into all ss*ss sub-samples (marked done); only the
# others are computed.  The centre sample is the mean position of the sub-samples, so for
# smooth regions the result equals the box-filtered average up to curvature terms.
ADAPT_CS = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer S1 { uint s1[]; };
layout(std430, binding = 1) readonly buffer R0 { uvec4 r0[]; };
layout(std430, binding = 3) buffer R1 { uvec4 r1[]; };
layout(std430, binding = 4) buffer Cnt { uint cnt[]; };
uniform sampler2D uCol;      // the centre field coloured (row 0 = bottom)
uniform uvec2 uSize;         // centre field size (may have one extra row above / below the tile)
uniform uint uRows;          // pixel rows of the target tile
uniform uint uRow0;          // centre-field row of the tile's first row
uniform uint uSS;            // sub-samples per pixel and axis in the target field
uniform uint uStride;
uniform uint uStatusWord;
uniform float uColThr;
uniform float uDEThr;
uniform float uDEScale;
shared uint sDone, sMaxIt, sInt, sLo, sHi, sFlag;
void main() {
    uint p = gl_GlobalInvocationID.x;
    if (gl_LocalInvocationIndex == 0u) { sDone = 0u; sMaxIt = 0u; sInt = 0u; sLo = 0xFFFFFFFFu; sHi = 0u; sFlag = 0u; }
    barrier();
    uint N = uSize.x * uRows;
    if (p < N) {
        int x = int(p % uSize.x), ty = int(p / uSize.x);
        int y = ty + int(uRow0);                   // row in the centre field
        uint pc = uint(y) * uSize.x + uint(x);
        uvec4 a = r0[2u * pc];
        uvec4 b = r0[2u * pc + 1u];
        ivec2 lim = ivec2(uSize) - 1;
        ivec2 tc = ivec2(x, lim.y - y);
        vec3 c0 = texelFetch(uCol, tc, 0).rgb;
        // second differences along x, y and both diagonals: a smooth colour ramp averages to
        // its centre value (only curvature changes the pixel mean), edges and texture do not
        float dmax = 0.0;
        ivec2 dirs[4] = ivec2[4](ivec2(1, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, -1));
        for (int k = 0; k < 4; ++k) {
            vec3 cp = texelFetch(uCol, clamp(tc + dirs[k], ivec2(0), lim), 0).rgb;
            vec3 cm = texelFetch(uCol, clamp(tc - dirs[k], ivec2(0), lim), 0).rgb;
            vec3 d = abs(cp + cm - 2.0 * c0);
            dmax = max(dmax, max(d.r, max(d.g, d.b)));
        }
        // class changes next to the pixel (escaped / proven interior / iteration limit) always
        // count, and samples stopped by the iteration limit are undecided: always refined
        uint cls = a.x == 0xFFFFFFFFu ? 2u : (a.x == 0xFFFFFFFEu ? 1u : 0u);
        bool inside = cls != 0u;
        bool mixed = cls == 2u;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                ivec2 q = clamp(ivec2(x + dx, y + dy), ivec2(0), lim);
                uint nq = r0[2u * (uint(q.y) * uSize.x + uint(q.x))].x;
                uint cq = nq == 0xFFFFFFFFu ? 2u : (nq == 0xFFFFFFFEu ? 1u : 0u);
                mixed = mixed || cq != cls;
            }
        float de = uintBitsToFloat(a.z);
        bool refine = a.x == 0u || mixed || dmax > uColThr || (!inside && de < uDEThr);
        uint SS = uSS;
        uint TW = uSize.x * SS;
        if (refine) {
            atomicAdd(sFlag, 1u);           // stays 'new' (the target was cleared): computed
        } else {
            uvec4 a2 = a;
            a2.z = floatBitsToUint(de * uDEScale);
            for (uint j = 0u; j < SS; ++j)
                for (uint i = 0u; i < SS; ++i) {
                    uint t = (uint(ty) * SS + j) * TW + uint(x) * SS + i;
                    s1[t * uStride + uStatusWord] = 2u;
                    r1[2u * t] = a2;
                    r1[2u * t + 1u] = b;
                }
            uint k = SS * SS;
            atomicAdd(sDone, k);
            if (a.x == 0xFFFFFFFFu) atomicAdd(sMaxIt, k);
            else if (a.x == 0xFFFFFFFEu) atomicAdd(sInt, k);
            else { atomicMin(sLo, a.x); atomicMax(sHi, a.x); }
        }
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        if (sDone > 0u) atomicAdd(cnt[1], sDone);
        if (sLo != 0xFFFFFFFFu) atomicMin(cnt[2], sLo);
        if (sHi != 0u) atomicMax(cnt[3], sHi);
        if (sMaxIt > 0u) atomicAdd(cnt[8], sMaxIt);
        if (sInt > 0u) atomicAdd(cnt[9], sInt);
        if (sFlag > 0u) atomicAdd(cnt[11], sFlag);
    }
}
"""

ADAPT_COL_THR = 4.0 / 255.0     # colour second difference that asks for sub-samples
ADAPT_DE_THR = 0.75             # pixels: closer to the set than this -> sub-samples


class GPUEngine:
    def __init__(self, ctx: moderngl.Context, options: EngineOptions | None = None):
        self.ctx = ctx
        self.opt = options or EngineOptions()
        self.has_fp64 = ("GL_ARB_gpu_shader_fp64" in ctx.extensions) or ctx.version_code >= 400
        self._progs: dict = {}
        self._orders: dict = {}
        self._fma_exact = None
        self.renderer_name = ctx.info.get("GL_RENDERER", "?")
        self.word_prog = ctx.compute_shader(WORD_SHADERS)
        self.reduce_prog = ctx.compute_shader(REDUCE_CS)
        self.adapt_prog = ctx.compute_shader(ADAPT_CS)
        self.octave_prog = ctx.compute_shader(OCTAVE_CS)
        self.display = Display(ctx)

    def perturb_program(self, double: bool, de: bool, interior: bool, stripes: bool = False,
                        formula: str = "mandelbrot"):
        key = (int(double), bool(de), bool(interior), bool(stripes), formula)
        if key not in self._progs:
            defs = ["#version 430", f"#define RUN_LENGTH {RUN_LENGTH}"]
            ff = int(double) == PREC_FF
            if formula == "ship":
                if ff:
                    raise ValueError("the Burning Ship has no float-float kernel (use float64)")
                defs.append("#define SHIP 1")
            if double and not ff:
                defs.append("#define USE_DOUBLE 1")
            if ff and self.fma_exact:
                defs.append("#define FF_FMA 1")
            if de:
                defs.append("#define WITH_DE 1")
            if interior:
                defs.append("#define WITH_INTERIOR 1")
            if stripes:
                defs.append("#define WITH_STRIPES 1")
            src = "\n".join(defs) + "\n" + _src("perturb_ff.comp" if ff else "perturb.comp")
            self._progs[key] = self.ctx.compute_shader(src)
        return self._progs[key]

    @property
    def fma_exact(self) -> bool:
        """Does fma() round once on this GPU?  (Float-float products need it; Mesa's llvmpipe,
        for one, computes a * b + c with two roundings.)"""
        if self._fma_exact is None:
            try:
                prog = self.ctx.compute_shader("""#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer O { float o[]; };
uniform float a;
uniform float b;
void main() { precise float p = a * b; precise float e = fma(a, b, -p); o[0] = e; }""")
                buf = self.ctx.buffer(reserve=16)
                buf.bind_to_storage_buffer(0)
                prog["a"].value = 1.0000001
                prog["b"].value = 3.0000002
                prog.run(1)
                self._fma_exact = bool(np.frombuffer(buf.read(), dtype=np.float32)[0] != 0.0)
                buf.release()
                prog.release()
            except Exception:
                self._fma_exact = False
        return self._fma_exact

    def julia_program(self, double: bool, de: bool, interior: bool, stripes: bool = False, formula=None):
        """Kernel for Julia sets (direct iteration; same buffers and results as perturbation)."""
        key = ("julia", bool(double), bool(de), bool(interior), bool(stripes))
        if key not in self._progs:
            defs = ["#version 430"]
            if double:
                defs.append("#define USE_DOUBLE 1")
            if de:
                defs.append("#define WITH_DE 1")
            if interior:
                defs.append("#define WITH_INTERIOR 1")
            if stripes:
                defs.append("#define WITH_STRIPES 1")
            self._progs[key] = self.ctx.compute_shader("\n".join(defs) + "\n" + _src("julia.comp"))
        return self._progs[key]

    def setup_julia(self, field: "Field", view: View, c, img_w: int, img_h: int, params: "FieldParams"):
        """Make ``field`` compute the Julia set of ``c`` for ``view`` (call after configure)."""
        field.program = self.julia_program(*params.variant())
        field.extra_uniforms = julia_uniforms(view, c, img_w, img_h)

    def warmup(self, double: bool = False):
        for de in (True, False):
            for it in (True, False):
                self.perturb_program(double, de, it)

    def order_buffer(self, W: int, H: int):
        key = (W, H)
        if key not in self._orders:
            if len(self._orders) > 8:
                for k, b in list(self._orders.items())[:4]:
                    b[0].release()
                    del self._orders[k]
            order, bounds = progressive_order(W, H, with_segments=True)
            self._orders[key] = (self.ctx.buffer(order.tobytes()), bounds)
        return self._orders[key]


# ---------------------------------------------------------------------------
# a field of samples being computed
# ---------------------------------------------------------------------------

class Field:
    """A grid of samples with persistent per-sample iteration state.

    Scheduling: the progressive levels (1/64, 3/64, 12/64, 48/64 of the samples)
    are swept one after another; samples that run out of per-invocation budget
    go to a pending list.  The next level starts once almost all samples of the
    current one are finished, so the coarse preview fills in evenly.
    Several dispatches are queued between CPU/GPU synchronisations.
    """

    def __init__(self, engine: GPUEngine, W: int, H: int, double: bool):
        self.engine = engine
        ctx = engine.ctx
        self.W, self.H = int(W), int(H)
        self.double = PREC_FF if int(double) == PREC_FF else bool(double)
        self.N = self.W * self.H
        self.stride = STATE_STRIDE[self.double]
        self.state = ctx.buffer(reserve=self.N * self.stride)
        self.result = ctx.buffer(reserve=self.N * RESULT_WORDS * 4)
        self.state2 = None
        self.result2 = None
        self.pend = [ctx.buffer(reserve=self.N * 4), ctx.buffer(reserve=self.N * 4)]
        zeros = bytes(self.N * 4)
        for b in self.pend:
            b.write(zeros)  # commit the memory now, not during the first picture (measured 40 ms)
        # The counters are updated by atomics of every work group.  The CPU never maps them:
        # it reads and writes through two small staging buffers and GPU-side copies.  (A buffer
        # the CPU reads back all the time may be moved to system memory by the driver, and
        # then every atomic on it crosses the PCIe bus.)
        self.counters = ctx.buffer(reserve=N_COUNTERS * 4)
        self.cnt_down = ctx.buffer(reserve=N_COUNTERS * 4, dynamic=True)
        self.cnt_up = ctx.buffer(reserve=N_COUNTERS * 4, dynamic=True)
        self.order, self.segments = engine.order_buffer(self.W, self.H)
        self.params: FieldParams | None = None
        self.gref: GPURef | None = None
        self.gbla: GPUBla | None = None
        self.program = None             # override (e.g. Julia kernel)
        self.base_budget = engine.opt.base_budget   # steps per invocation while sweeping the order list
        self.chunk = 32768
        self.batch_k = 2
        self.disp_est = 0.0
        self.budget = self.base_budget
        self.tag = None                 # free for the owner (e.g. which view was configured)
        self.flagged = 0
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def release(self):
        for b in (self.state, self.result, self.state2, self.result2, *self.pend, self.counters,
                  self.cnt_down, self.cnt_up):
            if b is not None:
                b.release()

    def buffer_names(self) -> dict:
        """GL object name -> what it is (to read driver messages)."""
        out = {self.state.glo: "field state", self.result.glo: "field results",
               self.pend[0].glo: "pending list", self.pend[1].glo: "pending list",
               self.counters.glo: "field counters", self.cnt_down.glo: "counter read-back copy",
               self.cnt_up.glo: "counter upload copy"}
        if self.state2 is not None:
            out[self.state2.glo] = "field state"
            out[self.result2.glo] = "field results"
        return out

    @property
    def nbytes(self) -> int:
        n = self.state.size + self.result.size + 2 * self.pend[0].size
        if self.state2 is not None:
            n += self.state2.size + self.result2.size
        return n

    def _write_counters(self, done=0, lo=0xFFFFFFFF, hi=0, maxit=0, interior=0):
        arr = np.array([0, done, lo, hi, 0, 0, 0, 0, maxit, interior, 0, 0], dtype=np.uint32)
        self._put_counters(arr.tobytes())

    def _put_counters(self, data: bytes):
        if self.engine.opt.staged_counters:
            self.cnt_up.write(data)
            self.engine.ctx.copy_buffer(self.counters, self.cnt_up)
        else:
            self.counters.write(data)

    def _start_sweep(self):
        self.seg_idx = 0
        self.order_pos, self.seg_end = self.segments[0]
        self.pend_in = 0
        self.pend_in_count = 0
        self.pend_in_pos = 0
        self.pend_checked = False
        self.complete = False
        self.finished_at = None
        # every sweep starts with dispatches big enough to fill the GPU (v3.0 carried a size
        # shrunk by a heavy view over to the next picture: 64-sample dispatches, 30-60x slower)
        self.chunk = self.engine.opt.chunk_start
        self.batch_k = 2
        self.disp_est = 0.0
        self.budget = self.base_budget
        self.started = time.perf_counter()

    def _clear_stats(self):
        self.dispatches = 0
        self.syncs = 0
        self.gpu_time = 0.0
        self.steps = 0
        self.bla_skips = 0
        self.bla_fails = 0
        self.rebases = 0

    def _word_dispatch(self, mode: int, mul=(1, 1), div=(1, 1), add=(0, 0), de_scale=1.0):
        p = self.engine.word_prog
        self.state.bind_to_storage_buffer(0)
        self.result.bind_to_storage_buffer(1)
        if mode == 0:
            self.state2.bind_to_storage_buffer(2)
            self.result2.bind_to_storage_buffer(3)
        self.counters.bind_to_storage_buffer(4)
        _setc(p, "uField", (self.W, self.H))
        _setc(p, "uMul", (int(mul[0]), int(mul[1])))
        _setc(p, "uDiv", (int(div[0]), int(div[1])))
        _setc(p, "uAdd", (int(add[0]), int(add[1])))
        _setc(p, "uDEScale", float(de_scale))
        _setc(p, "uStride", self.stride // 4)
        _setc(p, "uStatusWord", STATUS_WORD[self.double])
        _setc(p, "uMode", int(mode))
        p.run((self.N + 255) // 256)
        self.engine.ctx.memory_barrier()

    def reset(self):
        """Forget all samples (GPU-side clear: no CPU upload of the big buffers)."""
        self._word_dispatch(2)
        self.done_count = 0
        self.maxiter_count = 0
        self.interior_count = 0
        self.min_iter = None
        self.max_iter_seen = None
        self._clear_stats()
        self._start_sweep()
        self._write_counters()

    def configure(self, params: FieldParams, gref: GPURef | None, gbla: GPUBla | None):
        stride = (STATE_STRIDE_SHIP if params.formula == "ship" else STATE_STRIDE)[self.double]
        if stride != self.stride:
            # another formula keeps more per sample: new state buffers (all samples restart)
            for b in (self.state, self.state2, self.result2):
                if b is not None:
                    b.release()
            self.stride = stride
            self.state = self.engine.ctx.buffer(reserve=self.N * self.stride)
            self.state2 = self.result2 = None
            self.reset()
        self.params = params
        self.gref = gref
        self.gbla = gbla

    @property
    def progress(self) -> float:
        return min(1.0, self.done_count / max(self.N, 1))

    @property
    def level_done(self) -> int:
        """Finest progressive level that is (almost) complete: 4 = none, 0 = all."""
        if self.complete:
            return 0
        return 4 - self.seg_idx

    # ---------------------------------------------------------- reuse tools
    def shift(self, dx: int, dy: int):
        """Move already computed samples by (dx, dy) (same reference & scale).

        Copy and counting happen on the GPU; counters are refreshed at the next step()."""
        self.remap((1, 1), (1, 1), (int(dx), int(dy)))

    def remap(self, mul, div, add, de_scale: float = 1.0):
        """Keep the samples that land exactly on the new grid: target t <- source
        (t * mul - add) / div per axis (pan: mul = div = 1; zoom by 2^k: mul or div = 2^k).

        Samples are exact orbit states of their point c, so they stay valid on any grid that
        contains c (same reference).  Runs on the GPU, including the counting."""
        ctx = self.engine.ctx
        if self.state2 is None:
            self.state2 = ctx.buffer(reserve=self.state.size)
            self.result2 = ctx.buffer(reserve=self.result.size)
        self._write_counters()
        self._word_dispatch(0, mul, div, add, de_scale)
        self.state, self.state2 = self.state2, self.state
        self.result, self.result2 = self.result2, self.result
        self._clear_stats()
        self._start_sweep()

    def reopen_maxiter(self):
        """Continue samples that hit the iteration limit (after raising it)."""
        self._word_dispatch(1)
        self.done_count = max(0, self.done_count - self.maxiter_count)
        self.maxiter_count = 0
        self._write_counters(done=self.done_count,
                             lo=self.min_iter if self.min_iter is not None else 0xFFFFFFFF,
                             hi=self.max_iter_seen or 0, interior=self.interior_count)
        self._clear_stats()
        self._start_sweep()

    def seed_adaptive(self, centre: "Field", colour_tex, ss: int, row0: int = 0,
                      col_thr: float | None = None, de_thr: float | None = None):
        """Adaptive supersampling: call right after reset().  ``centre`` holds one finished
        sample per pixel (same view; rows ``row0`` .. ``row0 + H/ss`` are this tile, the rows
        around it only serve as neighbours), ``colour_tex`` its colours.  Smooth pixels are
        filled from it, the rest is left to compute."""
        rows = self.H // ss
        if centre.W * ss != self.W or rows * ss != self.H or row0 < 0 or row0 + rows > centre.H:
            raise ValueError("centre field does not match")
        p = self.engine.adapt_prog
        self.state.bind_to_storage_buffer(0)
        centre.result.bind_to_storage_buffer(1)
        self.result.bind_to_storage_buffer(3)
        self.counters.bind_to_storage_buffer(4)
        colour_tex.use(0)
        _setc(p, "uCol", 0)
        _setc(p, "uSize", (centre.W, centre.H))
        _setc(p, "uRows", int(rows))
        _setc(p, "uRow0", int(row0))
        _setc(p, "uSS", int(ss))
        _setc(p, "uStride", self.stride // 4)
        _setc(p, "uStatusWord", STATUS_WORD[self.double])
        _setc(p, "uColThr", float(ADAPT_COL_THR if col_thr is None else col_thr))
        _setc(p, "uDEThr", float(ADAPT_DE_THR if de_thr is None else de_thr))
        _setc(p, "uDEScale", float(ss))
        p.run((centre.W * rows + 255) // 256)
        self.engine.ctx.memory_barrier()

    def reference_candidates(self) -> np.ndarray:
        """(blocks_y, blocks_x, 2) best (n, index) per 8x8 block - a small read-back."""
        bx, by = (self.W + 7) // 8, (self.H + 7) // 8
        buf = self.engine.ctx.buffer(reserve=bx * by * 8)
        p = self.engine.reduce_prog
        self.result.bind_to_storage_buffer(1)
        buf.bind_to_storage_buffer(2)
        _setc(p, "uField", (self.W, self.H))
        _setc(p, "uBlocks", (bx, by))
        p.run((bx * by + 63) // 64)
        self.engine.ctx.memory_barrier()
        out = np.frombuffer(buf.read(), dtype=np.uint32).reshape(by, bx, 2).copy()
        buf.release()
        return out

    def iteration_octaves(self, limit: int) -> dict:
        """Octave statistics of the finished samples (see OCTAVE_CS); one small read-back."""
        ctx = self.engine.ctx
        buf = ctx.buffer(np.zeros(18, dtype=np.uint32).tobytes())
        p = self.engine.octave_prog
        self.result.bind_to_storage_buffer(1)
        buf.bind_to_storage_buffer(2)
        _setc(p, "uN", int(self.N))
        _setc(p, "uLimit", int(max(1, min(limit, 0xFFFFFFFD))))
        p.run(min(1024, (self.N + 255) // 256))
        ctx.memory_barrier()
        c = np.frombuffer(buf.read(), dtype=np.uint32).astype(np.int64)
        buf.release()
        return {"esc": c[0:8].tolist(), "int": c[8:16].tolist(), "und": int(c[16]), "done": int(c[17]),
                "N": int(self.N), "limit": int(limit)}

    # ------------------------------------------------------------- compute
    def _prepare(self):
        """Bind buffers and set the uniforms that are constant for this field."""
        prm = self.params
        prog = self.program or self.engine.perturb_program(*prm.variant())
        self.state.bind_to_storage_buffer(0)
        self.result.bind_to_storage_buffer(1)
        self.order.bind_to_storage_buffer(6)
        self.counters.bind_to_storage_buffer(9)
        _setc(prog, "uField", (self.W, self.H))
        _setc(prog, "uTileOrigin", tuple(int(v) for v in prm.tile_origin))
        _setc(prog, "uImgHalf", (prm.img_w / 2.0, prm.img_h / 2.0))
        if int(prm.double) == PREC_FF:
            o = ff_complex(prm.off)[0]
            _setc(prog, "uOff", tuple(float(v) for v in o))
            hh, hl = ff_split([prm.hm])
            _setc(prog, "uHm", (float(hh[0]), float(hl[0])))
        else:
            _setc(prog, "uOff", (prm.off.real, prm.off.imag))
            _setc(prog, "uHm", prm.hm)
        _setc(prog, "uRot", tuple(prm.rot))
        _setc(prog, "uHe", int(prm.he))
        _setc(prog, "uMaxIter", int(prm.max_iter))
        _setc(prog, "uBail2", float(prm.bail2))
        _setc(prog, "uLog2InteriorThr", float(prm.log2_interior_thr))
        _setc(prog, "uInteriorExp", int(math.ceil(prm.log2_interior_thr)))
        _setc(prog, "uJitter", tuple(float(v) for v in prm.jitter))
        _setc(prog, "uStripeFreq", float(prm.stripe_freq))
        _setc(prog, "uStripeDecay", float(prm.stripe_decay))
        _setc(prog, "uRun", int(self.engine.opt.run_length) if self.engine.opt.runs else 0)
        if self.gref is not None:
            self.gref.bf.bind_to_storage_buffer(2)
            self.gref.bm.bind_to_storage_buffer(3)
            self.gref.be.bind_to_storage_buffer(4)
            if self.gref.bxy is not None:
                self.gref.bxy.bind_to_storage_buffer(11)
            _setc(prog, "uRefLen", int(self.gref.length))
            _setc(prog, "uBaseB", int(self.gref.base_b))
            _setc(prog, "uLenA", int(self.gref.len_a))
            _setc(prog, "uLenB", int(self.gref.len_b))
            _setc(prog, "uPerA", int(self.gref.per_a))
            _setc(prog, "uPerB", int(self.gref.per_b))
        _setc(prog, "uJulia", 1 if prm.julia else 0)
        _setc(prog, "uDOne", 0.0 if prm.julia else 1.0)
        gb = self.gbla
        if gb is not None:
            gb.buf.bind_to_storage_buffer(5)
            order = gb.order if self.engine.opt.bla_order == 2 else 1
            (gb.qbuf if order == 2 else gb.buf).bind_to_storage_buffer(10)
            _setc(prog, "uBlaOrder", int(order))
            _setc(prog, "uBlaFast", 1 if self.engine.opt.bla_fast else 0)
            _setc(prog, "uBlaBackoff", 1 if self.engine.opt.bla_backoff else 0)
            _setc(prog, "uBlaPrefilter", 1 if self.engine.opt.bla_prefilter else 0)
            _setc(prog, "uLev0", int(gb.lev0))
            _setc(prog, "uLevels", int(gb.levels if prm.use_bla else 0))
            _setc(prog, "uLevOff", tuple(int(v) for v in gb.offsets))
            _setc(prog, "uLevCnt", tuple(int(v) for v in gb.counts))
            _setc(prog, "uUseBla", 1 if (prm.use_bla and gb.levels > 0) else 0)
            _setc(prog, "uBlaMaxLogR", float(gb.max_logR) if self.engine.opt.bla_gate else 1e30)
            _setc(prog, "uLevelsA", int(gb.levels_a if prm.use_bla else 0))
            _setc(prog, "uLevOffA", tuple(int(v) for v in gb.offsets_a))
            _setc(prog, "uLevCntA", tuple(int(v) for v in gb.counts_a))
            _setc(prog, "uBlaMaxLogRA", float(gb.max_logR_a) if self.engine.opt.bla_gate else 1e30)
            if prm.use_bla and gb.levels_a > 0:
                _setc(prog, "uUseBla", 1)
        else:
            _setc(prog, "uUseBla", 0)
            _setc(prog, "uLevels", 0)
            _setc(prog, "uLevelsA", 0)
        if self.program is not None and hasattr(self, "extra_uniforms"):
            for k, v in self.extra_uniforms.items():
                _setc(prog, k, v)
        return prog

    def _dispatch(self, prog, mode: int, start: int, count: int):
        self.pend[self.pend_in].bind_to_storage_buffer(7)
        self.pend[1 - self.pend_in].bind_to_storage_buffer(8)
        _setc(prog, "uBudget", int(self.base_budget if mode == 0 else self.budget))
        _setc(prog, "uMode", int(mode))
        _setc(prog, "uStart", int(start))
        _setc(prog, "uCount", int(count))
        prog.run((count + 63) // 64)
        # Dispatches of one batch touch disjoint samples (and append to the pending list
        # with an atomic counter), so they need no barrier between them: the GPU can start
        # the next one while the slowest warps of the previous one are still running.
        if self.engine.opt.barrier_each_dispatch:
            self.engine.ctx.memory_barrier()
        self.dispatches += 1

    def _read_counters(self):
        """Synchronise: make the shader writes visible, then read the 48-byte counter block
        (this waits for the queued dispatches; no separate glFinish needed)."""
        self.syncs += 1
        ctx = self.engine.ctx
        if self.engine.opt.finish_before_read:
            ctx.finish()
        ctx.memory_barrier(SYNC_BARRIER)
        if self.engine.opt.staged_counters:
            ctx.copy_buffer(self.cnt_down, self.counters)
            raw = self.cnt_down.read()
        else:
            raw = self.counters.read()
        return np.frombuffer(raw, dtype=np.uint32).copy()

    def _advance_segment(self):
        self.seg_idx += 1
        self.order_pos, self.seg_end = self.segments[self.seg_idx]
        self.pend_checked = False

    def step(self, time_budget: float = 0.012, dispatch_target: float = 0.004,
             max_budget: int = 1 << 22) -> bool:
        """Run compute work for about ``time_budget`` seconds; True when complete."""
        if self.complete:
            return True
        if self.params is None or (self.gref is None and self.program is None):
            return False
        prog = self._prepare()
        if self.engine.opt.sync_before_step:
            # Wait for the work queued before (the previous frame's display, panels, video
            # capture ...): otherwise the first batch's time includes it, the dispatch-time
            # estimate is inflated, dispatches shrink to the minimum and the stop-before-
            # overrunning rule allows one per frame - the v3.8 study's pictures took 4-8x longer.
            self.engine.ctx.finish()
        t0 = time.perf_counter()
        bt0 = t0
        batch = []                       # (mode, count) of queued dispatches
        last_seg = len(self.segments) - 1

        def flush():
            """Wait for the queued batch (by reading the counters); adapt the dispatch size."""
            nonlocal bt0, batch
            c = self._read_counters()
            if not batch:
                return c
            now = time.perf_counter()
            dt = now - bt0
            self.gpu_time += dt
            avg = dt / len(batch)
            mode, count = batch[-1]
            if avg < dispatch_target * 0.5:
                if mode == 1:
                    self.budget = min(self.budget * 2, max_budget)      # stragglers: go deeper
                if count >= self.chunk:
                    self.chunk = min(self.chunk * 2, MAX_GROUPS * 64)
            elif avg > dispatch_target * 1.5:
                f = dispatch_target / avg
                if mode == 1 and self.budget > self.base_budget:
                    self.budget = max(self.base_budget, int(self.budget * f))
                else:
                    self.chunk = max(self.engine.opt.chunk_min, int(self.chunk * f))
            # aim for ~2 synchronisations per time slice
            self.batch_k = int(min(16, max(1, (time_budget * 0.5) / max(avg, 1e-5))))
            self.disp_est = max(avg, 0.5 * (avg + self.disp_est)) if self.disp_est else avg
            bt0 = now
            batch = []
            return c

        c = None
        fresh = False                    # c holds the counters and nothing was queued since
        issued = 0
        predictive = self.engine.opt.predictive_budget
        while True:
            if predictive and issued and (bt0 - t0) + (len(batch) + 1) * (self.disp_est or dispatch_target) > time_budget:
                # the next dispatch would end after the time slice (the queued ones run since bt0):
                # stop now instead of overshooting the frame by up to a whole dispatch
                break
            if self.order_pos < self.seg_end:
                count = min(self.chunk, self.seg_end - self.order_pos, MAX_GROUPS * 64)
                self._dispatch(prog, 0, self.order_pos, count)
                self.order_pos += count
                batch.append((0, count))
                issued += 1
            elif self.pend_checked and self.pend_in_pos < self.pend_in_count:
                count = min(self.chunk, self.pend_in_count - self.pend_in_pos, MAX_GROUPS * 64)
                self._dispatch(prog, 1, self.pend_in_pos, count)
                self.pend_in_pos += count
                batch.append((1, count))
                issued += 1
            else:
                if self.seg_idx < last_seg and not self.engine.opt.sync_per_level:
                    # next progressive level right away: the GPU runs the dispatches in order,
                    # so coarse levels still come first; stragglers of every level are collected
                    # in the pending list and handled after the sweep (no wait per level)
                    self._advance_segment()
                    continue
                # need to know how many samples are still pending -> synchronise
                c = flush()
                fresh = True
                pend_out = int(c[0])
                remaining = (self.pend_in_count - self.pend_in_pos) + pend_out
                seg_size = self.segments[self.seg_idx][1] - self.segments[self.seg_idx][0]
                if self.seg_idx < last_seg and remaining <= max(256, seg_size // 32):
                    self._advance_segment()          # stragglers continue alongside the next level
                    continue
                if self.pend_in_pos < self.pend_in_count:
                    self.pend_checked = True
                    continue
                if pend_out > 0:
                    self.pend_in = 1 - self.pend_in
                    self.pend_in_count = pend_out
                    self.pend_in_pos = 0
                    self.pend_checked = True
                    c[0] = 0
                    self._put_counters(c.tobytes())
                    continue
                if self.seg_idx < last_seg:
                    self._advance_segment()
                    continue
                self.complete = True
                self.finished_at = time.perf_counter()
                break
            fresh = False
            if len(batch) >= self.batch_k:
                c = flush()
                fresh = True
                if time.perf_counter() - t0 > time_budget:
                    break
        if not fresh or c is None:
            c = flush()
        self.done_count = int(c[1])
        self.min_iter = None if c[2] == 0xFFFFFFFF else int(c[2])
        self.max_iter_seen = int(c[3]) if c[3] else None
        self.steps += int(c[4]) + (int(c[5]) << 32)
        self.bla_skips += int(c[6])
        self.bla_fails += int(c[10])
        self.rebases += int(c[7])
        self.maxiter_count = int(c[8])
        self.interior_count = int(c[9])
        self.flagged = int(c[11])       # adaptive supersampling: pixels that were refined
        c[4:8] = 0
        c[10] = 0
        self._put_counters(c.tobytes())
        return self.complete

    def run_to_completion(self, time_slice: float = 0.5, dispatch_target: float = 0.15,
                          timeout: float | None = None) -> bool:
        t0 = time.perf_counter()
        while not self.step(time_slice, dispatch_target):
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return False
        return True

    def read_results(self) -> np.ndarray:
        """(H, W, 8) uint32 array: n, frac, de, slope angle, arg z, stripe, -, - (floats as bits)."""
        return np.frombuffer(self.result.read(), dtype=np.uint32).reshape(self.H, self.W, RESULT_WORDS)


# ---------------------------------------------------------------------------
# colouring / display
# ---------------------------------------------------------------------------

MAPPINGS = ["linear", "log", "sqrt", "auto"]
METHODS = ["iterations", "distance"]
PATTERNS = ["none", "decomposition", "field lines", "grid"]


@dataclass
class ColorSettings:
    palette: str = "Gilded"
    method: int = 0                # 0 smooth iterations, 1 distance estimate
    density: float = 0.03
    offset: float = 0.0
    mapping: int = 3               # 0 linear, 1 log, 2 sqrt, 3 auto-contrast (histogram)
    auto_cycles: float = 5.0       # palette repetitions across the image (auto mapping)
    de_density: float = 0.12       # palette cycles per octave of distance (distance method)
    interior: tuple = (0.01, 0.01, 0.03)
    de_strength: float = 0.55      # dark edges along filaments
    de_scale: float = 1.0
    slope_strength: float = 0.35   # 3D relief lighting
    light_angle: float = 135.0     # degrees
    light_height: float = 1.6
    glow: float = 0.0
    glow_color: tuple = (1.0, 1.0, 1.0)
    stripes: float = 0.0           # stripe-average texture amount
    stripe_freq: float = 5.0
    pattern: int = 0               # 0 none, 1 binary decomposition, 2 field lines, 3 grid
    pattern_strength: float = 0.5
    pattern_freq: float = 2.0
    cycle_speed: float = 0.0       # palette cycles per second
    relative_iter: bool = False
    look: str = "Gilded"

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict) -> "ColorSettings":
        cs = ColorSettings()
        for k, v in d.items():
            if hasattr(cs, k):
                setattr(cs, k, tuple(v) if isinstance(getattr(cs, k), tuple) else v)
        return cs

    @staticmethod
    def from_look(name: str, base: "ColorSettings | None" = None) -> "ColorSettings":
        from .palettes import LOOKS
        defaults = ColorSettings()
        cs = ColorSettings() if base is None else replace(base)
        look = LOOKS.get(name, {})
        # parameters a look does not mention go back to their defaults (looks are complete styles)
        for k in ("method", "stripes", "pattern", "pattern_strength", "glow", "glow_color", "de_density",
                  "stripe_freq", "pattern_freq"):
            setattr(cs, k, getattr(defaults, k))
        for k, v in look.items():
            setattr(cs, k, tuple(v) if isinstance(v, list) else v)
        cs.look = name
        return cs

    @property
    def needs_de(self) -> bool:
        return (self.de_strength > 0 or self.slope_strength > 0 or self.glow > 0 or self.method == 1)

    @property
    def needs_stripes(self) -> bool:
        return self.stripes > 0


class AutoContrast:
    """Histogram equalisation of log2(iterations) for the 'auto' colour mapping."""

    BINS = 1024

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.compute_shader(HIST_CS)
        self.hist = ctx.buffer(reserve=self.BINS * 4)
        self.hist_zero = ctx.buffer(bytes(self.BINS * 4))   # cleared by a GPU copy, read through
        self.hist_down = ctx.buffer(reserve=self.BINS * 4, dynamic=True)  # a copy (never maps 'hist')
        self.tex = ctx.texture((self.BINS, 1), 1, dtype="f4")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        self.lo, self.hi = 0.0, 1.0
        self.cdf = np.linspace(0, 1, self.BINS, dtype=np.float64)
        self.valid = False
        self.tex.write(self.cdf.astype(np.float32).tobytes())

    def reset(self):
        self.valid = False

    def release(self):
        for o in (self.hist, self.hist_zero, self.hist_down, self.tex):
            o.release()

    def _cdf(self, h: np.ndarray, a0: float, a1: float, lo: float, hi: float) -> np.ndarray:
        """Colour mapping from a histogram of log2(iterations) over [a0, a1]: the fraction of
        samples below lo + k (hi - lo) / 1023 for texel k (exactly where the shader looks it
        up; v3.3 stored the count up to the *end* of each bin, one bin too high - visible when
        most of the picture lies within a few iterations), with a little linear-in-log mixed in."""
        x = np.linspace(a0, a1, self.BINS + 1)[1:]                 # upper bin edges
        cum = np.concatenate([[0.0], np.cumsum(h)])
        xs = np.linspace(lo, hi, self.BINS)
        cdf_raw = np.interp(xs, np.concatenate([[a0], x]), cum)     # counts below each texel position
        dens = np.diff(np.concatenate([[cdf_raw[0]], cdf_raw]))
        dens = np.maximum(dens, 0) + 0.02 * dens.sum() / self.BINS
        cdf = np.cumsum(dens)
        return (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-12)

    @staticmethod
    def _blend(old: np.ndarray, new: np.ndarray, smooth: float) -> np.ndarray:
        """Temporal smoothing of the colour mapping (videos).  Both are functions of the
        absolute log2(iterations), so the blend is too; it is *not* stretched to start at 0
        again (v3.3 did: whenever the smallest iteration count of the frame went up by one -
        every few frames of a zoom - all colours jumped)."""
        cdf = smooth * old + (1 - smooth) * new
        return np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)

    def _clear_hist(self):
        self.ctx.copy_buffer(self.hist, self.hist_zero)

    def _read_hist(self) -> np.ndarray:
        self.ctx.memory_barrier(SYNC_BARRIER)
        self.ctx.copy_buffer(self.hist_down, self.hist)
        return np.frombuffer(self.hist_down.read(), dtype=np.uint32).astype(np.float64)

    @staticmethod
    def _groups(n: int, stride: int) -> int:
        """Work groups for the histogram: each thread loops, at most 128 groups."""
        return max(1, min(128, ((n + stride - 1) // stride + 255) // 256))

    def update(self, field: "Field", smooth: float = 0.0, max_samples: int = 400_000) -> bool:
        """Recompute from the escaped samples of ``field``; smooth in [0,1) blends with the old CDF."""
        if field.min_iter is None or field.max_iter_seen is None:
            return False
        lo = math.log2(max(field.min_iter, 1))
        hi = math.log2(field.max_iter_seen + 1.0)
        if hi - lo < 1e-3:
            hi = lo + 1e-3
        stride = max(1, field.N // max_samples)
        self._clear_hist()
        field.result.bind_to_storage_buffer(1)
        self.hist.bind_to_storage_buffer(2)
        _setc(self.prog, "uN", int(field.N))
        _setc(self.prog, "uStride", int(stride))
        _setc(self.prog, "uRange", (float(lo), float(hi)))
        _setc(self.prog, "uW", int(field.W))
        _setc(self.prog, "uRect", (0, 0, int(field.W), int(field.H)))
        _setc(self.prog, "uXf", 0)
        self.prog.run(self._groups(field.N, stride))
        self.ctx.memory_barrier()
        h = self._read_hist()
        if h.sum() <= 0:
            return False
        cdf = self._cdf(h, lo, hi, lo, hi)
        if smooth > 0 and self.valid:
            x_new = np.linspace(lo, hi, self.BINS)
            x_old = np.linspace(self.lo, self.hi, self.BINS)
            old = np.interp(x_new, x_old, self.cdf, left=0.0, right=1.0)
            cdf = self._blend(old, cdf, smooth)
        self.lo, self.hi, self.cdf = lo, hi, cdf
        self.tex.write(cdf.astype(np.float32).tobytes())
        self.valid = True
        return True

    # ---- histogram over several fields (the tiles of a video frame), for the next frame
    def begin_accumulate(self, lo: float, hi: float):
        self.acc_range = (float(lo), float(max(hi, lo + 1e-3)))
        self._clear_hist()
        self.acc_count = 0

    def accumulate(self, field: "Field", max_samples: int = 2_000_000, rect=None, frame=None,
                   target=None, acc_range=None):
        """Add the escaped samples of ``field`` (only those in ``rect`` = (x0, y0, x1, y1), and
        with ``frame`` = (InvA, Off, (W/2, H/2)) only those with |InvA (pos + Off)| inside the half
        sizes: a rotated output frame inside a larger picture).  ``target`` = (buffer, byte
        offset): count into that histogram instead (many frames in one batch, read once)."""
        if field.min_iter is None or field.max_iter_seen is None:
            return
        if rect is None:
            rect = (0, 0, field.W, field.H)
        x0, y0, x1, y1 = (int(v) for v in rect)
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if area == 0:
            return
        stride = max(1, area // max_samples)
        field.result.bind_to_storage_buffer(1)
        if target is None:
            self.hist.bind_to_storage_buffer(2)
        else:
            target[0].bind_to_storage_buffer(2, int(target[1]), self.BINS * 4)
        _setc(self.prog, "uN", int(field.N))
        _setc(self.prog, "uStride", int(stride))
        _setc(self.prog, "uRange", tuple(float(v) for v in (acc_range or self.acc_range)))
        _setc(self.prog, "uW", int(field.W))
        _setc(self.prog, "uRect", (x0, y0, x1, y1))
        _setc(self.prog, "uXf", 1 if frame is not None else 0)
        if frame is not None:
            ia, off, half = frame
            _setc(self.prog, "uXfInvA", (float(ia.real), float(ia.imag)))
            _setc(self.prog, "uXfOff", (float(off.real), float(off.imag)))
            _setc(self.prog, "uXfHalf", (float(half[0]), float(half[1])))
        self.prog.run(self._groups(field.N, stride))
        if target is None:
            self.ctx.memory_barrier()
            self.acc_count += 1

    def end_accumulate(self, lo: float | None, hi: float | None, smooth: float = 0.0) -> bool:
        """Finish: the CDF over [lo, hi] (the actual iteration range of the frame; None: the
        range of the counted samples, to bin precision)."""
        if not getattr(self, "acc_count", 0):
            return False
        a0, a1 = self.acc_range
        return self.finish_hist(self._read_hist(), a0, a1, lo, hi, smooth)

    def finish_hist(self, h: np.ndarray, a0: float, a1: float, lo: float | None, hi: float | None,
                    smooth: float = 0.0, upload: bool = True) -> bool:
        """The colour mapping from histogram ``h`` (counted over [a0, a1]); see end_accumulate.
        ``upload=False`` only updates the state (lo, hi, cdf), not the texture."""
        if lo is None or hi is None:
            nz = np.nonzero(h)[0]
            if len(nz) == 0:
                return False
            bw = (a1 - a0) / self.BINS
            lo, hi = a0 + nz[0] * bw, a0 + (nz[-1] + 1) * bw
        x = np.linspace(a0, a1, self.BINS + 1)[1:]                 # upper bin edges
        keep = (x > lo) & (x <= hi + (a1 - a0) / self.BINS)
        if h[keep].sum() <= 0:
            return False
        xs = np.linspace(lo, hi, self.BINS)
        cdf = self._cdf(h, a0, a1, lo, hi)
        if smooth > 0 and self.valid:
            old = np.interp(xs, np.linspace(self.lo, self.hi, self.BINS), self.cdf, left=0.0, right=1.0)
            cdf = self._blend(old, cdf, smooth)
        self.lo, self.hi, self.cdf = lo, hi, cdf
        if upload:
            self.tex.write(cdf.astype(np.float32).tobytes())
        self.valid = True
        return True

    def state(self):
        return (self.lo, self.hi, self.cdf.copy(), self.valid)

    def set_state(self, st):
        self.lo, self.hi, cdf, self.valid = st
        self.cdf = cdf
        self.tex.write(cdf.astype(np.float32).tobytes())


class Display:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=FULLSCREEN_VS,
                                fragment_shader="#version 430\n" + _src("display.frag"))
        self.vao = ctx.vertex_array(self.prog, [])
        self.blit_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=BLIT_FS)
        self.blit_vao = ctx.vertex_array(self.blit_prog, [])
        self.palette_tex = ctx.texture((1024, 1), 3, dtype="f4")
        self.palette_tex.repeat_x = True
        self.palette_tex.repeat_y = False
        self.palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._palette_key = None
        self.dummy_tex = ctx.texture((1, 1), 4, data=bytes(4))
        self.consistent = True          # no dot grid while a field refines (v3.7; False: v3.6)
        self.flat_cdf = ctx.texture((2, 1), 1, dtype="f4", data=np.array([0.0, 1.0], np.float32).tobytes())
        self.flat_cdf.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def set_palette(self, name: str):
        from .palettes import palette_version
        key = (name, palette_version(name))
        if key != self._palette_key:
            self.palette_tex.write(make_palette(name, 1024).tobytes())
            self._palette_key = key

    def render(self, field: Field, fbo: moderngl.Framebuffer, viewport: tuple, scale: float,
               cs: ColorSettings, *, sample_origin=(0, 0), out_half=None, out_offset=(0.0, 0.0),
               prev_tex=None, prev_xform=None, prev_size=(1, 1), prev_q: float = 0.0,
               field_q: float | None = None, time_offset: float = 0.0,
               iter_base: int = 0, max_iter_show: int = 0xFFFFFFFF, fallback: bool = True,
               background=(0.0, 0.0, 0.0), auto: "AutoContrast | None" = None, box: bool = True,
               blend_add: bool = False, field_xform=None):
        """Colour ``field`` into ``fbo``.  ``field_xform`` = (A, (dx, dy)) maps screen pixels to
        the field's own view when the field was computed for a different view (reprojection);
        ``prev_xform`` does the same for ``prev_tex``."""
        ctx = self.ctx
        self.set_palette(cs.palette)
        p = self.prog
        x, y, w, h = viewport
        fbo.use()
        fbo.viewport = viewport
        field.result.bind_to_storage_buffer(1)
        self.palette_tex.use(0)
        _setc(p, "uPalette", 0)
        (prev_tex if prev_tex is not None else self.dummy_tex).use(1)
        _setc(p, "uPrev", 1)
        (auto.tex if auto is not None else self.flat_cdf).use(2)
        _setc(p, "uAutoTex", 2)
        have_prev = prev_tex is not None and prev_xform is not None
        _setc(p, "uHavePrev", 1 if have_prev else 0)
        if have_prev:
            a, dlt = prev_xform
            _setc(p, "uPrevA", (float(a.real), float(a.imag)))
            _setc(p, "uPrevD", (float(dlt[0]), float(dlt[1])))
        _setc(p, "uPrevQ", float(prev_q) if have_prev else -1.0)
        if field_xform is not None:
            fa, fd = field_xform
            _setc(p, "uFieldXf", 1)
            _setc(p, "uFieldA", (float(fa.real), float(fa.imag)))
            _setc(p, "uFieldD", (float(fd[0]), float(fd[1])))
        else:
            _setc(p, "uFieldXf", 0)
        if field_q is None:
            field_q = min(1.0, float(scale) ** 2)
        _setc(p, "uFieldQ", float(field_q))
        _setc(p, "uPrevSize", (float(prev_size[0]), float(prev_size[1])))
        if out_half is None:
            out_half = (w / 2.0, h / 2.0)
        _setc(p, "uOutHalf", tuple(float(v) for v in out_half))
        _setc(p, "uOutOffset", tuple(float(v) for v in out_offset))
        _setc(p, "uField", (field.W, field.H))
        _setc(p, "uVP", (float(x), float(y), float(w), float(h)))
        _setc(p, "uScale", float(scale) if box else min(float(scale), 0.999999))
        _setc(p, "uSampleOrigin", tuple(int(v) for v in sample_origin))
        _setc(p, "uBackground", tuple(float(v) for v in background))
        _setc(p, "uInterior", tuple(float(v) for v in cs.interior))
        _setc(p, "uMethod", int(cs.method))
        _setc(p, "uDensity", float(cs.density))
        _setc(p, "uDEDensity", float(cs.de_density))
        _setc(p, "uOffset", float(cs.offset + time_offset))
        mapping = int(cs.mapping)
        if mapping == 3 and (auto is None or not auto.valid):
            mapping = 1
        _setc(p, "uMapping", mapping)
        if auto is not None:
            _setc(p, "uAutoRange", (float(auto.lo), float(auto.hi)))
        _setc(p, "uAutoCycles", float(cs.auto_cycles))
        _setc(p, "uIterBase", int(iter_base))
        _setc(p, "uMaxIterShow", int(max_iter_show))
        _setc(p, "uDEStrength", float(cs.de_strength))
        _setc(p, "uDEScale", float(cs.de_scale))
        _setc(p, "uSlopeStrength", float(cs.slope_strength))
        _setc(p, "uGlow", float(cs.glow))
        _setc(p, "uGlowColor", tuple(float(v) for v in cs.glow_color))
        _setc(p, "uStripes", float(cs.stripes))
        _setc(p, "uPattern", int(cs.pattern))
        _setc(p, "uPatternStrength", float(cs.pattern_strength))
        _setc(p, "uPatternFreq", float(cs.pattern_freq))
        la = math.radians(cs.light_angle)
        _setc(p, "uLightDir", (math.cos(la), math.sin(la)))
        _setc(p, "uLightHeight", float(cs.light_height))
        _setc(p, "uShowFallback", 1 if fallback else 0)
        _setc(p, "uConsistent", 1 if self.consistent else 0)
        _setc(p, "uFieldDone", 1 if getattr(field, "complete", True) else 0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        if blend_add:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
        else:
            ctx.disable(moderngl.BLEND)
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        if blend_add:
            ctx.disable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def blit(self, tex: moderngl.Texture, fbo, viewport, scale: float = 1.0, src_offset=(0, 0)):
        fbo.use()
        fbo.viewport = viewport
        tex.use(0)
        _setc(self.blit_prog, "uTex", 0)
        _setc(self.blit_prog, "uScale", float(scale))
        _setc(self.blit_prog, "uSrcOffset", (int(src_offset[0]) - int(viewport[0]),
                                             int(src_offset[1]) - int(viewport[1])))
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.blit_vao.render(moderngl.TRIANGLES, vertices=3)
