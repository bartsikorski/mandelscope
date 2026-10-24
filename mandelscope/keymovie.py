"""Fast zoom movies: one large picture serves many frames.

A zoom movie at 60 frames/s spends 60 frames on every octave of zoom, and every frame is
computed from scratch.  Here the frames are cut out of *key pictures* instead: one picture
covers a stretch of the camera path (about one octave of a zoom, or a stretch of a pan or a
rotation) with enough samples for the deepest frame of that stretch, so every frame still
gets at least ``q`` samples per pixel and axis (q = the export's supersampling).  Each frame
is coloured from the key picture's iteration data (palette cycling, colour keyframes and the
'auto' histogram stay per frame) and filtered down to the frame with a box filter over the
pixel.  At the switch from one key picture to the next the frames cross-fade over a few
frames, so the fine noise of very detailed regions does not jump.

Key pictures are planned greedily along the frames: a window of frames grows while the cost
per frame (samples of the key picture / frames it serves) keeps falling.  For a steady zoom
that is just under one octave; for a pure pan it is several screen widths.  Where a key
picture would not be cheaper than computing the frames (very fast camera moves), those frames
are computed directly as before.
"""
from __future__ import annotations

import cmath
import math
import time
from dataclasses import dataclass, replace

from types import SimpleNamespace

import numpy as np
import moderngl

from .mp import View
from .renderer import FULLSCREEN_VS, SYNC_BARRIER, ColorSettings, _setc

PAD = 2                         # pixels of margin around the frames inside a key picture
BLEND_FRAMES = 6                # cross-fade between consecutive key pictures
MAX_SAMPLES = 36_000_000        # per key picture (1080p with q = 2: 7680 x 4320 = 33 M)
WINDOW_SLACK = 1.05             # keep growing a window while within 5 % of the best cost/frame
DIRECT_RATIO = 0.75             # use a key picture only if it costs < 75 % of direct frames


# ---------------------------------------------------------------------------- planning

@dataclass
class KeyPlan:
    view: View                  # the key picture's camera (its own size below)
    W: int
    H: int
    f0: int                     # frames f0 .. f1-1 (indices into the frame list) use it
    f1: int
    direct: bool = False        # frames computed directly instead

    @property
    def samples(self) -> int:
        return self.W * self.H


class _Local:
    """Frames expressed in the screen of one anchor frame (pixels, y down) as floats."""

    def __init__(self, views, i0, W, H):
        self.v0 = views[i0]
        self.views = views
        self.W, self.H = W, H
        self.corners = []                   # 4 complex corners per frame (anchor screen)
        self.scale = []                     # pixel size / anchor pixel size
        self.angle = []                     # angle - anchor angle

    def add(self, i):
        v = self.views[i]
        dx, dy = self.v0.delta_in_pixels(v, self.H)
        s = 2.0 ** (v.log2r - self.v0.log2r)
        a = v.angle - self.v0.angle
        r = s * cmath.exp(-1j * a)          # frame screen -> anchor screen: d0 = c + s e^{i(th0 - th)} d
        c = complex(dx, dy)
        W2, H2 = self.W / 2.0, self.H / 2.0
        self.corners.append([c + r * complex(x, y) for x, y in ((-W2, -H2), (W2, -H2), (W2, H2), (-W2, H2))])
        self.scale.append(s)
        self.angle.append(a)

    def geometry(self, q: float):
        """Key picture for all frames added so far: (samples, W_K, H_K, centre (anchor px),
        angle relative to the anchor, key pixel size in anchor pixels)."""
        ang = self.angle[(len(self.angle) - 1) // 2]     # oriented like the middle frame
        rot = cmath.exp(1j * ang)                        # anchor screen -> key orientation
        pts = np.asarray(self.corners, dtype=np.complex128) * rot
        h_rel = min(self.scale) / q
        x0, x1 = float(pts.real.min()), float(pts.real.max())
        y0, y1 = float(pts.imag.min()), float(pts.imag.max())
        Wk = int(math.ceil((x1 - x0) / h_rel)) + 2 * PAD
        Hk = int(math.ceil((y1 - y0) / h_rel)) + 2 * PAD
        Wk += (-Wk) % 16                                 # sizes that repeat between windows
        Hk += (-Hk) % 16
        centre = complex((x0 + x1) / 2, (y0 + y1) / 2) / rot
        return Wk * Hk, Wk, Hk, centre, ang, h_rel


def plan_key_pictures(views: list[View], W: int, H: int, q: float, *, direct_samples: int,
                      max_samples: int = MAX_SAMPLES, max_side: int = 16384,
                      max_window: int = 1200) -> list[KeyPlan]:
    """Split the frames into windows served by one key picture each (or computed directly)."""
    plans: list[KeyPlan] = []
    n = len(views)
    i = 0
    while i < n:
        loc = _Local(views, i, W, H)
        best = None                         # (cost per frame, last frame, geometry)
        i1 = i
        while i1 < n and i1 - i < max_window:
            loc.add(i1)
            g = loc.geometry(q)
            cost, Wk, Hk = g[0], g[1], g[2]
            if cost > max_samples or Wk > max_side or Hk > max_side:
                break
            per = cost / (i1 - i + 1)
            if best is None or per < best[0]:
                best = (per, i1, g)
            elif per > WINDOW_SLACK * best[0]:
                break
            else:
                best = (best[0], i1, g)      # nearly as good and longer: fewer switches
            i1 += 1
        if best is not None and best[1] < n - 1 and n - 1 - best[1] <= max(2, (best[1] - i + 1) // 4):
            # a short rest of the movie would get a picture of its own (or be computed
            # directly): take it into this window if that still fits
            for j in range(i + len(loc.scale), n):
                loc.add(j)
            g = loc.geometry(q)
            if g[0] <= max_samples and g[1] <= max_side and g[2] <= max_side:
                best = (g[0] / (n - i), n - 1, g)
        if best is None or best[0] > DIRECT_RATIO * direct_samples:
            # a key picture would not pay off here: compute this frame directly
            if plans and plans[-1].direct and plans[-1].f1 == i:
                plans[-1].f1 = i + 1
            else:
                plans.append(KeyPlan(views[i], W, H, i, i + 1, direct=True))
            i += 1
            continue
        _, i1, (cost, Wk, Hk, centre, ang, h_rel) = best
        v0 = views[i]
        kv = v0.offset(centre.real, centre.imag, H)
        kv.angle = v0.angle + ang
        kv.log2r = v0.log2r + math.log2(h_rel) + math.log2(Hk) - math.log2(H)
        kv.max_iter = max(v.max_iter for v in views[i:i1 + 1])
        kv.ensure_precision(Hk)
        plans.append(KeyPlan(kv, Wk, Hk, i, i1 + 1))
        i = i1 + 1
    return plans


def frame_xform(key: View, Hk: int, frame: View, H: int):
    """Frame pixels (y down, from the frame centre) -> key picture pixels (from its centre):
    q = D + A d.  Returns (A, D); |A| = key samples per frame pixel and axis."""
    A = 2.0 ** (frame.pixel_log2(H) - key.pixel_log2(Hk)) * cmath.exp(1j * (key.angle - frame.angle))
    dx, dy = key.delta_in_pixels(frame, Hk)
    return A, complex(dx, dy)


def footprint(A: complex, D: complex, W: int, H: int, Wk: int, Hk: int):
    """Bounding box (x0, y0, x1, y1) of the frame in key pixels (y down, from the top left)
    and whether the frame lies inside the key picture."""
    pts = [D + A * complex(x, y) for x, y in ((-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2))]
    xs = [p.real + Wk / 2 for p in pts]
    ys = [p.imag + Hk / 2 for p in pts]
    inside = min(xs) >= 0.5 and min(ys) >= 0.5 and max(xs) <= Wk - 0.5 and max(ys) <= Hk - 0.5
    x0 = max(0, int(math.floor(min(xs))) - 2)
    y0 = max(0, int(math.floor(min(ys))) - 2)
    x1 = min(Wk, int(math.ceil(max(xs))) + 2)
    y1 = min(Hk, int(math.ceil(max(ys))) + 2)
    return (x0, y0, x1, y1), inside


# ---------------------------------------------------------------------------- key pictures

class KeptTile:
    """The finished iteration data of one tile of a key picture (the result buffer only)."""

    def __init__(self, ctx, fld, row0: int):
        self.result = ctx.buffer(reserve=fld.result.size)
        ctx.copy_buffer(self.result, fld.result)
        self.W, self.H, self.N = fld.W, fld.H, fld.N
        self.row0 = row0
        self.min_iter = fld.min_iter
        self.max_iter_seen = fld.max_iter_seen

    def release(self):
        self.result.release()


class KeyPicture:
    def __init__(self, ctx, plan: KeyPlan):
        self.ctx = ctx
        self.plan = plan
        self.view = plan.view
        self.W, self.H = plan.W, plan.H
        self.tiles: list[KeptTile] = []
        self.tex = None
        self.fbo = None

    def ensure_texture(self):
        if self.tex is None:
            self.tex = self.ctx.texture((self.W, self.H), 4, dtype="f2")
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex.repeat_x = False
            self.tex.repeat_y = False
            self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])

    def iter_range(self):
        lo = [t.min_iter for t in self.tiles if t.min_iter is not None]
        hi = [t.max_iter_seen for t in self.tiles if t.max_iter_seen is not None]
        return (min(lo) if lo else None), (max(hi) if hi else None)

    def colour(self, display, cs: ColorSettings, rect, *, time_offset: float, max_iter_show: int, auto):
        """Colour the samples inside ``rect`` (key pixels, y down) into the texture."""
        self.ensure_texture()
        x0, y0, x1, y1 = rect
        for t in self.tiles:
            r0, r1 = max(y0, t.row0), min(y1, t.row0 + t.H)
            if r1 <= r0 or x1 <= x0:
                continue
            display.render(t, self.fbo, (x0, self.H - r1, x1 - x0, r1 - r0), 1.0, cs,
                           sample_origin=(x0, r0 - t.row0), time_offset=time_offset, iter_base=0,
                           max_iter_show=max_iter_show, fallback=False, auto=auto)

    def histogram(self, auto, rect, A: complex, D: complex, W: int, H: int, target=None, acc_range=None):
        """Add the samples that the frame (A, D, W x H) shows to ``auto``'s histogram (or to the
        histogram at ``target`` = (buffer, offset))."""
        x0, y0, x1, y1 = rect
        for t in self.tiles:
            r0, r1 = max(y0, t.row0), min(y1, t.row0 + t.H)
            if r1 > r0:
                off = complex(-self.W / 2.0, t.row0 - self.H / 2.0) - D
                auto.accumulate(t, rect=(x0, r0 - t.row0, x1, r1 - t.row0),
                                frame=(1.0 / A, off, (W / 2.0, H / 2.0)), target=target, acc_range=acc_range)

    def release(self):
        for t in self.tiles:
            t.release()
        self.tiles = []
        if self.fbo is not None:
            self.fbo.release()
            self.tex.release()
            self.fbo = self.tex = None


RESAMPLE_FS = """
#version 430
// A frame cut out of one or two key pictures: box filter over the output pixel, taps fetched
// bilinearly from the coloured key picture(s); the second picture is cross-faded in.
uniform sampler2D uImg0;
uniform sampler2D uImg1;
uniform vec2 uSize0;          // key picture sizes (pixels)
uniform vec2 uSize1;
uniform vec2 uA0;             // frame pixel (from the centre, y down) -> key pixel: D + A d
uniform vec2 uD0;
uniform vec2 uA1;
uniform vec2 uD1;
uniform int uTaps0;
uniform int uTaps1;
uniform float uMix;           // weight of picture 1 (0: picture 0 only)
uniform vec2 uOut;            // frame size
out vec4 fragColor;

vec3 boxSample(sampler2D img, vec2 size, vec2 A, vec2 D, int taps, vec2 d0) {
    vec3 sum = vec3(0.0);
    float inv = 1.0 / float(taps);
    for (int j = 0; j < taps; ++j)
        for (int i = 0; i < taps; ++i) {
            vec2 d = d0 + (vec2(float(i), float(j)) + 0.5) * inv - 0.5;
            vec2 q = D + vec2(A.x * d.x - A.y * d.y, A.y * d.x + A.x * d.y) + 0.5 * size;
            sum += texture(img, vec2(q.x / size.x, 1.0 - q.y / size.y)).rgb;
        }
    return sum * (inv * inv);
}

void main() {
    vec2 p = vec2(gl_FragCoord.x, uOut.y - gl_FragCoord.y);   // pixel centre, y down
    vec2 d0 = p - 0.5 * uOut;
    vec3 c = boxSample(uImg0, uSize0, uA0, uD0, uTaps0, d0);
    if (uMix > 0.0)
        c = mix(c, boxSample(uImg1, uSize1, uA1, uD1, uTaps1, d0), uMix);
    fragColor = vec4(c, 1.0);
}
"""


class Resampler:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=RESAMPLE_FS)
        self.vao = ctx.vertex_array(self.prog, [])

    def draw(self, fbo, size, parts):
        """parts: [(KeyPicture, A, D)] (1 or 2); the second one with weight ``mix``."""
        p = self.prog
        W, H = size
        fbo.use()
        fbo.viewport = (0, 0, W, H)
        _setc(p, "uOut", (float(W), float(H)))
        for k, (pic, A, D, w) in enumerate(parts[:2]):
            pic.tex.use(3 + k)
            _setc(p, f"uImg{k}", 3 + k)
            _setc(p, f"uSize{k}", (float(pic.W), float(pic.H)))
            _setc(p, f"uA{k}", (A.real, A.imag))
            _setc(p, f"uD{k}", (D.real, D.imag))
            _setc(p, f"uTaps{k}", int(min(8, max(1, math.ceil(abs(A) - 1e-6)))))
        if len(parts) < 2:
            parts[0][0].tex.use(4)
            _setc(p, "uImg1", 4)
        _setc(p, "uMix", float(parts[1][3]) if len(parts) > 1 else 0.0)
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self):
        self.vao.release()
        self.prog.release()


# ---------------------------------------------------------------------------- the movie

def render_sequence_fast_iter(engine, seq, path: str, settings, colors: ColorSettings, fps: float = 60.0,
                              codec: str = "auto", quality: int = 18, worker=None, png_dir: str | None = None,
                              frame_range: tuple | None = None, time_slice: float = 0.2,
                              max_samples: int | None = None, stats: dict | None = None, julia=None):
    """Like offline.render_sequence_iter (same yields), with frames cut out of key pictures."""
    from .offline import FrameRenderer, apply_color_overrides, draw_overlay
    from .video import FfmpegWriter, PngSaver
    ctx = engine.ctx
    fr = FrameRenderer(engine, settings)
    fr.auto_smooth = 0.75
    fr.hist_from_frames = True
    if julia is None and seq.meta.get("julia") is not None:
        julia = seq.meta["julia"]
    fr.julia = tuple(julia) if julia is not None else None
    fr.formula = seq.meta.get("formula", "mandelbrot")
    W, H = settings.width, settings.height
    q = max(1, int(settings.ss))
    n = seq.frame_count(fps)
    a, b = (0, n) if frame_range is None else frame_range
    times = [seq.start + i / fps for i in range(a, b)]
    vo = [seq.view_at(t) for t in times]
    views = [v for v, _ in vo]
    css = [apply_color_overrides(colors, o) for _, o in vo]
    max_side = min(16384, int(ctx.info.get("GL_MAX_TEXTURE_SIZE", 16384)))
    plans = plan_key_pictures(views, W, H, q, direct_samples=W * H * q * q,
                              max_samples=max_samples or MAX_SAMPLES, max_side=max_side)
    out_cost = 0.25 * W * H * q * q             # colouring + filtering a frame, in sample units
    total = sum((p.f1 - p.f0) * W * H * q * q if p.direct else p.samples + (p.f1 - p.f0) * out_cost
                for p in plans)
    if stats is not None:
        stats.update(key_pictures=sum(1 for p in plans if not p.direct),
                     direct_frames=sum(p.f1 - p.f0 for p in plans if p.direct),
                     samples=sum(p.samples for p in plans if not p.direct), frames=b - a,
                     direct_samples=(b - a) * W * H * q * q)
    writer = FfmpegWriter(path, W, H, fps, codec, quality) if path else None
    pngs = PngSaver() if png_dir else None
    rs = Resampler(ctx)
    display = engine.display
    t_start = time.perf_counter()
    done = 0.0
    prev = None
    # Frames are pipelined: frame k is coloured, filtered and copied into a pixel buffer while
    # the CPU takes frame k-1 out of the other one (no wait per frame); the 'auto' colour
    # histograms of a whole window are counted in one batch.
    BINS = fr.auto.BINS
    cdf_ring = []
    for _ in range(3):
        t = ctx.texture((BINS, 1), 1, dtype="f4")
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        cdf_ring.append(t)
    pbos = [ctx.buffer(reserve=W * H * 3) for _ in range(2)]
    pending = None                  # (frame index, slot): pixels on their way into pbos[slot]
    nframe = 0
    # bench: where the time goes (GPU times from timer queries, the rest measured on the CPU)
    prof = stats is not None
    if prof:
        q_col = [ctx.query(time=True) for _ in range(2)]
        q_fil = [ctx.query(time=True) for _ in range(2)]
        for k in ("t_keys", "t_hist", "t_colour", "t_filter", "t_issue", "t_read", "t_encode", "t_frames"):
            stats[k] = 0.0

    def status(i, phase, extra=0.0, **kw):
        p = min(1.0, (done + extra) / max(total, 1))
        el = time.perf_counter() - t_start
        return dict(frame=a + i, frames=n, phase=phase, progress=p, eta=el / max(p, 1e-6) - el,
                    codec=writer.codec if writer else None, **kw)

    def drain():
        """Generator: take the pending frame out of its pixel buffer and hand it on."""
        nonlocal pending, done
        if pending is None:
            return
        i, slot = pending
        pending = None
        t_r0 = time.perf_counter()
        data = pbos[slot].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
        if prof:
            stats["t_read"] += time.perf_counter() - t_r0
            stats["t_colour"] += q_col[slot].elapsed * 1e-9
            stats["t_filter"] += q_fil[slot].elapsed * 1e-9
        if settings.overlay:
            img = draw_overlay(img, views[i], settings.overlay)
        emit(i, img)
        done += out_cost
        yield status(i, "frame", image=img)

    def emit(i, img):
        t0 = time.perf_counter()
        if writer is not None:
            writer.write(img)
        if pngs is not None:
            pngs.save(img, f"{png_dir}/frame_{a + i:06d}.png")
        if prof:
            stats["t_encode"] += time.perf_counter() - t0

    try:
        for plan in plans:
            if plan.direct:
                if prev is not None:
                    prev.release()
                    prev = None
                for i in range(plan.f0, plan.f1):
                    t = times[i]
                    gen = fr.render_iter(views[i], css[i], time_offset=colors.cycle_speed * (t - seq.start),
                                         anchor=seq.deepest_in_segment(t), worker=worker, time_slice=time_slice)
                    while True:
                        try:
                            phase, pr = next(gen)
                            yield status(i, phase, pr * W * H * q * q)
                        except StopIteration as e:
                            img = e.value
                            break
                    emit(i, img)
                    done += W * H * q * q
                    yield status(i, "frame", image=img)
                continue
            win = css[plan.f0:plan.f1]
            t_key0 = time.perf_counter()
            gen = fr.key_picture_iter(plan, with_de=any(c.needs_de for c in win),
                                      with_stripes=any(c.needs_stripes for c in win),
                                      stripe_freq=win[0].stripe_freq,
                                      anchor=seq.deepest_in_segment(times[plan.f0]), worker=worker,
                                      time_slice=time_slice)
            while True:
                try:
                    phase, pr = next(gen)
                    yield status(plan.f0, phase, pr * plan.samples)
                except StopIteration as e:
                    pic = e.value
                    break
            if prof:
                stats["t_keys"] += time.perf_counter() - t_key0
            done += plan.samples
            n_win = plan.f1 - plan.f0
            m = min(BLEND_FRAMES, n_win // 3)
            xf = [frame_xform(pic.view, pic.H, views[i], H) for i in range(plan.f0, plan.f1)]
            # 'auto' colours: the histograms of all frames of the window in one batch (one wait),
            # then the per-frame mappings (smoothed over time) on the CPU
            t_h0 = time.perf_counter()
            maps = [None] * n_win
            lo, hi = pic.iter_range()
            auto_k = [k for k in range(n_win) if css[plan.f0 + k].mapping == 3]
            if auto_k and lo is not None and hi is not None:
                a0 = math.log2(max(lo, 1))
                a1 = max(math.log2(hi + 1.0), a0 + 1e-3)
                hb = ctx.buffer(reserve=len(auto_k) * BINS * 4)
                hb.write(bytes(len(auto_k) * BINS * 4))
                for j, k in enumerate(auto_k):
                    A, D = xf[k]
                    rect, _ = footprint(A, D, W, H, pic.W, pic.H)
                    pic.histogram(fr.auto, rect, A, D, W, H, target=(hb, j * BINS * 4), acc_range=(a0, a1))
                ctx.memory_barrier(SYNC_BARRIER)
                hs = np.frombuffer(hb.read(), dtype=np.uint32).reshape(len(auto_k), BINS).astype(np.float64)
                hb.release()
                for j, k in enumerate(auto_k):
                    fr.auto.finish_hist(hs[j], a0, a1, None, None, smooth=fr.auto_smooth, upload=False)
                    if fr.auto.valid:
                        maps[k] = (fr.auto.lo, fr.auto.hi, fr.auto.cdf.astype(np.float32).tobytes())
                if fr.auto.valid:                   # frames computed directly continue from it
                    fr.auto.tex.write(fr.auto.cdf.astype(np.float32).tobytes())
            elif fr.auto.valid:
                maps = [(fr.auto.lo, fr.auto.hi, fr.auto.cdf.astype(np.float32).tobytes())
                        if css[plan.f0 + k].mapping == 3 else None for k in range(n_win)]
            if prof:
                stats["t_hist"] += time.perf_counter() - t_h0
            t_w0 = time.perf_counter()
            for k in range(n_win):
                i = plan.f0 + k
                t_f0 = time.perf_counter()
                t = times[i]
                cs = css[i]
                A, D = xf[k]
                parts = [(pic, A, D, 1.0)]
                if prev is not None and k < m:
                    A1, D1 = frame_xform(prev.view, prev.H, views[i], H)
                    _, inside = footprint(A1, D1, W, H, prev.W, prev.H)
                    if inside and abs(A1) >= 0.6 * q:
                        parts.append((prev, A1, D1, (m - k) / (m + 1.0)))
                auto = None
                if maps[k] is not None:
                    tex = cdf_ring[nframe % 3]
                    tex.write(maps[k][2])
                    auto = SimpleNamespace(tex=tex, lo=maps[k][0], hi=maps[k][1], valid=True)
                slot = nframe % 2
                if prof:
                    q_col[slot].__enter__()
                for pc, Ak, Dk, _ in parts:
                    rect, _ = footprint(Ak, Dk, W, H, pc.W, pc.H)
                    # distance estimates are in sample units: as in a direct export with q x q
                    csk = replace(cs, de_scale=cs.de_scale * q / max(abs(Ak), 1e-9))
                    pc.colour(display, csk, rect, time_offset=colors.cycle_speed * (t - seq.start),
                              max_iter_show=views[i].max_iter * max(1, int(settings.iter_mult)), auto=auto)
                if prof:
                    q_col[slot].__exit__(None, None, None)
                    q_fil[slot].__enter__()
                rs.draw(fr.fbo, (W, H), parts)
                if prof:
                    q_fil[slot].__exit__(None, None, None)
                fr.fbo.read_into(pbos[slot], components=3, alignment=1)      # asynchronous
                if prof:
                    stats["t_issue"] += time.perf_counter() - t_f0
                yield from drain()                  # the previous frame meanwhile
                pending = (i, slot)
                nframe += 1
            yield from drain()
            if prof:
                stats["t_frames"] += time.perf_counter() - t_w0
            if prev is not None:
                prev.release()
            prev = pic
    finally:
        if prev is not None:
            prev.release()
        if pngs is not None:
            pngs.close()
        if writer is not None:
            rc, err = writer.close()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed ({rc}): {err}")
        rs.release()
        for o in (*cdf_ring, *pbos):
            o.release()
        fr.release()
