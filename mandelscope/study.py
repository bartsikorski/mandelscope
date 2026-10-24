"""Motion study: a scripted session that records what the explorer shows while moving.

    run.bat gui --motion-study            (writes footage\\<date-time>\\ in the project folder)

The explorer drives itself (mouse-wheel zooms, drags, rotations, held keys) at a few places,
twice: with the current display compositing ('new') and the one before it ('old').  Every
displayed frame goes into study.mp4 (the fractal only, no panels) and a line of frames.jsonl:

  t, dt             time stamp and frame interval (s)
  scen, act         place / variant and the action running
  view              log2 radius, angle
  xf                screen-to-screen similarity to the previous frame (A, dx, dy): the camera
                    motion, so the analysis can warp the previous frame onto this one
  ms_update, ms_render, ms_frame   CPU time in the compute step, compositing, whole frame
  field             progress, finest complete level, complete, size; kind of target, branch
  src               pixels per source: exact, exact reprojected, older picture, coarse levels
                    1-3, partial blend, nothing (counted on the GPU from the compositing pass)
  aa                anti-aliasing passes shown

tools/analyze_footage.py turns this into numbers (pacing, pops, blur, grid patterns).
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import moderngl

from .renderer import _setc

SOURCES = ["?", "exact", "exact_reprojected", "older_picture", "coarse1", "coarse2", "coarse3",
           "partial", "nothing"]

COUNT_CS = """
#version 430
// Per frame, on the uncompressed picture:
//   cnt[0..8]  pixels per source id (the display pass writes id / 15 into a 2nd render target)
//   cnt[9]     pixels whose source differs from the right or lower neighbour's (dot grids and
//              stripes of mixed sources make this large; clean pictures only have region edges)
//   cnt[10]    16 x sum of |this frame - previous frame moved by the camera motion| (0-255)
//   cnt[11]    pixels where the previous frame (moved) is available
//   cnt[12]    of those, pixels that changed by more than 32/255 (a visible pop)
layout(local_size_x = 256) in;
uniform sampler2D uSrc;
uniform sampler2D uCur;
uniform sampler2D uPrev;
uniform int uHavePrev;
uniform vec2 uA;            // current screen px (from the centre, y down) -> previous: D + A d
uniform vec2 uD;
uniform ivec2 uSize;
layout(std430, binding = 3) buffer Cnt { uint cnt[16]; };
shared uint sh[16];
void main() {
    uint lid = gl_LocalInvocationIndex;
    if (lid < 16u) sh[lid] = 0u;
    barrier();
    uint n = uint(uSize.x * uSize.y);
    uint total = gl_NumWorkGroups.x * 256u;
    vec2 S = vec2(uSize);
    for (uint k = gl_GlobalInvocationID.x; k < n; k += total) {
        ivec2 p = ivec2(int(k % uint(uSize.x)), int(k / uint(uSize.x)));     // texel, row 0 = bottom
        uint id = uint(texelFetch(uSrc, p, 0).r * 15.0 + 0.5);
        atomicAdd(sh[min(id, 8u)], 1u);
        uint idr = uint(texelFetch(uSrc, min(p + ivec2(1, 0), uSize - 1), 0).r * 15.0 + 0.5);
        uint idd = uint(texelFetch(uSrc, max(p - ivec2(0, 1), ivec2(0)), 0).r * 15.0 + 0.5);
        if (id != idr || id != idd) atomicAdd(sh[9], 1u);
        if (uHavePrev != 0) {
            vec2 d = vec2(float(p.x) + 0.5 - 0.5 * S.x, (S.y - float(p.y) - 0.5) - 0.5 * S.y);  // y down
            vec2 q = uD + vec2(uA.x * d.x - uA.y * d.y, uA.y * d.x + uA.x * d.y) + 0.5 * S;
            if (q.x >= 1.0 && q.y >= 1.0 && q.x <= S.x - 1.0 && q.y <= S.y - 1.0) {
                vec3 a = texelFetch(uCur, p, 0).rgb;
                vec3 b = texture(uPrev, vec2(q.x / S.x, 1.0 - q.y / S.y)).rgb;
                float dd = dot(abs(a - b), vec3(255.0 / 3.0));
                atomicAdd(sh[10], uint(dd * 16.0 + 0.5));
                atomicAdd(sh[11], 1u);
                if (dd > 32.0) atomicAdd(sh[12], 1u);
            }
        }
    }
    barrier();
    if (lid < 16u && sh[lid] != 0u) atomicAdd(cnt[lid], sh[lid]);
}
"""


class SourceCounter:
    """Counts the source ids of a frame on the GPU; results arrive one frame later."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.compute_shader(COUNT_CS)
        self.cnt = ctx.buffer(reserve=64)
        self.zero = ctx.buffer(bytes(64))
        self.down = [ctx.buffer(reserve=64, dynamic=True) for _ in range(2)]
        self.k = 0
        self.pending = None

    def count(self, tex, cur=None, xf=None, blit=None) -> list | None:
        """Count `tex` (this frame) and compare `cur` with the previous frame moved by `xf`;
        return the numbers of the previous call."""
        ctx = self.ctx
        ctx.copy_buffer(self.cnt, self.zero)
        p = self.prog
        tex.use(5)
        _setc(p, "uSrc", 5)
        _setc(p, "uSize", tuple(tex.size))
        have = cur is not None and xf is not None and getattr(self, "prev_tex", None) is not None \
            and self.prev_tex.size == cur.size
        _setc(p, "uHavePrev", 1 if have else 0)
        if cur is not None:
            cur.use(6)
            _setc(p, "uCur", 6)
        if have:
            self.prev_tex.use(7)
            _setc(p, "uPrev", 7)
            _setc(p, "uA", (float(xf[0]), float(xf[1])))
            _setc(p, "uD", (float(xf[2]), float(xf[3])))
        self.cnt.bind_to_storage_buffer(3)
        p.run(64)
        if cur is not None:                       # keep this frame for the next comparison
            if getattr(self, "prev_tex", None) is None or self.prev_tex.size != cur.size:
                self.prev_tex = ctx.texture(cur.size, 3)
                self.prev_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.prev_fbo = ctx.framebuffer(color_attachments=[self.prev_tex])
            blit(cur, self.prev_fbo)
        ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT | moderngl.BUFFER_UPDATE_BARRIER_BIT)
        slot = self.down[self.k % 2]
        ctx.copy_buffer(slot, self.cnt)
        prev, self.pending = self.pending, slot
        self.k += 1
        if prev is None:
            return None
        return np.frombuffer(prev.read(), dtype=np.uint32).tolist()

    def flush(self) -> list | None:
        if self.pending is None:
            return None
        out = np.frombuffer(self.pending.read(), dtype=np.uint32).tolist()
        self.pending = None
        return out


class FrameRecorder:
    """Every displayed frame into a video (asynchronous read-back, one frame late)."""

    def __init__(self, path, w, h, fps=60.0, quality=24):
        from .video import FfmpegWriter
        self.w, self.h = w - (w % 2), h - (h % 2)
        self.writer = FfmpegWriter(str(path), self.w, self.h, fps, "auto", quality, max_queue=32)
        self.pbos = []
        self.k = 0
        self.pending = None

    def capture(self, fbo, full_h):
        if not self.pbos:
            self.pbos = [fbo.ctx.buffer(reserve=self.w * self.h * 3) for _ in range(2)]
        pbo = self.pbos[self.k % 2]
        self.k += 1
        fbo.read_into(pbo, viewport=(0, full_h - self.h, self.w, self.h), components=3, alignment=1)
        self._flush()
        self.pending = pbo

    def _flush(self):
        if self.pending is not None:
            img = np.frombuffer(self.pending.read(), dtype=np.uint8).reshape(self.h, self.w, 3)[::-1]
            self.writer.write(img)
            self.pending = None

    def close(self):
        self._flush()
        for b in self.pbos:
            b.release()
        return self.writer.close()


class MotionStudyMixin:
    """Added to the scripted-session class (see make_study)."""

    def study_init(self):
        a = self.app
        self.rows: list[dict] = []
        self.marker = ("", "")
        self.rec = None
        self.counter = None
        self.prev_view = None
        self.t0 = None
        self.last_t = None
        self.reset_n = 0
        self.png_due: dict = {}
        self.png_threads: list = []
        a.study_mrt = True

    # ---------------------------------------------------------------- per frame
    def study_frame(self):
        a = self.app
        if a.screen_fbo is None or a.fb[0] < 16:
            return
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
            W, H = a.fb
            self.rec = FrameRecorder(self.out / "study.mp4", W, H)
            self.counter = SourceCounter(a.ctx)
        dt = now - self.last_t if self.last_t is not None else 0.0
        self.last_t = now
        W, H = a.fb
        v = a.view
        xf = None
        if self.prev_view is not None:
            x = a._xform(v, self.prev_view)                    # this screen -> previous screen
            if x is not None:
                A, (dx, dy) = x
                xf = [A.real, A.imag, dx, dy]
        self.prev_view = v
        f = a.field
        resets = a.reset_log_total - self.reset_n if hasattr(a, "reset_log_total") else 0
        self.reset_n = getattr(a, "reset_log_total", 0)
        dbg = getattr(a, "dbg", {})
        row = dict(i=len(self.rows), t=round(now - self.t0, 5), dt=round(dt, 5), scen=self.marker[0],
                   act=self.marker[1], view=[v.log2r, v.angle], xf=xf,
                   ms_update=round(getattr(a, "ms_update", 0.0), 3), ms_render=round(getattr(a, "ms_render", 0.0), 3),
                   ms_frame=round(a.frame_ms, 3),
                   field=[round(f.progress, 4), f.level_done, bool(f.complete), f.W, f.H] if f is not None else None,
                   kind=dbg.get("kind"), branch=dbg.get("branch"), resets=list(a.reset_log[-resets:]) if resets else [],
                   fq=dbg.get("fq"), snap_q=dbg.get("q"), fxf=dbg.get("fxf"), aa=[a.aa_count, dbg.get("aa_blit", False)],
                   field_T=round(a.field_T, 4), moving=a.is_moving(),
                   t_pre=dbg.get("t_pre"), t_step=dbg.get("t_step"), t_post=dbg.get("t_post"),
                   max_iter=int(v.max_iter), boost=getattr(a, "iter_boost", 1.0),
                   anim_dt=round(getattr(a, "anim_dt", 0.0), 5))
        self.rows.append(row)
        self.rec.capture(a.screen_fbo, H)
        if a.src_tex is not None:
            W_, H_ = a.screen_tex.size
            c = self.counter.count(a.src_tex, a.screen_tex, xf,
                                   lambda src, dst: a.engine.display.blit(src, dst, (0, 0, W_, H_)))
            a.ctx.screen.use()                    # the panels are drawn next, onto the screen
            if c is not None and len(self.rows) >= 2:
                self._put_counts(self.rows[-2], c)
        due = self.png_due.pop(len(self.rows) - 1, None)
        if due:
            # read now, encode in the background (a synchronous PNG save took 130-190 ms and
            # showed up as the biggest hitches of the v3.7 study)
            row["png"] = True
            img = a.screen_rgb()
            path = self.out / f"{due}.png"

            def save():
                from PIL import Image
                Image.fromarray(img).save(path, compress_level=1)
            th = threading.Thread(target=save, daemon=True)
            th.start()
            self.png_threads.append(th)

    @staticmethod
    def _put_counts(row, c):
        row["src"] = c[:9]
        n = max(1, sum(c[:9]))
        row["mixed"] = round(c[9] / n, 5)                         # share of pixels at a source change
        if c[11]:
            row["pop"] = round(c[10] / 16.0 / c[11], 3)           # mean change beyond the motion
            row["pop_px"] = round(c[12] / c[11], 5)               # share of pixels that changed visibly

    def study_finish(self):
        for th in self.png_threads:
            th.join(timeout=10.0)
        if self.counter is not None:
            c = self.counter.flush()
            if c is not None and self.rows:
                self._put_counts(self.rows[-1], c)
        if self.rec is not None:
            rc, err = self.rec.close()
            if rc != 0:
                self.report["errors"].append(("video", err))
        a = self.app
        with open(self.out / "frames.jsonl", "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")
        meta = dict(size=list(a.fb), gpu=a.engine.renderer_name, sources=SOURCES,
                    settings={k: getattr(a.settings, k) for k in ("render_scale", "motion_scale", "aa_passes",
                                                                  "adaptive_ss", "frame_budget_ms", "wheel_step",
                                                                  "zoom_speed", "inertia")},
                    frames=len(self.rows), video="study.mp4")
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1))

    # ---------------------------------------------------------------- the script
    def study_steps(self, quick: bool = False):
        from .app import InputSnapshot
        a = self.app
        W, H = a.fb
        st = self
        LOC = Path(__file__).parent / "locations"

        def act(fn):
            def f():
                fn()
                return True
            return f

        def mark(scen, action, pngs=()):
            def f():
                st.marker = (scen, action)
                n = len(st.rows)
                for off in pngs:
                    st.png_due[n + off] = f"{scen}_{action}_{off:02d}".replace("/", "_")
                return True
            return f

        def wait_s(sec):
            c = {}

            def f():
                c.setdefault("t0", time.perf_counter())
                return time.perf_counter() - c["t0"] >= sec
            return f

        def settle(max_s=2.5, extra=0.4):
            c = {}

            def f():
                now = time.perf_counter()
                c.setdefault("t0", now)
                if "done" not in c and (st.done_rendering() or now - c["t0"] > max_s):
                    c["done"] = now
                return "done" in c and now - c["done"] >= extra
            return f

        def wheel_burst(x, y, notches, interval=0.12):
            c = {"n": 0}

            def f():
                now = time.perf_counter()
                if c["n"] < abs(notches) and now >= c.get("next", 0.0):
                    st.snap = InputSnapshot(mouse=(x, y), wheel=math.copysign(1.0, notches))
                    c["n"] += 1
                    c["next"] = now + interval
                    return False
                if c["n"] >= abs(notches) and now >= c.get("next", 0.0):
                    return True
                st.snap = InputSnapshot(mouse=(x, y))
                return False
            return f

        def drag_for(vx, vy, secs, button=0):
            c = {}

            def f():
                now = time.perf_counter()
                dn = tuple(b == button for b in range(3))
                x0, y0 = W * 0.5, H * 0.5
                if "t0" not in c:
                    c["t0"] = now
                    st.snap = InputSnapshot(mouse=(x0, y0), down=dn, clicked=dn)
                    return False
                u = min(now - c["t0"], secs)
                pos = (x0 + vx * u, y0 + vy * u)
                if now - c["t0"] >= secs:
                    st.snap = InputSnapshot(mouse=pos, released=dn)
                    return True
                st.snap = InputSnapshot(mouse=pos, down=dn)
                return False
            return f

        def rotate_for(deg_per_s, secs, radius=None):
            radius = radius or 0.3 * H
            c = {}

            def f():
                now = time.perf_counter()
                dn = (False, True, False)
                cx, cy = W * 0.5, H * 0.5
                c.setdefault("t0", now)
                ang = math.radians(deg_per_s) * min(now - c["t0"], secs)
                pos = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
                if now == c["t0"]:
                    st.snap = InputSnapshot(mouse=pos, down=dn, clicked=dn)
                    return False
                if now - c["t0"] >= secs:
                    st.snap = InputSnapshot(mouse=pos, released=dn)
                    return True
                st.snap = InputSnapshot(mouse=pos, down=dn)
                return False
            return f

        def hold_for(key, secs, mouse):
            c = {}

            def f():
                now = time.perf_counter()
                first = "t0" not in c
                c.setdefault("t0", now)
                if first:
                    a.key_down_t[key] = now - 1.0                 # treat as held right away
                if now - c["t0"] >= secs:
                    st.snap = InputSnapshot(mouse=mouse)
                    return True
                st.snap = InputSnapshot(mouse=mouse, keys={key}, pressed={key} if first else set())
                return False
            return f

        def tap(key):
            def f():
                st.snap = InputSnapshot(mouse=(W * 0.5, H * 0.5), keys={key}, pressed={key})
                return True
            return f

        def load(name):
            def f():
                a.load_location(LOC / name)
                a.mark_moving(0.01)
                return True
            return f

        def variant(new):
            def f():
                a.engine.display.consistent = new      # no dot grid while refining (v3.7)
                a.settings.wheel_spring = new          # smooth zoom speed (v3.7)
                return True
            return f

        places = [("valley", "seahorse_valley.json", True), ("seahorse_1e-12", "seahorse_1e-12.json", True),
                  ("spiral_1e-100", "spiral_1e-100.json", False)]
        if quick:
            places = places[:1]
        steps = []
        for place, fname, full in places:
            for new in (True, False):
                scen = f"{place}/{'new' if new else 'old'}"
                steps += [
                    (f"{scen} variant", variant(new)),
                    (f"{scen} load", load(fname)),
                    (f"{scen} settle0", settle(6.0, 0.6)),
                    (f"{scen} m1", mark(scen, "wheel_in", pngs=(2, 5, 9))),
                    (f"{scen} wheel_in", wheel_burst(W * 0.62, H * 0.38, 6)),
                    (f"{scen} s1", settle()),
                    (f"{scen} m2", mark(scen, "wheel_out")),
                    (f"{scen} wheel_out", wheel_burst(W * 0.4, H * 0.6, -4)),
                    (f"{scen} s2", settle()),
                    (f"{scen} m3", mark(scen, "drag")),
                    (f"{scen} drag", drag_for(420.0, -260.0, 1.0)),
                    (f"{scen} s3", settle()),
                    (f"{scen} m4", mark(scen, "rotate", pngs=(6,))),
                    (f"{scen} rotate", rotate_for(50.0, 1.2)),
                    (f"{scen} s4", settle()),
                ]
                if full:
                    steps += [
                        (f"{scen} m5", mark(scen, "hold_zoom", pngs=(10,))),
                        (f"{scen} hold_zoom", hold_for("z", 1.5, (W * 0.55, H * 0.45))),
                        (f"{scen} s5", settle()),
                        (f"{scen} m6", mark(scen, "tap_plus", pngs=(3,))),
                        (f"{scen} tap_plus", tap("plus")),
                        (f"{scen} s6", settle()),
                    ]
                steps.append((f"{scen} end", mark(scen, "idle")))
        steps.append(("end", wait_s(0.2)))
        return steps


def make_study(app, outdir: str, quick: bool = False):
    """The scripted session for `gui --motion-study`."""
    from .selftest import SelfTest

    class MotionStudy(MotionStudyMixin, SelfTest):
        def __init__(self):
            SelfTest.__init__(self, app, outdir, quick=False)
            self.study_quick = quick
            self.study_init()

        def steps(self):
            return self.study_steps(self.study_quick)

        def after_frame(self):
            self.study_frame()
            SelfTest.after_frame(self)

        def _finish(self):
            try:
                self.study_finish()
            finally:
                SelfTest._finish(self)

    return MotionStudy()
