"""Scripted GUI session (`gui --selftest`, used by tests/test_gui.py): drives the explorer
through navigation, colours, keyframes, exports and both interface styles, taking screenshots
and recording what the engine did."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from imgui_bundle import hello_imgui

from gmpy2 import mpfr

from .mp import View
from .renderer import ColorSettings
from .app import InputSnapshot

if TYPE_CHECKING:
    from .app import App

class SelfTest:
    """Scripted UI session (used by the automated tests and `gui --selftest`)."""

    def __init__(self, app: App, outdir: str, quick: bool = False):
        self.app = app
        self.quick = quick
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MANDELSCOPE_HOME", str(self.out / "home"))
        self.frame = 0
        self.step = 0
        self.snap = InputSnapshot()
        self.pending_shot = None
        self.t_step = time.perf_counter()
        self.report = {"steps": [], "errors": []}
        self._steps = None

    def input(self) -> InputSnapshot:
        s = self.snap
        self.snap = InputSnapshot(mouse=s.mouse, down=s.down)
        return s

    def done_rendering(self):
        a = self.app
        return (a.field is not None and a.field.params is not None and a.field.complete
                and a.pending_ref is None and a.field_active and not a.is_moving())

    def shot(self, name):
        self.pending_shot = name

    def before_swap(self):
        if self.pending_shot:
            a = self.app
            W, H = a.fb
            data = a.ctx.screen.read(viewport=(0, 0, W, H), components=3, alignment=1)
            img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1]
            from PIL import Image
            Image.fromarray(img).save(self.out / f"{self.pending_shot}.png")
            self.pending_shot = None

    def record(self, name, **kw):
        a = self.app
        info = dict(name=name, frame=self.frame, t=round(time.perf_counter() - self.t_step, 2),
                    zoom=round(a.view.zoom_magnification_log10(), 3), angle=round(math.degrees(a.view.angle), 2),
                    max_iter=a.view.max_iter, field=(a.field.W, a.field.H) if a.field else None,
                    progress=round(a.field.progress, 3) if a.field else 0,
                    bla_skips=a.field.bla_skips if a.field else 0, aa=a.aa_count, **kw)
        self.report["steps"].append(info)
        if sys.stdout is not None:
            print("[selftest]", info, flush=True)

    def _finish(self):
        home = Path(os.environ.get("MANDELSCOPE_HOME", self.out / "home"))
        self.report["outputs"] = {str(p.relative_to(home)).replace("\\", "/"): p.stat().st_size
                                  for p in sorted(home.rglob("*")) if p.is_file()} if home.exists() else {}
        (self.out / "report.json").write_text(json.dumps(self.report, indent=1, default=str))
        hello_imgui.get_runner_params().app_shall_exit = True

    def after_frame(self):
        self.frame += 1
        if self._steps is None:
            self._steps = self.steps()
        steps = self._steps
        if self.step >= len(steps):
            self._finish()
            return
        name, fn = steps[self.step]
        try:
            res = fn()
        except Exception:
            self.report["errors"].append((name, traceback.format_exc()))
            traceback.print_exc()
            res = True
        if res:
            self.step += 1
            self.t_step = time.perf_counter()
        if self.frame > 30000:
            self.report["errors"].append(("timeout", self.step))
            self._finish()

    # each step returns True when finished
    def steps(self):
        a = self.app
        W, H = a.fb
        st = self
        LOC = Path(__file__).parent / "locations"

        def wait_done(name, timeout=900, aa=False):
            def f():
                ok = st.done_rendering() and (not aa or a.aa_count >= a.settings.aa_passes)
                if ok or time.perf_counter() - st.t_step > timeout:
                    st.record(name, complete=st.done_rendering())
                    st.shot(name)
                    return True
                return False
            return f

        def act(fn):
            def f():
                fn()
                return True
            return f

        def frames(n):
            c = {"n": 0}

            def f():
                c["n"] += 1
                return c["n"] >= n
            return f

        def wheel(x, y, amount):
            def f():
                st.snap = InputSnapshot(mouse=(x, y), wheel=amount)
                return True
            return f

        def drag(dx, dy, n=10, button=0, shift=False):
            c = {"i": 0}

            def f():
                i = c["i"]
                x0, y0 = W * 0.5, H * 0.5
                dn = tuple(b == button for b in range(3))
                if i == 0:
                    st.snap = InputSnapshot(mouse=(x0, y0), down=dn, clicked=dn, shift=shift)
                elif i <= n:
                    st.snap = InputSnapshot(mouse=(x0 + dx * i / n, y0 + dy * i / n), down=dn, shift=shift)
                else:
                    st.snap = InputSnapshot(mouse=(x0 + dx, y0 + dy), released=dn, shift=shift)
                    return True
                c["i"] += 1
                return False
            return f

        def arc(deg, n=12, radius=220.0, shift=False):
            """Right-drag the mouse along an arc around the centre (rotation)."""
            c = {"i": 0}

            def f():
                i = c["i"]
                cx, cy = W * 0.5, H * 0.5
                ang = math.radians(deg) * min(i, n) / n
                pos = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
                dn = (False, True, False)
                if i == 0:
                    st.snap = InputSnapshot(mouse=pos, down=dn, clicked=dn, shift=shift)
                elif i <= n:
                    st.snap = InputSnapshot(mouse=pos, down=dn, shift=shift)
                else:
                    st.snap = InputSnapshot(mouse=pos, released=dn, shift=shift)
                    return True
                c["i"] += 1
                return False
            return f

        def hold_key(key, n, mouse=None, shift=False):
            c = {"i": 0}

            def f():
                c["i"] += 1
                pressed = {key} if c["i"] == 1 else set()
                st.snap = InputSnapshot(mouse=mouse or (W * 0.5, H * 0.5), keys={key}, pressed=pressed,
                                        shift=shift)
                if c["i"] == 1:
                    a.key_down_t[key] = time.perf_counter() - 1.0     # treat as held
                return c["i"] >= n
            return f

        pad_ref = {}

        def pad_setup():
            from .gamepad import Gamepad
            if a.gamepad is None:
                a.gamepad = Gamepad(lib=False)          # no GLFW library: a stub
            from .control import PadState
            a.gamepad.name = "Simulated controller"
            a.gamepad.fake = PadState.make()        # (from now on no scan for real pads renames it)
            a.settings.pad_enabled = True
            a.settings.ui_style = "hud"
            v = a.view
            pad_ref.update(v=v.copy(), right=v.screen_to_c(W * 0.75, H * 0.5, W, H), look=a.colors.look,
                           boost=a.iter_boost)

        def pad_hold(n, **axes):
            from .control import PadState
            c = {"i": 0}

            def f():
                a.gamepad.fake = PadState.make(**axes) if c["i"] < n else PadState.make()
                c["i"] += 1
                return c["i"] > n
            return f

        def pad_press(*names):
            from .control import PadState
            c = {"i": 0}

            def f():
                a.gamepad.fake = PadState.make(pressed=names) if c["i"] == 0 else PadState.make()
                c["i"] += 1
                return c["i"] > 2
            return f

        def pad_after_pan():
            pad_ref["panned"] = a.view.copy()

        def pad_check():
            v0, v, vp = pad_ref["v"], a.view, pad_ref["panned"]
            # the stick pushed right: the centre moved along the old view's screen x axis
            right = complex(float(pad_ref["right"][0] - v0.re), float(pad_ref["right"][1] - v0.im))
            move = complex(float(vp.re - v0.re), float(vp.im - v0.im))
            along = (move * right.conjugate()).real / max(abs(move) * abs(right), 1e-300)
            res = dict(pad=a.pad_name, moved_right=abs(move) > 0 and along > 0.95,
                       moved_px=round(abs(move) / abs(right) * W * 0.25), zoomed_in=v.log2r < v0.log2r - 0.5,
                       turned=abs(v.angle - v0.angle) > 0.05, look=a.colors.look,
                       look_changed=a.colors.look != pad_ref["look"], more_iter=a.iter_boost > pad_ref["boost"])
            st.record("controller", **res)
            bad = [k for k in ("moved_right", "zoomed_in", "turned", "look_changed", "more_iter") if not res[k]]
            if bad or res["pad"] != "Simulated controller":
                st.report["errors"].append(("controller", res))

        stream = {}
        stash = {}
        STREAM_N = 2_000_000

        def stream_setup():
            # a long, non-periodic reference (c = i is pre-periodic): the picture must appear
            # while the orbit is still being computed, then continue when it is complete
            a.stream_after = 0.05
            a.settings.auto_iter = False
            # (2M iterations: several seconds of orbit even with the faster references, so the
            # first picture reliably comes from a partial orbit)
            a.manual_max_iter = STREAM_N
            v = View(mpfr("0", 128), mpfr("1", 128), -12.0, 0.0, STREAM_N)
            a.set_view(v)
            stream.update(t0=time.perf_counter(), first=None)

        def stream_first():
            f = a.field
            if (stream["first"] is None and f is not None and f.complete and a.field_active
                    and a.pending_ref is not None and a.refs.ref is not None):
                stream["first"] = dict(ref_len=a.refs.ref.length, cap=f.params.max_iter,
                                       secs=round(time.perf_counter() - stream["t0"], 2))
            return stream["first"] is not None or a.pending_ref is None or time.perf_counter() - st.t_step > 300

        def stream_check():
            first = stream["first"]
            r = a.refs.ref
            res = dict(first=first, final_len=r.length if r else None, cap=a.field.params.max_iter,
                       secs=round(time.perf_counter() - stream["t0"], 2))
            st.record("stream", **res)
            if first is None or first["ref_len"] >= STREAM_N + 1 or first["cap"] >= STREAM_N \
                    or r is None or r.length < STREAM_N + 1 or a.field.params.max_iter != STREAM_N:
                st.report["errors"].append(("stream", res))
            a.stream_after = 0.5
            a.settings.auto_iter = True
            a.manual_max_iter = None

        def tap(key, shift=False):
            def f():
                st.snap = InputSnapshot(mouse=(W * 0.5, H * 0.5), keys={key}, pressed={key}, shift=shift)
                return True
            return f

        def watch(label, inner):
            """Run another step while measuring, every frame, the uncovered (pure background)
            part of the picture and how often a complete picture arrived."""
            c = {"black": 0.0, "snaps": 0, "serial": None, "frames": 0, "flight0": None}

            def f():
                if c["flight0"] is None:
                    c["flight0"] = a.flight_frames
                    a.uncovered_colour = (1.0, 0.0, 1.0)      # magenta marks uncovered pixels
                done = inner()
                img = a.screen_rgb()
                b = float(np.mean(np.all(img == (255, 0, 255), axis=2)))
                if done:
                    a.uncovered_colour = (0.0, 0.0, 0.0)
                c.setdefault("series", []).append(round(b, 3))
                if b > c["black"] and os.environ.get("MANDELSCOPE_SELFTEST_WORST"):
                    from PIL import Image
                    Image.fromarray(img).save(st.out / f"worst_{label}.png")
                    c["worst_info"] = dict(frame=c["frames"], field_view=(a.field_view.log2r, a.field_view.angle) if a.field_view else None,
                                           view=(a.view.log2r, a.view.angle), snap=(a.snap_view.log2r, a.snap_view.angle) if a.snap_view else None,
                                           progress=a.field.progress if a.field else None, level=a.field.level_done if a.field else None,
                                           active=a.field_active)
                c["black"] = max(c["black"], b)
                if a.snap_serial != c["serial"]:
                    c["snaps"] += 1
                    c["serial"] = a.snap_serial
                c["frames"] += 1
                if done:
                    st.record(label, black=round(c["black"], 5), snapshots=c["snaps"], frames=c["frames"],
                              in_flight=a.flight_frames - c["flight0"], resets=list(a.reset_log[-6:]),
                              done=a.field.done_count if a.field else 0, N=a.field.N if a.field else 0,
                              field_T=round(a.field_T, 3), series=c.get("series"), worst=c.get("worst_info"))
                return done
            return f

        def check_pan_reuse():
            f = a.field
            st.record("pan_reuse", done=f.done_count, N=f.N, resets=list(a.reset_log[-12:]))
            return True

        def load(name):
            return act(lambda: a.load_location(str(LOC / name)))

        def tab(name):
            return act(lambda: setattr(a.settings, "sidebar_tab", name))

        def wait_previews(name, timeout=600):
            def f():
                p = a.previews
                ok = p.done_key is not None and p.done_key == p.view_key
                if ok or time.perf_counter() - st.t_step > timeout:
                    stds = {k: round(float(np.frombuffer(t.read(), dtype=np.uint8).std()), 1)
                            for k, t in p.textures.items()}
                    st.record(name, done=ok, stds=stds)
                    st.shot(name)
                    return True
                return False
            return f

        def julia_point_view(log2r, max_iter=1000):
            """A view at the repelling fixed point of the current Julia set (on the set at every scale)."""
            import gmpy2
            from gmpy2 import mpc, mpfr
            p = int(-log2r) + 80
            with gmpy2.context(gmpy2.get_context(), precision=p):
                cc = mpc(mpfr(a.julia_c[0], p), mpfr(a.julia_c[1], p))
                b = (1 + gmpy2.sqrt(1 - 4 * cc)) / 2
                o = gmpy2.mul_2exp(mpfr(0.31, p), int(log2r))
                return a.apply_auto_iter(View(b.real + o, b.imag - o, log2r, 0.4, max_iter))

        def julia_rec(name):
            def f():
                ns = a.field.read_results()[..., 0]
                st.record(name, deep=a.julia_deep(), double=a.field.double, program=a.field.program is not None,
                          distinct=int(len(np.unique(ns))), log2r=round(a.view.log2r, 2))
                return True
            return f

        def wait_job(name, timeout=900):
            def f():
                running = [j for j in a.jobs if j.status in ("queued", "running")]
                if not running or time.perf_counter() - st.t_step > timeout:
                    st.record(name, jobs=[(j.kind, j.status, round(j.elapsed, 1), j.error) for j in a.jobs])
                    return True
                return False
            return f

        def find_mb():
            a.find_minibrot_at(a.view.re, a.view.im)
            return True

        def wait_mb():
            if a.minibrot_job is None:
                st.record("minibrot", result=None if a.minibrot_result is None else
                          {k: a.minibrot_result[k] for k in ("period", "log2_size")})
                if a.minibrot_result:
                    a.go_minibrot(a.minibrot_result)
                return True
            return False

        zo = {}

        def zoom_out_frames(n, decades_out=5.0, max_frames=1500):
            """Hold Shift+minus from the deep minibrot for at least n frames and until the view is
            `decades_out` decades out (zoom speed is per second: 40 frames are 0.7 s at 60 fps on
            a GPU, one decade): the iteration limit must follow the depth down and the skip tables
            must stay on (the v3.11 lag)."""
            c = {"i": 0}

            def f():
                c["i"] += 1
                if c["i"] == 1:
                    zo.update(start_iter=a.view.max_iter, start_zoom=a.view.zoom_magnification_log10(),
                              no_bla=0, frames=0, boost=a.iter_boost)
                    a.key_down_t["minus"] = time.perf_counter() - 1.0
                fp = a.field.params if a.field is not None else None
                if c["i"] > 2 and fp is not None and not fp.use_bla:
                    zo["no_bla"] += 1
                zo["frames"] += 1
                st.snap = InputSnapshot(mouse=(W * 0.5, H * 0.5), keys={"minus"},
                                        pressed={"minus"} if c["i"] == 1 else set(), shift=True)
                out = zo["start_zoom"] - a.view.zoom_magnification_log10()
                if c["i"] >= max_frames or (c["i"] >= n and out >= decades_out):
                    v = a.view
                    st.record("zoom_out", **zo, end_iter=v.max_iter, end_zoom=v.zoom_magnification_log10(),
                              plain_iter=int(a.settings.iter_base + a.settings.iter_gain *
                                             max(0.0, v.zoom_magnification_log10())))
                    return True
                return False
            return f

        def wait_ship_mb():
            if a.minibrot_job is None:
                r = a.minibrot_result
                st.record("ship_minibrot", result=None if r is None else {k: r[k] for k in ("period", "log2_size")})
                if r:
                    a.go_minibrot(r)
                return True
            return False

        def ship_rec(name):
            def f():
                ns = a.field.read_results()[..., 0]
                st.record(name, formula=a.formula, field_formula=a.field.params.formula,
                          program=a.field.program is not None, distinct=int(len(np.unique(ns))),
                          loc_formula=a.location_dict().get("formula"), julia=a.julia_c)
                return True
            return f

        def script_steps():
            a.script_text = ('look Nebula\nstart home\nzoom to location "spiral_1e-100" over 4s spin 0.5\n'
                             'rotate 90 over 1s\nhold 0.5s\n')
            ok = a.apply_script()
            st.record("script", ok=ok, info=a.script_info, errors=a.script_errors,
                      keyframes=len(a.sequence.keyframes))
            return True

        def bad_script():
            ok = a.apply_script("start home\nzoom sideways\n")
            st.record("bad_script", ok=ok, errors=a.script_errors)
            return True

        def video_job():
            s = a.settings
            s.video_w, s.video_h, s.video_fps, s.video_ss = 160, 90, 8.0, 1
            s.video_name = "selftest"
            s.video_overlay = True
            a.queue_video()
            return True

        def save_lib():
            e = a.save_to_library("selftest spot")
            st.record("library_save", entry=e.name, thumb=bool(e.thumb))
            return True

        if os.environ.get("MANDELSCOPE_SELFTEST_ONLY") == "motion":
            # development: only the motion measurements
            return [
                ("load_valley", load("seahorse_valley.json")),
                ("valley_done", wait_done("m0_valley")),
                ("hold_out", watch("hold_zoom_out", hold_key("x", 30, (W * 0.6, H * 0.4)))),
                ("w1", wait_done("m1")),
                ("hold_rot", watch("hold_rotate", hold_key("e", 30))),
                ("w2", wait_done("m2")),
                ("key_minus", tap("minus")),
                ("minus_frames", watch("tap_zoom_out", frames(30))),
                ("w3", wait_done("m3")),
                ("rot90", tap("lbracket")),
                ("rot_frames", watch("rotate_90_anim", frames(30))),
                ("w4", wait_done("m4")),
                ("hold_zoom", watch("hold_zoom_in", hold_key("z", 30, (W * 0.45, H * 0.5)))),
                ("w5", wait_done("m5")),
            ]
        if self.quick:
            # short smoke test (used for the packaged .exe): GPU, worker processes, UI, export
            return [
                ("home", wait_done("01_home")),
                ("wheel", wheel(W * 0.62, H * 0.32, 4.0)),
                ("zoom_anim", frames(40)),
                ("after_wheel", wait_done("02_wheel_zoom")),
                ("drag", drag(-37, 23)),
                ("reuse", check_pan_reuse),
                ("inertia_frames", frames(30)),
                ("after_drag", wait_done("03_pan")),
                ("load_spiral", load("spiral_1e-100.json")),
                ("spiral", wait_done("07_spiral_1e-100")),
                ("colour_tab", tab("Colour")),
                ("look_frames", frames(30)),
                ("look_shot", wait_done("08_colour_tab")),
                ("library_tab", tab("Library")),
                ("library_save", save_lib),
                ("lib_frames", frames(5)),
                ("lib_shot", act(lambda: st.shot("13_library_tab"))),
                ("animate_tab", tab("Animate")),
                ("script", script_steps),
                ("export_tab", tab("Export")),
                ("video", video_job),
                ("video_wait", wait_job("video_export")),
                ("export_frames", frames(3)),
                ("export_shot", act(lambda: st.shot("16_export_tab"))),
                ("render_tab", tab("Render")),
                ("render_frames", frames(3)),
                ("render_shot", act(lambda: st.shot("17_render_tab"))),
                ("help", act(lambda: setattr(a, "show_help", True))),
                ("help_frames", frames(3)),
                ("help_shot", act(lambda: st.shot("20_help"))),
                ("end_frames", frames(2)),
            ]
        return [
            ("home", wait_done("01_home", aa=True)),
            ("wheel", wheel(W * 0.62, H * 0.32, 4.0)),
            ("zoom_anim", watch("wheel_zoom_4x", frames(40))),
            ("after_wheel", wait_done("02_wheel_zoom")),
            ("drag", drag(-37, 23)),
            ("reuse", check_pan_reuse),
            ("inertia_frames", frames(30)),
            ("after_drag", wait_done("03_pan")),
            ("load_valley", load("seahorse_valley.json")),
            ("valley_frames", frames(5)),
            ("valley_done", wait_done("04a_valley")),
            ("key_plus", tap("plus")),
            ("plus_reuse", watch("tap_zoom_2x", frames(2))),
            ("plus_frames", watch("tap_zoom_anim", frames(30))),
            ("hold_zoom", watch("hold_zoom_in", hold_key("z", 30, (W * 0.45, H * 0.5)))),
            ("after_hold", wait_done("04_keyboard_zoom")),
            ("hold_out", watch("hold_zoom_out", hold_key("x", 30, (W * 0.6, H * 0.4)))),
            ("hold_rot", watch("hold_rotate", hold_key("e", 30))),
            ("after_motion", wait_done("04b_after_motion")),
            ("rot_reset", tap("zero")),
            ("rot_reset_frames", frames(30)),
            ("key_minus", tap("minus")),
            ("minus_frames", watch("tap_zoom_out", frames(30))),
            ("after_minus", wait_done("04c_zoomed_out")),
            ("rot90", tap("lbracket")),
            ("rot_frames", watch("rotate_90_anim", frames(30))),
            ("after_rot90", wait_done("05_rotated_90")),
            ("rot_drag", arc(50.0, n=12, shift=True)),
            ("after_rot_drag", wait_done("06_rotate_snap")),
            ("rot0", tap("zero")),
            ("rot0_frames", frames(30)),
            ("load_spiral", load("spiral_1e-100.json")),
            ("spiral", wait_done("07_spiral_1e-100", aa=True)),
            ("colour_tab", tab("Colour")),
            ("look_frames", frames(60)),
            ("look_shot", wait_done("08_colour_tab")),
            ("look_ember", act(lambda: setattr(a, "colors", ColorSettings.from_look("Ember", a.colors)))),
            ("ember", wait_done("09_look_ember")),
            ("stripes", act(lambda: (setattr(a.colors, "stripes", 0.8)))),
            ("stripes_done", wait_done("10_stripes")),
            ("load_deep", load("deep_1e-1000.json")),
            ("deep", wait_done("11_deep_1e-1000")),
            ("precise", act(lambda: setattr(a.settings, "precision", 1))),
            ("deep_f64", wait_done("12_deep_f64")),
            ("deep_f64_rec", act(lambda: st.record("deep_f64", double=a.field.double))),
            ("precise_ff", act(lambda: setattr(a.settings, "precision", 2))),
            ("deep_ff", wait_done("12b_deep_ff")),
            ("deep_ff_rec", act(lambda: st.record("deep_ff", double=a.field.double,
                                                  label=a.delta_precision()))),
            ("fast", act(lambda: setattr(a.settings, "precision", 0))),
            ("library_tab", tab("Library")),
            ("lib_frames", frames(5)),
            ("library_save", save_lib),
            ("lib_shot", wait_done("13_library_tab")),
            ("animate_tab", tab("Animate")),
            ("script", script_steps),
            ("bad_script", bad_script),
            ("script2", script_steps),
            ("anim_shot", frames(5)),
            ("anim_shot2", act(lambda: st.shot("14_animate_tab"))),
            ("play", act(a.toggle_play)),
            ("playing", frames(40)),
            ("play_shot", act(lambda: st.shot("15_playback"))),
            ("stop", act(lambda: setattr(a, "playing", False))),
            ("export_tab", tab("Export")),
            ("video", video_job),
            ("video_wait", wait_job("video_export")),
            ("export_shot", frames(3)),
            ("export_shot2", act(lambda: st.shot("16_export_tab"))),
            ("still", act(lambda: (setattr(a.settings, "still_w", 320), setattr(a.settings, "still_h", 180),
                                   setattr(a.settings, "still_ss", 2), a.start_still()))),
            ("still_wait", wait_job("still_export")),
            ("live_start", act(lambda: (setattr(a.settings, "live_fps", 30.0), a.toggle_live()))),
            ("live_zoom", hold_key("x", 40)),
            ("live_stop", act(a.toggle_live)),
            ("path_start", act(a.toggle_path_recording)),
            ("path_zoom", hold_key("z", 30, (W * 0.3, H * 0.3))),
            ("path_pan", hold_key("d", 15)),
            ("path_stop", act(a.toggle_path_recording)),
            ("path_seq", act(lambda: st.record("path_sequence", keyframes=len(a.sequence.keyframes),
                                               duration=a.sequence.duration))),
            ("settings_tab", tab("Render")),
            ("load_sea", load("seahorse_1e-12.json")),
            ("sea", wait_done("17_seahorse")),
            ("minibrot", find_mb),
            ("minibrot_wait", wait_mb),
            ("minibrot_view", wait_done("18_minibrot")),
            ("more_iter", act(lambda: a.scale_iter(8.0))),            # as if the place had needed them
            ("zoom_out", zoom_out_frames(40)),
            ("zoom_out_done", wait_done("18b_zoomed_out")),
            ("back", act(a.back)),
            ("final", wait_done("19_back")),
            # Julia sets: preview of the point under the mouse, explore one, back
            ("julia_prev_on", act(lambda: setattr(a.settings, "julia_preview", True))),
            ("julia_mouse", lambda: (setattr(st, "snap", InputSnapshot(mouse=(W * 0.42, H * 0.47))), True)[1]),
            ("julia_prev_frames", frames(25)),
            ("julia_prev_shot", act(lambda: (st.record("julia_preview", c=a.jpreview.c,
                                                       done=bool(a.jpreview.field and a.jpreview.field.complete)),
                                             st.shot("19a_julia_preview")))),
            ("julia_enter", tap("j", shift=True)),
            ("julia_frames", frames(3)),
            ("julia_done", wait_done("19b_julia_set", aa=True)),
            ("julia_mode", act(lambda: st.record("julia_mode", c=a.julia_c, program=a.field.program is not None,
                                                 inside=a.field.interior_count, undecided=a.field.maxiter_count))),
            ("julia_wheel", wheel(W * 0.55, H * 0.45, 2.0)),
            ("julia_zoom_frames", watch("julia_wheel_zoom", frames(40))),
            ("julia_zoomed", wait_done("19c_julia_zoom")),
            ("julia_drag", drag(-31, 17)),
            ("julia_inertia", frames(30)),
            ("julia_panned", wait_done("19d_julia_pan")),
            # deep Julia zooms: past the old float64 limit (2^-39) by perturbation
            ("julia_deep", act(lambda: a.set_view(julia_point_view(-50.0)))),
            ("julia_deep_done", wait_done("19e_julia_deep")),
            ("julia_deep_rec", julia_rec("julia_deep")),
            ("julia_deeper", act(lambda: a.set_view(julia_point_view(-400.0)))),
            ("julia_deeper_done", wait_done("19g_julia_deeper")),
            ("julia_deeper_rec", julia_rec("julia_deeper")),
            ("julia_deeper_wheel", wheel(W * 0.47, H * 0.52, 2.0)),
            ("julia_deeper_wheel_frames", frames(30)),
            ("julia_deeper_zoomed", wait_done("19h_julia_deeper_wheel")),
            ("julia_looks_tab", tab("Colour")),
            ("julia_looks", wait_previews("19i_julia_deep_looks")),
            ("julia_looks_back", tab("Render")),
            ("julia_still", act(lambda: (setattr(a.settings, "still_w", 320), setattr(a.settings, "still_h", 180),
                                         setattr(a.settings, "still_ss", 1), a.start_still()))),
            ("julia_still_wait", wait_job("julia_still")),
            ("julia_loc", act(lambda: st.record("julia_location", loc=a.location_dict().get("julia")))),
            ("julia_leave", tap("j")),
            ("julia_leave_frames", frames(3)),
            ("julia_back", wait_done("19f_back_to_mandelbrot")),
            ("julia_back_rec", act(lambda: st.record("julia_left", c=a.julia_c,
                                                     program=a.field.program is not None if a.field else None))),
            ("julia_prev_off", act(lambda: setattr(a.settings, "julia_preview", False))),
            # the Burning Ship: B, zoom, a deep place, the mini finder, a still, J refused, back
            ("mandel_before_ship", act(lambda: stash.update(view=a.view.copy()))),
            ("ship_on", tap("b")),
            ("ship_frames", frames(3)),
            ("ship_home", wait_done("22_ship_home")),
            ("ship_home_rec", ship_rec("ship_home")),
            ("ship_wheel", wheel(W * 0.3, H * 0.62, 3.0)),
            ("ship_wheel_frames", frames(40)),
            ("ship_zoomed", wait_done("22b_ship_zoom")),
            ("ship_deep", load("ship_mini_1e-33.json")),
            ("ship_deep_frames", frames(3)),
            ("ship_deep_done", wait_done("22c_ship_mini_1e-33")),
            ("ship_deep_rec", ship_rec("ship_deep")),
            ("ship_mb", act(lambda: a.find_minibrot_at(a.view.re, a.view.im))),
            ("ship_mb_wait", wait_ship_mb),
            ("ship_mb_frames", frames(3)),
            ("ship_mb_view", wait_done("22d_ship_minibrot")),
            ("ship_still", act(lambda: (setattr(a.settings, "still_w", 320), setattr(a.settings, "still_h", 180),
                                        setattr(a.settings, "still_ss", 1), a.start_still()))),
            ("ship_still_wait", wait_job("ship_still")),
            ("ship_julia", tap("j", shift=True)),
            ("ship_julia_frames", frames(3)),
            ("ship_julia_rec", ship_rec("ship_julia")),
            ("ship_off", tap("b")),
            ("ship_off_frames", frames(3)),
            ("ship_back", wait_done("22e_back_to_mandelbrot")),
            ("ship_back_rec", act(lambda: st.record("ship_left", formula=a.formula,
                                                    field_formula=a.field.params.formula,
                                                    same_view=a.view.re == stash["view"].re
                                                    and abs(a.view.log2r - stash["view"].log2r) < 1e-9))),
            ("help", act(lambda: setattr(a, "show_help", True))),
            ("help_frames", frames(3)),
            ("help_shot", act(lambda: st.shot("20_help"))),
            ("help_off", act(lambda: setattr(a, "show_help", False))),
            ("rclick_down", lambda: (setattr(st, "snap", InputSnapshot(mouse=(W * 0.5, H * 0.5), down=(False, True, False),
                                                                     clicked=(False, True, False))), True)[1]),
            ("rclick_up", lambda: (setattr(st, "snap", InputSnapshot(mouse=(W * 0.5, H * 0.5),
                                                                   released=(False, True, False))), True)[1]),
            ("ctx_wait", frames(3)),
            ("ctx_shot", act(lambda: (st.record("context_menu", open=a.context_menu_c is not None),
                                      st.shot("21_context_menu")))),
            ("ctx_frames", frames(2)),
            ("hide_ui", tap("tab")),
            ("clean_frames", frames(3)),
            ("clean_shot", act(lambda: st.shot("22_clean_view"))),
            ("clean_frames2", frames(2)),
            ("show_ui", tap("tab")),
            ("hud_rec", act(lambda: st.record("ui_hud", style=ui_style_now(), hud_fonts="hud" in ui_fonts()))),
            ("classic_on", act(lambda: setattr(a.settings, "ui_style", "classic"))),
            ("classic_frames", frames(3)),
            ("classic_shot", act(lambda: (st.record("ui_classic", style=ui_style_now()),
                                          st.shot("23_classic_style")))),
            ("hud_back", act(lambda: setattr(a.settings, "ui_style", "hud"))),
            ("hud_frames", frames(2)),
            ("welcome_on", act(lambda: setattr(a, "force_welcome", True))),
            ("welcome_frames", frames(3)),
            ("welcome_shot", act(lambda: (st.record("welcome", shown=not a.settings.welcome_seen),
                                          st.shot("24_welcome")))),
            ("welcome_off", act(lambda: setattr(a, "force_welcome", False))),
            ("pad_setup", act(pad_setup)),
            ("pad_pan", pad_hold(20, lx=1.0)),
            ("pad_pan_rec", act(pad_after_pan)),
            ("pad_zoom", pad_hold(25, r2=1.0)),
            ("pad_turn", pad_hold(15, rx=1.0)),
            ("pad_look", pad_press("right")),
            ("pad_iter", pad_press("up")),
            ("pad_check", act(pad_check)),
            ("pad_wait", wait_done("25_controller")),
            ("pad_off", act(lambda: setattr(a.gamepad, "fake", None))),
            ("stream_setup", act(stream_setup)),
            ("stream_first", stream_first),
            ("stream_done", wait_done("26_stream")),
            ("stream_check", act(stream_check)),
            ("end_frames", frames(2)),
        ]


def ui_style_now() -> str:
    from . import ui
    return ui._state.get("style", ("?",))[0]


def ui_fonts() -> dict:
    from . import ui
    return ui.FONTS
