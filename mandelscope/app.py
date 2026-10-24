"""Interactive explorer: GLFW + Dear ImGui (hello_imgui) + moderngl compute.

The App owns the state (view, colours, sequence, jobs) and the per-frame loop:
input -> animations -> workers -> GPU compute (time budget) -> compositing.
All drawing of panels lives in ui.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np
import moderngl
from gmpy2 import mpfr
from imgui_bundle import imgui, hello_imgui

from .mp import View, to_str
from .renderer import (GPUEngine, Field, ColorSettings, AutoContrast, view_params, grid_map, snap_view_to_grid,
                       julia_params, julia_min_log2r, PREC_FF)
from .offline import ReferenceCache, JuliaReferences, FrameRenderer, RenderSettings, render_sequence_iter
from .formulas import FORMULAS, FORMULA_NAMES, home_view as formula_home
from .palettes import LOOKS, DEFAULT_LOOK, CUSTOM, set_custom
from .sequence import Sequence, Keyframe, PathRecorder, interpolate_views, ease
from .script import compile_script, sequence_to_script, ai_prompt
from .library import Library, user_root, location_from_png
from .video import FfmpegWriter
from .worker import Worker
from .bla import build_bla
from .control import (ZOOM_OMEGA, spring_step, UNDECIDED_OK, iteration_advice, FramePacer,  # noqa: F401
                      auto_motion_level, PadInput, pad_controls)
from . import software

CONFIG_DIR = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "mandelscope"
LEGACY_CONFIG_DIR = CONFIG_DIR.parent / "deepzoom"        # pre-release name: settings are read once
DEBUG = bool(os.environ.get("MANDELSCOPE_DEBUG"))
ASSETS = Path(__file__).parent / "assets"
DEFAULT_SCRIPT = """# Example: dive into the seahorse valley and spin
look Gilded
start home
zoom to location "seahorse_1e-12" over 25s
spin 1 over 6s
hold 2s
"""


def out_dir(kind: str) -> Path:
    return user_root() / kind


@dataclass
class Settings:
    precision: int = 0              # 0 fast f32, 1 precise f64, 2 precise float-float
    render_scale: float = 1.0       # samples per pixel (axis) when idle: 1 (2,3,4 = supersampling)
    motion_scale: float = 0.0       # samples per pixel while moving (0.5 = faster, softer; 0 = automatic)
    aa_passes: int = 8              # idle progressive anti-aliasing: jittered passes accumulated
    adaptive_ss: bool = True        # idle anti-aliasing passes only where the picture needs them
    adaptive_export: bool = False   # export supersampling only where needed (slower on an RTX 5060 Ti in v3.0)
    use_bla: bool = True
    eps_f32: float = -24.0
    eps_f64: float = -45.0
    interior: bool = True
    auto_iter: bool = True
    iter_base: int = 1000
    iter_gain: int = 400            # extra iterations per decade of zoom (auto mode)
    zoom_speed: float = 2.0         # octaves per second (keys, held)
    wheel_step: float = 0.5         # octaves per wheel notch
    pan_speed: float = 700.0        # pixels per second (keys)
    inertia: bool = True
    inertia_time: float = 0.35      # seconds (decay constant)
    wheel_spring: bool = True       # zoom animations follow a spring (smooth speed); False: v3.6
    frame_pacing: bool = True       # camera steps follow the display interval (no judder from timing jitter)
    ui_style: str = "hud"           # "hud": cinematic HUD (v3.8), "classic": toolbar + status bar
    welcome_seen: bool = False      # the first-run card with the basic controls was closed
    hud_accent: str = "ice"         # HUD accent colour: ice, gold, coral, lime, lilac
    hud_brackets: bool = True       # HUD: corner brackets
    hud_reticle: bool = True        # HUD: centre mark while moving
    show_fallback: bool = True
    frame_budget_ms: float = 10.0
    video_w: int = 1920
    video_h: int = 1080
    video_fps: float = 60.0
    video_ss: int = 2
    video_codec: str = "auto"
    video_quality: int = 18
    video_precision: int = 0
    video_iter_mult: int = 1        # videos: iteration limit x this (the explorer's stays)
    video_overlay: bool = False
    video_png: bool = False
    fast_movie: bool = True         # videos: frames cut out of shared key pictures (keymovie.py); v3.4
                                    # bench on the RTX 5060 Ti: 1.4-4.4x faster, same picture
    julia_preview: bool = False     # small live Julia set for the point under the cursor (J)
    video_name: str = "mandelscope"
    still_w: int = 3840
    still_h: int = 2160
    still_ss: int = 2
    still_iter_mult: int = 1        # stills: iteration limit x this
    live_fps: float = 60.0
    live_codec: str = "auto"
    live_quality: int = 20
    live_ui: bool = False
    auto_time_speed: float = 1.0
    seq_loop: bool = False
    show_timeline: bool = False     # shown automatically once there are keyframes
    sidebar_open: bool = True
    sidebar_tab: str = "Explore"
    sidebar_width: float = 380.0
    ui_scale: float = 0.0            # 0 = follow the Windows display scaling; e.g. 1.25 = larger UI
    pad_enabled: bool = True         # game controller (PlayStation / Xbox / ... through GLFW)
    pad_speed: float = 1.0           # controller speed factor (pan, zoom, turn)
    pad_deadzone: float = 0.15       # stick dead zone (0..0.5)
    pad_invert_zoom: bool = False    # right stick: down = zoom in


@dataclass
class InputSnapshot:
    """What the explorer needs from the input devices for one frame."""
    mouse: tuple = (0.0, 0.0)          # framebuffer pixels
    down: tuple = (False, False, False)
    clicked: tuple = (False, False, False)
    released: tuple = (False, False, False)
    double: bool = False
    wheel: float = 0.0
    keys: set = field(default_factory=set)      # held keys (names)
    pressed: set = field(default_factory=set)   # keys pressed this frame
    shift: bool = False
    ctrl: bool = False
    ui_mouse: bool = False
    typing: bool = False
    pad: PadInput | None = None                 # game controller (None: none / switched off)


KEYMAP = {"z": imgui.Key.z, "x": imgui.Key.x, "w": imgui.Key.w, "a": imgui.Key.a, "s": imgui.Key.s,
          "d": imgui.Key.d, "q": imgui.Key.q, "e": imgui.Key.e, "h": imgui.Key.h, "k": imgui.Key.k,
          "r": imgui.Key.r, "p": imgui.Key.p, "m": imgui.Key.m, "t": imgui.Key.t, "c": imgui.Key.c,
          "v": imgui.Key.v, "j": imgui.Key.j, "b": imgui.Key.b,
          "space": imgui.Key.space, "home": imgui.Key.home, "backspace": imgui.Key.backspace,
          "left": imgui.Key.left_arrow, "right": imgui.Key.right_arrow, "up": imgui.Key.up_arrow,
          "down": imgui.Key.down_arrow, "plus": imgui.Key.equal, "kp_plus": imgui.Key.keypad_add,
          "minus": imgui.Key.minus, "kp_minus": imgui.Key.keypad_subtract,
          "lbracket": imgui.Key.left_bracket, "rbracket": imgui.Key.right_bracket, "zero": imgui.Key._0,
          "tab": imgui.Key.tab, "f1": imgui.Key.f1, "f12": imgui.Key.f12, "escape": imgui.Key.escape,
          "pageup": imgui.Key.page_up, "pagedown": imgui.Key.page_down}


def read_input(fb_scale) -> InputSnapshot:
    io = imgui.get_io()
    sx, sy = fb_scale
    mp_ = io.mouse_pos
    snap = InputSnapshot()
    snap.mouse = (mp_.x * sx, mp_.y * sy)
    snap.down = tuple(bool(io.mouse_down[i]) for i in range(3))
    snap.clicked = tuple(bool(imgui.is_mouse_clicked(i)) for i in range(3))
    snap.released = tuple(bool(imgui.is_mouse_released(i)) for i in range(3))
    snap.double = bool(imgui.is_mouse_double_clicked(0))
    snap.wheel = float(io.mouse_wheel)
    snap.keys = {n for n, k in KEYMAP.items() if imgui.is_key_down(k)}
    snap.pressed = {n for n, k in KEYMAP.items() if imgui.is_key_pressed(k, False)}
    snap.shift = bool(io.key_shift)
    snap.ctrl = bool(io.key_ctrl)
    snap.ui_mouse = bool(io.want_capture_mouse)
    snap.typing = bool(io.want_text_input)
    return snap


R2_A, R2_B = 0.7548776662466927, 0.5698402909980532


def jitter(k: int) -> tuple:
    """Low-discrepancy (R2) sub-sample offsets in [-0.5, 0.5)."""
    return ((k * R2_A) % 1.0 - 0.5, (k * R2_B) % 1.0 - 0.5)


class LiveRecorder:
    def __init__(self, path, w, h, fps, codec, quality, include_ui):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = w - (w % 2), h - (h % 2)
        self.writer = FfmpegWriter(path, self.w, self.h, fps, codec, quality)
        self.fps = fps
        self.t0 = time.perf_counter()
        self.include_ui = include_ui
        self.path = path
        self.pbos = []                  # asynchronous read-back: two pixel buffers in turn
        self.pending = None             # (pbo, repeat) captured last frame, not read yet
        self.queued = 0

    def capture(self, fbo, full_h):
        """Start an asynchronous read-back of this frame and hand the previous one to ffmpeg
        (reading a frame back synchronously would stall the GPU pipeline every frame)."""
        due = int((time.perf_counter() - self.t0) * self.fps) + 1 - self.queued
        if due <= 0:
            return
        if not self.pbos:
            ctx = fbo.ctx
            self.pbos = [ctx.buffer(reserve=self.w * self.h * 3) for _ in range(2)]
        pbo = self.pbos[0] if (self.pending is None or self.pending[0] is self.pbos[1]) else self.pbos[1]
        fbo.read_into(pbo, viewport=(0, full_h - self.h, self.w, self.h), components=3, alignment=1)
        rep = min(due, 4)
        self.queued += rep
        self._flush_pending()
        self.pending = (pbo, rep)

    def _flush_pending(self):
        if self.pending is None:
            return
        pbo, rep = self.pending
        self.pending = None
        img = np.frombuffer(pbo.read(), dtype=np.uint8).reshape(self.h, self.w, 3)[::-1]
        self.writer.write(img, repeat=rep)

    def close(self):
        try:
            self._flush_pending()
        finally:
            for b in self.pbos:
                b.release()
            self.pbos = []
        return self.writer.close()


@dataclass
class Job:
    kind: str                       # 'video' | 'still' | 'draft'
    name: str
    path: str
    status: str = "queued"          # queued | running | paused | done | failed | cancelled
    progress: float = 0.0
    frame: int = 0
    frames: int = 1
    eta: float = 0.0
    error: str = ""
    codec: str = ""
    t0: float = 0.0
    elapsed: float = 0.0
    gen: object = None
    make: object = None             # callable creating the generator
    open_when_done: bool = False


class LookPreviews:
    """Live thumbnails of every look, computed from a small field of the current view."""

    def __init__(self, app: "App", w: int = 224):
        self.app = app
        self.w = w
        self.h = 126
        self.textures: dict = {}
        self.fbos: dict = {}
        self.field: Field | None = None
        self.view_key = None
        self.done_key = None
        self.auto = AutoContrast(app.ctx)
        self.wanted = False

    def tex(self, name: str):
        return self.textures.get(name)

    def _ensure(self):
        a = self.app
        W, H = a.fb
        aspect = min(0.75, max(0.45, H / max(W, 1)))      # keep the thumbnails landscape
        h = max(32, int(round(self.w * aspect)))
        if self.field is None or self.field.H != h:
            if self.field is not None:
                self.field.release()
            for f in self.fbos.values():
                f.release()
            for t in self.textures.values():
                t.release()
            self.fbos.clear()
            self.textures.clear()
            self.h = h
            self.field = Field(a.engine, self.w, h, False)
        return self.field

    def update(self, budget: float):
        """Called when idle and the Colour tab is visible."""
        a = self.app
        if a.field_view is None:
            return
        v = a.field_view
        julia = a.julia_c is not None
        deep = julia and a.julia_deep(v, self.h)       # perturbation, with the explorer's references
        refs = a.active_refs() if (deep or not julia) else None
        if refs is not None and refs.ref is None:
            return
        if refs is None:
            key = (str(v.re), str(v.im), v.log2r, v.angle, v.max_iter, "julia", a.julia_c)
        else:
            key = (str(v.re), str(v.im), v.log2r, v.angle, v.max_iter, id(refs.ref), deep,
                   getattr(refs, "gpu_serial", 0))
        f = self._ensure()
        if self.view_key != key and refs is None:
            fp = julia_params(v, self.w, self.h, with_de=True, with_interior=a.settings.interior,
                              with_stripes=True)
            f.reset()
            f.configure(fp, None, None)
            a.engine.setup_julia(f, v, a.julia_c, self.w, self.h, fp)
            self.view_key = key
        if self.view_key != key:
            if refs.ref_status(v, self.w, self.h) != "ok":
                self.view_key = None
                return
            gref, gbla = refs.gpu(False)
            key = key[:-1] + (getattr(refs, "gpu_serial", 0),)
            fp = view_params(v, refs.ref_re, refs.ref_im, self.w, self.h, with_de=True,
                             with_interior=a.settings.interior, with_stripes=True,
                             use_bla=gbla is not None and gbla.levels > 0, julia=deep,
                             formula="mandelbrot" if julia else a.formula)
            f.reset()
            f.configure(fp, gref, gbla)
            f.program = None
            self.view_key = key
        if not f.complete:
            f.step(budget, budget * 0.5)
            return
        if self.done_key == key:
            return
        self.auto.update(f)
        for name in LOOKS:
            if name not in self.textures:
                t = a.ctx.texture((self.w, self.h), 3)
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.textures[name] = t
                self.fbos[name] = a.ctx.framebuffer(color_attachments=[t])
            cs = ColorSettings.from_look(name)
            a.engine.display.render(f, self.fbos[name], (0, 0, self.w, self.h), 1.0, cs,
                                    fallback=False, auto=self.auto, max_iter_show=v.max_iter)
        self.done_key = key


JULIA_HOME = 1.0          # log2 of the half height of the whole-Julia-set view (|z| <= 2 fits)
def julia_home(fb=(16, 9)) -> View:
    W, H = fb
    lr = JULIA_HOME
    if W > 0 and H > 0:
        lr = math.log2(max(1.25, 1.35 * H / max(W, 1)) * 1.5)
    return View(mpfr(0), mpfr(0), lr, 0.0, 1000)


class JuliaPreview:
    """A small live picture of the Julia set for the point under the cursor.

    Each point c of the Mandelbrot set has its Julia set (z -> z^2 + c with z0 varying):
    connected for c inside the set, dust outside, and near the boundary it looks like the
    Mandelbrot set around c.  Recomputed whenever c changes (a few ms: it is small)."""

    MAX_ITER = 1500

    def __init__(self, app: "App"):
        self.app = app
        self.field: Field | None = None
        self.tex = None
        self.fbo = None
        self.auto = AutoContrast(app.ctx)
        self.c = None
        self.key = None
        self.drawn = None
        self.view = View(mpfr(0), mpfr(0), 1.0, 0.0, self.MAX_ITER)

    def size(self):
        W, H = self.app.fb
        w = int(min(420, max(220, 0.2 * W)))
        aspect = min(0.8, max(0.55, H / max(W, 1)))
        return w, int(round(w * aspect))

    def _ensure(self):
        a = self.app
        w, h = self.size()
        if self.field is None or (self.field.W, self.field.H) != (w, h):
            for o in (self.field, self.fbo, self.tex):
                if o is not None:
                    o.release()
            self.field = Field(a.engine, w, h, False)
            self.tex = a.ctx.texture((w, h), 3)
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.fbo = a.ctx.framebuffer(color_attachments=[self.tex])
            self.fbo.clear(0.05, 0.05, 0.07)
            self.key = self.drawn = None
            self.view = julia_home((w, h))
            self.view.max_iter = self.MAX_ITER
        return self.field

    def update(self, c, budget: float):
        a = self.app
        if c is None:
            return
        f = self._ensure()
        cs = a.colors
        key = (c, cs.needs_de, cs.needs_stripes, cs.stripe_freq, a.settings.interior)
        if key != self.key:
            w, h = f.W, f.H
            fp = julia_params(self.view, w, h, double=False, with_de=cs.needs_de, with_interior=True,
                              with_stripes=cs.needs_stripes, stripe_freq=float(cs.stripe_freq))
            f.reset()
            f.configure(fp, None, None)
            a.engine.setup_julia(f, self.view, c, w, h, fp)
            self.key = key
            self.c = c
        if not f.complete:
            done = f.step(budget, budget * 0.5)
            if cs.mapping == 3 and (done or f.level_done <= 3):
                self.auto.update(f)
        dk = (self.key, f.done_count, a.colors_key)
        if dk != self.drawn:
            auto = self.auto if cs.mapping == 3 and self.auto.valid else None
            a.engine.display.render(f, self.fbo, (0, 0, f.W, f.H), 1.0, cs, fallback=True, auto=auto,
                                    max_iter_show=self.view.max_iter, background=(0.05, 0.05, 0.07))
            self.drawn = dk

    def release(self):
        for o in (self.field, self.fbo, self.tex):
            if o is not None:
                o.release()
        self.field = self.fbo = self.tex = None
        self.key = self.drawn = None


class App:
    def __init__(self, location=None, size=(1600, 900), selftest=False, selftest_dir="selftest_out"):
        self.size = size
        self.selftest_mode = bool(selftest)
        self.cpu_mode = software.active()        # Mesa llvmpipe: the engine runs on the processor
        self.on_cpu = self.cpu_mode              # ... also when the system chose a software driver
        self.settings = Settings()
        self.colors = ColorSettings.from_look(DEFAULT_LOOK)
        self.script_text = DEFAULT_SCRIPT
        self._load_config()
        self.manual_max_iter = None
        self.iter_boost = 1.0
        self.iter_floor = 1.0
        self.view = self.apply_auto_iter(View.home())
        self.history: list[View] = []
        self.future: list[View] = []
        self.sequence = Sequence()
        self.selected_kf = -1
        self.playing = False
        self.play_t = 0.0
        self.path_rec: PathRecorder | None = None
        self.live: LiveRecorder | None = None
        self.jobs: list[Job] = []
        self.messages: list[tuple[float, str]] = []
        self.ui_visible = True
        self.show_help = False
        self.library = Library()
        self.script_errors: list = []
        self.script_info = ""
        self.pending_location = location
        self.study_mrt = False           # motion study: second render target with the pixel sources
        self.src_tex = None
        self.dbg: dict = {}              # what the last frame did (motion study log)
        self.reset_log_total = 0
        self.ms_update = self.ms_render = 0.0
        if selftest in ("study", "study-quick"):
            from .study import make_study
            self.selftest = make_study(self, selftest_dir, quick=(selftest == "study-quick"))
        else:
            from .selftest import SelfTest
            self.selftest = SelfTest(self, selftest_dir, quick=(selftest == "quick")) if selftest else None
            if selftest:
                # the scripted session checks sample reuse at full resolution (a software GPU
                # would otherwise zoom at half resolution: 'automatic' motion resolution)
                self.settings.motion_scale = 1.0
        # runtime (GL)
        self.ctx = None
        self.engine = None
        self.fb = (0, 0)
        self.fb_scale = (1.0, 1.0)
        self.fields: dict = {}
        self.field: Field | None = None
        self.field_view: View | None = None
        self.field_fp = None
        self.field_scale = 1.0
        self.field_serial = 0
        self.field_active = False
        self.need_restart = True
        self.reset_log: list[str] = []
        self.moving_until = 0.0
        self.last_time = time.perf_counter()
        self.pacer = FramePacer()
        self.anim_now = self.last_time         # camera clock: paced frame steps (see FramePacer)
        self.anim_dt = 0.0
        self.fps = 0.0
        self.frame_ms = 0.0
        self.zoom_anim = 0.0
        self.zoom_vel = 0.0
        self.zoom_focus = None
        self.transition = None
        self.drag_last = None
        self.drag_accum = [0.0, 0.0]
        self.drag_vel = (0.0, 0.0)
        self.drag_t = 0.0
        self.inertia = None
        self.pan_accum = [0.0, 0.0]
        self.rot_last = None
        self.rot_drag = None
        self.right_moved = False
        self.key_down_t: dict = {}
        self.ref_anchor: View | None = None
        self.pending_ref = None      # (job id, request dict)
        self.stream_after = 0.5      # s: a long reference is drawn with as soon as this much is done
        self.pending_bla = None
        self.ref_progress = 0.0
        self.minibrot_job = None
        self.minibrot_result = None
        self.context_menu_c = None
        self.open_context_menu = False
        self.preview_tex = None
        self.edit_re = ""
        self.edit_im = ""
        self.edit_zoom = ""
        self.snap_view: View | None = None
        self.snap_q = 0.0
        self.snap_serial = -1
        # motion model: velocity, expected time to finish a field, picture kept in flight
        self.vel = (0.0, 0.0, 0.0, 0.0)          # d log2r/dt, d angle/dt, centre px/s (x, y)
        self._mv_prev: View | None = None
        self.field_T = 0.12
        self.field_T_full = 0.12                 # the same for a full-resolution picture (estimate)
        self.motion_level = 0                    # automatic motion resolution: 1/2**level
        self.gamepad = None                      # gamepad.Gamepad (made in post_init: needs GLFW)
        self.pad_prev = None                     # the controller's buttons last frame
        self.pad_name = ""
        self.pad_active = False
        self.field_t0 = 0.0
        self.field_fresh = False
        self.field_epoch = 0                     # +1 per picture of a new view (reset / zoom remap)
        self.boost_epoch = -1                    # epoch of the last automatic iteration raise
        self.boost_l2r = None                    # depth (log2r) where iter_boost / iter_floor were set
        self.iter_advice = None                  # (step, reason, octave stats) of the last decision
        self.flight_frames = 0
        self.anim_end: View | None = None
        self.uncovered_colour = (0.0, 0.0, 0.0)  # screen parts no computed picture covers yet
        # early inside detection costs 6-10 % per iteration: only while the view has inside parts
        self.interior_needed = True
        self.aa_count = 0
        self.aa_serial = -1
        self.aa_ctex = self.aa_cfbo = None
        self.aa_ctex_serial = -1
        self.colors_key = None
        self.auto_ready_serial = -1
        self.colour_tab_visible = False
        self.save_name = ""
        # Julia sets: c while exploring one (None: the Mandelbrot set); what to return to
        self.julia_c: tuple | None = None
        self.mandel_saved = None
        # the fractal: "mandelbrot" or "ship" (the Burning Ship); the view of the other one
        self.formula = "mandelbrot"
        self.formula_saved: dict = {}
        self.mouse = (0.0, 0.0, True)          # last mouse position, over the UI?
        self.jpreview = None
        self._sync_edit_fields()

    # ================================================================ config
    def _config_path(self) -> Path:
        # CPU mode keeps its own settings (fewer anti-aliasing passes, ...): switching back to
        # the graphics card finds the old ones unchanged
        return CONFIG_DIR / ("settings_cpu.json" if self.cpu_mode else "settings.json")

    def _load_config(self):
        if self.selftest_mode:
            return               # the scripted test runs with the defaults and leaves the user's config alone
        path = self._config_path()
        if self.cpu_mode and not path.exists():
            self.settings.aa_passes = 4              # CPU mode, first start: lighter defaults
        try:
            if not path.exists():
                path = CONFIG_DIR / "settings.json"
            if not path.exists() and (LEGACY_CONFIG_DIR / "settings.json").exists():
                path = LEGACY_CONFIG_DIR / "settings.json"
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version", 1) < 2:
                return               # v1 settings: start with the new defaults
            for k, v in d.get("settings", {}).items():
                if hasattr(self.settings, k):
                    setattr(self.settings, k, v)
            if d.get("version", 2) < 3 and self.settings.motion_scale == 1.0:
                self.settings.motion_scale = 0.0          # v3.9: 'automatic' (full unless heavy)
            for name, stops in d.get("custom_palettes", {}).items():
                set_custom(name, [tuple(s) for s in stops])
            self.colors = ColorSettings.from_dict(d.get("colors", {}))
            self.script_text = d.get("script", self.script_text)
            if self.cpu_mode and path != self._config_path():
                self.settings.aa_passes = min(self.settings.aa_passes, 4)
        except Exception:
            pass

    def _save_config(self):
        if self.selftest_mode:
            return
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            d = {"version": 3, "settings": asdict(self.settings), "colors": self.colors.to_dict(),
                 "custom_palettes": {k: [list(s) for s in v] for k, v in CUSTOM.items()},
                 "script": self.script_text, "last_view": self.view.to_dict()}
            self._config_path().write_text(json.dumps(d, indent=1), encoding="utf-8")
        except Exception:
            pass

    def notify(self, text: str):
        self.messages.append((time.perf_counter(), text))
        self.messages = self.messages[-8:]
        if sys.stdout is not None:
            try:
                print("[mandelscope]", text, flush=True)
            except Exception:
                pass

    # ============================================================== location
    def load_location(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.apply_location_dict(d)

    def apply_location_dict(self, d: dict, animate: bool = False):
        v = View.from_dict(d["view"] if "view" in d else d)
        self.place_title = d.get("title") or None       # shown by the HUD while near this place
        self.place_view = v.copy()
        if "colors" in d:
            self.colors = ColorSettings.from_dict({**self.colors.to_dict(), **d["colors"]})
        f = d.get("formula", "mandelbrot")
        if f != self.formula:
            self._adopt_iterations(v)
            self.set_formula(f, view=v)
            self.ref_anchor = v.copy()
            return
        jc = d.get("julia")
        if jc is not None:
            self._adopt_iterations(v)
            self.enter_julia(float(jc[0]), float(jc[1]), view=v)
            return
        if self.julia_c is not None:
            self.leave_julia()
            animate = False
        self._adopt_iterations(v)
        if animate:
            dz = abs(v.log2r - self.view.log2r) * 0.30103
            self.go_to(v, min(8.0, 0.6 + 0.12 * dz))
        else:
            self.set_view(v, push=True)
        self.ref_anchor = v.copy()

    def _adopt_iterations(self, v: View):
        """Keep the automatic iteration mode, but never use fewer iterations than `v` asks for."""
        self.set_boost(1.0, v, floor=1.0)
        if self.settings.auto_iter:
            base = max(1, self.auto_iter_for(v))
            if v.max_iter > base:
                b = v.max_iter / base
                self.set_boost(b, v, floor=b)           # "never fewer than v asks for" (near v)
            v.max_iter = self.auto_iter_for(v)
        else:
            self.manual_max_iter = v.max_iter

    def location_dict(self):
        d = {"view": self.view.to_dict(), "colors": self.colors.to_dict()}
        if self.julia_c is not None:
            d["julia"] = [self.julia_c[0], self.julia_c[1]]
        if self.formula != "mandelbrot":
            d["formula"] = self.formula
        return d

    def set_view(self, v: View, push: bool = False):
        if push:
            self.push_history()
        self.view = v
        self.transition = None
        self._sync_edit_fields()

    def push_history(self):
        if not self.history or self.history[-1] is not self.view:
            self.history.append(self.view.copy())
            self.future.clear()
            self.history = self.history[-500:]

    def _sync_edit_fields(self):
        p = max(self.view.precision, 64)
        digits = int(math.ceil(self.view.needed_precision(max(self.fb[1], 256)) * 0.30103)) + 2
        self.edit_re = format(mpfr(self.view.re, p), f".{max(digits, 17)}e")
        self.edit_im = format(mpfr(self.view.im, p), f".{max(digits, 17)}e")
        self.edit_zoom = f"{self.view.zoom_magnification_log10():.4f}"

    BOOST_KEEP = 2.0      # octaves of zooming out that keep a raised iteration limit

    def _decayed(self, b: float, v: View) -> float:
        """A raised boost belongs to the depth where it was set: zooming out more than
        BOOST_KEEP octaves beyond it halves it per octave (the v3.11 lag: the x44 of a deep
        minibrot stayed, and shallow views were computed with 100,000-240,000 iterations)."""
        if b <= 1.0 or self.boost_l2r is None:
            return b
        out = v.log2r - self.boost_l2r - self.BOOST_KEEP
        return max(1.0, b * 2.0 ** -out) if out > 0 else b

    def boost_for(self, v: View) -> float:
        return self._decayed(self.iter_boost, v)

    def floor_for(self, v: View) -> float:
        return self._decayed(getattr(self, "iter_floor", 1.0), v)

    def set_boost(self, b: float, v: View | None = None, floor: float | None = None):
        """Set the iteration boost (and floor) as valid at the depth of v (default: the view)."""
        self.iter_boost = b
        if floor is not None:
            self.iter_floor = floor
        self.boost_l2r = (v or self.view).log2r

    def auto_iter_for(self, v: View) -> int:
        s = self.settings
        dec = max(0.0, v.zoom_magnification_log10())
        return int((s.iter_base + s.iter_gain * dec) * self.boost_for(v))

    def apply_auto_iter(self, v: View) -> View:
        s = self.settings
        if s.auto_iter:
            v.max_iter = self.auto_iter_for(v)
        elif getattr(self, "manual_max_iter", None):
            v.max_iter = int(self.manual_max_iter)
        return v

    # ================================================================ GL init
    def post_init(self):
        self.ctx = moderngl.create_context(require=430)
        self.engine = GPUEngine(self.ctx)
        self.engine.perturb_program(False, True, True)
        self.refs = ReferenceCache(self.engine)
        self.jrefs = JuliaReferences(self.engine)   # deep Julia sets (perturbation, two references)
        self.auto = AutoContrast(self.ctx)
        self.previews = LookPreviews(self)
        self.jpreview = JuliaPreview(self)
        self.ref_worker = Worker()
        self.aux_worker = Worker()
        try:
            from .gamepad import Gamepad
            self.gamepad = Gamepad(mappings=CONFIG_DIR / "gamecontrollerdb.txt")
            if self.gamepad.lib is not None and not self.gamepad.same_instance_as(hello_imgui.get_glfw_window_address()):
                print("controller: the GLFW library found is not the window's", flush=True)
        except Exception as e:
            print(f"controller support off: {e}", flush=True)
            self.gamepad = None
        self.video_worker = None
        self.screen_tex = self.prev_tex = self.accum_tex = None
        self.scratch_tex = self.scratch_fbo = None
        self.thumb_cache: dict = {}
        self.strip_cache: dict = {}
        self.notify(f"GPU: {self.engine.renderer_name}  (OpenGL {self.ctx.info['GL_VERSION']})")
        if self.cpu_mode:
            if "llvmpipe" in self.engine.renderer_name.lower():
                self.notify(f"CPU mode - {software.threads()} threads. A graphics card with OpenGL 4.3 is "
                            "many times faster.")
            else:
                self.notify(f"CPU mode was asked for, but OpenGL runs on {self.engine.renderer_name}")
        self.on_cpu = self.cpu_mode or software.is_software_renderer(self.engine.renderer_name)
        if not self.engine.has_fp64:
            self.notify("fp64 not supported - precise mode disabled")
        if self.pending_location:
            try:
                p = str(self.pending_location)
                if p.lower().endswith(".png"):
                    d = location_from_png(p)
                    if d:
                        self.apply_location_dict(d)
                else:
                    self.load_location(p)
            except Exception as e:
                self.notify(f"cannot open {self.pending_location}: {e}")

    def setup_imgui_config(self):
        io = imgui.get_io()
        io.config_flags &= ~imgui.ConfigFlags_.nav_enable_keyboard.value
        io.config_windows_move_from_title_bar_only = True
        # a Python exception in the middle of the UI code must not kill the app:
        # let Dear ImGui close the windows that were left open instead of asserting
        io.config_error_recovery = True
        io.config_error_recovery_enable_assert = False
        io.config_error_recovery_enable_tooltip = False
        io.config_error_recovery_enable_debug_log = True

    def before_exit(self):
        try:
            if self.live:
                self.live.close()
            for j in self.jobs:
                if j.gen is not None:
                    try:
                        j.gen.close()
                    except Exception:
                        pass
            self._save_config()
        finally:
            for w in (getattr(self, "ref_worker", None), getattr(self, "aux_worker", None),
                      getattr(self, "video_worker", None)):
                if w is not None:
                    w.close()

    def _ensure_screen(self, W, H):
        if self.screen_tex is not None and self.screen_tex.size == (W, H):
            return
        for t in ("screen_fbo", "screen_tex", "prev_fbo", "prev_tex", "scratch_fbo", "scratch_tex",
                  "accum_fbo", "accum_tex", "src_tex"):
            o = getattr(self, t, None)
            if o is not None:
                o.release()
        self.screen_tex = self.ctx.texture((W, H), 3)
        self.screen_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        if self.study_mrt:
            self.src_tex = self.ctx.texture((W, H), 1)        # source id of every pixel (study)
            self.screen_fbo = self.ctx.framebuffer(color_attachments=[self.screen_tex, self.src_tex])
        else:
            self.screen_fbo = self.ctx.framebuffer(color_attachments=[self.screen_tex])
        # snapshots keep the per-pixel quality in alpha (half floats)
        self.prev_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.prev_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.prev_fbo = self.ctx.framebuffer(color_attachments=[self.prev_tex])
        self.scratch_tex = self.ctx.texture((W, H), 4, dtype="f2")
        self.scratch_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scratch_fbo = self.ctx.framebuffer(color_attachments=[self.scratch_tex])
        self.accum_tex = self.ctx.texture((W, H), 4, dtype="f4")
        self.accum_fbo = self.ctx.framebuffer(color_attachments=[self.accum_tex])
        self.screen_fbo.clear(0, 0, 0)
        self.prev_fbo.clear(0, 0, 0, 0)
        self.snap_view = None
        self.need_restart = True
        self.aa_count = 0

    # =============================================================== fields
    def get_field(self, W, H, double, kind="main") -> Field:
        key = (W, H, double, kind)
        f = self.fields.get(key)
        if f is None:
            # keep memory bounded: at most 2 main + 1 aa field
            same_kind = [k for k in self.fields if k[3] == kind]
            limit = 2 if kind == "main" else 1
            while len(same_kind) >= limit:
                k = same_kind.pop(0)
                old = self.fields.pop(k)
                if old is self.field:
                    self.field = None
                old.release()
            f = Field(self.engine, W, H, double)
            self.fields[key] = f
        return f

    def is_moving(self) -> bool:
        return (time.perf_counter() < self.moving_until or self.playing or self.transition is not None
                or abs(self.zoom_anim) > 1e-4 or self.inertia is not None or self.drag_last is not None)

    def mark_moving(self, dur=0.25):
        self.moving_until = time.perf_counter() + dur

    # ============================================================== input
    def _read_pad(self) -> PadInput | None:
        g, s = self.gamepad, self.settings
        if g is None or not s.pad_enabled:
            self.pad_active = False
            return None
        st = g.poll()
        if g.name != self.pad_name:
            if g.name:
                self.notify(f"controller connected: {g.name}  (F1 lists the buttons)")
            elif self.pad_name:
                self.notify("controller disconnected")
            self.pad_name = g.name
        pi = pad_controls(st, self.pad_prev, s.pad_deadzone, s.pad_invert_zoom)
        self.pad_prev = st.buttons if st is not None else None
        self.pad_active = pi.active
        return pi if st is not None else None

    def scale_iter(self, k: float):
        """More (k > 1) or fewer iterations, chosen by hand."""
        if self.settings.auto_iter:
            b = max(0.125, min(1024.0, self.boost_for(self.view) * k))
            self.set_boost(b, floor=b)              # chosen by hand: not lowered automatically (here)
            self.view = self.apply_auto_iter(self.view.copy())
        else:
            nv = self.view.copy()
            nv.max_iter = max(100, int(nv.max_iter * k))
            self.manual_max_iter = nv.max_iter
            self.view = nv

    def cycle_look(self, d: int):
        names = list(LOOKS)
        i = names.index(self.colors.look) if self.colors.look in names else -1
        name = names[(i + d) % len(names)]
        self.colors = ColorSettings.from_look(name, self.colors)
        self.notify(f"look: {name}")

    def _handle_pad(self, pad: PadInput, v: View, dt: float):
        """Game controller: sticks and triggers move the camera, buttons do the common actions.
        Returns (view, moved)."""
        W, H = self.fb
        s = self.settings
        moved = False
        mult = (3.0 if pad.fast else (0.3 if pad.fine else 1.0)) * s.pad_speed
        if pad.pan != (0.0, 0.0):
            step = 1.4 * s.pan_speed * mult * dt
            v, did = self._apply_pan(v, -pad.pan[0] * step, -pad.pan[1] * step, self.pan_accum)
            moved |= did
            self.stop_inertia()
        if pad.zoom:
            v = self.apply_auto_iter(v.zoom_at(W / 2.0, H / 2.0, pad.zoom * 1.25 * s.zoom_speed * mult * dt, W, H))
            self.ref_anchor = View(v.re, v.im, v.log2r)
            self.zoom_anim = 0.0
            self.zoom_vel = 0.0
            moved = True
        if pad.rot:
            v = v.rotate_by(pad.rot * dt * math.radians(90.0) * mult)
            moved = True
        if moved:
            self.transition = None               # taking over the camera, like a mouse click
            self.playing = False
        p = pad.pressed
        if "cross" in p:
            self.find_minibrot_at(v.re, v.im)
        if "circle" in p:
            self.back()
            v, moved = self.view, False
        if "square" in p:
            if self.julia_c is not None:
                self.leave_julia()
                v, moved = self.view, False
            else:
                self.settings.julia_preview = not self.settings.julia_preview
        if "triangle" in p:
            self.add_keyframe()
        if "left" in p or "right" in p:
            self.cycle_look(1 if "right" in p else -1)
        if "up" in p or "down" in p:
            self.view = v
            self.scale_iter(2.0 if "up" in p else 0.5)
            v = self.view
            self.notify(f"iterations: {v.max_iter:,}")
        if "l3" in p:
            self.go_home()
            v, moved = self.view, False
        if "r3" in p:
            self.rotate_to(0.0)
        if "options" in p:
            self.ui_visible = not self.ui_visible
        if "create" in p:
            self.start_still(quick=True)
        if "ps" in p:
            self.show_help = not self.show_help
        return v, moved

    def handle_input(self, inp: InputSnapshot, dt: float):
        W, H = self.fb
        s = self.settings
        mx, my = inp.mouse
        self.mouse = (mx, my, bool(inp.ui_mouse))
        moved = False
        v = self.view
        now = time.perf_counter()
        mult = 3.0 if inp.shift else (0.3 if inp.ctrl else 1.0)
        # ---------------- mouse on the fractal
        if not inp.ui_mouse:
            if inp.wheel != 0.0:
                if inp.shift:
                    self.rotate_by(math.radians(5.0) * inp.wheel, animate=False)
                    v = self.view
                else:
                    step = 0.15 if inp.ctrl else s.wheel_step
                    self.zoom_anim += inp.wheel * step
                    self.zoom_focus = (mx, my)
                self.stop_inertia()
            if inp.double:
                self.zoom_to_point(mx, my, 2.0)
                v = self.view
            if inp.clicked[0]:
                self.drag_last = (mx, my)
                self.drag_accum = [0.0, 0.0]
                self.drag_vel = (0.0, 0.0)
                self.drag_t = now
                self.stop_inertia()
                self.transition = None
            if inp.clicked[1]:
                self.rot_last = (mx, my)
                self.rot_drag = (v.angle, 0.0)
                self.right_moved = False
                self.stop_inertia()
            if inp.clicked[2]:
                self.centre_on_point(mx, my)
                v = self.view
        # ---------------- left drag = pan (integer pixels -> sample reuse)
        if inp.down[0] and self.drag_last is not None:
            dx, dy = mx - self.drag_last[0], my - self.drag_last[1]
            self.drag_last = (mx, my)
            if dt > 0 and (dx or dy):
                a = min(1.0, dt / 0.06)
                self.drag_vel = (self.drag_vel[0] * (1 - a) + dx / dt * a, self.drag_vel[1] * (1 - a) + dy / dt * a)
                self.drag_t = now
            v, did = self._apply_pan(v, dx, dy, self.drag_accum)
            moved |= did
        if not inp.down[0] and self.drag_last is not None:
            self.drag_last = None
            vx, vy = self.drag_vel
            if s.inertia and now - self.drag_t < 0.08 and math.hypot(vx, vy) > 80:
                self.inertia = [vx, vy]
            self.push_history()
        # ---------------- right drag = rotate around the centre (Shift snaps to 15 degrees)
        if inp.down[1] and self.rot_last is not None:
            ax = math.atan2(self.rot_last[1] - H / 2, self.rot_last[0] - W / 2)
            bx = math.atan2(my - H / 2, mx - W / 2)
            da = (bx - ax + math.pi) % (2 * math.pi) - math.pi
            if math.hypot(mx - W / 2, my - H / 2) > 10 and abs(da) > 1e-5:
                a0, acc = self.rot_drag
                acc += da            # the picture follows the mouse (larger angle = clockwise on screen)
                self.rot_drag = (a0, acc)
                target = a0 + acc
                if inp.shift:
                    st = math.radians(15.0)
                    target = round(target / st) * st
                if abs(target - v.angle) > 1e-9:
                    v = v.copy()
                    v.angle = target
                    moved = True
                if abs(mx - self.rot_last[0]) + abs(my - self.rot_last[1]) > 2:
                    self.right_moved = True
            self.rot_last = (mx, my)
        if inp.released[1] and self.rot_last is not None:
            if not self.right_moved and not inp.ui_mouse:
                cre, cim = v.screen_to_c(mx, my, W, H)
                self.context_menu_c = (cre, cim, mx, my)
                self.open_context_menu = True
            elif self.right_moved:
                self.push_history()
            self.rot_last = None
        # ---------------- keyboard
        if not inp.typing:
            k, p = inp.keys, inp.pressed
            for name in p:
                self.key_down_t[name] = now
            held = lambda n: n in k and now - self.key_down_t.get(n, now) > 0.3
            # zoom at the centre: + / - (tap = 2x, hold = continuous)
            zin = ("plus" in p) or ("kp_plus" in p) or ("pageup" in p)
            zout = ("minus" in p) or ("kp_minus" in p) or ("pagedown" in p)
            if zin or zout:
                self.zoom_anim += (1.0 if zin else -1.0) * (0.25 if inp.ctrl else 1.0)
                self.zoom_focus = (W / 2.0, H / 2.0)
                self.stop_inertia()
            cont = 0.0
            if held("plus") or held("kp_plus") or held("pageup"):
                cont += 1.0
            if held("minus") or held("kp_minus") or held("pagedown"):
                cont -= 1.0
            if cont:
                v = self.apply_auto_iter(v.zoom_at(W / 2.0, H / 2.0, cont * s.zoom_speed * mult * dt, W, H))
                self.ref_anchor = View(v.re, v.im, v.log2r)
                self.zoom_anim = 0.0
                self.zoom_vel = 0.0
                moved = True
            # zoom at the cursor: Z / X (tap = 2x, hold = continuous)
            if "z" in p or "x" in p:
                self.zoom_anim += (1.0 if "z" in p else -1.0) * (2.0 if inp.shift else (0.25 if inp.ctrl else 1.0))
                self.zoom_focus = (mx, my)
            oct_ = 0.0
            if held("z"):
                oct_ += s.zoom_speed * mult * dt
            if held("x"):
                oct_ -= s.zoom_speed * mult * dt
            if oct_:
                v = self.apply_auto_iter(v.zoom_at(mx, my, oct_, W, H))
                self.ref_anchor = View(*v.screen_to_c(mx, my, W, H), v.log2r)
                self.zoom_anim = 0.0
                self.zoom_vel = 0.0
                moved = True
            # pan
            pdx = (("d" in k) or ("right" in k)) - (("a" in k) or ("left" in k))
            pdy = (("s" in k) or ("down" in k)) - (("w" in k) or ("up" in k))
            if pdx or pdy:
                step = s.pan_speed * mult * dt
                v, did = self._apply_pan(v, -pdx * step, -pdy * step, self.pan_accum)
                moved |= did
                self.stop_inertia()
            # rotation
            rot = ("e" in k) - ("q" in k)
            if rot:
                v = v.rotate_by(rot * dt * math.radians(60.0) * mult)
                moved = True
            if "lbracket" in p:          # [ = rotate the picture 90 degrees counter-clockwise
                self.rotate_by(math.radians(-90.0))
            if "rbracket" in p:          # ] = 90 degrees clockwise
                self.rotate_by(math.radians(90.0))
            if "zero" in p:
                self.rotate_to(0.0)
            if "h" in p or "home" in p:
                self.go_home()
            if "backspace" in p:
                self.forward() if inp.shift else self.back()
            if "k" in p:
                self.add_keyframe()
            if "space" in p:
                self.toggle_play()
            if "r" in p and not inp.ctrl:
                self.toggle_live()
            if "p" in p:
                self.toggle_path_recording()
            if "m" in p:
                cre, cim = v.screen_to_c(mx, my, W, H)
                self.find_minibrot_at(cre, cim)
            if "b" in p:
                # B: the Burning Ship <-> the Mandelbrot set
                self.set_formula("mandelbrot" if self.formula == "ship" else "ship")
                v, moved = self.view, False
            if "j" in p and self.formula != "mandelbrot":
                self.notify("Julia sets: Mandelbrot set only (B: back to it)")
            elif "j" in p:
                # J: Julia preview on / off; Shift+J: explore the Julia set of the point under the
                # cursor; J while exploring one: back to the Mandelbrot set
                if self.julia_c is not None:
                    self.leave_julia()
                    v, moved = self.view, False
                elif inp.shift:
                    cre, cim = v.screen_to_c(mx, my, W, H) if not inp.ui_mouse else (v.re, v.im)
                    self.enter_julia(cre, cim)
                    v, moved = self.view, False
                else:
                    self.settings.julia_preview = not self.settings.julia_preview
            if "t" in p:
                self.settings.show_timeline = not self.settings.show_timeline
            if "tab" in p:
                self.ui_visible = not self.ui_visible
            if "f1" in p:
                self.show_help = not self.show_help
            if "f12" in p:
                self.start_still(quick=True)
            if "escape" in p:
                self.show_help = False
                self.stop_inertia()
                self.zoom_anim = 0.0
                self.zoom_vel = 0.0
                self.transition = None
                if self.playing:
                    self.playing = False
            if inp.ctrl and "c" in p:
                imgui.set_clipboard_text(json.dumps(self.location_dict(), indent=1))
                self.notify("location copied")
            if inp.ctrl and "v" in p:
                self.paste_location()
        # ---------------- game controller
        if inp.pad is not None:
            self.view = v
            v, did = self._handle_pad(inp.pad, v, dt)
            moved |= did
        # ---------------- animations: inertia, wheel zoom, transitions
        if self.inertia is not None:
            vx, vy = self.inertia
            v, did = self._apply_pan(v, vx * dt, vy * dt, self.pan_accum)
            moved |= did
            f = math.exp(-dt / max(s.inertia_time, 1e-3))
            self.inertia = [vx * f, vy * f]
            if math.hypot(*self.inertia) < 15:
                self.inertia = None
                self.push_history()
        if abs(self.zoom_anim) > 1e-4 and self.zoom_focus is not None:
            if s.wheel_spring:
                # critically damped spring towards the zoom target: every wheel notch moves the
                # target, the speed changes smoothly (v3.6: the speed jumped up at every notch
                # and decayed - a sawtooth when the wheel turns)
                rem0, vel0 = self.zoom_anim, self.zoom_vel
                rem1, vel1 = spring_step(rem0, vel0, dt, ZOOM_OMEGA)
                if (abs(rem1) < 0.002 and abs(vel1) < 0.05) or rem1 * rem0 < 0.0:
                    rem1, vel1 = 0.0, 0.0
                step = rem0 - rem1
                self.zoom_vel = vel1
            else:
                step = self.zoom_anim * (1.0 - math.exp(-dt * 12.0))
                if abs(self.zoom_anim) < 0.005:
                    step = self.zoom_anim
            self.zoom_anim -= step
            fx, fy = self.zoom_focus
            v = self.apply_auto_iter(v.zoom_at(fx, fy, step, W, H))
            self.ref_anchor = View(*v.screen_to_c(fx, fy, W, H), v.log2r)
            moved = True
            if abs(self.zoom_anim) <= 1e-4:
                e = self.anim_end
                if e is not None and abs(e.log2r - v.log2r) < 1e-6:
                    ex, ey = v.delta_in_pixels(e, H)
                    if math.hypot(ex, ey) < 1.0:
                        v = self.apply_auto_iter(e.copy())   # land on the grid of the reused samples
                self.anim_end = None
                self.push_history()
        if self.transition is not None:
            tr = self.transition
            u = min(1.0, (self.anim_now - tr["t0"]) / tr["dur"])
            v = interpolate_views(tr["v0"], tr["v1"], ease("smooth", u))
            if not self.settings.auto_iter:
                v.max_iter = tr["v1"].max_iter
            else:
                v = self.apply_auto_iter(v)
            moved = True
            if u >= 1.0:
                v = tr["v1"].copy()
                v.angle = math.remainder(v.angle, 2 * math.pi)     # 0 deg, not 360 deg, after turning back
                if self.settings.auto_iter:
                    v = self.apply_auto_iter(v)
                self.transition = None
        if moved:
            self.mark_moving()
            self.view = v
            self._sync_edit_fields()
            if self.playing and (inp.down[0] or inp.down[1] or inp.wheel):
                self.playing = False

    def _apply_pan(self, v: View, dx: float, dy: float, acc: list):
        """Move the content by (dx, dy) screen pixels using whole field samples (reusable)."""
        sc = self.field_scale or 1.0
        acc[0] += dx
        acc[1] += dy
        ix = math.trunc(acc[0] * sc) / sc
        iy = math.trunc(acc[1] * sc) / sc
        if ix or iy:
            v = v.offset(-ix, -iy, self.fb[1])
            acc[0] -= ix
            acc[1] -= iy
            return v, True
        return v, False

    def stop_inertia(self):
        self.inertia = None

    # --------------------------------------------------- navigation commands
    def go_to(self, v1: View, dur: float = 0.5):
        self.push_history()
        self.transition = {"v0": self.view.copy(), "v1": v1.copy(), "t0": self.anim_now, "dur": dur}
        self.ref_anchor = View(v1.re, v1.im, v1.log2r)

    def zoom_to_point(self, sx, sy, octaves: float):
        W, H = self.fb
        cre, cim = self.view.screen_to_c(sx, sy, W, H)
        v1 = self.view.copy()
        v1.re, v1.im = cre, cim
        v1.log2r = self.view.log2r - octaves
        v1.ensure_precision(H)
        self.go_to(self.apply_auto_iter(v1), 0.45)

    def centre_on_point(self, sx, sy):
        W, H = self.fb
        cre, cim = self.view.screen_to_c(sx, sy, W, H)
        v1 = self.view.copy()
        v1.re, v1.im = cre, cim
        self.go_to(v1, 0.35)

    def rotate_by(self, dang: float, animate: bool = True):
        base = self.transition["v1"] if self.transition is not None else self.view
        v1 = base.copy()
        v1.angle = base.angle + dang
        if animate:
            self.go_to(v1, 0.3)
        else:
            self.view = v1
            self.mark_moving()

    def rotate_to(self, angle: float):
        a = self.view.angle
        d = (angle - a + math.pi) % (2 * math.pi) - math.pi
        self.rotate_by(d)

    def home_view(self) -> View:
        """The whole set, fitted to the window shape (also tall / narrow windows)."""
        if self.julia_c is not None:
            return julia_home(self.fb)
        return formula_home(self.formula, self.fb)

    def go_home(self):
        self.iter_boost = 1.0
        self.iter_floor = 1.0
        self.go_to(self.apply_auto_iter(self.home_view()), 0.6)

    # ============================================================ Julia sets
    def _mode_switched(self):
        """Mandelbrot <-> Julia: nothing of the old picture carries over."""
        self.transition = None
        self.zoom_anim = 0.0
        self.zoom_vel = 0.0
        self.anim_end = None
        self.inertia = None
        self.playing = False
        self.snap_view = None
        self.need_restart = True
        self.aa_count = 0
        self.interior_needed = True
        self.mark_moving(0.05)

    def delta_precision(self, setting: int | None = None, formula: str | None = None) -> int:
        """The kernels' delta precision for a precision setting: 0 float32, 1 float64, 2 float-float.
        Without float64 on the GPU, float-float stands in for it; the Burning Ship has no
        float-float kernel and uses float64 instead."""
        p = self.settings.precision if setting is None else int(setting)
        f = (self.formula if self.julia_c is None else "mandelbrot") if formula is None else formula
        has64 = self.engine is not None and self.engine.has_fp64
        if f == "ship":
            return 1 if (p != 0 and has64) else 0
        if p == 1 and not has64:
            return PREC_FF
        return p if p in (0, 1, PREC_FF) else 0

    # ============================================================ formulas
    def set_formula(self, f: str, view: View | None = None):
        """Switch between the Mandelbrot set and the Burning Ship.  Each keeps its own view and
        history; `view` (from a location) replaces the remembered one."""
        if f not in FORMULAS:
            f = "mandelbrot"
        if f == self.formula and view is None:
            return
        if self.julia_c is not None:
            self.leave_julia()
        if f != self.formula:
            self.formula_saved[self.formula] = dict(view=self.view.copy(), history=self.history,
                                                    future=self.future, iter_boost=self.boost_for(self.view),
                                                    ref_anchor=self.ref_anchor)
            st = self.formula_saved.pop(f, None)
            self.formula = f
            self.refs.formula = f
            self.pending_ref = None
            self.pending_bla = None
            self.minibrot_result = None
            self._mode_switched()
            if st is not None and view is None:
                self.history, self.future = st["history"], st["future"]
                self.set_boost(st["iter_boost"], st["view"], floor=1.0)
                self.ref_anchor = st["ref_anchor"]
                self.view = self.apply_auto_iter(st["view"])
            else:
                self.history, self.future = [], []
                self.iter_boost = 1.0
                self.iter_floor = 1.0
                self.ref_anchor = None
                self.view = self.apply_auto_iter(view.copy() if view is not None else self.home_view())
        else:
            self.set_view(view.copy(), push=True)
        self._sync_edit_fields()
        self.notify(FORMULA_NAMES[f])

    def enter_julia(self, cre, cim, view: View | None = None):
        """Explore the Julia set of c (the view starts with the whole set)."""
        if self.formula != "mandelbrot":
            self.notify("Julia sets: Mandelbrot set only")
            return
        c = (float(cre), float(cim))
        if self.julia_c is None:
            self.mandel_saved = dict(view=self.view.copy(), history=self.history, future=self.future,
                                     iter_boost=self.boost_for(self.view), ref_anchor=self.ref_anchor)
            self.history, self.future = [], []
        self.julia_c = c
        self._mode_switched()
        self.iter_boost = 1.0
        self.iter_floor = 1.0
        self.view = self.apply_auto_iter(view.copy() if view is not None else self.home_view())
        self._sync_edit_fields()
        self.notify(f"Julia set of c = {c[0]:.10g} {'+' if c[1] >= 0 else '-'} {abs(c[1]):.10g}i"
                    f"   (J: back to the Mandelbrot set)")

    def leave_julia(self):
        if self.julia_c is None:
            return
        self.julia_c = None
        st, self.mandel_saved = self.mandel_saved, None
        self._mode_switched()
        if st is not None:
            self.view = st["view"]
            self.history, self.future = st["history"], st["future"]
            self.set_boost(st["iter_boost"], st["view"], floor=1.0)
            self.ref_anchor = st["ref_anchor"]
        else:
            self.iter_boost = 1.0
            self.iter_floor = 1.0
            self.view = self.apply_auto_iter(self.home_view())
        for k in [k for k in self.fields if k[3] == "julia"]:
            f = self.fields.pop(k)
            if f is self.field:
                self.field = None
            f.release()
        self._sync_edit_fields()
        self.notify("back to the Mandelbrot set")

    def julia_deep(self, v: View | None = None, Hf: int | None = None) -> bool:
        """A Julia view too deep for the direct float32 kernel: perturbation with two references
        (any depth, like the Mandelbrot set)."""
        if self.julia_c is None:
            return False
        v = self.view if v is None else v
        Hf = (self.fb[1] or 1080) if Hf is None else Hf
        return v.log2r < julia_min_log2r(False, Hf) + 1.0

    def active_refs(self):
        """The reference cache drawing the current view (Mandelbrot or deep Julia)."""
        if self.julia_c is not None:
            self.jrefs.select(self.julia_c)
            return self.jrefs
        self.refs.formula = self.formula
        return self.refs

    def preview_c(self):
        """c for the Julia preview: the point under the cursor (or the centre)."""
        W, H = self.fb
        mx, my, ui = self.mouse
        if ui or not (0 <= mx < W and 0 <= my < H):
            if self.jpreview is not None and self.jpreview.c is not None:
                return self.jpreview.c
            return (float(self.view.re), float(self.view.im))
        cre, cim = self.view.screen_to_c(mx, my, W, H)
        return (float(cre), float(cim))

    def back(self):
        if self.history:
            self.future.append(self.view.copy())
            self.view = self.history.pop()
            self.transition = None
            self._sync_edit_fields()

    def forward(self):
        if self.future:
            self.history.append(self.view.copy())
            self.view = self.future.pop()
            self.transition = None
            self._sync_edit_fields()

    def paste_location(self):
        try:
            d = json.loads(imgui.get_clipboard_text() or "")
            self.apply_location_dict(d, animate=False)
            self.notify("location pasted")
        except Exception:
            self.notify("clipboard has no location")

    # ========================================================= reference mgmt
    def _poll_workers(self):
        for msg in self.ref_worker.poll():
            kind = msg["kind"]
            if self.pending_ref and msg["id"] == self.pending_ref[0]:
                if kind == "progress":
                    self.ref_progress = msg["value"]
                elif kind == "partial":
                    # the orbit so far (a long one is still being computed): draw with it, the
                    # iterations capped at its length; the rest continues the same samples
                    ref = msg["ref"]
                    old = self.refs.ref
                    same = old is not None and _same_point(old, ref)
                    if not same or ref.length > old.length:
                        self.refs.set_reference(ref, msg["bla"])
                        if not same:
                            self.need_restart = True
                elif kind == "reference":
                    ref = msg["ref"]
                    old = self.refs.ref
                    self.refs.set_reference(ref, msg["bla"])
                    self.pending_ref = None
                    if not (old is not None and _same_point(old, ref)):
                        self.need_restart = True
                    # else: the same reference point, only longer / more precise - the orbit
                    # prefix is identical, so every computed sample stays valid
                    if DEBUG:
                        self.notify(f"reference: {ref.length:,} iters, {ref.prec} bits ({ref.elapsed:.2f}s)")
                elif kind == "julia_reference":
                    self.pending_ref = None
                    jr = self.jrefs
                    old = (jr.ref_re, jr.ref_im)
                    if self.julia_c is not None and jr.select(self.julia_c) == (msg["ref_b"].re, msg["ref_b"].im):
                        jr.set_references(msg["ref_a"], msg["ref_b"], msg["bla_a"], msg["bla_b"], msg["log2_eps"])
                        if old != (jr.ref_re, jr.ref_im):
                            self.need_restart = True
                elif kind in ("error", "cancelled"):
                    if kind == "error":
                        self.notify("reference error: " + msg.get("error", "")[-300:])
                    self.pending_ref = None
            elif self.pending_bla and msg["id"] == self.pending_bla[0]:
                if kind == "bla":
                    if self.refs.ref is not None and msg["bla"].ref_id == self.refs.ref.ref_id:
                        self.refs.set_bla(msg["bla"])     # computed samples stay valid
                    self.pending_bla = None
                elif kind == "error":
                    self.pending_bla = None
        for msg in self.aux_worker.poll():
            if self.minibrot_job and msg["id"] == self.minibrot_job:
                if msg["kind"] == "minibrot":
                    self.minibrot_job = None
                    self.minibrot_result = msg["result"]
                    what = "mini ship" if self.formula == "ship" else "minibrot"
                    if msg["result"] is None:
                        self.notify(f"no {what} found there")
                    else:
                        r = msg["result"]
                        self.notify(f"{what}: period {r['period']:,}, size 1e{r['log2_size'] * 0.30103:.1f}"
                                    " - press 'Go to minibrot'")
                elif msg["kind"] == "error":
                    self.minibrot_job = None
                    self.notify("minibrot search failed")

    def _auto_reference_point(self, v: View) -> View | None:
        """A long-lived reference: a non-escaping sample (or the deepest one) of the
        current field near the zoom focus.  Uses a GPU reduction (small read-back)."""
        f, fv = self.field, self.field_view
        if (f is None or f.params is None or fv is None or f.progress < 0.25
                or abs(fv.log2r - v.log2r) > 3 or not self.field_active):
            return None
        try:
            cand = f.reference_candidates()
        except Exception:
            return None
        H, W = f.H, f.W
        by, bx = cand.shape[:2]
        if self.ref_anchor is not None:
            ax, ay = fv.delta_in_pixels(self.ref_anchor, self.fb[1])
            ax = (ax + self.fb[0] / 2.0) * W / self.fb[0]
            ay = (ay + self.fb[1] / 2.0) * H / self.fb[1]
        else:
            ax, ay = W / 2.0, H / 2.0
        gy, gx = np.mgrid[0:by, 0:bx]
        dist = np.hypot(gx * 8 + 4 - ax, gy * 8 + 4 - ay)
        near = dist < 0.25 * H
        n = cand[..., 0].astype(np.int64)
        bounded = near & (n >= 0xFFFFFFFE)
        if bounded.any():
            d = np.where(bounded, dist, np.inf)
            iy, ix = np.unravel_index(np.argmin(d), d.shape)
        else:
            esc = np.where(near & (n > 0), n, -1)
            if esc.max() <= 0:
                return None
            iy, ix = np.unravel_index(np.argmax(esc), esc.shape)
        idx = int(cand[iy, ix, 1])
        if idx == 0xFFFFFFFF:
            return None
        sx, sy = idx % W, idx // W
        k = self.fb[0] / W
        cre, cim = fv.screen_to_c((sx + 0.5) * k, (sy + 0.5) * k, self.fb[0], self.fb[1])
        cand_v = View(cre, cim, v.log2r)
        dx, dy = v.delta_in_pixels(cand_v, self.fb[1])
        if math.hypot(dx, dy) > 0.6 * max(self.fb):
            return None
        return cand_v

    def _request_reference(self, view: View, Wf: int, Hf: int, anchor: View | None):
        # head-room: extra precision and iterations so zooming on does not need a new reference
        vv = view.copy()
        vv.max_iter = int(view.max_iter * 1.5) + 1000
        refs = self.active_refs()
        req = refs.reference_request(vv, Wf, Hf, anchor, extra_prec=96)
        mode = refs.c if self.julia_c is not None else (None if self.formula == "mandelbrot" else self.formula)
        if self.pending_ref is not None:
            old = self.pending_ref[1]
            if (old.get("mode") == mode and old["prec"] >= view.needed_precision(Hf)
                    and old["max_iter"] >= view.max_iter):
                ov = View(mpfr(old["re"], old["prec"]), mpfr(old["im"], old["prec"]), view.log2r)
                dx, dy = view.delta_in_pixels(ov, Hf)
                if math.hypot(dx, dy) < 1.5 * max(Wf, Hf):
                    return      # the pending reference will do
        eps = self.settings.eps_f64 if self.delta_precision() else self.settings.eps_f32
        if self.julia_c is not None:
            # deep Julia set: the orbit of the view's point and the critical orbit of c
            jid = self.ref_worker.submit("julia_reference", log2_eps=eps, **req)
        elif self.formula == "ship":
            jid = self.ref_worker.submit("reference", log2_cmax=None, log2_eps=eps, stream=True,
                                         partial_after=self.stream_after, **req)
        else:
            fp = view_params(view, mpfr(req["re"], req["prec"]), mpfr(req["im"], req["prec"]), Wf, Hf)
            jid = self.ref_worker.submit("reference", log2_cmax=fp.log2_cmax + 1.0, log2_eps=eps, stream=True,
                                         partial_after=self.stream_after, **req)
        req["mode"] = mode
        self.pending_ref = (jid, req)
        self.ref_progress = 0.0

    AUTO_HALF_ABOVE = 0.20      # s for a full-resolution picture: zoom / turn at half resolution
    AUTO_FULL_BELOW = 0.10      # ... and back to full resolution (hysteresis)

    def _auto_motion_scale(self) -> float:
        """'While moving: automatic'.  Zooms and turns restart the picture all the time; when a
        full-resolution one takes longer than ~12 frames, they are computed at half resolution
        (4x fewer samples: finished pictures instead of ever more stretched old ones).  Pans
        keep full resolution - they reuse almost every sample."""
        vz, va, _, _ = self.vel
        zooming = (self.playing or self.transition is not None or abs(self.zoom_anim) > 1e-4
                   or abs(vz) > 0.05 or abs(va) > 0.02)
        scale, self.motion_level = auto_motion_level(zooming, self.field_T_full, self.motion_level,
                                                     2 if self.on_cpu else 1,
                                                     self.AUTO_HALF_ABOVE, self.AUTO_FULL_BELOW)
        return scale

    def _field_dims(self, scale):
        W, H = self.fb
        return max(8, int(round(W * scale))), max(8, int(round(H * scale)))

    # ========================================================== motion model
    def _xform(self, src: View, dst: View):
        """Similarity from screen pixels of `src` to screen pixels of `dst` (window size, y down,
        relative to the centre): q = dlt + A * d.  None when numerically meaningless."""
        dl = src.log2r - dst.log2r
        if abs(dl) > 40:
            return None
        A = 2.0 ** dl * complex(math.cos(dst.angle - src.angle), math.sin(dst.angle - src.angle))
        dlt = dst.delta_in_pixels(src, self.fb[1])
        if max(abs(dlt[0]), abs(dlt[1])) > 1e7:
            return None
        return A, dlt

    def _coverage(self, base: View, v: View) -> float:
        """Factor by which the screen of `base` would have to grow to contain the screen of `v`
        (<= 1: `base` already covers it)."""
        xf = self._xform(v, base)
        if xf is None:
            return math.inf
        A, (dx, dy) = xf
        W, H = self.fb
        s = 0.0
        for cx, cy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
            q = complex(dx, dy) + A * complex(cx * W, cy * H)
            s = max(s, abs(q.real) / (0.5 * W), abs(q.imag) / (0.5 * H))
        return s

    def _track_motion(self, dt: float):
        """Camera velocity (octaves/s, rad/s, screen px/s) for predicting the next views."""
        v = self.view
        p = self._mv_prev
        self._mv_prev = v
        if not self.is_moving():
            self.vel = (0.0, 0.0, 0.0, 0.0)
            return
        if p is None or dt <= 0:
            return
        if p is v:
            self.vel = (0.0, 0.0, 0.0, 0.0)          # nothing moved this frame: no momentum to predict
            return
        W, H = self.fb
        dl = v.log2r - p.log2r
        da = math.remainder(v.angle - p.angle, 2 * math.pi)
        try:
            dx, dy = p.delta_in_pixels(v, H)
        except Exception:
            dx = dy = math.inf
        if abs(dl) > 1.0 or abs(da) > 1.0 or abs(dx) > 2 * W or abs(dy) > 2 * H:
            self.vel = (0.0, 0.0, 0.0, 0.0)          # a jump (location loaded, history), not a motion
            return
        a = min(1.0, dt / 0.08)
        new = (dl / dt, da / dt, dx / dt, dy / dt)
        self.vel = tuple(o * (1.0 - a) + n * a for o, n in zip(self.vel, new))

    def _view_after(self, tau: float) -> View:
        """Where the camera will be in `tau` seconds: exact for animations, transitions and
        playback, extrapolated from the current velocity otherwise."""
        v = self.view
        if tau <= 0:
            return v
        W, H = self.fb
        if self.playing and len(self.sequence.keyframes) >= 2:
            return self.sequence.view_at(min(self.play_t + tau, self.sequence.end))[0]
        if self.transition is not None:
            tr = self.transition
            u = min(1.0, (self.anim_now + tau - tr["t0"]) / tr["dur"])
            return self.apply_auto_iter(interpolate_views(tr["v0"], tr["v1"], ease("smooth", u)))
        if abs(self.zoom_anim) > 1e-4 and self.zoom_focus is not None:
            if self.settings.wheel_spring:
                rem = self.zoom_anim - spring_step(self.zoom_anim, self.zoom_vel, tau, ZOOM_OMEGA)[0]
            else:
                rem = self.zoom_anim * (1.0 - math.exp(-12.0 * tau))
            return self.apply_auto_iter(v.zoom_at(self.zoom_focus[0], self.zoom_focus[1], rem, W, H))
        vz, va, vx, vy = self.vel
        out = v.offset(vx * tau, vy * tau, H) if (vx or vy) else v.copy()
        out.log2r = v.log2r + vz * tau
        out.angle = v.angle + va * tau
        return self.apply_auto_iter(out.ensure_precision(H))

    def _snap_for_reuse(self, tv: View, Wf: int, Hf: int) -> View:
        """Nudge an animation's end view by less than half a pixel so that its sample grid is
        the current field's grid zoomed by an exact power of two: every 2nd (4th) sample then
        coincides with an already computed one and is reused."""
        f, fv = self.field, self.field_view
        if f is None or fv is None or f.params is None or not self.field_active or (f.W, f.H) != (Wf, Hf):
            return tv
        return snap_view_to_grid(fv, tv, Wf, Hf)

    def _field_target(self, moving: bool, Wf: int, Hf: int):
        """The view a (re)started field is computed for.

        Idle: the current view.  Short zoom animations: their end view (it lies inside the
        screen when zooming in and covers it when zooming out), snapped for exact reuse.
        Other zooms/rotations: a view that covers the screens of the next ~1.25 field times,
        so the field stays useful while it is being computed.  Pans: the current view (integer
        pixel reuse)."""
        v = self.view
        self.anim_end = None
        if not moving:
            return v, "current"
        if self.field_T < 0.8 * self.settings.frame_budget_ms / 1000.0:
            # the GPU finishes a whole picture within one frame: compute exactly the view on
            # the screen every frame - shown 1:1, nothing reprojected, nothing blurred
            return v, "current"
        W, H = self.fb
        end = None
        tr = self.transition
        if tr is not None:
            if tr["t0"] + tr["dur"] - self.anim_now <= 0.75 and abs(tr["v1"].log2r - v.log2r) <= 3.0:
                end = tr["v1"]
        elif abs(self.zoom_anim) > 1e-4 and self.zoom_focus is not None and abs(self.zoom_anim) <= 3.0:
            end = self.apply_auto_iter(v.zoom_at(self.zoom_focus[0], self.zoom_focus[1], self.zoom_anim, W, H))
        if (end is not None and not self.playing
                and abs(math.remainder(end.angle - v.angle, 2 * math.pi)) < 1e-9
                and (self._coverage(v, end) <= 1.0 + 1e-6 or self._coverage(end, v) <= 1.0 + 1e-6)):
            fv = self.field_view
            if (fv is not None and self.field is not None and (self.field.W, self.field.H) == (Wf, Hf)
                    and self._same_place(fv, end, Hf)):
                snapped = fv.copy()                  # already computing it (snapped earlier)
                snapped.max_iter = end.max_iter
            else:
                snapped = self._snap_for_reuse(end, Wf, Hf)
            if tr is not None:
                snapped.max_iter = tr["v1"].max_iter
                tr["v1"] = snapped                   # the transition ends exactly there
                if self.settings.auto_iter:
                    snapped = self.apply_auto_iter(snapped.copy())
            else:
                self.anim_end = snapped              # handle_input lands the zoom exactly there
            return snapped, "endpoint"
        vz, va, _, _ = self.vel
        animated = self.playing or tr is not None or abs(self.zoom_anim) > 1e-4
        if not animated and abs(vz) < 1e-3 and abs(va) < 1e-4:
            return v, "current"
        T = min(0.6, max(0.05, 1.25 * self.field_T))
        views = [v] + [self._view_after(T * k / 3.0) for k in (1, 2, 3)]
        base = views[2].copy()
        base.log2r = max(x.log2r for x in views)
        s = max(self._coverage(base, x) for x in views)
        if s > 1.0 + 1e-6:
            base.log2r += math.log2(min(s, 1.5))   # wider costs sharpness (same number of samples)
        base.ensure_precision(Hf)
        return self.apply_auto_iter(base), "predicted"

    @staticmethod
    def _same_place(a: View, b: View, Hf: int, px: float = 1.0) -> bool:
        """Same scale and angle, centres less than `px` samples apart."""
        if abs(a.log2r - b.log2r) > 1e-6 or abs(math.remainder(a.angle - b.angle, 2 * math.pi)) > 1e-9:
            return False
        dx, dy = a.delta_in_pixels(b, Hf)
        return math.hypot(dx, dy) < px

    def _reuse_relation(self, field: Field, fpo, fv: View | None, fp, tv: View, Wf: int, Hf: int, same_ref: bool):
        """How the samples of `field` (computed for `fv`) can serve the target view `tv`:
        ('same', reopen) | ('remap', mul, div, add, de_scale, reopen) | None."""
        if (fpo is None or fv is None or self.need_restart or field.params is None or not same_ref
                or fpo.variant() != fp.variant() or fpo.stripe_freq != fp.stripe_freq):
            return None
        m = grid_map(fv, tv, Wf, Hf)
        if m is None:
            return None
        reopen = fp.max_iter > fpo.max_iter
        if m == "same":
            return ("same", reopen)
        return ("remap", *m, reopen)

    def _in_flight_useful(self, field: Field, tv: View) -> bool:
        """While moving: keep computing the unfinished field (instead of restarting every frame)
        as long as it still covers the screen, or the view the camera is heading to."""
        fv, v = self.field_view, self.view
        if fv is None or time.perf_counter() - self.field_t0 > 2.0:
            return False
        dens = field.W / max(self.fb[0], 1) * 2.0 ** (v.log2r - fv.log2r)     # samples per screen pixel
        if dens < 1.0 / 3.0:
            return False
        return self._coverage(fv, v) <= 1.002 or self._coverage(fv, tv) <= 1.002

    # ============================================================== compute
    def update_field(self, budget: float):
        """Decide reuse / restart for the current view, then compute for `budget` seconds."""
        self._uf_t0 = time.perf_counter()
        s = self.settings
        moving = self.is_moving()
        scale = s.render_scale
        if moving:
            ms = s.motion_scale if s.motion_scale > 0 else self._auto_motion_scale()
            scale = min(scale, max(ms, 0.25))
        double = self.delta_precision()
        Wf, Hf = self._field_dims(scale)
        v = self.view
        julia = self.julia_c is not None
        if julia and not self.julia_deep(v, Hf):
            return self._update_field_rest(budget, moving, scale, *self._julia_setup(v, moving, double, Wf, Hf))
        refs = self.active_refs()
        status = refs.ref_status(v, Wf, Hf)
        if status != "ok":
            if status == "iterations":
                # only too short: extend the same reference point, so all computed samples
                # stay valid when the longer orbit arrives
                anchor = View(refs.ref_re, refs.ref_im, v.log2r)
            else:
                anchor = self._auto_reference_point(v) or (None if julia else self.ref_anchor)
            self._request_reference(v, Wf, Hf, anchor)
            if status == "missing":
                self.field_active = False
                self.need_restart = True
                return False
        ref = refs.ref
        tv, kind = self._field_target(moving, Wf, Hf)
        if kind != "current":
            st_t = refs.ref_status(tv, Wf, Hf)
            if st_t == "missing":
                tv, kind, st_t = v, "current", status
        else:
            st_t = status
        v_eff = tv
        if st_t == "iterations":
            v_eff = tv.copy()
            v_eff.max_iter = max(1, min(tv.max_iter, ref.length - 1))
        eps = s.eps_f64 if double else s.eps_f32
        cs = self.colors
        ship = self.formula == "ship" and not julia
        if (not self.interior_needed and getattr(self, "interior_off_l2r", None) is not None
                and v_eff.log2r > self.interior_off_l2r + 1.0):
            # zoomed out more than an octave since the detection was switched off: the wider view
            # may hold inside points, and without the detection they would all run to the limit
            self.interior_needed = True
        fp = view_params(v_eff, refs.ref_re, refs.ref_im, Wf, Hf, double=double,
                         use_bla=s.use_bla and not ship, with_de=cs.needs_de,
                         with_interior=s.interior and self.interior_needed,
                         with_stripes=cs.needs_stripes, stripe_freq=float(cs.stripe_freq), julia=julia,
                         formula="ship" if ship else "mandelbrot")
        want = False
        if julia:
            if not refs.bla_usable(fp, eps):
                fp.use_bla = False             # (tables for another precision: new ones are coming)
        elif ship:
            pass                               # (no BLA for the Burning Ship)
        else:
            # A table holds for pixel offsets |c| <= 2^log2_cmax.  Zooming out widens the offsets
            # by vel[0] octaves per second: then a table gets head-room for the time the next one
            # takes, and the next one is asked for before this one runs out (v3.11 lag: zooming
            # out of a deep minibrot built a table every octave and drew every other frame without
            # skips while waiting for it).  When the camera stops, a tight table again.
            b = self.refs.bla
            valid = (b is not None and b.ref_id == ref.ref_id and abs(b.log2_eps - eps) < 1e-9
                     and fp.log2_cmax <= b.log2_cmax)
            slack = (b.log2_cmax - fp.log2_cmax) if valid else -1.0
            out = max(0.0, self.vel[0]) if moving else 0.0
            margin = 1.0 + min(6.0, 0.6 * out)
            want = (not valid or slack >= 6.0
                    or (out > 0.05 and slack < 0.35 * out + 0.25)
                    or (not moving and slack > 2.5))
        if not julia and not ship and want:
            if ref.length < 60000:
                nb = build_bla(ref.zm, ref.ze, fp.log2_cmax + margin, eps, max_iter=ref.max_iter)
                nb.ref_id = ref.ref_id
                self.refs.set_bla(nb)
            else:
                if self.pending_bla is None:
                    jid = self.ref_worker.submit("bla", cancel_previous=False, ref_id=ref.ref_id,
                                                 log2_cmax=fp.log2_cmax + margin, log2_eps=eps)
                    self.pending_bla = (jid,)
                if not valid:
                    fp.use_bla = False          # correct but slower until the new table arrives
                    if b is not None:
                        self.refs.set_bla(None)
        gref, gbla = refs.gpu(double)
        field = self.get_field(Wf, Hf, double)
        # Samples stay valid when only the BLA table changes (it is rebuilt while panning away
        # from the reference): they are exact orbit states, so reuse needs the same reference only.
        ref_point = (refs.ref_re, refs.ref_im) if not julia else ("julia-deep", refs.c, refs.ref_re, refs.ref_im)
        same_ref = field.gref is gref or getattr(field, "ref_point", None) == ref_point
        return self._update_field_rest(budget, moving, scale, Wf, Hf, kind, v_eff, fp, eps, gref, gbla,
                                       field, ref_point, same_ref)

    def _julia_setup(self, v: View, moving: bool, double: bool, Wf: int, Hf: int):
        """update_field for a Julia set: no reference orbit, the field runs the Julia kernel."""
        s = self.settings
        double = double == 1          # (the direct kernel: float32 or float64; shallow anyway)
        cs = self.colors
        tv, kind = self._field_target(moving, Wf, Hf)
        fp = julia_params(tv, Wf, Hf, double=double, with_de=cs.needs_de,
                          with_interior=s.interior and self.interior_needed,
                          with_stripes=cs.needs_stripes, stripe_freq=float(cs.stripe_freq))
        field = self.get_field(Wf, Hf, double, kind="julia")
        ref_point = ("julia", self.julia_c)
        same_ref = getattr(field, "ref_point", None) == ref_point
        return Wf, Hf, kind, tv, fp, None, None, None, field, ref_point, same_ref

    def _update_field_rest(self, budget, moving, scale, Wf, Hf, kind, v_eff, fp, eps, gref, gbla,
                           field, ref_point, same_ref):
        fpo = self.field_fp if field is self.field else None
        fv_old = self.field_view if field is self.field else None
        self.dbg["kind"] = kind
        rel = self._reuse_relation(field, fpo, fv_old, fp, v_eff, Wf, Hf, same_ref)
        if rel is None and moving and fpo is not None and not field.complete and self.field_active \
                and not self.need_restart and same_ref and fpo.variant() == fp.variant() \
                and fpo.stripe_freq == fp.stripe_freq and self._in_flight_useful(field, v_eff):
            # keep computing the picture in flight; it is shown reprojected
            self.dbg["branch"] = "flight"
            fpk = fpo
            if field.gbla is not gbla or field.gref is not gref:
                if fpk.use_bla and not self.active_refs().bla_usable(fpk, eps):
                    fpk = replace(fpk, use_bla=False)
                field.configure(fpk, gref, gbla)
                self.field_fp = fpk
            self.flight_frames += 1
        elif rel is None:
            self.dbg["branch"] = "reset"
            if fpo is None:
                why = "new field"
            elif self.need_restart:
                why = "restart requested"
            elif not same_ref:
                why = "new reference"
            elif fpo.variant() != fp.variant() or fpo.stripe_freq != fp.stripe_freq:
                why = "colouring channels"
            elif moving:
                why = f"moving ({kind})"
            elif abs(v_eff.log2r - fv_old.log2r) > 1e-9 or abs(v_eff.angle - fv_old.angle) > 1e-12:
                why = "zoom/rotation"
            else:
                why = "offset/iterations"
            self.reset_log.append(why)
            self.reset_log_total += 1
            del self.reset_log[:-40]
            self._snapshot_if_better()
            field.reset()
            field.configure(fp, gref, gbla)
            self.field_serial += 1
            self.field_t0 = time.perf_counter()
            self.field_fresh = True
            self.field_epoch += 1
            self.field_view = v_eff.copy()
            self.field_fp = fp
        elif rel[0] == "same":
            self.dbg["branch"] = "same"
            reopen = rel[1]
            fpk = replace(fpo, max_iter=fp.max_iter, use_bla=fp.use_bla, log2_cmax=fpo.log2_cmax)
            if reopen:
                field.reopen_maxiter()
            field.configure(fpk, gref, gbla)      # may also pick up a new reference / BLA table
            if fpk.max_iter != fpo.max_iter:
                self.field_serial += 1
                self.field_view = fv_old.copy()
                self.field_view.max_iter = v_eff.max_iter
            self.field_fp = fpk
        else:
            self.dbg["branch"] = "remap"
            _, mul, div, add, de_scale, reopen = rel
            if mul != (1, 1) or div != (1, 1):
                self._snapshot_if_better()        # the remapped field starts partly empty
                self.reset_log.append(f"zoom x{div[0] / mul[0]:g}: reused")
                self.reset_log_total += 1
                del self.reset_log[:-40]
                self.field_t0 = time.perf_counter()
                self.field_fresh = True
                self.field_epoch += 1
            else:
                self.field_fresh = False
            if reopen:                           # first: it uses the current counters ...
                field.reopen_maxiter()
            field.remap(mul, div, add, de_scale)  # ... which the remap recounts on the GPU
            field.configure(fp, gref, gbla)
            self.field_serial += 1
            self.field_view = v_eff.copy()
            self.field_fp = fp
        field.ref_point = ref_point
        if self.julia_c is not None and not self.field_fp.julia:
            self.engine.setup_julia(field, self.field_view, self.julia_c, Wf, Hf, self.field_fp)
        else:
            field.program = None
        self.field = field
        self.field_scale = scale
        self.field_active = True
        self.need_restart = False
        was_complete = field.complete
        # the time slice is for the whole update: what deciding / restarting took is deducted
        t_step = time.perf_counter()
        pre = t_step - getattr(self, "_uf_t0", t_step)
        done = field.step(max(0.3 * budget, budget - pre), dispatch_target=budget * 0.5)
        t_post = time.perf_counter()
        if done and not was_complete:
            self._on_field_complete(moving)
        elif (not done and self.colors.mapping == 3 and field.level_done <= 3
              and getattr(self, "auto_serial", -1) != self.field_serial):
            # 'auto' colours: adapt as soon as the coarse level of a new view is there
            # (otherwise a jump to a very different depth shows one flat colour until done)
            self.auto.update(field, smooth=0.6 if moving else 0.0)
            self.auto_serial = self.field_serial
        self.dbg.update(t_pre=round(pre * 1000, 2), t_step=round((t_post - t_step) * 1000, 2),
                        t_post=round((time.perf_counter() - t_post) * 1000, 2))
        return done

    def _on_field_complete(self, moving: bool):
        f = self.field
        s = self.settings
        r = self.active_refs().ref if (self.julia_c is None or self.julia_deep()) else None
        capped = r is not None and not r.complete_for(self.view.max_iter)     # reference still short
        if self.field_fresh:
            dt = time.perf_counter() - self.field_t0
            self.field_T = min(1.0, 0.5 * self.field_T + 0.5 * dt)
            area = (f.W * f.H) / max(1, self.fb[0] * self.fb[1])     # field samples per screen pixel
            if area > 0.05:
                self.field_T_full = min(4.0, 0.5 * self.field_T_full + 0.5 * dt / min(1.0, area))
        if self.colors.mapping == 3:
            self.auto.update(f, smooth=0.6 if moving else 0.0)
        self._take_snapshot(min(1.0, (f.W / max(self.fb[0], 1)) ** 2))
        if capped:
            return          # drawn with a reference that is still being computed: decide later
        if s.interior:
            if f.params is not None and f.params.with_interior:
                # nothing inside (and nothing undecided): the next pictures skip the detection
                self.interior_needed = f.interior_count > 0 or f.maxiter_count > 0
                if not self.interior_needed and self.field_view is not None:
                    self.interior_off_l2r = self.field_view.log2r
            elif f.maxiter_count > 0:
                # computed without detection and something stopped at the limit: it may be
                # inside - detect it (instead of raising the iteration limit)
                self.interior_needed = True
                if not moving:
                    self.need_restart = True
                return
        if s.auto_iter and not moving and self.transition is None and f.params is not None:
            boost = self.boost_for(self.view)
            if f.maxiter_count <= UNDECIDED_OK * f.N and boost <= 1.0:
                return                          # the common case: nothing to decide, no read-back
            st = f.iteration_octaves(f.params.max_iter)
            # lower only on a picture of a new view (never right after raising it here)
            step, why = iteration_advice(st, boost, may_lower=self.boost_epoch != self.field_epoch)
            self.iter_advice = (step, why, st)
            if step > 0:
                # the limit cuts off outside detail: double it (continues incrementally)
                self.set_boost(boost * 2.0, floor=self.floor_for(self.view))
                self.boost_epoch = self.field_epoch
                self.view = self.apply_auto_iter(self.view.copy())
                self.notify(f"more iterations: {self.view.max_iter:,}")
            elif step < 0 and boost / 2.0 >= max(1.0, self.floor_for(self.view)):
                self.set_boost(boost / 2.0, floor=self.floor_for(self.view))
                self.view = self.apply_auto_iter(self.view.copy())

    def step_aa(self, budget: float):
        """Idle progressive anti-aliasing: jittered passes accumulated into a float texture."""
        s = self.settings
        f = self.field
        if (s.aa_passes <= 1 or f is None or not f.complete or not self.field_active or self.is_moving()
                or self.colors.cycle_speed != 0.0 or self.aa_count >= s.aa_passes):
            return
        W, H = self.fb
        if self.aa_count == 0 or self.aa_serial != self.field_serial:
            self.accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._render_field(f, self.accum_fbo, blend_add=True)
            self.aa_count = 1
            self.aa_serial = self.field_serial
            self.aa_field_key = None
        aa = self.get_field(f.W, f.H, f.double, kind="aa")
        aa.program = f.program                          # Julia kernel or None (perturbation)
        if f.program is not None:
            aa.extra_uniforms = dict(f.extra_uniforms)
        key = (self.field_serial, self.aa_count)
        if getattr(self, "aa_field_key", None) != key:
            self.aa_fp = replace(self.field_fp, jitter=jitter(self.aa_count))
            aa.reset()
            if s.adaptive_ss and (f.W, f.H) == (W, H):
                # only pixels at edges / near the set get new jittered samples; the others keep
                # the centre sample (their average would not change)
                if self.aa_ctex is None or self.aa_ctex.size != (W, H):
                    if self.aa_ctex is not None:
                        self.aa_cfbo.release()
                        self.aa_ctex.release()
                    self.aa_ctex = self.ctx.texture((W, H), 4)
                    self.aa_cfbo = self.ctx.framebuffer(color_attachments=[self.aa_ctex])
                if self.aa_ctex_serial != (self.field_serial, self.colors_key):
                    self._render_field(f, self.aa_cfbo)
                    self.aa_ctex_serial = (self.field_serial, self.colors_key)
                aa.seed_adaptive(f, self.aa_ctex, 1)
            self.aa_field_key = key
        aa.configure(self.aa_fp, f.gref, f.gbla)      # the BLA table may have been replaced
        if aa.step(budget, dispatch_target=budget * 0.5):
            self._render_field(aa, self.accum_fbo, blend_add=True)
            self.aa_count += 1

    def _render_field(self, f: Field, fbo, blend_add=False):
        W, H = self.fb
        auto = self.auto if self.colors.mapping == 3 and self.auto.valid else None
        self.engine.display.render(f, fbo, (0, 0, W, H), f.W / W, self.colors, fallback=False,
                                   max_iter_show=self.field_view.max_iter, auto=auto,
                                   iter_base=self._iter_base(), blend_add=blend_add,
                                   time_offset=self.colors.cycle_speed * time.perf_counter())

    def _iter_base(self):
        f = self.field
        if self.colors.relative_iter and f is not None and f.min_iter:
            return max(0, f.min_iter - 1)
        return 0

    # ============================================================== display
    def _snap_xform(self):
        """Reprojection of the last snapshot onto the screen and its quality there."""
        if self.snap_view is None:
            return None, -1.0
        xf = self._xform(self.view, self.snap_view)
        if xf is None:
            return None, -1.0
        q = min(1.0, abs(xf[0]) ** 2)         # x the per-pixel quality stored in the snapshot
        if abs(math.sin(2 * (self.snap_view.angle - self.view.angle))) > 1e-6:
            q *= 0.9                          # resampled rotation is slightly soft
        return xf, q

    def _snap_coverage(self) -> float:
        """Fraction of the screen the last snapshot covers (sampled on a 9 x 5 grid)."""
        if self.snap_view is None:
            return 0.0
        xf = self._xform(self.view, self.snap_view)
        if xf is None:
            return 0.0
        A, (dx, dy) = xf
        W, H = self.fb
        inside = 0
        for j in range(5):
            for i in range(9):
                d = complex((i / 8.0 - 0.5) * W, (j / 4.0 - 0.5) * H)
                q = complex(dx, dy) + A * d
                inside += abs(q.real) <= 0.5 * W and abs(q.imag) <= 0.5 * H
        return inside / 45.0

    def _field_xform(self):
        """(xform or None if the field is aligned with the screen, quality of its samples)."""
        f, fv = self.field, self.field_view
        scale = f.W / max(self.fb[0], 1)
        xf = self._xform(self.view, fv)
        if xf is None:
            return False, None, -1.0
        A, (dx, dy) = xf
        if abs(A - 1.0) < 1e-9 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return True, None, min(1.0, scale ** 2)
        q = min(1.0, (abs(A) * scale) ** 2)
        if abs(math.sin(2 * (fv.angle - self.view.angle))) > 1e-6:
            q *= 0.9
        return True, xf, q

    def _take_snapshot(self, q: float):
        """Keep the field's picture (at the field's own view) as the reprojection source for
        the following frames; parts the field has not computed yet come from the old one."""
        f, fv = self.field, self.field_view
        if f is None or f.params is None or fv is None or self.scratch_fbo is None:
            return
        W, H = self.fb
        cs = self.colors
        # Snapshot at the field's own view (its samples 1:1) when that covers the screen; a
        # field inside the screen (the end view of a zoom-in) goes into a snapshot of the screen
        # view instead, so the wider old picture around it is not lost.
        target = fv if self._coverage(fv, self.view) <= 1.002 else self.view
        fxf, fq = None, min(1.0, (f.W / W) ** 2)
        if target is not fv:
            fxf = self._xform(target, fv)
            if fxf is None:
                return
            fq = min(1.0, (abs(fxf[0]) * f.W / W) ** 2)
        pxf, pq = None, -1.0
        if self.snap_view is not None:
            pxf = self._xform(target, self.snap_view)
            if pxf is not None:
                pq = min(1.0, abs(pxf[0]) ** 2)
        auto = self.auto if cs.mapping == 3 and self.auto.valid else None
        self.engine.display.render(
            f, self.scratch_fbo, (0, 0, W, H), f.W / W, cs, out_half=(W / 2.0, H / 2.0),
            prev_tex=self.prev_tex if pxf else None, prev_xform=pxf, prev_size=(W, H), prev_q=pq,
            field_q=fq, field_xform=fxf, time_offset=cs.cycle_speed * time.perf_counter(),
            iter_base=self._iter_base(), max_iter_show=fv.max_iter, fallback=True, auto=auto)
        self.prev_tex, self.scratch_tex = self.scratch_tex, self.prev_tex
        self.prev_fbo, self.scratch_fbo = self.scratch_fbo, self.prev_fbo
        self.snap_view = target.copy()
        self.snap_q = q
        self.snap_serial = self.field_serial

    def _snapshot_if_better(self):
        """Before a restart: merge what the field has into the snapshot when that improves the
        screen - the stored picture covers too little of it (zooming out, rotating) or is much
        coarser (zoomed far since the last completed frame).  The snapshot keeps the better
        source per pixel, so merging a partial field never loses detail."""
        if self.field is None or not self.field_active or self.snap_serial == self.field_serial:
            return
        if self.field.params is None or self.field_view is None:
            return
        f = self.field
        if not f.complete and f.done_count == 0:
            return                             # nothing computed yet
        ok, _, fq = self._field_xform()
        if not ok:
            return
        _, q = self._snap_xform()
        if (f.complete or self._snap_coverage() < 0.98 or q < 0.02
                or (f.progress > 0.5 and q < 0.25 * fq)):
            self._take_snapshot(1.0)

    def render_frame(self):
        W, H = self.fb
        cs = self.colors
        xf, q = self._snap_xform()
        f = self.field
        auto = self.auto if cs.mapping == 3 and self.auto.valid else None
        ok = False
        if f is not None and self.field_active and f.params is not None and self.field_view is not None:
            ok, fxf, fq = self._field_xform()
        self.dbg.update(q=round(q, 4), fq=None, fxf=None, aa_blit=False)
        if ok:
            self.dbg.update(fq=round(fq, 4), fxf=fxf is not None)
            if (fxf is None and self.aa_count >= 2 and self.aa_serial == self.field_serial and f.complete):
                self.dbg["aa_blit"] = True
                self.engine.display.blit(self.accum_tex, self.screen_fbo, (0, 0, W, H), 1.0 / self.aa_count)
            else:
                self.engine.display.render(
                    f, self.screen_fbo, (0, 0, W, H), f.W / W, cs,
                    out_half=(W / 2.0, H / 2.0), prev_tex=self.prev_tex if xf else None,
                    prev_xform=xf, prev_size=(W, H), prev_q=q, field_q=fq, field_xform=fxf,
                    time_offset=cs.cycle_speed * time.perf_counter(), iter_base=self._iter_base(),
                    max_iter_show=self.field_view.max_iter, fallback=self.settings.show_fallback, auto=auto,
                    background=self.uncovered_colour)
        elif xf is not None:
            # waiting for a reference: show the reprojected last image
            self.engine.display.render(self._empty_field(), self.screen_fbo, (0, 0, W, H), 1.0, cs,
                                       prev_tex=self.prev_tex, prev_xform=xf, prev_size=(W, H), prev_q=q,
                                       fallback=False, background=self.uncovered_colour)
        self.engine.display.blit(self.screen_tex, self.ctx.screen, (0, 0, W, H))

    def _empty_field(self):
        if getattr(self, "_empty", None) is None:
            self._empty = Field(self.engine, 8, 8, False)
        return self._empty

    def screen_rgb(self) -> np.ndarray:
        W, H = self.fb
        data = self.screen_fbo.read(components=3, alignment=1)
        return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()

    # ================================================================ frame
    def internal_error(self, where: str):
        """Log an unexpected exception once, tell the user briefly, keep running."""
        tb = traceback.format_exc()
        seen = self.__dict__.setdefault("_errors_seen", set())
        key = tb[-400:]
        if key in seen:
            return
        seen.add(key)
        last = self.__dict__.get("_err_line", "")
        line = tb.strip().splitlines()[-1][:160]
        if line != last:
            self.notify(f"internal error ({where}): {line} - details in the log")
            self._err_line = line
        if sys.stderr is not None:
            try:
                print(f"[mandelscope] internal error in {where}:\n{tb}", file=sys.stderr, flush=True)
            except Exception:
                pass
        if self.selftest is not None:
            self.selftest.report["errors"].append((where, tb[-3000:]))

    def custom_background(self):
        try:
            self._frame()
        except Exception:
            self.internal_error("render")

    def _frame(self):
        io = imgui.get_io()
        sc = io.display_framebuffer_scale
        self.fb_scale = (sc.x or 1.0, sc.y or 1.0)
        W = int(io.display_size.x * self.fb_scale[0])
        H = int(io.display_size.y * self.fb_scale[1])
        if W < 16 or H < 16:
            return
        if (W, H) != self.fb:
            first = self.fb == (0, 0)
            self.fb = (W, H)
            self._ensure_screen(W, H)
            if first and self.pending_location is None and not self.history:
                self.view = self.apply_auto_iter(self.home_view())
                self._sync_edit_fields()
        now = time.perf_counter()
        raw = min(0.1, now - self.last_time)
        self.last_time = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(raw, 1e-4))
        dt = self.pacer.step(raw) if self.settings.frame_pacing else raw
        self.anim_now += dt                       # the camera's clock (transitions use it too)
        if abs(self.anim_now - now) > 0.1:
            self.anim_now = now
        self.anim_dt = dt
        inp = self.selftest.input() if self.selftest else read_input(self.fb_scale)
        inp.pad = self._read_pad()
        self.handle_input(inp, dt)
        if self.playing:
            self._advance_playback(dt)
        self._track_motion(dt)
        self._poll_workers()
        if self.path_rec is not None:          # sampled every frame (pauses are kept too)
            self.path_rec.add(now, self.view, {"offset": self.colors.offset})
        if self.live is not None and (self.live.w > W or self.live.h > H):
            self.notify("window resized - live recording stopped")
            self.toggle_live()
        # colour changes -> restart anti-aliasing (and the field if channels changed)
        ck = (tuple(sorted((k, str(v)) for k, v in self.colors.to_dict().items())),)
        if ck != self.colors_key:
            self.colors_key = ck
            self.aa_count = 0
            if self.colors.mapping == 3 and self.field is not None and self.field.complete:
                self.auto.update(self.field)
        budget = self.settings.frame_budget_ms / 1000.0
        t1 = time.perf_counter()
        job_running = any(j.status == "running" for j in self.jobs)
        if job_running:
            self._step_jobs(0.030 if self.is_moving() else 0.060)
            budget = min(budget, 0.006)
        self.update_field(budget)
        if not job_running:
            self.step_aa(budget * 0.8)
        if self.colour_tab_visible and not self.is_moving() and self.field is not None and self.field.complete:
            self.previews.update(0.004)
        if (self.settings.julia_preview and self.julia_c is None and self.formula == "mandelbrot"
                and self.ui_visible and self.jpreview is not None):
            self.jpreview.update(self.preview_c(), 0.003)
        t2 = time.perf_counter()
        self.render_frame()
        t3 = time.perf_counter()
        self.ms_update, self.ms_render = (t2 - t1) * 1000.0, (t3 - t2) * 1000.0
        self.frame_ms = (time.perf_counter() - now) * 1000.0
        if DEBUG:
            f = self.field
            print(f"[frame] update {1000 * (t2 - t1):.1f} ms  field={None if f is None else (f.W, f.H, round(f.progress, 3))} "
                  f"zoom={self.view.zoom_magnification_log10():.3f} aa={self.aa_count}", flush=True)
        if self.live is not None and not self.live.include_ui:
            self.live.capture(self.screen_fbo, H)
        if self.selftest:
            self.selftest.after_frame()
        elif self.on_cpu:
            # on the processor every frame costs real time: when nothing is left to compute or
            # animate, let hello_imgui wait for input (about 9 frames per second) instead of
            # redrawing the same picture 60 times a second
            try:
                hello_imgui.get_runner_params().fps_idling.enable_idling = self.quiet(job_running)
            except Exception:
                pass

    def restart_renderer(self, cpu: bool):
        """Start Mandelscope again on the processor (cpu=True) or the graphics card, then close."""
        if cpu and sys.platform == "win32" and not software.ensure_mesa_interactive(
                "CPU mode uses Mesa's software renderer, which is not installed yet."):
            return
        self._save_config()
        try:
            software.restart(cpu)
        except Exception as e:
            self.notify(f"could not restart: {e}")
            return
        hello_imgui.get_runner_params().app_shall_exit = True

    def quiet(self, job_running: bool = False) -> bool:
        """Nothing to compute, animate or record: the picture would not change without input."""
        s, f = self.settings, self.field
        if f is None or not f.complete or job_running or self.is_moving():
            return False
        if (self.live is not None or self.path_rec is not None or self.pending_ref is not None
                or self.pending_bla is not None or self.minibrot_job is not None):
            return False
        if self.colors.cycle_speed != 0.0 or self.colour_tab_visible:
            return False
        if s.aa_passes > 1 and self.field_active and self.aa_count < s.aa_passes:
            return False
        if getattr(self, "pad_active", False):
            return False
        return True

    def before_swap(self):
        if self.live is not None and self.live.include_ui:
            self.live.capture(self.ctx.screen, self.fb[1])
        if self.selftest:
            self.selftest.before_swap()

    # ============================================================= sequence
    def add_keyframe(self):
        cur = list(self.julia_c) if self.julia_c is not None else None
        meta = self.sequence.meta
        if self.sequence.keyframes and (meta.get("julia") != cur or meta.get("formula", "mandelbrot") != self.formula):
            what = ("a Julia set" if meta.get("julia") else FORMULA_NAMES[meta.get("formula", "mandelbrot")])
            self.notify(f"the timeline is for {what} - clear it first (Animate tab)")
            return
        if not self.sequence.keyframes:
            if cur is None:
                meta.pop("julia", None)
            else:
                meta["julia"] = cur
            if self.formula == "mandelbrot":
                meta.pop("formula", None)
            else:
                meta["formula"] = self.formula
        t = self.sequence.end + 3.0 if self.sequence.keyframes else 0.0
        if self.playing or (self.sequence.keyframes and self.sequence.start <= self.play_t < self.sequence.end
                            and self.selected_kf < 0):
            t = self.play_t
        kf = Keyframe(self.view.copy(), t, "linear", self.colour_state())
        if self.sequence.keyframes and any(abs(k.time - t) < 1e-6 for k in self.sequence.keyframes):
            t = self.sequence.end + 3.0
            kf.time = t
        self.sequence.add(kf)
        self.selected_kf = self.sequence.keyframes.index(kf)
        self.play_t = t
        self.settings.show_timeline = True
        self.notify(f"keyframe {self.selected_kf + 1} at {t:.1f}s")

    def colour_state(self) -> dict:
        c = self.colors
        return {k: getattr(c, k) for k in ("palette", "method", "mapping", "density", "offset", "auto_cycles",
                                           "de_density", "de_strength", "de_scale", "slope_strength",
                                           "light_angle", "light_height", "glow", "glow_color", "stripes",
                                           "stripe_freq", "pattern", "pattern_strength", "pattern_freq",
                                           "interior", "look")}

    def toggle_play(self):
        if len(self.sequence.keyframes) < 2:
            self.notify("add at least 2 keyframes (K) or apply a script to play")
            return
        self.playing = not self.playing
        if self.playing and (self.play_t >= self.sequence.end or self.play_t < self.sequence.start):
            self.play_t = self.sequence.start

    def _advance_playback(self, dt):
        seq = self.sequence
        if len(seq.keyframes) < 2:
            self.playing = False
            return
        self.play_t += dt
        if self.play_t > seq.end:
            if self.settings.seq_loop:
                self.play_t = seq.start
            else:
                self.play_t = seq.end
                self.playing = False
        self.seek(self.play_t)

    def seek(self, t):
        if not self.sequence.keyframes:
            return
        v, over = self.sequence.view_at(t)
        for k, val in over.items():
            if hasattr(self.colors, k):
                setattr(self.colors, k, val)
        jc = self.sequence.meta.get("julia")
        cur = list(self.julia_c) if self.julia_c is not None else None
        sf = self.sequence.meta.get("formula", "mandelbrot")
        if sf != self.formula:              # a Burning Ship timeline plays as the Burning Ship
            playing = self.playing
            self.set_formula(sf, view=v)
            self.playing = playing
        if jc != cur:                       # a Julia timeline plays in Julia mode (and back)
            playing = self.playing
            if jc is not None:
                self.enter_julia(jc[0], jc[1], view=v)
            else:
                self.leave_julia()
            self.playing = playing
        self.view = v
        self.transition = None
        self.ref_anchor = self.sequence.deepest_in_segment(t)
        self.mark_moving()
        self._sync_edit_fields()

    def toggle_path_recording(self):
        if self.path_rec is None:
            self.path_rec = PathRecorder()
            self.path_rec.add(time.perf_counter(), self.view, {"offset": self.colors.offset})
            self.notify("recording your flight - press P again to stop")
        else:
            rec = self.path_rec
            self.path_rec = None
            rec.add(time.perf_counter() + 1.0, self.view, {"offset": self.colors.offset})
            if len(rec.samples) >= 2:
                self.sequence = rec.to_sequence(self.settings.video_fps, smooth=2)
                if self.julia_c is not None:
                    self.sequence.meta["julia"] = list(self.julia_c)
                if self.formula != "mandelbrot":
                    self.sequence.meta["formula"] = self.formula
                for k in self.sequence.keyframes:
                    k.color = {**self.colour_state(), **k.color}
                self.selected_kf = -1
                self.play_t = 0.0
                self.settings.show_timeline = True
                self.notify(f"flight recorded: {self.sequence.duration:.1f}s -> timeline (render it in Export)")
            else:
                self.notify("flight too short")

    def apply_script(self, text: str | None = None) -> bool:
        text = self.script_text if text is None else text
        looks = LOOKS
        res = compile_script(text, resolve_location=self.library.view, auto_iter=self.auto_iter_for,
                             current_view=self.view, base_colors=self.colour_state(), looks=looks,
                             current_formula=self.formula)
        self.script_errors = res.errors + [(ln, "warning: " + m) for ln, m in res.warnings]
        if not res.ok:
            self.script_info = f"{len(res.errors)} error(s) - nothing changed"
            return False
        self.sequence = res.sequence
        self.settings.video_fps = res.fps
        if res.size:
            self.settings.video_w, self.settings.video_h = res.size
        self.selected_kf = -1
        self.play_t = self.sequence.start
        self.settings.show_timeline = True
        self.seek(self.play_t)
        self.script_info = (f"{len(res.sequence.keyframes)} keyframes, {res.sequence.duration:.1f}s"
                            + (f", look {res.look}" if res.look else ""))
        return True

    def script_from_timeline(self):
        if not self.sequence.keyframes:
            self.notify("timeline is empty")
            return
        self.script_text = sequence_to_script(self.sequence)
        self.script_errors = []
        self.script_info = "script generated from the timeline"

    def copy_ai_prompt(self):
        locs = [f'"{e.name}" ({e.label}, zoom 10^{e.zoom:.0f})' for e in self.library.entries]
        txt = ai_prompt(self.script_text, self.view, locs, list(LOOKS))
        imgui.set_clipboard_text(txt)
        self.notify("AI prompt copied - paste it into Claude (or another assistant), then paste the script back")

    def zoom_movie_here(self, seconds_per_decade: float = 2.0, spin: float = 0.0):
        """Script: from the whole set into the current view at constant speed."""
        z = max(0.5, self.view.zoom_magnification_log10())
        dur = max(4.0, z * seconds_per_decade)
        v = self.view
        digits = int(math.ceil(v.needed_precision(2160) * 0.30103)) + 3
        re_s = format(mpfr(v.re, max(v.precision, 64)), f".{max(digits, 17)}e")
        im_s = format(mpfr(v.im, max(v.precision, 64)), f".{max(digits, 17)}e")
        target = (f"re={re_s} im={im_s} zoom=10^{z:.6f} angle={math.degrees(v.angle):.4f}"
                  + (f" iter={v.max_iter}" if not self.settings.auto_iter else ""))
        self.script_text = (
            f"# zoom movie into the current view ({z:.0f} decades)\n"
            + (f"fractal {self.formula}\n" if self.formula != "mandelbrot" else "") +
            f"look {self.colors.look}\nstart home\n"
            f"zoom to {target} over {dur:.1f}s" + (f" spin {spin:g}" if spin else "") +
            "\nhold 2s\n")
        self.apply_script()

    def zoom_movie_to(self, entry, seconds_per_decade: float = 2.0):
        """Script: from the whole set into a library location."""
        z = max(0.5, entry.zoom)
        dur = max(4.0, z * seconds_per_decade)
        f = getattr(entry, "formula", "mandelbrot")
        self.script_text = (f"# zoom movie into {entry.label}\n"
                            + (f"fractal {f}\n" if f != "mandelbrot" else "") +
                            f"look {self.colors.look}\nstart home\n"
                            f"zoom to location \"{entry.name}\" over {dur:.1f}s\nhold 2s\n")
        self.apply_script()

    # ============================================================== live rec
    def toggle_live(self):
        if self.live is None:
            s = self.settings
            try:
                path = _unique(str(out_dir("Videos") / time.strftime("live_%Y%m%d_%H%M%S.mp4")))
                self.live = LiveRecorder(path, self.fb[0], self.fb[1], s.live_fps, s.live_codec,
                                         s.live_quality, s.live_ui)
                self.notify(f"recording ({self.live.writer.codec}) - press R to stop")
            except Exception as e:
                self.notify(f"cannot start recording: {e}")
                self.live = None
        else:
            rc, err = self.live.close()
            self.notify(f"recording saved: {self.live.path}"
                        + ("" if rc == 0 else f" (ffmpeg error {rc}: {err[-200:]})"))
            self.last_output = self.live.path
            self.live = None

    # ============================================================ jobs (queue)
    def _ensure_video_worker(self):
        if self.video_worker is None:
            self.video_worker = Worker()
        return self.video_worker

    def queue_video(self, draft: bool = False, path: str | None = None):
        if len(self.sequence.keyframes) < 2:
            self.notify("the timeline needs at least 2 keyframes (K, or apply a script)")
            return None
        s = self.settings
        w, h = (s.video_w, s.video_h)
        ss = s.video_ss
        if draft:
            h2 = 360
            w, h, ss = int(round(w * h2 / h / 2) * 2), h2, 1
        w -= w % 2
        h -= h % 2
        rs = RenderSettings(width=w, height=h, ss=ss,
                            double=self.delta_precision(s.video_precision, self.sequence.meta.get("formula", "mandelbrot")),
                            iter_mult=1 if draft else max(1, int(s.video_iter_mult)),
                            use_bla=s.use_bla, interior=s.interior, overlay="depth" if s.video_overlay else "",
                            adaptive=s.adaptive_export and not s.fast_movie, fast=s.fast_movie)
        name = (s.video_name or "mandelscope") + ("_draft" if draft else "")
        path = path or _unique(str(out_dir("Videos") / f"{name}_{w}x{h}.mp4"))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        png_dir = None
        if s.video_png and not draft:
            png_dir = str(Path(path).with_suffix("")) + "_frames"
            Path(png_dir).mkdir(parents=True, exist_ok=True)
        seq = Sequence.from_dict(self.sequence.to_dict())
        colors = replace(self.colors)
        fps, codec, quality = s.video_fps, s.video_codec, s.video_quality
        worker = self._ensure_video_worker()

        julia = seq.meta.get("julia")

        def make():
            return render_sequence_iter(self.engine, seq, path, rs, colors, fps, codec, quality,
                                        worker=worker, time_slice=0.02, png_dir=png_dir, julia=julia)
        job = Job("draft" if draft else "video", Path(path).name, path, make=make,
                  frames=seq.frame_count(fps), open_when_done=draft)
        self.jobs.append(job)
        self.notify(f"queued: {job.name} ({job.frames} frames)")
        return job

    def start_still(self, quick=False):
        s = self.settings
        w, h, ss = (self.fb[0], self.fb[1], 2) if quick else (s.still_w, s.still_h, s.still_ss)
        rs = RenderSettings(width=w, height=h, ss=ss, double=self.delta_precision(),
                            iter_mult=1 if quick else max(1, int(s.still_iter_mult)),
                            use_bla=s.use_bla, interior=s.interior, adaptive=s.adaptive_export)
        path = _unique(str(out_dir("Images") / time.strftime(f"mandelscope_%Y%m%d_%H%M%S_{w}x{h}.png")))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        view = self.view.copy()
        colors = replace(self.colors)
        worker = self._ensure_video_worker()
        engine = self.engine
        julia = self.julia_c
        formula = self.formula
        loc = {"view": view.to_dict(), "colors": colors.to_dict()}
        if julia is not None:
            loc["julia"] = list(julia)
        if formula != "mandelbrot":
            loc["formula"] = formula

        def make():
            fr = FrameRenderer(engine, rs)
            fr.julia = julia
            fr.formula = formula

            def gen():
                try:
                    g = fr.render_iter(view, colors, worker=worker, time_slice=0.02)
                    while True:
                        try:
                            phase, p = next(g)
                            yield dict(phase=phase, progress=p, frame=0, frames=1, eta=0.0)
                        except StopIteration as e:
                            img = e.value
                            break
                    from PIL import Image, PngImagePlugin
                    meta = PngImagePlugin.PngInfo()
                    meta.add_text("mandelscope", json.dumps(loc))
                    Image.fromarray(img).save(path, pnginfo=meta)
                    yield dict(phase="frame", progress=1.0, frame=0, frames=1, eta=0.0, image=img)
                finally:
                    fr.release()
            return gen()
        job = Job("still", Path(path).name, path, make=make)
        self.jobs.append(job)
        if quick:
            self.notify("screenshot queued (2x supersampled)")
        return job

    def _step_jobs(self, slice_s: float):
        job = next((j for j in self.jobs if j.status == "running"), None)
        if job is None:
            return
        t0 = time.perf_counter()
        try:
            if job.gen is None:
                job.gen = job.make()
                job.t0 = time.perf_counter()
            while time.perf_counter() - t0 < slice_s:
                st = next(job.gen)
                job.progress = st.get("progress", job.progress)
                job.frame = st.get("frame", job.frame)
                job.frames = st.get("frames", job.frames)
                job.eta = st.get("eta", job.eta)
                job.codec = st.get("codec") or job.codec
                if "image" in st:
                    self._show_preview(st["image"])
                    break
            job.elapsed = time.perf_counter() - job.t0
        except StopIteration:
            job.status = "done"
            job.progress = 1.0
            job.elapsed = time.perf_counter() - job.t0
            job.gen = None
            self.last_output = job.path
            self.notify(f"finished: {job.name} ({job.elapsed:.1f}s)")
            if job.open_when_done:
                open_path(job.path)
            self._start_next_job()
        except Exception as e:
            if sys.stderr is not None:
                traceback.print_exc()
            job.status = "failed"
            job.error = str(e)[-300:]
            job.gen = None
            self.notify(f"{job.name} failed: {e}")
            self._start_next_job()

    def _start_next_job(self):
        if any(j.status == "running" for j in self.jobs):
            return
        nxt = next((j for j in self.jobs if j.status == "queued"), None)
        if nxt is not None:
            nxt.status = "running"

    def process_queue(self):
        """Start the first queued job if nothing runs (called every frame)."""
        if not any(j.status in ("running",) for j in self.jobs):
            self._start_next_job()

    def pause_job(self, job: Job):
        if job.status == "running":
            job.status = "paused"
        elif job.status == "paused":
            if any(j.status == "running" for j in self.jobs):
                job.status = "queued"
            else:
                job.status = "running"

    def cancel_job(self, job: Job):
        if job.gen is not None:
            try:
                job.gen.close()
            except Exception:
                pass
            job.gen = None
        if job.status in ("running", "paused", "queued"):
            job.status = "cancelled"
        self._start_next_job()

    def _show_preview(self, img):
        h, w = img.shape[:2]
        if self.preview_tex is None or self.preview_tex.size != (w, h):
            if self.preview_tex is not None:
                self.preview_tex.release()
            self.preview_tex = self.ctx.texture((w, h), 3)
            self.preview_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.preview_tex.write(np.ascontiguousarray(img).tobytes())

    # ============================================================= minibrot
    def find_minibrot_at(self, cre, cim):
        if self.julia_c is not None:
            self.notify("minibrots are in the Mandelbrot set (J: back to it)")
            return
        H = self.fb[1]
        log2_radius = self.view.pixel_log2(H) + 5
        p = max(self.view.precision, self.view.needed_precision(H))
        self.minibrot_result = None
        self.minibrot_job = self.aux_worker.submit("minibrot", re=to_str(cre, p), im=to_str(cim, p),
                                                   log2_radius=log2_radius, formula=self.formula)
        self.notify("searching for a mini ship..." if self.formula == "ship" else "searching for a minibrot...")

    def go_minibrot(self, r):
        # (a mini Burning Ship is skewed and its size only estimated: a little more room)
        v = View.from_strings(r["re"], r["im"], r["log2_size"] + (2.5 if self.formula == "ship" else 1.5),
                              self.view.angle, max(int(r["period"]) * 20, 1000))
        self._adopt_iterations(v)
        self.ref_anchor = v.copy()
        self.go_to(v, 0.8)

    # ============================================================= library
    def save_to_library(self, name: str):
        thumb = self.screen_rgb()
        e = self.library.save(name or time.strftime("location %Y-%m-%d %H%M%S"), self.location_dict(), thumb)
        self.notify(f"saved to library: {e.name}")
        return e

    # ===================================================================== UI
    def show_gui(self):
        try:
            from . import ui
            ui.draw(self)
            self.process_queue()
        except Exception:
            self.internal_error("interface")


def _same_point(a, b) -> bool:
    """Two reference orbits start from the same point C (decimal strings, any precision)."""
    if getattr(a, "formula", "mandelbrot") != getattr(b, "formula", "mandelbrot"):
        return False
    try:
        p = max(int(a.prec), int(b.prec), 64)
        return mpfr(a.re, p) == mpfr(b.re, p) and mpfr(a.im, p) == mpfr(b.im, p)
    except Exception:
        return False


def _unique(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return str(p)
    for i in range(1, 10000):
        q = p.with_name(f"{p.stem}_{i:03d}{p.suffix}")
        if not q.exists():
            return str(q)
    return str(p)


def open_path(path: str | Path):
    """Open a file or folder with the system default application."""
    try:
        p = str(path)
        if os.name == "nt":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", p])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ============================================================== self test
def load_fonts(app=None):
    """Roboto + Font Awesome 6 icons, Roboto bold for headings, Inconsolata for scripts."""
    from . import ui
    ui.FONTS["regular"] = hello_imgui.load_font_ttf_with_font_awesome_icons("fonts/Roboto/Roboto-Regular.ttf", 16.0)
    try:
        ui.FONTS["bold"] = hello_imgui.load_font_ttf_with_font_awesome_icons("fonts/Roboto/Roboto-Bold.ttf", 16.0)
        ui.FONTS["mono"] = hello_imgui.load_font_ttf("fonts/Inconsolata-Medium.ttf", 15.0)
    except Exception:
        pass
    # cinematic HUD: Inter (labels, with the icons) and JetBrains Mono (numbers); SIL OFL 1.1
    for key, path, icons in (("hud", "fonts/Inter/Inter-Regular.ttf", True),
                             ("hud_semibold", "fonts/Inter/Inter-SemiBold.ttf", True),
                             ("hud_mono", "fonts/JetBrainsMono/JetBrainsMono-Regular.ttf", False),
                             ("hud_mono_light", "fonts/JetBrainsMono/JetBrainsMono-Light.ttf", False),
                             ("hud_mono_medium", "fonts/JetBrainsMono/JetBrainsMono-Medium.ttf", False)):
        try:
            load = hello_imgui.load_font_ttf_with_font_awesome_icons if icons else hello_imgui.load_font_ttf
            ui.FONTS[key] = load(path, 16.0)
        except Exception:
            pass


def run_app(location=None, size=(1600, 900), selftest=False, selftest_dir="selftest_out"):
    from . import ui
    if selftest:
        os.environ["MANDELSCOPE_HOME"] = str(Path(selftest_dir).resolve() / "home")
    app = App(location, size, selftest, selftest_dir)
    if ASSETS.exists():
        hello_imgui.set_assets_folder(str(ASSETS))
    rp = hello_imgui.RunnerParams()
    rp.app_window_params.window_title = "Mandelscope"
    rp.app_window_params.window_geometry.size = tuple(size)
    rp.app_window_params.restore_previous_geometry = not selftest
    ui_scale = float(os.environ.get("MANDELSCOPE_UI_SCALE") or app.settings.ui_scale or 0.0)
    if ui_scale > 0:
        rp.dpi_aware_params.dpi_window_size_factor = ui_scale
    rp.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    gl = hello_imgui.OpenGlOptions()
    gl.major_version = 4
    gl.minor_version = 3
    gl.use_core_profile = True
    gl.glsl_version = "#version 430"
    rp.renderer_backend_options.open_gl_options = gl
    rp.fps_idling.enable_idling = False
    rp.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    rp.ini_filename = "mandelscope/imgui_layout_v2.ini"
    if selftest:
        rp.ini_disable = True
    rp.callbacks.post_init = app.post_init
    rp.callbacks.custom_background = app.custom_background
    rp.callbacks.show_gui = app.show_gui
    rp.callbacks.before_exit = app.before_exit
    rp.callbacks.before_swap = app.before_swap
    rp.callbacks.setup_imgui_config = app.setup_imgui_config
    rp.callbacks.setup_imgui_style = ui.setup_style
    rp.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
    rp.callbacks.load_additional_fonts = load_fonts
    hello_imgui.run(rp)
    if selftest:
        rep = json.loads((Path(selftest_dir) / "report.json").read_text())
        return 1 if rep["errors"] else 0
    return 0
