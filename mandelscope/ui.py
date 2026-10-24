"""Dear ImGui interface: toolbar, sidebar with tabs, timeline, status bar, overlays."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import moderngl
from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa

from .mp import View
from .palettes import (LOOKS, make_palette, palette_stops, all_palette_names, palette_version, set_custom,
                       delete_custom, random_stops, to_hex, from_hex, CUSTOM)
from .renderer import ColorSettings, MAPPINGS, METHODS, PATTERNS
from .sequence import EASINGS, Sequence
from .script import SCRIPT_HELP
from .video import available_codecs
from .library import user_dir, user_root, location_from_png

V2 = imgui.ImVec2
V4 = imgui.ImVec4
FONTS: dict = {}
CLASSIC_ACCENT = (0.95, 0.68, 0.32)
ACCENT = CLASSIC_ACCENT
S = 1.0               # UI scale (font size / 16 px): follows Windows display scaling
TOOLBAR_H = 44
STATUS_H = 28
TIMELINE_H = 132
TABS = ["Explore", "Colour", "Library", "Animate", "Export", "Render"]
TAB_ICONS = {"Explore": fa.ICON_FA_COMPASS, "Colour": fa.ICON_FA_PALETTE, "Library": fa.ICON_FA_BOOKMARK,
             "Animate": fa.ICON_FA_FILM, "Export": fa.ICON_FA_VIDEO, "Render": fa.ICON_FA_GEAR}
RES_PRESETS = [("720p", 1280, 720), ("1080p", 1920, 1080), ("1440p", 2560, 1440), ("4K", 3840, 2160),
               ("8K", 7680, 4320), ("Vertical 9:16", 1080, 1920), ("Square", 1080, 1080)]
FPS_PRESETS = [24.0, 25.0, 30.0, 50.0, 60.0, 120.0]
_state = {"tab_drawn": None, "edit_stop": 0, "drag_stop": None, "tl_drag": None, "rename": None,
          "codecs": None, "palette_open": False, "editor_open": False, "script_help": False}


def col(r, g, b, a=1.0) -> int:
    return imgui.color_convert_float4_to_u32(V4(r, g, b, a))


ACC_U32 = None


# ----------------------------------------------------------------- style
def setup_style():
    imgui.style_colors_dark()
    st = imgui.get_style()
    st.window_rounding = 8.0
    st.child_rounding = 6.0
    st.frame_rounding = 5.0
    st.popup_rounding = 6.0
    st.grab_rounding = 5.0
    st.tab_rounding = 5.0
    st.scrollbar_rounding = 6.0
    st.window_border_size = 0.0
    st.frame_border_size = 0.0
    st.popup_border_size = 1.0
    st.window_padding = V2(12, 10)
    st.frame_padding = V2(8, 5)
    st.item_spacing = V2(8, 6)
    st.item_inner_spacing = V2(6, 5)
    st.scrollbar_size = 12.0
    st.grab_min_size = 10.0
    ar, ag, ab = ACCENT
    C = imgui.Col_
    colors = {
        C.text: (0.92, 0.92, 0.94, 1.0),
        C.text_disabled: (0.56, 0.58, 0.63, 1.0),
        C.window_bg: (0.07, 0.075, 0.095, 0.90),
        C.child_bg: (0.0, 0.0, 0.0, 0.0),
        C.popup_bg: (0.09, 0.095, 0.115, 0.97),
        C.border: (1.0, 1.0, 1.0, 0.08),
        C.frame_bg: (1.0, 1.0, 1.0, 0.06),
        C.frame_bg_hovered: (1.0, 1.0, 1.0, 0.10),
        C.frame_bg_active: (1.0, 1.0, 1.0, 0.14),
        C.title_bg: (0.07, 0.075, 0.095, 1.0),
        C.title_bg_active: (0.09, 0.095, 0.12, 1.0),
        C.menu_bar_bg: (0.07, 0.075, 0.095, 1.0),
        C.scrollbar_bg: (0.0, 0.0, 0.0, 0.0),
        C.scrollbar_grab: (1.0, 1.0, 1.0, 0.12),
        C.scrollbar_grab_hovered: (1.0, 1.0, 1.0, 0.2),
        C.scrollbar_grab_active: (ar, ag, ab, 0.6),
        C.check_mark: (ar, ag, ab, 1.0),
        C.slider_grab: (ar, ag, ab, 0.85),
        C.slider_grab_active: (ar, ag, ab, 1.0),
        C.button: (1.0, 1.0, 1.0, 0.07),
        C.button_hovered: (ar, ag, ab, 0.35),
        C.button_active: (ar, ag, ab, 0.6),
        C.header: (ar, ag, ab, 0.22),
        C.header_hovered: (ar, ag, ab, 0.32),
        C.header_active: (ar, ag, ab, 0.45),
        C.separator: (1.0, 1.0, 1.0, 0.08),
        C.separator_hovered: (ar, ag, ab, 0.5),
        C.separator_active: (ar, ag, ab, 0.8),
        C.resize_grip: (1.0, 1.0, 1.0, 0.05),
        C.resize_grip_hovered: (ar, ag, ab, 0.4),
        C.resize_grip_active: (ar, ag, ab, 0.7),
        C.tab: (1.0, 1.0, 1.0, 0.04),
        C.tab_hovered: (ar, ag, ab, 0.35),
        C.tab_selected: (ar, ag, ab, 0.25),
        C.tab_selected_overline: (ar, ag, ab, 1.0),
        C.plot_histogram: (ar, ag, ab, 1.0),
        C.text_selected_bg: (ar, ag, ab, 0.35),
        C.nav_cursor: (ar, ag, ab, 0.8),
        C.table_row_bg_alt: (1.0, 1.0, 1.0, 0.03),
    }
    for k, v in colors.items():
        st.set_color_(k, V4(*v))


def _bold():
    f = FONTS.get("hud_semibold") if _state.get("style", ("classic",))[0] == "hud" else None
    f = f or FONTS.get("bold")
    if f is not None:
        imgui.push_font(f, 0.0)
        return True
    return False


def section(title: str, icon: str = ""):
    if _state.get("style", ("classic",))[0] == "hud":
        from . import hud
        hud.section(title)
        return
    imgui.spacing()
    b = _bold()
    imgui.text_colored(V4(*ACCENT, 1.0), f"{icon}  {title}" if icon else title)
    if b:
        imgui.pop_font()
    imgui.separator()


def help_marker(text: str):
    """A (?) after the previous widget - or, if there is no room left on the line,
    the help becomes the tooltip of that widget instead of spilling over the edge."""
    right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
    need = imgui.calc_text_size(fa.ICON_FA_CIRCLE_QUESTION).x + imgui.get_style().item_spacing.x
    if imgui.get_item_rect_max().x + need > right + 0.5:
        if imgui.is_item_hovered():
            imgui.set_tooltip(text)
        return
    imgui.same_line()
    imgui.text_disabled(fa.ICON_FA_CIRCLE_QUESTION)
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def icon_button(icon: str, tip: str, active: bool = False, color=None, size=None) -> bool:
    pushed = 0
    if active:
        imgui.push_style_color(imgui.Col_.button, col(*ACCENT, 0.45))
        pushed += 1
    if color is not None:
        imgui.push_style_color(imgui.Col_.text, col(*color))
        pushed += 1
    r = imgui.button(icon, size or V2(34 * S, 30 * S))
    if pushed:
        imgui.pop_style_color(pushed)
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
    return r


def tex_ref(tex) -> imgui.ImTextureRef:
    return imgui.ImTextureRef(tex.glo)


# textures rendered by the GPU (framebuffers) store the bottom row first: show them upright
FBO_UV = (imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))


# ----------------------------------------------------------------- textures
def palette_strip_tex(app, name: str):
    key = (name, palette_version(name))
    t = app.strip_cache.get(key)
    if t is None:
        for k in [k for k in app.strip_cache if k[0] == name]:
            app.strip_cache.pop(k).release()
        rgb = make_palette(name, 256)
        t = app.ctx.texture((256, 1), 3, data=np.clip(rgb * 255, 0, 255).astype(np.uint8).tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        app.strip_cache[key] = t
    return t


def image_tex(app, path: Path | None):
    if path is None or not Path(path).exists():
        return None
    key = (str(path), Path(path).stat().st_mtime)
    t = app.thumb_cache.get(key)
    if t is None:
        from PIL import Image
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            return None
        t = app.ctx.texture(im.size, 3, data=im.tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        for k in [k for k in app.thumb_cache if k[0] == str(path)]:
            app.thumb_cache.pop(k).release()
        app.thumb_cache[key] = t
    return t


# ================================================================== draw
def apply_style(app):
    """Switch the theme when the interface style (or the HUD accent) changed."""
    global ACCENT
    s = app.settings
    hud_ok = "hud" in FONTS
    want = ("hud", s.hud_accent) if (s.ui_style == "hud" and hud_ok) else ("classic",)
    if _state.get("style") == want:
        return want[0]
    from . import hud
    if want[0] == "hud":
        ACCENT = hud.accent(app)
        hud.setup_style(ACCENT)
    else:
        ACCENT = CLASSIC_ACCENT
        setup_style()
    _state["style"] = want
    return want[0]


WELCOME = [
    (fa.ICON_FA_COMPUTER_MOUSE, "Wheel", "zoom in / out at the cursor (also + and -)"),
    (fa.ICON_FA_HAND, "Drag", "move around - it glides when you let go"),
    (fa.ICON_FA_ROTATE, "Right-drag", "rotate (Shift snaps to 15 degrees)"),
    (fa.ICON_FA_MAGNIFYING_GLASS_PLUS, "Double-click", "zoom x4 into that point"),
    (fa.ICON_FA_BOOKMARK, "Library", "famous places, and the ones you save"),
    (fa.ICON_FA_SNOWFLAKE, "J", "Julia set of the point under the cursor"),
    (fa.ICON_FA_GAMEPAD, "Controller", "left stick moves, R2 / L2 zoom, right stick turns"),
    (fa.ICON_FA_KEYBOARD, "F1", "every key and button; Tab hides the panels"),
]


def _welcome(app, W, H):
    """First-run card with the basic controls: a modal (the picture keeps rendering behind it),
    closed for good with either button."""
    if app.settings.welcome_seen or (app.selftest and not getattr(app, "force_welcome", False)):
        return
    if not imgui.is_popup_open("Welcome##card"):
        imgui.open_popup("Welcome##card")
    imgui.set_next_window_pos(V2(W / 2, H / 2), imgui.Cond_.always, V2(0.5, 0.5))
    imgui.set_next_window_size(V2(min(W - 40, 560 * S), 0))
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(24 * S, 20 * S))
    opened = imgui.begin_popup_modal("Welcome##card", None, imgui.WindowFlags_.no_title_bar |
                                     imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move |
                                     imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize)
    imgui.pop_style_var()
    if not (opened[0] if isinstance(opened, tuple) else opened):
        return
    b = _bold()
    imgui.text_colored(V4(*ACCENT, 1.0), "Welcome to Mandelscope")
    if b:
        imgui.pop_font()
    imgui.push_text_wrap_pos(0.0)
    imgui.text_disabled("Dive into the Mandelbrot set - as deep as you like. Every frame is computed live "
                        "on your graphics card, down to magnifications of 10^1000 and beyond.")
    imgui.pop_text_wrap_pos()
    imgui.dummy(V2(1, 8 * S))
    if imgui.begin_table("##keys", 2, imgui.TableFlags_.sizing_fixed_fit):
        for icon, key, what in WELCOME:
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            imgui.text(f"{icon}  {key}   ")
            imgui.table_set_column_index(1)
            imgui.text_disabled(what)
        imgui.end_table()
    imgui.dummy(V2(1, 10 * S))
    w = (imgui.get_content_region_avail().x - 8 * S) / 2
    imgui.push_style_color(imgui.Col_.button, V4(*ACCENT, 0.30))
    close = imgui.button("Start exploring", V2(w, 36 * S))
    imgui.pop_style_color()
    imgui.same_line(0, 8 * S)
    fly = imgui.button(fa.ICON_FA_WAND_MAGIC_SPARKLES + "  Fly to a famous place", V2(w, 36 * S))
    if close or fly:
        app.settings.welcome_seen = True
        app.force_welcome = False
        imgui.close_current_popup()
    if fly:
        built = [e for e in app.library.entries if e.builtin and e.name != "home"]
        if built:
            import random
            app.apply_location_dict(random.choice(built).load(), animate=True)
    imgui.end_popup()


def draw(app):
    global S, TOOLBAR_H, STATUS_H, TIMELINE_H
    S = max(0.75, hello_imgui.em_size() / 16.0)
    TOOLBAR_H, STATUS_H, TIMELINE_H = int(round(44 * S)), int(round(28 * S)), int(round(132 * S))
    io = imgui.get_io()
    W, H = io.display_size.x, io.display_size.y
    app.colour_tab_visible = False
    if apply_style(app) == "hud":
        from . import hud
        TOOLBAR_H, STATUS_H = int(round(84 * S)), 0      # (julia preview below the wordmark; no status bar)
        hud.draw(app, W, H)
        _welcome(app, W, H)
        return
    _context_menu(app)
    if app.ui_visible:
        _toolbar(app, W)
        tl_h = TIMELINE_H if app.settings.show_timeline else 0
        if app.settings.sidebar_open and not app.show_help:         # the help card is wide
            _sidebar(app, W, H, tl_h)
        if tl_h:
            _timeline(app, W, H)
        _status_bar(app, W, H)
        _julia_preview(app)
    _toasts(app, H)
    _welcome(app, W, H)
    if app.show_help:
        _help_overlay(app, W, H)
    if _state["script_help"]:
        _script_help(app, W, H)


FIXED = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move |
         imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings |
         imgui.WindowFlags_.no_bring_to_front_on_focus)


# ----------------------------------------------------------------- toolbar
def _toolbar(app, W):
    imgui.set_next_window_pos(V2(0, 0))
    imgui.set_next_window_size(V2(W, TOOLBAR_H))
    imgui.set_next_window_bg_alpha(0.86)
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(8 * S, 7 * S))
    imgui.begin("##toolbar", None, FIXED | imgui.WindowFlags_.no_scrollbar)
    b = _bold()
    imgui.text_colored(V4(*ACCENT, 1.0), fa.ICON_FA_INFINITY + "  Mandelscope")
    if b:
        imgui.pop_font()
    imgui.same_line(0, 18 * S)
    if icon_button(fa.ICON_FA_HOUSE, "Home - whole set (H)"):
        app.go_home()
    imgui.same_line()
    if icon_button(fa.ICON_FA_ARROW_LEFT, "Back (Backspace)"):
        app.back()
    imgui.same_line()
    if icon_button(fa.ICON_FA_ARROW_RIGHT, "Forward (Shift+Backspace)"):
        app.forward()
    imgui.same_line(0, 14 * S)
    if icon_button(fa.ICON_FA_MAGNIFYING_GLASS_PLUS, "Zoom in x2 (+, or Z at the cursor)"):
        app.zoom_anim += 1.0
        app.zoom_focus = (app.fb[0] / 2, app.fb[1] / 2)
    imgui.same_line()
    if icon_button(fa.ICON_FA_MAGNIFYING_GLASS_MINUS, "Zoom out x2 (-, or X at the cursor)"):
        app.zoom_anim -= 1.0
        app.zoom_focus = (app.fb[0] / 2, app.fb[1] / 2)
    imgui.same_line(0, 14 * S)
    if icon_button(fa.ICON_FA_ROTATE_LEFT, "Rotate 90 degrees left ([)"):
        app.rotate_by(math.radians(-90))
    imgui.same_line()
    if imgui.button("0" + "°", V2(34 * S, 30 * S)):
        app.rotate_to(0.0)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Reset rotation to 0 degrees (0)")
    imgui.same_line()
    if icon_button(fa.ICON_FA_ROTATE_RIGHT, "Rotate 90 degrees right (])"):
        app.rotate_by(math.radians(90))
    imgui.same_line(0, 14 * S)
    if icon_button(fa.ICON_FA_CROSSHAIRS, "Find the nearest minibrot (M at the cursor)"):
        app.find_minibrot_at(app.view.re, app.view.im)
    imgui.same_line()
    if icon_button(fa.ICON_FA_BOOKMARK, "Save this location to the library"):
        app.save_to_library("")
    imgui.same_line(0, 14 * S)
    if app.julia_c is None:
        if icon_button(fa.ICON_FA_SNOWFLAKE, "Julia set preview for the point under the cursor (J)\n"
                       "Shift+J: explore the Julia set of that point", app.settings.julia_preview):
            app.settings.julia_preview = not app.settings.julia_preview
    else:
        if icon_button(fa.ICON_FA_ARROW_TURN_UP, "Back to the Mandelbrot set (J)", True, (0.6, 0.85, 1.0, 1.0)):
            app.leave_julia()
    # right-hand group
    right = [(fa.ICON_FA_PAUSE if app.playing else fa.ICON_FA_PLAY, "Play / pause the timeline (Space)", app.playing, None, app.toggle_play),
             (fa.ICON_FA_DIAMOND, "Add keyframe at the current view (K)", False, None, app.add_keyframe),
             (fa.ICON_FA_CIRCLE, "Record the screen live (R)", app.live is not None,
              (1.0, 0.35, 0.35, 1.0), app.toggle_live),
             (fa.ICON_FA_ROUTE, "Record your flight as a timeline (P)", app.path_rec is not None,
              (0.45, 1.0, 0.55, 1.0) if app.path_rec is not None else None, app.toggle_path_recording),
             (fa.ICON_FA_CAMERA, "Screenshot, 2x supersampled (F12)", False, None, lambda: app.start_still(quick=True)),
             (fa.ICON_FA_TIMELINE, "Show / hide the timeline (T)", app.settings.show_timeline, None,
              lambda: setattr(app.settings, "show_timeline", not app.settings.show_timeline)),
             (fa.ICON_FA_TABLE_COLUMNS, "Show / hide the side panel (Tab hides everything)", app.settings.sidebar_open, None,
              lambda: setattr(app.settings, "sidebar_open", not app.settings.sidebar_open)),
             (fa.ICON_FA_KEYBOARD, "Keys, mouse and controller (F1)", app.show_help, None,
              lambda: setattr(app, "show_help", not app.show_help))]
    x = W - 8 * S - len(right) * 40 * S
    imgui.same_line(max(imgui.get_cursor_pos_x() + 10, x))
    for i, (ic, tip, active, color, fn) in enumerate(right):
        if i:
            imgui.same_line(0, 6 * S)
        if icon_button(ic, tip, active, color):
            fn()
    imgui.end()
    imgui.pop_style_var(2)


# ----------------------------------------------------------------- sidebar
def _sidebar(app, W, H, tl_h):
    sw = app.settings.sidebar_width * S
    y0 = TOOLBAR_H + 6
    h = H - y0 - STATUS_H - tl_h - 12
    imgui.set_next_window_pos(V2(W - sw - 6, y0))
    imgui.set_next_window_size(V2(sw, max(200, h)))
    imgui.set_next_window_bg_alpha(0.90)
    imgui.begin("##sidebar", None, FIXED)
    _tab_strip(app)
    name = app.settings.sidebar_tab if app.settings.sidebar_tab in TABS else TABS[0]
    imgui.begin_child("##tabbody", V2(0, 0))
    {"Explore": _tab_explore, "Colour": _tab_colour, "Library": _tab_library,
     "Animate": _tab_animate, "Export": _tab_export, "Render": _tab_render}[name](app)
    imgui.end_child()
    imgui.end()


def _tab_strip(app):
    """Segmented tab control: the active tab shows icon + name, the others just icons."""
    s = app.settings
    avail = imgui.get_content_region_avail().x
    gap = 4.0
    n = len(TABS)
    sel_w = min(124.0 * S, avail * 0.36)
    other_w = (avail - sel_w - gap * (n - 1)) / (n - 1)
    ar, ag, ab = ACCENT
    for i, name in enumerate(TABS):
        if i:
            imgui.same_line(0, gap)
        active = s.sidebar_tab == name
        if active:
            imgui.push_style_color(imgui.Col_.button, V4(ar, ag, ab, 0.26))
            imgui.push_style_color(imgui.Col_.button_hovered, V4(ar, ag, ab, 0.34))
            imgui.push_style_color(imgui.Col_.text, V4(1.0, 0.86, 0.62, 1.0))
            label = f"{TAB_ICONS[name]}  {name}###tab{name}"
        else:
            label = f"{TAB_ICONS[name]}###tab{name}"
        if imgui.button(label, V2(sel_w if active else other_w, 30 * S)):
            s.sidebar_tab = name
        if active:
            imgui.pop_style_color(3)
        elif imgui.is_item_hovered():
            imgui.set_tooltip(name)
    imgui.dummy(V2(0, 2 * S))


def _full(width_frac=1.0):
    imgui.push_item_width(-1 if width_frac >= 1.0 else imgui.get_content_region_avail().x * width_frac)


# --------------------------------------------------------------- Explore
_PREC_ORDER = (0, 2, 1)          # menu order: float32, float-float, float64 (settings keep 0 / 1 / 2)
_PREC_LABELS = ("float32 (fast)", "float-float (precise, fast)", "float64 (precise)")


_ITER_MULTS = (1, 2, 4, 8, 16, 64, 256)


def _iter_mult_combo(label: str, value: int, now: int | None) -> int:
    """Iteration limit of an export: the explorer's, or that many times more (more detail at
    the edge of the set and inside minibrots; time grows mostly where the extra iterations
    are needed)."""
    i = _ITER_MULTS.index(value) if value in _ITER_MULTS else 0
    names = ["as in the explorer"] + [f"{m}x the explorer's" for m in _ITER_MULTS[1:]]
    _, i = imgui.combo(label, i, names)
    if imgui.is_item_hovered():
        tip = ("The iteration limit of the export.  More iterations show more detail at the edge\n"
               "of the set and in minibrots, where pixels need longer to escape; elsewhere they\n"
               "cost nothing.  Very large limits need a long reference orbit (about 1 us and\n"
               "70 bytes per iteration, computed before the picture - the queue shows it),\n"
               "except at minibrots, whose periodic orbit is stored only once.")
        if now:
            tip += f"\n\nHere: up to {now * _ITER_MULTS[i]:,} iterations."
        imgui.set_tooltip(tip)
    return _ITER_MULTS[i]


def _precision_combo(label: str, value: int) -> int:
    i = _PREC_ORDER.index(value) if value in _PREC_ORDER else 0
    _, i = imgui.combo(label, i, list(_PREC_LABELS))
    return _PREC_ORDER[i]


def _tab_explore(app):
    v = app.view
    s = app.settings
    section("Fractal", fa.ICON_FA_SAILBOAT)
    bw = (imgui.get_content_region_avail().x - 8) / 2
    for i, (f, label) in enumerate((("mandelbrot", "Mandelbrot set"), ("ship", "Burning Ship"))):
        if i:
            imgui.same_line()
        on = app.formula == f
        if on:
            imgui.push_style_color(imgui.Col_.button, V4(*ACCENT, 0.55))
        if imgui.button(label, V2(bw, 0)) and not on:
            app.set_formula(f)
        if on:
            imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip("B switches.  The Burning Ship: z -> (|Re z| + i|Im z|)² + c\n"
                          "(Michelitsch and Rössler, 1992), drawn upright.  Each fractal keeps\n"
                          "its own view; locations remember which one they belong to.")
    v = app.view
    section("Location", fa.ICON_FA_LOCATION_DOT)
    imgui.text_disabled("real")
    _full()
    _, app.edit_re = imgui.input_text("##re", app.edit_re)
    imgui.pop_item_width()
    imgui.text_disabled("imaginary")
    _full()
    _, app.edit_im = imgui.input_text("##im", app.edit_im)
    imgui.pop_item_width()
    imgui.text_disabled("zoom (log10 of magnification)")
    imgui.push_item_width(150 * S)
    ch, app.edit_zoom = imgui.input_text("##zoom", app.edit_zoom, imgui.InputTextFlags_.enter_returns_true)
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button(fa.ICON_FA_LOCATION_ARROW + " Go") or ch:
        try:
            l10 = float(app.edit_zoom)
            nv = View.from_strings(app.edit_re, app.edit_im, 1.0 - l10 / 0.30102999566398, v.angle, v.max_iter)
            app.go_to(app.apply_auto_iter(nv), 0.6)
        except Exception as e:
            app.notify(f"invalid location: {e}")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_COPY):
        imgui.set_clipboard_text(json.dumps(app.location_dict(), indent=1))
        app.notify("location copied (Ctrl+C)")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Copy location (Ctrl+C)")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_PASTE):
        app.paste_location()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Paste location (Ctrl+V)")
    imgui.text_disabled(f"magnification 10^{v.zoom_magnification_log10():.2f}   "
                        f"precision {v.needed_precision(app.fb[1] or 1000)} bits")

    section("Rotation", fa.ICON_FA_ROTATE)
    deg = (math.degrees(v.angle) + 180.0) % 360.0 - 180.0
    imgui.push_item_width(-1)
    ch, nd = imgui.slider_float("##rot", deg, -180.0, 180.0, "%.1f°")
    imgui.pop_item_width()
    if ch:
        nv = v.copy()
        nv.angle = v.angle + math.radians(nd - deg)
        app.view = nv
        app.mark_moving()
    bw = (imgui.get_content_region_avail().x - 16) / 3
    if imgui.button(fa.ICON_FA_ROTATE_LEFT + "  90°", V2(bw, 0)):
        app.rotate_by(math.radians(-90))
    imgui.same_line()
    if imgui.button("0°", V2(bw, 0)):
        app.rotate_to(0.0)
    imgui.same_line()
    if imgui.button("90°  " + fa.ICON_FA_ROTATE_RIGHT, V2(bw, 0)):
        app.rotate_by(math.radians(90))
    imgui.text_disabled("Right-drag rotates, Shift snaps to 15°.  Keys [ ] 0  Q E")

    section("Julia sets", fa.ICON_FA_SNOWFLAKE)
    if app.formula != "mandelbrot":
        imgui.text_disabled("Julia sets: for the Mandelbrot set (B switches back).")
    elif app.julia_c is None:
        _, s.julia_preview = imgui.checkbox("preview for the point under the cursor (J)", s.julia_preview)
        if imgui.button(fa.ICON_FA_SNOWFLAKE + "  Explore the Julia set of the centre", V2(-1, 0)):
            app.enter_julia(v.re, v.im)
        imgui.text_disabled("Every point c has a Julia set (z -> z² + c, start z varies).\n"
                            "Inside the Mandelbrot set it is connected, outside it is dust;\n"
                            "on the edge it looks like the Mandelbrot set around c.\n"
                            "Shift+J explores the one under the cursor.")
    else:
        c = app.julia_c
        imgui.text(f"c = {c[0]:.12g}")
        imgui.text(f"    {'+' if c[1] >= 0 else '-'} {abs(c[1]):.12g} i")
        if imgui.button(fa.ICON_FA_ARROW_TURN_UP + "  Back to the Mandelbrot set (J)", V2(-1, 0)):
            app.leave_julia()
        imgui.text_disabled("Deep zooms use the perturbation engine (a reference\n"
                            "orbit at the view's point and the orbit of 0): no depth limit.")

    section("Iterations", fa.ICON_FA_LAYER_GROUP)
    ch, auto = imgui.checkbox("automatic", s.auto_iter)
    if ch:
        s.auto_iter = auto
        app.manual_max_iter = v.max_iter
        app.iter_boost = 1.0
        app.iter_floor = 1.0
        app.view = app.apply_auto_iter(v.copy())
    help_marker("Iterations grow with depth: base + gain x decades of zoom.\n"
                "When the limit cuts off outside detail the program doubles it, and lowers it\n"
                "again later when the extra iterations change nothing.  Pixels at the edge of\n"
                "a black component that converge very slowly stay undecided (shown as inside).")
    if s.auto_iter:
        imgui.push_item_width(-130 * S)
        c1, s.iter_base = imgui.input_int("at the start", s.iter_base, 100, 1000)
        c2, s.iter_gain = imgui.input_int("more per decade", s.iter_gain, 50, 500)
        imgui.pop_item_width()
        s.iter_base = max(100, s.iter_base)
        s.iter_gain = max(0, s.iter_gain)
        if c1 or c2:
            app.view = app.apply_auto_iter(app.view.copy())
    else:
        imgui.push_item_width(160 * S)
        ch, mi = imgui.input_int("limit", v.max_iter, 1000, 100000)
        imgui.pop_item_width()
        if ch and mi > 10:
            nv = v.copy()
            nv.max_iter = mi
            app.manual_max_iter = mi
            app.view = nv
    imgui.same_line()
    if imgui.small_button("x2"):
        _scale_iter(app, 2.0)
    imgui.same_line()
    if imgui.small_button("/2"):
        _scale_iter(app, 0.5)
    f = app.field
    if f is not None and f.params is not None:
        imgui.text_disabled(f"now {app.view.max_iter:,}   escapes {f.min_iter or 0:,} .. {f.max_iter_seen or 0:,}"
                            f"   undecided {f.maxiter_count:,}")

    section("Navigation", fa.ICON_FA_HAND)
    imgui.push_item_width(-150 * S)
    _, s.zoom_speed = imgui.slider_float("zoom speed", s.zoom_speed, 0.25, 8.0, "%.2f oct/s")
    _, s.wheel_step = imgui.slider_float("wheel step", s.wheel_step, 0.1, 2.0, "%.2f oct")
    _, s.pan_speed = imgui.slider_float("key pan speed", s.pan_speed, 100.0, 3000.0, "%.0f px/s")
    _, s.frame_pacing = imgui.checkbox("even motion steps", s.frame_pacing)
    help_marker("Moves the camera by the display interval instead of the (jittery) time between\n"
                "frame starts, so zooming and gliding advance evenly on the screen.")
    _, s.inertia = imgui.checkbox("glide after dragging", s.inertia)
    if s.inertia:
        _, s.inertia_time = imgui.slider_float("glide time", s.inertia_time, 0.1, 1.5, "%.2f s")
    imgui.pop_item_width()

    section("Controller", fa.ICON_FA_GAMEPAD)
    _, s.pad_enabled = imgui.checkbox("use a game controller", s.pad_enabled)
    help_marker("PlayStation, Xbox, Switch Pro and other pads (USB or Bluetooth).\n"
                "Left stick moves, R2 / L2 zoom, right stick zooms and turns.\n"
                "F1 lists every button.")
    g = app.gamepad
    if s.pad_enabled:
        if g is not None and g.name:
            imgui.text_colored(V4(*ACCENT, 1.0), fa.ICON_FA_GAMEPAD + "  " + g.name)
        else:
            imgui.text_disabled((g.error if g is not None and g.error else "no controller connected")
                                if g is not None else "not available")
        imgui.push_item_width(-150 * S)
        _, s.pad_speed = imgui.slider_float("controller speed", s.pad_speed, 0.25, 4.0, "%.2fx")
        _, s.pad_deadzone = imgui.slider_float("stick dead zone", s.pad_deadzone, 0.0, 0.5, "%.2f")
        imgui.pop_item_width()
        _, s.pad_invert_zoom = imgui.checkbox("right stick down zooms in", s.pad_invert_zoom)

    section("Find", fa.ICON_FA_CROSSHAIRS)
    if imgui.button(fa.ICON_FA_MAGNIFYING_GLASS + "  Find minibrot near the centre"):
        app.find_minibrot_at(v.re, v.im)
    help_marker("Ball-period detection + Newton's method find the nearest small copy\n"
                "of the set. Press M to search under the mouse cursor.")
    if app.minibrot_job is not None:
        imgui.same_line()
        imgui.text_disabled("searching...")
    r = app.minibrot_result
    if r:
        imgui.text(f"period {r['period']:,}   size 10^{r['log2_size'] * 0.30103:.1f}")
        imgui.same_line()
        if imgui.button("Go to minibrot"):
            app.go_minibrot(r)
    imgui.spacing()
    imgui.text_disabled(f"history: {len(app.history)} back, {len(app.future)} forward")


def _scale_iter(app, k):
    app.scale_iter(k)


# ---------------------------------------------------------------- Colour
def _tab_colour(app):
    app.colour_tab_visible = True
    c = app.colors
    section("Looks", fa.ICON_FA_WAND_MAGIC_SPARKLES)
    imgui.text_disabled("Live previews of this view - click to apply")
    avail = imgui.get_content_region_avail().x
    n_cols = 2
    tw = (avail - 8 * (n_cols - 1)) / n_cols
    pv = app.previews
    th = tw * pv.h / pv.w
    for i, name in enumerate(LOOKS):
        if i % n_cols:
            imgui.same_line()
        imgui.begin_group()
        t = pv.tex(name)
        selected = (c.look == name)
        p0 = imgui.get_cursor_screen_pos()
        if t is not None:
            imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(3 * S, 3 * S))
            clicked = imgui.image_button(f"##look{i}", tex_ref(t), V2(tw - 6, th - 6), *FBO_UV)
            imgui.pop_style_var()
        else:
            clicked = imgui.button(f"{name}##lk{i}", V2(tw, th))
        if selected:
            dl = imgui.get_window_draw_list()
            dl.add_rect(p0, V2(p0.x + tw, p0.y + th), col(*ACCENT, 1.0), 6.0, 2.5)
        if clicked:
            app.colors = ColorSettings.from_look(name, app.colors)
        imgui.text(("  " + fa.ICON_FA_CHECK + " " if selected else "  ") + name)
        imgui.end_group()
    if imgui.button(fa.ICON_FA_DICE + "  Surprise me"):
        _random_look(app)
    help_marker("Random palette (saved as a custom gradient) with random shading.")

    section("Palette", fa.ICON_FA_SWATCHBOOK)
    _palette_strip(app, c.palette, V2(imgui.get_content_region_avail().x, 22))
    if imgui.begin_combo("##palette", c.palette, imgui.ComboFlags_.height_large):
        for name in all_palette_names():
            t = palette_strip_tex(app, name)
            sel = imgui.selectable(f"##p_{name}", name == c.palette, 0, V2(0, 20 * S))[0]
            imgui.same_line(8)
            imgui.image(tex_ref(t), V2(120 * S, 16 * S))
            imgui.same_line()
            imgui.text(name + ("  (custom)" if name in CUSTOM else ""))
            if sel:
                c.palette = name
        imgui.end_combo()
    imgui.same_line()
    ch, _state["editor_open"] = imgui.checkbox("edit", _state["editor_open"])
    if _state["editor_open"]:
        _gradient_editor(app)

    section("Colouring", fa.ICON_FA_BRUSH)
    for i, m in enumerate(METHODS):
        if i:
            imgui.same_line()
        if imgui.radio_button(m, c.method == i):
            c.method = i
    help_marker("iterations: bands by escape time (classic)\n"
                "distance: colour by distance to the set - soft glowing halos")
    imgui.push_item_width(-120 * S)
    if c.method == 0:
        ch, c.mapping = imgui.combo("mapping", c.mapping, MAPPINGS)
        help_marker("auto: colours spread evenly over the image at any depth (histogram)\n"
                    "log / sqrt / linear: fixed spacing (set the density)")
        if c.mapping == 3:
            _, c.auto_cycles = imgui.slider_float("cycles", c.auto_cycles, 0.5, 20.0, "%.1f")
        else:
            ch, d = imgui.slider_float("density", math.log10(max(c.density, 1e-6)), -4.0, 1.0, "")
            if ch:
                c.density = 10 ** d
            imgui.same_line()
            imgui.text(f"{c.density:.4f}")
    else:
        _, c.de_density = imgui.slider_float("cycles / octave", c.de_density, 0.01, 1.0, "%.3f")
    _, c.offset = imgui.slider_float("offset", c.offset % 1.0, 0.0, 1.0, "%.3f")
    _, c.cycle_speed = imgui.slider_float("animate", c.cycle_speed, -0.5, 0.5, "%.2f cyc/s")
    _, c.relative_iter = imgui.checkbox("start colours at the lowest iteration", c.relative_iter)
    imgui.pop_item_width()

    section("Texture", fa.ICON_FA_WATER)
    imgui.push_item_width(-120 * S)
    _, c.stripes = imgui.slider_float("stripes", c.stripes, 0.0, 2.0, "%.2f")
    help_marker("Stripe-average colouring: silky flowing lines (needs a little more GPU time).")
    if c.stripes > 0:
        _, c.stripe_freq = imgui.slider_float("stripe density", c.stripe_freq, 1.0, 12.0, "%.1f")
    _, c.pattern = imgui.combo("pattern", c.pattern, PATTERNS)
    if c.pattern:
        _, c.pattern_strength = imgui.slider_float("strength", c.pattern_strength, 0.0, 1.0, "%.2f")
        if c.pattern >= 2:
            _, c.pattern_freq = imgui.slider_float("lines", c.pattern_freq, 1.0, 16.0, "%.0f")
    imgui.pop_item_width()

    section("Light & edges", fa.ICON_FA_SUN)
    imgui.push_item_width(-120 * S)
    _, c.slope_strength = imgui.slider_float("3D relief", c.slope_strength, 0.0, 1.0, "%.2f")
    if c.slope_strength > 0:
        _, c.light_angle = imgui.slider_float("light angle", c.light_angle % 360.0, 0.0, 360.0, "%.0f°")
        _, c.light_height = imgui.slider_float("light height", c.light_height, 0.1, 4.0, "%.2f")
    _, c.de_strength = imgui.slider_float("dark edges", c.de_strength, 0.0, 1.0, "%.2f")
    _, c.de_scale = imgui.slider_float("edge width", c.de_scale, 0.05, 4.0, "%.2f")
    _, c.glow = imgui.slider_float("glow", c.glow, 0.0, 1.0, "%.2f")
    if c.glow > 0:
        ch, g = imgui.color_edit3("glow colour", list(c.glow_color))
        if ch:
            c.glow_color = tuple(g)
    ch, ic = imgui.color_edit3("inside colour", list(c.interior))
    if ch:
        c.interior = tuple(ic)
    imgui.pop_item_width()


def _random_look(app):
    i = 1
    while f"Random {i}" in CUSTOM:
        i += 1
    name = f"Random {i}"
    set_custom(name, random_stops())
    rng = np.random.default_rng()
    c = app.colors
    c.palette = name
    c.look = name
    c.mapping = 3
    c.auto_cycles = float(rng.choice([2.0, 3.0, 4.0, 6.0, 8.0]))
    c.slope_strength = float(rng.choice([0.0, 0.3, 0.6]))
    c.de_strength = float(rng.choice([0.2, 0.5, 0.8]))
    c.stripes = float(rng.choice([0.0, 0.0, 0.6]))
    c.glow = float(rng.choice([0.0, 0.0, 0.3]))
    c.light_angle = float(rng.uniform(0, 360))


def _palette_strip(app, name, size):
    t = palette_strip_tex(app, name)
    imgui.image(tex_ref(t), size)


def _gradient_editor(app):
    c = app.colors
    if c.palette not in CUSTOM:
        imgui.text_disabled("Built-in palettes are read-only.")
        if imgui.button(fa.ICON_FA_PEN + "  Edit a copy"):
            name = f"My {c.palette}"
            k = 2
            while name in CUSTOM:
                name = f"My {c.palette} {k}"
                k += 1
            stops = palette_stops(c.palette) or random_stops()
            set_custom(name, list(stops))
            c.palette = name
        imgui.same_line()
        if imgui.button(fa.ICON_FA_DICE + "  New random"):
            _random_look(app)
        return
    stops = list(CUSTOM[c.palette])
    sel = min(_state["edit_stop"], len(stops) - 1)
    w = imgui.get_content_region_avail().x
    p0 = imgui.get_cursor_screen_pos()
    _palette_strip(app, c.palette, V2(w, 26))
    # handles
    dl = imgui.get_window_draw_list()
    hy = p0.y + 28
    imgui.invisible_button("##handles", V2(w, 16))
    mp = imgui.get_io().mouse_pos
    for i, (pos, hx) in enumerate(stops):
        x = p0.x + pos * w
        r, g, b = from_hex(hx)
        dl.add_triangle_filled(V2(x, hy), V2(x - 7, hy + 13), V2(x + 7, hy + 13), col(r, g, b, 1.0))
        dl.add_triangle(V2(x, hy), V2(x - 7, hy + 13), V2(x + 7, hy + 13),
                        col(*ACCENT, 1.0) if i == sel else col(1, 1, 1, 0.7), 2.0 if i == sel else 1.0)
    if imgui.is_item_clicked(0):
        best = min(range(len(stops)), key=lambda i: abs(p0.x + stops[i][0] * w - mp.x))
        if abs(p0.x + stops[best][0] * w - mp.x) < 10:
            sel = best
            _state["drag_stop"] = best
        else:
            # add a stop at the click, colour sampled from the gradient
            pos = min(0.999, max(0.0, (mp.x - p0.x) / w))
            rgb = make_palette(c.palette, 1024)[int(pos * 1023)]
            stops.append((pos, to_hex(rgb)))
            stops.sort(key=lambda t: t[0])
            sel = [s[0] for s in stops].index(pos)
            set_custom(c.palette, stops)
    if _state["drag_stop"] is not None:
        if imgui.is_mouse_down(0):
            i = _state["drag_stop"]
            if i < len(stops):
                pos = min(0.999, max(0.0, (mp.x - p0.x) / w))
                hexc = stops[i][1]
                stops[i] = (pos, hexc)
                stops_sorted = sorted(stops, key=lambda t: t[0])
                sel = stops_sorted.index((pos, hexc))
                _state["drag_stop"] = sel
                set_custom(c.palette, stops_sorted)
                stops = stops_sorted
        else:
            _state["drag_stop"] = None
    _state["edit_stop"] = sel
    imgui.text_disabled("click below the strip to add a stop, drag stops to move")
    pos, hx = stops[sel]
    ch, rgb = imgui.color_edit3(f"stop {sel + 1}", list(from_hex(hx)))
    if ch:
        stops[sel] = (pos, to_hex(rgb))
        set_custom(c.palette, stops)
    if imgui.button(fa.ICON_FA_TRASH + " stop") and len(stops) > 2:
        stops.pop(sel)
        _state["edit_stop"] = max(0, sel - 1)
        set_custom(c.palette, stops)
    imgui.same_line()
    if imgui.button(fa.ICON_FA_ARROWS_LEFT_RIGHT + " reverse"):
        set_custom(c.palette, [((1.0 - p) % 1.0, h) for p, h in stops])
    imgui.same_line()
    if imgui.button(fa.ICON_FA_GRIP_LINES_VERTICAL + " even"):
        n = len(stops)
        set_custom(c.palette, [(i / n, h) for i, (_, h) in enumerate(stops)])
    imgui.same_line()
    if imgui.button(fa.ICON_FA_DICE):
        set_custom(c.palette, random_stops())
    if imgui.is_item_hovered():
        imgui.set_tooltip("Randomise this gradient")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_TRASH + " palette"):
        delete_custom(c.palette)
        c.palette = "Gilded"


# --------------------------------------------------------------- Library
def _tab_library(app):
    lib = app.library
    section("Save this view", fa.ICON_FA_FLOPPY_DISK)
    imgui.push_item_width(-90 * S)
    _, app.save_name = imgui.input_text("##savename", app.save_name)
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button(fa.ICON_FA_FLOPPY_DISK + " Save", V2(80 * S, 0)):
        app.save_to_library(app.save_name.strip())
        app.save_name = ""
    imgui.text_disabled(f"saved in {user_dir()}")
    mine = [e for e in lib.entries if not e.builtin]
    built = [e for e in lib.entries if e.builtin]
    section(f"My locations ({len(mine)})", fa.ICON_FA_STAR)
    if not mine:
        imgui.text_disabled("Nothing saved yet - use Save above or the bookmark button.")
    _grid(app, mine, "m")
    section("Built-in", fa.ICON_FA_MAP)
    _grid(app, built, "b")
    section("Open", fa.ICON_FA_FOLDER_OPEN)
    if imgui.button(fa.ICON_FA_FILE_IMPORT + "  Open location or Mandelscope image..."):
        _open_any(app)
    if imgui.button(fa.ICON_FA_FOLDER_OPEN + "  Open the locations folder"):
        from .app import open_path
        user_dir().mkdir(parents=True, exist_ok=True)
        open_path(user_dir())
    if imgui.button(fa.ICON_FA_ARROWS_ROTATE + "  Refresh"):
        lib.refresh()


def _grid(app, entries, tag):
    avail = imgui.get_content_region_avail().x
    n = 2
    tw = (avail - 8 * (n - 1)) / n
    th = tw * 144 / 256
    for i, e in enumerate(entries):
        if i % n:
            imgui.same_line()
        imgui.begin_group()
        t = image_tex(app, e.thumb)
        if t is not None:
            imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(3 * S, 3 * S))
            clicked = imgui.image_button(f"##{tag}{i}", tex_ref(t), V2(tw - 6, th - 6))
            imgui.pop_style_var()
        else:
            clicked = imgui.button(f"{fa.ICON_FA_IMAGE}##{tag}{i}", V2(tw, th))
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{e.label}" + ("  (Burning Ship)" if e.formula == "ship" else "") +
                              f"\nzoom 10^{e.zoom:.1f}\nin scripts: location \"{e.name}\"\n"
                              "click = jump, right-click = more")
        if clicked:
            try:
                app.apply_location_dict(e.load(), animate=False)
                app.notify(f"opened {e.label}")
            except Exception as ex:
                app.notify(f"cannot open {e.label}: {ex}")
        if imgui.begin_popup_context_item(f"ctx{tag}{i}"):
            if imgui.menu_item("Open", "", False)[0]:
                app.apply_location_dict(e.load())
            if imgui.menu_item("Fly there (animated)", "", False)[0]:
                app.apply_location_dict(e.load(), animate=True)
            if imgui.menu_item("Zoom movie to here", "", False)[0]:
                app.zoom_movie_to(e)
            if imgui.menu_item("Copy script name", "", False)[0]:
                imgui.set_clipboard_text(f'location "{e.name}"')
            if not e.builtin:
                if imgui.menu_item("Overwrite with the current view", "", False)[0]:
                    app.library.save(e.name, app.location_dict(), app.screen_rgb(), overwrite=True)
                if imgui.menu_item("Delete", "", False)[0]:
                    app.library.delete(e.name)
            imgui.end_popup()
        label = e.label if len(e.label) < 24 else e.label[:22] + ".."
        imgui.text(label)
        imgui.text_disabled(f"10^{e.zoom:.0f}")
        imgui.end_group()


def _open_any(app):
    try:
        from imgui_bundle import portable_file_dialogs as pfd
        res = pfd.open_file("Open location", str(user_dir()), ["Mandelscope", "*.json *.png", "All", "*"]).result()
        if not res:
            return
        p = res[0]
        if p.lower().endswith(".png"):
            d = location_from_png(p)
            if d is None:
                app.notify("this PNG has no Mandelscope location inside")
                return
            app.apply_location_dict(d)
        else:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if "keyframes" in d:
                app.sequence = Sequence.from_dict(d)
                app.notify(f"sequence with {len(app.sequence.keyframes)} keyframes loaded")
            else:
                app.apply_location_dict(d)
    except Exception as e:
        app.notify(f"open failed: {e}")


# --------------------------------------------------------------- Animate
def _tab_animate(app):
    seq = app.sequence
    s = app.settings
    section("Timeline", fa.ICON_FA_TIMELINE)
    imgui.text(f"{len(seq.keyframes)} keyframes, {seq.duration:.1f} s")
    if imgui.button(fa.ICON_FA_DIAMOND + " Add (K)"):
        app.add_keyframe()
    imgui.same_line()
    has_sel = 0 <= app.selected_kf < len(seq.keyframes)
    imgui.begin_disabled(not has_sel)
    if imgui.button(fa.ICON_FA_ARROWS_ROTATE + " Update"):
        k = seq.keyframes[app.selected_kf]
        k.view = app.view.copy()
        k.color = app.colour_state()
        seq.invalidate()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Replace the selected keyframe with the current view and colours")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_TRASH):
        seq.keyframes.pop(app.selected_kf)
        seq.invalidate()
        app.selected_kf = -1
    imgui.end_disabled()
    imgui.same_line()
    if imgui.button("Clear all"):
        app.sequence = Sequence(fps=s.video_fps)
        app.selected_kf = -1
    if has_sel:
        k = seq.keyframes[app.selected_kf]
        imgui.push_item_width(110 * S)
        ch, t = imgui.input_float("time (s)##kt", k.time, 0.1, 1.0, "%.2f")
        if ch:
            seq.retime(app.selected_kf, t)
        imgui.same_line()
        ch, ei = imgui.combo("easing", EASINGS.index(k.ease) if k.ease in EASINGS else 0, EASINGS)
        if ch:
            k.ease = EASINGS[ei]
            seq.invalidate()
        imgui.pop_item_width()
        imgui.text_disabled(f"keyframe {app.selected_kf + 1}: zoom 10^{k.view.zoom_magnification_log10():.1f}, "
                            f"{math.degrees(k.view.angle):.0f}°, {k.view.max_iter:,} iterations")
    ch, seq.smooth = imgui.checkbox("smooth camera", seq.smooth)
    if ch:
        seq.invalidate()
    help_marker("Zoom speed changes smoothly through keyframes instead of jumping.")
    imgui.same_line()
    _, s.seq_loop = imgui.checkbox("loop", s.seq_loop)
    if imgui.button(fa.ICON_FA_RULER_HORIZONTAL + "  Auto-time"):
        seq.auto_time(max(0.01, s.auto_time_speed))
    imgui.same_line()
    imgui.push_item_width(-28 * S)
    _, s.auto_time_speed = imgui.slider_float("##ats", s.auto_time_speed, 0.1, 4.0, "%.2f octaves / s")
    imgui.pop_item_width()
    help_marker("Spreads the keyframes so the zoom speed is constant.")
    if imgui.button(fa.ICON_FA_FILM + "  Zoom movie into this view", V2(-1, 0)):
        app.zoom_movie_here(_state.get("spd", 2.0), _state.get("spin", 0.0))
    half = (imgui.get_content_region_avail().x - 8) / 2
    imgui.push_item_width(half)
    _, _state["spd"] = imgui.slider_float("##spd", _state.get("spd", 2.0), 0.3, 6.0, "%.1f s per decade")
    imgui.same_line()
    _, _state["spin"] = imgui.slider_float("##spin", _state.get("spin", 0.0), -5.0, 5.0, "spin %.1f turns")
    imgui.pop_item_width()
    label = (fa.ICON_FA_STOP + "  Stop flight recording (P)") if app.path_rec is not None else \
            (fa.ICON_FA_ROUTE + "  Record a flight (P)")
    if imgui.button(label):
        app.toggle_path_recording()
    help_marker("Fly around with mouse and keys; stopping turns your flight into a timeline\n"
                "that you can render in high quality from the Export tab.")

    section("Script", fa.ICON_FA_CODE)
    mono = FONTS.get("mono")
    if mono is not None:
        imgui.push_font(mono, 0.0)
    _, app.script_text = imgui.input_text_multiline("##script", app.script_text, V2(-1, 260 * S),
                                                    imgui.InputTextFlags_.allow_tab_input)
    if mono is not None:
        imgui.pop_font()
    if imgui.button(fa.ICON_FA_PLAY + "  Apply script"):
        app.apply_script()
    imgui.same_line()
    if imgui.button(fa.ICON_FA_ARROW_DOWN_WIDE_SHORT + " From timeline"):
        app.script_from_timeline()
    imgui.same_line()
    if imgui.button(fa.ICON_FA_BOOK + " Reference"):
        _state["script_help"] = True
    if imgui.button(fa.ICON_FA_ROBOT + "  Copy prompt for an AI assistant"):
        app.copy_ai_prompt()
    help_marker("Copies the script reference, your locations and the current script.\n"
                "Paste it into Claude, describe the movie, paste the answer back here and Apply.")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_FLOPPY_DISK):
        _save_script(app)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Save script...")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_FOLDER_OPEN):
        _load_script(app)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Load script...")
    if app.script_info:
        imgui.text_colored(V4(0.55, 0.9, 0.6, 1.0) if not app.script_errors else V4(1.0, 0.6, 0.4, 1.0),
                           app.script_info)
    for ln, msg in app.script_errors[:12]:
        imgui.text_colored(V4(1.0, 0.55, 0.45, 1.0) if not msg.startswith("warning") else V4(1.0, 0.85, 0.4, 1.0),
                           f"line {ln}: {msg}" if ln else msg)


def _save_script(app):
    try:
        from imgui_bundle import portable_file_dialogs as pfd
        from .app import out_dir
        d = out_dir("Scripts")
        d.mkdir(parents=True, exist_ok=True)
        res = pfd.save_file("Save script", str(d / "movie.mzs"), ["Mandelscope script", "*.mzs *.dzs *.txt"]).result()
        if res:
            Path(res).write_text(app.script_text, encoding="utf-8")
            app.notify(f"saved {res}")
    except Exception as e:
        app.notify(f"save failed: {e}")


def _load_script(app):
    try:
        from imgui_bundle import portable_file_dialogs as pfd
        from .app import out_dir
        res = pfd.open_file("Load script", str(out_dir("Scripts")), ["Mandelscope script", "*.mzs *.dzs *.txt", "All", "*"]).result()
        if res:
            app.script_text = Path(res[0]).read_text(encoding="utf-8")
            app.notify(f"loaded {res[0]}")
    except Exception as e:
        app.notify(f"load failed: {e}")


def _script_help(app, W, H):
    imgui.set_next_window_size(V2(min(760, W - 40), min(640, H - 80)), imgui.Cond_.appearing)
    imgui.set_next_window_pos(V2(W / 2, H / 2), imgui.Cond_.appearing, V2(0.5, 0.5))
    exp, opened = imgui.begin("Script reference", True, imgui.WindowFlags_.no_saved_settings)
    if exp:
        if imgui.button(fa.ICON_FA_COPY + " Copy"):
            imgui.set_clipboard_text(SCRIPT_HELP)
        imgui.same_line()
        if imgui.button("Insert as script"):
            app.script_text = SCRIPT_HELP
        mono = FONTS.get("mono")
        if mono is not None:
            imgui.push_font(mono, 0.0)
        imgui.input_text_multiline("##ref", SCRIPT_HELP, V2(-1, -1), imgui.InputTextFlags_.read_only)
        if mono is not None:
            imgui.pop_font()
    imgui.end()
    if opened is False:
        _state["script_help"] = False


# ---------------------------------------------------------------- Export
def _tab_export(app):
    s = app.settings
    section("Video from the timeline", fa.ICON_FA_VIDEO)
    names = [p[0] for p in RES_PRESETS] + ["Custom"]
    cur = next((i for i, p in enumerate(RES_PRESETS) if (p[1], p[2]) == (s.video_w, s.video_h)), len(RES_PRESETS))
    imgui.push_item_width(-110 * S)
    ch, i = imgui.combo("size", cur, names)
    if ch and i < len(RES_PRESETS):
        s.video_w, s.video_h = RES_PRESETS[i][1], RES_PRESETS[i][2]
    if cur == len(RES_PRESETS) or (ch and i == len(RES_PRESETS)):
        imgui.push_item_width(90 * S)
        _, s.video_w = imgui.input_int("##vw", s.video_w, 0)
        imgui.same_line()
        _, s.video_h = imgui.input_int("w x h##vh", s.video_h, 0)
        imgui.pop_item_width()
    fps_list = list(FPS_PRESETS) + ([float(s.video_fps)] if float(s.video_fps) not in FPS_PRESETS else [])
    fi = fps_list.index(float(s.video_fps))
    ch, fi = imgui.combo("frames/s", fi, [f"{f:g}" for f in fps_list])
    if ch:
        s.video_fps = fps_list[fi]
    _, s.video_ss = imgui.slider_int("supersampling", s.video_ss, 1, 4, f"{s.video_ss}x{s.video_ss}")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Each pixel averages ss x ss samples: smooth, noise-free filaments.\n"
                          "2x2 is a good default; 3-4 for the final cut.")
    _, s.fast_movie = imgui.checkbox("fast: frames share pictures (recommended)", s.fast_movie)
    if imgui.is_item_hovered():
        imgui.set_tooltip("One large picture is computed for about an octave of zoom (or a stretch of\n"
                          "a pan or turn) and every frame of that stretch is cut out of it, with at\n"
                          "least the supersampling chosen above per pixel.  Colours, palette cycling\n"
                          "and colour keyframes stay per frame.  Typically 5-15x fewer samples;\n"
                          "very fast moves fall back to computing frames one by one.\n"
                          "Needs ~3 GB of graphics memory at 1080p with 2x2.")
    if s.video_ss > 1 and not s.fast_movie:
        _, s.adaptive_export = imgui.checkbox("adaptive (experimental)", s.adaptive_export)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Sub-samples only for pixels at edges, texture or near the set.\n"
                              "Same picture; whether it is faster depends on the GPU -\n"
                              "'run.bat bench --experiments' compares both.")
    if _state["codecs"] is None:
        _state["codecs"] = available_codecs()
    codes = [c for c, _ in _state["codecs"]]
    labels = [l for _, l in _state["codecs"]]
    ci = codes.index(s.video_codec) if s.video_codec in codes else 0
    ch, ci = imgui.combo("codec", ci, labels)
    s.video_codec = codes[ci]
    _, s.video_quality = imgui.slider_int("quality", s.video_quality, 10, 35, "%d (lower = better)")
    s.video_precision = _precision_combo("precision##v", s.video_precision)
    s.video_iter_mult = _iter_mult_combo("iterations##v", s.video_iter_mult, None)
    _, s.video_overlay = imgui.checkbox("show zoom depth", s.video_overlay)
    imgui.same_line()
    _, s.video_png = imgui.checkbox("also save PNG frames", s.video_png)
    _, s.video_name = imgui.input_text("file name", s.video_name)
    imgui.pop_item_width()
    seq = app.sequence
    nfr = seq.frame_count(s.video_fps) if len(seq.keyframes) >= 2 else 0
    imgui.text_disabled(f"{nfr:,} frames, {seq.duration:.1f} s  ->  {user_root() / 'Videos'}")
    bw = (imgui.get_content_region_avail().x - 8) / 2
    if imgui.button(fa.ICON_FA_BOLT + "  Quick draft (360p)", V2(bw, 0)):
        app.queue_video(draft=True)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Fast low-resolution preview; opens in your video player when done")
    imgui.same_line()
    if imgui.button(fa.ICON_FA_FILM + "  Render video", V2(bw, 0)):
        app.queue_video()

    section("Queue", fa.ICON_FA_LIST_CHECK)
    if not app.jobs:
        imgui.text_disabled("Nothing rendering. Exploring stays possible while a job runs.")
    for j in list(app.jobs)[-8:]:
        imgui.push_id(str(id(j)))
        icon = {"queued": fa.ICON_FA_HOURGLASS, "running": fa.ICON_FA_SPINNER, "paused": fa.ICON_FA_PAUSE,
                "done": fa.ICON_FA_CHECK, "failed": fa.ICON_FA_TRIANGLE_EXCLAMATION,
                "cancelled": fa.ICON_FA_XMARK}.get(j.status, "")
        imgui.text(f"{icon}  {j.name}")
        if j.status in ("running", "paused", "queued"):
            txt = (f"{j.frame + 1}/{j.frames}  ETA {_fmt_t(j.eta)}" if j.kind != "still" else f"{100 * j.progress:.0f}%")
            imgui.progress_bar(j.progress, V2(-90 * S, 0), txt)
            imgui.same_line()
            if imgui.small_button(fa.ICON_FA_PLAY if j.status == "paused" else fa.ICON_FA_PAUSE):
                app.pause_job(j)
            imgui.same_line()
            if imgui.small_button(fa.ICON_FA_XMARK):
                app.cancel_job(j)
        elif j.status == "done":
            imgui.text_disabled(f"done in {_fmt_t(j.elapsed)}  {j.codec or ''}")
            imgui.same_line()
            if imgui.small_button(fa.ICON_FA_PLAY + " open"):
                from .app import open_path
                open_path(j.path)
            imgui.same_line()
            if imgui.small_button(fa.ICON_FA_FOLDER_OPEN):
                from .app import open_path
                open_path(Path(j.path).parent)
        elif j.status == "failed":
            imgui.text_colored(V4(1, 0.5, 0.4, 1), j.error[-120:])
        imgui.pop_id()
    if app.preview_tex is not None and any(j.status in ("running", "paused") for j in app.jobs):
        w, h = app.preview_tex.size
        pw = imgui.get_content_region_avail().x
        imgui.image(tex_ref(app.preview_tex), V2(pw, pw * h / max(w, 1)))

    section("Still image", fa.ICON_FA_IMAGE)
    imgui.push_item_width(90 * S)
    _, s.still_w = imgui.input_int("##sw", s.still_w, 0)
    imgui.same_line()
    _, s.still_h = imgui.input_int("w x h##sh", s.still_h, 0)
    imgui.same_line()
    _, s.still_ss = imgui.input_int("ss", s.still_ss, 0)
    imgui.pop_item_width()
    s.still_ss = max(1, min(8, s.still_ss))
    imgui.push_item_width(-150 * S)
    s.still_iter_mult = _iter_mult_combo("iterations##s", s.still_iter_mult, app.view.max_iter)
    imgui.pop_item_width()
    if imgui.button(fa.ICON_FA_IMAGE + "  Render still"):
        app.start_still()
    imgui.same_line()
    imgui.text_disabled("F12 = screenshot")

    section("Live recording", fa.ICON_FA_CIRCLE)
    imgui.push_item_width(-110 * S)
    _, s.live_fps = imgui.input_float("frames/s##live", s.live_fps, 5.0, 10.0, "%.0f")
    ci = codes.index(s.live_codec) if s.live_codec in codes else 0
    ch, ci = imgui.combo("codec##live", ci, labels)
    s.live_codec = codes[ci]
    _, s.live_quality = imgui.slider_int("quality##live", s.live_quality, 10, 35)
    _, s.live_ui = imgui.checkbox("include the panels", s.live_ui)
    imgui.pop_item_width()
    if imgui.button((fa.ICON_FA_STOP + "  Stop recording (R)") if app.live is not None
                    else (fa.ICON_FA_CIRCLE + "  Start recording (R)")):
        app.toggle_live()
    imgui.same_line()
    if imgui.button(fa.ICON_FA_FOLDER_OPEN + " Videos"):
        from .app import open_path, out_dir
        out_dir("Videos").mkdir(parents=True, exist_ok=True)
        open_path(out_dir("Videos"))


def _fmt_t(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"


# ---------------------------------------------------------------- Render
def _tab_render(app):
    s = app.settings
    section("Interface", fa.ICON_FA_WINDOW_MAXIMIZE)
    styles = [("hud", "Cinematic HUD"), ("classic", "Classic")]
    si = next((i for i, (k, _) in enumerate(styles) if k == s.ui_style), 0)
    imgui.push_item_width(-150 * S)
    ch, si = imgui.combo("style", si, [n for _, n in styles])
    imgui.pop_item_width()
    if ch:
        s.ui_style = styles[si][0]
    help_marker("Cinematic HUD: the picture fills the window, thin outlined panels and corner readouts.\n"
                "Classic: toolbar, side panel and status bar.")
    if imgui.small_button("show the welcome card again"):
        s.welcome_seen = False
    if s.ui_style == "hud":
        from . import hud
        imgui.text_disabled("accent")
        for name, rgb in hud.ACCENTS.items():
            imgui.same_line()
            if imgui.color_button(f"{name}##acc", V4(*rgb, 1.0), imgui.ColorEditFlags_.no_tooltip,
                                  V2(22 * S, 22 * S)):
                s.hud_accent = name
            if name == s.hud_accent:
                mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                imgui.get_window_draw_list().add_rect(V2(mn.x - 3, mn.y - 3), V2(mx.x + 3, mx.y + 3),
                                                      col(1, 1, 1, 0.9), 3.0, 1.5)
            if imgui.is_item_hovered():
                imgui.set_tooltip(name)
        _, s.hud_brackets = imgui.checkbox("corner brackets", s.hud_brackets)
        imgui.same_line()
        _, s.hud_reticle = imgui.checkbox("centre mark while moving", s.hud_reticle)
    section("Renderer", fa.ICON_FA_DESKTOP)
    from . import software
    if app.on_cpu:
        imgui.text_wrapped(f"Processor (CPU mode): Mesa llvmpipe, {software.threads()} threads.")
        if app.cpu_mode and imgui.button("restart on the graphics card"):
            app.restart_renderer(False)
        help_marker("CPU mode runs the same engine with Mesa's software OpenGL: every feature works,\n"
                    "only slower: a graphics card with OpenGL 4.3 is tens to hundreds of times faster.")
    else:
        imgui.text_wrapped(f"Graphics card: {app.engine.renderer_name if app.engine else '?'}")
        if imgui.button("restart in CPU mode"):
            app.restart_renderer(True)
        help_marker("Runs the engine on the processor with Mesa's software OpenGL (llvmpipe), for\n"
                    "computers without an OpenGL 4.3 graphics driver. Much slower; useful to try it.")
    section("Quality", fa.ICON_FA_GEM)
    imgui.push_item_width(-150 * S)
    s.precision = _precision_combo("precision", s.precision)
    help_marker("float32 runs at full speed.  The precise modes remove the last bit of noise in\n"
                "very chaotic places.  float-float (two float32 per number, about 48 bits) is\n"
                "4-7x faster than float64 on graphics cards for games (RTX 5060 Ti: seahorse\n"
                "1e-12 in 122 instead of 914 ms); 'bench --f64' compares them on yours.\n"
                "(The Burning Ship uses float64 for both.)")
    aa_opts = [1, 4, 8, 16, 32]
    ai = aa_opts.index(s.aa_passes) if s.aa_passes in aa_opts else 2
    ch, ai = imgui.combo("smoothing when idle", ai, ["off", "4 passes", "8 passes", "16 passes", "32 passes"])
    s.aa_passes = aa_opts[ai]
    help_marker("When you stop moving, extra slightly shifted passes are averaged:\n"
                "smooth anti-aliased edges without slowing down navigation.")
    ch, s.adaptive_ss = imgui.checkbox("only where needed (adaptive)", s.adaptive_ss)
    if ch:
        app.aa_count = 0
    help_marker("Smoothing passes compute extra samples only for pixels at edges, texture or near\n"
                "the set; the others keep their (identical) centre sample.")
    ss_opts = [1.0, 2.0, 3.0, 4.0]
    si = ss_opts.index(s.render_scale) if s.render_scale in ss_opts else 0
    ch, si = imgui.combo("supersampling", si, ["1x (recommended)", "2x2", "3x3", "4x4"])
    s.render_scale = ss_opts[si]
    mo = [0.0, 1.0, 0.5]
    mi = mo.index(s.motion_scale) if s.motion_scale in mo else 0
    ch, mi = imgui.combo("while moving", mi, ["automatic", "full resolution", "half resolution (faster)"])
    s.motion_scale = mo[mi]
    help_marker("Automatic: zooms and turns use half resolution while a full picture would take\n"
                "longer than ~12 frames (heavy places), full resolution otherwise (CPU mode: down\n"
                "to a quarter).  Pans always use full resolution (they reuse almost every sample).")
    _, s.frame_budget_ms = imgui.slider_float("CPU time / frame" if app.on_cpu else "GPU time / frame",
                                              s.frame_budget_ms, 4.0, 40.0, "%.0f ms")
    sizes = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0]
    zi = sizes.index(s.ui_scale) if s.ui_scale in sizes else 0
    ch, zi = imgui.combo("interface size", zi, ["automatic", "100 %", "125 %", "150 %", "175 %", "200 %"])
    if ch:
        s.ui_scale = sizes[zi]
        app.notify("interface size changes at the next start")
    help_marker("Automatic follows the Windows display scaling. Takes effect after a restart.")
    imgui.pop_item_width()
    section("Algorithm", fa.ICON_FA_MICROCHIP)
    ch, s.use_bla = imgui.checkbox("skip iterations (BLA)", s.use_bla)
    if ch:
        app.need_restart = True
    imgui.push_item_width(-150 * S)
    if s.precision == 0:
        _, s.eps_f32 = imgui.slider_float("BLA tolerance", s.eps_f32, -32.0, -16.0, "2^%.0f")
    else:
        _, s.eps_f64 = imgui.slider_float("BLA tolerance", s.eps_f64, -60.0, -30.0, "2^%.0f")
    imgui.pop_item_width()
    ch, s.interior = imgui.checkbox("detect the inside early", s.interior)
    if ch:
        app.need_restart = True
    _, s.show_fallback = imgui.checkbox("progressive preview", s.show_fallback)
    section("Statistics", fa.ICON_FA_CHART_SIMPLE)
    f = app.field
    if f is not None and f.params is not None:
        imgui.text_disabled(f"field {f.W}x{f.H}   {100 * f.progress:.0f}%   dispatches {f.dispatches}  syncs {f.syncs}")
        imgui.text_disabled(f"GPU {f.gpu_time * 1000:.1f} ms   steps {f.steps:,}")
        imgui.text_disabled(f"BLA skips {f.bla_skips:,}   rebases {f.rebases:,}")
        imgui.text_disabled(f"anti-aliasing passes {app.aa_count}/{s.aa_passes}")
        imgui.text_disabled(f"time to finish a picture {app.field_T * 1000:.0f} ms, "
                            f"frames kept in flight {app.flight_frames:,}")
    r = app.refs.ref if hasattr(app, "refs") else None
    if r is not None:
        imgui.text_disabled(f"reference: {r.length:,} iterations, {r.prec} bits, "
                            f"{'escaped' if r.escaped else ('period %d' % r.period if r.period else 'bounded')}"
                            f", {r.elapsed:.2f}s")
    b = app.refs.bla if hasattr(app, "refs") else None
    if b is not None:
        imgui.text_disabled(f"BLA table: {b.levels} levels, {b.size:,} entries")
    imgui.text_disabled(app.engine.renderer_name if app.engine else "")
    imgui.text_disabled(f"frame {app.frame_ms:.1f} ms")


# ---------------------------------------------------------------- timeline
def _timeline(app, W, H):
    y = H - STATUS_H - TIMELINE_H
    imgui.set_next_window_pos(V2(6, y))
    imgui.set_next_window_size(V2(W - 12, TIMELINE_H - 6))
    imgui.set_next_window_bg_alpha(0.90)
    imgui.begin("##timeline", None, FIXED | imgui.WindowFlags_.no_scrollbar)
    seq = app.sequence
    # controls
    imgui.begin_group()
    if icon_button(fa.ICON_FA_PAUSE if app.playing else fa.ICON_FA_PLAY, "Play / pause (Space)", app.playing):
        app.toggle_play()
    imgui.same_line()
    if icon_button(fa.ICON_FA_BACKWARD_STEP, "Go to start"):
        app.playing = False
        app.play_t = seq.start
        app.seek(seq.start)
    imgui.same_line()
    if icon_button(fa.ICON_FA_REPEAT, "Loop", app.settings.seq_loop):
        app.settings.seq_loop = not app.settings.seq_loop
    if icon_button(fa.ICON_FA_DIAMOND, "Add keyframe (K)"):
        app.add_keyframe()
    imgui.same_line()
    if icon_button(fa.ICON_FA_FILM, "Render this timeline (Export tab)"):
        app.settings.sidebar_tab = "Export"
        app.settings.sidebar_open = True
    imgui.text_disabled(f"{app.play_t:6.2f} / {seq.duration:.2f} s")
    imgui.end_group()
    imgui.same_line(0, 12 * S)
    p0 = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    w, h = avail.x - 4, avail.y - 2
    imgui.invisible_button("##track", V2(max(w, 10), max(h, 10)))
    hovered = imgui.is_item_hovered()
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(p0, V2(p0.x + w, p0.y + h), col(0, 0, 0, 0.25), 6.0)
    if len(seq.keyframes) < 1:
        dl.add_text(V2(p0.x + 14, p0.y + h / 2 - 8), col(1, 1, 1, 0.5),
                    "No keyframes yet - press K to add the current view, record a flight (P), "
                    "or write a script in the Animate tab")
        imgui.end()
        return
    t0, t1 = seq.start, max(seq.end, seq.start + 1.0)
    pad = 14 * S
    tx = lambda t: p0.x + pad + (t - t0) / (t1 - t0) * (w - 2 * pad)
    xt = lambda x: t0 + (x - p0.x - pad) / max(w - 2 * pad, 1) * (t1 - t0)
    # ruler
    span = t1 - t0
    step = next((s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if span / s <= 12), 1200)
    t = math.ceil(t0 / step) * step
    while t <= t1 + 1e-9:
        x = tx(t)
        dl.add_line(V2(x, p0.y + 2), V2(x, p0.y + 10 * S), col(1, 1, 1, 0.35), 1.0)
        dl.add_text(V2(x + 3, p0.y + 1), col(1, 1, 1, 0.45), f"{t:g}s")
        t += step
    # zoom depth curve
    lane_top, lane_bot = p0.y + 24 * S, p0.y + h - 26 * S
    if len(seq.keyframes) >= 2:
        zs = [k.view.zoom_magnification_log10() for k in seq.keyframes]
        zmin, zmax = min(zs), max(zs)
        if zmax - zmin < 1e-6:
            zmax = zmin + 1.0
        pts = []
        n = int(min(240, max(24, w / 4)))
        cache_key = (id(seq), len(seq.keyframes), tuple(round(k.time, 4) for k in seq.keyframes),
                     tuple(round(k.view.log2r, 4) for k in seq.keyframes),
                     tuple(k.ease for k in seq.keyframes), seq.smooth, n)
        if _state.get("curve_key") != cache_key:
            vals = []
            for i in range(n + 1):
                tt = t0 + (t1 - t0) * i / n
                try:
                    vals.append((tt, seq.view_at(tt)[0].zoom_magnification_log10()))
                except Exception:
                    pass
            _state["curve_key"] = cache_key
            _state["curve"] = vals
        for tt, z in _state.get("curve", []):
            pts.append(V2(tx(tt), lane_bot - (z - zmin) / (zmax - zmin) * (lane_bot - lane_top)))
        for i in range(len(pts) - 1):
            dl.add_line(pts[i], pts[i + 1], col(*ACCENT, 0.55), 2.0)
        dl.add_text(V2(p0.x + w - 120 * S, lane_top - 2), col(*ACCENT, 0.7), f"zoom 10^{zmax:.0f}")
    # keyframes
    ky = p0.y + h - 13 * S
    mp = imgui.get_io().mouse_pos
    hover_k = -1
    for i, k in enumerate(seq.keyframes):
        x = tx(k.time)
        r = 7.0 * S
        is_sel = i == app.selected_kf
        near = abs(mp.x - x) < 9 * S and abs(mp.y - ky) < 12 * S
        if near:
            hover_k = i
        pts = [V2(x, ky - r), V2(x + r, ky), V2(x, ky + r), V2(x - r, ky)]
        fillc = col(*ACCENT, 1.0) if is_sel else (col(1, 1, 1, 0.9) if near else col(0.8, 0.82, 0.88, 0.75))
        dl.add_quad_filled(pts[0], pts[1], pts[2], pts[3], fillc)
        dl.add_quad(pts[0], pts[1], pts[2], pts[3], col(0, 0, 0, 0.6), 1.0)
        if k.ease != "linear":
            dl.add_text(V2(x + 9, ky - 8), col(1, 1, 1, 0.35), k.ease[:6])
    # playhead
    px = tx(app.play_t)
    ph = tuple(min(1.0, 0.35 + 0.7 * c) for c in ACCENT)          # playhead: a light accent
    dl.add_line(V2(px, p0.y), V2(px, p0.y + h), col(*ph, 0.95), 2.0)
    dl.add_triangle_filled(V2(px - 6 * S, p0.y), V2(px + 6 * S, p0.y), V2(px, p0.y + 8 * S), col(*ph, 1.0))
    # interaction
    if imgui.is_item_clicked(0):
        if hover_k >= 0:
            app.selected_kf = hover_k
            _state["tl_drag"] = ("key", hover_k, mp.x)
        else:
            _state["tl_drag"] = ("seek",)
            app.playing = False
    if imgui.is_mouse_double_clicked(0) and hover_k >= 0 and hovered:
        app.play_t = seq.keyframes[hover_k].time
        app.seek(app.play_t)
    drag = _state.get("tl_drag")
    if drag is not None:
        if imgui.is_mouse_down(0):
            if drag[0] == "seek":
                app.play_t = min(max(xt(mp.x), seq.start), seq.end)
                app.seek(app.play_t)
            elif drag[0] == "key" and abs(mp.x - drag[2]) > 3 and drag[1] > 0:
                seq.retime(drag[1], xt(mp.x))
        else:
            _state["tl_drag"] = None
    if hovered and hover_k >= 0:
        k = seq.keyframes[hover_k]
        imgui.set_tooltip(f"keyframe {hover_k + 1}  t={k.time:.2f}s  zoom 10^{k.view.zoom_magnification_log10():.1f}\n"
                          "drag: move in time, double-click: go there, right-click: options")
    if hovered and imgui.is_mouse_clicked(1) and hover_k >= 0:
        app.selected_kf = hover_k
        imgui.open_popup("kfmenu")
    if imgui.begin_popup("kfmenu"):
        i = app.selected_kf
        if 0 <= i < len(seq.keyframes):
            k = seq.keyframes[i]
            if imgui.menu_item("Go to this keyframe", "", False)[0]:
                app.play_t = k.time
                app.seek(k.time)
            if imgui.menu_item("Update from the current view", "", False)[0]:
                k.view = app.view.copy()
                k.color = app.colour_state()
                seq.invalidate()
            imgui.separator()
            for e in EASINGS:
                if imgui.menu_item(f"easing: {e}", "", k.ease == e)[0]:
                    k.ease = e
                    seq.invalidate()
            imgui.separator()
            if imgui.menu_item("Delete", "", False)[0]:
                seq.keyframes.pop(i)
                seq.invalidate()
                app.selected_kf = -1
        imgui.end_popup()
    imgui.end()


# ---------------------------------------------------------------- status bar
def _status_bar(app, W, H):
    imgui.set_next_window_pos(V2(0, H - STATUS_H))
    imgui.set_next_window_size(V2(W, STATUS_H))
    imgui.set_next_window_bg_alpha(0.88)
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(10 * S, 5 * S))
    imgui.begin("##status", None, FIXED | imgui.WindowFlags_.no_scrollbar)
    v = app.view
    f = app.field
    imgui.text(f"{fa.ICON_FA_MAGNIFYING_GLASS} 10^{v.zoom_magnification_log10():.2f}")
    imgui.same_line(0, 16 * S)
    imgui.text(f"{fa.ICON_FA_LAYER_GROUP} {v.max_iter:,}")
    imgui.same_line(0, 16 * S)
    imgui.text(f"{math.degrees(v.angle) % 360:.0f}°")
    imgui.same_line(0, 16 * S)
    if f is not None and f.params is not None:
        imgui.push_item_width(110 * S)
        imgui.progress_bar(f.progress, V2(110 * S, 16 * S), "" if f.complete else f"{100 * f.progress:.0f}%")
        imgui.pop_item_width()
        imgui.same_line()
        if f.complete and f.finished_at:
            imgui.text_disabled(f"{(f.finished_at - f.started) * 1000:.0f} ms")
            imgui.same_line()
        if app.aa_count > 1:
            imgui.text_disabled(f"AA {app.aa_count}/{app.settings.aa_passes}")
            imgui.same_line()
    imgui.text_disabled(f"{app.fps:4.0f} fps")
    if app.julia_c is not None:
        c = app.julia_c
        imgui.same_line(0, 16 * S)
        imgui.text_colored(V4(0.6, 0.85, 1.0, 1), f"{fa.ICON_FA_SNOWFLAKE} Julia set  c = {c[0]:.8g} "
                                                f"{'+' if c[1] >= 0 else '-'} {abs(c[1]):.8g}i   (J: back)")
    if app.pending_ref is not None:
        imgui.same_line(0, 16 * S)
        imgui.text_colored(V4(1, 0.8, 0.35, 1), f"{fa.ICON_FA_SPINNER} reference {100 * app.ref_progress:.0f}%")
    if app.pending_bla is not None:
        imgui.same_line(0, 12 * S)
        imgui.text_colored(V4(1, 0.8, 0.35, 1), "building skip table")
    # right side indicators
    items = []
    if app.live is not None:
        items.append(((1, 0.35, 0.35, 1), f"{fa.ICON_FA_CIRCLE} REC {app.live.writer.frames}"))
    if app.path_rec is not None:
        items.append(((0.45, 1, 0.55, 1), f"{fa.ICON_FA_ROUTE} flight {len(app.path_rec.samples)}"))
    run = next((j for j in app.jobs if j.status == "running"), None)
    if run is not None:
        items.append(((0.55, 0.8, 1, 1), f"{fa.ICON_FA_FILM} {run.name} {100 * run.progress:.0f}% ETA {_fmt_t(run.eta)}"))
    if app.playing:
        items.append(((0.55, 0.8, 1, 1), f"{fa.ICON_FA_PLAY} {app.play_t:.1f}s"))
    if items:
        txt_w = sum(imgui.calc_text_size(t).x + 20 for _, t in items)
        imgui.same_line(max(imgui.get_cursor_pos_x() + 20, W - txt_w - 10))
        for i, (c, t) in enumerate(items):
            if i:
                imgui.same_line(0, 20 * S)
            imgui.text_colored(V4(*c), t)
    imgui.end()
    imgui.pop_style_var(2)


def _julia_preview(app):
    """Small window with the Julia set of the point under the cursor (click: explore it)."""
    jp = app.jpreview
    if (jp is None or jp.tex is None or jp.c is None or app.julia_c is not None or not app.settings.julia_preview
            or app.formula != "mandelbrot"):
        return
    fx = app.fb_scale[0] or 1.0
    w, h = jp.tex.size
    imgui.set_next_window_pos(V2(8 * S, TOOLBAR_H + 8 * S))
    imgui.set_next_window_bg_alpha(0.85)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(6 * S, 5 * S))
    imgui.begin("##juliaprev", None, FIXED | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_scrollbar)
    c = jp.c
    imgui.text_disabled(f"{fa.ICON_FA_SNOWFLAKE} Julia set   c = {c[0]:.6f} {'+' if c[1] >= 0 else '-'} {abs(c[1]):.6f}i")
    imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(0, 0))
    if imgui.image_button("##jimg", tex_ref(jp.tex), V2(w / fx, h / fx), *FBO_UV):
        app.enter_julia(*c)
    imgui.pop_style_var()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Click (or Shift+J over the picture): explore this Julia set\nJ: hide the preview")
    imgui.end()
    imgui.pop_style_var()


def _toasts(app, H):
    now = time.perf_counter()
    msgs = [(t, m) for t, m in app.messages if now - t < 6.0][-4:]
    if not msgs:
        return
    tl = TIMELINE_H if (app.ui_visible and app.settings.show_timeline) else 0
    y = H - (STATUS_H if app.ui_visible else 0) - tl - 10
    imgui.set_next_window_pos(V2(12 * S, y), 0, V2(0, 1))
    imgui.set_next_window_bg_alpha(0.75)
    imgui.begin("##toasts", None, imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize |
                imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_focus_on_appearing |
                imgui.WindowFlags_.no_nav | imgui.WindowFlags_.no_inputs)
    for t, m in msgs:
        a = max(0.0, min(1.0, (6.0 - (now - t)) / 1.0))
        imgui.text_colored(V4(0.92, 0.92, 0.95, a), m)
    imgui.end()


# ---------------------------------------------------------------- overlays
SHORTCUTS = [
    ("Mouse", ""),
    ("wheel", "zoom at the cursor (Ctrl: fine, Shift: rotate)"),
    ("left drag", "pan (glides when released)"),
    ("right drag", "rotate (Shift snaps to 15°)"),
    ("double-click", "zoom x4 into that point"),
    ("middle click", "centre that point"),
    ("right click", "menu: find minibrot, keyframe, save ..."),
    ("Keyboard", ""),
    ("+ / -  (PgUp / PgDn)", "zoom at the centre: tap = 2x, hold = continuous"),
    ("Z / X", "zoom at the cursor: tap = 2x, hold = continuous"),
    ("Shift / Ctrl", "faster / finer"),
    ("arrows or W A S D", "pan"),
    ("Q / E", "rotate"),
    ("[  /  ]", "rotate 90° left / right"),
    ("0", "rotation back to 0°"),
    ("H", "home (whole set)"),
    ("Backspace", "back (Shift: forward)"),
    ("M", "find the minibrot (or mini ship) under the cursor"),
    ("B", "Burning Ship / Mandelbrot set"),
    ("J", "Julia set preview on / off (exploring one: back to the Mandelbrot set)"),
    ("Shift+J", "explore the Julia set of the point under the cursor"),
    ("K", "add keyframe"),
    ("Space", "play / pause the timeline"),
    ("T", "show / hide the timeline"),
    ("R", "start / stop live recording"),
    ("P", "start / stop recording a flight"),
    ("F12", "screenshot (2x supersampled PNG)"),
    ("Ctrl+C / Ctrl+V", "copy / paste the location"),
    ("Tab", "hide / show all panels"),
    ("F1", "this help"),
]


def _pad_rows():
    from .control import PAD_ACTIONS
    return [("Controller", "")] + list(PAD_ACTIONS) + [
        ("Xbox names", "A B X Y = cross circle square triangle"),
        ("", "LB RB LT RT = L1 R1 L2 R2, Menu = options, View = create")]


def _key_cells(k, d, c0):
    imgui.table_set_column_index(c0)
    if not d:
        if k.startswith(("Controller", "Mouse", "Keyboard")):
            imgui.text_colored(V4(*ACCENT, 1.0), k)
        elif k:
            imgui.text_disabled(k)
        return
    imgui.text(k)
    imgui.table_set_column_index(c0 + 1)
    imgui.text_disabled(d)


def _help_overlay(app, W, H):
    """Every key, mouse action and controller button: side by side on wide windows, one list
    on narrow ones; scrolls when it does not fit."""
    imgui.set_next_window_pos(V2(W / 2, H / 2), imgui.Cond_.always, V2(0.5, 0.5))
    imgui.set_next_window_size_constraints(V2(0, 0), V2(max(200.0, W - 40), max(200.0, H - 40)))
    imgui.set_next_window_bg_alpha(0.95)
    exp, opened = imgui.begin("Keyboard, mouse & controller", True,
                              imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings |
                              imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_move)
    if exp:
        pad = _pad_rows()
        wide = W >= 1180 * S
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit
        if wide and imgui.begin_table("keys", 5, flags):
            for c in range(5):
                if c == 2:
                    imgui.table_setup_column("gap", imgui.TableColumnFlags_.width_fixed, 24 * S)
                else:
                    imgui.table_setup_column(f"c{c}")
            for i in range(max(len(SHORTCUTS), len(pad))):
                imgui.table_next_row()
                if i < len(SHORTCUTS):
                    _key_cells(*SHORTCUTS[i], 0)
                if i < len(pad):
                    _key_cells(*pad[i], 3)
            imgui.end_table()
        elif not wide and imgui.begin_table("keys", 2, flags):
            for k, d in SHORTCUTS + [("", "")] + pad:
                imgui.table_next_row()
                _key_cells(k, d, 0)
            imgui.end_table()
        imgui.text_disabled("Esc, F1 or the PS button closes this window")
    imgui.end()
    if opened is False:
        app.show_help = False


def _context_menu(app):
    if app.open_context_menu:
        app.open_context_menu = False
        imgui.open_popup("fractal_ctx")
    if imgui.begin_popup("fractal_ctx"):
        c = app.context_menu_c
        if c is not None:
            cre, cim, mx, my = c
            if imgui.menu_item(fa.ICON_FA_MAGNIFYING_GLASS_PLUS + "  Zoom x4 into this point", "dbl-click", False)[0]:
                app.zoom_to_point(mx, my, 2.0)
            if imgui.menu_item(fa.ICON_FA_CROSSHAIRS + "  Centre here", "middle click", False)[0]:
                app.centre_on_point(mx, my)
            if app.julia_c is None:
                if imgui.menu_item(fa.ICON_FA_MAGNIFYING_GLASS + "  Find a minibrot here", "M", False)[0]:
                    app.find_minibrot_at(cre, cim)
                if imgui.menu_item(fa.ICON_FA_SNOWFLAKE + "  Explore the Julia set of this point", "Shift+J", False)[0]:
                    app.enter_julia(cre, cim)
                if imgui.menu_item(fa.ICON_FA_THUMBTACK + "  Put the reference orbit here", "", False)[0]:
                    app.ref_anchor = View(cre, cim, app.view.log2r)
                    app.refs.ref = None
                    app.need_restart = True
            else:
                if imgui.menu_item(fa.ICON_FA_ARROW_TURN_UP + "  Back to the Mandelbrot set", "J", False)[0]:
                    app.leave_julia()
            imgui.separator()
            if imgui.menu_item(fa.ICON_FA_DIAMOND + "  Add keyframe", "K", False)[0]:
                app.add_keyframe()
            if imgui.menu_item(fa.ICON_FA_BOOKMARK + "  Save location to the library", "", False)[0]:
                app.save_to_library("")
            if imgui.menu_item(fa.ICON_FA_COPY + "  Copy location", "Ctrl+C", False)[0]:
                imgui.set_clipboard_text(json.dumps(app.location_dict(), indent=1))
            if imgui.menu_item(fa.ICON_FA_ROTATE_LEFT + "  Rotation to 0°", "0", False)[0]:
                app.rotate_to(0.0)
        imgui.end_popup()
