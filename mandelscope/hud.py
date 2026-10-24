"""'Cinematic HUD' interface style: the picture fills the window, thin outlined panels float over
it, readouts sit in the corners (Inter for labels, JetBrains Mono for numbers).

The classic style (ui.py) stays available (Render tab > Interface).  Both draw the same tab
contents; this module adds the HUD theme, the corner readouts, the top-right icon row, the
outlined side panel with its text tab strip and the toasts."""
from __future__ import annotations

import math
import time

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from . import ui
from .ui import V2, V4, FIXED

ACCENTS = {"ice": (0.561, 0.827, 1.0), "gold": (0.949, 0.757, 0.306), "coral": (1.0, 0.541, 0.502),
           "lime": (0.714, 0.941, 0.612), "lilac": (0.788, 0.722, 1.0)}
PANEL_BG = (0.018, 0.026, 0.042, 0.82)   # no backdrop blur in ImGui: denser than the mock-up
LINE = 0.20                      # panel / control outline alpha
_meas: dict = {}


def accent(app) -> tuple:
    return ACCENTS.get(app.settings.hud_accent, ACCENTS["ice"])


def c32(r, g, b, a=1.0) -> int:
    return imgui.color_convert_float4_to_u32(V4(r, g, b, a))


def white(a: float) -> int:
    return c32(1.0, 1.0, 1.0, a)


# ----------------------------------------------------------------- theme
def setup_style(acc):
    imgui.style_colors_dark()
    st = imgui.get_style()
    st.window_rounding = 3.0
    st.child_rounding = 2.0
    st.frame_rounding = 2.0
    st.popup_rounding = 3.0
    st.grab_rounding = 2.0
    st.tab_rounding = 2.0
    st.scrollbar_rounding = 2.0
    st.window_border_size = 1.0
    st.child_border_size = 0.0
    st.frame_border_size = 1.0
    st.popup_border_size = 1.0
    st.window_padding = V2(14, 12)
    st.frame_padding = V2(8, 5)
    st.item_spacing = V2(8, 7)
    st.item_inner_spacing = V2(6, 5)
    st.scrollbar_size = 8.0
    st.grab_min_size = 8.0
    ar, ag, ab = acc
    C = imgui.Col_
    colors = {
        C.text: (0.94, 0.95, 0.96, 1.0),
        C.text_disabled: (1.0, 1.0, 1.0, 0.56),
        C.window_bg: PANEL_BG,
        C.child_bg: (0.0, 0.0, 0.0, 0.0),
        C.popup_bg: (0.03, 0.04, 0.06, 0.96),
        C.border: (1.0, 1.0, 1.0, LINE),
        C.border_shadow: (0.0, 0.0, 0.0, 0.0),
        C.frame_bg: (1.0, 1.0, 1.0, 0.03),
        C.frame_bg_hovered: (1.0, 1.0, 1.0, 0.08),
        C.frame_bg_active: (ar, ag, ab, 0.16),
        C.title_bg: PANEL_BG,
        C.title_bg_active: (0.03, 0.04, 0.06, 0.9),
        C.menu_bar_bg: PANEL_BG,
        C.scrollbar_bg: (0.0, 0.0, 0.0, 0.0),
        C.scrollbar_grab: (1.0, 1.0, 1.0, 0.16),
        C.scrollbar_grab_hovered: (1.0, 1.0, 1.0, 0.26),
        C.scrollbar_grab_active: (ar, ag, ab, 0.7),
        C.check_mark: (ar, ag, ab, 1.0),
        C.slider_grab: (ar, ag, ab, 0.9),
        C.slider_grab_active: (ar, ag, ab, 1.0),
        C.button: (1.0, 1.0, 1.0, 0.0),
        C.button_hovered: (ar, ag, ab, 0.16),
        C.button_active: (ar, ag, ab, 0.30),
        C.header: (ar, ag, ab, 0.14),
        C.header_hovered: (ar, ag, ab, 0.22),
        C.header_active: (ar, ag, ab, 0.32),
        C.separator: (1.0, 1.0, 1.0, 0.14),
        C.separator_hovered: (ar, ag, ab, 0.5),
        C.separator_active: (ar, ag, ab, 0.8),
        C.resize_grip: (1.0, 1.0, 1.0, 0.0),
        C.resize_grip_hovered: (ar, ag, ab, 0.4),
        C.resize_grip_active: (ar, ag, ab, 0.7),
        C.tab: (1.0, 1.0, 1.0, 0.0),
        C.tab_hovered: (ar, ag, ab, 0.2),
        C.tab_selected: (ar, ag, ab, 0.14),
        C.tab_selected_overline: (ar, ag, ab, 1.0),
        C.plot_lines: (ar, ag, ab, 1.0),
        C.plot_histogram: (ar, ag, ab, 1.0),
        C.table_header_bg: (1.0, 1.0, 1.0, 0.04),
        C.table_border_strong: (1.0, 1.0, 1.0, 0.16),
        C.table_border_light: (1.0, 1.0, 1.0, 0.08),
        C.table_row_bg_alt: (1.0, 1.0, 1.0, 0.025),
        C.text_selected_bg: (ar, ag, ab, 0.35),
        C.nav_cursor: (ar, ag, ab, 0.8),
        C.modal_window_dim_bg: (0.0, 0.0, 0.0, 0.5),
    }
    for k, v in colors.items():
        st.set_color_(k, V4(*v))
    _meas.clear()


# ----------------------------------------------------------------- text helpers
def _font(name):
    return ui.FONTS.get(name)


def text_w(font, size: float, text: str) -> float:
    """Width in pixels of `text` at `size` (unscaled, like push_font)."""
    key = (id(font), size, text)
    w = _meas.get(key)
    if w is None:
        imgui.push_font(font, size)
        w = imgui.calc_text_size(text).x
        imgui.pop_font()
        if len(_meas) > 4000:
            _meas.clear()
        _meas[key] = w
    return w


def spaced_w(font, size, text, tracking):
    return sum(text_w(font, size, ch) for ch in text) + tracking * size * ui.S * max(0, len(text) - 1)


SHADOW = c32(0.0, 0.0, 0.0, 0.6)


def put(dl, font, size, pos, colour, text, tracking=0.0, shadow=True):
    """Draw text at pos (top-left) with letter spacing `tracking` (em) and a soft drop shadow
    (keeps the readouts legible over bright pictures); returns its width."""
    f = font or imgui.get_font()
    px = size * ui.S
    o = max(1.0, round(ui.S))
    if tracking == 0.0:
        if shadow:
            dl.add_text(f, px, V2(pos.x + o, pos.y + o), SHADOW, text)
        dl.add_text(f, px, pos, colour, text)
        return text_w(font, size, text)
    x = pos.x
    for ch in text:
        if ch != " ":
            if shadow:
                dl.add_text(f, px, V2(x + o, pos.y + o), SHADOW, ch)
            dl.add_text(f, px, V2(x, pos.y), colour, ch)
        x += text_w(font, size, ch) + tracking * px
    return x - pos.x - tracking * px


def section(title: str):
    """Small-caps section label with a hairline (replaces ui.section in this style)."""
    imgui.dummy(V2(1, 4 * ui.S))
    dl = imgui.get_window_draw_list()
    p = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_avail().x
    size = 10.5
    put(dl, _font("hud_semibold"), size, V2(p.x, p.y), white(0.74), title.upper(), 0.22, shadow=False)
    y = p.y + size * ui.S + 7 * ui.S
    dl.add_line(V2(p.x, y), V2(p.x + w, y), white(0.14), 1.0)
    imgui.dummy(V2(1, size * ui.S + 11 * ui.S))


# ----------------------------------------------------------------- frame
def draw(app, W, H):
    font = _font("hud")
    if font is not None:
        imgui.push_font(font, 14.5)
    try:
        ui._context_menu(app)
        tl_h = ui.TIMELINE_H if (app.ui_visible and app.settings.show_timeline) else 0
        if app.ui_visible:
            _overlay(app, W, H, tl_h)
            _icon_row(app, W)
            if app.settings.sidebar_open and not app.show_help:     # the help card is wide
                _panel(app, W, H, tl_h)
            if tl_h:
                ui._timeline(app, W, H)
            ui._julia_preview(app)
        _toasts(app, W)
        if app.show_help:
            ui._help_overlay(app, W, H)
        if ui._state["script_help"]:
            ui._script_help(app, W, H)
    finally:
        if font is not None:
            imgui.pop_font()


# ----------------------------------------------------------------- overlay
def _status(app):
    """(text, progress 0..1, busy) of the picture on the screen."""
    f = app.field
    s = app.settings
    if getattr(app, "pending_ref", None) is not None:
        return f"REFERENCE {100 * app.ref_progress:.0f}%", app.ref_progress, True
    if f is None or f.params is None:
        return "STARTING", 0.0, True
    if not f.complete:
        return f"REFINING {100 * f.progress:.0f}%", f.progress, True
    if s.aa_passes > 1 and app.aa_count < s.aa_passes and not app.is_moving() and app.colors.cycle_speed == 0.0:
        return f"SMOOTHING {app.aa_count}/{s.aa_passes}", app.aa_count / s.aa_passes, True
    return "READY", 1.0, False


def _place(app) -> str:
    if app.julia_c is not None:
        c = app.julia_c
        return f"JULIA SET  c = {c[0]:.6f} {'+' if c[1] >= 0 else '-'} {abs(c[1]):.6f}i"
    title, pv = getattr(app, "place_title", None), getattr(app, "place_view", None)
    if title and pv is not None:
        v = app.view
        try:
            dx, dy = v.delta_in_pixels(pv, max(app.fb[1], 1))
            near = abs(v.log2r - pv.log2r) < 4.0 and math.hypot(dx, dy) < 1.5 * max(app.fb[1], 1)
        except Exception:
            near = False
        if near:
            return title.upper()
    return "BURNING SHIP" if getattr(app, "formula", "mandelbrot") == "ship" else "MANDELBROT SET"


def _gpu_short(name: str) -> str:
    if "llvmpipe" in (name or "").lower():
        from .software import threads
        return f"CPU · {threads()} THREADS"
    n = (name or "?").split("/")[0]
    for p in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "Intel(R) "):
        if n.startswith(p):
            n = n[len(p):]
    return n.upper()


def _ruler_max(depth: float) -> float:
    for m in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
        if depth <= m:
            return float(m)
    return float(10 ** math.ceil(math.log10(depth)))


def _overlay(app, W, H, tl_h):
    S = ui.S
    s = app.settings
    dl = imgui.get_background_draw_list()
    acc = accent(app)
    A = c32(*acc)
    sb = _font("hud_semibold")
    mono, light = _font("hud_mono"), _font("hud_mono_light")
    # scrims so the readouts stay legible over bright pictures
    dl.add_rect_filled_multi_color(V2(0, 0), V2(W, 140 * S), c32(0, 0, 0, 0.6), c32(0, 0, 0, 0.6),
                                   c32(0, 0, 0, 0.0), c32(0, 0, 0, 0.0))
    yb = H - tl_h
    dl.add_rect_filled_multi_color(V2(0, yb - 240 * S), V2(W, yb), c32(0, 0, 0, 0.0), c32(0, 0, 0, 0.0),
                                   c32(0, 0, 0, 0.7), c32(0, 0, 0, 0.7))
    if s.hud_brackets:
        a, L, col = 16 * S, 20 * S, white(0.45)
        for (x, y, sx, sy) in ((a, a, 1, 1), (W - a, a, -1, 1), (a, yb - a, 1, -1), (W - a, yb - a, -1, -1)):
            dl.add_line(V2(x, y), V2(x + sx * L, y), col, 1.0)
            dl.add_line(V2(x, y), V2(x, y + sy * L), col, 1.0)
    if s.hud_reticle and app.is_moving():
        cx, cy, g, L, col = W / 2, H / 2, 6 * S, 7 * S, white(0.55)
        dl.add_line(V2(cx - g - L, cy), V2(cx - g, cy), col, 1.0)
        dl.add_line(V2(cx + g, cy), V2(cx + g + L, cy), col, 1.0)
        dl.add_line(V2(cx, cy - g - L), V2(cx, cy - g), col, 1.0)
        dl.add_line(V2(cx, cy + g), V2(cx, cy + g + L), col, 1.0)
    # wordmark, place, picture status
    x, y = 40 * S, 30 * S
    dl.add_circle_filled(V2(x + 3 * S, y + 6.5 * S), 3 * S, A)
    x += 20 * S
    x += put(dl, sb, 12, V2(x, y), white(0.96), "MANDELSCOPE", 0.32) + 14 * S
    dl.add_line(V2(x, y + 1 * S), V2(x, y + 14 * S), white(0.35), 1.0)
    x += 14 * S
    put(dl, mono, 12, V2(x, y), white(0.82), _place(app), 0.08)
    text, prog, busy = _status(app)
    y2 = 56 * S
    put(dl, _font("hud_mono_medium") or mono, 10, V2(60 * S, y2), A if busy else white(0.6), text, 0.16)
    bw = 120 * S
    dl.add_rect_filled(V2(60 * S, y2 + 17 * S), V2(60 * S + bw, y2 + 19 * S), white(0.18))
    dl.add_rect_filled(V2(60 * S, y2 + 17 * S), V2(60 * S + bw * max(0.0, min(1.0, prog)), y2 + 19 * S),
                       A if busy else white(0.5))
    # magnification, depth ruler, chips (bottom left)
    v = app.view
    depth = max(0.0, v.zoom_magnification_log10())
    x0, y0 = 40 * S, yb - 184 * S
    put(dl, sb, 10, V2(x0, y0), white(0.78), "MAGNIFICATION", 0.32)
    big = 54
    w10 = put(dl, light, big, V2(x0, y0 + 20 * S), white(0.97), "10")
    put(dl, light, 26, V2(x0 + w10 + 4 * S, y0 + 22 * S), white(0.97), f"{depth:.2f}")
    rw, ry = 420 * S, y0 + 92 * S
    m = _ruler_max(depth)
    dl.add_line(V2(x0, ry), V2(x0 + rw, ry), white(0.3), 1.0)
    dl.add_line(V2(x0, ry), V2(x0 + rw * min(1.0, depth / m), ry), A, 1.0)
    for k in range(11):
        tx = x0 + rw * k / 10
        major = k in (0, 5, 10)
        dl.add_line(V2(tx, ry - (4 if major else 2) * S), V2(tx, ry + (5 if major else 3) * S),
                    white(0.55 if major else 0.4), 1.0)
    dl.add_circle_filled(V2(x0 + rw * min(1.0, depth / m), ry), 3.5 * S, A)
    put(dl, mono, 9, V2(x0, ry + 9 * S), white(0.55), "0", 0.12)
    lbl = "DEPTH · DECADES"
    put(dl, mono, 9, V2(x0 + rw / 2 - spaced_w(mono, 9, lbl, 0.12) / 2, ry + 9 * S), white(0.55), lbl, 0.12)
    ms = f"{m:g}"
    put(dl, mono, 9, V2(x0 + rw - spaced_w(mono, 9, ms, 0.12), ry + 9 * S), white(0.55), ms, 0.12)
    re, im = float(v.re), float(v.im)
    chips = [f"{re:.6f} {'+' if im >= 0 else '-'}{abs(im):.6f}i", f"{v.max_iter:,} IT",
             f"ROT {math.degrees(v.angle) % 360:.0f}°", f"{app.fps:.0f} FPS"]
    cx = x0
    for t in chips:
        cx += put(dl, mono, 12, V2(cx, ry + 30 * S), white(0.86), t, 0.04) + 22 * S
    # performance / jobs (bottom right)
    lines = []
    run = next((j for j in app.jobs if j.status == "running"), None)
    if run is not None:
        lines.append((A, f"{run.name.upper()} {100 * run.progress:.0f}% · ETA {ui._fmt_t(run.eta)}"))
    if app.playing:
        lines.append((A, f"PLAYING {app.play_t:.1f} S"))
    if app.path_rec is not None:
        lines.append((c32(0.45, 1.0, 0.55), f"FLIGHT · {len(app.path_rec.samples)} SAMPLES"))
    f = app.field
    if f is not None and f.complete and f.finished_at and not busy:
        lines.append((white(0.78), f"PICTURE {(f.finished_at - f.started) * 1000:.0f} MS"))
    lines.append((white(0.78), f"FRAME {app.frame_ms:.0f} MS · BUDGET {s.frame_budget_ms:.0f} MS"))
    from .renderer import PRECISION_NAMES
    prec = PRECISION_NAMES.get(int(app.delta_precision()), "float32").upper() if app.engine is not None else "FLOAT32"
    lines.append((white(0.78), f"{prec} · {_gpu_short(app.engine.renderer_name if app.engine else '')}"))
    y = yb - 40 * S - 18 * S * len(lines)
    for colour, t in lines:
        w = spaced_w(mono, 10, t, 0.14)
        put(dl, mono, 10, V2(W - 40 * S - w, y), colour, t, 0.14)
        y += 18 * S
    # recording: a red pill at the top centre
    if app.live is not None:
        secs = app.live.writer.frames / max(1.0, getattr(app.live, "fps", 30.0) or 30.0)
        t = f"REC  {int(secs) // 60:02d}:{int(secs) % 60:02d}"
        w = spaced_w(_font("hud_mono_medium") or mono, 10, t, 0.16) + 34 * S
        px, py = W / 2 - w / 2, 26 * S
        dl.add_rect_filled(V2(px, py), V2(px + w, py + 26 * S), c32(0.94, 0.44, 0.40, 0.9), 13 * S)
        dl.add_circle_filled(V2(px + 14 * S, py + 13 * S), 3.5 * S, white(1.0))
        put(dl, _font("hud_mono_medium") or mono, 10, V2(px + 24 * S, py + 7 * S), white(1.0), t, 0.16)
    # picture progress along the bottom edge
    dl.add_rect_filled(V2(0, yb - 2 * S), V2(W, yb), white(0.16))
    dl.add_rect_filled(V2(0, yb - 2 * S), V2(W * max(0.0, min(1.0, prog)), yb), A if busy else c32(*acc, 0.55))


# ----------------------------------------------------------------- icon row (top right)
def _ghost(icon: str, tip: str, active=False, colour=None, dim=False) -> bool:
    S = ui.S
    acc = ui.ACCENT
    if active:
        imgui.push_style_color(imgui.Col_.text, V4(*acc, 1.0))
    elif colour is not None:
        imgui.push_style_color(imgui.Col_.text, V4(*colour))
    else:
        imgui.push_style_color(imgui.Col_.text, V4(1, 1, 1, 0.45 if dim else 0.86))
    imgui.push_style_color(imgui.Col_.button, V4(0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_hovered, V4(1, 1, 1, 0.08))
    imgui.push_style_color(imgui.Col_.button_active, V4(*acc, 0.22))
    r = imgui.button(icon, V2(38 * S, 38 * S))
    imgui.pop_style_color(4)
    if active:
        mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        imgui.get_window_draw_list().add_circle_filled(V2((mn.x + mx.x) / 2, mx.y - 4 * S), 1.6 * S, c32(*acc))
    if imgui.is_item_hovered():
        imgui.set_tooltip(tip)
    return r


def _bar_sep():
    S = ui.S
    imgui.same_line(0, 8 * S)
    p = imgui.get_cursor_screen_pos()
    imgui.get_window_draw_list().add_line(V2(p.x, p.y + 11 * S), V2(p.x, p.y + 27 * S), white(0.28), 1.0)
    imgui.dummy(V2(1, 38 * S))
    imgui.same_line(0, 8 * S)


def _rec_pill(app) -> bool:
    S = ui.S
    mono = _font("hud_mono_medium") or _font("hud_mono")
    on = app.live is not None
    label = "STOP" if on else "REC"
    w = spaced_w(mono, 10, label, 0.14) + 34 * S
    p = imgui.get_cursor_screen_pos()
    y = p.y + 5 * S
    clicked = imgui.invisible_button("##rec", V2(w, 38 * S))
    hov = imgui.is_item_hovered()
    dl = imgui.get_window_draw_list()
    if on:
        dl.add_rect_filled(V2(p.x, y), V2(p.x + w, y + 28 * S), c32(0.94, 0.44, 0.40, 0.85), 14 * S)
    dl.add_rect(V2(p.x, y), V2(p.x + w, y + 28 * S), white(0.6 if hov else 0.38), 14 * S, 1.0)
    dl.add_circle_filled(V2(p.x + 14 * S, y + 14 * S), 3.5 * S, white(1.0) if on else c32(0.94, 0.44, 0.40))
    put(dl, mono, 10, V2(p.x + 24 * S, y + 7.5 * S), white(0.95), label, 0.14, shadow=False)
    if hov:
        imgui.set_tooltip("Stop the live recording (R)" if on else "Record the screen live (R)")
    return clicked


def _icon_row(app, W):
    S = ui.S
    imgui.set_next_window_pos(V2(W - 30 * S, 18 * S), imgui.Cond_.always, V2(1.0, 0.0))
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(4 * S, 2 * S))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, V2(0, 0))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
    imgui.begin("##hudbar", None, FIXED | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_scrollbar |
                imgui.WindowFlags_.always_auto_resize)
    if _ghost(fa.ICON_FA_HOUSE, "Home - whole set (H)"):
        app.go_home()
    imgui.same_line()
    if _ghost(fa.ICON_FA_ARROW_LEFT, "Back (Backspace)", dim=not app.history):
        app.back()
    imgui.same_line()
    if _ghost(fa.ICON_FA_ARROW_RIGHT, "Forward (Shift+Backspace)", dim=not app.future):
        app.forward()
    _bar_sep()
    if _ghost(fa.ICON_FA_CROSSHAIRS, "Find the nearest minibrot (M at the cursor)"):
        app.find_minibrot_at(app.view.re, app.view.im)
    imgui.same_line()
    ship = app.formula == "ship"
    if _ghost(fa.ICON_FA_SAILBOAT, "Back to the Mandelbrot set (B)" if ship else "The Burning Ship (B)", ship):
        app.set_formula("mandelbrot" if ship else "ship")
    imgui.same_line()
    if ship:
        pass
    elif app.julia_c is None:
        if _ghost(fa.ICON_FA_SNOWFLAKE, "Julia set preview for the point under the cursor (J)\n"
                  "Shift+J: explore the Julia set of that point", app.settings.julia_preview):
            app.settings.julia_preview = not app.settings.julia_preview
    else:
        if _ghost(fa.ICON_FA_ARROW_TURN_UP, "Back to the Mandelbrot set (J)", True):
            app.leave_julia()
    imgui.same_line()
    if _ghost(fa.ICON_FA_BOOKMARK, "Save this location to the library"):
        app.save_to_library("")
    _bar_sep()
    if _rec_pill(app):
        app.toggle_live()
    imgui.same_line(0, 4 * S)
    if _ghost(fa.ICON_FA_ROUTE, "Record your flight as a timeline (P)", app.path_rec is not None):
        app.toggle_path_recording()
    imgui.same_line()
    if _ghost(fa.ICON_FA_CAMERA, "Screenshot, 2x supersampled (F12)"):
        app.start_still(quick=True)
    _bar_sep()
    if _ghost(fa.ICON_FA_PAUSE if app.playing else fa.ICON_FA_PLAY, "Play / pause the timeline (Space)", app.playing):
        app.toggle_play()
    imgui.same_line()
    if _ghost(fa.ICON_FA_DIAMOND, "Add keyframe at the current view (K)"):
        app.add_keyframe()
    imgui.same_line()
    if _ghost(fa.ICON_FA_TIMELINE, "Show / hide the timeline (T)", app.settings.show_timeline):
        app.settings.show_timeline = not app.settings.show_timeline
    imgui.same_line()
    if _ghost(fa.ICON_FA_TABLE_COLUMNS, "Show / hide the side panel (Tab hides everything)", app.settings.sidebar_open):
        app.settings.sidebar_open = not app.settings.sidebar_open
    imgui.same_line()
    if _ghost(fa.ICON_FA_KEYBOARD, "Keys, mouse and controller (F1)", app.show_help):
        app.show_help = not app.show_help
    imgui.end()
    imgui.pop_style_var(3)


# ----------------------------------------------------------------- side panel
TAB_LABEL = {"Colour": "LOOK"}


def _tab_strip(app):
    S = ui.S
    s = app.settings
    acc = ui.ACCENT
    dl = imgui.get_window_draw_list()
    p0 = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail().x
    h = 42 * S
    sb = _font("hud_semibold")
    names = ui.TABS
    cur = s.sidebar_tab if s.sidebar_tab in names else names[0]
    label = TAB_LABEL.get(cur, cur.upper())
    icon_w = text_w(None, 14.5, ui.TAB_ICONS[cur])
    active_w = icon_w + 8 * S + spaced_w(sb, 10, label, 0.22) + 26 * S
    other_w = (avail - active_w) / max(1, len(names) - 1)
    x = p0.x
    for name in names:
        w = active_w if name == cur else other_w
        imgui.set_cursor_screen_pos(V2(x, p0.y))
        if imgui.invisible_button(f"##hudtab{name}", V2(w, h)):
            s.sidebar_tab = name
        hov = imgui.is_item_hovered()
        icon = ui.TAB_ICONS[name]
        iw = text_w(None, 14.5, icon)
        ih = imgui.get_font_size()
        if name == cur:
            colour = c32(*acc)
            tx = x + 13 * S
            dl.add_text(V2(tx, p0.y + (h - ih) / 2), colour, icon)
            put(dl, sb, 10, V2(tx + iw + 8 * S, p0.y + (h - 10 * S) / 2 - 1 * S), colour, label, 0.22, shadow=False)
            dl.add_rect_filled(V2(x + 4 * S, p0.y + h - 2 * S), V2(x + w - 4 * S, p0.y + h), colour)
        else:
            dl.add_text(V2(x + (w - iw) / 2, p0.y + (h - ih) / 2), white(1.0 if hov else 0.72), icon)
            if hov:
                imgui.set_tooltip(TAB_LABEL.get(name, name).capitalize())
        x += w
    dl.add_line(V2(p0.x - 6 * S, p0.y + h), V2(p0.x + avail + 6 * S, p0.y + h), white(0.14), 1.0)
    imgui.set_cursor_screen_pos(V2(p0.x, p0.y + h + 1))
    imgui.dummy(V2(1, 1))


def _panel(app, W, H, tl_h):
    S = ui.S
    sw = app.settings.sidebar_width * S
    y0 = 92 * S
    h = H - tl_h - y0 - 128 * S
    imgui.set_next_window_pos(V2(W - sw - 40 * S, y0))
    imgui.set_next_window_size(V2(sw, max(220 * S, h)))
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(6 * S, 0))
    imgui.begin("##hudpanel", None, FIXED | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)
    imgui.pop_style_var()
    _tab_strip(app)
    name = app.settings.sidebar_tab if app.settings.sidebar_tab in ui.TABS else ui.TABS[0]
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(12 * S, 10 * S))
    imgui.begin_child("##hudbody", V2(0, 0), imgui.ChildFlags_.always_use_window_padding)
    {"Explore": ui._tab_explore, "Colour": ui._tab_colour, "Library": ui._tab_library,
     "Animate": ui._tab_animate, "Export": ui._tab_export, "Render": ui._tab_render}[name](app)
    imgui.end_child()
    imgui.pop_style_var()
    imgui.end()


# ----------------------------------------------------------------- toasts (top centre)
def _toasts(app, W):
    S = ui.S
    now = time.perf_counter()
    msgs = [(t, m) for t, m in app.messages if now - t < 6.0][-3:]
    if not msgs:
        return
    dl = imgui.get_foreground_draw_list()
    mono = _font("hud_mono_medium") or _font("hud_mono")
    y = (62 if app.live is not None else 26) * S
    for t, m in msgs:
        a = max(0.0, min(1.0, (6.0 - (now - t)) / 1.0))
        w = text_w(mono, 11, m) + 28 * S
        x = W / 2 - w / 2
        dl.add_rect_filled(V2(x, y), V2(x + w, y + 26 * S), c32(0, 0, 0, 0.5 * a), 13 * S)
        dl.add_rect(V2(x, y), V2(x + w, y + 26 * S), white(0.28 * a), 13 * S, 1.0)
        put(dl, mono, 11, V2(x + 14 * S, y + 6.5 * S), white(0.92 * a), m, shadow=False)
        y += 32 * S
