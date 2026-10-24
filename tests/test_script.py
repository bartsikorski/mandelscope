"""Script language, timeline <-> script round trip, location library, palettes, launcher."""
import math

import pytest

from mandelscope.mp import View
from mandelscope.script import (SCRIPT_HELP, compile_script, sequence_to_script, ai_prompt,
                             parse_duration, parse_log10_magnitude)
from mandelscope.sequence import Sequence, Keyframe
from mandelscope import palettes


def _lib(name):
    if name.lower() in ("seahorse", "seahorse valley"):
        return View.from_strings("-0.7453", "0.1127", math.log2(0.0065), 0.0, 2000)
    return None


def test_help_text_is_a_valid_script():
    r = compile_script(SCRIPT_HELP, resolve_location=_lib)
    assert r.ok, r.errors
    seq = r.sequence
    assert len(seq.keyframes) >= 7
    times = [k.time for k in seq.keyframes]
    assert times == sorted(times)
    assert r.fps == 60 and r.size == (1920, 1080) and r.look == "Nebula"


def test_values():
    assert parse_duration("1.5min") == 90.0
    assert parse_duration("500ms") == 0.5
    assert parse_duration("20") == 20.0
    assert parse_log10_magnitude("1e5000") == pytest.approx(5000.0)
    assert parse_log10_magnitude("10^800") == pytest.approx(800.0)
    assert parse_log10_magnitude("2^10") == pytest.approx(10 * math.log10(2))


def test_shots_and_timing():
    r = compile_script("start home\nzoom in 1e6 over 20s spin 1\nhold 2s\nrotate 90 over 3s\n"
                       "zoom to location seahorse over 10s\n", resolve_location=_lib)
    assert r.ok, r.errors
    ks = r.sequence.keyframes
    assert [k.time for k in ks] == [0.0, 20.0, 22.0, 25.0, 35.0]
    z0 = ks[0].view.zoom_magnification_log10()
    assert ks[1].view.zoom_magnification_log10() - z0 == pytest.approx(6.0, abs=1e-6)
    assert math.degrees(ks[1].view.angle - ks[0].view.angle) == pytest.approx(360.0)
    assert math.degrees(ks[3].view.angle - ks[2].view.angle) == pytest.approx(90.0)
    assert ks[4].view.zoom_magnification_log10() == pytest.approx(_lib("seahorse").zoom_magnification_log10())


def test_zoom_to_coordinates_and_colour_animation():
    r = compile_script("look Ember\nstart home\n"
                       "zoom to re=-0.743643887037158704752191506114774 im=0.131825904205311970493132056385139 "
                       "zoom=1e40 angle=30 over 40s\nanimate offset by 1 over 5s\n")
    assert r.ok, r.errors
    k = r.sequence.keyframes
    assert k[1].view.zoom_magnification_log10() == pytest.approx(40.0, abs=1e-9)
    assert format(k[1].view.re, ".30e").startswith("-7.43643887037158704752191506114")
    assert math.degrees(k[1].view.angle) == pytest.approx(30.0)
    assert k[2].color["offset"] == pytest.approx(k[1].color.get("offset", 0.0) + 1.0)
    assert k[0].color["palette"] == palettes.LOOKS["Ember"]["palette"]


def test_errors_have_line_numbers_and_change_nothing():
    r = compile_script("start home\nzoom sideways\nzoom in 1e3 over 2s\nfly to nowhere over 3s\n")
    assert not r.ok
    lines = [ln for ln, _ in r.errors]
    assert 2 in lines and 4 in lines
    assert compile_script("").errors


def test_timeline_script_round_trip_is_exact():
    v0 = View.from_strings("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139",
                           math.log2(1e-30), 0.4, 9000)
    v1 = v0.copy()
    v1.log2r = math.log2(1e-35)
    seq = Sequence([Keyframe(View.home(), 0.0, "smooth"), Keyframe(v0, 7.25), Keyframe(v1, 12.0, "ease-out")], fps=30)
    txt = sequence_to_script(seq)
    back = compile_script(txt).sequence
    assert len(back.keyframes) == 3 and back.fps == 30
    for a, b in zip(seq.keyframes, back.keyframes):
        assert a.time == pytest.approx(b.time)
        assert a.ease == b.ease
        assert a.view.max_iter == b.view.max_iter
        assert a.view.log2r == pytest.approx(b.view.log2r, abs=1e-5)
        assert a.view.angle == pytest.approx(b.view.angle, abs=1e-6)
        h = 2.0 ** a.view.log2r
        assert abs(float(a.view.re - b.view.re)) < 1e-6 * h
        assert abs(float(a.view.im - b.view.im)) < 1e-6 * h


def test_ai_prompt_contains_reference_and_script():
    p = ai_prompt("start home\nzoom in 1e3 over 5s", View.home(), ['"home"'], ["Gilded"])
    assert "zoom in 1e3 over 5s" in p and "start home" in p and "Gilded" in p
    assert "key t=" in p        # the reference part


def test_smooth_zoom_speed_is_monotone():
    vs = [View.home()]
    for d in (5, 6, 30):
        v = vs[-1].copy()
        v.log2r -= d
        vs.append(v)
    seq = Sequence([Keyframe(v, t) for v, t in zip(vs, (0.0, 5.0, 8.0, 30.0))])
    ts = [i * 0.05 for i in range(int(30 / 0.05) + 1)]
    l2 = [seq.view_at(t)[0].log2r for t in ts]
    assert all(b <= a + 1e-9 for a, b in zip(l2, l2[1:]))       # never zooms back out
    # no speed jumps at keyframes: neighbouring speed differences are small
    sp = [a - b for a, b in zip(l2, l2[1:])]
    jumps = [abs(b - a) for a, b in zip(sp, sp[1:])]
    assert max(jumps) < 0.05


def test_library_save_find_rename_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("MANDELSCOPE_HOME", str(tmp_path))
    import numpy as np
    from mandelscope.library import Library, location_from_png
    lib = Library()
    builtins = [e for e in lib.entries if e.builtin]
    assert len(builtins) >= 10 and all(e.thumb is not None for e in builtins)
    # shallow to deep, the Mandelbrot set's places first, then the Burning Ship's
    assert [e.formula for e in builtins] == sorted((e.formula for e in builtins), key=lambda f: f != "mandelbrot")
    for f in ("mandelbrot", "ship"):
        z = [e.zoom for e in builtins if e.formula == f]
        assert len(z) >= 5 and z == sorted(z)
    assert lib.view("ship_armada").formula == "ship" and lib.view("home").formula == "mandelbrot"
    assert lib.view("Julia island") is not None          # by title
    assert lib.view("julia_island") is not None          # by file name
    loc = {"view": View.home().to_dict()}
    e = lib.save("my spot", loc, np.zeros((90, 160, 3), np.uint8))
    assert e.thumb is not None and e.thumb.exists()
    e2 = lib.save("my spot", loc)
    assert e2.name == "my spot (2)"
    assert lib.rename("my spot (2)", "other")
    assert lib.find("other") is not None
    assert lib.delete("other") and lib.find("other") is None
    assert not lib.delete("home")                          # built-ins are read-only
    assert location_from_png(str(e.thumb)) is None


def test_palettes_and_looks():
    for name in palettes.palette_names():
        p = palettes.make_palette(name, 64)
        assert p.shape == (64, 3) and float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    for look, d in palettes.LOOKS.items():
        assert d["palette"] in palettes.PALETTES, look
    st = palettes.random_stops(seed=3)
    pos = [p for p, _ in st]
    assert pos == sorted(pos) and pos[0] == 0.0 and min(b - a for a, b in zip(pos, pos[1:])) > 0.05
    assert palettes.to_hex(palettes.from_hex("#12abef")) == "#12abef"
    palettes.set_custom("test custom", st)
    assert "test custom" in palettes.all_palette_names()
    v = palettes.palette_version("test custom")
    palettes.set_custom("test custom", st[:3])
    assert palettes.palette_version("test custom") == v + 1
    palettes.delete_custom("test custom")


def test_launcher_arguments():
    from mandelscope.launcher import _args
    assert _args([]) == ["gui"]
    assert _args(["C:/x/spot.json"]) == ["gui", "C:/x/spot.json"]
    assert _args(["render", "a.json"]) == ["render", "a.json"]
    assert _args(["--help"]) == ["--help"]


def test_scripts_for_the_burning_ship():
    """'fractal ship' (or a Burning Ship place) makes the timeline a Burning Ship one; mixing
    fractals is an error; the timeline script keeps the fractal."""
    def lib(name):
        if name == "armada":
            v = View.from_strings("-1.7615", "0.029", math.log2(0.034), 0.0, 3000)
            v.formula = "ship"
            return v
        return _lib(name)

    r = compile_script("fractal ship\nstart home\nzoom in 1e3 over 4s\n", resolve_location=lib)
    assert r.ok, r.errors
    assert r.sequence.meta.get("formula") == "ship" and r.formula == "ship"
    assert float(r.sequence.keyframes[0].view.re) == pytest.approx(-0.48)       # the ship's home
    r = compile_script("start location armada\nzoom in 1e3 over 4s\n", resolve_location=lib)
    assert r.ok and r.sequence.meta.get("formula") == "ship"
    r = compile_script("start home\nzoom to location armada over 4s\n", resolve_location=lib)
    assert not r.ok and "Burning Ship" in r.errors[0][1]
    r = compile_script("start home\nfractal ship\n", resolve_location=lib)
    assert not r.ok
    r = compile_script("start here\nzoom in 4x over 1s\n", current_view=View.home(), current_formula="ship")
    assert r.ok and r.sequence.meta.get("formula") == "ship"
    r = compile_script("start home\nzoom in 4x over 1s\n")
    assert r.ok and "formula" not in r.sequence.meta
    seq = compile_script("fractal ship\nstart home\nzoom in 1e3 over 4s\n").sequence
    again = compile_script(sequence_to_script(seq))
    assert again.ok and again.sequence.meta.get("formula") == "ship"


def test_categorical_colour_settings_stay_integers_between_keyframes():
    """(Mapping, method and pattern are choices: interpolating them made 3.0 out of 3 and
    broke the colour panel.)"""
    a = Keyframe(View.home(), 0.0, "linear", {"mapping": 3, "method": 0, "density": 0.03})
    b = Keyframe(View.home(), 2.0, "linear", {"mapping": 1, "method": 0, "density": 0.06})
    seq = Sequence([a, b], fps=10)
    for t in (0.0, 0.7, 1.4, 2.0):
        _, over = seq.view_at(t)
        assert type(over["mapping"]) is int and type(over["method"]) is int
        assert over["mapping"] == (3 if t < 1.0 else 1)
    assert seq.view_at(1.0)[1]["density"] == pytest.approx(0.045)


def test_readme_clips_compile():
    """The README animations (tools/readme_media.py) are scripts: they must stay valid."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "tools" / "readme_media.py"
    spec = importlib.util.spec_from_file_location("readme_media", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name, (text, (w, h), fps, julia) in m.CLIPS.items():
        r = compile_script(text, resolve_location=m.resolve)
        assert r.ok and not r.warnings, (name, r.errors, r.warnings)
        assert w % 2 == 0 and h % 2 == 0, name           # (video sizes must be even)
        assert r.sequence.duration > 5, name
    for name, (loc, *_rest) in m.STILLS.items():
        assert isinstance(loc, dict) or m.resolve(loc) is not None, name
