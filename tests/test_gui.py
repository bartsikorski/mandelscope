"""End-to-end GUI session (scripted).  Opens a window (or uses Xvfb on Linux).

Exercises: wheel zoom, drag pan with sample reuse, keyboard zoom, rotation,
presets down to 1e-1000, float64 mode, colours, keyframes + playback, live
recording, path recording, offline video and still export, minibrot finder.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]


def test_scripted_gui_session(tmp_path):
    cmd = [sys.executable, "-m", "mandelscope", "gui", "--size", "960x540", "--selftest",
           "--selftest-dir", str(tmp_path)]
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        if not shutil.which("xvfb-run"):
            pytest.skip("no display and no xvfb-run")
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1400x900x24"] + cmd
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["errors"], report["errors"]
    names = {s["name"] for s in report["steps"]}
    shots = ["01_home", "02_wheel_zoom", "03_pan", "04_keyboard_zoom", "05_rotated_90", "06_rotate_snap",
             "07_spiral_1e-100", "08_colour_tab", "09_look_ember", "10_stripes", "11_deep_1e-1000",
             "12_deep_f64", "13_library_tab", "14_animate_tab", "15_playback", "16_export_tab",
             "17_seahorse", "18_minibrot", "19_back", "20_help", "21_context_menu", "22_clean_view",
             "23_classic_style", "24_welcome", "25_controller"]
    for shot in shots:
        assert (tmp_path / f"{shot}.png").exists(), shot
    for name in shots[:13]:
        if name not in ("08_colour_tab",):
            assert name in names, name
    pad = [st for st in report["steps"] if st["name"] == "controller"]
    assert pad and pad[0]["moved_right"] and pad[0]["zoomed_in"] and pad[0]["turned"], pad
    steps = {s["name"]: s for s in report["steps"]}
    assert steps["pan_reuse"]["done"] > 0.3 * steps["pan_reuse"]["N"]        # samples were reused
    assert steps["05_rotated_90"]["angle"] == -90.0                          # [ = 90 deg counter-clockwise
    assert steps["06_rotate_snap"]["angle"] == -45.0                         # drag follows the mouse, 15 deg snap
    assert steps["script"]["ok"] and steps["script"]["keyframes"] >= 4
    assert not steps["bad_script"]["ok"] and steps["bad_script"]["errors"]
    assert steps["library_save"]["thumb"]
    assert steps["path_sequence"]["keyframes"] > 5
    assert steps["minibrot"]["result"] is not None
    # zooming out of a deep place: the raised iteration limit fades with the depth and the
    # skip tables stay on (v3.11: the limit stayed x44 and every other frame had no skips)
    zo = steps["zoom_out"]
    assert zo["end_zoom"] < zo["start_zoom"] - 4, zo
    assert zo["end_iter"] < 1.5 * zo["plain_iter"], zo
    # (frames without skips while the next table is built: the v3.11 lag was 49 of 67; a fast GPU
    # shows more frames in the same build time - RTX 5060 Ti: 8 of ~176)
    assert zo["no_bla"] <= max(3, 0.1 * zo["frames"]), zo
    assert steps["context_menu"]["open"]
    assert steps["ui_hud"]["style"] == "hud" and steps["ui_hud"]["hud_fonts"]      # v3.8 default style
    assert steps["ui_classic"]["style"] == "classic"
    assert all(j[1] == "done" for j in steps["video_export"]["jobs"]), steps["video_export"]
    # v3 motion: zoom steps by powers of two reuse samples; pictures in flight are kept while
    # moving and the screen stays covered when zooming in
    assert any("zoom x4: reused" in r for r in steps["wheel_zoom_4x"]["resets"]), steps["wheel_zoom_4x"]
    assert any("zoom x2: reused" in r for r in steps["tap_zoom_2x"]["resets"]), steps["tap_zoom_2x"]
    assert steps["tap_zoom_2x"]["done"] > 0.2 * steps["tap_zoom_2x"]["N"]
    # (frames that kept computing an unfinished picture instead of restarting it; a fast GPU
    # finishes most pictures within a frame - RTX 5060 Ti: 5 in the 30 frames of the hold)
    assert steps["hold_zoom_in"]["in_flight"] > 0
    assert steps["hold_zoom_in"]["black"] == 0.0                # 'black' = pixels no picture covers
    for name in ("hold_zoom_out", "hold_rotate", "tap_zoom_out", "rotate_90_anim"):
        assert name in steps
    out = report["outputs"]
    assert any(k.startswith("Videos/selftest") and k.endswith(".mp4") and v > 1000 for k, v in out.items()), out
    assert any(k.startswith("Videos/live_") and v > 1000 for k, v in out.items()), out
    assert any(k.startswith("Images/") and k.endswith(".png") for k, v in out.items()), out
    assert any(k.startswith("Locations/selftest spot") for k in out), out
    # precise modes: float64, then float-float (two float32 per number)
    assert steps["12_deep_f64"]["complete"] and steps["12b_deep_ff"]["complete"]
    assert steps["deep_f64"]["double"] is True and steps["deep_ff"]["double"] == 2, (steps["deep_f64"], steps["deep_ff"])
    # Julia sets
    assert steps["julia_preview"]["c"] is not None and steps["julia_preview"]["done"], steps["julia_preview"]
    jm = steps["julia_mode"]
    assert jm["c"] is not None and jm["program"], jm
    assert steps["19b_julia_set"]["complete"] and steps["19c_julia_zoom"]["complete"]
    assert steps["19d_julia_pan"]["complete"] and steps["19e_julia_deep"]["complete"]
    for k in ("julia_deep", "julia_deeper"):                # perturbation, any depth (2^-50, 2^-400)
        jd = steps[k]
        assert jd["deep"] and not jd["program"] and jd["distinct"] > 20, jd
    assert steps["19g_julia_deeper"]["complete"] and steps["19h_julia_deeper_wheel"]["complete"]
    assert steps["19h_julia_deeper_wheel"]["zoom"] > steps["19g_julia_deeper"]["zoom"]
    jl = steps["19i_julia_deep_looks"]
    assert jl["done"] and len(jl["stds"]) >= 6 and min(jl["stds"].values()) > 3, jl     # not flat
    assert all(j[1] == "done" for j in steps["julia_still"]["jobs"]), steps["julia_still"]
    assert steps["julia_location"]["loc"] == list(jm["c"])
    assert steps["julia_left"]["c"] is None and steps["julia_left"]["program"] is False
    assert steps["19f_back_to_mandelbrot"]["complete"]
    # the Burning Ship
    for k in ("ship_home", "ship_deep"):
        sh = steps[k]
        assert sh["formula"] == "ship" and sh["field_formula"] == "ship" and not sh["program"], sh
        assert sh["distinct"] > 20 and sh["loc_formula"] == "ship", sh
    assert steps["22_ship_home"]["complete"] and steps["22b_ship_zoom"]["complete"]
    assert steps["22b_ship_zoom"]["zoom"] > steps["22_ship_home"]["zoom"]
    assert steps["22c_ship_mini_1e-33"]["complete"] and steps["22c_ship_mini_1e-33"]["zoom"] > 30
    mb = steps["ship_minibrot"]["result"]
    assert mb is not None and mb["period"] == 228, mb               # the mini ship of that place
    assert steps["22d_ship_minibrot"]["complete"]
    assert all(j[1] == "done" for j in steps["ship_still"]["jobs"]), steps["ship_still"]
    assert steps["ship_julia"]["julia"] is None and steps["ship_julia"]["formula"] == "ship"
    sl = steps["ship_left"]
    assert sl["formula"] == "mandelbrot" and sl["field_formula"] == "mandelbrot" and sl["same_view"], sl
    assert steps["22e_back_to_mandelbrot"]["complete"]
    for shot in ("19a_julia_preview", "19b_julia_set", "19c_julia_zoom", "19d_julia_pan", "19f_back_to_mandelbrot"):
        assert (tmp_path / f"{shot}.png").exists(), shot
    assert "internal error" not in r.stdout, r.stdout[-3000:]
