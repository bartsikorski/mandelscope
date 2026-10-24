"""Video encoding and offline sequence rendering."""
import re
import subprocess

import numpy as np
import pytest
from PIL import Image

from mandelscope.offline import RenderSettings, FrameRenderer, render_sequence_iter
from mandelscope.sequence import Sequence, Keyframe
from mandelscope.video import FfmpegWriter, find_ffmpeg, resolve_codec


def count_frames(path):
    ff = find_ffmpeg()
    r = subprocess.run([ff, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(m[-1]) if m else 0


def test_ffmpeg_available():
    assert find_ffmpeg() is not None
    assert resolve_codec("auto") in ("hevc_nvenc", "h264_nvenc", "av1_nvenc", "hevc_amf", "hevc_qsv",
                                     "libx264", "libx265")


def test_writer_writes_all_frames(tmp_path):
    p = tmp_path / "t.mp4"
    w = FfmpegWriter(str(p), 64, 48, 30, "auto", 20)
    for i in range(12):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, :, 0] = i * 20
        w.write(img)
    rc, err = w.close()
    assert rc == 0, err
    assert count_frames(p) == 12


@pytest.mark.gpu
def test_sequence_video_and_png_frames_match_stills(engine, locations, tmp_path):
    from mandelscope.cli import load_location
    v1, colors = load_location(str(locations / "spiral_1e-100.json"))
    colors.mapping = 1          # 'auto' contrast is smoothed over time in videos (no flicker), so
                                # use a per-frame mapping for the exact still == frame comparison
    v0 = v1.copy()
    v0.log2r = v1.log2r + 12.0
    seq = Sequence([Keyframe(v0, 0.0), Keyframe(v1, 1.0)], fps=8)
    rs = RenderSettings(width=96, height=54, ss=1)
    out = tmp_path / "seq.mp4"
    frames = list(render_sequence_iter(engine, seq, str(out), rs, colors, fps=8.0, codec="auto",
                                       quality=20, png_dir=str(tmp_path)))
    done = [f for f in frames if f["phase"] == "frame"]
    assert len(done) == seq.frame_count(8.0) == 9
    assert count_frames(out) == 9
    # frame 0 and the last frame are exactly the keyframes
    fr = FrameRenderer(engine, rs)
    first = np.asarray(Image.open(tmp_path / "frame_000000.png"))
    last = np.asarray(Image.open(tmp_path / "frame_000008.png"))
    assert np.array_equal(first, fr.render(v0, colors))
    assert np.array_equal(last, fr.render(v1, colors))
    # zoom progresses: consecutive frames differ
    mid = np.asarray(Image.open(tmp_path / "frame_000004.png"))
    assert np.abs(mid.astype(int) - first.astype(int)).mean() > 1.0


def test_png_saver_is_lossless_and_background(tmp_path):
    from PIL import Image
    from mandelscope.video import PngSaver
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 256, (37, 53, 3), dtype=np.uint8) for _ in range(5)]
    s = PngSaver(workers=2)
    for i, im in enumerate(imgs):
        s.save(im, tmp_path / f"f{i}.png")
        im[:] = 0                           # the saver keeps its own copy
    s.close()
    rng = np.random.default_rng(3)
    for i in range(5):
        assert np.array_equal(np.asarray(Image.open(tmp_path / f"f{i}.png")),
                              rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))


@pytest.mark.gpu
def test_fast_zoom_movie_matches_direct_frames(engine, locations, tmp_path):
    """Frames cut out of shared key pictures look like frames computed one by one (zoom with
    rotation, 'auto' colours smoothed over time), with far fewer samples."""
    from mandelscope.cli import load_location
    from mandelscope.keymovie import render_sequence_fast_iter
    v1, colors = load_location(str(locations / "spiral_1e-100.json"))
    assert colors.mapping == 3
    v0 = v1.copy()
    v0.log2r = v1.log2r + 3.0
    v0.angle = v1.angle - 1.0
    seq = Sequence([Keyframe(v0, 0.0), Keyframe(v1, 1.5)], fps=12)
    rs = RenderSettings(width=96, height=54, ss=2)
    direct = [f["image"] for f in render_sequence_iter(engine, seq, None, rs, colors, fps=12.0)
              if f["phase"] == "frame"]
    st = {}
    out = tmp_path / "fast.mp4"
    fast = [f["image"] for f in render_sequence_fast_iter(engine, seq, str(out), rs, colors, fps=12.0, stats=st)
            if f["phase"] == "frame"]
    assert len(fast) == len(direct) == seq.frame_count(12.0)
    assert count_frames(out) == len(fast)
    assert st["key_pictures"] >= 2 and st["direct_frames"] == 0         # a switch (cross-fade) happens
    assert st["samples"] < 0.7 * st["direct_samples"]          # (tiny frames: margins cost a lot)
    diffs = [np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(direct, fast)]
    assert max(diffs) < 6.0, diffs
    # as close to a much finer render (4x4 samples per pixel) as the direct 2x2 frames are
    fine = [f["image"] for f in render_sequence_iter(engine, seq, None, RenderSettings(width=96, height=54, ss=4),
                                                     colors, fps=12.0) if f["phase"] == "frame"]
    err = lambda imgs: np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(imgs, fine)])
    assert err(fast) <= 1.3 * err(direct), (err(fast), err(direct))
    # the same through the normal entry point (RenderSettings.fast)
    rs2 = RenderSettings(width=96, height=54, ss=2, fast=True)
    again = [f["image"] for f in render_sequence_iter(engine, seq, None, rs2, colors, fps=12.0)
             if f["phase"] == "frame"]
    assert all(np.array_equal(a, b) for a, b in zip(again, fast))


def test_fast_movie_plan():
    """Key pictures: about one octave of a steady zoom each, one for a still or panning shot,
    none for a zoom too fast to pay off; every frame inside its picture with >= q samples/pixel."""
    from mandelscope.keymovie import plan_key_pictures, frame_xform, footprint
    from mandelscope.mp import View
    from gmpy2 import mpfr
    v1 = View(mpfr("-0.743643887037158704752191506114774"), mpfr("0.131825904205311970493132056385139"),
              -40.0, 0.3, 5000)
    W, H, q = 320, 180, 2

    def plan(octaves, seconds, fps, rot=0.0, pan=0.0):
        v0 = v1.copy()
        v0.log2r = v1.log2r + octaves
        v0.angle = v1.angle - rot
        if pan:
            v0 = v0.offset(pan, 0, H)
        seq = Sequence([Keyframe(v0, 0.0), Keyframe(v1, seconds)], fps=fps)
        views = [seq.view_at(i / fps)[0] for i in range(seq.frame_count(fps))]
        plans = plan_key_pictures(views, W, H, q, direct_samples=W * H * q * q)
        assert plans[0].f0 == 0 and plans[-1].f1 == len(views)
        assert all(a.f1 == b.f0 for a, b in zip(plans, plans[1:]))
        for p in plans:
            if p.direct:
                continue
            for i in range(p.f0, p.f1):
                A, D = frame_xform(p.view, p.H, views[i], H)
                _, inside = footprint(A, D, W, H, p.W, p.H)
                assert inside and q - 1e-6 <= abs(A) <= 2.5 * q, (i, abs(A))
        return plans

    zoom = plan(8, 8, 30)                       # 1 octave/s
    assert 7 <= len(zoom) <= 12 and not any(p.direct for p in zoom)
    assert len(plan(0, 4, 30, rot=2.0)) == 1     # turning on the spot: one picture
    assert len(plan(0, 4, 30, pan=600)) == 1     # a slow pan: one picture
    spiral = plan(6, 6, 30, rot=6.0)
    assert not any(p.direct for p in spiral)
    fast = plan(8, 0.5, 30)                     # 16 octaves/s: 2 frames per octave
    assert all(p.direct for p in fast)


def test_auto_codec_skips_hardware_encoders_for_tiny_frames(monkeypatch):
    """NVENC refuses frames below its minimum size (96x54 on an RTX 5060 Ti): 'auto' takes x264 there."""
    from mandelscope import video
    monkeypatch.setattr(video, "probe_encoder", lambda c: True)
    assert video.resolve_codec("auto", 96, 54) == "libx264"
    assert video.resolve_codec("auto", 1920, 1080) == video.AUTO_ORDER[0]
    assert video.resolve_codec("auto") == video.AUTO_ORDER[0]
    assert video.resolve_codec("hevc_nvenc", 96, 54) == "hevc_nvenc"      # asked for: as given
