"""Analyse a motion study (mandelscope gui --motion-study): smoothness numbers per action.

    python tools/analyze_footage.py footage/20261021_191300 [--out DIR]

Needs numpy, scipy, Pillow, imageio-ffmpeg (matplotlib optional, for the plots).

Per frame (measured in the explorer on the uncompressed picture, see mandelscope/study.py):
  dt          frame interval; 'janky' = longer than 1.5 frames at 60 Hz
  pop         mean change (0-255) after moving the previous frame by the camera motion: what
              changed beyond the motion itself (refinement, sources switching, colours)
  pop_px      share of pixels that changed visibly (> 32/255) beyond the motion
  spikes      frames whose pop_px is far above the frames around them (a visible jump)
  mixed       share of pixels whose source differs from a neighbour's: high for dot grids /
              stripes of mixed sources ('grid'), low for clean pictures
  src         share of pixels from each source (exact, older picture, coarse levels ...)
  zoom rate   octaves per second of the camera; 'rough' = mean speed change per frame
              relative to the peak speed (a sawtooth of wheel notches is rough)
  jit         mean deviation (ms) of the camera time step from the display interval (frame
              pacing: uneven steps on an even display make motion judder)
  iter        largest iteration limit in the segment
Frames that follow a PNG snapshot of the study itself (v3.7 saved them synchronously: 130-190 ms)
are left out of the timing numbers and counted as 'study stalls'; frames that only show the
anti-aliased picture (no source ids written) are left out of 'mixed' and 'sources'.
From the video (compressed, so only relative): grid = spectrum peaks at the lattice periods.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_frames(path: Path, size):
    import imageio_ffmpeg
    w, h = size
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(gen)
    w, h = meta["size"]
    for raw in gen:
        yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def gray(img):
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def warp_prev(prev, xf):
    """Previous frame resampled at the positions the current pixels had there."""
    from scipy.ndimage import map_coordinates
    H, W = prev.shape[:2]
    A = complex(xf[0], xf[1])
    D = complex(xf[2], xf[3])
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = (xs + 0.5 - W / 2) + 1j * (ys + 0.5 - H / 2)
    q = D + A * d
    px = q.real + W / 2 - 0.5
    py = q.imag + H / 2 - 0.5
    valid = (px >= 1) & (py >= 1) & (px <= W - 2) & (py <= H - 2)
    out = np.zeros_like(prev, dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(prev[..., c].astype(np.float32), [py, px], order=1, mode="nearest")
    return out, valid


def grid_score(g):
    """Peak-to-surroundings ratio of the spectrum at the lattice periods 8, 4, 2 px."""
    H, W = g.shape
    hp = g - g.mean()
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    F = np.abs(np.fft.rfft2(hp * win)) ** 2
    scores = []
    for period in (8, 4, 2):
        fx = W // period
        fy = H // period
        for (yy, xx) in ((0, fx), (fy, 0), (fy, fx)):
            if xx >= F.shape[1] or yy >= F.shape[0]:
                continue
            peak = F[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2].max()
            ring = F[max(0, yy - 6):yy + 7, max(0, xx - 6):min(F.shape[1], xx + 7)]
            base = np.median(ring) + 1e-9
            scores.append(peak / base)
    return float(np.max(scores)) if scores else 1.0


def laplacian_energy(g):
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(np.mean(np.abs(lap)))


def analyse(folder: Path, out: Path, every_grid: int = 1, save_worst: int = 4):
    meta = json.loads((folder / "meta.json").read_text())
    rows = [json.loads(l) for l in (folder / "frames.jsonl").read_text().splitlines() if l.strip()]
    names = meta.get("sources")
    out.mkdir(parents=True, exist_ok=True)
    per = []                                  # per-frame metrics
    has_png = any("png" in r for r in rows)
    stall = set()
    for k, r in enumerate(rows):
        if has_png:
            if k > 0 and rows[k - 1].get("png"):
                stall.add(k)
        elif r["dt"] > 0.1 and r["ms_frame"] < 300.0 * r["dt"]:
            stall.add(k)                      # older studies: a long gap the frame itself did not take
    for k, img in enumerate(read_frames(folder / meta.get("video", "study.mp4"), meta["size"])):
        if k >= len(rows):
            break
        r = rows[k]
        g = gray(img)
        m = dict(i=k, scen=r["scen"], act=r["act"], dt=r["dt"], moving=r["moving"],
                 pop=r.get("pop"), pop_px=r.get("pop_px"), mixed=r.get("mixed"), stall=k in stall,
                 anim_dt=r.get("anim_dt"), max_iter=r.get("max_iter"),
                 t_pre=r.get("t_pre"), t_step=r.get("t_step"), t_post=r.get("t_post"))
        if every_grid and k % every_grid == 0:
            m["grid"] = grid_score(g)
        src = r.get("src")
        if r.get("aa") and r["aa"][1]:
            src = None                        # the anti-aliased picture is shown: no source ids
            m["mixed"] = None
        if src and sum(src) > 0:
            tot = float(sum(src))
            m["src"] = {names[j]: src[j] / tot for j in range(1, min(len(src), len(names))) if src[j]}
        m["aa_blit"] = r["aa"][1] if r.get("aa") else False
        v = r["view"]
        m["log2r"], m["angle"] = v
        m["fprog"] = r["field"][0] if r.get("field") else None
        m["kind"], m["branch"] = r.get("kind"), r.get("branch")
        per.append(m)
    # zoom rate
    for a, b in zip(per, per[1:]):
        if b["dt"] > 0:
            b["zoom_rate"] = -(b["log2r"] - a["log2r"]) / b["dt"]
            b["rot_rate"] = math.degrees(math.remainder(b["angle"] - a["angle"], 2 * math.pi)) / b["dt"]
    # visible jumps: pop_px far above the frames around
    for j, m in enumerate(per):
        if m.get("pop_px") is None:
            continue
        around = [x.get("pop_px") for x in per[max(0, j - 5):j] + per[j + 1:j + 6] if x.get("pop_px") is not None]
        base = float(np.median(around)) if around else 0.0
        m["spike"] = m["pop_px"] > max(0.03, 3.0 * base)
    # per segment (place/variant, action): only the frames while moving + 0.5 s after
    segs = defaultdict(list)
    for m in per:
        if m["scen"] and m["act"] not in ("", "idle"):
            segs[(m["scen"], m["act"])].append(m)

    def stats(vals, f=np.mean):
        vals = [v for v in vals if v is not None]
        return float(f(vals)) if vals else float("nan")

    table = []
    for (scen, act), ms in segs.items():
        mov = [m for m in ms if m["moving"]] or ms
        dts = [m["dt"] for m in ms if m["dt"] > 0 and not m["stall"]]
        stalls = sum(1 for m in ms if m["stall"])
        # pacing: the camera's time step against the display interval (median frame interval)
        steps = [m["anim_dt"] if m.get("anim_dt") is not None else m["dt"] for m in mov
                 if m["dt"] > 0 and not m["stall"]]
        med = float(np.median(dts)) if dts else 0.0
        jit = 1000 * float(np.mean([abs(x - med * max(1, round(x / med))) for x in steps])) if steps and med > 0 else float("nan")
        iters = [m["max_iter"] for m in ms if m.get("max_iter")]
        tim = {k3: stats([m.get(k3) for m in mov], np.median) for k3 in ("t_pre", "t_step", "t_post")}
        src = defaultdict(float)
        for m in mov:
            for k2, v in (m.get("src") or {}).items():
                src[k2] += v / len(mov)
        zr = [m.get("zoom_rate", 0.0) for m in mov]
        peak = max((abs(z) for z in zr), default=0.0)
        rough = (float(np.mean(np.abs(np.diff(zr)))) / peak) if peak > 0.05 and len(zr) > 2 else float("nan")
        table.append(dict(
            scen=scen, act=act, frames=len(ms), moving=len(mov),
            dt_p50=stats(dts, np.median) * 1000, dt_p95=stats(dts, lambda x: np.percentile(x, 95)) * 1000,
            dt_max=stats(dts, np.max) * 1000, janky=100.0 * sum(d > 1.5 / 60 for d in dts) / max(len(dts), 1),
            pop=stats([m.get("pop") for m in ms]), pop_px=100 * stats([m.get("pop_px") for m in ms]),
            spikes=sum(1 for m in ms if m.get("spike")),
            mixed=100 * stats([m.get("mixed") for m in mov]), mixed_max=100 * stats([m.get("mixed") for m in mov], np.max),
            grid=stats([m.get("grid") for m in mov]),
            zoom_peak=peak, rough=rough, stalls=stalls, jit=jit, iter=max(iters) if iters else None, **tim,
            src={k2: round(100 * v, 1) for k2, v in sorted(src.items(), key=lambda kv: -kv[1])}))
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, segments=table), indent=1))
    # worst frames (pops and grids), side by side with the warped previous frame
    try:
        _worst_frames(folder, meta, rows, per, out, save_worst)
    except Exception as e:                     # pragma: no cover
        print("worst frames:", e)
    try:
        _plots(per, out)
    except Exception as e:                     # pragma: no cover
        print("plots:", e)
    return table, per


def _worst_frames(folder, meta, rows, per, out, n):
    from PIL import Image
    want = {}
    for key in ("pop_px", "mixed"):
        cand = sorted([m for m in per if m.get(key) is not None and m["moving"]], key=lambda m: -m[key])[:n]
        for m in cand:
            want.setdefault(m["i"], []).append(key)
    if not want:
        return
    prev = None
    for k, img in enumerate(read_frames(folder / meta.get("video", "study.mp4"), meta["size"])):
        if k in want and prev is not None and rows[k].get("xf"):
            w, _ = warp_prev(prev, rows[k]["xf"])
            pair = np.concatenate([np.clip(w, 0, 255).astype(np.uint8), np.full((img.shape[0], 4, 3), 255, np.uint8), img], 1)
            tag = "_".join(want[k])
            scen = rows[k]["scen"].replace("/", "_")
            Image.fromarray(pair).save(out / f"worst_{tag}_{k:05d}_{scen}_{rows[k]['act']}.png")
        prev = img
        if k > max(want):
            break


def _plots(per, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scens = sorted({m["scen"] for m in per if m["scen"]})
    for scen in scens:
        ms = [m for m in per if m["scen"] == scen]
        t = np.arange(len(ms))
        fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
        ax[0].plot(t, [1000 * m["dt"] for m in ms], lw=0.8)
        ax[0].set_ylabel("frame ms")
        ax[1].plot(t, [m.get("zoom_rate", 0) for m in ms], lw=0.8, label="zoom oct/s")
        ax[1].plot(t, [m.get("rot_rate", 0) / 90 for m in ms], lw=0.8, label="rotation (90°/s)")
        ax[1].legend(fontsize=7)
        ax[2].plot(t, [100 * (m.get("pop_px") or 0) for m in ms], lw=0.8)
        ax[2].set_ylabel("pop px %")
        ax[3].plot(t, [100 * (m.get("mixed") or 0) for m in ms], lw=0.8)
        ax[3].set_ylabel("mixed %")
        keys = ["exact", "exact_reprojected", "older_picture", "coarse1", "coarse2", "coarse3", "partial", "nothing"]
        bottom = np.zeros(len(ms))
        for k2 in keys:
            vals = np.array([(m.get("src") or {}).get(k2, 0.0) for m in ms])
            if vals.any():
                ax[4].fill_between(t, bottom, bottom + vals, step="mid", label=k2)
                bottom += vals
        ax[4].legend(fontsize=7, ncol=4)
        ax[4].set_ylabel("sources")
        acts, last = [], None
        for i, m in enumerate(ms):
            if m["act"] != last:
                acts.append((i, m["act"]))
                last = m["act"]
        for a_ in ax:
            for i, name in acts:
                a_.axvline(i, color="k", lw=0.4, alpha=0.4)
        for i, name in acts:
            ax[0].text(i, ax[0].get_ylim()[1] * 0.9, name, fontsize=7, rotation=90, va="top")
        fig.suptitle(scen)
        fig.tight_layout()
        fig.savefig(out / f"plot_{scen.replace('/', '_')}.png", dpi=90)
        plt.close(fig)


def print_table(table):
    hdr = (f"{'place/variant':24s}{'action':11s}{'frames':>7s}{'dt p50':>8s}{'p95':>7s}{'max':>7s}{'jank%':>7s}{'jit':>6s}"
           f"{'pop':>6s}{'pop%':>6s}{'spikes':>7s}{'mixed%':>7s}{'max':>6s}{'zoom':>6s}{'rough':>7s}{'iter':>9s}"
           f"{'pre/step/post ms':>18s}  sources %")
    print(hdr)
    n_stall = 0
    for r in sorted(table, key=lambda r: (r["scen"].split("/")[0], r["act"], r["scen"])):
        n_stall += r["stalls"]
        tim = (f"{r['t_pre']:5.1f}/{r['t_step']:5.1f}/{r['t_post']:4.1f}" if not math.isnan(r.get("t_step", float("nan")))
               else "-")
        print(f"{r['scen']:24s}{r['act']:11s}{r['frames']:7d}{r['dt_p50']:8.1f}{r['dt_p95']:7.1f}{r['dt_max']:7.1f}"
              f"{r['janky']:7.1f}{r['jit']:6.1f}{r['pop']:6.1f}{r['pop_px']:6.1f}{r['spikes']:7d}{r['mixed']:7.1f}{r['mixed_max']:6.1f}"
              f"{r['zoom_peak']:6.2f}{r['rough']:7.2f}{(r['iter'] or 0):9,d}{tim:>18s}  "
              + ", ".join(f"{k} {v}" for k, v in list(r["src"].items())[:4]))
    if n_stall:
        print(f"\n{n_stall} frames after the study's own PNG snapshots left out of the timing numbers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid-every", type=int, default=0, help="spectrum grid score every N frames (0: off)")
    args = ap.parse_args()
    folder = Path(args.folder)
    out = Path(args.out) if args.out else folder / "analysis"
    table, _ = analyse(folder, out, args.grid_every)
    print_table(table)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
