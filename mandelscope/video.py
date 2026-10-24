"""ffmpeg based video encoding (NVENC when available, CPU encoders otherwise)."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from functools import lru_cache

import numpy as np

CODECS = [
    ("auto", "Auto (best available)"),
    ("hevc_nvenc", "HEVC / H.265 - NVIDIA NVENC"),
    ("av1_nvenc", "AV1 - NVIDIA NVENC (RTX 40/50)"),
    ("h264_nvenc", "H.264 - NVIDIA NVENC"),
    ("libx264", "H.264 - x264 (CPU)"),
    ("libx265", "HEVC - x265 (CPU)"),
    ("hevc_amf", "HEVC - AMD AMF"),
    ("hevc_qsv", "HEVC - Intel QSV"),
]
AUTO_ORDER = ["hevc_nvenc", "h264_nvenc", "av1_nvenc", "hevc_amf", "hevc_qsv", "libx264", "libx265"]


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _creationflags():
    return 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW


@lru_cache(maxsize=None)
def probe_encoder(codec: str) -> bool:
    """True if the encoder actually works on this machine (driver present...)."""
    ff = find_ffmpeg()
    if not ff:
        return False
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           "color=c=black:s=256x256:r=30:d=0.2", "-frames:v", "3", "-c:v", codec, "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_creationflags())
        return r.returncode == 0
    except Exception:
        return False


def available_codecs() -> list[tuple[str, str]]:
    out = [CODECS[0]]
    for c, label in CODECS[1:]:
        if probe_encoder(c):
            out.append((c, label))
    return out


# smallest frame the hardware encoders are used for by "auto" (NVENC refuses tiny frames:
# "Frame dimensions are less than the minimum supported value" at 96x54 on an RTX 5060 Ti)
HW_MIN_SIZE = (256, 144)
SOFTWARE_CODECS = ("libx264", "libx265")


def resolve_codec(codec: str, width: int | None = None, height: int | None = None) -> str:
    if codec != "auto":
        return codec
    small = width is not None and height is not None and (width < HW_MIN_SIZE[0] or height < HW_MIN_SIZE[1])
    for c in AUTO_ORDER:
        if small and c not in SOFTWARE_CODECS:
            continue
        if probe_encoder(c):
            return c
    return "libx264"


def codec_args(codec: str, quality: int) -> list[str]:
    q = str(int(quality))
    if codec.endswith("_nvenc"):
        args = ["-c:v", codec, "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", q, "-b:v", "0"]
        if codec == "hevc_nvenc":
            args += ["-tag:v", "hvc1"]
        return args
    if codec == "libx264":
        return ["-c:v", "libx264", "-preset", "slow", "-crf", q]
    if codec == "libx265":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", q, "-tag:v", "hvc1"]
    if codec == "hevc_amf":
        return ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", q, "-qp_p", q]
    if codec == "hevc_qsv":
        return ["-c:v", "hevc_qsv", "-global_quality", q]
    return ["-c:v", codec]


class PngSaver:
    """Saves PNG frames on background threads (zlib releases the GIL), so a frame sequence
    export does not wait ~0.3-0.6 s per 1080p frame for compression.

    PNG is lossless at every compression level; on a rendered 1080p frame level 3 took 330 ms
    instead of 700 ms (Pillow's default 6) for a 14 % larger file."""

    def __init__(self, workers: int | None = None, compress_level: int = 3, max_pending: int | None = None):
        from concurrent.futures import ThreadPoolExecutor
        n = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
        self.pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="png")
        self.level = int(compress_level)
        self.max_pending = max_pending or 2 * n
        self.pending: list = []
        self.saved = 0

    @staticmethod
    def _save(img: np.ndarray, path: str, level: int):
        from PIL import Image
        Image.fromarray(img).save(path, compress_level=level)

    def _reap(self, block_until: int):
        while self.pending and (len(self.pending) > block_until or self.pending[0].done()):
            f = self.pending.pop(0)
            f.result()                    # re-raises a failed save here
            self.saved += 1

    def save(self, img: np.ndarray, path: str):
        self._reap(self.max_pending - 1)
        self.pending.append(self.pool.submit(self._save, np.array(img, copy=True), str(path), self.level))

    def close(self):
        try:
            self._reap(0)
        finally:
            self.pool.shutdown(wait=True)


class FfmpegWriter:
    """Streams RGB frames into ffmpeg on a background thread."""

    def __init__(self, path: str, width: int, height: int, fps: float, codec: str = "auto",
                 quality: int = 18, max_queue: int = 8):
        ff = find_ffmpeg()
        if not ff:
            raise RuntimeError("ffmpeg not found (install imageio-ffmpeg or put ffmpeg on PATH)")
        if width % 2 or height % 2:
            raise ValueError("video width and height must be even")
        self.codec = resolve_codec(codec, width, height)
        self.path = path
        self.width, self.height = width, height
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}",
               "-i", "-", *codec_args(self.codec, quality), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", path]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_creationflags())
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.frames = 0
        self.error = None
        self._stderr = b""
        self._err_thread = threading.Thread(target=self._read_err, daemon=True)
        self._err_thread.start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _read_err(self):
        try:
            self._stderr = self.proc.stderr.read()
        except Exception:
            pass

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self.proc.stdin.write(item)
            except Exception as e:      # ffmpeg died
                self.error = str(e)
                break

    def write(self, frame: np.ndarray, repeat: int = 1):
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame shape {frame.shape} != {(self.height, self.width, 3)}")
        data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        for _ in range(max(1, repeat)):
            self.q.put(data)
            self.frames += 1

    def close(self) -> tuple[int, str]:
        self.q.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        self._err_thread.join(timeout=5)
        return rc, self._stderr.decode("utf-8", "replace")[-2000:]
