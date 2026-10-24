"""Read the OpenGL driver's debug messages (KHR_debug message log), for diagnostics.

NVIDIA reports when it moves a buffer between video and system memory ("Buffer performance
warning: Buffer object 12 (...) is being copied/moved from VIDEO memory to HOST memory").
The bench uses this to show which of our buffers the driver moved.

Uses glGetDebugMessageLog through ctypes (no callback, so no calls from driver threads).
Everything fails soft: if a function cannot be found the log simply stays empty."""
from __future__ import annotations

import ctypes
import re
import sys

GL_DEBUG_OUTPUT = 0x92E0
GL_DONT_CARE = 0x1100
GL_MAX_DEBUG_MESSAGE_LENGTH = 0x9143
GL_DEBUG_TYPE_PERFORMANCE = 0x8250
GL_DEBUG_SEVERITY_LOW = 0x9148
TYPES = {0x824C: "error", 0x824D: "deprecated", 0x824E: "undefined", 0x824F: "portability",
         0x8250: "performance", 0x8251: "other", 0x8268: "marker"}

_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


def _get_proc(name: str):
    names = name.encode()
    if sys.platform == "win32":
        gl = ctypes.WinDLL("opengl32")
        gpa = gl.wglGetProcAddress
        gpa.restype = ctypes.c_void_p
        gpa.argtypes = [ctypes.c_char_p]
        addr = gpa(names)
        if addr and addr not in (1, 2, 3, ctypes.c_void_p(-1).value):
            return addr
        try:
            return ctypes.cast(getattr(gl, name), ctypes.c_void_p).value
        except AttributeError:
            return None
    for lib, fn in (("libEGL.so.1", "eglGetProcAddress"), ("libGL.so.1", "glXGetProcAddressARB"),
                    ("libGL.so.1", "glXGetProcAddress")):
        try:
            dll = ctypes.CDLL(lib)
            gpa = getattr(dll, fn)
        except (OSError, AttributeError):
            continue
        gpa.restype = ctypes.c_void_p
        gpa.argtypes = [ctypes.c_char_p]
        addr = gpa(names)
        if addr:
            return addr
    return None


class DebugLog:
    """Turn on the driver's debug output for the current context and collect its messages."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.ok = False
        self.messages: list[tuple[str, str]] = []      # (type, text)
        try:
            a_log = _get_proc("glGetDebugMessageLog")
            a_ctl = _get_proc("glDebugMessageControl")
            if not a_log:
                return
            u, i, p = ctypes.c_uint, ctypes.c_int, ctypes.c_void_p
            self._log = _FUNCTYPE(u, u, i, p, p, p, p, p, p)(a_log)
            ctx.enable_direct(GL_DEBUG_OUTPUT)
            if a_ctl:
                ctl = _FUNCTYPE(None, u, u, u, i, p, ctypes.c_ubyte)(a_ctl)
                ctl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, None, 1)    # everything, also 'low'
            self.drain()
            self.messages.clear()
            self.ok = True
        except Exception:                 # pragma: no cover - driver/platform dependent
            self.ok = False

    def drain(self) -> int:
        """Move the driver's queued messages into self.messages; returns how many."""
        if not getattr(self, "_log", None):
            return 0
        got = 0
        n, size = 64, 64 * 1024
        sources = (ctypes.c_uint * n)()
        types = (ctypes.c_uint * n)()
        ids = (ctypes.c_uint * n)()
        sev = (ctypes.c_uint * n)()
        lens = (ctypes.c_int * n)()
        text = ctypes.create_string_buffer(size)
        for _ in range(100):
            k = self._log(n, size, sources, types, ids, sev, lens, text)
            if not k:
                break
            pos = 0
            for j in range(k):
                ln = lens[j]
                msg = text.raw[pos:pos + ln].split(b"\0", 1)[0].decode("utf-8", "replace").strip()
                pos += ln
                self.messages.append((TYPES.get(types[j], hex(types[j])), msg))
            got += k
        return got

    def close(self):
        try:
            self.ctx.disable_direct(GL_DEBUG_OUTPUT)
        except Exception:                 # pragma: no cover
            pass


MOVE_RE = re.compile(r"[Bb]uffer object (\d+).*?(?:moved|copied/moved)\s+from\s+(\w+)\s+memory\s+to\s+(\w+)",
                     re.S)


def buffer_moves(messages, names: dict) -> list[tuple[str, str, str, int]]:
    """(what, from, to, count) for 'buffer moved between memories' driver messages."""
    out: dict = {}
    for _, msg in messages:
        m = MOVE_RE.search(msg)
        if not m:
            continue
        what = names.get(int(m.group(1)), f"buffer {m.group(1)}")
        key = (what, m.group(2).lower(), m.group(3).lower())
        out[key] = out.get(key, 0) + 1
    return [(w, a, b, n) for (w, a, b), n in out.items()]
