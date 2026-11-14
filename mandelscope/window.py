"""Full screen (F11) through GLFW, which runs the explorer's window.

Like gamepad.py, this loads the GLFW library that imgui_bundle ships and talks to the window
hello_imgui made (the current OpenGL context's window).  Full screen covers the monitor the
window is on (the one it overlaps most) at that monitor's current video mode, so the display
does not switch modes; the windowed position and size come back when leaving it.  Full-screen
windows do not minimise when another window gets the focus (a second monitor stays usable).
"""
from __future__ import annotations

import ctypes

from .gamepad import glfw_library_path

GLFW_AUTO_ICONIFY = 0x00020006


class _VidMode(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("red", ctypes.c_int),
                ("green", ctypes.c_int), ("blue", ctypes.c_int), ("refresh", ctypes.c_int)]


def overlap(a: tuple, b: tuple) -> int:
    """Area shared by two rectangles (x, y, w, h)."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def best_monitor(window_rect: tuple, monitor_rects: list) -> int:
    """Index of the monitor the window overlaps most (the first one when it overlaps none)."""
    best, area = 0, -1
    for i, r in enumerate(monitor_rects):
        a = overlap(window_rect, r)
        if a > area:
            best, area = i, a
    return best


class FullScreen:
    def __init__(self, window_address: int | None = None):
        self.lib = None
        self.address = window_address or None   # hello_imgui.get_glfw_window_address()
        self.error = ""
        self.windowed = None          # (x, y, w, h) to return to
        self._load()

    def _load(self):
        path = glfw_library_path()
        if path is None:
            self.error = "GLFW library not found"
            return
        try:
            lib = ctypes.CDLL(path)
            vp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
            lib.glfwGetCurrentContext.argtypes = []
            lib.glfwGetCurrentContext.restype = vp
            lib.glfwGetWindowMonitor.argtypes = [vp]
            lib.glfwGetWindowMonitor.restype = vp
            lib.glfwGetMonitors.argtypes = [ip]
            lib.glfwGetMonitors.restype = ctypes.POINTER(vp)
            lib.glfwGetPrimaryMonitor.argtypes = []
            lib.glfwGetPrimaryMonitor.restype = vp
            lib.glfwGetMonitorPos.argtypes = [vp, ip, ip]
            lib.glfwGetMonitorPos.restype = None
            lib.glfwGetVideoMode.argtypes = [vp]
            lib.glfwGetVideoMode.restype = ctypes.POINTER(_VidMode)
            lib.glfwGetWindowPos.argtypes = [vp, ip, ip]
            lib.glfwGetWindowPos.restype = None
            lib.glfwGetWindowSize.argtypes = [vp, ip, ip]
            lib.glfwGetWindowSize.restype = None
            lib.glfwSetWindowMonitor.argtypes = [vp, vp] + [ctypes.c_int] * 5
            lib.glfwSetWindowMonitor.restype = None
            lib.glfwSetWindowAttrib.argtypes = [vp, ctypes.c_int, ctypes.c_int]
            lib.glfwSetWindowAttrib.restype = None
            self.lib = lib
        except Exception as e:          # pragma: no cover - depends on the GLFW build
            self.error = f"GLFW: {e}"

    def _window(self):
        if self.lib is None:
            return None
        return self.address or self.lib.glfwGetCurrentContext() or None

    def active(self) -> bool:
        w = self._window()
        return bool(w and self.lib.glfwGetWindowMonitor(w))

    def _window_rect(self, w) -> tuple:
        x, y, cw, ch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self.lib.glfwGetWindowPos(w, ctypes.byref(x), ctypes.byref(y))
        self.lib.glfwGetWindowSize(w, ctypes.byref(cw), ctypes.byref(ch))
        return x.value, y.value, cw.value, ch.value

    def _monitors(self) -> list:
        n = ctypes.c_int()
        arr = self.lib.glfwGetMonitors(ctypes.byref(n))
        out = []
        for i in range(n.value):
            m = arr[i]
            x, y = ctypes.c_int(), ctypes.c_int()
            self.lib.glfwGetMonitorPos(m, ctypes.byref(x), ctypes.byref(y))
            mode = self.lib.glfwGetVideoMode(m)
            if not mode:
                continue
            out.append((m, (x.value, y.value, mode.contents.width, mode.contents.height), mode.contents.refresh))
        return out

    def set(self, on: bool) -> bool:
        """Enter or leave full screen; False when the window could not be reached."""
        w = self._window()
        if not w:
            return False
        if on == self.active():
            return True
        if on:
            mons = self._monitors()
            if not mons:
                return False
            rect = self._window_rect(w)
            self.windowed = rect
            m, (mx, my, mw, mh), hz = mons[best_monitor(rect, [r for _, r, _ in mons])]
            self.lib.glfwSetWindowAttrib(w, GLFW_AUTO_ICONIFY, 0)
            self.lib.glfwSetWindowMonitor(w, m, mx, my, mw, mh, hz)
        else:
            x, y, cw, ch = self.windowed or (100, 100, 1600, 900)
            self.lib.glfwSetWindowMonitor(w, None, x, y, cw, ch, 0)
        return True

    def toggle(self) -> bool:
        return self.set(not self.active())
