"""Game controllers (PlayStation, Xbox, Switch Pro, ...) through GLFW's gamepad API.

GLFW runs the window already; imgui_bundle ships it as a shared library (glfw3.dll /
libglfw.so.3), and its import points PYGLFW_LIBRARY at that file.  Loading the same file with
ctypes gives the *same* GLFW instance, so no extra package and no second copy of GLFW is
needed.  GLFW maps PlayStation 4 / 5 and other pads to the standard layout with its built-in
copy of the SDL controller database (XInput pads on Windows need no mapping).  For a pad it
does not know, a `gamecontrollerdb.txt` from github.com/mdqinc/SDL_GameControllerDB in the
settings folder (%APPDATA%\\mandelscope) adds the newer mappings.

`Gamepad.poll()` returns a `control.PadState` (or None without a controller); the mapping to
camera motion lives in `control.pad_controls`, which the tests drive with made-up states.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

from .control import PadState

JOYSTICK_LAST = 15


class _GamepadState(ctypes.Structure):
    _fields_ = [("buttons", ctypes.c_ubyte * 15), ("axes", ctypes.c_float * 6)]


def glfw_library_path() -> str | None:
    p = os.environ.get("PYGLFW_LIBRARY")
    if p and os.path.exists(p):
        return p
    try:
        import imgui_bundle
        d = Path(imgui_bundle.__file__).parent
    except Exception:
        return None
    names = {"win32": ["glfw3.dll"], "darwin": ["libglfw.3.dylib"]}.get(sys.platform, ["libglfw.so.3", "libglfw.so"])
    for base in (d, Path(sys.executable).parent):
        for n in names:
            if (base / n).exists():
                return str(base / n)
    return None


class Gamepad:
    def __init__(self, lib=None, mappings: Path | None = None):
        self.lib = lib
        self.error = ""
        self.jid = -1
        self.name = ""
        self.next_scan = 0.0
        self.fake: PadState | None = None       # the scripted test drives the app with these
        self._state = _GamepadState()
        self.mappings_loaded = False
        if self.lib is None:
            self._load()
        if self.lib and mappings is not None and Path(mappings).is_file():
            self.load_mappings(Path(mappings))

    def load_mappings(self, path: Path) -> bool:
        """Add SDL-format mappings (gamecontrollerdb.txt) to GLFW's built-in ones."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            self.mappings_loaded = bool(self.lib.glfwUpdateGamepadMappings(text.encode("utf-8")))
        except Exception as e:
            self.error = f"cannot read {path.name}: {e}"
            self.mappings_loaded = False
        return self.mappings_loaded

    def _load(self):
        path = glfw_library_path()
        if path is None:
            self.error = "GLFW library not found"
            return
        try:
            lib = ctypes.CDLL(path)
            lib.glfwJoystickIsGamepad.argtypes = [ctypes.c_int]
            lib.glfwJoystickIsGamepad.restype = ctypes.c_int
            lib.glfwJoystickPresent.argtypes = [ctypes.c_int]
            lib.glfwJoystickPresent.restype = ctypes.c_int
            lib.glfwGetGamepadState.argtypes = [ctypes.c_int, ctypes.POINTER(_GamepadState)]
            lib.glfwGetGamepadState.restype = ctypes.c_int
            lib.glfwGetGamepadName.argtypes = [ctypes.c_int]
            lib.glfwGetGamepadName.restype = ctypes.c_char_p
            lib.glfwGetJoystickName.argtypes = [ctypes.c_int]
            lib.glfwGetJoystickName.restype = ctypes.c_char_p
            lib.glfwGetCurrentContext.argtypes = []
            lib.glfwGetCurrentContext.restype = ctypes.c_void_p
            lib.glfwUpdateGamepadMappings.argtypes = [ctypes.c_char_p]
            lib.glfwUpdateGamepadMappings.restype = ctypes.c_int
            self.lib = lib
        except (OSError, AttributeError) as e:
            self.error = f"cannot use {path}: {e}"
            self.lib = None

    def same_instance_as(self, window_address: int) -> bool:
        """True when this GLFW is the one that opened the window (the joystick functions of a
        second, uninitialised copy would always report 'nothing connected')."""
        if self.lib is None:
            return False
        try:
            return int(self.lib.glfwGetCurrentContext() or 0) == int(window_address)
        except Exception:
            return False

    def _scan(self):
        self.jid, self.name = -1, ""
        if not self.lib:
            return
        self.error = ""
        for j in range(JOYSTICK_LAST + 1):
            if self.lib.glfwJoystickIsGamepad(j):
                self.jid = j
                raw = self.lib.glfwGetGamepadName(j) or self.lib.glfwGetJoystickName(j) or b"controller"
                self.name = raw.decode("utf-8", "replace")
                return
        # a joystick GLFW has no mapping for: say so once (it is not used)
        for j in range(JOYSTICK_LAST + 1):
            if self.lib.glfwJoystickPresent(j):
                raw = self.lib.glfwGetJoystickName(j) or b"?"
                self.error = f"controller '{raw.decode('utf-8', 'replace')}' is not known to GLFW"
                return

    def poll(self, now: float | None = None) -> PadState | None:
        if self.fake is not None:
            return self.fake
        if self.lib is None:
            return None
        now = time.perf_counter() if now is None else now
        if self.jid < 0 and now >= self.next_scan:
            self.next_scan = now + 1.0              # connecting a pad: noticed within a second
            self._scan()
        if self.jid < 0:
            return None
        if not self.lib.glfwGetGamepadState(self.jid, ctypes.byref(self._state)):
            self.jid, self.name = -1, ""            # unplugged
            return None
        st = self._state
        return PadState(axes=tuple(float(st.axes[i]) for i in range(6)),
                        buttons=tuple(bool(st.buttons[i]) for i in range(15)))
