"""Windowed entry point (no console window).

Used by the Desktop / Start-menu shortcut (``pythonw -m mandelscope.launcher``) and by
the stand-alone ``Mandelscope.exe``.  Output goes to a log file instead of a console,
and a fatal error is shown in a message box instead of disappearing silently.
"""
from __future__ import annotations

import faulthandler
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "mandelscope"


def _message_box(title: str, text: str):
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)      # MB_ICONERROR
            return
        except Exception:
            pass
    try:
        print(f"{title}: {text}", file=sys.__stderr__ or sys.stderr)
    except Exception:
        pass


def _redirect_output() -> Path | None:
    """Without a console sys.stdout/stderr are None: send them to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    d = log_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        path = d / "mandelscope.log"
        if path.exists() and path.stat().st_size > 2_000_000:
            path.replace(d / "mandelscope.old.log")
        f = open(path, "a", encoding="utf-8", buffering=1)
        f.write(f"\n===== Mandelscope started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = f
        sys.stderr = f
        try:
            faulthandler.enable(f)
        except Exception:
            pass
        return path
    except Exception:
        return None


def _args(argv: list[str]) -> list[str]:
    """``Mandelscope.exe some_location.json`` (e.g. a file dropped on the icon) opens it."""
    if not argv:
        return ["gui"]
    first = argv[0]
    if not first.startswith("-") and first not in ("gui", "render", "video", "info", "bench", "get-mesa") \
            and Path(first).suffix.lower() in (".json", ".png"):
        return ["gui", first] + argv[1:]
    return argv


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    log = _redirect_output()
    try:
        from .cli import main as cli_main
        return int(cli_main(_args(list(sys.argv[1:] if argv is None else argv))) or 0)
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) else 0
    except BaseException as e:            # noqa: BLE001 - last line of defence
        tb = traceback.format_exc()
        try:
            print(tb, file=sys.stderr, flush=True)
        except Exception:
            pass
        hint = ""
        low = (str(e) + tb).lower()
        if "opengl" in low or "glcontext" in low or "require=430" in low or "context" in low:
            hint = ("\n\nMandelscope needs OpenGL 4.3. Update the graphics driver "
                    "(NVIDIA: GeForce Experience / nvidia.com/drivers), or start it in CPU mode: "
                    "Mandelscope.exe --software (run.bat --software from the source folder).")
        _message_box("Mandelscope could not start",
                     f"{type(e).__name__}: {e}{hint}" + (f"\n\nDetails are in\n{log}" if log else ""))
        return 1


if __name__ == "__main__":
    sys.exit(main())
