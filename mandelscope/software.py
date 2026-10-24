"""CPU mode: the same engine and interface on Mesa's llvmpipe, a software OpenGL 4.5 driver.

llvmpipe compiles the compute shaders for the processor (SIMD over 8 or 16 pixels, all cores),
so every feature works and the pictures are the same as on a graphics card, only slower.

* Linux: Mesa is part of the system; `enable()` only selects llvmpipe.
* Windows: Mesa is not part of the system.  The release zip carries it in `software\\`
  (built by github.com/mmozeiko/build-mesa, MIT licence).  From a source checkout,
  `mandelscope get-mesa` (or the offer at start-up) downloads the same build.  `enable()` loads
  its opengl32.dll by full path *before* anything else loads OpenGL; GLFW, Dear ImGui and
  moderngl then get that module when they ask Windows for "opengl32.dll".

`enable()` must run before imgui_bundle is imported (its native module links OpenGL).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = "MANDELSCOPE_SOFTWARE"          # "1": CPU mode, "0": graphics card (no automatic fallback)
MESA_REPO = "mmozeiko/build-mesa"
MESA_PAGE = f"https://github.com/{MESA_REPO}/releases"
ASSET_RE = re.compile(r"llvmpipe.*x64.*\.(7z|zip)$", re.IGNORECASE)
PRELOAD_FIRST = ("libglapi.dll", "libgallium_wgl.dll")

_active = False
_mesa_dir: Path | None = None
_orig_env: dict | None = None
_handles: list = []                   # keep the preloaded DLLs (and the DLL directory) alive


class MesaMissing(RuntimeError):
    pass


# ----------------------------------------------------------------------------- state
def active() -> bool:
    """CPU mode was switched on in this process."""
    return _active


def mesa_dir() -> Path | None:
    return _mesa_dir


def threads() -> int:
    try:
        n = int(os.environ.get("LP_NUM_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else (os.cpu_count() or 1)


def is_software_renderer(renderer: str) -> bool:
    r = (renderer or "").lower()
    return any(k in r for k in ("llvmpipe", "softpipe", "swrast", "lavapipe", "gdi generic",
                                "basic render"))


def take_flags(argv: list[str]) -> list[str]:
    """Remove --software / --cpu / --gpu from a command line and set the mode for this process
    and the processes it starts."""
    out = []
    for a in argv:
        if a in ("--software", "--cpu"):
            os.environ[ENV] = "1"
        elif a == "--gpu":
            os.environ[ENV] = "0"
        else:
            out.append(a)
    return out


def requested() -> bool:
    return os.environ.get(ENV) == "1"


# ----------------------------------------------------------------------------- where Mesa is
def user_mesa_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "mandelscope" / "mesa"


def candidate_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("MANDELSCOPE_MESA_DIR"):
        dirs.append(Path(os.environ["MANDELSCOPE_MESA_DIR"]))
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent / "software")
    dirs.append(Path(__file__).resolve().parent.parent / "software")      # source checkout
    dirs.append(user_mesa_dir())
    return dirs


def find_mesa() -> Path | None:
    for d in candidate_dirs():
        if (d / "opengl32.dll").is_file():
            return d
    return None


# ----------------------------------------------------------------------------- switching on
def enable() -> str:
    """Switch this process to llvmpipe.  Returns a short description; raises MesaMissing on
    Windows when no Mesa build is found."""
    global _active, _mesa_dir, _orig_env
    if _active:
        return describe()
    if _orig_env is None:
        _orig_env = dict(os.environ)
    os.environ[ENV] = "1"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    if sys.platform == "win32":
        d = find_mesa()
        if d is None:
            raise MesaMissing("Mesa (opengl32.dll) not found in: " + ", ".join(str(p) for p in candidate_dirs()))
        _preload_windows(d)
        _mesa_dir = d
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"      # glvnd: Mesa even next to an NVIDIA driver
        for p in ("/usr/share/glvnd/egl_vendor.d/50_mesa.json", "/etc/glvnd/egl_vendor.d/50_mesa.json"):
            if os.path.exists(p):
                os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = p
                break
    _active = True
    return describe()


def describe() -> str:
    where = f" from {_mesa_dir}" if _mesa_dir else ""
    return f"CPU mode: Mesa llvmpipe, {threads()} threads{where}"


def _preload_windows(d: Path):
    import ctypes
    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    try:
        _handles.append(os.add_dll_directory(str(d)))
    except (AttributeError, OSError):
        pass
    names = [n for n in PRELOAD_FIRST if (d / n).is_file()] + ["opengl32.dll"]
    for n in names:
        try:
            # full path: Windows also searches this folder for the DLL's own dependencies
            _handles.append(ctypes.WinDLL(str(d / n)))
        except OSError as e:
            raise MesaMissing(f"cannot load {d / n}: {e}") from e


# ----------------------------------------------------------------------------- restarting
def child_env(software: bool) -> dict:
    env = dict(_orig_env if _orig_env is not None else os.environ)
    env[ENV] = "1" if software else "0"
    return env


def restart_command(args: list[str]) -> list[str]:
    """The command that starts this program again with `args` (frozen exe, launcher or CLI)."""
    if getattr(sys, "frozen", False):
        return [sys.executable] + args
    exe = sys.executable
    windowed = Path(sys.argv[0]).stem == "launcher"
    if windowed and sys.platform == "win32" and exe.lower().endswith("python.exe"):
        w = exe[:-10] + "pythonw.exe"
        exe = w if os.path.exists(w) else exe
    return [exe, "-m", "mandelscope.launcher" if windowed else "mandelscope"] + args


def restart(software: bool, args: list[str] | None = None) -> subprocess.Popen:
    args = take_flags(list(sys.argv[1:] if args is None else args))
    os.environ.pop(ENV, None)
    cmd = restart_command(args)
    return subprocess.Popen(cmd, env=child_env(software), cwd=os.getcwd(), close_fds=True)


# ----------------------------------------------------------------------------- graphics check
def gpu_check() -> tuple[bool, str]:
    """Can this process get an OpenGL 4.3 core context from the installed driver?  (Creates a
    hidden one.  After a failure the process has the system OpenGL loaded, so CPU mode needs a
    new process: see `restart`.)"""
    try:
        from .cli import make_headless_context
        ctx = make_headless_context()
        name = ctx.info.get("GL_RENDERER", "?")
        ctx.release()
        if is_software_renderer(name) and "llvmpipe" not in name.lower():
            return False, f"only {name}"
        return True, name
    except Exception as e:          # noqa: BLE001 - any failure means: no usable driver
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        return False, (lines[-1] if lines else type(e).__name__)[:200]


# ----------------------------------------------------------------------------- download (Windows)
def _get(url: str, accept: str = "application/octet-stream"):
    from urllib.request import Request, urlopen
    headers = {"User-Agent": "mandelscope", "Accept": accept}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {tok}"
    return urlopen(Request(url, headers=headers), timeout=60)


def pick_asset(release: dict) -> dict:
    assets = [a for a in release.get("assets", []) if ASSET_RE.search(a.get("name", ""))]
    if not assets:
        raise RuntimeError(f"no llvmpipe x64 archive in release {release.get('tag_name')} of {MESA_REPO}")
    # prefer .zip (needs no extractor), then the shortest name (no debug / symbols packages)
    assets.sort(key=lambda a: (not a["name"].lower().endswith(".zip"), "debug" in a["name"].lower(), len(a["name"])))
    return assets[0]


def extract(archive: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        return
    tried = []
    tools = [["tar", "-xf", str(archive), "-C", str(out)]]          # Windows 10 1803+: bsdtar
    for p in (shutil.which("7z"), shutil.which("7za"),
              os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "7-Zip", "7z.exe"),
              os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "7-Zip", "7z.exe")):
        if p and os.path.exists(p):
            tools.append([p, "x", "-y", f"-o{out}", str(archive)])
    for cmd in tools:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and any(out.rglob("opengl32.dll")):
                return
            tried.append(f"{Path(cmd[0]).name}: {(r.stderr or r.stdout).strip()[:200]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            tried.append(f"{Path(cmd[0]).name}: {e}")
    raise RuntimeError("could not unpack the .7z archive (install 7-Zip: winget install 7zip.7zip).\n"
                       + "\n".join(tried))


def install_from_archive(archive: Path, dest: Path, source: str = "") -> Path:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        extract(archive, tmp)
        hits = sorted(tmp.rglob("opengl32.dll"), key=lambda p: ("x64" not in str(p).lower(), len(p.parts)))
        if not hits:
            raise RuntimeError("the archive has no opengl32.dll")
        src = hits[0].parent
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.dll"):
            old.unlink()
        for f in src.iterdir():
            if f.is_file() and (f.suffix.lower() == ".dll" or "license" in f.name.lower()
                                or "copying" in f.name.lower()):
                shutil.copy2(f, dest / f.name)
    (dest / "SOURCE.txt").write_text(
        "Mesa llvmpipe (software OpenGL) for Mandelscope's CPU mode.\n"
        f"From {source or archive.name} ({MESA_PAGE}).\n"
        "Mesa: MIT licence (https://docs.mesa3d.org/license.html); it contains LLVM:\n"
        "Apache-2.0 WITH LLVM-exception (https://llvm.org/LICENSE.txt).\n", encoding="utf-8")
    return dest


def download_mesa(dest: Path | None = None, log=print) -> Path:
    """Download the latest llvmpipe build from github.com/mmozeiko/build-mesa into `dest`
    (default: %LOCALAPPDATA%\\mandelscope\\mesa)."""
    dest = Path(dest) if dest else user_mesa_dir()
    with _get(f"https://api.github.com/repos/{MESA_REPO}/releases/latest", "application/vnd.github+json") as r:
        rel = json.load(r)
    a = pick_asset(rel)
    log(f"downloading {a['name']} ({a.get('size', 0) / 1e6:.0f} MB) from {MESA_PAGE}/tag/{rel.get('tag_name')}")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / a["name"]
        h = hashlib.sha256()
        with _get(a["browser_download_url"]) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                h.update(b)
                f.write(b)
        want = (a.get("digest") or "").lower()
        if want.startswith("sha256:") and want[7:] != h.hexdigest():
            raise RuntimeError(f"checksum mismatch for {a['name']}")
        install_from_archive(path, dest, f"{a['name']}, release {rel.get('tag_name')}")
    log(f"installed into {dest}")
    return dest


# ----------------------------------------------------------------------------- start-up (Windows)
def _message(text: str, title: str = "Mandelscope", yes_no: bool = False) -> bool:
    if sys.platform == "win32":
        try:
            import ctypes
            flags = 0x24 if yes_no else 0x40            # MB_YESNO|MB_ICONQUESTION / MB_ICONINFORMATION
            return ctypes.windll.user32.MessageBoxW(None, text, title, flags) == 6     # IDYES
        except Exception:
            pass
    print(f"{title}: {text}", file=sys.__stderr__ or sys.stderr)
    return False


def ensure_mesa_interactive(reason: str) -> bool:
    """Windows: Mesa is needed but missing - offer to download it.  True when installed."""
    if find_mesa() is not None:
        return True
    if not _message(f"{reason}\n\nMandelscope can run on the processor instead, with Mesa's software "
                    f"renderer (slower). Download it now? (some tens of MB from\n{MESA_PAGE})",
                    yes_no=True):
        return False
    try:
        download_mesa()
        return True
    except Exception as e:          # noqa: BLE001
        _message(f"The download failed:\n{e}\n\nYou can download a 'mesa-llvmpipe-x64' archive from "
                 f"{MESA_PAGE} and put its DLL files into\n{user_mesa_dir()}")
        return False


def fallback_if_needed() -> bool:
    """Before the window opens (Windows): without an OpenGL 4.3 driver, start again in CPU mode.
    True when a CPU-mode process was started (this one should exit)."""
    if sys.platform != "win32" or active() or os.environ.get(ENV) == "0":
        return False
    ok, why = gpu_check()
    if ok:
        return False
    if not ensure_mesa_interactive(f"This computer's graphics driver does not offer OpenGL 4.3 ({why})."):
        return False
    restart(True)
    return True
