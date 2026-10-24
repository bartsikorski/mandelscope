"""CPU mode (Mesa llvmpipe): flags, finding / installing Mesa, the Windows start-up path.

The Windows parts run here with the platform and the DLL loader replaced by fakes; the real
thing runs in the Windows job of the CI."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from mandelscope import software

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(software, "_active", False)
    monkeypatch.setattr(software, "_mesa_dir", None)
    monkeypatch.setattr(software, "_orig_env", None)
    monkeypatch.setattr(software, "_handles", [])
    monkeypatch.delenv(software.ENV, raising=False)
    monkeypatch.delenv("MANDELSCOPE_MESA_DIR", raising=False)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))      # enable() prepends Mesa's folder
    for k in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE", "__GLX_VENDOR_LIBRARY_NAME", "__EGL_VENDOR_LIBRARY_FILENAMES"):
        monkeypatch.delenv(k, raising=False)
    yield


def fake_mesa(d: Path, extra=("libgallium_wgl.dll",)) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in ("opengl32.dll",) + tuple(extra):
        (d / n).write_bytes(b"MZ")
    return d


def test_flags_are_taken_out_of_the_command_line(monkeypatch):
    assert software.take_flags(["gui", "--software", "x.json"]) == ["gui", "x.json"]
    assert software.requested()
    assert software.take_flags(["--gpu", "bench"]) == ["bench"]
    assert not software.requested()
    assert software.take_flags(["render", "--cpu"]) == ["render"] and software.requested()


def test_find_mesa_search_order(tmp_path, monkeypatch):
    user = tmp_path / "user"
    monkeypatch.setattr(software, "user_mesa_dir", lambda: user)
    assert software.find_mesa() is None or software.find_mesa() == ROOT / "software"
    fake_mesa(user)
    if not (ROOT / "software" / "opengl32.dll").exists():
        assert software.find_mesa() == user
    own = fake_mesa(tmp_path / "own")
    monkeypatch.setenv("MANDELSCOPE_MESA_DIR", str(own))
    assert software.find_mesa() == own                       # the environment variable wins
    monkeypatch.delenv("MANDELSCOPE_MESA_DIR")
    exe = tmp_path / "app" / "Mandelscope.exe"
    fake_mesa(exe.parent / "software")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert software.find_mesa() == exe.parent / "software"   # next to the stand-alone exe


def test_linux_enable_selects_llvmpipe(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    info = software.enable()
    assert software.active() and "llvmpipe" in info
    import os
    assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1" and os.environ["GALLIUM_DRIVER"] == "llvmpipe"
    assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "mesa"
    assert software.child_env(False)[software.ENV] == "0"
    assert "LIBGL_ALWAYS_SOFTWARE" not in software.child_env(False)   # a GPU restart gets the old environment


def test_windows_enable_preloads_mesa_first(tmp_path, monkeypatch):
    import ctypes
    import os
    d = fake_mesa(tmp_path / "mesa", extra=("libgallium_wgl.dll", "libglapi.dll", "vulkan_lvp.dll"))
    monkeypatch.setenv("MANDELSCOPE_MESA_DIR", str(d))
    monkeypatch.setattr(sys, "platform", "win32")
    loaded = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda p: loaded.append(Path(p).name) or p, raising=False)
    dirs = []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: dirs.append(p) or p, raising=False)
    software.enable()
    # dependencies first, opengl32.dll last, by full path; unrelated DLLs are not touched
    assert loaded == ["libglapi.dll", "libgallium_wgl.dll", "opengl32.dll"]
    assert dirs == [str(d)] and os.environ["PATH"].split(os.pathsep)[0] == str(d)
    assert software.mesa_dir() == d and os.environ["GALLIUM_DRIVER"] == "llvmpipe"


def test_windows_enable_without_mesa(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(software, "candidate_dirs", lambda: [tmp_path / "nothing"])
    with pytest.raises(software.MesaMissing):
        software.enable()
    assert not software.active()


def test_windows_load_error_is_reported(tmp_path, monkeypatch):
    import ctypes
    d = fake_mesa(tmp_path / "mesa")
    monkeypatch.setenv("MANDELSCOPE_MESA_DIR", str(d))
    monkeypatch.setattr(sys, "platform", "win32")

    def fail(p):
        raise OSError("[WinError 193] not a valid Win32 application")
    monkeypatch.setattr(ctypes, "WinDLL", fail, raising=False)
    with pytest.raises(software.MesaMissing, match="cannot load"):
        software.enable()


RELEASE = {"tag_name": "25.3.0", "assets": [
    {"name": "mesa-d3d12-x64-25.3.0.7z", "browser_download_url": "u1", "size": 1},
    {"name": "mesa-llvmpipe-arm64-25.3.0.7z", "browser_download_url": "u2", "size": 1},
    {"name": "mesa-llvmpipe-x64-25.3.0.7z", "browser_download_url": "u3", "size": 1},
    {"name": "mesa-llvmpipe-x86-25.3.0.7z", "browser_download_url": "u4", "size": 1},
    {"name": "mesa-lavapipe-x64-25.3.0.7z", "browser_download_url": "u5", "size": 1}]}


def test_pick_asset():
    assert software.pick_asset(RELEASE)["name"] == "mesa-llvmpipe-x64-25.3.0.7z"
    rel = json.loads(json.dumps(RELEASE))
    rel["assets"].append({"name": "mesa-llvmpipe-x64-25.3.0.zip", "browser_download_url": "u6"})
    assert software.pick_asset(rel)["name"].endswith(".zip")          # no extractor needed
    with pytest.raises(RuntimeError):
        software.pick_asset({"tag_name": "x", "assets": RELEASE["assets"][:2]})


def test_install_from_archive(tmp_path):
    arc = tmp_path / "mesa-llvmpipe-x64-25.3.0.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("mesa-llvmpipe-x64/opengl32.dll", b"MZ1")
        z.writestr("mesa-llvmpipe-x64/libgallium_wgl.dll", b"MZ2")
        z.writestr("mesa-llvmpipe-x64/LICENSE.txt", b"MIT")
        z.writestr("mesa-llvmpipe-x64/readme.md", b"ignored")
        z.writestr("mesa-llvmpipe-x86/opengl32.dll", b"wrong arch")
    dest = tmp_path / "dest"
    fake_mesa(dest, extra=("old.dll",))
    software.install_from_archive(arc, dest)
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["LICENSE.txt", "SOURCE.txt", "libgallium_wgl.dll", "opengl32.dll"]
    assert (dest / "opengl32.dll").read_bytes() == b"MZ1"
    assert "MIT" in (dest / "SOURCE.txt").read_text()


def test_restart_commands(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\\Mandelscope\\Mandelscope.exe")
    assert software.restart_command(["gui"]) == [r"C:\\Mandelscope\\Mandelscope.exe", "gui"]
    monkeypatch.setattr(sys, "frozen", False)
    monkeypatch.setattr(sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(sys, "argv", ["/x/mandelscope/__main__.py", "gui"])
    assert software.restart_command(["gui"]) == ["/venv/bin/python", "-m", "mandelscope", "gui"]
    monkeypatch.setattr(sys, "argv", ["/x/mandelscope/launcher.py"])
    assert software.restart_command([]) == ["/venv/bin/python", "-m", "mandelscope.launcher"]


def test_windows_fallback_restarts_in_cpu_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(software, "gpu_check", lambda: (False, "requested OpenGL 4.3, got 1.1"))
    started = []
    monkeypatch.setattr(software, "restart", lambda cpu, args=None: started.append(cpu))
    monkeypatch.setattr(software, "find_mesa", lambda: tmp_path)
    assert software.fallback_if_needed() and started == [True]
    # Mesa missing and the offer declined: start normally (the usual error message follows)
    monkeypatch.setattr(software, "find_mesa", lambda: None)
    monkeypatch.setattr(software, "_message", lambda *a, **k: False)
    assert not software.fallback_if_needed() and started == [True]
    # a working driver, or --gpu: no check / no restart
    monkeypatch.setattr(software, "gpu_check", lambda: (True, "NVIDIA GeForce RTX 5060 Ti"))
    assert not software.fallback_if_needed()
    monkeypatch.setenv(software.ENV, "0")
    monkeypatch.setattr(software, "gpu_check", lambda: pytest.fail("checked despite --gpu"))
    assert not software.fallback_if_needed()


def test_gpu_check_accepts_this_driver():
    # in a separate process: the check makes (and drops) its own OpenGL context, which would
    # leave the test session's shared context no longer current
    code = ("import os; from mandelscope import software\n"
            "if os.environ.get('MANDELSCOPE_SOFTWARE') == '1': software.enable()\n"
            "print(repr(software.gpu_check()))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    ok, name = eval(r.stdout.strip().splitlines()[-1])
    assert ok, name


def test_cli_flag_end_to_end():
    r = subprocess.run([sys.executable, "-m", "mandelscope", "--software", "info"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "CPU mode: Mesa llvmpipe" in r.stdout and "llvmpipe" in r.stdout.split("GPU")[1]
