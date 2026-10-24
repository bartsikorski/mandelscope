# PyInstaller spec - one-folder build with two programs sharing the same files:
#   Mandelscope.exe      windowed explorer (double-click)
#   mandelscope-cli.exe  console tool: render / video / bench / info
# Build with "Build Mandelscope.exe.bat" (or: uv run --with pyinstaller pyinstaller packaging/mandelscope.spec)
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
PKG = os.path.join(ROOT, "mandelscope")

datas = [
    (os.path.join(PKG, "shaders"), "mandelscope/shaders"),
    (os.path.join(PKG, "locations"), "mandelscope/locations"),
    (os.path.join(PKG, "assets"), "mandelscope/assets"),
]
binaries = []
hiddenimports = collect_submodules("mandelscope") + collect_submodules("glcontext") + ["moderngl"]
SKIP = ("demos_python", "demos_cpp", "demos_assets", "imgui_terminal", "immvision", "doc")


def _keep(name):
    parts = name.replace("\\", "/").replace(".", "/").split("/")
    return not any(p in SKIP for p in parts)


for pkg in ("imgui_bundle", "gmpy2", "imageio_ffmpeg"):
    d, b, h = collect_all(pkg)
    datas += [x for x in d if _keep(x[1]) and _keep(os.path.basename(x[0]))]
    binaries += b
    hiddenimports += [x for x in h if _keep(x)]

excludes = ["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest", "PyQt5", "PySide6",
            "cv2", "pydantic", "pydantic_core", "yaml", "pyte", "torch", "sympy",
            "imgui_bundle.demos_python", "imgui_bundle.demos_cpp", "imgui_bundle.imgui_terminal"]
ICON = os.path.join(PKG, "assets", "mandelscope.ico")


def analysis(script):
    return Analysis([os.path.join(SPECPATH, script)], pathex=[ROOT], binaries=binaries, datas=datas,
                    hiddenimports=hiddenimports, excludes=excludes, noarchive=False)


a_gui = analysis("Mandelscope.py")
a_cli = analysis("mandelscope_cli.py")
pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="Mandelscope", console=False,
              icon=ICON if sys.platform == "win32" else None, upx=False)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="mandelscope-cli", console=True,
              icon=ICON if sys.platform == "win32" else None, upx=False)

coll = COLLECT(exe_gui, exe_cli,
               a_gui.binaries, a_gui.datas, a_cli.binaries, a_cli.datas,
               upx=False, name="Mandelscope")
