# Third-party notices

Mandelscope is MIT-licensed (see `LICENSE`). It bundles the following fonts, each under its own licence. The licence texts sit next to the font files in `mandelscope/assets/fonts/`.

| File | Licence | Copyright |
|---|---|---|
| `Inter/Inter-Regular.ttf`, `Inter-SemiBold.ttf` | SIL Open Font License 1.1 (`Inter/OFL.txt`) | The Inter Project Authors |
| `JetBrainsMono/JetBrainsMono-*.ttf` | SIL Open Font License 1.1 (`JetBrainsMono/OFL.txt`) | The JetBrains Mono Project Authors |
| `Roboto/Roboto-*.ttf` | Apache License 2.0 (`Roboto/LICENSE.txt`) | Google |
| `Inconsolata-Medium.ttf` | SIL Open Font License 1.1 (`OFL-Inconsolata.txt`) | The Inconsolata Project Authors |
| `Font_Awesome_6_Free-Solid-900.otf` | Font: SIL OFL 1.1; icons: CC BY 4.0 (`OFL-FontAwesome.txt`) | Fonticons, Inc. |

Inter and JetBrains Mono were converted from the WOFF2 files of their npm packages to TTF. The private-use glyphs were removed from Inter so that they don't hide the icon font. Roboto, Inconsolata and Font Awesome come from the `imgui-bundle` assets.

## CPU mode on Windows (in the release zip, or downloaded by `get-mesa`)

| Files | Licence | Source |
|---|---|---|
| `software/opengl32.dll` and the other DLLs next to it | Mesa: MIT (`software/MESA-LICENSE.rst`); it contains LLVM: Apache-2.0 WITH LLVM-exception (`software/LLVM-LICENSE.txt`) | built by [mmozeiko/build-mesa](https://github.com/mmozeiko/build-mesa) from the Mesa release named in `software/SOURCE.txt` |

These files are not in the repository. The release workflow downloads them and copies their licences next to them.

## Installed dependencies (not part of this repository)

| Package | Licence |
|---|---|
| imgui-bundle (Dear ImGui, hello_imgui) | MIT |
| GLFW (inside imgui-bundle; game controllers use its gamepad API and built-in copy of the SDL controller database) | zlib |
| moderngl, glcontext | MIT |
| numpy | BSD-3-Clause |
| gmpy2 (GMP, MPFR, MPC) | LGPL-3.0-or-later |
| Pillow | MIT-CMU |
| imageio-ffmpeg | BSD-2-Clause (its ffmpeg binary: LGPL/GPL, see the package) |
| pytest, ruff (development only) | MIT |

The stand-alone Windows build (`Build Mandelscope.exe.bat`, PyInstaller) bundles these packages. If you redistribute that build, include their licences as well. The release workflow copies this file and `LICENSE` into the zip.

## Algorithms

The engine is an independent implementation of published techniques; no code was taken from other programs. [CREDITS.md](CREDITS.md) names each technique, its origin and the projects we studied.
