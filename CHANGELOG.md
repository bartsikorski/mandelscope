# Changelog

## 1.0.0 — first public release

Mandelscope is a GPU deep-zoom Mandelbrot explorer.

### Exploring
- Live exploration to magnifications of 10^3000 and beyond. Every frame is computed on the GPU with perturbation, rebasing and **second-order bilinear approximation**, which needs 3–4× fewer GPU iteration steps at deep, high-iteration places than first-order skips.
- Smooth navigation:
  - spring-damped wheel zoom;
  - gliding pans and right-drag rotation;
  - 2× and 4× zooms that reuse the samples already computed;
  - camera steps paced to the display refresh.
- The GPU work per frame stays within its time budget. Zooms and turns at heavy places switch to half resolution automatically and sharpen again when you stop.
- **Automatic iteration limit.** It doubles only when that reveals detail outside the set and lowers again when the extra iterations change nothing; points next to large black components no longer trigger runaway limits. A limit raised at a deep place fades as you zoom out of it, and the skip tables are built ahead while you zoom out (both made zooming out of deep minibrots lag).
- Minibrot finder, Julia set preview and full Julia set exploration **to any depth** (perturbation with a reference at the view's point and the orbit of 0), and a location library with famous places.
- **The Burning Ship** (B): the same engine with perturbation through the absolute values, exact also next to the needle, distance estimation and lighting from its 2×2 derivative, a mini-ship finder, six places down to 10^106, and scripts (`fractal ship`).
- **Game controllers** (PlayStation, Xbox, Switch Pro, ... over USB or Bluetooth): left stick moves, R2 / L2 zoom, right stick zooms and turns, buttons for back, home, minibrot, looks, iterations, keyframes, screenshot. Speed, dead zone and zoom direction are adjustable.
- **CPU mode** for computers without an OpenGL 4.3 graphics driver: the same engine on Mesa's llvmpipe. Windows switches to it by itself (the release zip includes Mesa; from source it is downloaded on request); `--software` forces it, `--gpu` turns the automatic switch off. Zooms drop to a quarter of the resolution where needed, and the window stops redrawing when nothing changes.

### Interface
- **Cinematic HUD** (default) or the **Classic** style:
  - corner readouts with a depth ruler;
  - an outlined side panel;
  - five accent colours.
- A welcome card on first start with the basic controls.

### Colour
- 17 looks with live previews, including Silk (stripe texture), Midnight (glowing filaments), Blueprint (distance colouring with a drafting grid), Opal and Graphite.
- Gradient editor, random palettes, auto-contrast mapping, distance estimation, 3-D lighting, stripes, field lines.

### Animation and export
- Keyframe timeline and a small script language for camera paths. "Copy AI prompt" helps an assistant write scripts.
- Videos up to 8K with NVENC / AMF / QSV / x264, and fast zoom movies that share key pictures between frames. Stills come with the location embedded in the PNG.
- **More iterations for stills and videos**: up to 256× the explorer's limit (Export tab, `--iter-mult`). Periodic reference orbits (at minibrots) are stored once instead of up to the limit, so limits in the millions cost no memory there.

### Speed
- **Precise mode in float-float**: two float32 per number (about 48 bits) with a separate exponent, for graphics cards that run float64 slowly: 4–7.5× faster than float64 on an RTX 5060 Ti (seahorse 1e-12: 122 instead of 914 ms). `bench --f64` compares float32, float64 and float-float on your GPU.
- Shortest stored skip of the second-order BLA: 2^6 iterations for limits below 65,536, 2^7 above, and plain-float runs of 32 iterations. On an RTX 5060 Ti (1920x1080, `bench --bla`): spiral 10^100 5.0 ms (7.0 with 2^7 everywhere), 10^1000 5.6 (6.9), the period-6006 minibrot 688 (761 with 2^6 everywhere), the period-153231 one 749 (761); earlier pre-release builds used 2^4 (957 and 781 ms at the two minibrots).
- Reference orbits: long orbits are computed up to 1.8× faster (4 million iterations: 4.9 → 2.8 s), and the picture no longer waits for them: after half a second it is drawn with the orbit computed so far and continues when the rest arrives. An orbit cancelled by moving on is resumed when you come back.
- Cheaper skips: lookups are made only where the shortest skip can succeed, and skips whose terms fit the float range are done in plain arithmetic. `bench --bla` shows the variants.
- The explorer's background worker built the tables of long references with 4-iteration skips instead of the default; at the period-6006 minibrot that took 1,585 ms per picture instead of 801 ms.

### Fixes
- Videos smaller than 256x144 with the codec on "auto" use x264: NVENC refuses frames below its minimum size.

### Tools
- `tools/readme_media.py` and `tools/readme_tour.py` make the animations and pictures of the README (zoom movies as GIFs; a tour recorded from the explorer window).
- `bench`: the pan step of the interactive table is timed after a warm-up pan (the first one also compiled a shader: home showed 6.3 ms on the RTX 5060 Ti).
- `bench`, `bench --experiments` and `bench --bla` measure the engine on your GPU.
- `gui --motion-study` records and measures smoothness; `tools/analyze_footage.py` turns the recording into tables and plots.
- `tools/look_sheet.py` renders contact sheets of the looks.
- `get-mesa` installs the software OpenGL for CPU mode on Windows. The Windows CI job runs the whole test suite, including the scripted GUI session, in CPU mode.

### Migrating from the pre-release DeepZoom builds
Settings in `%APPDATA%\deepzoom` are read once. Your files in `Documents\DeepZoom` keep being used until a `Documents\Mandelscope Files` folder exists. PNGs and scripts saved by DeepZoom still open.

The development history before this release, with the measurements behind each change, is in `docs/design-notes.md`.
