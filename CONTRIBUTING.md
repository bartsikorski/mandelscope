# Contributing to Mandelscope

Thanks for your interest! Bug reports, benchmark results from your GPU, new colour looks, famous places and code are all welcome.

## Setting up

```
git clone https://github.com/bartsikorski/mandelscope
cd mandelscope
uv sync --all-extras          # Python 3.12 + dependencies into .venv (https://docs.astral.sh/uv/)
uv run python -m mandelscope  # the explorer
uv run pytest -q              # the test suite (needs OpenGL 4.3; Linux without a display: install xvfb)
uv run ruff check .           # lint (pyflakes rules)
```

On Windows, `Install Mandelscope.bat` does the same and creates shortcuts.

## How the code is organised

| Module | What it does |
|---|---|
| `mp.py` | arbitrary-precision `View` (centre in MPFR, log2 radius, angle) |
| `reference.py`, `bla.py` | reference orbit (MPFR), second-order BLA tables |
| `shaders/perturb.comp` | the deep-zoom kernel (perturbation, rebasing, BLA, rescaled floats) |
| `renderer.py` | GPU engine: fields, scheduling, sample reuse, colouring, display |
| `control.py` | camera and work control without graphics (spring zoom, pacing, automatic iterations, controller mapping) |
| `gamepad.py` | game controllers through GLFW's gamepad API |
| `software.py` | CPU mode: Mesa llvmpipe, finding / downloading it, the start-up fallback |
| `app.py`, `ui.py`, `hud.py` | the application and its two interface styles |
| `offline.py`, `keymovie.py`, `video.py` | still images and videos |
| `selftest.py`, `study.py` | scripted GUI session and motion study |

`docs/design-notes.md` records why things are the way they are, with the measurements behind each decision. Please read the part about the area you are changing.

## Guidelines

- **Measure first.** Speed changes come with numbers: `run.bat bench --experiments` (or `--bla` for deep renders), or a motion study plus `tools/analyze_footage.py` for anything you see while moving. Please include the GPU model.
- **Keep pictures exact.** The tests compare GPU results with an MPFR ground truth and with each other (reuse equals a fresh render, BLA on equals off). A speed-up that changes pictures needs a test showing that the difference stays within float precision.
- **Add a test** for every bug fix and feature. Tests without a GPU go in `tests/test_control.py` or `tests/test_math.py`.
- **Windows without a GPU:** `run.bat get-mesa` once, then `set MANDELSCOPE_SOFTWARE=1` before `uv run pytest -q` runs the whole suite in CPU mode (that is what the Windows CI job does).
- **Small pull requests** with one topic each are easiest to review.

## Adding a look or a place

- **Looks** live in `mandelscope/palettes.py` (`PALETTES` and `LOOKS`). Check yours with `python tools/look_sheet.py --looks YourLook --out sheet.png` at several places before opening a pull request.
- **Places** are small JSON files in `mandelscope/locations/`. Save the place in the explorer's Library, copy the `.json`, add a `"title"`, and render the thumbnail with `python tools/make_thumbnails.py`.

## Reporting bugs

Please include:
- your GPU and driver version (`run.bat info` prints them);
- what you did and what you expected;
- the log, `%LOCALAPPDATA%\mandelscope\mandelscope.log`.

For a display or smoothness problem, a short motion study helps a lot: `run.bat gui --motion-study-quick`, then attach `frames.jsonl` and `meta.json`.
