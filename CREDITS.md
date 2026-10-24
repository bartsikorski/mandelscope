# Credits, references and inspirations

Mandelscope is written from scratch in this repository and is MIT-licensed. Its engine is built on ideas that the fractal community worked out and published in the open. This file names every technique we follow, where it comes from, and the projects we studied.

**What "followed" means here.** We read papers, blog posts, forum threads and project READMEs, and implemented the ideas ourselves. **No source code was copied from any project listed below.** Several of them are GPL- or AGPL-licensed; we did not take code from them, so their licences do not apply to Mandelscope's code. Where a formula appears in our code, the comment in the code names its origin too.

Third-party *files* that are bundled or downloaded (fonts, Mesa) are listed with their licences in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Deep-zoom mathematics

| Technique (where in Mandelscope) | Origin | References |
|---|---|---|
| Perturbation: one high-precision reference orbit, low-precision deltas per pixel (`shaders/perturb.comp`) | Kevin I. Martin, SuperFractalThing (2013) | [Mu-Ency: Perturbation methods](https://mrob.com/pub/muency/perturbationmethods.html), [Mu-Ency: Martin, Kevin](https://mrob.com/pub/muency/martinkevin.html) |
| Rescaled deltas with a separate exponent (floatexp), full floatexp steps where the reference is tiny (`perturb.comp`, `reference.py`) | Claude Heiland-Allen, in Kalles Fraktaler 2+ | [Deep zoom theory and practice](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html) (mathr.co.uk, 2021) |
| Rebasing: switching a pixel back to the start of the reference when it gets closer to it (glitch-free with one reference) | Zhuoran, fractalforums.org, December 2021 | [Another solution to perturbation glitches](http://fractalforums.org/fractal-mathematics-and-new-theories/28/another-solution-to-perturbation-glitches/4360), [Mu-Ency: Bivariate linear approximation](https://mrob.com/pub/muency/bivariatelinearapproximati.html) |
| Bilinear approximation (BLA): skip tables with doubling levels, merge rules and validity radii (`bla.py`) | Zhuoran (same thread); used in Kalles Fraktaler and Fraktaler 3 by Claude Heiland-Allen, and in Imagina | as above; [Kalles Fraktaler](https://mrob.com/pub/muency/kallesfraktaler.html) |
| Second-order BLA (skips that keep the quadratic terms) (`bla.py`) | The idea is described in the README of [Desarso/mandelbrot-webgpu](https://github.com/Desarso/mandelbrot-webgpu) (GPL-3.0). Our formulas, radii and tables were derived independently | `bla.py` docstring, `docs/design-notes.md` |
| Periodicity shortcut for references on an attracting cycle (`reference.py`) | standard technique | — |
| Deep Julia sets: a reference at the view's point plus the orbit of the critical point 0, rebasing from the first to the second (`perturb.comp`, `offline.py`) | Rebasing is Zhuoran's (above); using the critical orbit as the reference to rebase to is how Mandelscope applies it to Julia sets | `docs/design-notes.md` |
| Float-float ("precise") deltas: two float32 per number, exponent kept separately (`shaders/perturb_ff.comp`) | The "2×32 + exponent" number types of [FractalShark](https://github.com/mattsaccount364/FractalShark) (GPL-3.0) gave the idea; the arithmetic is the classic error-free sum and product: T. J. Dekker, *A floating-point technique for extending the available precision*, Numerische Mathematik 18 (1971); D. E. Knuth, *The Art of Computer Programming*, vol. 2; for GPUs, A. Thall, *Extended-precision floating-point numbers for GPU computation* (2006) | `perturb_ff.comp` header |

## The Burning Ship

| Technique (where in Mandelscope) | Origin |
|---|---|
| The fractal, z → (\|Re z\| + i\|Im z\|)² + c | Michael Michelitsch and Otto E. Rössler, *The "Burning Ship" and its quasi-Julia sets*, Computers & Graphics 16(4), 435–438 (1992). Mandelscope draws it upright, as nearly all pictures of it are shown, by iterating at the conjugate point (`formulas.py`) |
| Perturbation through the absolute values with `diffabs(c, d) = \|c + d\| − \|c\|` (`perturb.comp`) | "laser blaster" on fractalforums.com, [Perturbation formula for Burning Ship](http://www.fractalforums.com/new-theories-and-research/perturbation-formula-for-burning-ship-(hopefully-correct-p)/msg74090/#msg74090); the rescaled form as described by Claude Heiland-Allen in [Deep zoom theory and practice](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html), which also describes the "deep needle" problem (one part of the reference far below the other). Mandelscope's answer to it, storing the product XY of each reference value with its own exponent, is its own |
| Distance estimate and lighting from the real 2×2 derivative, written as v → P v + Q v̄ (Wirtinger form) | standard complex analysis |
| Mini-ship finder: ball-period detection and Newton's method with the 2×2 Jacobian (`nucleus.py`) | the ball method of the minibrot finder (above), which carries over because folding into the first quadrant moves no two points further apart; Newton in two real variables |

## Finding minibrots

| Technique | Origin |
|---|---|
| Ball-period detection, Newton's method for the nucleus, size estimate, Misiurewicz points (`nucleus.py`) | Robert Munafo's [Mu-Ency](https://mrob.com/pub/muency.html) (period detection, atom domains) and Claude Heiland-Allen's `mandelbrot-numerics` library and blog ([overview on Wikibooks](https://en.wikibooks.org/wiki/Fractals/mandelbrot-graphics)) |

## Colouring

| Technique | Origin |
|---|---|
| Smooth (continuous) iteration count | Linas Vepstas, *Renormalizing the Mandelbrot Escape* (1997); see [Plotting algorithms for the Mandelbrot set](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set) |
| Exterior distance estimation; 3-D (slope) lighting from its derivative | H.-O. Peitgen and D. Saupe (eds.), *The Science of Fractal Images* (Springer, 1988); [Inigo Quilez's articles](https://iquilezles.org/articles/msetsmooth) |
| Stripe average colouring | Jussi Härkönen, *On Smooth Fractal Coloring Techniques*, MSc thesis, Åbo Akademi University (2007) |
| Binary decomposition and field lines | H.-O. Peitgen and P. H. Richter, *The Beauty of Fractals* (Springer, 1986) |
| Interior detection from the derivative of the orbit (attracting cycles) | standard technique, described in the deep-zoom literature above |

## Rendering and interaction

| Technique | Origin |
|---|---|
| Reusing computed pixels while zooming and panning | [XaoS](https://xaos-project.github.io/) (Jan Hubička, Thomas Marsh and contributors); the description in [ttsiodras/MandelbrotSSE](https://github.com/ttsiodras/MandelbrotSSE) (GPL-2.0) |
| Coordinate-preserving sample reuse and reprojection of finished pictures while the camera moves | the README of [byronbuzz/DeepGPU-Zoomer](https://github.com/byronbuzz/DeepGPU-Zoomer) (GPL-3.0) |
| Catmull–Rom interpolation of the coarse progressive levels | E. Catmull and R. Rom, *A class of local interpolating splines* (1974) |
| R2 low-discrepancy sequence for the anti-aliasing jitter (`app.py`) | Martin Roberts, [The Unreasonable Effectiveness of Quasirandom Sequences](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) (2018) |
| Critically damped spring for wheel zooming (`control.py`) | standard technique, e.g. T. Lowe, *Critically Damped Ease-In/Ease-Out Smoothing*, Game Programming Gems 4 (2004) |
| CPU mode: the unchanged engine on a software OpenGL driver | [Mesa](https://www.mesa3d.org/)'s llvmpipe, Windows builds by [mmozeiko/build-mesa](https://github.com/mmozeiko/build-mesa) |
| Game controllers | [GLFW](https://www.glfw.org/)'s gamepad API with its copy of the [SDL controller database](https://github.com/mdqinc/SDL_GameControllerDB) |

## Projects we studied

These shaped decisions (what to try, what to measure), see `docs/design-notes.md`, section 1:

- [Kalles Fraktaler 2+](https://mrob.com/pub/muency/kallesfraktaler.html) (Karl Runmo; Claude Heiland-Allen) and Fraktaler 3 (Claude Heiland-Allen)
- [Imagina](https://github.com/5E-324/Imagina) (AGPL-3.0): linear approximation with stages (a possible next step)
- [FractalShark](https://github.com/mattsaccount364/FractalShark) (GPL-3.0): the "2×32 + exponent" number type, the inspiration for a faster precise mode
- [Fractalshades](https://github.com/GBillotey/Fractal-shades) (G. Billotey): deep zooms from Python
- [Desarso/mandelbrot-webgpu](https://github.com/Desarso/mandelbrot-webgpu) (GPL-3.0), [byronbuzz/DeepGPU-Zoomer](https://github.com/byronbuzz/DeepGPU-Zoomer) (GPL-3.0), [gcollombet/mandelbrot](https://github.com/gcollombet/mandelbrot), [ttsiodras/MandelbrotSSE](https://github.com/ttsiodras/MandelbrotSSE) (GPL-2.0)
- the many contributors at [fractalforums.org](https://fractalforums.org)

## Libraries

Dear ImGui (Omar Cornut) through [imgui-bundle](https://github.com/pthom/imgui_bundle) and hello_imgui (Pascal Thomet), GLFW, moderngl and glcontext, gmpy2 with GMP, MPFR and MPC, NumPy, Pillow, imageio-ffmpeg with FFmpeg. Licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How it was made

Mandelscope was developed with Claude (Anthropic) as an AI programming assistant. The measurements behind each decision are in `docs/design-notes.md`.

If you find a technique here that is not credited correctly, please open an issue.
