# Mandelscope — research notes and design decisions

> Development notes, written while the program was built (as "DeepZoom" until the first public release). Each section records what was measured and why things changed; the numbers come from the GPUs named there.

## 1. What existing projects do

| Project | Stack | Depth / precision | Notable tricks | Video |
|---|---|---|---|---|
| gcollombet/mandelbrot (Rust + wgpu, web) | WebGPU, float32 | perturbation + rebasing, ~1e-40 | renders only the changed part of the screen, auto iterations, derivative-based early exit | – |
| ttsiodras/MandelbrotSSE (C, SDL2) | AVX/SSE + OpenMP (CUDA variant) | double (~1e-13) | XaoS reuse: only ~0.75 % of pixels recomputed per frame while zooming; 1/16-res pass to skip the black interior | autopilot only |
| byronbuzz/DeepGPU-Zoomer (TS, WebGPU) | WebGPU compute, BigInt reference in a worker | deep, compensated f32 + exponent ("Wide") | rebasing, hierarchical BLA, coordinate-preserving sample reuse, reprojection, pending regions, bounded continuation, recolour without recompute, tiled PNG | – |
| Kalles Fraktaler / Fraktaler-3 (mathr) | C++, OpenCL/GLSL | unlimited | perturbation, rescaled iterations, BLA, many number types | zoomasm (keyframe assembly) |
| FractalShark | CUDA | unlimited | "2×32 + exponent" type, CUDA LA, multithreaded MPIR reference, reference compression, GPU reference orbit (NTT) | – |
| Imagina (Zhuoran) | C++ CPU | unlimited | best CPU LA, periodicity detection, reference compression | – |
| Fractalshades | Python + numba (CPU) | 1e-2000+ | chained BLA, GUI + scripting | movie module |
| Phylliida/fractals (WebGL2) | GLSL | to 2^-340 | float32 deltas get noisy at high iteration counts -> df64 | – |

Key theory (Claude Heiland-Allen's *Deep Zoom* notes, Zhuoran's fractalforums thread):
perturbation `z' = 2Zz + z² + c`, rebasing when `|Z+z| < |z|`, BLA
`z_{m+l} = A z_m + B c` with single-step radius `ε|Z| − |c|/|2Z|` and merge
`A = AyAx, B = AyBx + By, R = max(0, min(Rx, (Ry − |Bx||c|)/|Ax|))`.

## 2. Decisions (taken together with the user)

| Decision | Choice | Why |
|---|---|---|
| Target | Windows PC, **RTX 5060 Ti 16 GB** | user hardware |
| GPU API | **OpenGL 4.3 compute shaders via moderngl** | runs on NVIDIA/AMD/Intel, same context for compute + display, fp64 available, and testable headlessly (Mesa llvmpipe supports GL 4.5 + fp64) |
| UI | **Dear ImGui (imgui-bundle / hello_imgui, GLFW)** | single GPU window, panels over the fractal, tight render loop |
| Precision | **float32 default**, float64 "precise"; both **rescaled** (mantissa + int exponent) with floatexp fallback | consumer NVIDIA FP64 = 1/64 rate; rescaling gives unlimited depth with hardware floats |
| Reference orbit | **gmpy2 (MPFR) in a spawned worker process** | ~0.6 µs/iter at 128 bits, ~7 µs at 4096 bits; separate process avoids GIL stalls |
| Acceleration | rebasing + BLA (levels ≥ 2, log2 radii; tolerance 2^-24 for float32, 2^-45 for float64) + interior detection + periodicity shortcut + automatic long-lived reference (non-escaping or highest-iteration pixel) | dominant speed-ups at depth |
| Responsiveness | resumable per-sample state, adaptive dispatch size/budget, progressive order, reprojection, dynamic resolution, pan sample reuse | interactivity + Windows TDR safety |
| Video | **offline keyframe render**, **live recording**, **record path → re-render**; a fast mode with key pictures shared by many frames came in v3.4 | user selection |
| Interpolation | log-spiral (similarity group) between keyframes + easing | constant zoom speed, target fixed, exact at depth |
| Encoder | ffmpeg pipe (imageio-ffmpeg build has NVENC), auto-probe HEVC/AV1/H.264 NVENC → x264 | quality and speed on RTX |
| Environment | **uv** + `pyproject.toml` / `uv.lock`, Python 3.12, `setup_windows.ps1`, `run.bat` | fast, reproducible |

## 3. Verification approach

* Ground truth = brute-force MPFR iteration of the exact sample coordinate.
* Depths 1 … 1e-1000 (Misiurewicz spiral), minibrots (period 6006 and 153 231, up to 280k
  iterations), float32 and float64: zero non-chaotic mismatches.  A mismatch counts as
  "chaotic" only if the true value changes when the sample point moves by 1e-3 px (f32) /
  1e-6 px (f64).
* Unit tests for BLA steps vs exact perturbation, reference orbit vs direct iteration, tiny
  floatexp values (2^-4000), periodicity shortcut, nucleus/size/Misiurewicz solvers, view
  transforms, keyframe interpolation invariants.
* Engine tests: resumption with tiny budgets, BLA on/off agreement, pan sample reuse equals a
  fresh render, incremental max-iteration increase equals a fresh render, tiling, orientation.
* Video tests: frame counts, first/last frames identical to stills.
* Scripted GUI session under a virtual display: wheel/drag/keys/rotation, presets to 1e-1000,
  f64 mode, colours, keyframes + playback, live recording, path recording, video & still
  export, minibrot finder, history, context menu — with screenshots.

## 4. Version 2 (after the first round of user feedback)

| Feedback | Diagnosis | Change |
|---|---|---|
| "every 1080p frame takes ~350 ms, whatever the depth" (bench) | fixed cost, not compute: the field clear mapped and zero-filled ~130 MB on the CPU, and panning read whole result buffers back | clear / shift / reopen / counting are compute passes; per batch only 12 counters are read back; one sync per batch; the bench prints a breakdown |
| "moving with the arrows blurs everything" | moving used half resolution, restarted every frame, and re-projected already re-projected images | full resolution while moving; integer-pixel panning reuses samples (now also when the BLA table is rebuilt); display composites by quality: exact samples > last *complete* frame re-projected > bilinear coarse levels; snapshots only of complete frames; idle anti-aliasing passes |
| rotation blurs | rotation re-projected a degraded snapshot | same compositing; 90° buttons / keys, one click back to 0°, right-drag follows the mouse, Shift snaps 15° |
| keyboard zoom | – | + / − (centre) and Z / X (cursor): tap = 2×, hold = continuous; Shift / Ctrl faster / finer |
| nicer UI, see every colour option | – | toolbar + tabbed side panel + status bar + timeline, Roboto + Font Awesome 6, dark theme with a gold accent; *Looks* gallery with live previews of the current view; palette strips; gradient editor; random palettes |
| nicer default colours | – | OKLab gradients, histogram auto-contrast ("auto" mapping), distance-estimate edges, 3-D slope light, stripe-average texture, decomposition / field-line / grid patterns, glow; 12 curated looks, default "Gilded" |
| sequences: script **and** timeline | – | line-based script language (`start`, `zoom in/out/to`, `fly to`, `pan`, `rotate`, `spin`, `hold`, `animate`, `look`, exact `key` lines) ↔ timeline; PCHIP-smooth zoom speed through keyframes; "Copy AI prompt"; render queue with pause/cancel, quick drafts, 8K/vertical/square presets, depth overlay |
| locations | – | library: built-in famous places with thumbnails + the user's own (`Documents/Mandelscope Files/Locations`, JSON + PNG), PNG screenshots carry their location |
| open without PowerShell / .exe | – | `Install Mandelscope.bat` (setup + Desktop/Start-menu shortcuts to `pythonw -m mandelscope.launcher`), windowed launcher with log file and error dialog, PyInstaller one-folder build (`Mandelscope.exe` + `mandelscope-cli.exe`) |

## 5. Version 3 (speed round)

The v2 bench on the RTX 5060 Ti (1080p) showed the fixed ~350 ms per frame gone (spiral /
deep frames 370 ms -> 27 ms, 7-8 ms in the explorer).  What was left:

| Finding | Diagnosis | Change |
|---|---|---|
| zooming looks blurry and "slow" | every frame of a continuous zoom/rotation **restarted** the field; a picture that needs more than one frame's 10 ms budget (seahorse 1080p ~ 90 ms) never finished, so the screen showed an ever more stretched old frame | the picture in flight is **kept while moving** and shown reprojected (new: the display can reproject the field too); it is restarted only when it is finished or no longer covers the screen. Targets: short zoom animations compute their **end view** directly; other zooms/rotations compute a view that covers where the camera will be during the next ~1.25 field times (velocity / animation prediction). Snapshots keep a **per-pixel quality** (alpha) and partial pictures are merged into them, so nothing computed is lost and uncovered borders are rare |
| tap / wheel / double-click zooms recompute everything | a 2^k zoom of the same grid contains 1/4 (1/16) of the old samples exactly | **exact reuse for 2x / 4x zooms** (in and out): the end view is nudged by < 1/2 pixel onto the old grid, a GPU remap keeps the coinciding samples (distance estimates rescaled); tests compare with a fresh render |
| export 2-4x slower than the explorer for the same frame | exports used an 8192-step per-sample budget and 250 ms dispatches | exports use the explorer's scheduling (2048 steps, ~15 ms dispatches) |
| home view 0.5 Gsteps/s | ~1 rebase every 2 iterations; after each rebase the step at Z_0 = 0 took the slow floatexp path | **plain-float step at Z_0** when the delta is in range (identical iteration results) — 30 % faster at home on the software GPU |
| BLA lookups cost time even where no skip can succeed | per lookup: log2 + table reads | **exponent-only gate** against the largest BLA radius (bit-identical) |
| GPU idle between dispatches | full memory barrier after every dispatch + glFinish per batch | no barrier between independent dispatches; one barrier + counter read per batch |
| PNG frame export ~0.6 s / frame on the CPU, in the render loop | synchronous PIL save, compression level 6 | PNG frames on worker threads, level 3 (lossless, 2x faster, +14 % size); live recording reads frames back asynchronously (two pixel buffers) |
| supersampling cost | every pixel gets ss x ss samples | **adaptive supersampling** for exports and the idle smoothing passes: one sample per pixel (+ distance estimate) first; pixels whose colour has no curvature against their neighbours, that are not near the set and not undecided keep that sample, the rest get all sub-samples. Same picture within 1-3/255. Measured (software GPU, 480x270, 2x2): 9-23 % fewer iterations on most test views, 2-9 % more on very dense ones (Julia island), because the pixels that need sub-samples are also the expensive ones; sequences skip it for 15 frames when > 60 % of a frame needed refining. The idle smoothing passes have no extra centre pass (the explorer's picture is the centre) |

**v3.1 after the RTX bench.** The experiments table showed ~300 ms rows for pictures that
take 5-10 ms: the per-dispatch sample count was adapted from timings but *carried over* from
one picture to the next, so after a heavy view it stayed at 64-170 samples (a few work groups
on a 36-SM GPU) - thousands of tiny dispatches.  Every sweep now starts at 65 536 samples
and never goes below 8 192.  Progressive levels are dispatched back to back (the GPU keeps
their order) and the CPU waits only for the pending lists: 2-3 waits per light picture
instead of 6.  Adaptive export supersampling was slower than computing every sub-sample on
the RTX (fixed costs of the extra pass dominate when a 1080p 2x2 frame takes 40 ms), so it is
opt-in; the idle smoothing passes keep it (they have no extra pass).  The bench now gives
each location its own reference, interleaves repeats, prints dispatches / waits, the cost of
one CPU-GPU wait, and the colour pre-pass time of exports.

**v3.2 (second RTX bench).** One CPU-GPU wait costs only ~0.05 ms on the user's PC, so
GPU-side pending lists were *not* built.  Changes:

* *Plain-float runs* in the kernel: after a normal step, up to 16 further iterations run
  without the main loop's bookkeeping (interior check, limits, BLA attempts that cannot
  succeed, renormalisation of the derivatives).  A run stops before any iteration that needs
  the main loop (escape, rebase, tiny reference value, BLA opportunity, |w| leaving the safe
  range), so iteration counts, smooth fractions and angles are identical; distance estimates
  differ by < 4e-4 relative (rounding).  The smallest interior derivative of a run is handed to
  the main loop, so inside detection stays exact.  Software GPU: 1.07-1.7x on heavy views.
* *Exports take the 'auto' colour histogram from the frame's own samples* (one tile, which now
  covers 1080p with 2x2 supersampling) - no pre-pass; tiled video frames use the previous
  frame's histogram; stills in tiles keep the pre-pass.
* *Sharp motion*: when a whole picture takes less than 80 % of the frame budget, moving
  computes exactly the view on the screen every frame - shown 1:1, never reprojected.  Slower
  views keep pictures in flight; their reprojection now uses Catmull-Rom filtering (clamped,
  no ringing) for both the field and the snapshot, the widening for rotation / zoom-out is
  capped at 1.5x, and a zoom-in end view is snapshotted into the screen view so the wider
  picture around it is kept.
* The experiments table checks every picture against a fresh render (steps, undecided and
  inside counts) and adds "new field each time" / "reference re-uploaded" rows to isolate the
  ~300 ms rows seen on the RTX (not reproducible on the software GPU).

**v3.3 (third RTX bench).** The experiments are clean (no more ~300 ms rows; every picture
checked).  On the RTX the plain-float runs give 1.5-1.8x (seahorse 84 -> 46 ms per 1080p
picture), the BLA gate 1.2x, waiting per level / barriers cost 10-200 %, the per-sample budget
does not matter.  Exports had become slower (spiral 11 -> 44 ms): the colour histogram added
one global atomicAdd per sample (~400 k) onto a few hot bins, which NVIDIA serialises - the
same histogram also ran after every finished picture in the explorer.  It now counts in shared
memory per work group and adds non-empty bins once.  Inside detection costs 6-10 % where nothing
is inside: it is now switched off after a picture with no inside / undecided pixels and back on
(with a recompute instead of an iteration boost) when a picture without it finds undecided
pixels.  Pending lists of new fields are committed at allocation (a field's first picture was
40 ms slower).

Measured but not changed (A/B in `mandelscope bench --experiments` on the real GPU first):
interior detection costs ~6-20 % per iteration when nothing is inside; per-sample budget
1024 vs 2048; tile-shaped (8x4) sample order (no gain on the software GPU).

**v3.4 (fourth RTX bench).** The ~300 ms rows were back, but only when one field object was
reused for many pictures ("new field each time": 6 ms) - and the explorer reuses its field
all the time.  The CPU read the field's counters by mapping the buffer every few milliseconds;
drivers move buffers the CPU maps often into system memory, and then every work group's
atomics on them cross the PCIe bus.  The CPU now reads and writes the counters (and the
colour histogram) only through two small staging buffers and GPU copies.  The bench has a
diagnostic for exactly this: one field reused for 6 x 4 pictures with the old and the new
path and with a new field each time (wall and GPU time per picture), and it reads the
driver's debug messages (KHR_debug message log) to report any buffer the driver moved
between video and system memory.

*Fast zoom movies* (`keymovie.py`, like zoomasm / KF's keyframe movies, but for any camera
path): frames are planned into windows; each window gets one key picture covering all its
frames (bounding box in the orientation of the middle frame) at the resolution of its
deepest frame times the supersampling q.  A window grows while the samples per frame keep
falling (for a steady zoom that is ~0.9 octave; a still or slow pan: one picture); where a
key picture would cost more than 75 % of computing the frames, those frames are computed
directly.  Each frame colours the samples it covers with its own colour settings (cycling,
colour keyframes, distance estimates rescaled to the frame's pixels, 'auto' histogram of
exactly the samples inside the rotated frame) into a half-float texture, then a box filter
over each output pixel (ceil(rho) x ceil(rho) bilinear taps, rho = key samples per pixel,
q <= rho <= 2q) makes the frame; the first ~6 frames of a window cross-fade from the
previous key picture.  On the software GPU: frames within 1-2/255 of computing them one by
one (rotating zoom, auto colours), as close to a 4x4-supersampled reference as the direct
2x2 frames, ~13 % of the samples for one octave per second at 30 fps.

Two colour fixes found on the way (they affected every video with 'auto' colours): the
temporal smoothing renormalised the blended mapping so that it started at 0 again - every
time the frame's smallest iteration count went up by one (every few frames of a zoom), all
colours jumped; and the mapping stored the share of samples up to the *end* of each bin,
one bin too high (visible when most of the picture lies within a few iterations).

*Julia sets*: a second kernel (`julia.comp`) with the same buffers, scheduling and result
format as the perturbation kernel, so fields, reuse (pans, 2^k zooms), colouring,
anti-aliasing, stills and videos work unchanged.  Direct iteration in float32 or float64
(automatic below the float32 limit), dz/dz0 for distance estimates and inside detection.
The explorer shows a live preview for the point under the cursor and switches between the
Mandelbrot set and a Julia set keeping each one's view and history.  (Fixed on the way: the
colour-look thumbnails were shown upside down - textures rendered by the GPU store the bottom
row first.)

Before the next two speed items, the bench measures what they would improve: `bench --f64`
compares float32 and float64 pictures (a float-float "precise" mode can only win where
float64 is used), and a new table times the CPU reference orbits (software-GPU machine: 3-36 ms
for the built-in places, up to 1e-1000 - a faster reference only matters for much deeper or
higher-iteration spots).

**v3.5 (fourth RTX bench, after v3.4).** The driver confirmed the diagnosis: with the old path
it reported moving the field's counter buffer from video to host memory 130 times during the
24 pictures; with staging copies only the small read-back copy goes to (intended) DMA memory.
All rows of the reuse diagnostic and the experiments are now within noise of a new field.
Fast zoom movies: 1.4x (spiral, deep) to 4.4x (seahorse) faster at 13 % of the samples - never
slower, so the fast mode is now the default.  For the light places a frame costs ~12 ms of
per-frame work (colouring the key picture's samples, filtering, read-back, CPU) next to 2-3 ms
of its share of the key picture; the movie bench now breaks this down (GPU timer queries for
colouring and filtering, CPU times for the histogram, waits, read-back, encoding) and runs
both modes once more with video encoding, before that part is optimised.

The breakdown (fifth RTX bench, spiral at 1080p 2x2, ms per frame): key pictures 4.8, colour
histogram 1.7 (a GPU round trip per frame), waiting for the GPU 2.1 (colouring 1.6, filter
0.3), read-back 3.2, rest 1.  "With encoding" looked 7-9 ms per frame more expensive, but the
writer already runs on its own thread: that was mostly starting and finishing ffmpeg (process,
NVENC session, moving the index to the front of the file - once per video) spread over a
31-frame clip; the bench now measures that fixed time separately.  So the frame loop was
pipelined: the 'auto' histograms of all frames of a window are counted in one batch (one
read-back per window; the mappings are then made frame by frame on the CPU with the same
smoothing), each frame's mapping goes into its own small texture (ring of 3), and frames are
read back through two pixel buffers - frame k is coloured, filtered and copied while the CPU
takes frame k-1 out and hands it to the encoder.  Same frames as before (software GPU).

Decided from the same bench: float64 is 3.6x (home) to 17x (seahorse) slower than float32 -
a float-float precise mode would help, but only users of precise mode; CPU reference orbits
take 3-42 ms for every built-in place (up to 1e-1000), so faster references are dropped for
now; adaptive export supersampling stays opt-in (still 10-40 % slower than every sub-sample);
inside detection costs 11 % at seahorse but saves work elsewhere, kept.

**v3.7 (smoothness, from footage).** The user saw "a grid" while pictures refine and found
wheel zooms a bit out of place.  A motion study mode now records every displayed frame with
a per-frame log, and measures on the GPU (uncompressed) which source each pixel shows, the
share of pixels next to a different source ("mixed": dot grids and stripes make it large) and
the change beyond the camera motion.  Software-GPU run, 3 places x 6 actions, old vs new:

| action | mixed % old (max) | mixed % new (max) |
|---|---|---|
| wheel zoom in | 5-15 (38-43) | 0.2-2.3 (1-5) |
| wheel zoom out | 3-25 (21-62) | 0.7-4 (1-6) |
| drag | 23-34 (53-81) | 0.6-1.5 (1-3) |
| held zoom key | 2-14 (19-42) | 0.1-0.2 (0.4-1.4) |

Cause: a refining field's finished samples form sparse lattices (every 8th, 4th, 2nd sample,
then rows of the last level).  They were drawn sharp between pixels that came from the older
(stretched) picture - a dot grid, then stripes - and partially finished pixels were blended
in.  Now a finished sample is shown only when at least 3 of its 4 neighbours are finished
too; before that the field takes part through its coarse levels, interpolated with
Catmull-Rom (bilinear through a sparse lattice draws straight-edged "tents" that look like a
grid too), and region by region the older picture or the field wins, never alternating per
pixel.  Wheel / key zooms follow a critically damped spring (exact integration) instead of
an exponential approach restarted at every notch: same ~0.25 s per notch, but the speed no
longer jumps up at each notch (simulated 60 fps, 6 notches: speed roughness 0.066 vs 0.095).
Both are switches (display.consistent, settings.wheel_spring) so the study records old and
new in the same session.

**v3.8 (the user's footage, interface, looks).** The study ran on the RTX 5060 Ti (3,673
frames, 1600x900).  Findings and changes:

* *The dot grid is gone on the real GPU*: mixed sources 0-0.9 % of pixels on average (max
  4 %) with the v3.7 display, 0.2-5 % (max 34-73 %) with the old one.
* *Iteration runaway.*  "more iterations: 4,000 ... 271,483" at Seahorse valley after the
  wheel zoom-out: the view holds a large piece of the main cardioid.  Points just inside its
  edge converge so slowly that recognising them as inside takes ~7/(1-|multiplier|)
  iterations, so the undecided share only halves with every doubling (measured: 34 % at
  2,000, 0.27 % at 256,000 iterations), while the old rule doubled whenever more than 0.3 %
  were undecided.  The pictures were the same (the undecided are drawn like inside points;
  only 0.03-1 % of pixels escaped per doubling), but the boost stayed at 128x for the rest of
  the place: the valley drag ran at 20.8 ms per frame instead of 16.7.  Now a small GPU pass
  counts, per octave below the limit, the samples that escaped and those recognised as inside
  (the result keeps the iteration of that recognition), and the limit is doubled only if the
  last octave resolved almost nothing (deep zooms: nothing escapes yet) or the escapes per
  octave predict that a fair share of the undecided samples would escape; it is halved on a
  new view when the lower limit would not ask for more (no flip-flop).  Simulated at 320x180:
  Seahorse/Elephant valley and the zoomed valley stay at the base limit (v3.7: 4-128x), the
  period-6006 minibrot still climbs 5,430 -> 173,760 (the escapes there need it), the
  period-153231 minibrot 9,579 -> 306,528.
* *Frame budget.*  New pictures took 15-17 ms of a 10 ms slice: the step loop only checked
  the time after a dispatch.  It now stops before a dispatch that would end after the slice
  (median overshoot +7.9 ms -> -2.2 ms in the software-GPU test), and the time the frame
  already spent deciding / restarting is taken off the slice.
* *Pacing.*  Frames were shown every 16.7 ms but started 22.7 / 12.4 / 15.0 ms apart in turn,
  and the camera moved by those raw intervals: uneven steps on an even display.  The camera
  now steps by the median of the last intervals plus a share of the time not yet handed out
  (never drifts; a real hitch passes at once): jitter 6 ms -> < 1.5 ms in the test.
* *Study artefacts.*  The 130-190 ms "hitches" were the study's own synchronous PNG saves
  (now in a thread); frames that only show the anti-aliased picture wrote no source ids
  (left out of 'mixed').  The log has the iteration limit and the time spent before / in /
  after the compute step, so the next footage shows where the remaining frame time goes.
* *Interface.*  "Cinematic HUD" style (the user's pick from four mock-ups, refined): the
  picture fills the window; corner readouts (magnification as 10^x with a depth ruler,
  coordinates, iterations, rotation, fps; picture status, frame time, precision, GPU), an
  icon row top right, an outlined panel with a text tab strip, toasts at the top, the picture
  progress along the bottom edge, optional corner brackets and a centre mark while moving;
  Inter and JetBrains Mono (SIL OFL, bundled), five accents.  The classic style stays in
  Render > Interface: both draw the same tab contents, so keeping it costs little.
* *Looks.*  A contact sheet (tools/look_sheet.py: every look at eight places) showed that
  Sunset and Nebula were near twins, Relief washed out and Neon wire flattened dense areas
  into teal.  Sunset is warmer now, Relief has a dark brown for depth, Neon wire is darker
  with a thinner glow, Ink less busy; five looks use features no look used before: Silk
  (stripe-average texture), Blueprint (colour by distance + drafting grid), Midnight (gold
  glow on navy), Opal (pastel with field lines), Graphite (black-and-white relief).

**v3.8.1 (second RTX study).** Pacing is fixed on the real GPU: frame time p95 17.0-18.4 ms
(v3.7: 24-29), slow frames 0-1.7 % (v3.7: 2-10 %), camera-step jitter 0.3-0.9 ms (v3.7:
2-4); Seahorse valley stays at 4,241-9,449 iterations (v3.7: 271,483).  But pictures took
4-8x longer: drags showed coarse levels (mixed sources 14-25 %) because fields never
finished.  Cause: the first batch of every time slice also waited for the GPU work queued
before it (the previous frame's display, panels, capture), so the dispatch-time estimate was
inflated, dispatches shrank to the minimum (8,192 samples) and the new stop-before-overrunning
rule then allowed one per frame.  v3.7 got away with it because it kept dispatching after the
first batch.  Now each slice first waits for that earlier work (one extra synchronisation),
so its own dispatches are timed cleanly.  Simulated 5 ms backlog, 10 ms slices: 8,192 ->
22-25k samples per slice (test_earlier_gpu_work_does_not_shrink_dispatches).

**v3.9 (deep renders, responsiveness, first run).**  Where deep, high-iteration pictures
spend their time (CPU prototype of one pixel's path, period-153231 minibrot at 10^21,
3,064,620 iterations): ~1,300 BLA skips but ~8,400 plain iterations per pixel, almost all
in the last 13k iterations before escape, where the delta is about 2^-22 of |Z| - just above
the first-order validity (~2^-24 |Z| in float32).  Keeping the quadratic terms in the skips
(second-order BLA: z' = A z + B c + C z^2 + D z c + E c^2, see bla.py) makes a skip valid up
to ~sqrt(eps) |Z|, which covers 93-95 % of those plain iterations.  Prototype (double): steps
per pixel 9,300 -> 2,200 with a 10x smaller median error against plain perturbation; GPU
(software renderer): 10,300 -> 2,900 steps, 1.12 -> 0.26 s; seahorse 1e-12: 1,286 -> 425
steps.  A second-order skip costs more than a few plain-float iterations, so the shortest
stored skip went from 4 to 16 iterations (on the software renderer longer minimum skips kept
winning; `bench --bla` measures order x shortest skip on the user's GPU to set the default).
Reference orbits (0.24 s at the period-153231 minibrot, 0.49 s at 10^3000) and BLA tables
(<0.1 s) are not the bottleneck.  Raising the iteration limit no longer reproduces a fresh
picture bit for bit with second order (skips split differently at the old limit): 1-2 % of
the continued, chaotic samples differ - first order stays exact (both tested).

Responsiveness: "while moving: automatic" (new default, also for saved settings that had full
resolution) computes zooms and turns at half resolution while a full-resolution picture
takes longer than ~0.2 s (back above 0.1 s: hysteresis), so heavy places show finished
pictures instead of ever more stretched old ones; pans keep full resolution (they reuse
almost every sample).  First run: a welcome card with the basic controls and a "fly to a
famous place" button.

**v3.10 / 1.0.0 (RTX bench, CPU mode, controllers).**  `bench --bla` on the user's RTX 5060 Ti
(1920x1080, wall ms; period-153231 / period-6006 minibrot): first order 1772 / 1351 at a
shortest skip of 2^2, 931 / 865 at 2^4; second order 899 / 1585 at 2^2, 781 / 957 at 2^4,
690 / 895 at 2^5, 686 / 801 at 2^6 (seahorse 1e-12: 30-31 ms from 2^4 on, 10^1000: 3-4 ms
everywhere).  The steps per pixel *rise* with longer skips (2,848 -> 12,985 at the first place)
while the time falls: a skip costs a table lookup and ends the fast run of plain iterations, and
a failed lookup costs the same.  Defaults now 2^6 (second order) and 2^4 (first).  The same
data exposed a bug: the background worker, which builds the tables of references longer than
60,000 iterations - exactly the deep minibrots - passed lev0=2 instead of the default, so the
explorer ran the slowest variant there (1585 instead of 801 ms at the period-6006 minibrot);
test_worker_tables_use_the_default_shortest_skip.  `bench --bla` now also prints the skips per
pixel (to separate the cost of a skip from that of a plain iteration) and tries 2^7, 2^8, 2^10.

CPU mode.  Options compared: a second engine for the CPU (numba / C) or the existing one on
Mesa's llvmpipe, which compiles GLSL compute shaders to SIMD code over 8 pixels per AVX2
instruction on every core.  llvmpipe needs no second implementation, gives identical pictures
and every feature, and is the renderer the test suite already runs on.  Measured on 2 cloud
cores (1280x720, full pictures): home 0.39 s, 10^100 spiral 0.66 s, 10^1000 0.63 s, seahorse
1e-12 5.5 s (0.23 G steps/s) - 60-330x more time per pixel than the RTX; colouring a frame 26-31
ms.  A compiled scalar loop (C, -O3 -march=native, double, the same perturbation step with
escape and rebase checks) did 0.12-0.13 G steps/s per core on the same machine, against llvmpipe's
0.115: a dedicated engine could win 1-2x, not the 10x that would justify a second code base.
Windows: Mesa is loaded by full path before imgui_bundle (whose native module links
opengl32.dll) and before moderngl, so GLFW, Dear ImGui and moderngl all get it when they ask
for "opengl32.dll" (the loader returns an already-loaded module of the same name).  Builds come
from github.com/mmozeiko/build-mesa (llvmpipe only, static); the release zip carries them in
software\, a source checkout downloads them with `get-mesa` or when asked at start-up.  Without
an OpenGL 4.3 driver a hidden test context fails and Mandelscope starts again in CPU mode.  On
the CPU, zooms may drop to a quarter of the resolution (auto_motion_level: a level is left when
a picture at it would take > 0.2 s), smoothing defaults to 4 passes (own settings file), and
hello_imgui idles (about 9 frames/s) while nothing is computed, animated or recorded.

Controllers.  GLFW's gamepad API (glfwGetGamepadState) maps PlayStation 4/5, Xbox, Switch Pro
and others to one layout with its built-in copy of the SDL controller database.  imgui_bundle
ships GLFW as a shared library and points PYGLFW_LIBRARY at it; ctypes on that file is the same
GLFW instance as the window (checked with glfwGetCurrentContext), so no package is added.
Dear ImGui's own gamepad navigation is left off: it would move the focus inside the panels
while the sticks fly the camera.  Mapping (control.pad_controls): radial dead zone + squared
response for the sticks, triggers 0..1 squared; left stick pans (1.4x key speed), R2 - L2 and
the right stick's y zoom (1.25x key speed), its x turns (90 deg/s); R1 / L1 = faster / finer.

**v3.11 (the five "next steps": references, skips, deep Julia sets, the Burning Ship,
float-float).**  The user's RTX bench of v3.10 put the period-6006 minibrot at 675 ms with a
shortest skip of 2^8 against 841 ms at 2^6, and the period-153231 one at 688 ms at 2^6, and
reference orbits at 3-10 us per iteration.

*Reference orbits.*  Values go into growing NumPy buffers (amortised doubling instead of
lists and concatenation), and the periodicity check runs after every further quarter of the
orbit instead of every few chunks (it scans up to 2M values: O(N) in total instead of O(N^2)).
Bit-identical orbits; c = i with 4M iterations: 4.94 -> 2.78 s (1.24 -> 0.70 us/iteration),
1M: 0.79 -> 0.66 s, short orbits unchanged.  More important for the explorer: the worker
sends the orbit computed so far after 0.5 s and at doubling intervals ("partial" messages,
with their BLA tables); the explorer draws with it, the iteration limit capped at its length
(status "iterations"), and the samples continue when the longer orbit arrives - the same
point, so the prefix is identical.  A cancelled orbit (the view moved on) is kept and resumed
when the same point is asked for again.  The tables of the short levels are built in
complex128 with a floatexp redo only for blocks whose values leave its range.

*Cheaper skips* (perturb.comp; `bench --bla` shows the variants).  On the software renderer
a skip cost about 17 plain iterations and a failed lookup about 11.  Three changes: inside the
fast runs of plain iterations a lookup is left for only when the shortest skip's radius at
that iteration admits the delta (prefilter, one load); a skip whose terms all fit the plain
float range is done in plain arithmetic in the frame 2^(k + Ae) instead of five floatexp
products and four normalised sums; the run length is capped at 16.  A back-off after failed
lookups was measured and left off.  The GPU numbers are the user's to measure.

*Deep Julia sets* (the user's screenshot: no zoom past 10^12, flat look previews).  The direct
kernel in float64 runs out of mantissa at ~10^12 (and the float32 previews long before).
Perturbation for a Julia set of c: every sample has the same c, so there is no dc term; the
delta is the offset of the sample's z0 from the view's point.  Reference A is the orbit of the
view's point, reference B the orbit of 0 (the critical point; it is the Mandelbrot reference
of c).  Samples start on A; rebasing (|z| < |dz|) moves them to the start of B, Z = 0, from
where they continue like Mandelbrot samples after a rebase.  Both orbits sit in one buffer
(B from uBaseB on) with their BLA tables (A's levels first), one flag bit in the sample state
says which one a sample follows, and dz/dz0 replaces dz/dc (no +1).  The switch happens
where float32 can no longer resolve the samples (julia_min_log2r + 1).  MPFR ground truth at
10^6, 10^30, 10^100 (basilica-like c) and 10^40 (rabbit), float32 / float64 / float-float: no
hard mismatches.  A test detail that cost time: a view centre given as a 53-bit float next to a
repelling fixed point puts the deep view somewhere featureless - centres are computed in MPFR.

*The Burning Ship.*  z -> (|x| + i|y|)^2 + c, drawn upright: y' = 2|xy| - Im c (the original map
at conj c; nearly all published pictures are mirrored that way).  Deltas use diffabs, the
exact |XY + d| - |XY| by cases, rescaled by 2^k.  Ground truth found the "deep needle" problem:
next to the needle (c near the real axis) the reference's imaginary part is 10^-300 while the
real part is about 2 - float32 and float64 round it to 0, and diffabs then takes the wrong sign.
The reference now carries X*Y of every value as (mantissa, exponent), computed from the MPFR
parts; with it 10^-300 and 10^-1000 next to the needle match MPFR in float32.  dz/dc is a real
2x2 matrix: stored as v -> P v + Q conj(v) (the extra vec2 makes the Burning Ship's sample
state 8 bytes longer: fields grow their buffers when the formula changes).  The fold is
multiplication by 2z or v -> conj(2z v); the distance estimate is |z|^2 log|z| / |J^T z| and the
light direction J^T z, checked against MPFR finite differences.  Interior detection uses
|dz/dz1| = prod 2|z| unchanged.  No BLA (a future step: skips only valid while no sign
changes).  Mini ships: ball-period detection works unchanged (folding moves no two points
further apart), Newton with the 2x2 Jacobian; chained searches gave the built-in places of
period 39 (10^-11), 228 (10^-33) and 1730 (10^-106).  Their exterior is chaotic at 10^-6 of a
pixel, so ground truth there can only say "no hard mismatch".  Found on the way: in float64
mode the float derivatives took vec2(z) of values below 10^-38 (next to a deep minibrot) as 0,
a false "inside"; such steps now take the floatexp path.

*Float-float precise mode* (perturb_ff.comp).  The "2x32 + exponent" idea (FractalShark): every
delta mantissa is hi + lo in two float32 (about 48 bits), the exponent separate as before, the
derivatives float32 as in float64 mode.  Error-free sums (Knuth) and products: fma where it
rounds once, else Dekker's split - the engine tests fma at start, because Mesa's llvmpipe
computes it with two roundings (products then only had float32 precision; a GPU test now
measures 2^-46 median error for each operation).  Same state layout as float64.  Against
float64 pictures at chaotic places: float-float 99.6-100 % identical samples, float32 85-99 %.
On the CPU (llvmpipe) float-float is 1.0-1.6x slower than float64, as expected without fast
float64 hardware; on GeForce cards (float64 at 1/64 rate) it should win - `bench --f64` prints
f32, f64 and ff frames.  The Burning Ship has no float-float kernel (float64 is used).

**v3.12 (the user's RTX bench of v3.11; lag when zooming out; more iterations).**  1920x1080,
ms per picture (float32 / float64 / float-float): home 14 / 49 / 20, seahorse 1e-12 41 / 914 /
123, 10^100 9 / 57 / 14, 10^1000 10 / 59 / 14 - float-float is 2.5-7.5x faster than float64 there,
so it is listed first among the precise modes.  The Burning Ship: 11-72 G steps/s without BLA
(10^-106 mini ship 412 ms).  `bench --bla`: prefilter and plain-arithmetic skips changed little
on the RTX (failed lookups 125 -> 13 per pixel, time within +-5 %: a failed lookup is cheap
there); 2^7 instead of 2^6 shortest skip: period-6006 minibrot 740 instead of 819 ms, the others
equal; runs of 32 instead of 16 plain iterations: 0-6 % faster.  New defaults 2^7 and 32; the
bench now compares 2^6 / 2^7 / 2^8 with runs of 16 / 32 / 64.

*Lag when zooming out after a deep zoom* (the user).  A frame-by-frame study (Shift + minus held
from the period-6006 minibrot to the whole set, software renderer, 800x450): the iteration limit
of the deep place (x44 of the automatic one, also the "floor" a loaded place sets) stayed on the
way out - 100,000-240,000 iterations at magnifications 10^3-10^10, where 2,500-17,000 do - and
the skip table, valid for pixel offsets up to 2x the screen, ran out after every octave: every
other frame was computed without skips while the worker built the next one (49 of 67 frames).
Measured pictures with the old and new limits: 10^6.5 4.0 -> 1.8 s, 10^9.4 16.0 -> 11.6 s
(software renderer).  Now a raised limit belongs to the depth where it was set: more than 2
octaves above it, it halves per octave (back to where it was when zooming in again); while
zooming out, tables get head-room for vel x 0.6 s (up to 7 octaves) and the next one is asked for
before the current one runs out, and a tight table is built again when the camera stops: 1
frame without skips of 68.  Inside detection, switched off after pictures without inside
points, is switched on again after zooming out by an octave (otherwise a wide view's inside
points would all run to the limit).  The GUI session holds the key for 40 frames from its
minibrot and checks both.

*More iterations for stills.*  There was no fixed cap (the explorer's automatic limit stops at
1024x, the typed limit at 2^31), but a reference orbit costs ~1 us and ~70 bytes (CPU + GPU) per
iteration, and a periodic one was tiled up to the limit.  Periodic references now keep their
cycle once and the kernels wrap the index by the period at the end (`uPerA/B`; equal pictures to
the tiled reference in the test), so any limit at a minibrot costs no memory.  Stills and videos
take a multiplier of the explorer's limit (x1 ... x256, Export tab; `--iter-mult`); key movies
show the samples up to the multiplied limit.

**v3.13 (October 23, 2026: the user's RTX bench of v3.12; publishing).**  `bench --f64` and two
`bench --bla` runs on the RTX 5060 Ti, 1920x1080.  The new defaults (shortest skip 2^7, runs of
32) were measured in full frames too, and there 2^7 costs at the shallower deep places, which
need few iterations:

| ms per picture (steps / skips per pixel) | 2^6 run32 | 2^7 run32 |
|---|---|---|
| spiral 10^100 (5,000 iterations), full frame | 5.0 | 7.0 |
| 10^1000 (20,000), full frame | 5.6 | 6.9 |
| 10^1000, bla table | 4 (107 / 6) | 5 (177 / 5) |
| seahorse 10^12 (20,000), full frame | 36.9 | 37.1 |
| minibrot p6006 (120,120) | 761 (27,962 / 5) | 688 (28,300 / 0) |
| minibrot p153231 (3 million) | 761 (12,985 / 31) | 749 (13,466 / 23) |

Below about 10^5 iterations a skip of 64 still pays (it replaces 64 plain iterations, and the
pictures need only a few hundred per pixel, so the 70 extra plain steps per pixel at 2^7 are
40 % of the work); near deep minibrots, where pixels iterate tens of thousands of times between
rebases, the longer runs of 2^7 win.  The table is now built with 2^6 when the reference is for
fewer than 65,536 iterations and with 2^7 above (`bla.default_lev0`; the table belongs to the
reference, so its limit decides; a table built for a lower limit stays valid when the
explorer raises it).  `bench --bla` has an "auto" column (the default) and the spiral 10^100.
On the software renderer (320x180): spiral 25 / 29 ms, 10^1000 25 / 33 ms, auto equal to 2^6
in steps and skips.  Deep Julia sets keep 2^7 (not measured for them).

The user's RTX bench of this change (1920x1080): full frames spiral 10^100 4.9 ms (7.0 before),
10^1000 5.5 (6.9), seahorse 33.5 (36.9); deep renders, auto against the best fixed 2^6 / 2^7 run32:
p153231 762 / 766 / 768, p6006 690 / 770 / 700, seahorse 33 / 33 / 35, spiral 3 / 3 / 4, 10^1000
4 / 4 / 5 - auto is the best of the two everywhere.  Worth a later look: at p6006, 2^8 run32 (669)
and 2^7 run64 (671) beat it, but 2^8 is 9 % slower at p153231 (833).

The interactive table's "pan step" timed the first pan of the run, which also compiles the
sample-copy shader (home 6.3 ms on the RTX, 84 ms on the software renderer against 17-23 ms for
the following ones); it is now timed after a warm-up pan (RTX: home 1.8 ms instead of 6.3).

*The test suite on the RTX.*  The suite had only run on the software renderer (CI); on the
user's GPU 6 of 145 failed:
- three video tests: "auto" picked hevc_nvenc, which refuses frames below its minimum size (the
  tests use 64x48 and 96x54).  "auto" now takes x264 below 256x144 (`video.HW_MIN_SIZE`).
- the GUI session's zoom out of the deep minibrot held the key for 40 frames: 0.67 s at 60 fps,
  one decade, where the check wants four (the software renderer's frames are longer).  The step
  now holds the key for at least 40 frames and until 5 decades out.  Then its "frames without skips" check (at most 3) failed with 8: the RTX shows ~176 frames
  while the worker builds the next tables in the same CPU time.  The limit is now 10 % of the
  frames (at least 3); the v3.11 lag it guards against was 49 of 67.  Next, the hold-zoom check wanted more than 5 frames that kept computing an unfinished picture
  ("in flight"); the RTX finishes most pictures within a frame and had exactly 5.  It now asks
  for at least one.  With these, the user's session on the RTX (report.json) passes every
  check of the test: rotation snap -45 deg, zoom out 15.05 -> 10.05 with 11 of 176 frames
  without skips, deep Julia sets in the explorer at 2^-50 / 2^-400 with 397 / 681 distinct
  values.  (One earlier failure of the rotation snap, -105 deg, did not come back: a real mouse
  or controller over the window can turn the view during the session.)
- `test_float_float_agrees_with_float64[minibrot_p6006]`: 96.5 % of float-float samples equal
  float64 (float32 84.5 %), the test wants 98 %.  Not caused by this round (shaders and that
  test's tables unchanged); the MPFR ground truth of float-float at the same minibrot passes on
  the RTX.  To be looked at: NVIDIA may contract float64 products into fma in perturb.comp,
  which shifts the float64 "reference" picture at a chaotic place.
- `test_deep_julia_stills_and_movies`: a flat picture at 10^-60.  The Julia tables of FrameRenderer
  had got 2^6 from the new rule; with 2^7 again the rerun gave the same picture (std 2.0548046676563256,
  equal to the last digit), so it is older than this round and specific to the GPU: the Julia
  accuracy tests to 10^-100 (their own render path) pass on the RTX.  Open; needs a session on
  the RTX.  The explorer's deep Julia sets and the GUI session's deep Julia still show detail on
  the RTX, so the flat picture is specific to that test's view (c = -0.8 + 0.156i, 10^-60, next to
  the repelling fixed point).  The Julia tables keep 2^7 until it is measured for them.

Not changed, with the reason: float64 and float-float get no BLA skips at the seahorse 10^12
(0 against 10.7 million in float32): their tolerance 2^-45 makes the validity radii smaller than
the deltas there.  A looser tolerance would lose the precision the modes are chosen for.

README animations (`tools/readme_media.py`, `tools/readme_tour.py`): zoom movies rendered as
fast key movies (2x2 samples) and turned into GIFs with ffmpeg's two-pass palette (palettegen
with per-frame statistics, ordered dithering, which compresses better than error diffusion:
a zoom changes every pixel of every frame, so GIFs grow quickly; 360 pixels wide, 12 fps, 6-7 MB
each).  The tour runs the explorer with a scripted session (the selftest machinery) and keeps
every displayed frame, but only the last frame while a picture refines, so the software
renderer gives the same animation as a GPU.

## 6. Possible next steps

* BLA for the Burning Ship (skips are valid only while the deltas change no signs).
* Make float-float or float64 the precise mode by default per GPU, from the bench.
* Multi-threaded or GPU reference orbits for > 10 000 decimal digits (mandelbrot-webgpu
  computes them in WGSL with fixed-point limbs; ours take 3-490 ms for the built-in places).
* Imagina-style "LA with stages" (Zhuoran): approximations that also cover the rebases at
  the near-periods of the reference, so pixels that have left the reference near a deep
  minibrot keep skipping.  Second-order BLA already took the steps per pixel near the
  period-153231 minibrot from ~9,300 to ~2,200; the published description of LA is forum
  posts and AGPL code, so it needs a clean derivation first.
* Cheaper BLA skips.  With the skip counts that `bench --bla` now prints (software renderer,
  320x180, period-6006 minibrot): second order at 2^4 takes 16,141 steps incl. 704 skips per
  pixel in 15.7 s, at 2^6 27,953 steps incl. 5 skips in 13.6 s - a skip costs about as much as 24
  plain iterations while it replaces about 17.  The skip does all its arithmetic in floatexp
  (five products, four normalised additions) and leaves the fast run of plain iterations; a
  plain-float path for skips whose exponents are in range (as the plain iteration has) could
  make short skips pay again.  Also to measure: waiting longer after a failed lookup, and a
  compact array of the lowest-level radii for the lookup.
* Reference orbits in compiled code: 3-10 us per iteration in Python + gmpy2 (bench); a place
  that needs 10 million iterations waits a minute for its reference.
* Centre-first sample order inside each progressive level (the centre sharpens first).

## 7. Further ideas

* Other formulas (higher powers, other abs variations).
