"""Mandelscope sequence scripts: a small, line based language for camera paths.

The language is deliberately simple so that people *and* language models can
write and tweak it.  See SCRIPT_HELP (also shown in the app) for the reference.
"""
from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field

from .mp import View, LOG10_2
from .sequence import Keyframe, Sequence
from .formulas import parse_formula, home_view, FORMULA_NAMES

SCRIPT_HELP = r"""# Mandelscope sequence script - reference
#
# One command per line ('#' starts a comment, ';' separates commands on a line).
# Times: 20s, 1.5min, 500ms or a bare number (seconds).  Angles in degrees.
# Zoom factors: 1e6, 64x, 2^20, "20 octaves" (= 2^20).  Magnification: 1e50 or 10^50
# (1 = the whole Mandelbrot set visible).  Easing: linear | smooth | in | out.
#
# --- settings (optional) -----------------------------------------------------
fps 60                       # frames per second for rendering
size 1920x1080               # suggested video size
smooth on                    # smooth zoom speed through keyframes (default on)
iterations auto              # or a number: iterations 20000
look Nebula                  # colour look (Looks list in the Colour tab)
# fractal ship               # the Burning Ship (default: the fractal of the location / mandelbrot)
#
# --- where to start ----------------------------------------------------------
start home                   # whole set     | start here (current view)
# start location "spiral_1e-100"             # a saved / built-in location
# start at re=-0.7436438870371587 im=0.1318259042053119 zoom=1e6 angle=0
#
# --- shots: each one adds time -------------------------------------------------
zoom in 1e6 over 20s                    # factor; options: spin <turns>, ease <e>
zoom to 1e12 over 15s spin 1 ease smooth   # absolute magnification
hold 2s                                 # pause
rotate 90 over 3s                       # relative; 'rotate to 0 over 2s' is absolute
spin 1 over 5s                          # full turns
# zoom to location "seahorse_1e-12" over 30s   # fly to a location (centre, depth, angle)
# zoom to re=-0.7453 im=0.1127 zoom=1e3 over 10s  # fly to coordinates
# pan to re=-0.74 im=0.13 over 4s
# pan by 0.5 0 over 3s                  # in screen heights (x right, y down)
animate offset by 1 over 10s            # colour parameter: offset, density, auto_cycles,
                                        # stripes, glow, slope_strength, de_strength ...
# look Ember                             # switch colour look from here on
zoom out 1e12 over 10s
#
# --- exact keyframes (what "Script from timeline" produces) -----------------------
# key t=12.5 re=-0.74364 im=0.13182 zoom=1e9 angle=30 iter=5000 ease=smooth
#   (on a key line, ease= is the easing of the segment that starts at that key)
"""

EASE_ALIASES = {"linear": "linear", "smooth": "smooth", "in": "ease-in", "out": "ease-out",
                "ease-in": "ease-in", "ease-out": "ease-out", "easein": "ease-in", "easeout": "ease-out"}
COLOR_PARAMS = {"offset", "density", "auto_cycles", "cycles", "stripes", "stripe_freq", "glow",
                "slope_strength", "light", "de_strength", "edges", "de_scale", "light_angle",
                "light_height", "pattern_strength", "pattern_freq", "de_density", "cycle_speed"}
PARAM_ALIASES = {"cycles": "auto_cycles", "light": "slope_strength", "edges": "de_strength"}


class ScriptError(Exception):
    pass


@dataclass
class CompileResult:
    sequence: Sequence | None
    errors: list = field(default_factory=list)       # (line number, message)
    warnings: list = field(default_factory=list)
    fps: float = 60.0
    size: tuple | None = None
    look: str | None = None
    formula: str = "mandelbrot"

    @property
    def ok(self) -> bool:
        return not self.errors and self.sequence is not None


# ------------------------------------------------------------------ values
def parse_duration(tok: str) -> float:
    t = tok.strip().lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*(ms|s|sec|secs|seconds?|m|min|mins|minutes?)?", t)
    if not m:
        raise ScriptError(f"bad duration '{tok}' (use e.g. 20s, 1.5min, 500ms)")
    v = float(m.group(1))
    unit = m.group(2) or "s"
    if unit == "ms":
        v /= 1000.0
    elif unit.startswith("m") and unit != "ms":
        v *= 60.0
    if v < 0:
        raise ScriptError("duration must be positive")
    return v


def parse_number(tok: str) -> float:
    t = tok.strip().lower().replace("_", "")
    m = re.fullmatch(r"10\^([+-]?[0-9.]+)", t)
    if m:
        return 10.0 ** float(m.group(1))
    m = re.fullmatch(r"2\^([+-]?[0-9.]+)", t)
    if m:
        return 2.0 ** float(m.group(1))
    try:
        return float(t)
    except ValueError:
        raise ScriptError(f"expected a number, got '{tok}'")


def parse_log10_magnitude(tok: str) -> float:
    """log10 of a (possibly astronomically large) magnitude like 1e-300 / 1e5000 / 10^800."""
    t = tok.strip().lower().replace("_", "").rstrip("x")
    m = re.fullmatch(r"10\^([+-]?[0-9.]+)", t)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"2\^([+-]?[0-9.]+)", t)
    if m:
        return float(m.group(1)) * LOG10_2
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)(?:e([+-]?[0-9]*\.?[0-9]+))?", t)
    if not m:
        raise ScriptError(f"expected a magnitude like 1e50 or 10^50, got '{tok}'")
    mant = float(m.group(1))
    if mant <= 0:
        raise ScriptError("magnitude must be positive")
    return math.log10(mant) + (float(m.group(2)) if m.group(2) else 0.0)


def parse_factor_log2(words: list[str]) -> tuple[float, int]:
    """Zoom factor -> log2(factor); returns (value, words consumed)."""
    if not words:
        raise ScriptError("missing zoom factor (e.g. 1e6, 64x, 20 octaves)")
    w0 = words[0].lower()
    if len(words) > 1 and words[1].lower() in ("octave", "octaves", "oct"):
        return parse_number(w0), 2
    m = re.fullmatch(r"([0-9.]+)(oct|octaves?)", w0)
    if m:
        return float(m.group(1)), 1
    return parse_log10_magnitude(w0) / LOG10_2, 1


def _kv(words: list[str]) -> dict:
    out = {}
    for w in words:
        if "=" not in w:
            raise ScriptError(f"expected key=value, got '{w}'")
        k, v = w.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def mag_to_log2r(log10_mag: float) -> float:
    return 1.0 - log10_mag / LOG10_2


# ------------------------------------------------------------------ compiler
class _Compiler:
    def __init__(self, resolve_location, auto_iter, current_view, base_colors: dict, looks: dict,
                 current_formula: str = "mandelbrot"):
        self.resolve_location = resolve_location
        self.current_formula = current_formula
        self.formula: str | None = None       # set by 'fractal', or by the first place used
        self.auto_iter = auto_iter
        self.current_view = current_view
        self.colors = dict(base_colors)
        self.looks = looks
        self.iter_mode = "auto"
        self.iter_fixed = None
        self.keys: list[Keyframe] = []
        self.t = 0.0
        self.view: View | None = None
        self.res = CompileResult(None)
        self.smooth = True

    # -- helpers
    def iters_for(self, v: View) -> int:
        if self.iter_mode == "fixed" and self.iter_fixed:
            return int(self.iter_fixed)
        return int(self.auto_iter(v))

    def push(self, v: View, dur: float, ease_name: str = "linear", absolute_t: float | None = None):
        if self.view is None:
            raise ScriptError("no starting point: use 'start home' (or start here / location / at)")
        if self.keys:
            self.keys[-1].ease = ease_name
        self.t = absolute_t if absolute_t is not None else self.t + dur
        v.max_iter = v.max_iter if self.iter_mode == "keep" else self.iters_for(v)
        self.keys.append(Keyframe(v, self.t, "linear", dict(self.colors)))
        self.view = v

    def start(self, v: View):
        if self.keys:
            raise ScriptError("'start' must come before the first shot")
        v.max_iter = self.iters_for(v) if not getattr(v, "_fixed_iter", False) else v.max_iter
        self.view = v
        self.keys.append(Keyframe(v, 0.0, "linear", dict(self.colors)))

    def options(self, words: list[str]) -> tuple[float, str, float]:
        """Parse 'over <dur>' / 'in <dur>' / 'spin <turns>' / 'ease <e>' options."""
        dur, ease_name, spin = None, "linear", 0.0
        i = 0
        while i < len(words):
            w = words[i].lower()
            if w in ("over", "in", "for", "during") and i + 1 < len(words):
                dur = parse_duration(words[i + 1])
                i += 2
            elif w == "spin" and i + 1 < len(words):
                spin = parse_number(words[i + 1])
                i += 2
                if i < len(words) and words[i].lower() in ("turn", "turns"):
                    i += 1
            elif w == "ease" and i + 1 < len(words):
                e = words[i + 1].lower()
                if e not in EASE_ALIASES:
                    raise ScriptError(f"unknown easing '{words[i + 1]}' (linear, smooth, in, out)")
                ease_name = EASE_ALIASES[e]
                i += 2
            elif w in EASE_ALIASES and w != "in":
                ease_name = EASE_ALIASES[w]
                i += 1
            else:
                raise ScriptError(f"unexpected '{words[i]}' (options: over <time>, spin <turns>, ease <e>)")
        if dur is None:
            raise ScriptError("missing duration: add 'over <time>', e.g. 'over 10s'")
        return dur, ease_name, spin

    def use_formula(self, f: str, what: str):
        if self.formula is None:
            self.formula = f
        elif f != self.formula:
            raise ScriptError(f"{what} is in {FORMULA_NAMES[f]}, this script is for {FORMULA_NAMES[self.formula]}")

    def location(self, name: str) -> View:
        v = self.resolve_location(name)
        if v is None:
            raise ScriptError(f"unknown location '{name}' (Library tab lists the names)")
        self.use_formula(getattr(v, "formula", "mandelbrot"), f"'{name}'")
        return v.copy()

    def apply_look(self, name: str):
        key = next((k for k in self.looks if k.lower() == name.lower()), None)
        if key is None:
            raise ScriptError(f"unknown look '{name}'. Known: {', '.join(self.looks)}")
        self.colors.update(self.looks[key])
        self.colors["look"] = key
        self.res.look = self.res.look or key
        if self.keys:
            self.keys[-1].color = dict(self.colors)

    # -- commands
    def run_line(self, words: list[str]):
        cmd = words[0].lower()
        rest = words[1:]
        if cmd == "fps":
            self.res.fps = parse_number(rest[0])
        elif cmd == "size":
            m = re.fullmatch(r"(\d+)\s*[x*]\s*(\d+)", "".join(rest).lower())
            if not m:
                raise ScriptError("size must look like 1920x1080")
            self.res.size = (int(m.group(1)), int(m.group(2)))
        elif cmd == "smooth":
            self.smooth = (not rest) or rest[0].lower() in ("on", "yes", "true", "1")
        elif cmd in ("iterations", "iter", "iters"):
            if not rest:
                raise ScriptError("iterations auto | iterations <number>")
            if rest[0].lower() == "auto":
                self.iter_mode = "auto"
            else:
                self.iter_mode = "fixed"
                self.iter_fixed = int(parse_number(rest[0]))
        elif cmd == "look":
            self.apply_look(" ".join(rest))
        elif cmd in ("fractal", "formula", "set_fractal"):
            f = parse_formula(" ".join(rest))
            if f is None:
                raise ScriptError("fractal mandelbrot | fractal ship (the Burning Ship)")
            if self.keys:
                raise ScriptError("'fractal' must come before 'start'")
            if self.formula is not None and f != self.formula:
                raise ScriptError(f"this script is already for {FORMULA_NAMES[self.formula]}")
            self.formula = f
        elif cmd == "palette":
            self.colors["palette"] = " ".join(rest).strip('"')
            if self.keys:
                self.keys[-1].color = dict(self.colors)
        elif cmd == "set":
            if len(rest) < 2:
                raise ScriptError("set <parameter> <value>")
            p = PARAM_ALIASES.get(rest[0].lower(), rest[0].lower())
            self.colors[p] = parse_number(rest[1])
            if self.keys:
                self.keys[-1].color = dict(self.colors)
        elif cmd == "start":
            self.cmd_start(rest)
        elif cmd == "zoom":
            self.cmd_zoom(rest)
        elif cmd in ("fly", "goto", "go"):
            if rest and rest[0].lower() == "to":
                rest = rest[1:]
            self.cmd_zoom(["to", "location"] + rest)
        elif cmd == "pan":
            self.cmd_pan(rest)
        elif cmd in ("rotate", "turn"):
            self.cmd_rotate(rest)
        elif cmd == "spin":
            if not rest:
                raise ScriptError("spin <turns> over <time>")
            turns = parse_number(rest[0])
            r2 = rest[1:]
            if r2 and r2[0].lower() in ("turn", "turns"):
                r2 = r2[1:]
            self.cmd_rotate([str(turns * 360.0)] + r2)
        elif cmd in ("hold", "wait", "pause"):
            if not rest:
                raise ScriptError("hold <time>")
            dur = parse_duration(rest[0] if rest[0].lower() not in ("for",) else rest[1])
            self.push(self.view.copy(), dur)
        elif cmd in ("animate", "fade", "color", "colour"):
            self.cmd_animate(rest)
        elif cmd == "key":
            self.cmd_key(rest)
        else:
            raise ScriptError(f"unknown command '{words[0]}'")

    def cmd_start(self, rest):
        if not rest:
            raise ScriptError("start home | start here | start location <name> | start at re=.. im=.. zoom=..")
        what = rest[0].lower()
        if what == "home":
            v = home_view(self.formula or "mandelbrot")
            self.use_formula(self.formula or "mandelbrot", "home")
        elif what == "here":
            if self.current_view is None:
                raise ScriptError("'start here' needs the app's current view")
            self.use_formula(self.current_formula, "the current view")
            v = self.current_view.copy()
        elif what == "location":
            v = self.location(" ".join(rest[1:]).strip('"'))
        elif what == "at":
            v = self.view_from_kv(_kv(rest[1:]))
        else:
            v = self.location(" ".join(rest).strip('"'))
        self.start(v)

    def view_from_kv(self, kv: dict, base: View | None = None) -> View:
        base = base or self.view or View.home()
        re_s = kv.get("re", None)
        im_s = kv.get("im", None)
        if re_s is None or im_s is None:
            if base is None:
                raise ScriptError("need re=... and im=...")
            re_s = re_s or format(base.re, ".40e")
            im_s = im_s or format(base.im, ".40e")
        if "zoom" in kv:
            log2r = mag_to_log2r(parse_log10_magnitude(kv["zoom"]))
        elif "log2r" in kv:
            log2r = parse_number(kv["log2r"])
        else:
            log2r = base.log2r
        angle = math.radians(parse_number(kv["angle"])) if "angle" in kv else base.angle
        v = View.from_strings(re_s, im_s, log2r, angle, 1000)
        if "iter" in kv or "iterations" in kv:
            v.max_iter = int(parse_number(kv.get("iter", kv.get("iterations"))))
            v._fixed_iter = True
        return v

    def cmd_zoom(self, rest):
        if not rest:
            raise ScriptError("zoom in <factor> | zoom out <factor> | zoom to <magnification> | zoom to location <name>")
        mode = rest[0].lower()
        v0 = self.view
        if v0 is None:
            raise ScriptError("no starting point: use 'start home' first")
        if mode in ("in", "out"):
            l2, n = parse_factor_log2(rest[1:])
            opts = rest[1 + n:]
            dur, e, spin = self.options(opts)
            v = v0.copy()
            v.log2r = v0.log2r - l2 if mode == "in" else v0.log2r + l2
            v.angle = v0.angle + spin * 2 * math.pi
            self.push(v, dur, e)
        elif mode == "to":
            if len(rest) > 1 and "=" in rest[1]:
                # zoom to re=.. im=.. zoom=.. [angle=..] over <t>
                kvs = [w for w in rest[1:] if "=" in w]
                opts = [w for w in rest[1:] if "=" not in w]
                kv = _kv(kvs)
                v = self.view_from_kv(kv, v0)
                dur, e, spin = self.options(opts)
                if "angle" not in kv:
                    v.angle = v0.angle
                v.angle += spin * 2 * math.pi
                self.push(v, dur, e)
            elif len(rest) > 1 and rest[1].lower() in ("location", "loc"):
                # name may be quoted or several words until 'over'/'in'
                j = 2
                name_words = []
                while j < len(rest) and rest[j].lower() not in ("over", "in", "for", "spin", "ease"):
                    name_words.append(rest[j])
                    j += 1
                dur, e, spin = self.options(rest[j:])
                v = self.location(" ".join(name_words).strip('"'))
                v.angle += spin * 2 * math.pi
                self.push(v, dur, e)
            else:
                lm = parse_log10_magnitude(rest[1])
                dur, e, spin = self.options(rest[2:])
                v = v0.copy()
                v.log2r = mag_to_log2r(lm)
                v.angle = v0.angle + spin * 2 * math.pi
                self.push(v, dur, e)
        else:
            raise ScriptError(f"zoom {rest[0]}? use zoom in / zoom out / zoom to")

    def cmd_pan(self, rest):
        if not rest:
            raise ScriptError("pan to re=.. im=.. over <t> | pan by <dx> <dy> over <t>")
        mode = rest[0].lower()
        if mode == "to":
            kvs = [w for w in rest[1:] if "=" in w]
            opts = [w for w in rest[1:] if "=" not in w]
            v = self.view_from_kv(_kv(kvs), self.view)
            v.log2r, v.angle = self.view.log2r, self.view.angle
            dur, e, _ = self.options(opts)
            self.push(v, dur, e)
        elif mode == "by":
            if len(rest) < 3:
                raise ScriptError("pan by <dx> <dy> over <time>  (in screen heights)")
            dx, dy = parse_number(rest[1]), parse_number(rest[2])
            dur, e, _ = self.options(rest[3:])
            H = 1000
            v = self.view.offset(dx * H, dy * H, H)
            self.push(v, dur, e)
        else:
            raise ScriptError("pan to ... | pan by ...")

    def cmd_rotate(self, rest):
        if not rest:
            raise ScriptError("rotate <degrees> over <t> | rotate to <degrees> over <t>")
        v = self.view.copy()
        if rest[0].lower() == "to":
            target = math.radians(parse_number(rest[1]))
            # shortest way round
            d = (target - v.angle + math.pi) % (2 * math.pi) - math.pi
            v.angle = v.angle + d
            dur, e, _ = self.options(rest[2:])
        else:
            v.angle = v.angle + math.radians(parse_number(rest[0]))
            dur, e, _ = self.options(rest[1:])
        self.push(v, dur, e)

    def cmd_animate(self, rest):
        if len(rest) < 3:
            raise ScriptError("animate <parameter> to|by <value> over <time>")
        p = PARAM_ALIASES.get(rest[0].lower(), rest[0].lower())
        if p not in COLOR_PARAMS and p not in PARAM_ALIASES.values():
            raise ScriptError(f"unknown parameter '{rest[0]}'. Try: {', '.join(sorted(COLOR_PARAMS))}")
        how = rest[1].lower()
        val = parse_number(rest[2])
        dur, e, _ = self.options(rest[3:])
        cur = float(self.colors.get(p, 0.0) or 0.0)
        self.colors[p] = val if how == "to" else cur + val
        self.push(self.view.copy(), dur, e)

    def cmd_key(self, rest):
        kv = _kv(rest)
        if "t" not in kv:
            raise ScriptError("key needs t=<seconds>")
        t = parse_duration(kv.pop("t"))
        e = EASE_ALIASES.get(kv.pop("ease", "linear").lower(), "linear")
        v = self.view_from_kv(kv, self.view)
        if self.view is None:
            if t != 0:
                self.res.warnings.append((0, "first key is not at t=0; it starts the sequence"))
            self.start(v)
            self.keys[-1].time = t
            self.keys[-1].ease = e
            self.t = t
            return
        if t <= self.t:
            raise ScriptError(f"key times must increase (t={t} after {self.t})")
        # on a key line, ease= is the easing of the segment that STARTS at this key
        prev_ease = self.keys[-1].ease if self.keys else "linear"
        if getattr(v, "_fixed_iter", False):
            mode, self.iter_mode = self.iter_mode, "keep"
            self.push(v, 0.0, prev_ease, absolute_t=t)
            self.iter_mode = mode
        else:
            self.push(v, 0.0, prev_ease, absolute_t=t)
        self.keys[-1].ease = e


def split_commands(text: str) -> list[tuple[int, list[str]]]:
    out = []
    for ln, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0]
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                words = shlex.split(part, posix=True)
            except ValueError as e:
                out.append((ln, ["__error__", f"cannot read line: {e}"]))
                continue
            out.append((ln, words))
    return out


def compile_script(text: str, resolve_location=None, auto_iter=None, current_view: View | None = None,
                   base_colors: dict | None = None, looks: dict | None = None,
                   current_formula: str = "mandelbrot") -> CompileResult:
    """Compile a script into a Sequence (keyframes with views + colour parameters)."""
    if resolve_location is None:
        resolve_location = lambda name: None
    if auto_iter is None:
        auto_iter = lambda v: int(1000 + 400 * max(0.0, v.zoom_magnification_log10()))
    if looks is None:
        from .palettes import LOOKS
        looks = LOOKS
    c = _Compiler(resolve_location, auto_iter, current_view, base_colors or {}, looks, current_formula)
    for ln, words in split_commands(text):
        try:
            if words[0] == "__error__":
                raise ScriptError(words[1])
            c.run_line(words)
        except ScriptError as e:
            c.res.errors.append((ln, str(e)))
        except (IndexError, ValueError) as e:
            c.res.errors.append((ln, f"incomplete or invalid command ({e})"))
    if not c.keys and not c.res.errors:
        c.res.errors.append((0, "the script has no 'start' and no shots"))
    if len(c.keys) == 1 and not c.res.errors:
        c.res.warnings.append((0, "only a starting point - add shots like 'zoom in 1e6 over 20s'"))
    if c.keys:
        seq = Sequence(c.keys, c.res.fps, smooth=c.smooth)
        seq.meta = {"size": list(c.res.size) if c.res.size else None, "look": c.res.look}
        c.res.formula = c.formula or "mandelbrot"
        if c.res.formula != "mandelbrot":
            seq.meta["formula"] = c.res.formula
        c.res.sequence = seq
    return c.res


def sequence_to_script(seq: Sequence, digits: int | None = None) -> str:
    """Exact script for a sequence (one 'key' line per keyframe)."""
    lines = ["# generated from the timeline - every keyframe is exact",
             f"fps {seq.fps:g}", f"smooth {'on' if seq.smooth else 'off'}"]
    if seq.meta.get("formula", "mandelbrot") != "mandelbrot":
        lines.append(f"fractal {seq.meta['formula']}")
    for k in seq.keyframes:
        d = k.view.to_dict()
        z = k.view.zoom_magnification_log10()
        lines.append(f"key t={k.time:.3f}s re={d['re']} im={d['im']} zoom=10^{z:.6f} "
                     f"angle={math.degrees(k.view.angle):.4f} iter={k.view.max_iter} ease={k.ease}")
    return "\n".join(lines) + "\n"


def ai_prompt(script: str, current_view: View | None, locations: list[str], looks: list[str]) -> str:
    """Text to paste into an AI assistant to write or edit a script."""
    here = ""
    if current_view is not None:
        d = current_view.to_dict()
        here = (f"\nThe current view in the app is: re={d['re'][:60]} im={d['im'][:60]} "
                f"zoom=1e{current_view.zoom_magnification_log10():.2f} "
                f"angle={math.degrees(current_view.angle):.1f} (use 'start here' to start from it).\n")
    return (
        "You are helping me write a camera script for Mandelscope, a Mandelbrot deep-zoom video tool.\n"
        "Write ONLY the script, following this reference exactly:\n\n" + SCRIPT_HELP +
        "\nAvailable looks: " + ", ".join(looks) +
        "\nAvailable locations: " + ", ".join(locations) + "\n" + here +
        "\nMy current script is:\n```\n" + (script.strip() or "(empty)") + "\n```\n"
        "\nWhat I want: <describe the movie here>\n")
