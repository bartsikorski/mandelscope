"""Camera and work control without graphics: zoom spring, frame pacing, automatic motion
resolution, the automatic iteration limit and the game-controller mapping (unit-tested on
their own)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

ZOOM_OMEGA = 20.0         # wheel / key zoom spring: 1/s (95 % of a notch in ~0.24 s)


def spring_step(rem: float, vel: float, dt: float, w: float):
    """Critically damped spring, exact over dt: remaining distance `rem` to the target and
    velocity `vel` towards it -> (rem, vel) after dt."""
    e = math.exp(-w * dt)
    b = -vel + w * rem
    rem1 = (rem + b * dt) * e
    vel1 = -(b - w * (rem + b * dt)) * e
    return rem1, vel1


UNDECIDED_OK = 0.003      # share of samples that may stay undecided at the iteration limit
DETAIL_MIN = 0.0002       # share that must escape in the top octave for a doubling to pay off
MAX_BOOST = 256.0


def _wants_more(esc, ins, und, N) -> tuple[bool, str]:
    """Would doubling the iteration limit pay off?  esc[k] / ins[k]: samples that escaped / were
    recognised as inside in the k-th octave below the limit, und: undecided samples.

    The top octave shows what the last doubling did.  If it resolved almost nothing the limit is
    far too low (deep zooms: nothing escapes yet).  Otherwise the escapes per octave predict how
    many undecided samples the next doubling would turn into outside colours: when that is a
    fair share of them the limit cuts off outside detail; when it is not, the undecided samples
    are points converging slowly to a cycle (next to the edge of a component: their number only
    halves with every doubling, and they are shown like inside points anyway)."""
    if und <= UNDECIDED_OK * N:
        return False, "resolved"
    resolved = esc[0] + ins[0]
    if resolved < 0.1 * (und + resolved):
        return True, "far"
    decay = min(1.0, esc[0] / esc[1]) if esc[1] > 0 else 1.0
    predicted = esc[0] * decay                   # escapes expected in the next octave
    if predicted >= 0.1 * und and predicted >= DETAIL_MIN * N:
        return True, "detail"
    return False, "inside"


def iteration_advice(st: dict, boost: float, may_lower: bool = True) -> tuple[int, str]:
    """+1: double the iteration limit, -1: halve it, 0: keep (from Field.iteration_octaves)."""
    esc, ins, und, N = st["esc"], st["int"], st["und"], max(1, st["N"])
    up, why = _wants_more(esc, ins, und, N)
    if up:
        return (1, why) if boost < MAX_BOOST else (0, why + " (at the cap)")
    if boost > 1.0 and may_lower:
        # one octave lower: the top octave's samples would be undecided; lower if that limit
        # would not ask for more either (no flip-flop) - only esc[0] samples change colour
        lower_up, _ = _wants_more(esc[1:] + [0], ins[1:] + [0], und + esc[0] + ins[0], N)
        if not lower_up:
            return -1, "not needed"
    return 0, why


class FramePacer:
    """Camera time steps for an even display.

    The picture changes once per display refresh, but the moments the frames start jitter by
    several ms (the v3.7 study measured 22.7 / 12.4 / 15.0 ms in turn at 60 Hz).  Moving the
    camera by those raw intervals gives uneven steps on an even display: judder.  The step is
    the median of the last intervals plus a share of the time not yet handed out, so the camera
    clock never drifts from the real one; a real hitch (much longer than usual) passes at once."""

    def __init__(self, n: int = 5, gain: float = 0.25):
        self.win: deque = deque(maxlen=n)
        self.gain = gain
        self.debt = 0.0          # real time not yet given to the camera (may be negative)

    def step(self, raw: float) -> float:
        med = float(np.median(self.win)) if len(self.win) >= 3 else None
        self.win.append(raw)
        if med is None or raw > 2.5 * med or raw < 0.0:
            self.debt = 0.0
            return raw
        out = max(0.25 * med, med + self.gain * self.debt)
        self.debt += raw - out
        if abs(self.debt) > 0.1:
            out += self.debt
            self.debt = 0.0
        return out


def auto_motion_level(zooming: bool, t_full: float, level: int, max_level: int = 1,
                      above: float = 0.20, below: float = 0.10) -> tuple[float, int]:
    """(field scale while moving, new level): level L computes zooms and turns at 1/2**L of the
    resolution (4**L times fewer samples).  A level is left for a coarser one when a picture at
    it would take more than `above` seconds, and for the finer one when that would take less
    than `below` (hysteresis).  The GPU uses levels 0-1, CPU mode 0-2."""
    level = max(0, min(level, max_level))
    while level < max_level and t_full / 4.0 ** level > above:
        level += 1
    while level > 0 and t_full / 4.0 ** (level - 1) < below:
        level -= 1
    return (0.5 ** level if zooming else 1.0), level


def auto_motion_scale(zooming: bool, t_full: float, half: bool, above: float = 0.20,
                      below: float = 0.10) -> tuple[float, bool]:
    """(field scale while moving, new half-resolution state) - with hysteresis between
    `below` and `above` seconds per full-resolution picture."""
    scale, level = auto_motion_level(zooming, t_full, int(half), 1, above, below)
    return scale, bool(level)


# ------------------------------------------------------------------ game controllers
# GLFW's standard gamepad layout (SDL names), with the PlayStation names used in the app
PAD_BUTTONS = ("cross", "circle", "square", "triangle", "l1", "r1", "create", "options", "ps",
               "l3", "r3", "up", "right", "down", "left")
PAD_XBOX = {"cross": "A", "circle": "B", "square": "X", "triangle": "Y", "l1": "LB", "r1": "RB",
            "create": "View", "options": "Menu", "ps": "Xbox", "l3": "LS", "r3": "RS",
            "l2": "LT", "r2": "RT", "up": "D-pad up", "down": "D-pad down", "left": "D-pad left",
            "right": "D-pad right"}
# what each button does (shown in the help card and the Explore tab)
PAD_ACTIONS = (
    ("left stick", "move"),
    ("R2 / L2", "zoom in / out (press harder: faster)"),
    ("right stick", "up / down: zoom, left / right: turn"),
    ("R1 / L1 (hold)", "faster / finer"),
    ("cross", "find the minibrot at the centre"),
    ("circle", "back"),
    ("square", "Julia set preview of the centre (in a Julia set: back)"),
    ("triangle", "add a keyframe"),
    ("D-pad left / right", "previous / next look"),
    ("D-pad up / down", "more / fewer iterations"),
    ("L3 / R3", "home / rotation back to 0\u00b0"),
    ("options", "hide / show the panels"),
    ("create", "screenshot"),
    ("PS", "help"),
)


@dataclass
class PadState:
    """One reading of a controller: GLFW axes (left x, left y, right x, right y, L2, R2; sticks
    -1..1 with y down, triggers -1 released .. 1 pressed) and the 15 buttons of PAD_BUTTONS."""
    axes: tuple = (0.0, 0.0, 0.0, 0.0, -1.0, -1.0)
    buttons: tuple = (False,) * 15

    @staticmethod
    def make(lx=0.0, ly=0.0, rx=0.0, ry=0.0, l2=0.0, r2=0.0, pressed=()) -> "PadState":
        """Test helper: triggers 0..1, buttons by name."""
        return PadState((lx, ly, rx, ry, 2 * l2 - 1, 2 * r2 - 1),
                        tuple(n in pressed for n in PAD_BUTTONS))


@dataclass
class PadInput:
    """What the camera takes from a controller in one frame."""
    pan: tuple = (0.0, 0.0)         # -1..1, x right, y down (the view moves that way)
    zoom: float = 0.0               # -1..1, + = in
    rot: float = 0.0                # -1..1, + = like the E key
    fast: bool = False
    fine: bool = False
    pressed: set = field(default_factory=set)
    active: bool = False            # any stick, trigger or button in use


def _stick(x: float, y: float, dead: float) -> tuple:
    """Radial dead zone, then a squared response (fine control near the centre)."""
    m = math.hypot(x, y)
    if m <= dead:
        return 0.0, 0.0
    u = min(1.0, (m - dead) / max(1e-6, 1.0 - dead))
    k = u * u / m
    return x * k, y * k


def _axis(v: float, dead: float) -> float:
    a = abs(v)
    if a <= dead:
        return 0.0
    u = min(1.0, (a - dead) / max(1e-6, 1.0 - dead))
    return math.copysign(u * u, v)


def _trigger(v: float, dead: float = 0.06) -> float:
    t = 0.5 * (v + 1.0)                    # GLFW: -1 released .. 1 pressed
    if t <= dead:
        return 0.0
    u = min(1.0, (t - dead) / (1.0 - dead))
    return u * u


def pad_controls(st: PadState | None, prev: tuple | None, dead: float = 0.15,
                 invert_zoom: bool = False) -> PadInput:
    """Map a controller reading to camera input.  `prev` are the previous frame's buttons (for
    'pressed this frame')."""
    if st is None:
        return PadInput()
    ax = list(st.axes) + [0.0] * (6 - len(st.axes))
    pan = _stick(ax[0], ax[1], dead)
    rx, ry = _axis(ax[2], dead), _axis(ax[3], dead)
    zoom_stick = ry if invert_zoom else -ry          # stick up (negative y) = zoom in
    zoom = max(-1.0, min(1.0, _trigger(ax[5]) - _trigger(ax[4]) + zoom_stick))
    b = tuple(st.buttons) + (False,) * (15 - len(st.buttons))
    pb = tuple(prev) + (False,) * (15 - len(prev)) if prev is not None else (False,) * 15
    pressed = {PAD_BUTTONS[i] for i in range(15) if b[i] and not pb[i]}
    held = {PAD_BUTTONS[i] for i in range(15) if b[i]}
    active = bool(pan != (0.0, 0.0) or zoom or rx or held)
    return PadInput(pan=pan, zoom=zoom, rot=rx, fast="r1" in held, fine="l1" in held,
                    pressed=pressed, active=active)
