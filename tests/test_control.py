"""Camera / work control without graphics: zoom spring, frame pacing, automatic motion
resolution, automatic iteration limit."""
import math

import numpy as np


def test_wheel_zoom_spring_is_smooth():
    """Wheel zoom: a critically damped spring - a notch is done in ~0.25 s like before, the speed
    changes smoothly (no jump at each notch) and it never overshoots."""
    import math
    from mandelscope.control import spring_step, ZOOM_OMEGA

    def run(notches, spring):
        rem = vel = 0.0
        rates, t, nxt, n, dt = [], 0.0, 0.0, 0, 1 / 60
        while t < 1.5:
            if n < notches and t >= nxt - 1e-9:
                rem += 0.5
                n += 1
                nxt += 0.12
            if spring:
                r1, vel = spring_step(rem, vel, dt, ZOOM_OMEGA)
            else:
                r1 = rem * math.exp(-12 * dt)
            assert r1 >= -1e-9                             # no overshoot
            rates.append((rem - r1) / dt)
            rem, t = r1, t + dt
        return np.array(rates)
    one = run(1, True)
    assert 0.2 <= (np.argmax(np.cumsum(one) / 60 >= 0.475) + 1) / 60 <= 0.3
    old, new = run(6, False), run(6, True)
    rough = lambda r: np.mean(np.abs(np.diff(r))) / r.max()
    assert rough(new) < 0.8 * rough(old)
    assert np.abs(np.diff(new)).max() < 0.6 * np.abs(np.diff(old)).max()


def test_iteration_advice_rules():
    """Automatic iteration limit: double while it reveals outside detail or nothing resolves
    yet, stop when the undecided samples converge to cycles (edge of a component), lower again
    when the top octave is not needed - consistently (no flip-flop between two limits)."""
    from mandelscope.control import iteration_advice
    N = 100000

    def st(esc, ins, und):
        return {"esc": list(esc) + [0] * (8 - len(esc)), "int": list(ins) + [0] * (8 - len(ins)),
                "und": und, "N": N}
    assert iteration_advice(st([60000], [], 40000), 1.0)[0] == 1            # outside detail cut off
    assert iteration_advice(st([], [], N), 1.0) == (1, "far")              # deep zoom: nothing escapes yet
    assert iteration_advice(st([200], [], 100), 4.0)[0] == -1               # top octave hardly used
    assert iteration_advice(st([100, 3000], [], 100), 1.0)[0] == 0          # few undecided: done
    assert iteration_advice(st([], [], N), 256.0)[0] == 0                   # at the cap
    # edge of the main cardioid (measured): undecided and resolved halve with each doubling,
    # 93 % of what resolves is inside -> no raise at any limit, lowering down to the base
    esc0 = [2252, 4910, 8517, 9000, 7000, 5000, 3000, 2000]      # at the base limit (per octave)
    ins0 = [7345, 2363, 2505, 1000, 500, 200, 100, 50]
    for k in range(8):
        und = 31837 // 2 ** k
        # k doublings later: k new octaves on top, each with half the previous one's samples
        esc = [esc0[0] // 2 ** (k - j) for j in range(k)] + esc0
        ins = [ins0[0] // 2 ** (k - j) for j in range(k)] + ins0
        esc = esc[:7] + [sum(esc[7:])]
        ins = ins[:7] + [sum(ins[7:])]
        s = st(esc, ins, und)
        step, why = iteration_advice(s, 2.0 ** k)
        assert step == (0 if k == 0 else -1) and (k > 0 or why == "inside"), (k, step, why)
    # right after a raise that revealed detail, the limit is not lowered again
    s = st([3000, 9000], [100, 100], 5000)
    assert iteration_advice(s, 2.0)[0] == 1


def test_frame_pacer_evens_out_jittered_frame_starts():
    """Frames shown every 16.7 ms whose starts jitter (pattern measured on the RTX 5060 Ti): the
    camera steps become even, the camera clock does not drift, a real hitch passes at once."""
    from mandelscope.control import FramePacer
    p = FramePacer()
    raw = [0.0227, 0.0124, 0.0150] * 40
    out = [p.step(x) for x in raw]
    tail = np.array(out[15:])
    assert np.abs(tail - 1 / 60).max() < 0.0015                  # raw: up to 6 ms off
    assert abs(sum(out) - sum(raw)) < 0.02                         # no drift
    assert p.step(0.080) == 0.080                                  # a hitch is not spread out
    # steady variable rate (vsync off / VRR): follows it
    q = FramePacer()
    seq = [0.010 + 0.004 * math.sin(k / 10) for k in range(200)]
    o = [q.step(x) for x in seq]
    assert abs(sum(o) - sum(seq)) < 0.02 and np.abs(np.array(o[20:]) - np.array(seq[20:])).max() < 0.003


def test_automatic_motion_resolution():
    """Zooms / turns drop to half resolution only at heavy places (with hysteresis); pans never."""
    from mandelscope.control import auto_motion_scale
    half = False
    seq = []
    for t in (0.05, 0.15, 0.25, 0.15, 0.12, 0.08, 0.15):
        sc, half = auto_motion_scale(True, t, half)
        seq.append(sc)
    assert seq == [1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert auto_motion_scale(False, 0.5, True) == (1.0, True)        # panning: full resolution


def test_automatic_motion_resolution_cpu_levels():
    """CPU mode may go down to a quarter of the resolution; each step has its own hysteresis."""
    from mandelscope.control import auto_motion_level
    lv, seq = 0, []
    for t in (0.1, 0.5, 1.0, 3.5, 2.0, 1.5, 0.3, 0.15, 0.09, 0.05):
        sc, lv = auto_motion_level(True, t, lv, 2)
        seq.append(sc)
    assert seq == [1.0, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0]
    assert auto_motion_level(True, 5.0, 0, 1) == (0.5, 1)             # the GPU stops at half
    assert auto_motion_level(False, 5.0, 2, 2) == (1.0, 2)            # panning: full resolution


# ------------------------------------------------------------------ game controller
def test_pad_mapping_dead_zone_and_curves():
    from mandelscope.control import PadState, pad_controls
    assert not pad_controls(PadState.make(lx=0.1, ly=-0.1, rx=0.1, l2=0.04), None).active   # resting pad
    full = pad_controls(PadState.make(lx=1.0), None)
    assert full.pan == (1.0, 0.0) and full.active
    half = pad_controls(PadState.make(lx=0.575), None, dead=0.15)
    assert abs(half.pan[0] - 0.25) < 1e-6                      # squared response past the dead zone
    diag = pad_controls(PadState.make(lx=0.6, ly=0.8), None)     # radial: the direction is kept
    assert abs(diag.pan[1] / diag.pan[0] - 0.8 / 0.6) < 1e-9
    assert pad_controls(PadState.make(r2=1.0), None).zoom == 1.0
    assert pad_controls(PadState.make(l2=1.0), None).zoom == -1.0
    assert pad_controls(PadState.make(ry=-1.0), None).zoom == 1.0              # stick up = in
    assert pad_controls(PadState.make(ry=-1.0), None, invert_zoom=True).zoom == -1.0
    assert pad_controls(PadState.make(r2=1.0, ry=-1.0), None).zoom == 1.0      # clamped
    assert pad_controls(PadState.make(rx=-1.0), None).rot == -1.0
    assert pad_controls(None, None).pan == (0.0, 0.0)


def test_pad_buttons_fire_once_per_press():
    from mandelscope.control import PadState, pad_controls
    down = PadState.make(pressed=("cross", "r1"))
    first = pad_controls(down, None)
    again = pad_controls(down, down.buttons)
    assert first.pressed == {"cross", "r1"} and first.fast
    assert again.pressed == set() and again.fast and again.active      # held: still fast, no new press
    assert pad_controls(PadState.make(pressed=("l1",)), down.buttons).fine


class FakeGLFW:
    """Stands in for the GLFW library: joystick 3 is a gamepad until `unplug`."""
    def __init__(self):
        self.present = {3}
        self.calls = 0

    def glfwJoystickIsGamepad(self, j):
        return int(j in self.present)

    def glfwJoystickPresent(self, j):
        return int(j in self.present)

    def glfwGetGamepadName(self, j):
        return b"PS5 Controller"

    def glfwGetJoystickName(self, j):
        return b"DualSense Wireless Controller"

    def glfwGetGamepadState(self, j, ref):
        self.calls += 1
        if j not in self.present:
            return 0
        st = ref._obj
        st.axes[0] = 0.5
        st.axes[5] = 1.0
        st.buttons[1] = 1
        return 1

    def glfwGetCurrentContext(self):
        return 1234

    def glfwUpdateGamepadMappings(self, text):
        self.mappings = text
        return 1


def test_gamepad_reader_with_a_fake_glfw():
    import ctypes
    from mandelscope.gamepad import Gamepad, _GamepadState
    assert ctypes.sizeof(_GamepadState) == 40                 # GLFWgamepadstate: 15 bytes + pad + 6 floats
    lib = FakeGLFW()
    g = Gamepad(lib=lib)
    st = g.poll(now=0.0)
    assert g.name == "PS5 Controller" and g.jid == 3
    assert st.axes[0] == 0.5 and st.axes[5] == 1.0 and st.buttons[1] and not st.buttons[0]
    assert g.same_instance_as(1234) and not g.same_instance_as(99)
    lib.present.clear()                                       # unplugged
    assert g.poll(now=0.1) is None and g.name == ""
    assert g.poll(now=0.5) is None                            # no rescan before a second has passed
    lib.present.add(0)
    assert g.poll(now=1.2) is not None and g.jid == 0         # plugged in again


def test_gamepad_loads_the_glfw_library_of_imgui_bundle():
    """The GLFW library imgui_bundle ships is found and loads (no controller here)."""
    import imgui_bundle  # noqa: F401  (sets PYGLFW_LIBRARY)
    from mandelscope.gamepad import Gamepad, glfw_library_path
    assert glfw_library_path() is not None
    g = Gamepad()
    assert g.lib is not None, g.error
    assert g.poll() is None                                   # GLFW not initialised in this process: nothing


def test_gamepad_extra_mappings(tmp_path):
    from mandelscope.gamepad import Gamepad
    db = tmp_path / "gamecontrollerdb.txt"
    db.write_text("# SDL_GameControllerDB\n030000004c050000f20d000000000000,DualSense Edge,a:b1,platform:Windows,\n")
    lib = FakeGLFW()
    g = Gamepad(lib=lib, mappings=db)
    assert g.mappings_loaded and b"DualSense Edge" in lib.mappings
    assert not Gamepad(lib=FakeGLFW(), mappings=tmp_path / "missing.txt").mappings_loaded
