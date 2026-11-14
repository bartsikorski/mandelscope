"""Free ride (ride.py): the minibrots it flies to, and the timing of its legs."""
import math
import random

import pytest
from gmpy2 import mpc, mpfr

from mandelscope import ride
from mandelscope.nucleus import find_minibrot


def test_local_minis_are_nuclei():
    """Every entry of LOCAL_MINIS is the nucleus of a minibrot of that period."""
    for re, im, q in ride.LOCAL_MINIS:
        r = find_minibrot(re, im, -20)
        assert r is not None and r["period"] == q, (re, im, q)
        assert abs(float(r["re"]) - float(re)) < 1e-9 and abs(float(r["im"]) - float(im)) < 1e-9
        assert -20 < r["log2_size"] < -4


def test_embedded_copy_has_tuned_period_and_size():
    """The copy of the period-3 antenna minibrot inside the period-4 one above the top bulb."""
    parent = find_minibrot("-0.1565", "1.0322", -8)
    assert parent["period"] == 4
    parent = dict(parent, re=str(parent["re"]), im=str(parent["im"]))
    r = ride.embedded(parent, ride.LOCAL_MINIS[0], False)
    assert r is not None and r["period"] == 12
    # size: about the parent's times the local one (2^-5.7)
    assert abs(r["log2_size"] - (parent["log2_size"] - 5.7)) < 1.0
    # it is itself a nucleus: the ball-period finder at it gives the same point and period
    again = find_minibrot(r["re"], r["im"], r["log2_size"] - 4)
    assert again["period"] == 12
    d = abs(mpc(mpfr(again["re"], 200), mpfr(again["im"], 200)) - mpc(mpfr(r["re"], 200), mpfr(r["im"], 200)))
    assert float(d) < 2.0 ** (r["log2_size"] - 20)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ride_dives_deeper_until_the_period_limit(seed):
    out = ride.plan({"re": "-0.75", "im": "0", "log2r": 0.43, "seed": seed})
    assert out["parent"]["period"] == 1                 # the home view shows the whole set: dive in
    legs = []
    while out["target"] is not None:
        t = out["target"]
        if legs:
            assert t["period"] % legs[-1]["period"] == 0            # a copy inside the last one
            assert t["log2_size"] < legs[-1]["log2_size"] - 1.5      # deeper
        legs.append(t)
        out = ride.plan({"parent": t, "seed": seed * 100 + len(legs), "max_period": 20000})
        assert len(legs) < 30
    assert len(legs) >= 3
    assert all(t["period"] <= 20000 for t in legs)


def test_first_target_near_a_small_minibrot():
    """Zoomed out a little from the period-9 minibrot: it is the first target."""
    re, im = "-0.616532685850", "0.612431354461"
    out = ride.plan({"re": re, "im": im, "log2r": -9.0, "seed": 1})
    assert out["parent"] is None and out["target"]["period"] == 9


def test_framing_of_the_whole_set_is_home():
    re, im, log2r, angle = ride.framing(ride.MAIN_SET)
    assert float(re) == -0.75 and float(im) == 0.0 and angle == 0.0
    assert abs(log2r - math.log2(1.35)) < 1e-12


@pytest.mark.parametrize("dur,ramp", [(20.0, 3.0), (5.0, 3.0), (60.0, 0.0)])
def test_leg_progress_is_smooth_and_monotone(dur, ramp):
    ts = [dur * k / 400 for k in range(401)]
    us = [ride.leg_progress(t, dur, ramp) for t in ts]
    assert us[0] == 0.0 and abs(us[-1] - 1.0) < 1e-12
    assert all(b >= a for a, b in zip(us, us[1:]))
    # speed: continuous (no jumps larger than the ramp's acceleration allows)
    sp = [(b - a) / (dur / 400) for a, b in zip(us, us[1:])]
    top = max(sp)
    if ramp > 0:
        assert max(abs(b - a) for a, b in zip(sp, sp[1:])) < top * 0.1
    else:
        assert max(sp) - min(sp) < 1e-9


def test_leg_duration_follows_the_speed():
    assert ride.leg_duration(-20.0, 0.0, 0.5, 3.0) == pytest.approx(43.0)
    assert ride.leg_duration(-1.0, 0.0, 0.5, 3.0) == pytest.approx(5.0)
    assert ride.leg_duration(0.0, 0.1, 2.0, 0.0) == 4.0            # never shorter than 4 s


def test_random_choices_differ_between_rides():
    parent = dict(ride.MAIN_SET)
    picks = {ride.next_target(parent, random.Random(s))["re"][:12] for s in range(8)}
    assert len(picks) >= 3
