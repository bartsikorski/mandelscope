"""Full screen: choosing the monitor the window is on."""
from mandelscope.window import best_monitor, overlap


def test_overlap():
    assert overlap((0, 0, 10, 10), (5, 5, 10, 10)) == 25
    assert overlap((0, 0, 10, 10), (20, 0, 10, 10)) == 0


def test_best_monitor_is_the_one_the_window_is_mostly_on():
    mons = [(0, 0, 1920, 1080), (1920, 0, 2560, 1440)]
    assert best_monitor((100, 100, 1600, 900), mons) == 0
    assert best_monitor((1500, 100, 1600, 900), mons) == 1        # more of it on the right one
    assert best_monitor((-5000, -5000, 100, 100), mons) == 0      # off every monitor: the first
