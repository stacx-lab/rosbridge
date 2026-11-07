import numpy as np
import pytest

from rosbridge_behavior.backend import WALL_MARGIN, inside


def test_an_overview_stays_inside_the_robots_room():
    """An overview 1.8 m out whose room ends 1.0 m from the robot stands WALL_MARGIN in front of the boundary."""
    offset = np.array([1.8 * np.cos(.7), 1.8 * np.sin(.7)])
    assert inside(offset, lambda f: f * 1.8 < 1.0, .05) * 1.8 == pytest.approx(1.0 - WALL_MARGIN, abs=.05)
    assert inside(offset, lambda f: True, .05) == 1.
    assert inside(offset, lambda f: False, .05) == .01  # at a wall already: as close as the ray test allows
