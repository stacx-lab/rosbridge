"""Which of cuRobo's IK solutions a RoboDojo arm executes for a pose target, without Isaac."""
import numpy as np
import pytest

from rosbridge_robodojo.backend import PATH_SAMPLE_RAD, choose, nearest_turn, path

NAMES = [f"joint{i}" for i in range(1, 7)]
LOWER, UPPER = np.r_[[-10.] * 5, -3.14], np.r_[[10.] * 5, 3.14]
PREVIOUS = np.zeros(6)
STEP = .2  # the X5's 5 rad/s native velocity limit over one 0.04 s frame


def pick(candidates, path_free=lambda rows: True):
    return choose(candidates, PREVIOUS, LOWER, UPPER, STEP, path_free, NAMES)


def test_a_solution_takes_the_whole_turn_nearest_the_commanded_joints_within_its_limits():
    q = nearest_turn(np.r_[2 * np.pi + .1, -2 * np.pi, 0., 0., 0., 3.], np.r_[0., 0., 0., 0., 0., -3.],
                     LOWER, UPPER)
    assert q == pytest.approx([.1, 0., 0., 0., 0., 3.])  # joint6's limits allow no other turn


def test_the_first_ranked_solution_one_frame_can_execute_is_taken():
    near, nearer = np.r_[.1, 0, 0, 0, 0, 0], np.r_[.05, 0, 0, 0, 0, 0]
    flip = np.r_[0, 0, 0, 0, 3.1, 0]
    q, why = pick([flip, near, nearer])
    assert why is None and q == pytest.approx(near)  # cuRobo's ranking, not the nearest
    q, why = pick([flip])
    assert q is None and why == "moves joint5 +3.10 rad in one frame, beyond 0.20 rad"
    q, why = pick([np.r_[0, 0, 0, 0, 0, np.pi]])  # joint6's limits hold neither pi nor -pi
    assert q is None and why == "lies outside the joint limits"


def test_a_solution_whose_joint_path_collides_is_passed_over():
    blocked, clear = np.r_[.15, 0, 0, 0, 0, 0], np.r_[0, .15, 0, 0, 0, 0]
    seen = []

    def path_free(rows):
        seen.append(rows)
        return not rows[:, 0].any()

    q, why = pick([blocked, clear], path_free)
    assert why is None and q == pytest.approx(clear)
    q, why = pick([blocked], path_free)
    assert q is None and why == "has a joint path that collides with the arm itself or the table"
    # The path leaves the commanded joints out and samples at most PATH_SAMPLE_RAD apart, ending at the solution.
    rows = seen[0]
    assert len(rows) == 6 and np.diff(np.vstack([PREVIOUS, rows]), axis=0).max() <= PATH_SAMPLE_RAD + 1e-12
    assert rows[-1] == pytest.approx(blocked)
    assert len(path(PREVIOUS, PREVIOUS)) == 1  # an unchanged pose is one sample, the solution itself
