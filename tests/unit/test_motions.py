"""Tip goals: pacing, their ends, and world waits that end with them. No ROS."""
import numpy as np
import pytest

from rosbridge.core.motions import (GOAL_TOLERANCE_VIOLATED, INVALID_GOAL, PATH_TOLERANCE_VIOLATED, SUCCESSFUL,
                                    Motion)
from rosbridge.runtime.world import WorldRuntime
from tests.toy_backend import PairPoseToy

UPRIGHT = (0., 0., 0., 1.)


def pose(x, y, quaternion=UPRIGHT):
    return np.array([x, y, 0., *quaternion])


def world(**kwargs):
    """The two planar arms, tips at (0.5, -0.5), stepping 0.1 s: a tip moves 0.1 m a step at most."""
    sim = PairPoseToy(**kwargs)
    return WorldRuntime(sim, "stepped", lambda: None), sim


def move(runtime, group, *points, times=None, **options):
    """Start a goal through points; its end and the poses commanded for it."""
    end, commanded = [], []
    runtime.move(group, lambda *outcome: end.append(outcome), lambda time_s, target, measured: commanded.append(target),
                 points, times or [0.] * len(points), **options)
    return end, commanded


def wait(runtime, *ends, duration=0., timeout=0., expire=None):
    """A world wait for duration and the goals with these ends, run to its reply: the result and which had
    finished."""
    replies = []
    until = [lambda end=end: bool(end) for end in ends]
    runtime.wait(duration, lambda result: replies.append((result, [done() for done in until])), until, timeout, expire)
    while not replies:
        runtime.tick()
    return replies[0]


def test_a_motion_steps_from_the_measured_pose_at_its_pace():
    half_turn = (0., 0., 1., 0.)
    motion = Motion(pose(0., 0.), 0., .1, [pose(1., 0.), pose(1., 0., half_turn)], [0., 0.])
    assert motion.reference() == pytest.approx(pose(.1, 0.))  # 1 m/s for a 0.1 s step
    assert motion.observe(pose(.998, 0.), .1) is None and motion.index == 1  # within tolerance: on to the next pose
    turned = motion.reference()
    assert turned[:3] == pytest.approx([1., 0., 0.]) and 2 * np.arccos(turned[6]) == pytest.approx(.33)
    slow = Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [4.])
    assert slow.reference() == pytest.approx(pose(.025, 0.))  # a quarter of the pace for four times the time
    assert Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [.5]).reference() == pytest.approx(pose(.1, 0.))  # no faster
    # A tip that lags is late settle_s after the goal's nominal end, the 1 s its pace needs.
    late = Motion(pose(0., 0.), 0., .1, [pose(1., 0.)], [0.], settle_s=.2)
    outcomes = [late.observe(pose(.05 * step, 0.), .1 * step) for step in range(1, 13)]
    assert outcomes[:11] == [None] * 11 and outcomes[11] == (GOAL_TOLERANCE_VIOLATED, "not reached in time")


def test_a_goal_reaches_its_pose_and_ends_the_wait_for_it():
    runtime, sim = world()
    end, commanded = move(runtime, "left", pose(1., -.5))
    result, finished = wait(runtime, end)
    assert result.success and finished == [True] and result.waited_s == pytest.approx(.5)  # 0.5 m at 0.1 m a step
    assert end == [(SUCCESSFUL, "reached: 0.000 m and 0.000 rad from the goal after 0.50 s")]
    assert [target[0] for target in commanded] == pytest.approx([.6, .7, .8, .9, 1.])
    assert sim.position[:2] == pytest.approx([1., -.5]) and not runtime.motions
    again, _ = move(runtime, "left", pose(1.004, -.5))  # already within tolerance: no step
    assert again[0][0] == SUCCESSFUL and runtime.steps == 5


def test_a_wait_ends_once_its_duration_has_passed_and_its_goals_have_finished():
    runtime, _ = world()
    end, _ = move(runtime, "left", pose(.7, -.5))  # two steps
    result, _ = wait(runtime, duration=1.)  # without goals: its duration, whatever goals do
    assert result.waited_s == pytest.approx(1.) and end[0][0] == SUCCESSFUL
    left, _ = move(runtime, "left", pose(.5, -.5))
    right, _ = move(runtime, "right", pose(1., -.5))  # five steps
    result, finished = wait(runtime, left, right)  # with goals alone: until the last finishes
    assert result.success and finished == [True, True] and result.waited_s == pytest.approx(.5)
    back, _ = move(runtime, "right", pose(.8, -.5))
    result, finished = wait(runtime, back, duration=.5)  # with both: the later of the two
    assert finished == [True] and result.waited_s == pytest.approx(.5)
    out, _ = move(runtime, "right", pose(1., -.5))
    assert wait(runtime, out, duration=.1)[0].waited_s == pytest.approx(.2)
    result, finished = wait(runtime, left, right, out)  # all finished: nothing to wait for
    assert result.success and finished == [True] * 3 and result.waited_s == 0. and runtime.steps == 22


def test_goals_still_executing_at_a_waits_timeout_are_cancelled():
    runtime, sim = world()
    left, _ = move(runtime, "left", pose(.7, -.5))
    right, _ = move(runtime, "right", pose(1., -.5))
    late = lambda: runtime.end("right", "too slow", GOAL_TOLERANCE_VIOLATED, hold=True)
    result, finished = wait(runtime, left, right, timeout=.3, expire=late)
    assert result.success and finished == [True, True] and result.waited_s == pytest.approx(.3)
    assert left[0][0] == SUCCESSFUL
    assert right == [(GOAL_TOLERANCE_VIOLATED, "too slow: 0.200 m and 0.000 rad from the goal after 0.30 s")]
    result, _ = wait(runtime, duration=.5)
    assert sim.position[2] == pytest.approx(.8)  # it holds where it was cancelled
    again, _ = move(runtime, "right", pose(1., -.5))
    result, finished = wait(runtime, again, duration=.5, timeout=.4, expire=late)  # in time: its duration
    assert again[0][0] == SUCCESSFUL and result.waited_s == pytest.approx(.5)


def test_a_goal_takes_its_tolerance_and_times():
    runtime, sim = world()
    end, _ = move(runtime, "left", pose(.9, -.5), tolerance=(.15, 0.))
    wait(runtime, end)
    assert end[0][0] == SUCCESSFUL and sim.position[0] == pytest.approx(.8)  # within 0.15 m
    slow, _ = move(runtime, "right", pose(1., -.5), times=[2.])  # 0.5 m in 2 s: 0.025 m a step
    result, _ = wait(runtime, slow)
    assert slow[0][0] == SUCCESSFUL and result.waited_s == pytest.approx(2.)


def test_a_stalled_goal_ends_holding_the_measured_joints():
    runtime, sim = world()
    end, _ = move(runtime, "left", pose(1.5, -.5))  # the wall stops the tip at x = 1.2
    result, _ = wait(runtime, end)
    assert end == [(PATH_TOLERANCE_VIOLATED, "stalled: 0.300 m and 0.000 rad from the goal after 1.00 s")]
    assert result.waited_s == pytest.approx(1.) and runtime.commands.action(9.)[:2] == pytest.approx([1.2, -.5])


def test_a_step_without_joints_ends_the_goal_where_it_is():
    runtime, sim = world()
    end, _ = move(runtime, "left", pose(.5, 1.9))
    result, _ = wait(runtime, end)
    code, text = end[0]
    assert code == INVALID_GOAL and text.startswith("the tip pose is beyond the 1.5 m reach: ")
    assert sim.position[1] == pytest.approx(1.4) and result.waited_s == pytest.approx(1.9)  # the progress stays
    with pytest.raises(ValueError, match="left goal pose needs a unit quaternion"):
        move(runtime, "left", pose(0., 0., (0., 0., 0., 0.)))


def test_a_newer_goal_a_stop_and_the_episodes_end_end_the_active_goal():
    runtime, sim = world(max_steps=3)
    first, _ = move(runtime, "left", pose(1., -.5))
    second, _ = move(runtime, "left", pose(.5, 0.))
    assert first[0] == (INVALID_GOAL, "replaced by a newer goal: 0.500 m and 0.000 rad from the goal after 0.00 s")
    wait(runtime, duration=.1)
    runtime.hold("left")
    assert second[0][1].startswith("stopped: ") and sim.position[:2] == pytest.approx([.5, -.4])
    result, finished = wait(runtime, second)  # finished already: no step
    assert finished == [True] and runtime.steps == 1
    last, _ = move(runtime, "right", pose(1., -.5))
    result, finished = wait(runtime, last)
    assert finished == [True] and runtime.steps == 3  # the end finished the goal the wait was for
    assert result.message == "ok; the episode ended because the time limit was reached"
    assert last[0] == (INVALID_GOAL, "the episode ended because the time limit was reached: "
                                     "0.300 m and 0.000 rad from the goal after 0.20 s")
    with pytest.raises(ValueError, match="the episode ended because the time limit was reached"):
        move(runtime, "right", pose(1., -.5))
