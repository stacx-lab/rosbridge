"""A backend's action goals in the world: their commands, their ends, and waits that end with them. No ROS.

The pair toy's move steps an arm's tip 0.1 m a 0.1 s step from (0.5, -0.5) to the goal's last pose; a wall stops
the tips at x = 1.2.
"""
import pytest

from rosbridge.runtime.world import WorldRuntime
from tests.toy_backend import PairPoseToy

REACHED = ("succeeded", {"error_code": 0, "error_string": "reached"})


def world(**kwargs):
    sim = PairPoseToy(**kwargs)
    return WorldRuntime(sim, "stepped", lambda: None), sim


def aborted(reason):
    return "aborted", {"error_code": -1, "error_string": reason}


def start(runtime, arm, x, y):
    """Start an arm's goal; its end and the x its feedback says each step commands."""
    end, commanded = [], []
    runtime.start(f"{arm}/move", {"multi_dof_trajectory": {"points": [{"transforms": [{"translation": {"x": x, "y": y}}]}]}},
                  lambda *outcome: end.append(outcome),
                  lambda fields: commanded.append(fields["multi_dof_desired"]["transforms"][0]["translation"]["x"]))
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


def test_a_goal_runs_in_the_backend_and_the_wait_for_it_ends_with_it():
    runtime, sim = world()
    end, commanded = start(runtime, "left", 1., -.5)
    result, finished = wait(runtime, end)
    assert result.success and finished == [True] and result.waited_s == pytest.approx(.5)
    assert end == [REACHED] and commanded == pytest.approx([.6, .7, .8, .9, 1.]) and not runtime.goals
    assert sim.position[:2] == pytest.approx([1., -.5])
    assert runtime.commands.action(9.)[:2] == pytest.approx([1., -.5])  # the arm keeps the goal's last command
    again, _ = start(runtime, "left", 1., -.5)  # already there: no step
    assert again == [REACHED] and runtime.steps == 5
    refused, _ = start(runtime, "left", 1., 1.9)  # a goal the backend cannot finish ends where it gets
    assert wait(runtime, refused)[0].waited_s == pytest.approx(1.6)
    assert refused == [aborted("the tip pose is beyond the 1.5 m reach")] and sim.position[1] == pytest.approx(1.1)


def test_a_wait_ends_once_its_duration_has_passed_and_its_goals_have_finished():
    runtime, _ = world()
    end, _ = start(runtime, "left", .7, -.5)  # two steps
    result, _ = wait(runtime, duration=1.)  # without goals: its duration, whatever goals do
    assert result.waited_s == pytest.approx(1.) and end == [REACHED]
    left, _ = start(runtime, "left", .5, -.5)
    right, _ = start(runtime, "right", 1., -.5)  # five steps
    result, finished = wait(runtime, left, right)  # with goals alone: until the last finishes
    assert result.success and finished == [True, True] and result.waited_s == pytest.approx(.5)
    back, _ = start(runtime, "right", .8, -.5)
    result, finished = wait(runtime, back, duration=.5)  # with both: the later of the two
    assert finished == [True] and result.waited_s == pytest.approx(.5)
    out, _ = start(runtime, "right", 1., -.5)
    assert wait(runtime, out, duration=.1)[0].waited_s == pytest.approx(.2)
    result, finished = wait(runtime, left, right, out)  # all finished: nothing to wait for
    assert result.success and finished == [True] * 3 and result.waited_s == 0. and runtime.steps == 22


def test_goals_still_executing_at_a_waits_timeout_are_ended():
    runtime, sim = world()
    left, _ = start(runtime, "left", .7, -.5)
    right, _ = start(runtime, "right", 1., -.5)
    result, finished = wait(runtime, left, right, timeout=.3, expire=lambda: runtime.end("right/move", "too slow"))
    assert result.success and finished == [True, True] and result.waited_s == pytest.approx(.3)
    assert left == [REACHED] and right == [aborted("too slow")]
    wait(runtime, duration=.5)
    assert sim.position[2] == pytest.approx(.8)  # it stays where the backend left it


def test_a_goal_the_backend_ends_leaves_its_group_where_the_backend_puts_it():
    runtime, sim = world()
    end, _ = start(runtime, "left", 1.5, -.5)  # the wall stops the tip at x = 1.2
    result, _ = wait(runtime, end)
    assert end == [aborted("stalled")] and result.waited_s == pytest.approx(.8)
    assert runtime.commands.action(9.)[:2] == pytest.approx([1.2, -.5])


def test_a_goal_and_a_groups_targets_replace_each_other():
    runtime, sim = world()
    goal, _ = start(runtime, "left", 1., -.5)
    wait(runtime, duration=.2)
    runtime.commands.set_trajectory("left", runtime.time_s, [.2], [[.5, -.5]])  # as a joint_target does
    runtime.release("left", "replaced by a command")
    assert goal == [aborted("replaced by a command")]
    wait(runtime, duration=.1)
    assert sim.position[:2] == pytest.approx([.65, -.5])  # from the goal's command, 0.8: no jump back
    wait(runtime, duration=.1)
    runtime.commands.set_trajectory("left", runtime.time_s, [1.], [[1.1, -.5]])
    wait(runtime, duration=.2)
    goal, _ = start(runtime, "left", .5, 0.)  # a goal takes the group from its trajectory
    result, _ = wait(runtime, goal)
    assert goal == [REACHED] and sim.position[:2] == pytest.approx([.5, 0.])
    wait(runtime, duration=1.)
    assert sim.position[:2] == pytest.approx([.5, 0.])  # the trajectory is gone


def test_a_newer_goal_a_stop_and_the_episodes_end_end_the_active_goal():
    runtime, sim = world(max_steps=3)
    first, _ = start(runtime, "left", 1., -.5)
    second, _ = start(runtime, "left", .5, 0.)
    assert first == [aborted("replaced by a newer goal")]
    wait(runtime, duration=.1)
    runtime.hold("left")
    assert second == [aborted("stopped")] and sim.position[:2] == pytest.approx([.5, -.4])
    result, finished = wait(runtime, second)  # finished already: no step
    assert finished == [True] and runtime.steps == 1
    last, _ = start(runtime, "right", 1., -.5)
    result, finished = wait(runtime, last)
    assert finished == [True] and runtime.steps == 3  # the end finished the goal the wait was for
    assert result.message == "ok; the episode ended because the time limit was reached"
    assert last == [aborted("the episode ended because the time limit was reached")]
    late, _ = start(runtime, "right", 1., -.5)
    assert late == [aborted("the episode ended because the time limit was reached")]
