"""Command buffer and pacing; no ROS, no MuJoCo."""

import numpy as np
import pytest

from rosbridge.core.commands import CommandBuffer, CommandError
from rosbridge.runtime.pacing import Pacer

from rosbridge.core.backend import ActionLayout, CommandGroup

LAYOUT = ActionLayout(tuple(f"joint{i}" for i in range(14)), (
    CommandGroup("left", tuple(range(6)), (-1.,)*6, (1.,)*6),
    CommandGroup("left_gripper", (6,), (0.,), (1.,), scalar=True),
    CommandGroup("right", tuple(range(7, 13)), (-1.,)*6, (1.,)*6),
    CommandGroup("right_gripper", (13,), (0.,), (1.,), scalar=True),
))


def buffer():
    return CommandBuffer(LAYOUT, np.zeros(14))


def test_hold_and_validate():
    b = buffer()
    b.set_group("left", [0.5] * 6)
    b.set_group("right_gripper", [1.0])
    a = b.action(1.0)
    assert a[:6].tolist() == [0.5] * 6 and a[13] == 1.0
    for bad in ([2.0] * 6, [0.0] * 5, [float("nan")] * 6):
        with pytest.raises(CommandError):
            b.set_group("left", bad)
    with pytest.raises(CommandError):
        b.set_group("left_gripper", [1.5])
    assert np.array_equal(b.action(2.0), a)  # rejected commands change nothing
    b.close()
    with pytest.raises(CommandError):
        b.set_group("left", [0.0] * 6)


def test_trajectory_interpolates_then_holds():
    b = buffer()
    b.set_trajectory("left", 10.0, [1.0, 2.0], [[0.4] * 6, [0.8] * 6])
    assert b.action(10.5)[0] == pytest.approx(0.2)  # from held 0 toward first point
    assert b.action(11.5)[0] == pytest.approx(0.6)
    assert b.action(20.0)[0] == pytest.approx(0.8)
    b.set_group("left", [0.1] * 6)  # direct command replaces the held target
    assert b.action(30.0)[0] == pytest.approx(0.1)
    with pytest.raises(CommandError):
        b.set_trajectory("left", 0.0, [1.0, 1.0], [[0.0] * 6, [0.0] * 6])


def test_velocity_groups_start_at_rest_and_expire():
    from rosbridge.core.devices import device_inventory
    layout = ActionLayout(("arm", "x", "y", "yaw"), (
        CommandGroup("arm", (0,), (-1.,), (1.,)),
        CommandGroup("base", (1, 2, 3), (-1.,)*3, (1.,)*3, velocity=True)))
    assert [d.kind for d in device_inventory(layout, {})] == ["joint", "locomotion"]
    b = CommandBuffer(layout, [0.3, 5., 6., 7.])  # measured odometry is never a command
    assert b.action(0.05).tolist() == [0.3, 0., 0., 0.]
    b.set_group("base", [0.5, 0., -0.2], time_s=1.0)
    assert b.action(1.5).tolist() == [0.3, 0.5, 0., -0.2]
    assert b.action(1.55).tolist() == [0.3, 0., 0., 0.]  # 0.5 s without a new command
    b.set_group("base", [0.5, 0., 0.], time_s=2.0)
    b.stop("base", [0.] * 3)
    assert b.action(2.05)[1] == 0.
    with pytest.raises(CommandError):
        b.set_group("base", [0.5, 0., 0.])  # velocity commands carry their world time
    with pytest.raises(CommandError):
        b.set_group("base", [1.5, 0., 0.], time_s=3.0)
    with pytest.raises(CommandError):
        b.set_trajectory("base", 3.0, [1.0], [[0.1, 0., 0.]])
    with pytest.raises(ValueError):
        ActionLayout(("x", "y", "yaw"), (CommandGroup("bad", (0, 1, 2), (-1.,)*3, (1.,)*3,
                                                      scalar=True, velocity=True),))


def test_realtime_catches_up_without_skipping():
    now = [0.0]
    p = Pacer("realtime", 0.1, clock=lambda: now[0])
    steps = 0
    now[0] = 0.35  # loop was late by 3.5 periods
    while p.ready():
        p.advanced()
        steps += 1
    assert steps == 4 and p.wait_s() == pytest.approx(0.05)


def test_stepped_only_runs_while_waits_are_pending():
    p = Pacer("stepped", 0.1)
    assert not p.ready()
    assert p.ready(pending=True)
    p.advanced()
    assert not p.ready()
    with pytest.raises(ValueError):
        Pacer("fast", 0.1)
