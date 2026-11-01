"""Exercise extension boundaries without importing ROS, MuJoCo, or ABC."""
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge.core.backend import ActionLayout, CommandGroup
from rosbridge.core.commands import CommandBuffer, CommandError
from rosbridge_runtime.conversions import trajectory
from rosbridge.core import registry


def test_non_abc_layout_and_atomic_rejection():
    layout = ActionLayout(("finger", "axis_b", "axis_a"), (
        CommandGroup("arm", (2, 1), (-2., -3.), (2., 3.)),
        CommandGroup("finger", (0,), (-1.,), (1.,), scalar=True),
    ))
    commands = CommandBuffer(layout, [0, 0, 0])
    commands.set_group("arm", [1, 2])
    commands.set_group("finger", [-0.5])  # no ABC-specific [0, 1] constraint
    assert commands.action(0).tolist() == [-0.5, 2., 1.]
    with pytest.raises(CommandError):
        commands.set_group("arm", [0, 99])
    assert commands.action(1).tolist() == [-0.5, 2., 1.]
    commands.set_trajectory("arm", 1, [1, 2], [[0, 0], [-1, -2]])
    assert commands.action(1.5).tolist() == [-0.5, 1., 0.5]
    target = np.array([0.3, 0.4])
    commands.set_group("arm", target)
    target[:] = 99  # caller cannot mutate the held target
    assert commands.action(20).tolist() == [-0.5, 0.4, 0.3]
    commands.stop("arm", [5., 5.])  # a measured hold may lie outside command limits
    assert commands.action(21).tolist() == [-0.5, 5., 5.]
    commands.close()
    with pytest.raises(CommandError, match="episode is done"):
        commands.set_group("finger", [0.])


def test_targets_within_a_rounding_tolerance_of_a_limit_are_clipped():
    """Interpolation rounding (0.2 - 1e-17) and a measured position a soft limit lets slightly past its bound are
    valid targets, held at the limit; anything further out is still refused whole."""
    layout = ActionLayout(("pelvis", "pinky"), (CommandGroup("pelvis", (0,), (.2,), (.78,)),
                                                CommandGroup("pinky", (1,), (0.,), (1.6,))))
    commands = CommandBuffer(layout, [.74, 0.])
    commands.set_group("pelvis", [0.19999999999999996])
    commands.set_group("pinky", [-0.0005])
    assert commands.action(0).tolist() == [.2, 0.]
    commands.set_trajectory("pelvis", 0, [1.], [[.78 + 1e-12]])
    assert commands.action(1.).tolist() == [.78, 0.]
    with pytest.raises(CommandError, match="outside"):
        commands.set_group("pinky", [-0.01])
    assert commands.action(2.).tolist() == [.78, 0.]


def test_a_clipping_layout_clips_targets_beyond_a_limit_and_reports_them():
    """With clip, joint targets, trajectory points and twists beyond a limit act at the limit instead of being
    refused whole; in-limit targets are untouched and nothing is reported."""
    groups = (CommandGroup("base", (0, 1, 2), (-.5,) * 3, (.5,) * 3, velocity=True),
              CommandGroup("torso", (3, 4), (-2.6, -.52), (2.6, .52)))
    names = ("vx", "vy", "wz", "waist_yaw", "waist_pitch")
    commands = CommandBuffer(ActionLayout(names, groups, clip=True), [0.] * 5)
    assert commands.set_group("torso", [.3, .6]) == {"waist_pitch": (.6, .52)}
    assert commands.action(0).tolist() == [0., 0., 0., .3, .52]
    assert commands.set_trajectory("torso", 0, [1., 2.], [[-.1, .2], [-3., -.9]]) == {"waist_yaw": (-3., -2.6),
                                                                                         "waist_pitch": (-.9, -.52)}
    assert commands.action(2.).tolist() == [0., 0., 0., -2.6, -.52]
    assert commands.set_group("base", [.8, 0., -.1], time_s=3.) == {"vx": (.8, .5)}
    assert commands.action(3.).tolist() == [.5, 0., -.1, -2.6, -.52]
    assert commands.set_group("torso", [.1, .52 + 1e-12]) == {}  # within the rounding tolerance: not reported
    with pytest.raises(CommandError):
        commands.set_group("torso", [float("nan"), 0.])  # malformed targets are still refused
    with pytest.raises(CommandError, match="outside"):
        CommandBuffer(ActionLayout(names, groups), [0.] * 5).set_group("torso", [.3, .6])
    actions = []
    for clip in (False, True):  # in-limit commands give the same actions either way
        b = CommandBuffer(ActionLayout(names, groups, clip=clip), [0.] * 5)
        assert b.set_group("base", [.5, -.2, .1], time_s=0.) == b.set_group("torso", [.2, .52 + 1e-4]) == {}
        assert b.set_trajectory("torso", .1, [.3, .7], [[2.6, -.1], [-.4, -.52 - 1e-4]]) == {}
        actions.append([b.action(t).tobytes() for t in np.arange(0., 1.2, .02)])
    assert actions[0] == actions[1]


def test_trajectory_conversion_ignores_named_measured_joints():
    """A trajectory echoing a group's joint state names its measured joints too; their positions are dropped."""
    msg = SimpleNamespace(
        joint_names=["pelvis_pitch", "pelvis_height", "pelvis_roll"],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
        points=[SimpleNamespace(positions=[.4, .5, .01], time_from_start=SimpleNamespace(sec=2, nanosec=0))])
    ignored = ("pelvis_pitch", "pelvis_roll")
    assert trajectory(msg, ["pelvis_height"], ignored) == (None, [2.], [[.5]])
    with pytest.raises(ValueError):
        trajectory(msg, ["pelvis_height"])  # only declared measured joints may be named
    msg.joint_names = ["pelvis_pitch", "pelvis_pitch", "pelvis_height"]
    with pytest.raises(ValueError):
        trajectory(msg, ["pelvis_height"], ignored)
    msg.joint_names, msg.points[0].positions = ["pelvis_roll"], [.01]
    with pytest.raises(ValueError):
        trajectory(msg, ["pelvis_height"], ignored)  # the commanded joints must all be named


@pytest.mark.parametrize("indices,lower,upper", [
    ((0, 0), (-1, -1), (1, 1)), ((1,), (-1,), (1,)),
    ((0,), (2,), (1,)), ((0,), (float("nan"),), (1,)),
])
def test_invalid_layout(indices, lower, upper):
    with pytest.raises(ValueError):
        ActionLayout(("q",), (CommandGroup("arm", indices, lower, upper),))


def test_measured_joints_extend_a_groups_joint_state_but_never_its_commands():
    groups = (CommandGroup("base", (0, 1, 2), (-1.,) * 3, (1.,) * 3, velocity=True),
              CommandGroup("pelvis", (3,), (.2,), (.78,)), CommandGroup("arm", (4, 5), (-1.,) * 2, (1.,) * 2))
    names = ("x", "y", "yaw", "pelvis_height", "a", "b")
    layout = ActionLayout(names, groups, measured=(("pelvis", ("pelvis_pitch", "pelvis_roll")),))
    assert layout.joint_state("pelvis") == (["pelvis_height", "pelvis_pitch", "pelvis_roll"], [3, 6, 7])
    assert layout.joint_state("arm") == (["a", "b"], [4, 5])
    assert ActionLayout(names, groups).joint_state("pelvis") == (["pelvis_height"], [3])
    commands = CommandBuffer(layout, [0, 0, 0, .74, 0, 0])  # commands span the action channels only
    with pytest.raises(CommandError):
        commands.set_group("pelvis", [.5, .1, 0.])
    for measured in ((("base", ("tilt",)),), (("pelvis", ("pelvis_height",)),), (("hand", ("q",)),),
                     (("pelvis", ()),), (("pelvis", ("p",)), ("pelvis", ("r",)))):
        with pytest.raises(ValueError):
            ActionLayout(names, groups, measured=measured)


@pytest.mark.parametrize("start,times", [(float("nan"), [1]), (0, [float("inf")]),
                                         (0, [-1]), (0, [2, 1]), (0, [])])
def test_invalid_trajectory_is_atomic(start, times):
    layout = ActionLayout(("q",), (CommandGroup("arm", (0,), (-1,), (1,)),))
    commands = CommandBuffer(layout, [0.5])
    with pytest.raises(CommandError):
        commands.set_trajectory("arm", start, times, [[0] for _ in times])
    assert commands.action(100).tolist() == [0.5]


def test_registry_loads_only_selected_plugin(monkeypatch):
    loaded = []
    def plugin(name):
        return SimpleNamespace(name=name, load=lambda: loaded.append(name) or name)
    monkeypatch.setattr(registry, "entry_points", lambda **_: [plugin("fake"), plugin("other")])
    assert registry.load_backend("fake") == "fake"
    assert loaded == ["fake"]
    with pytest.raises(ValueError, match="installed backends"):
        registry.load_backend("missing")
    monkeypatch.setattr(registry, "entry_points", lambda **_: [plugin("fake"), plugin("fake")])
    with pytest.raises(ValueError, match="duplicate"):
        registry.load_backend("fake")


def test_trajectory_conversion_without_ros():
    msg = SimpleNamespace(
        joint_names=["b", "a"], header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
        points=[SimpleNamespace(positions=[2, 1], time_from_start=SimpleNamespace(sec=1, nanosec=5))],
    )
    start, times, points = trajectory(msg, ["a", "b"])
    assert start is None and times == [1.000000005] and points == [[1, 2]]
    msg.joint_names = ["a", "a"]
    with pytest.raises(ValueError):
        trajectory(msg, ["a", "b"])
