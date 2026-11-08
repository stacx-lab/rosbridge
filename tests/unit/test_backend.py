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


def test_targets_within_a_tolerance_of_a_limit_are_held_at_it():
    """A measured position slightly past a limit, or a client's interpolation rounding past one, is a valid target;
    anything further out is still refused whole."""
    layout = ActionLayout(("lift", "pinky"), (CommandGroup("lift", (0,), (.2,), (.78,)),
                                              CommandGroup("pinky", (1,), (0.,), (1.6,))))
    commands = CommandBuffer(layout, [.74, 0.])
    commands.set_group("lift", [0.19999999999999996])
    commands.set_group("pinky", [-0.0005])
    assert commands.action(0).tolist() == [.2, 0.]
    commands.set_trajectory("lift", 0, [1.], [[.78 + 1e-6]])
    assert commands.action(1.).tolist() == [.78, 0.]
    with pytest.raises(CommandError, match="outside"):
        commands.set_trajectory("lift", 1, [1., 2.], [[.5], [.78 + 2e-3]])
    with pytest.raises(CommandError, match="outside"):
        commands.set_group("pinky", [-0.01])
    assert commands.action(2.).tolist() == [.78, 0.]


@pytest.mark.parametrize("indices,lower,upper", [
    ((0, 0), (-1, -1), (1, 1)), ((1,), (-1,), (1,)),
    ((0,), (2,), (1,)), ((0,), (float("nan"),), (1,)),
])
def test_invalid_layout(indices, lower, upper):
    with pytest.raises(ValueError):
        ActionLayout(("q",), (CommandGroup("arm", indices, lower, upper),))


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
