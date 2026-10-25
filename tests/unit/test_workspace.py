"""Selected-package builds and Python-environment isolation."""
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "workspace", Path(__file__).resolve().parents[2] / "src/rosbridge/provisioning/workspace.py")
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def test_build_selects_one_simulator_and_rosbridges_own_interfaces(tmp_path, monkeypatch):
    simulator = tmp_path / "simulators/fake"
    simulator.mkdir(parents=True)
    (simulator / "simulator.toml").write_text('ros_package="fake_ros"\n')
    (tmp_path / "simulators/other").mkdir()
    (tmp_path / "src/rosbridge").mkdir(parents=True)
    (tmp_path / "pixi.toml").write_text("")
    monkeypatch.setattr(sys, "argv", ["workspace.py", "--simulator", "fake", "--project", str(tmp_path)])
    calls = []
    monkeypatch.setattr(workspace.subprocess, "call", lambda command, **_: calls.append(command) or 0)
    monkeypatch.delenv("KINEX_ROS_PACKAGES", raising=False)  # no Kinex checkout is needed
    with pytest.raises(SystemExit) as finished:
        workspace.main()
    assert finished.value.code == 0
    command = calls[0]
    paths = command[command.index("--base-paths") + 1:command.index("--packages-up-to")]
    assert paths == [str(tmp_path / "ros2"), str(simulator)]
    assert command[command.index("--packages-up-to") + 1] == "fake_ros"
    assert str(tmp_path / ".workspaces/fake/install") in command
    assert (tmp_path / ".workspaces/fake/interpreter.json").is_file()


def test_vendored_wait_matches_kinex():
    """ROS 2 matches services by type name and definition hash, so the copy must not drift."""
    project = Path(__file__).resolve().parents[2]
    vendored = project / "ros2/kinex_interfaces/srv/Wait.srv"
    kinex = project.parent / "kinex2.0/packages/core/assets/ros/kinex_interfaces/srv/Wait.srv"
    if not kinex.is_file():
        pytest.skip("needs a Kinex checkout next to this one")
    assert vendored.read_bytes() == kinex.read_bytes()
    assert sorted(p.name for p in vendored.parent.iterdir()) == ["Wait.srv"]
