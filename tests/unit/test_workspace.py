"""Selected-package builds and Python-environment isolation."""
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "workspace", Path(__file__).resolve().parents[2] / "src/rosbridge/provisioning/workspace.py")
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def test_build_selects_one_simulator_with_kinex_packages(tmp_path, monkeypatch):
    simulator = tmp_path / "simulators/fake"
    simulator.mkdir(parents=True)
    (simulator / "simulator.toml").write_text('ros_package="fake_ros"\n')
    (tmp_path / "simulators/other").mkdir()
    (tmp_path / "src/rosbridge").mkdir(parents=True)
    (tmp_path / "pixi.toml").write_text("")
    kinex = tmp_path / "kinex/assets/ros"
    (kinex / "kinex_interfaces").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["workspace.py", "--simulator", "fake", "--project", str(tmp_path)])
    calls = []
    monkeypatch.setattr(workspace.subprocess, "call", lambda command, **_: calls.append(command) or 0)
    monkeypatch.delenv("KINEX_ROS_PACKAGES", raising=False)
    with pytest.raises(ValueError, match="KINEX_ROS_PACKAGES"):
        workspace.main()
    (kinex / "kinex_interfaces/package.xml").write_text("")
    monkeypatch.setenv("KINEX_ROS_PACKAGES", str(kinex))
    with pytest.raises(SystemExit) as finished:
        workspace.main()
    assert finished.value.code == 0
    command = calls[0]
    assert str(simulator) in command and str(kinex) in command
    assert str(tmp_path / "simulators/other") not in command
    assert command[command.index("--packages-up-to") + 1] == "fake_ros"
    assert str(tmp_path / ".workspaces/fake/install") in command
    assert (tmp_path / ".workspaces/fake/interpreter.json").is_file()
