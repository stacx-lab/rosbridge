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
    with pytest.raises(SystemExit) as finished:
        workspace.main()
    assert finished.value.code == 0
    command = calls[0]
    paths = command[command.index("--base-paths") + 1:command.index("--packages-up-to")]
    assert paths == [str(tmp_path / "ros2"), str(simulator)]
    assert command[command.index("--packages-up-to") + 1] == "fake_ros"
    assert str(tmp_path / ".workspaces/fake/install") in command
    assert (tmp_path / ".workspaces/fake/interpreter.json").is_file()


def test_the_launch_files_forward_every_world_node_parameter():
    import ast

    def literal(path, name):
        """The dict or tuple literal assigned to `name` anywhere in the file; a dict's keys."""
        for node in ast.walk(ast.parse((root / path).read_text())):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                value = node.value
                return {key.value for key in value.keys} if isinstance(value, ast.Dict) else {
                    item.value for item in value.elts}
        raise AssertionError(f"{name} not in {path}")
    root = Path(__file__).resolve().parents[2]
    package = "ros2/rosbridge_runtime"
    # Launch never sets randomize; every other parameter of the world node must reach it, max_steps included.
    parameters = literal(f"{package}/rosbridge_runtime/sim_node.py", "defaults") - {"randomize"}
    assert literal(f"{package}/launch/sim.launch.py", "TYPES") == parameters
    assert literal(f"{package}/launch/demo.launch.py", "WORLD") == parameters
