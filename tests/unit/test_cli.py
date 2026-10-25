"""CLI discovery and lifecycle plumbing without ROS or simulators."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rosbridge.storage import write_json
from rosbridge.catalog import resolve, tasks
from rosbridge.cli.main import main
from rosbridge.project import project_root
from rosbridge.provisioning.environment import overlay_command
from rosbridge.runs.domains import DomainLease
from rosbridge.runs import api
from rosbridge.runs.records import SCHEMA, alive, identity, select

ROOT = Path(__file__).resolve().parents[2]


def test_catalog_aliases_and_cli_json(capsys):
    _, _, default = resolve("abc", ROOT)
    assert default["id"] == "abc/put-plastic-bottles-in-bin"
    assert resolve(default["id"], ROOT)[2] == default
    assert resolve("abc", ROOT, "put_plastic_bottles_in_bin")[2] == default
    assert main(["tasks", "list", "--project", str(ROOT), "--simulator", "abc", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tasks"] == len(tasks(ROOT, "abc")) and default["id"] in data["ids"]
    assert main(["prepare", "missing", "--json"]) == 1
    assert "Unknown simulator" in json.loads(capsys.readouterr().out)["error"]


def test_cli_imports_no_native_dependencies():
    code = '''import sys
from rosbridge.cli.main import main
assert main(["tasks", "list", "--simulator", "abc"]) == 0
assert not any(x in sys.modules for x in ("rclpy", "mujoco", "numpy", "abc_sim"))
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True, capture_output=True)


def test_project_resolution_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROSBRIDGE_PROJECT", str(ROOT))
    assert project_root() == ROOT
    with pytest.raises(ValueError, match="checkout"):
        project_root(tmp_path)


def test_domain_leases_are_exclusive_and_released():
    first = DomainLease()
    second = DomainLease()
    try:
        assert first.domain != second.domain
        with pytest.raises(ValueError, match="already managed"):
            DomainLease(first.domain)
        chosen = first.domain
    finally:
        first.close()
        second.close()
    again = DomainLease(chosen)
    again.close()
    with pytest.raises(ValueError, match="between"):
        DomainLease(102)


def test_cache_and_log_environment_is_checkout_local(tmp_path):
    from rosbridge.storage import local_environment
    inherited = {key: "/nethome/small" for key in
                 ("XDG_CACHE_HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
                  "PIXI_CACHE_DIR", "ROS_HOME", "ROS_LOG_DIR", "CUDA_CACHE_PATH")}
    inherited["HOME"] = "/nethome/user"
    env = local_environment(tmp_path, inherited)
    assert env["HOME"] == inherited["HOME"]
    for key in inherited.keys() - {"HOME", "UV_PYTHON_INSTALL_DIR"}:
        assert Path(env[key]).is_relative_to(tmp_path / ".cache/rosbridge")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / ".runtime/python")
    assert env["ROSBRIDGE_PROJECT"] == str(tmp_path)


def test_stale_owner_is_never_signalled(tmp_path, monkeypatch):
    owner = identity(os.getpid())
    record = {"schema": SCHEMA, "uid": os.getuid(), "state": "ready", "owner": {**owner, "start": "stale"}}
    path = tmp_path / "run.json"
    write_json(path, record)
    assert not alive(record)
    monkeypatch.setattr(api.os, "kill", lambda *a: pytest.fail("signalled a reused PID"))
    result = api.stop(str(path), project=ROOT)
    assert result["running"] is False


def test_default_selection_requires_one_live_run(tmp_path):
    for name in ("one", "two"):
        path = tmp_path / ".cache/rosbridge/runs" / name / "run.json"
        path.parent.mkdir(parents=True)
        write_json(path, {"schema": SCHEMA, "uid": os.getuid(), "state": "ready", "owner": identity(os.getpid())})
    with pytest.raises(ValueError, match="2 live runs"):
        select(tmp_path)


def test_overlay_preserves_command_arguments(tmp_path):
    overlay = tmp_path / ".workspaces/fake/install/local_setup.bash"
    overlay.parent.mkdir(parents=True)
    overlay.write_text("export ROSBRIDGE_TEST=1\n")
    values = ["space here", "$(touch should-not-exist)", "`touch also-not`", "line1\nline2"]
    command = overlay_command(tmp_path, {"name": "fake"}, [sys.executable, "-c",
        "import json,sys; print(json.dumps(sys.argv[1:]))", *values])
    result = subprocess.check_output(command, cwd=tmp_path, text=True)
    assert json.loads(result) == values
    assert not (tmp_path / "should-not-exist").exists()


def test_invalid_flags_fail_before_launch():
    for args in (["launch", "abc", "--domain-id", "102"],
                 ["launch", "abc", "--timeout", "nan"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2


def test_preparation_cannot_modify_a_running_environment(tmp_path):
    from rosbridge.provisioning.setup import workspace_lock
    with workspace_lock(tmp_path, "fake", shared=True):
        with workspace_lock(tmp_path, "fake", shared=True):
            with pytest.raises(ValueError, match="stop its runs"):
                with workspace_lock(tmp_path, "fake"):
                    pytest.fail("preparation entered while worlds were active")
    with workspace_lock(tmp_path, "fake"):
        with pytest.raises(ValueError, match="preparation to finish"):
            with workspace_lock(tmp_path, "fake", shared=True):
                pytest.fail("launch entered during preparation")
