"""Every catalog rig against Kinex's profile schema and readiness plan."""
from pathlib import Path
from unittest.mock import patch

import pytest

from rosbridge.catalog import tasks
from rosbridge.catalog.robots import list_robots, setup

PROJECT = Path(__file__).resolve().parents[3]


def test_kinex_schema_and_readiness_accept_every_rig(tmp_path):
    """Kinex's own profile schema parses every export and its readiness plan completes.

    Kinex resolves URDFs in its own Python environment, a multi-gigabyte Pixi
    project it would otherwise provision in every workspace. Set
    KINEX_PYTHON_PROJECT to one prepared `.kinex/python` directory to share it.
    """
    import json
    import os
    import shutil
    import subprocess
    dist = PROJECT.parent / "kinex2.0/packages/core/dist"
    python = os.environ.get("KINEX_PYTHON_PROJECT")
    if not python or not shutil.which("node") or not (dist / "device/profile/store.js").is_file():
        pytest.skip("needs Node, a built Kinex core next to this checkout and KINEX_PYTHON_PROJECT")
    robots = list_robots(PROJECT)
    workspaces = [tmp_path / robot.replace("/", "-") for robot in robots]
    for robot, workspace in zip(robots, workspaces):
        with patch("rosbridge.catalog.robots.changes", lambda *args: {}):  # the rig, without a task's changes
            setup(next(task["id"] for task in tasks(PROJECT).values() if task["robot"] == robot), workspace,
                  project=PROJECT)
        (workspace / ".kinex/python").symlink_to(Path(python).resolve(), target_is_directory=True)
    result = subprocess.run(["node", str(Path(__file__).with_name("profiles.mjs")), str(dist), *map(str, workspaces)],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    for robot, plan in zip(robots, json.loads(result.stdout)):
        assert plan["robot"] == robot.split("/")[1]
        assert {item["status"] for item in plan["items"]} == {"done"}, (robot, plan["items"])
