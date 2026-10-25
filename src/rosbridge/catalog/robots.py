"""Export Kinex workspaces from the robot catalog without importing ROS or simulators.

Each robot is a complete workspace skeleton whose profiles carry its calibration.
Files shared by several robots are symlinks; export copies their content.
"""
from pathlib import Path
import json
import shutil
import tempfile

import yaml

from rosbridge.project import project_root
from rosbridge.runs.domains import check_domain


def list_robots(project=None):
    root = project_root(project) / "catalog/robots"
    return sorted(str(file.parent.relative_to(root)) for file in root.glob("*/*/workspace.yaml"))


def setup(identifier, directory=None, *, project=None, domain_id=0):
    check_domain(domain_id)
    root = project_root(project)
    if identifier not in list_robots(root):
        raise ValueError(f"unknown robot {identifier!r}")
    target = Path(directory or Path.cwd()).resolve() / ".kinex"
    if any((target / name).exists() for name in ("workspace.yaml", "devices")):
        raise ValueError("workspace already contains device configuration; export into a new directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kinex-export-", dir=target.parent) as temporary:
        staging = Path(temporary) / "kinex"
        shutil.copytree(root / "catalog/robots" / identifier, staging,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for profile in staging.glob("devices/*/profile.yaml"):
            value = yaml.safe_load(profile.read_text())
            value["loaded"] = False
            value["runtime"] = {**value["runtime"], "domain_id": domain_id}
            profile.write_text(yaml.safe_dump(value, sort_keys=False))
        target.mkdir(exist_ok=True)
        for name in ("workspace.yaml", "devices"):
            (staging / name).rename(target / name)
    config_file = target / "kinex.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    config.setdefault("python", {"provider": "ros-jazzy"})
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    return {"robot": identifier, "workspace": str(target), "loaded": False}
