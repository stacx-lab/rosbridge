"""Export the Kinex workspace of a task and seed from the robot catalog, without importing ROS or simulators.
`rosbridge setup` exports it in the simulator's environment with the episode's task description
(rosbridge.runs.setup).

Each robot is a complete workspace skeleton whose profiles carry its calibration; it may also ship Kinex
workspace content (SHIPPED). Files shared by several robots are symlinks; export copies their content.

A simulator that declares configurations in its manifest, alternative device sets such as RoboDojo's RGB-D
cameras, gives each of its rigs a `configurations/<name>/` directory: its optional `profiles.yaml` holds
profile changes by device, and its SHIPPED directories add files over the rig's.

A simulator whose devices differ by task or seed, such as HandsOn's cameras, names a `workspace` hook in its
manifest: "module:function" under its python_path, called with its runtime directory, the native task, the
seed and the robot's profiles by device, which returns profile changes by device.

Profile changes are JSON merge patches (see merge).
"""
import importlib
from pathlib import Path
import json
import shutil
import sys
import tempfile

import yaml

from rosbridge.catalog import check_configuration, resolve
from rosbridge.project import project_root
from rosbridge.runs.domains import check_domain

# Kinex workspace content a rig or configuration may ship besides its profiles: its function library
# (kinex.lib.<module>), saved programs and skills.
SHIPPED = ("lib", "programs", "skills")
# The episode's task description, which `rosbridge setup` adds to the workspace.
DESCRIPTION = "task_description.json"
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc")


def list_robots(project=None):
    root = project_root(project) / "catalog/robots"
    return sorted(str(file.parent.relative_to(root)) for file in root.glob("*/*/workspace.yaml"))


def merge(profile, changes):
    """Change a profile in place by a JSON merge patch (RFC 7386): mappings merge key by key, None removes,
    other values replace. Unlike RFC 7386, a mapping new to the profile is assigned whole, nulls included:
    profiles keep null facts, such as a camera's fps."""
    for key, value in changes.items():
        if value is None:
            profile.pop(key, None)
        elif isinstance(value, dict) and isinstance(profile.get(key), dict):
            merge(profile[key], value)
        else:
            profile[key] = value


def apply(profiles, changes, source):
    """Merge profile changes by device into profiles by device; source names them in errors."""
    unknown = set(changes) - set(profiles)
    if unknown:
        raise ValueError(f"{source} changes unknown devices {sorted(unknown)}")
    for name, change in changes.items():
        merge(profiles[name], change)


def configuration_directory(root, identifier, configuration):
    """A robot's directory for a configuration of its simulator; None for its standard devices."""
    if configuration is None:
        return None
    directory = root / "catalog/robots" / identifier / "configurations" / configuration
    if not directory.is_dir():
        raise ValueError(f"robot {identifier} has no configuration {configuration!r}")
    return directory


def catalog_profiles(root, identifier, configuration=None):
    """A catalog robot's device profiles by device name, in a configuration of its simulator."""
    devices = root / "catalog/robots" / identifier / "devices"
    profiles = {path.parent.name: yaml.safe_load(path.read_text()) for path in sorted(devices.glob("*/profile.yaml"))}
    overlay = configuration_directory(root, identifier, configuration)
    if overlay is not None and (overlay / "profiles.yaml").is_file():
        apply(profiles, yaml.safe_load((overlay / "profiles.yaml").read_text()) or {},
              f"configuration {configuration!r}")
    return profiles


def changes(root, config, task, seed, profiles=None):
    """The simulator's profile changes for a task and seed, by device, given the robot's profiles (default: its
    catalog profiles); none without a workspace hook."""
    if not config.get("workspace"):
        return {}
    module, function = config["workspace"].split(":")
    path = str(root / config["python_path"])
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        hook = getattr(importlib.import_module(module), function)
    finally:
        if added:
            sys.path.remove(path)
    return hook(root / ".runtime" / config["name"], task["native_task"], seed,
                catalog_profiles(root, task["robot"]) if profiles is None else profiles)


def export(selection, directory=None, *, seed=None, project=None, domain_id=0, configuration=None, describe=None):
    """The workspace a launch of the same task, seed and configuration serves, every device loaded (Kinex activates
    loaded devices when it starts): the task's robot in that
    configuration, with its simulator's profile changes for that task and seed and the content it ships. The seed
    defaults as a launch's does. With describe(root, config, task, seed), the episode's task description is added
    as task_description.json once the selection checks out."""
    check_domain(domain_id)
    root, config, task = resolve(selection, project)
    check_configuration(config, configuration)
    seed = seed if seed is not None else task.get("seed", 0)
    identifier = task["robot"]
    if identifier not in list_robots(root):
        raise ValueError(f"task {task['id']} names an unknown robot {identifier!r}")
    rig = root / "catalog/robots" / identifier
    # The rig's shipped content, then the configuration's over it.
    sources = [source for source in (rig, configuration_directory(root, identifier, configuration)) if source]
    profiles = catalog_profiles(root, identifier, configuration)
    apply(profiles, changes(root, config, task, seed, profiles), f"{config['name']}'s workspace hook")
    exported = ["workspace.yaml", "devices",
                *(name for name in SHIPPED if any((source / name).is_dir() for source in sources)),
                *([DESCRIPTION] if describe else [])]
    target = Path(directory or Path.cwd()).resolve() / ".kinex"
    present = [name for name in exported if (target / name).exists()]
    if present:
        raise ValueError(f"workspace already contains {', '.join(present)}; export into a new directory")
    description = describe(root, config, task, seed) if describe else None
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kinex-export-", dir=target.parent) as temporary:
        staging = Path(temporary)
        shutil.copy(rig / "workspace.yaml", staging)
        shutil.copytree(rig / "devices", staging / "devices", ignore=IGNORED)
        for source in sources:
            for name in SHIPPED:
                if (source / name).is_dir():
                    shutil.copytree(source / name, staging / name, ignore=IGNORED, dirs_exist_ok=True)
        for name, value in profiles.items():
            value["loaded"] = True
            value["runtime"] = {**value["runtime"], "domain_id": domain_id}
            (staging / "devices" / name / "profile.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
        if description is not None:
            (staging / DESCRIPTION).write_text(json.dumps(description, indent=2, ensure_ascii=False) + "\n")
        target.mkdir(exist_ok=True)
        for name in exported:
            (staging / name).rename(target / name)
    config_file = target / "kinex.json"
    config_value = json.loads(config_file.read_text()) if config_file.exists() else {}
    config_value.setdefault("python", {"provider": "ros-jazzy"})
    config_file.write_text(json.dumps(config_value, indent=2) + "\n")
    return {"task": task["id"], "seed": seed, "configuration": configuration, "robot": identifier,
            "workspace": str(target), "loaded": True,
            **({"task_description": str(target / DESCRIPTION)} if description is not None else {})}
