"""Idempotent preparation and read-only readiness reports."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from rosbridge.assets import file_lock
from rosbridge.catalog import preparation_tasks, resolve
from rosbridge.storage import local_environment
from .environment import activated, native_activated, overlay_command
from .workspace import kinex_packages


@contextmanager
def workspace_lock(root, name, *, shared=False):
    try:
        with file_lock(root / ".runtime" / f"prepare-{name}.lock", shared=shared, blocking=False) as handle:
            yield handle
    except BlockingIOError:
        raise ValueError(f"{name} is running or being prepared; stop its runs before preparing, "
                         "or wait for preparation to finish before launching") from None


def build_identity(root, config, kinex):
    """Inputs of a simulator's ROS build, including the Kinex packages it was built with."""
    files = [(root, root / name) for name in
             ("pyproject.toml", "pixi.toml", "pixi.lock", f"simulators/{config['name']}/simulator.toml")]
    for base in (root / "ros2", root / "simulators" / config["name"], kinex):
        files += [(base, p) for p in base.rglob("*") if p.name in
                  ("setup.py", "setup.cfg", "package.xml", "CMakeLists.txt")
                  or p.suffix in (".msg", ".srv")
                  or (p.parent.name == "launch" and p.suffix == ".py")]
    digest = hashlib.sha256()
    for base, path in sorted(files):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(path.read_bytes())
    return {"schema": 2, "package": config["ros_package"], "kinex_packages": str(kinex),
            "build_hash": digest.hexdigest()}


def native(config):
    """Configuration of the simulator's native interpreter; the ROS environment by default."""
    return {**config, "environment": config.get("native_environment") or config["environment"]}


def environment_interpreter(python, env):
    return json.loads(subprocess.check_output(
        [str(python), "-c", "import json; from rosbridge.provisioning.workspace import interpreter; "
         "print(json.dumps(interpreter()))"], env=env, text=True))


def doctor(selection, *, project=None, task=None, all_tasks=False):
    if all_tasks:
        root, config, selected = preparation_tasks(selection, project, task)
    else:
        root, config, entry = resolve(selection, project, task)
        selected = [entry]
    issues = []
    source = root / config["source"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if config["source_revision"] != revision:
            issues.append("Upstream is not at the pinned revision")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
        if dirty:
            issues.append("Upstream has tracked modifications; store changes as patches")
        from .source import fingerprint, nested_sources
        for nested in nested_sources(config):
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / nested["source"],
                                           text=True).strip()
            if head != nested["source_revision"]:
                issues.append(f"{nested['source']} is not at its pinned revision")
        prepared = json.loads((root / ".runtime" / config["name"] / ".prepared.json").read_text())
        if prepared != fingerprint(root, config, revision):
            issues.append("Generated upstream source or patches are stale")
    except (OSError, ValueError, subprocess.CalledProcessError):
        issues.append("Pinned upstream or prepared source is missing")
    workspace = root / ".workspaces" / config["name"]
    try:
        built = json.loads((workspace / "project.json").read_text())
        if built != build_identity(root, config, Path(built["kinex_packages"])):
            issues.append("ROS build is stale")
    except (OSError, ValueError, KeyError):
        issues.append("ROS build is missing")
    try:
        python, env = activated(root, config)
        command = [python, "-m", "rosbridge.provisioning.check", config["name"]]
        for entry in selected:
            command += ["--task", entry["native_task"]]
        command = overlay_command(root, config, command)
        result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=30)
        if result.returncode:
            issues.append((result.stderr or result.stdout).strip()[-1500:])
        if config.get("native_environment"):
            native_python, native_env = native_activated(root, config)
            command = [native_python, "-m", "rosbridge.provisioning.check", config["name"], "--native"]
            for entry in selected:
                command += ["--task", entry["native_task"]]
            result = subprocess.run(command, env=native_env, cwd=root, capture_output=True, text=True,
                                    timeout=120)
            if result.returncode:
                issues.append((result.stderr or result.stdout).strip()[-1500:])
        marker = json.loads((workspace / "interpreter.json").read_text())
        if marker != environment_interpreter(python, env):
            issues.append("Build interpreter differs from the simulator environment")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        issues.append(str(error))
    task_option = "" if all_tasks else f" --task {selected[0]['native_task']}"
    return {"simulator": config["name"], "task": selected[0]["id"] if len(selected) == 1 else None,
            "tasks": [entry["id"] for entry in selected], "ready": not issues,
            "issues": issues, "next_steps": [] if not issues else
            [f"rosbridge prepare {config['name']}{task_option}"]}


def prepare(selection, *, project=None, task=None):
    root, config, selected = preparation_tasks(selection, project, task)
    all_tasks = task is None and selection == config["name"]
    with workspace_lock(root, config["name"]):
        def run(command, **kwargs):
            subprocess.run(list(map(str, command)), cwd=root, stdout=sys.stderr, stderr=sys.stderr,
                           check=True, **kwargs)
        for environment in dict.fromkeys((config["environment"], native(config)["environment"])):
            run(["pixi", "install", "--locked", "--manifest-path", root / "pixi.toml",
                 "-e", environment], env=local_environment(root))
        python, env = activated(root, config)
        from .source import sync_source
        sync_source(root, config)
        # Upstream source and assets are prepared by the interpreter that imports them.
        source_python, source_env = (native_activated(root, config) if config.get("native_environment")
                                     else (python, env))
        command = [source_python, "-m", "rosbridge.provisioning.source", config["name"], "--project", root]
        if not all_tasks:
            command += ["--task", selected[0]["native_task"]]
        run(command, env=source_env)
        workspace = root / ".workspaces" / config["name"]
        marker = workspace / "project.json"
        identity = build_identity(root, config, kinex_packages())
        previous = json.loads(marker.read_text()) if marker.is_file() else None
        interpreter_marker = workspace / "interpreter.json"
        old_interpreter = json.loads(interpreter_marker.read_text()) if interpreter_marker.exists() else None
        if workspace.exists() and (previous != identity or old_interpreter != environment_interpreter(python, env)):
            print(f"[prepare] rebuilding generated workspace for {config['name']}", file=sys.stderr)
            shutil.rmtree(workspace)
        run([python, "-m", "rosbridge.provisioning.workspace", "--project", root,
             "--simulator", config["name"]], env=env)
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    result = doctor(selection, project=root, task=task, all_tasks=all_tasks)
    if not result["ready"]:
        raise ValueError("Preparation checks failed: " + "; ".join(result["issues"]))
    return result
