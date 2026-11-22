"""Activate a prepared Pixi environment without solving, installing, or shell interpolation."""
import json
import os
import shutil
import subprocess

from rosbridge.storage import local_environment


# A native worker's interpreter must not see the parent environment's Python, ROS or Pixi state.
INHERITED = ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "AMENT_PREFIX_PATH",
             "CMAKE_PREFIX_PATH", "COLCON_PREFIX_PATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
             "CONDA_SHLVL", "VIRTUAL_ENV", "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION")


def activated(root, config, base=None):
    if not shutil.which("pixi", path=(base or os.environ).get("PATH")):
        raise ValueError("Pixi is required; install Pixi, then run rosbridge prepare")
    python = root / ".pixi/envs" / config["environment"] / "bin/python"
    if not python.is_file():
        raise ValueError(f"Environment missing; run rosbridge prepare {config['name']}")
    env = local_environment(root, base)
    result = subprocess.run(
        ["pixi", "shell-hook", "--as-is", "--json", "--manifest-path", str(root / "pixi.toml"),
         "-e", config["environment"]], env=env, capture_output=True, text=True, check=True)
    variables = json.loads(result.stdout)
    # Pixi exposes activation variables under environment_variables.
    env.update(variables["environment_variables"])
    env = local_environment(root, env)
    env.pop("VIRTUAL_ENV", None)
    env["ROSBRIDGE_PROJECT"] = str(root)
    env["PYTHONNOUSERSITE"] = "1"
    return python, env


def native_activated(root, config):
    """The simulator's native interpreter and a clean environment for it."""
    base = {key: value for key, value in os.environ.items()
            if key not in INHERITED and not key.startswith(("PIXI_", "AMENT_", "COLCON_"))}
    base["PATH"] = os.pathsep.join(part for part in os.environ.get("PATH", "").split(os.pathsep)
                                   if f"{root}/.pixi/envs/" not in part)
    python, env = activated(root, {**config, "environment": config.get("native_environment")
                                   or config["environment"]}, base)
    env["PYTHONPATH"] = os.pathsep.join(native_path(root, config))
    return python, env


def native_path(root, config):
    """PYTHONPATH of a simulator's native worker: rosbridge, its adapter and generated upstream."""
    return [str(root / "src"), str(root / config["python_path"]), str(root / ".runtime" / config["name"])]


def overlay_command(root, config, command):
    overlay = root / ".workspaces" / config["name"] / "install/local_setup.bash"
    if not overlay.is_file():
        raise ValueError(f"ROS workspace missing; run rosbridge prepare {config['name']}")
    return ["bash", "-c", 'source "$1" && shift && exec "$@"',
            "rosbridge", str(overlay), *map(str, command)]
