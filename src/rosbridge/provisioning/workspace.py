"""Build a selected simulator's isolated ROS overlay inside its environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from rosbridge.catalog import simulator


def interpreter():
    # Keep the environment path: resolving symlinks would collapse distinct venvs.
    return {"executable": os.path.abspath(sys.executable), "prefix": sys.prefix,
            "version": list(sys.version_info[:3])}


def kinex_packages():
    """Kinex core's ROS packages; every graph serves kinex_interfaces services."""
    packages = os.environ.get("KINEX_ROS_PACKAGES")
    if not packages or not (Path(packages) / "kinex_interfaces/package.xml").is_file():
        raise ValueError("set KINEX_ROS_PACKAGES to Kinex core's packages/core/assets/ros directory")
    return Path(packages).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", required=True)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    root, directory, config = simulator(args.simulator, args.project)
    workspace = root / ".workspaces" / args.simulator
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "interpreter.json").write_text(json.dumps(interpreter(), indent=2) + "\n")
    command = [sys.executable, "-m", "colcon", "--log-base", str(workspace / "log"),
               "build", "--symlink-install", "--build-base", str(workspace / "build"),
               "--install-base", str(workspace / "install"),
               "--base-paths", str(root / "ros2"), str(directory), str(kinex_packages()),
               "--packages-up-to", config["ros_package"], "--cmake-args",
               f"-DPython3_EXECUTABLE={sys.executable}", f"-DPYTHON_EXECUTABLE={sys.executable}"]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
