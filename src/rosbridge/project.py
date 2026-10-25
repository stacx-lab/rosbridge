"""Source project discovery, independent of ROS and simulator installations."""
import os
from pathlib import Path


def project_root(value=None):
    explicit = value or os.environ.get("ROSBRIDGE_PROJECT")
    candidates = ([Path(explicit)] if explicit else
                  [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "src/rosbridge").is_dir() and (root / "pixi.toml").is_file():
            return root
    raise ValueError("Select a rosbridge checkout with --project or ROSBRIDGE_PROJECT")
