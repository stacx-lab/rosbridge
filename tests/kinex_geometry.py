"""Exported catalog rigs evaluated with Kinex's own URDF kinematics.

Simulator geometry tests compare these poses with native measurements. They
need Kinex Python (KINEX_PYTHON=packages/core/assets/python) and mark
themselves with `requires_kinex`. A module-level skip would end collection of
sibling test modules under ROS's launch_testing pytest plugin.
"""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import yaml

from rosbridge.catalog.robots import setup

if os.environ.get("KINEX_PYTHON") and os.environ["KINEX_PYTHON"] not in sys.path:
    sys.path.insert(0, os.environ["KINEX_PYTHON"])

PROJECT = Path(__file__).resolve().parents[1]


def available():
    try:
        return importlib.util.find_spec("kinex.lib.joints") is not None
    except ModuleNotFoundError:
        return False


requires_kinex = pytest.mark.skipif(not available(),
                                    reason="needs Kinex Python; set KINEX_PYTHON")


def compose(a, b):
    from kinex.lib.geometry import compose_rt
    return compose_rt(a, b)


def xyzq(value):
    from kinex.lib.geometry import xyzq_to_rt
    return xyzq_to_rt(value)


def rt(position, quaternion_wxyz):
    return xyzq([*position, *quaternion_wxyz[1:], quaternion_wxyz[0]])


class Rig:
    def __init__(self, identifier, directory):
        setup(identifier, directory, project=PROJECT)
        self.directory = Path(directory) / ".kinex/devices"
        self.profiles = {path.parent.name: yaml.safe_load(path.read_text())
                         for path in self.directory.glob("*/profile.yaml")}

    def mount(self, name, frame="base"):
        value = self.profiles[name]["frames"][frame]
        return xyzq(value["translation_m"] + value["rotation_xyzw"])

    def link(self, name, positions, link=None):
        """A URDF link pose in the device base; grippers name a finger link."""
        from kinex.lib.joints import JointModel
        profile = self.profiles[name]
        if profile["type"] == "gripper":
            profile = {"type": "joint", "name": name,
                       "model": {**profile["model"], "tip_link": link}}
        return xyzq(JointModel(profile, self.directory / name).fk(list(positions), link))

    def flange(self, arm, positions):
        """The arm tip in the world, through its calibrated mount."""
        return compose(self.mount(arm), self.link(arm, positions))
