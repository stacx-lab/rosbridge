"""rosbridge's interface package against Kinex's."""
from pathlib import Path

import pytest


def test_the_interfaces_keep_the_definitions_of_kinexs():
    """rosbridge's Wait and Json are its own types; their requests and replies stay those of Kinex's."""
    project = Path(__file__).resolve().parents[3]
    own = project / "ros2/rosbridge_interfaces/srv"
    kinex = project.parent / "kinex2.0/packages/core/assets/ros/kinex_interfaces/srv"
    if not kinex.is_dir():
        pytest.skip("needs a Kinex checkout next to this one")
    assert sorted(p.name for p in own.iterdir()) == ["Json.srv", "Wait.srv"]
    for definition in own.iterdir():
        assert definition.read_bytes() == (kinex / definition.name).read_bytes()
