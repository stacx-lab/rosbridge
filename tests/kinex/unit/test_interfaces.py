"""rosbridge's copy of custom_interfaces against Kinex's."""
from pathlib import Path

import pytest


def test_the_interfaces_are_the_ones_kinex_builds():
    """Both projects build custom_interfaces from their own copy; the same definitions make the same types."""
    project = Path(__file__).resolve().parents[3]
    own = project / "ros2/custom_interfaces/srv"
    kinex = project.parent / "kinex2.0/packages/core/assets/ros/custom_interfaces/srv"
    if not kinex.is_dir():
        pytest.skip("needs a Kinex checkout next to this one")
    assert sorted(p.name for p in own.iterdir()) == sorted(p.name for p in kinex.iterdir()) == ["Json.srv", "Wait.srv"]
    for definition in own.iterdir():
        assert definition.read_bytes() == (kinex / definition.name).read_bytes()
