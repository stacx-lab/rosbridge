"""Without Kinex Python, skip its client checks without losing the remaining graph tests.

A module-level importorskip would end collection of every sibling module under
ROS's launch_testing pytest plugin, and an installed plugin answers
pytest_ignore_collect first, so collect_ignore is not consulted.
"""
import pytest

from tests.kinex_geometry import available

KINEX_MODULES = {"test_kinex.py", "test_kinex_frames.py", "test_device_nodes.py"}


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    if collection_path.name in KINEX_MODULES and not available():
        return True
    return None
