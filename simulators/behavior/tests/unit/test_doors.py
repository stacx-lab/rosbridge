"""Doors out of a partial BEHAVIOR scene's loaded rooms are closed and locked. OmniGibson is faked: a room map that
splits the world at x = 0 and y = 0, and doors with a closed-frame box, a pose and joints."""
import math
from types import SimpleNamespace

import pytest

from rosbridge_behavior.environment import DOOR_LOCKED, doors_out, lock_doors_out


class Rooms:
    """kitchen_0 at x > 0, hall_0 at x < 0 and y > 0, no room at x < 0 and y < 0 (outside)."""
    def get_room_instance_by_point(self, xy):
        x, y = xy
        return "kitchen_0" if x > 0 else "hall_0" if y > 0 else None


class Joint:
    """OmniGibson's limits, read back as none when equal."""
    def __init__(self, lower, upper, position):
        self._limits, self.position = [lower, upper], position

    @property
    def lower_limit(self):
        lower, upper = self._limits
        return -1000. if lower == upper else lower

    @lower_limit.setter
    def lower_limit(self, value):
        self._limits = [value, self.upper_limit]
        assert self._limits[0] < self._limits[1], "the range collapsed"

    @property
    def upper_limit(self):
        lower, upper = self._limits
        return 1000. if lower == upper else upper

    @upper_limit.setter
    def upper_limit(self, value):
        self._limits = [self.lower_limit, value]
        assert self._limits[0] < self._limits[1], "the range collapsed"

    def set_pos(self, position):
        self.position = position


def door(name, x, y, yaw=0., size=(.05, .9, 2.), category="door", joints=None):
    """A door whose closed box is thin along its own x, turned by yaw about z."""
    half = yaw / 2
    return SimpleNamespace(name=name, category=category, native_bbox=size, scale=(1., 1., 1.),
                           scaled_bbox_center_in_base_frame=(0., 0., 1.),
                           get_position_orientation=lambda: ((x, y, 0.), (0., 0., math.sin(half), math.cos(half))),
                           joints={"j": Joint(0., 1.57, .3)} if joints is None else joints)


def scene(*objects):
    return SimpleNamespace(objects=objects, seg_map=Rooms())


def test_a_door_with_a_loaded_room_on_one_side_only_is_out():
    kitchen_hall = door("kitchen_hall", 0., 5.)  # faces along x: kitchen one side, hall the other
    turned = door("turned", 0., 5., yaw=math.pi / 2, size=(.9, .05, 2.))  # thin along its own y, still along x
    outside = door("outside", -5., 0., yaw=math.pi / 2)  # faces along y: hall above, outside below
    handle = door("handle", 0., 5., joints={})  # no joints: not a door one opens
    shelf = door("shelf", 0., 5., category="shelf")
    world = scene(kitchen_hall, turned, outside, handle, shelf)
    assert [d.name for d in doors_out(world, ["kitchen_0"])] == ["kitchen_hall", "turned"]
    assert [d.name for d in doors_out(world, ["hall_0"])] == ["kitchen_hall", "turned", "outside"]
    assert [d.name for d in doors_out(world, ["kitchen_0", "hall_0"])] == ["outside"]
    assert doors_out(world, ["bathroom_0"]) == []  # neither side loaded: out of reach


@pytest.mark.parametrize("lower,upper", [(0., 1.57), (-1.57, 0.), (-1.57, 1.57), (0., .9)])
def test_a_door_out_is_closed_and_locked_whatever_way_it_opens(lower, upper):
    joint = Joint(lower, upper, .4)
    out, inside = door("out", 0., 5., joints={"j": joint}), door("inside", 5., 5.)
    assert lock_doors_out(scene(out, inside), ["kitchen_0"]) == [out]
    assert joint.position == 0. and (joint.lower_limit, joint.upper_limit) == (0., DOOR_LOCKED)
    assert inside.joints["j"].position == .3 and inside.joints["j"].upper_limit == 1.57
