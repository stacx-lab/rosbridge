"""The scene's room doors (rosbridge_behavior.scene_doors): which way a hinged leaf swings from its joint, the closed
leaf from an open one, which doors open onto no floor, and the world text's door paragraph. The scene reads (rays,
collision boxes) run in the world; tests/native and native probes check them there."""
import math

import numpy as np
import pytest

from rosbridge_behavior import scene_doors as sd


def leaf_points(width=.8, thickness=.04, height=2., hinge=(0., 0.), angle=0.):
    """A box leaf hinged on a vertical axis at `hinge`, extending along +x when closed, turned by angle about +z."""
    xs, ys, zs = np.meshgrid([0., width], [-thickness / 2, thickness / 2], [0., height], indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    points = points @ sd.turn([0., 0., 1.], angle).T
    return points + [hinge[0], hinge[1], 0.]


@pytest.mark.parametrize("axis, limits, side", [
    ((0., 0., 1.), (0., 1.57), 1),     # positive turn about +z moves a leaf along +x toward +y
    ((0., 0., 1.), (-1.57, 0.), -1),
    ((0., 0., -1.), (0., 1.57), -1),
    ((0., 0., -1.), (-1.57, 0.), 1),
])
def test_a_hinged_leaf_swings_toward_the_side_its_opening_limit_turns_it(axis, limits, side):
    g = sd.closed_geometry(leaf_points(), np.array(axis), np.zeros(3), 0., True)
    normal_side = 1 if g["normal"][1] > 0 else -1  # the SVD may pick either normal; the answer follows it
    assert sd.opening_side(np.array(axis), np.zeros(3), g["centre"], g["normal"], *limits) == side * normal_side


def test_an_open_leaf_is_taken_closed_from_its_joint_position():
    angle = math.radians(80)
    g = sd.closed_geometry(leaf_points(hinge=(1., 2.), angle=angle), np.array([0., 0., 1.]), np.array([1., 2., 0.]),
                           angle, True)
    assert g["centre"] == pytest.approx([1.4, 2., 1.], abs=1e-9)
    assert abs(g["normal"][1]) == pytest.approx(1.) and abs(g["wide"][0]) == pytest.approx(1.)
    assert (g["width"], g["thickness"], g["bottom"], g["height"]) == pytest.approx((.8, .04, 0., 2.))
    slid = sd.closed_geometry(leaf_points() + [.5, 0., 0.], np.array([1., 0., 0.]), np.zeros(3), .5, False)
    assert slid["centre"] == pytest.approx([.4, 0., 1.])


@pytest.mark.parametrize("floors, beyond", [
    ({1: 9, -1: 0}, -1), ({1: 0, -1: 9}, 1), ({1: 5, -1: 0}, -1),
    ({1: 9, -1: 9}, None), ({1: 9, -1: 1}, None), ({1: 4, -1: 0}, None), ({1: 0, -1: 0}, None),
])
def test_a_door_opens_onto_no_floor_only_with_floor_on_its_other_side(floors, beyond):
    assert sd.classify(floors) == beyond


def door(centre, normal, swing, no_floor=None, blocked=False, rooms=("bedroom_0", "bathroom_0"), category="door",
         kind="hinged", pivot=None):
    d = {"object": "door_x_0", "category": category, "joint": "j_leaf", "kind": kind, "swing": swing,
         "centre": list(centre), "normal": list(normal), "wide": [-normal[1], normal[0], 0.], "width": .84,
         "pivot": list(pivot if pivot is not None else centre), "no_floor_side": no_floor,
         "sides": {"1": {"floor_points": 9, "room": rooms[0]}, "-1": {"floor_points": 0 if no_floor else 9,
                                                                    "room": rooms[1]}}}
    if blocked:
        d["blocked"] = "/World/rosbridge_scene_edge/door_x_0_j_leaf"
    return d


def test_the_door_paragraph_states_position_swing_and_boundary_in_the_start_frame():
    # The robot starts at (10, 0) facing +y: world +y is base/odom +x, world -x is base/odom +y.
    origin = (np.array([10., 0., 0.]), sd.turn([0., 0., 1.], math.pi / 2))
    inner = door((10., 2., 1.), (0., 1., 0.), swing=1, pivot=(10.4, 2., 1.))       # 2 m ahead, swings away from the start
    edge = door((8., 0., 1.), (-1., 0., 0.), swing=-1, no_floor=1, blocked=True)  # 2 m to the left, void beyond
    text = sd.describe([inner, edge], origin)
    assert text.startswith(sd.HEADER + ": (1) ")
    assert ("a door at (2.0, 0.0), 0.84 m wide, between bedroom_0 (its +x side) and bathroom_0 (its -x side), hinged at "
            "(2.0, -0.4): it swings open toward its +x side, so it opens by a push from its -x side and by a pull from its "
            "+x side; the start position is on its push side") in text
    assert ("a door at (0.0, 2.0), 0.84 m wide, hinged at (0.0, 2.0): it swings open toward its -y side, so it opens by a "
            "push from its +y side and by a pull from its -y side; the start position is on its pull side; past its +y side "
            f"there is no floor (it leads out of the loaded rooms), and {sd.BLOCKED}") in text
    first, second = text.split("; (2) ")
    assert first.endswith(sd.BLOCKED) and "between" not in first  # sorted by position; no "between" at a boundary door
    unblocked = sd.describe([door((8., 0., 1.), (-1., 0., 0.), swing=-1, no_floor=1)], origin)
    assert "walking through it drops the robot out of the world" in unblocked
    assert sd.describe([], origin) == sd.HEADER + ": none."
    sliding = sd.describe([door((10., 2., 1.), (0., 1., 0.), 0, category="sliding_door", kind="sliding")], origin)
    assert "a sliding door at (2.0, 0.0)" in sliding and "it slides sideways along its wall" in sliding


def test_a_leaf_the_code_could_not_read_is_left_out_of_the_text():
    origin = (np.zeros(3), np.eye(3))
    text = sd.describe([{"object": "door_y_0", "category": "door", "joint": "j", "error": "ValueError: x"},
                        door((2., 0., 1.), (1., 0., 0.), 1)], origin)
    assert "(1) a door at (2.0, 0.0)" in text and "(2)" not in text


def test_doors_off_axis_are_named_by_their_bearing():
    assert sd._bearing([1., .2]) == "+x" and sd._bearing([0., -1.]) == "-y" and sd._bearing([-1., -.1]) == "-x"
    assert sd._bearing([math.cos(math.radians(-73)), math.sin(math.radians(-73))]) == "bearing -73 degrees"


def test_the_record_and_text_go_to_the_run_directory(tmp_path):
    import json

    origin = (np.zeros(3), np.eye(3))
    text = sd.write([door((2., 0., 1.), (1., 0., 0.), 1)], origin, folder=tmp_path)
    assert (tmp_path / "scene-doors.txt").read_text() == text + "\n"
    assert json.loads((tmp_path / "scene-doors.json").read_text())["text"] == text


def test_a_handle_on_a_door_is_not_a_leaf():
    handle = sd.closed_geometry(leaf_points(width=.15, height=.04), np.array([1., 0., 0.]), np.zeros(3), 0., True)
    assert not sd.is_leaf(handle) and sd.is_leaf(sd.closed_geometry(leaf_points(), np.array([0., 0., 1.]), np.zeros(3),
                                                                    0., True))


def test_a_lower_floor_beyond_a_door_keeps_it_open_but_walls_without_floor_do_not():
    """A floor below the band (a step down) is part of the loaded scene: not walled off. Something above floor level with
    no floor under it (a wall, furniture of another room) does not make a floor."""
    assert sd.classify({1: 9, -1: 0}, {1: 0, -1: 0}) == -1
    assert sd.classify({1: 9, -1: 0}, {1: 0, -1: 3}) == -1  # a few low pieces (clean_a_trumpet's door_pxhbim_2) are no floor
    assert sd.classify({1: 9, -1: 0}, {1: 0, -1: sd.FLOOR_MIN_POINTS}) is None  # a lower floor
    assert sd.classify({1: 0, -1: 7}, {1: 0, -1: 0}) == 1

