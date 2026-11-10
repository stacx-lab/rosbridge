"""behavior-dex's door friction rule (door_friction.py) on synthetic joints, without OmniGibson: a fixed object's part
keeps the static friction OmniGibson's coefficient gives it at rest, load-independent and at most the cap, and
gravity goes off on a hinged part it would turn."""
import math

import numpy as np
import pytest

from rosbridge_behavior.door_friction import (COUNTERBALANCE, DYNAMIC, GRAVITY, DoorFriction, Part, gravity_load, rule,
                                              static_friction)

CAP = DoorFriction(hinge=1.5, slide=6.)
UP, ACROSS = np.array([0., 0., 1.]), np.array([0., 1., 0.])


def hinge(mass, centre, axis=UP, lower=0., upper=math.pi / 2, position=0., origin=(0., 0., 0.), coefficient=.2):
    """A one-link part on a revolute joint at origin, its centre of mass at centre, both at the joint's position."""
    return Part(revolute=True, coefficient=coefficient, position=position, lower=lower, upper=upper,
                origin=np.asarray(origin, float), axis=np.asarray(axis, float) / np.linalg.norm(axis),
                masses=np.array([mass]), centres=np.array([centre], float))


def slide(mass, axis=(1., 0., 0.), lower=0., upper=.4, coefficient=.3):
    return Part(revolute=False, coefficient=coefficient, position=0., lower=lower, upper=upper,
                origin=np.zeros(3), axis=np.asarray(axis, float) / np.linalg.norm(axis),
                masses=np.array([mass]), centres=np.zeros((1, 3)))  # a drawer's joint frame sits at its centre


def test_heavy_doors_start_at_the_cap_and_light_parts_keep_their_friction_at_rest():
    room_door = hinge(15.2, (.42, 0., 1.))  # 15 kg leaf on a vertical hinge: 0.2 * (149 + 62) = 42 N m today
    assert static_friction(room_door, CAP) == 1.5
    cabinet_door = hinge(.16, (.15, 0., .3))  # 0.16 kg: 0.2 * (1.57 + 0.24) = 0.36 N m, below the cap: kept
    assert static_friction(cabinet_door, CAP) == pytest.approx(.2 * .16 * GRAVITY * 1.15)
    assert static_friction(slide(3.5), CAP) == 6.  # a 3.5 kg drawer: 0.3 * 34 N today
    assert static_friction(slide(.106), CAP) == pytest.approx(.3 * .106 * GRAVITY)  # an oven rack: 0.3 * 1.04 N
    assert static_friction(hinge(.16, (.15, 0., .3), coefficient=0.), CAP) == 0.  # a joint without friction keeps none
    assert 0 < DYNAMIC < 1
    assert not any(rule(part, CAP)[1] for part in (room_door, cabinet_door, slide(3.5)))  # gravity does not turn them


def test_parts_gravity_turns_are_counterbalanced_whatever_their_mass():
    trunk = hinge(3.11, (-.36, 0., -.05), axis=ACROSS, upper=1.885)  # a car trunk on a horizontal hinge
    oven_door = hinge(4.34, (0., -.02, .22), axis=(1., 0., 0.))  # upright when closed, falls open about x
    dishwasher_door = hinge(6.36, (0., -.03, .31), axis=(-1., 0., 0.), lower=-math.pi / 2, upper=0.)
    small_lid = hinge(.3, (0., .12, .02), axis=(1., 0., 0.), upper=1.6)  # 0.3 kg, 12 cm from its hinge
    for part in (trunk, oven_door, dishwasher_door, small_lid):
        static, balanced = rule(part, CAP)
        assert balanced and gravity_load(part) > COUNTERBALANCE * static
    # A part on a vertical slide keeps its weight: a deep fryer's basket comes down into the fryer, as today.
    basket = slide(.9, axis=(0., 0., 1.), coefficient=.3)
    assert gravity_load(basket) == pytest.approx(.9 * GRAVITY) and not rule(basket, CAP)[1]


def test_a_hinge_out_of_plumb_or_a_lid_that_barely_tilts_is_held_by_friction_alone():
    # A 10 kg room door whose hinge leans 1 mrad, as in the scenes: 0.05 N m of gravity against 1.5 N m.
    leaning = hinge(9.9, (.51, 0., 1.), axis=(.001, 0., 1.), upper=1.745)
    assert gravity_load(leaning) == pytest.approx(9.9 * GRAVITY * .51 * .001, rel=.05)
    assert rule(leaning, CAP) == (1.5, False)
    # A flap balanced over its hinge that turns only 0.1 rad either way: gravity is evaluated over its range only.
    balanced_flap = hinge(1., (0., 0., .2), axis=(1., 0., 0.), lower=-.1, upper=.1)
    assert gravity_load(balanced_flap) == pytest.approx(GRAVITY * .2 * math.sin(.1), rel=1e-3)
    assert not rule(balanced_flap, CAP)[1]
    # Over a whole turn it would be counterbalanced.
    assert rule(balanced_flap._replace(lower=-math.pi, upper=math.pi), CAP)[1]


def test_the_rule_does_not_depend_on_which_way_an_axis_points_or_where_the_part_stands_in_its_range():
    trunk = hinge(3.11, (-.36, 0., -.05), axis=ACROSS, upper=1.885)
    flipped = trunk._replace(axis=-trunk.axis, lower=-1.885, upper=0.)
    assert rule(flipped, CAP) == rule(trunk, CAP)
    assert gravity_load(flipped) == pytest.approx(gravity_load(trunk))
    # The same trunk read half open: its centre of mass turned by 0.9 rad about the hinge, the range shifted with it.
    c, s = math.cos(.9), math.sin(.9)
    turned = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]) @ trunk.centres[0]
    half_open = trunk._replace(position=.9, centres=np.array([turned]))
    assert gravity_load(half_open) == pytest.approx(gravity_load(trunk), rel=1e-3)


def test_only_the_dexterous_embodiment_departs_from_omnigibsons_friction():
    from rosbridge_behavior.g1_sharpa import G1Sharpa
    from rosbridge_behavior.r1pro import R1Pro
    from rosbridge_behavior.r1pro_sharpa import R1ProSharpa

    assert R1Pro.door_friction is None  # the evaluator's gripper robot: OmniGibson's coefficient friction
    assert R1ProSharpa.door_friction == CAP
    assert getattr(G1Sharpa, "door_friction", None) is None
