"""behavior-dex's door friction rule (door_friction.py) on synthetic joints, without OmniGibson: a fixed object's part
keeps the static friction OmniGibson's coefficient gives it at rest, load-independent and at most the cap."""
import math

import numpy as np
import pytest

from rosbridge_behavior.door_friction import DYNAMIC, GRAVITY, DoorFriction, Part, static_friction

CAP = DoorFriction(hinge=1.5, slide=6.)
UP = np.array([0., 0., 1.])


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


def test_only_the_dexterous_embodiment_departs_from_omnigibsons_friction():
    from rosbridge_behavior.g1_sharpa import G1Sharpa
    from rosbridge_behavior.r1pro import R1Pro
    from rosbridge_behavior.r1pro_sharpa import R1ProSharpa

    assert R1Pro.door_friction is None  # the evaluator's gripper robot: OmniGibson's coefficient friction
    assert R1ProSharpa.door_friction == CAP
    assert getattr(G1Sharpa, "door_friction", None) is None
