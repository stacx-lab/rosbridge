"""Load-independent friction for the doors, lids and drawers of fixed BEHAVIOR objects: the embodiment option
door_friction (behavior-dex; the evaluator's R1Pro keeps OmniGibson's).

OmniGibson gives every revolute joint a friction coefficient of 0.2 and every prismatic joint 0.3
(DatasetObject._initialize), and PhysX resists a joint's motion with that coefficient times the force and torque the
joint transmits. A part's own weight is most of that load, so the effort that starts a part grows with its mass, and
below it the part creeps instead of holding. Measured in five scenes as the joint effort, ramped, at which a part has
moved 0.02 rad or 5 mm: room doors 10 to 17 N m, fridge, washer and dryer doors 2.6 to 6.6, a car
trunk 18 and a grill lid 11 to 13 to lift, drawers, racks and sliding doors up to 34 N. The evaluator's assisted grasp is a rigid joint; a Sharpa hand's grip is not.

With the option each single-axis joint of a fixed object gets PhysX's load-independent friction instead, a static
and a dynamic friction effort, which replace the coefficient; movable objects and robots are untouched:

- static: what OmniGibson's coefficient gives the part at rest, its coefficient times the weight its joint carries
  plus that weight's moment about the joint (PhysX adds the two), at most the option's cap, 1.5 N m for a hinge and
  6 N for a slide. Parts lighter than the cap keep what they have at rest, without the creep.
- dynamic: DYNAMIC (0.7) of the static.

Measured over five scenes: doors start at 1.6 to 2.2 N m and drawers at 6.2 to 7.8 N whatever their mass; nothing
moves in 10 s idle; locked doors keep their locking limits.
"""
from typing import NamedTuple

import numpy as np

GRAVITY = 9.81  # m/s^2, OmniGibson's simulator
DYNAMIC = .7  # the dynamic friction effort, as a fraction of the static


class DoorFriction(NamedTuple):
    """The most static friction effort a fixed object's joint keeps: a hinge's in N m, a slide's in N."""
    hinge: float
    slide: float


class Part(NamedTuple):
    """One joint of a fixed object and the links it moves, in the world frame at its current position."""
    revolute: bool
    coefficient: float  # OmniGibson's friction coefficient
    position: float  # the joint's position (rad or m) and its range
    lower: float
    upper: float
    origin: np.ndarray  # a point on the joint's axis
    axis: np.ndarray  # the axis, a unit vector
    masses: np.ndarray  # the moving links' masses (kg) and centres of mass
    centres: np.ndarray


def moment(part):
    """The moving links' first moment of mass about the joint's origin (kg m)."""
    return (part.masses[:, None] * (part.centres - part.origin)).sum(0)


def static_friction(part, friction):
    """The part's static friction effort: its coefficient times the weight its joint carries at rest plus that weight's
    moment about the joint, as PhysX loads OmniGibson's coefficient, at most the cap for its joint type."""
    weight = part.masses.sum() * GRAVITY
    torque = GRAVITY * float(np.hypot(*moment(part)[:2]))  # |first moment x gravity|
    return min(part.coefficient * (weight + torque), friction.hinge if part.revolute else friction.slide)


def _matrix(quaternion_xyzw):
    x, y, z, w = np.asarray(quaternion_xyzw, float) / np.linalg.norm(quaternion_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def parts(obj):
    """The single-axis joints of an OmniGibson object as Parts, by joint name, with their DOF indices and the names
    of the links each moves."""
    import networkx as nx
    from omnigibson.utils.constants import JointType

    by_path = {link.prim_path: name for name, link in obj.links.items()}
    q = _array(obj.get_joint_positions())
    scale = _array(obj.scale)
    found = {}
    for name, joint in obj.joints.items():
        if joint.joint_type not in (JointType.JOINT_REVOLUTE, JointType.JOINT_PRISMATIC) or joint.n_dof != 1:
            continue
        parent, child = by_path.get(str(joint.body0)), by_path.get(str(joint.body1))
        if parent is None or child is None:
            continue
        dof = int(joint.dof_indices[0])
        links = [child, *nx.descendants(obj.articulation_tree, child)]
        position, orientation = obj.links[parent].get_position_orientation()
        turn = _matrix(_array(orientation))
        axis = np.zeros(3)
        axis["XYZ".index(joint.axis)] = 1.
        masses, centres = [], []
        for link_name in links:
            link = obj.links[link_name]
            link_position, link_orientation = link.get_position_orientation()
            masses.append(float(link.mass))
            centres.append(_array(link_position) + _matrix(_array(link_orientation)) @ _array(link.center_of_mass))
        # The joint frame's offset on its parent link is in the object's scaled frame.
        found[name] = (dof, links, Part(
            # OmniGibson writes its coefficient to the joint (JointPrim.friction); reading it from PhysX would go
            # through the tensor API's deprecated coefficient getter.
            revolute=joint.joint_type == JointType.JOINT_REVOLUTE,
            coefficient=float(joint.get_attribute("physxJoint:jointFriction") or 0.),
            position=float(q[dof]), lower=float(joint.lower_limit), upper=float(joint.upper_limit),
            origin=_array(position) + turn @ (scale * _array(joint.local_position_0)),
            axis=turn @ _matrix(_array(joint.local_orientation_0)) @ axis,
            masses=np.asarray(masses), centres=np.asarray(centres)))
    return found


def ease_doors(scene, friction):
    """Give the joints of the scene's fixed objects load-independent friction (static_friction); returns
    {object: {joint: static friction effort}}."""
    import torch
    from omnigibson.objects.dataset_object import DatasetObject

    eased = {}
    for obj in scene.objects:
        if not isinstance(obj, DatasetObject) or not obj.fixed_base or obj.n_dof == 0:
            continue
        found = parts(obj)
        if not found:
            continue
        # Once a DOF has friction efforts, PhysX no longer applies its coefficient: a room door started at 2.1 N m with
        # the coefficient left at 0.2 as with it at 0, and at 11 N m on the coefficient alone.
        view = obj._articulation_view._physics_view
        properties = torch.as_tensor(_array(view.get_dof_friction_properties()), dtype=torch.float32)
        eased[obj.name] = {}
        for name, (dof, links, part) in found.items():
            static = static_friction(part, friction)
            properties[0, dof, 0], properties[0, dof, 1] = static, DYNAMIC * static
            eased[obj.name][name] = round(static, 4)
        view.set_dof_friction_properties(properties, torch.arange(view.count, dtype=torch.int32))
    return eased
