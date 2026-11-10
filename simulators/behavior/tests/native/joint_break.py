"""Run by test_attachment.py in a process of its own: a shelf attached to a bookcase's back panel is pulled until the
attachment joint breaks, on steps that do not render, as rosbridge steps an episode. Writes as JSON to argv[1] what
the break left, after each stage, so that a process that dies leaves how far it came."""
import json
import math
import sys

from rosbridge_behavior.environment import boot

boot("behavior")
import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
import torch as th  # noqa: E402
from omnigibson.object_states import AttachedTo  # noqa: E402
from omnigibson.object_states.attached_to import m as attachment  # noqa: E402
from omnigibson.utils.physx_utils import apply_force_at_pos  # noqa: E402

env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [
    {"type": "DatasetObject", "name": "back", "category": "bookcase_back", "model": "gjsnrt",
     "abilities": {"attachable": {}}, "position": [0., 0., .01]},
    {"type": "DatasetObject", "name": "shelf", "category": "bookcase_shelf", "model": "ymtnqa",
     "abilities": {"attachable": {}}, "position": [0., 2., 1.]}]})
back, shelf = (env.scene.object_registry("name", name) for name in ("back", "shelf"))
attached = shelf.states[AttachedTo]
report = {}


def stage(label, **values):
    report[label] = values
    with open(sys.argv[1], "w") as file:
        json.dump(report, file)


def place():
    """The shelf at its slot in the back panel, as OmniGibson's own attachment test places it."""
    back.set_position_orientation(position=[0., 0., .01], orientation=[0., 0., 0., 1.])
    back.keep_still()
    shelf.set_position_orientation(position=[0., .0286, .2],
                                   orientation=T.euler2quat(th.tensor([0., -math.pi / 2., math.pi / 2.])))
    shelf.keep_still()


def exists(joint):
    """Whether the attachment joint's prim is on the stage."""
    return bool(joint and lazy.isaacsim.core.utils.prims.is_prim_path_valid(joint))


attachment.DEFAULT_BREAK_FORCE = 100.  # N, before the first attachment reads it; OmniGibson's is 5000
shelf.root_link.mass = .1  # so that its weight alone does not break the joint
place()
with og.sim.render_on_step(False):  # as rosbridge steps an episode
    for _ in range(15):
        og.sim.step()  # the shelf falls into its slot and attaches on contact
    joint = attached.attachment_joint_prim_path
    stage("attached", attached=bool(attached.get_value(back)), joint=exists(joint))
    # One step of pulling at each of its physics steps: the first breaks the joint, the others carry the shelf off,
    # or it would still be in its slot at the end of the step and attach again
    force, at = th.tensor([0., 0., 300.]), shelf.get_position_orientation()[0]
    pull = og.sim._physics_context._physx_interface.subscribe_physics_on_step_events(
        lambda _: apply_force_at_pos(shelf.root_link, force, at), pre_step=True, order=5)
    og.sim.step()
    pull = None  # ends the subscription
    stage("broken", attached=bool(attached.get_value(back)), joint=exists(joint),
          child=back.states[AttachedTo].children.get(joint.split("/")[-2]) is not None,
          rose=float(shelf.get_position_orientation()[0][2]) > .5)
    shelf.set_position_orientation(position=[0., 2., .5])  # to fall beside the back panel, not into its slot
    shelf.keep_still()
    for _ in range(30):
        og.sim.step()
    stage("later", steps=30, attached=bool(attached.get_value(back)))
    place()
    for _ in range(15):
        og.sim.step()
    stage("again", attached=bool(attached.get_value(back)), joint=exists(attached.attachment_joint_prim_path))
og.shutdown()
