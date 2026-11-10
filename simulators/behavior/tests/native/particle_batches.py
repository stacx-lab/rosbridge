"""Run by test_particles.py in a process of its own: adds and removes macro particles in an empty scene beside a
jointed cabinet and writes, as JSON to argv[1], how many physics handle updates (og.sim.update_handles) each call
made, what it left, and how far the cabinet's joints, set afterwards, read from where they were set."""
import json
import sys

from rosbridge_behavior.environment import boot

boot("behavior")
import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402

env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [
    {"type": "PrimitiveObject", "name": "box", "primitive_type": "Cube", "size": .4, "position": [0., 0., .2]},
    {"type": "DatasetObject", "name": "cabinet", "category": "bottom_cabinet", "model": "immwzb", "fixed_base": True,
     "position": [2., 0., 1.]}]})
Simulator, updates = type(og.sim), [0]
update_handles = Simulator.update_handles


def counted(self):
    updates[-1] += 1
    update_handles(self)


Simulator.update_handles = counted
report = {}


def call(label, function, **kwargs):
    updates.append(0)
    function(**kwargs)
    report[label] = {"updates": updates[-1], "left": function.__self__.n_particles}
    updates.append(0)  # the joint check below is not part of the call
    report[label]["joints"] = joint_error(.1 + .1 * len(report))
    return report[label]


def joint_error(fraction):
    """Sets the cabinet's joints this far into their ranges and returns how far they read from there, or the error.
    Deleting any prim drops the handle of every articulation view, and through a dropped handle a joint write does
    nothing and a joint read fails."""
    cabinet = env.scene.object_registry("name", "cabinet")
    target = cabinet.joint_lower_limits + fraction * (cabinet.joint_upper_limits - cabinet.joint_lower_limits)
    try:
        cabinet.set_joint_positions(target)
        return float((cabinet.get_joint_positions() - target).abs().max())
    except Exception as error:  # reported, so that the test names the call it follows
        return f"{type(error).__name__}: {error}"


popcorn, dust = env.scene.get_system("cooked__popcorn"), env.scene.get_system("dust")
grid = th.stack(th.meshgrid(th.arange(5) * .06 - .12, th.arange(4) * .06 - .09, indexing="ij"), dim=-1).reshape(-1, 2)
positions = th.cat([grid, th.ones(len(grid), 1)], dim=1)
generated = call("generate", popcorn.generate_particles, positions=positions)
# Read back as the centres of the particles' sphere approximations, offset from where they were placed
generated["error"] = float((popcorn.get_particles_position_orientation()[0] - positions).norm(dim=1).max())
generated["offset"] = float(popcorn._particle_offset.norm())
og.sim.step()
call("remove", popcorn.remove_particles, idxs=th.arange(10))
call("sync up", popcorn._sync_particles, n_particles=15)
call("sync down", popcorn._sync_particles, n_particles=5)
call("remove all", popcorn.remove_all_particles)

box = env.scene.object_registry("name", "box")
dust.create_attachment_group(box)
group = dust.get_group_name(box)
top = th.cat([grid * .5, th.full((len(grid), 1), .4)], dim=1)
call("visual generate", dust.generate_group_particles, group=group, positions=top,
     link_prim_paths=[box.root_link.prim_path] * len(top))
saved = dust.dump_state(serialized=False)
call("visual remove", dust.remove_particles, idxs=th.arange(10))
# Restoring a state removes the group whose particles differ and creates it again, as loading a task instance does
call("visual restore", dust.load_state, state=saved, serialized=False)
call("visual remove group", dust.remove_all_group_particles, group=group)
for _ in range(10):
    og.sim.step()
report["box z"] = float(box.get_position_orientation()[0][2])
with open(sys.argv[1], "w") as file:
    json.dump(report, file)
og.shutdown()
