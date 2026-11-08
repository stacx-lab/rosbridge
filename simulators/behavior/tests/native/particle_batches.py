"""Run by test_particles.py in a process of its own: adds and removes macro particles in an empty scene and writes, as
JSON to argv[1], how many physics handle updates (og.sim.update_handles) each call made and what it left."""
import json
import sys

from rosbridge_behavior.environment import boot

boot("behavior")
import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402

env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [
    {"type": "PrimitiveObject", "name": "box", "primitive_type": "Cube", "size": .4, "position": [0., 0., .2]}]})
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
    return report[label]


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
call("visual remove", dust.remove_particles, idxs=th.arange(10))
call("visual remove group", dust.remove_all_group_particles, group=group)
report["view valid"] = bool(og.sim.physics_sim_view.is_valid)
for _ in range(10):
    og.sim.step()
report["box z"] = float(box.get_position_orientation()[0][2])
with open(sys.argv[1], "w") as file:
    json.dump(report, file)
og.shutdown()
