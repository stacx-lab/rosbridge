"""Run by test_removers.py in a process of its own: a broom, moved by its velocity alone on steps that do not render,
passes over dust on a box. Writes as JSON to argv[1] how many particles were left, how many were at most still there
after a step that brought them inside the broom's removal box, and how far the box the broom's remover reads fell
behind the broom."""
import json
import sys

from rosbridge_behavior.environment import boot

boot("behavior")
import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402
from omnigibson.object_states import ParticleRemover  # noqa: E402
from omnigibson.object_states.particle_modifier import m as modifier  # noqa: E402

SPEED, CLEAR = .3, .25
env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [
    {"type": "PrimitiveObject", "name": "box", "primitive_type": "Cube", "size": .4, "position": [0., 0., .2]},
    {"type": "DatasetObject", "name": "broom", "category": "broom", "model": "tpyvbt", "position": [0., 3., 1.5]}]})
box, broom = (env.scene.object_registry("name", name) for name in ("box", "broom"))
dust = env.scene.get_system("dust")
dust.create_attachment_group(box)
grid = th.stack(th.meshgrid(th.arange(5) * .03 - .06, th.arange(4) * .03 - .045, indexing="ij"), dim=-1).reshape(-1, 2)
dust.generate_group_particles(group=dust.get_group_name(box), link_prim_paths=[box.root_link.prim_path] * len(grid),
                              positions=th.cat([grid, th.full((len(grid), 1), .4)], dim=1))
points = dust.get_particles_position_orientation()[0]
link = broom.states[ParticleRemover].link
broom.disable_gravity()
for _ in range(3):
    og.sim.step()  # the dust system's own objects are initialized in a step: before the pass, not in it


def put(position):
    """Sets the broom there, at rest, and returns its remover's box. Setting a pose writes Fabric too, so the box
    read after it is where the broom is."""
    broom.set_position_orientation(position=position, orientation=[0., 0., 0., 1.])
    broom.keep_still()
    return link.visual_aabb


aside = th.tensor([0., 3., 1.5])
low, high = put(aside)
# The broom starts with its box CLEAR short of the dust, centred on it across, and 8 mm above it, inside the
# remover's margin
start = aside + th.stack([points[:, 0].min() - CLEAR - high[0], -(low[1] + high[1]) / 2, .408 - low[2]])
low, high = put(start)
margin = modifier.PARTICLE_MODIFIER_ADJACENCY_AREA_MARGIN
steps = int((float(high[0] - low[0]) + CLEAR + .12 + .1) / (SPEED * og.sim.get_sim_step_dt()))
report = {"particles": dust.n_particles, "steps": steps, "late": 0, "lag": 0.}
with og.sim.render_on_step(False):  # as rosbridge steps an episode
    for _ in range(steps):
        broom.set_linear_velocity(th.tensor([SPEED, 0., 0.]))
        broom.set_angular_velocity(th.zeros(3))
        og.sim.step()
        moved = broom.get_position_orientation()[0] - start
        if dust.n_particles:
            left = dust.get_particles_position_orientation()[0]
            inside = ((low + moved - margin < left) & (left < high + moved + margin)).all(dim=-1)
            report["late"] = max(report["late"], int(inside.sum()))
        report["lag"] = max(report["lag"], float((link.visual_aabb[0] - low - moved).norm()))
report.update(left=dust.n_particles, travel=float(moved[0]), drift=float(moved[1:].abs().max()))
with open(sys.argv[1], "w") as file:
    json.dump(report, file)
og.shutdown()
