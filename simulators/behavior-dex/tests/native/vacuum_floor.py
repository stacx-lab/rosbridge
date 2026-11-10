"""Run by test_vacuum.py in a process of its own: BEHAVIOR's vacuum, the only particle remover by projection of its
taxonomy, standing under gravity on a floor box with dust, stepped without rendering as rosbridge steps an episode.
Writes to argv[1], as JSON, where the vacuum's removal box lies against the floor and what each trial removed:
  off      standing over a particle, switched off
  beside   switched on, with particles 5 mm and 15 mm beside the box's long side and 15 mm beyond its end
  plate    switched on, standing on a 5 cm plate over a particle of the floor
  pushed   switched on, pushed 0.9 m along the floor by a floating block that moves at 0.2 m/s, through a lane of
           particles and past particles 3 cm beside the lane
  on       switched on, standing over a particle
  lifted   switched on, held 1, 2, 20, 23, 30 and 33 mm above where it stands, over a particle each"""
import json
import sys
import traceback

import numpy as np

from rosbridge_behavior.environment import boot, rotation, values

boot("behavior-dex")
import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402
from omnigibson.object_states import ParticleRemover, ToggledOn  # noqa: E402
from omnigibson.utils import transform_utils as T  # noqa: E402

TOP, PLATE = .1, .05  # the floor's top and the plate's thickness (m)
SPEED, TRAVEL = .2, .9
LIFTS = (.001, .002, .02, .023, .03, .033)
BLOCK = .5  # the edge of the block that pushes the vacuum: 125 kg, floating 2 mm above the floor
MODE = sys.argv[2] if len(sys.argv) > 2 else "block"  # "self": the vacuum moves by its own velocity instead
env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [
    {"type": "PrimitiveObject", "name": "floor", "primitive_type": "Cube", "size": 1., "scale": [8., 4., TOP],
     "position": [0., 0., TOP / 2], "fixed_base": True},
    {"type": "PrimitiveObject", "name": "plate", "primitive_type": "Cube", "size": 1., "scale": [.6, .6, PLATE],
     "position": [-3., -1., TOP + PLATE / 2], "fixed_base": True},
    {"type": "PrimitiveObject", "name": "block", "primitive_type": "Cube", "size": BLOCK,
     "position": [3., 1.5, TOP + BLOCK / 2 + .002]},
    {"type": "DatasetObject", "name": "vacuum", "category": "vacuum", "model": "bdmsbr",
     "position": [0., 0., TOP + .35]}]})
scene, dt = env.scene, og.sim.get_sim_step_dt()
floor, plate, block, vacuum = (scene.object_registry("name", name) for name in ("floor", "plate", "block", "vacuum"))
remover = vacuum.states[ParticleRemover]
link = remover.link
mesh = next(iter(link.visual_meshes.values()))
report = {"errors": {}}


def pose(prim):
    position, orientation = prim.get_position_orientation()
    return values(position), rotation(orientation)


def box():
    """The removal box as the remover reads it: centre, axes (columns) and half extents in the world."""
    transform = values(mesh.scaled_transform)
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    return transform[:3, 3], transform[:3, :3] / scale, scale * float(mesh.get_attribute("size")) / 2


def stand(point, height=TOP, settle=20):
    """Stands the vacuum, turned so that it faces +x, with its removal box centred over the floor point, and lets
    it settle under gravity."""
    position = np.array([point[0], point[1], height + REST]) - YAW @ LOCAL * [1., 1., 0.]
    vacuum.set_position_orientation(position=th.tensor(position, dtype=th.float32),
                                    orientation=T.mat2quat(th.tensor(YAW, dtype=th.float32)))
    vacuum.keep_still()
    for _ in range(settle):
        og.sim.step()


def switch(on):
    vacuum.states[ToggledOn].set_value(on)


def left():
    return values(dust.get_particles_position_orientation()[0]).reshape(-1, 3) if dust.n_particles else np.zeros((0, 3))


def there(point):
    """Whether the particle generated at this floor point still exists."""
    return bool((np.linalg.norm(left()[:, :2] - np.asarray(point)[:2], axis=1) < 1e-3).any())


def tilt():
    return float(np.degrees(np.arccos(np.clip(pose(vacuum)[1][2, 2] / YAW[2, 2], -1., 1.))))


def trial(label, run):
    try:
        run()
    except Exception:
        report["errors"][label] = traceback.format_exc()[-3000:]
    print(label, json.dumps(report.get(label, report["errors"].get(label)))[:1500], flush=True)


with og.sim.render_on_step(False):
    block.disable_gravity()
    block.keep_still()
    for _ in range(40):  # the vacuum settles standing
        og.sim.step()
    origin, turn = pose(vacuum)
    centre, axes, half = box()
    order = np.argsort(-half)
    wide, deep = axes[:, order[0]], axes[:, order[1]]  # the box's long side and the direction a vacuum is pushed in
    # The vacuum's turn about z that points `deep` along +x, and the box's centre in its frame
    angle = -np.arctan2(deep[1], deep[0])
    YAW = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]) @ turn
    LOCAL, REST = turn.T @ (centre - origin), float(origin[2] - TOP)
    WIDE, DEEP = float(half[order[0]]), float(half[order[1]])
    report["standing"] = {
        "box top above the floor mm": round(1e3 * float(centre[2] + half[order[2]] - TOP), 2),
        "box bottom above the floor mm": round(1e3 * float(centre[2] - half[order[2]] - TOP), 2),
        "box mm": np.round(1e3 * 2 * half[order], 1).tolist(), "tilt deg": round(float(np.degrees(np.arccos(
            np.clip(turn[2, 2], -1., 1.)))), 2), "mass kg": round(float(vacuum.mass), 3),
        "root above the floor": round(REST, 4), "aabb": [np.round(values(v), 3).tolist() for v in vacuum.aabb]}
    print("standing", json.dumps(report["standing"]), flush=True)

    # Floor points, by where the vacuum will stand (its box centred at the station; its long side along y)
    OFF, BESIDE, UNDER, LANE, ON = (np.array(p) for p in ([-3., 1.], [-1.5, 1.], [-3., -1.], [0., -1.], [-1.5, -1.]))
    points = {"off": OFF, "near": BESIDE + [DEEP + .005, 0.], "side": BESIDE + [-DEEP - .015, 0.],
              "end": BESIDE + [0., WIDE + .015], "under": UNDER, "on": ON}
    lane = [LANE + [.3 + .08 * i, y] for i, y in enumerate((0., -.12, .12, -.06, .06, 0., WIDE - .02, -WIDE + .02))]
    outside = [LANE + [.4, WIDE + .03], LANE + [.6, -WIDE - .03]]
    lifted = [np.array([-.3 + .55 * i, 1.]) for i in range(len(LIFTS))]
    every = [*points.values(), *lane, *outside, *lifted]
    dust = scene.get_system("dust")
    dust.create_attachment_group(floor)
    dust.generate_group_particles(
        group=dust.get_group_name(floor), link_prim_paths=[floor.root_link.prim_path] * len(every),
        positions=th.tensor(np.array([[x, y, TOP] for x, y in every]), dtype=th.float32))
    for _ in range(3):
        og.sim.step()  # the dust system's own objects are initialized in a step
    report["particles"] = dust.n_particles

    def off():
        stand(OFF)
        switch(False)
        for _ in range(10):
            og.sim.step()
        report["off"] = {"left": there(points["off"]), "tilt deg": round(tilt(), 2)}

    def beside():
        switch(False)
        stand(BESIDE)
        centre, axes, half = box()
        report["beside"] = {"box off the station mm": np.round(1e3 * (centre[:2] - BESIDE), 2).tolist(),
                            "box top above the floor mm": round(1e3 * float(centre[2] + half[order[2]] - TOP), 2)}
        switch(True)
        for _ in range(10):
            og.sim.step()
        switch(False)
        report["beside"].update({f"{name} left": there(points[name]) for name in ("near", "side", "end")})

    def under():
        switch(False)
        stand(UNDER, height=TOP + PLATE)
        centre, axes, half = box()
        report["plate"] = {"box bottom above the floor mm": round(1e3 * float(centre[2] - half[order[2]] - TOP), 2)}
        switch(True)
        for _ in range(10):
            og.sim.step()
        switch(False)
        report["plate"]["left"] = there(points["under"])

    def pushed():
        """A block behind the vacuum moves at SPEED and pushes it; nothing else moves the vacuum."""
        switch(False)
        stand(LANE)
        start, _ = pose(vacuum)
        low, high = (values(v) for v in vacuum.aabb)
        behind = np.array([low[0] - BLOCK / 2 - .003, LANE[1], TOP + BLOCK / 2 + .002])
        block.set_position_orientation(position=th.tensor(behind, dtype=th.float32),
                                       orientation=th.tensor([0., 0., 0., 1.]))
        block.keep_still()
        for _ in range(5):
            og.sim.step()
        switch(True)
        grown = np.array([DEEP, WIDE]) + .01 - .002  # the box and its margin, less 2 mm
        result = {"particles": len(lane), "late": 0, "lag mm": 0., "tilt deg": 0., "lift mm": 0., "removed at": []}
        count = dust.n_particles
        for k in range(int(TRAVEL / (SPEED * dt))):
            if MODE == "block":
                block.set_linear_velocity(th.tensor([SPEED, 0., 0.]))
                block.set_angular_velocity(th.zeros(3))
            else:
                vacuum.set_linear_velocity(th.tensor([SPEED, 0., float(values(vacuum.get_linear_velocity())[2])]))
                vacuum.set_angular_velocity(th.zeros(3))
            og.sim.step()
            origin, turn = pose(vacuum)
            true = origin + turn @ LOCAL  # where the box is by the vacuum's pose
            read, axes, _ = box()
            remaining = left()
            if len(remaining):
                local = (remaining - true) @ axes[:, [order[1], order[0]]]
                result["late"] = max(result["late"], int((np.abs(local) < grown).all(axis=1).sum()))
            if dust.n_particles < count:
                result["removed at"].append([k, count - dust.n_particles, round(float(true[0] - LANE[0]), 3)])
                count = dust.n_particles
            result["lag mm"] = max(result["lag mm"], round(1e3 * float(np.linalg.norm(read - true)), 3))
            result["tilt deg"] = max(result["tilt deg"], round(tilt(), 2))
            result["lift mm"] = max(result["lift mm"], round(1e3 * abs(float(origin[2] - start[2])), 2))
        switch(False)
        origin, turn = pose(vacuum)
        result.update(travel=round(float(origin[0] - start[0]), 3), drift=round(float(origin[1] - start[1]), 3),
                      yaw=round(float(np.degrees(np.arctan2(*(YAW.T @ turn)[[1, 0], 0]))), 2),
                      left=sum(there(p) for p in lane), outside=sum(there(p) for p in outside))
        report["pushed"] = result
        block.set_position_orientation(position=th.tensor([3., 1.5, TOP + BLOCK / 2 + .002]),
                                       orientation=th.tensor([0., 0., 0., 1.]))
        block.keep_still()

    def on():
        switch(False)
        stand(ON)
        switch(True)
        for _ in range(10):
            og.sim.step()
        switch(False)
        report["on"] = {"left": there(points["on"])}

    def lifts():
        """Held still above where it stands (no gravity), over a particle of its own for each height."""
        report["lifted"] = {}
        for lift, station in zip(LIFTS, lifted):
            switch(False)
            stand(station)
            vacuum.disable_gravity()
            origin, turn = pose(vacuum)
            vacuum.set_position_orientation(position=th.tensor(origin + [0., 0., lift], dtype=th.float32))
            switch(True)
            for _ in range(6):
                vacuum.keep_still()
                og.sim.step()
            centre, axes, half = box()
            report["lifted"][f"{round(1e3 * lift)} mm"] = {
                "left": there(station),
                "box top above the floor mm": round(1e3 * float(centre[2] + half[order[2]] - TOP), 2),
                "box bottom above the floor mm": round(1e3 * float(centre[2] - half[order[2]] - TOP), 2)}
            switch(False)
            vacuum.enable_gravity()

    for label, run in (("off", off), ("beside", beside), ("plate", under), ("pushed", pushed), ("on", on),
                       ("lifted", lifts)):
        trial(label, run)
    report["left"] = dust.n_particles

with open(sys.argv[1], "w") as file:
    json.dump(report, file, indent=1)
print("REPORT WRITTEN", flush=True)
og.shutdown()
