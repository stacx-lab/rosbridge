"""Run by test_behavior.py in a process of its own: an episode start, seen from inside the backend's process.
Writes to argv[1], as JSON:
  events    the seeds seed_everything received and where the environment was built, in order
  teleports the robot poses env.reset set, [[x, y, z], [qx, qy, qz, qw], frame]
  robot     the robot's scene-frame pose after reset, [x, y, z, qx, qy, qz, qw]
  sampled   the instance's presampled robot pose 0, the one upstream's env.reset places it at
  max_steps, timeout   the episode's step limit and upstream's Timeout limit"""
import json
import sys

import torch as th

from rosbridge_behavior.backend import R1ProSim
from rosbridge_behavior.environment import boot

boot()
import omnigibson as og  # noqa: E402
import omnigibson.eval.utils.eval_utils as eval_utils  # noqa: E402
from omnigibson.tasks.behavior_task import get_presampled_robot_poses  # noqa: E402

events, teleports, seed_everything, environment = [], [], eval_utils.seed_everything, og.Environment
eval_utils.seed_everything = lambda seed, *args, **kwargs: (events.append(["seed", seed]),
                                                            seed_everything(seed, *args, **kwargs))[1]


def build(*args, **kwargs):
    """The environment, recording the robot poses its env.reset sets."""
    events.append(["environment"])
    env = environment(*args, **kwargs)
    robot, reset, place, inside = env.scene.robots[0], env.reset, env.scene.robots[0].set_position_orientation, []

    def placed(position=None, orientation=None, frame="world", **options):
        if inside:
            teleports.append([th.as_tensor(position).tolist(), th.as_tensor(orientation).tolist(), frame])
        return place(position, orientation, frame=frame, **options)

    def resetting(*args, **kwargs):
        inside.append(True)
        try:
            return reset(*args, **kwargs)
        finally:
            inside.clear()
    robot.set_position_orientation, env.reset = placed, resetting
    return env


og.Environment = build
sim = R1ProSim()
sim.reset("turning_on_radio", 0)
robot, scene = sim.episode.robot, sim.episode.env.scene
position, orientation = robot.get_position_orientation(frame="scene")
pose = get_presampled_robot_poses(scene.get_task_metadata(key="robot_poses"), robot_model=robot.model)[0]
with open(sys.argv[1], "w") as stream:
    json.dump({"events": events, "teleports": teleports, "robot": th.cat([position, orientation]).tolist(),
               "sampled": th.cat([th.as_tensor(pose["position"]), th.as_tensor(pose["orientation"])]).tolist(),
               "max_steps": sim.max_steps, "timeout": sim.episode.max_steps}, stream)
    stream.flush()
sim.close()  # OmniGibson's shutdown ends the process
