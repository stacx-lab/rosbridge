"""BEHAVIOR's R1Pro with SharpaWave hands in place of its parallel grippers (dex_model).

Everything but the hands is the evaluator's R1Pro (r1pro.R1Pro): its base, torso and arm
controllers, its cameras at their full resolution and pose, its base mass and head optics. Each
hand is a joint device of its 22 joints, tracking absolute targets in radians on position drives
with Sharpa's Isaac gains; grasping is physical, by the fingers' contacts alone.
"""
import copy
import json

import yaml

from .dex_model import MODEL, SIDES
from .door_friction import DoorFriction
from .environment import data_root
from .r1pro import R1Pro

SIMULATOR = "behavior-dex"
PARTS = {"base": "base", "torso": "trunk", "left_arm": "arm_left", "left_hand": "gripper_left",
         "right_arm": "arm_right", "right_hand": "gripper_right"}
TORSO = [f"torso_joint{i}" for i in range(1, 5)]


class R1ProSharpa(R1Pro):
    parts = PARTS
    scalar_grippers = False
    pose_key = "r1pro"  # the presampled R1Pro start of each task instance
    # Fixed objects' doors, lids and drawers start at no more than 1.5 N m (hinges) or 6 N (slides) whatever their
    # mass, and hinged parts that gravity turns are counterbalanced (door_friction.py): a hand's grip slips where the
    # evaluator's assisted grasp holds, and lost the heavier fridge door's handle under OmniGibson's friction.
    door_friction = DoorFriction(hinge=1.5, slide=6.)

    def __init__(self):
        super().__init__()
        model = data_root(SIMULATOR) / "omnigibson-robot-assets/models" / MODEL
        self.drives = json.loads((model / "drives.json").read_text())
        self.hands = yaml.safe_load((model / f"{MODEL}.yaml").read_text())["manipulation"]["finger_joint_names"]
        # The evaluator's posture, by joint: its reset vector lists 6 base, 4 torso, 14 arm and 4
        # finger joints; the hands start open, at zero.
        self.pose = dict(zip(TORSO, self.config["reset_joint_pos"][6:10], strict=True))

    def robot_config(self):
        config = copy.deepcopy(super().robot_config())
        config["model"] = MODEL
        del config["reset_joint_pos"]  # the R1Pro's joint vector; configure sets ours by name
        config["grasping_mode"] = "physical"
        for side in SIDES:
            joints = self.hands[side]
            config["controller_config"][f"gripper_{side}"] = {
                "name": "JointController", "motor_type": "position", "use_delta_commands": False,
                "use_impedances": False, "command_input_limits": None, "command_output_limits": None,
                "isaac_kp": [self.drives[joint]["stiffness"] for joint in joints],
                "isaac_kd": [self.drives[joint]["damping"] for joint in joints]}
        return config

    def configure(self, robot):
        import torch

        super().configure(robot)
        robot.reset_joint_pos = torch.tensor([self.pose.get(name, 0.) for name in robot.dof_names_ordered])
