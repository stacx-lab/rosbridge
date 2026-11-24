"""Synthetic position/scalar backend for shared conformance tests."""
import threading
import numpy as np
from rosbridge.core import Action, ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory


class ToyBackend:
    """A differently shaped backend: a reordered arm group and a scalar gripper."""
    layout = ActionLayout(("slide", "finger", "rotate"), (
        CommandGroup("axes", (2, 0), (-2., -3.), (2., 3.)),
        CommandGroup("finger", (1,), (-1.,), (1.,), scalar=True),
    ))
    devices = device_inventory(layout, {})
    default_task = "toy"
    cameras = spectators = ()
    reward = 0.
    period = 0.1

    def __init__(self, max_steps=None, early_stop=False):
        self.owner = threading.get_ident()
        self.max_steps, self.early_stop = max_steps, early_stop
        self.closed = False
        self.reset("toy", 0, False)

    def configure_spectators(self, width, height, views=None):
        pass

    def reset(self, task, seed, randomize):
        assert threading.get_ident() == self.owner
        if task not in ("toy", "succeeds", "fails", "stumbles"):
            raise ValueError("unknown task")
        self.task, self.seed = task, seed
        self.description = {"instruction": f"Move the toy's axes ({task}, seed {seed})."}
        self.steps, self.time_s, self.done, self.success, self.stop = 0, 0., False, False, None
        self.position = np.zeros(3)

    def state(self):
        assert threading.get_ident() == self.owner
        return self.position.copy(), np.zeros(3)

    def step(self, action):
        assert threading.get_ident() == self.owner
        if self.task == "fails" and self.steps == 1:
            raise FloatingPointError("the toy's physics failed")  # its second step
        self.position = action.copy()
        self.steps += 1
        self.time_s = self.steps * self.period
        self.success = self.task == "succeeds" and self.steps >= 2  # its goal holds from the second step
        if self.early_stop and self.task == "stumbles" and self.steps == 2:
            self.stop = ("failure", "toy_stumbled")  # a task failure of its own at its second step
        self.done = (self.max_steps is not None and self.steps >= self.max_steps
                     or self.early_stop and self.success or self.stop is not None)

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True


class SlowCameraToy(ToyBackend):
    """A backend whose cameras sample twice per simulated second, as an expensive renderer's would."""
    camera_rate_hz = 2.


class CameraToy(ToyBackend):
    """The toy with an RGB-D camera that renders a constant scene."""
    devices = device_inventory(ToyBackend.layout, {"camera": "top"}, depth=("camera",))

    def intrinsics(self, source):
        return 4, 2, 2., 2., 2., 1.

    def render(self, source):
        assert threading.get_ident() == self.owner
        return np.zeros((2, 4, 3), np.uint8)

    def render_depth(self, source):
        assert threading.get_ident() == self.owner
        return np.ones((2, 4), np.float32)


class DepthToy(CameraToy):
    """The toy whose camera is RGB, or RGB-D in its depth configuration."""

    def __init__(self, max_steps=None, early_stop=False, configuration=None):
        if configuration not in (None, "depth"):
            raise ValueError(f"the toy has no configuration {configuration!r}")
        self.devices = device_inventory(ToyBackend.layout, {"camera": "top"},
                                        depth=("camera",) if configuration else ())
        super().__init__(max_steps, early_stop)


class LensToy(CameraToy):
    """The camera toy rendering through a lens: HandsOn's head camera lens on a 64 x 48 image."""
    distortion = {"top": (0.00646656053141, -0.0479530096054, -0.000352804054273, -0.000104829683551,
                          0.0332237109542)}

    def intrinsics(self, source):
        return 64, 48, 40., 40., 31.5, 23.5

    def render(self, source):
        assert threading.get_ident() == self.owner
        return np.zeros((48, 64, 3), np.uint8)

    def render_depth(self, source):
        assert threading.get_ident() == self.owner
        return np.ones((48, 64), np.float32)


class PoseToy(ToyBackend):
    """A planar arm commanded by its tip pose: joints x and y place the tip at (x, y, 0), unrotated."""
    layout = ActionLayout(("x", "y"), (CommandGroup("reach", (0, 1), (-2., -2.), (2., 2.), tip="tip"),))
    devices = device_inventory(layout, {})

    def reset(self, task, seed, randomize):
        super().reset(task, seed, randomize)
        self.position, self.commanded = np.array([.5, -.5]), []

    def state(self):
        assert threading.get_ident() == self.owner
        return self.position.copy(), np.zeros(2)

    def step(self, action, poses):
        assert threading.get_ident() == self.owner
        pose = poses["reach"]
        self.commanded.append(pose)
        if pose is not None:  # its controller moves the tip to the target, within its reach
            self.position = np.clip(pose[:2], -2., 2.)
        self.steps += 1
        self.time_s = self.steps * self.period

    def tip_pose(self, group):
        assert threading.get_ident() == self.owner and group == "reach"
        return np.r_[self.position, 0., 0., 0., 0., 1.]


class PairPoseToy(PoseToy):
    """Two planar arms, left and right, as joint groups that take joint targets and, by an action of this backend's
    own, goals: move steps an arm's tip 0.1 m a step to the last pose of a goal's multi_dof_trajectory. A step
    leaving an arm's 1.5 m reach ends the goal, and the tips cannot pass a wall at x = 1.2, where a goal stalls."""
    layout = ActionLayout(("x", "y", "x", "y"), tuple(
        CommandGroup(name, indices, (-2., -2.), (2., 2.), tip="tip", joint_targets=True)
        for name, indices in (("left", (0, 1)), ("right", (2, 3)))))
    devices = device_inventory(layout, {})
    actions = tuple(Action(arm, "move", "control_msgs/action/FollowJointTrajectory") for arm in ("left", "right"))
    JOINTS = {"left": slice(0, 2), "right": slice(2, 4)}

    def reset(self, task, seed, randomize):
        super().reset(task, seed, randomize)
        self.position = np.array([.5, -.5, .5, -.5])
        self.goals, self.targets = {}, {}  # each arm's goal as the world reads it; the tip targets of those executing

    def state(self):
        return self.position.copy(), np.zeros(4)

    def tip_pose(self, group):
        return np.r_[self.position[self.JOINTS[group]], 0., 0., 0., 0., 1.]

    def start(self, operation, goal):
        points = goal["multi_dof_trajectory"]["points"]
        if not points:
            self.targets.pop(operation, None)
            return self.finish(operation, "aborted", "a goal needs a tip pose")
        to = points[-1]["transforms"][0]["translation"]
        self.targets[operation] = np.array([to["x"], to["y"]])
        self.follow(operation)

    def end(self, operation, reason):
        if self.targets.pop(operation, None) is not None:
            self.finish(operation, "aborted", reason)

    def finish(self, operation, status, text):
        arm = operation.partition("/")[0]
        self.goals[operation] = {"status": status, "feedback": None, "commands": {arm: self.position[self.JOINTS[arm]]},
                                 "result": {"error_code": 0 if status == "succeeded" else -1, "error_string": text}}

    def follow(self, operation, moved=True):
        """The goal's next step, or its end: there, stalled, or out of reach."""
        arm = operation.partition("/")[0]
        gap = self.targets[operation] - self.position[self.JOINTS[arm]]
        distance = np.linalg.norm(gap)
        command = self.position[self.JOINTS[arm]] + gap * min(1., .1 / max(distance, 1e-9))
        end = (("succeeded", "reached") if distance < 1e-9 else ("aborted", "stalled") if not moved else
               ("aborted", "the tip pose is beyond the 1.5 m reach") if np.linalg.norm(command) > 1.5 else None)
        if end:
            del self.targets[operation]
            return self.finish(operation, *end)
        self.goals[operation] = {"status": None, "result": None, "commands": {arm: command}, "feedback": {
            "multi_dof_joint_names": ["tip"], "multi_dof_desired": {"transforms": [{"translation": {
                "x": float(command[0]), "y": float(command[1])}}]}}}

    def step(self, action):
        assert threading.get_ident() == self.owner
        before, self.position = self.position, np.array(action, dtype=float)
        for operation in self.targets:  # an arm with a goal takes the goal's step
            arm = operation.partition("/")[0]
            self.position[self.JOINTS[arm]] = self.goals[operation]["commands"][arm]
        self.position[::2] = np.minimum(self.position[::2], 1.2)
        self.steps += 1
        self.time_s = self.steps * self.period
        self.done = self.max_steps is not None and self.steps >= self.max_steps
        for operation in list(self.targets):
            arm = self.JOINTS[operation.partition("/")[0]]
            self.follow(operation, moved=not np.allclose(self.position[arm], before[arm]))


class SceneToy(ToyBackend):
    """Selects scenes in its own terms: a layout, which reset takes as a keyword and the record states."""

    def reset(self, task, seed, randomize, *, layout=0):
        super().reset(task, seed, randomize)
        self.scene_layout = layout

    def measures(self):
        return {"layout": self.scene_layout}
