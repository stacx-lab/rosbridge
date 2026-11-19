"""Synthetic position/scalar backend for shared conformance tests."""
import threading
import numpy as np
from rosbridge.core import ActionLayout, CommandGroup
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


class SceneToy(ToyBackend):
    """Selects scenes in its own terms: a layout, which reset takes as a keyword and the record states."""

    def reset(self, task, seed, randomize, *, layout=0):
        super().reset(task, seed, randomize)
        self.scene_layout = layout

    def measures(self):
        return {"layout": self.scene_layout}
