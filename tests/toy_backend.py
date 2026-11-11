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

    def __init__(self, max_steps=None, terminate_on_success=False):
        self.owner = threading.get_ident()
        self.max_steps, self.terminate_on_success = max_steps, terminate_on_success
        self.closed = False
        self.reset("toy", 0, False)

    def configure_spectators(self, width, height, views=None):
        pass

    def reset(self, task, seed, randomize):
        assert threading.get_ident() == self.owner
        if task not in ("toy", "succeeds"):
            raise ValueError("unknown task")
        self.task, self.seed = task, seed
        self.steps, self.time_s, self.done, self.success = 0, 0., False, False
        self.position = np.zeros(3)

    def state(self):
        assert threading.get_ident() == self.owner
        return self.position.copy(), np.zeros(3)

    def step(self, action):
        assert threading.get_ident() == self.owner
        self.position = action.copy()
        self.steps += 1
        self.time_s = self.steps * self.period
        self.success = self.task == "succeeds" and self.steps >= 2  # its goal holds from the second step
        self.done = (self.max_steps is not None and self.steps >= self.max_steps
                     or self.terminate_on_success and self.success)

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True
