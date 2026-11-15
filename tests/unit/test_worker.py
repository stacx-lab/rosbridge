"""A native backend served from another interpreter behaves like an in-process one."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.worker import WorkerBackend, WorkerError, decode, encode

ROOT = Path(__file__).resolve().parents[2]


class ToyWorker(WorkerBackend):
    target = "tests.toy_backend:ToyBackend"


def worker(**kwargs):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return ToyWorker(python=sys.executable, env=env, **kwargs)


def test_round_trip_keeps_layout_state_and_errors():
    sim = worker(max_steps=2)
    try:
        assert sim._process.pid != os.getpid()
        assert sim.layout.joint_names == ("slide", "finger", "rotate")
        assert [d.name for d in sim.devices] == ["axes", "finger"]
        assert sim.task == "toy" and sim.period == pytest.approx(.1)
        sim.step(np.array([.5, -.25, 1.]))
        positions, velocities = sim.state()
        np.testing.assert_array_equal(positions, [.5, -.25, 1.])
        assert positions.flags.writeable and velocities.shape == (3,)
        assert sim.steps == 1 and sim.time_s == pytest.approx(.1) and not sim.done
        with pytest.raises(ValueError, match="unknown task"):
            sim.reset("missing", 1, False)
        assert sim.task == "toy" and sim.steps == 1
        sim.step(np.zeros(3))
        assert sim.done
        sim.max_steps = None
        sim.reset("toy", 3, False)
        assert sim.max_steps is None and sim.seed == 3 and not sim.done
        with pytest.raises(WorkerError, match="AttributeError") as error:
            sim.native("missing_method")
        assert error.value.kind == "AttributeError"
    finally:
        sim.close()
    assert sim._process.returncode == 0
    with pytest.raises(WorkerError, match="closed"):
        sim.state()


def test_a_call_interrupted_awaiting_its_reply_keeps_calls_and_replies_paired():
    """A stop's KeyboardInterrupt lands while the world awaits a step, and the worker completes the step anyway."""
    sim = worker()
    try:
        connection = sim._connection

        class Interrupted:
            def send(self, data):
                connection.send(data)

            def recv(self):
                sim._connection = connection
                raise KeyboardInterrupt
        sim._connection = Interrupted()
        with pytest.raises(KeyboardInterrupt):
            sim.step(np.array([.5, -.25, 1.]))
        positions, _ = sim.state()  # its own reply, not the step's
        np.testing.assert_array_equal(positions, [.5, -.25, 1.])
        assert sim.steps == 1
    finally:
        sim.close()
    assert sim._process.returncode == 0


def test_worker_death_is_reported():
    sim = worker()
    try:
        sim._process.kill()
        sim._process.wait()
        with pytest.raises(WorkerError, match="exited"):
            sim.state()
    finally:
        sim.close()


def test_encoding_is_numpy_version_neutral():
    layout = ActionLayout(("a", "b"), (CommandGroup("arm", (0, 1), (np.float64(-1), -2.), (1., 2.)),))
    value = encode({"layout": layout, "image": np.ones((2, 3, 3), np.uint8), "focal": np.float32(1.5)})
    assert type(value["focal"]) is float and type(value["layout"].groups[0].lower[0]) is float
    decoded = decode(value)
    assert decoded["layout"] == layout
    assert decoded["image"].dtype == np.uint8 and decoded["image"].shape == (2, 3, 3)


def test_worker_dies_with_its_proxy_process():
    import subprocess
    import time
    script = ("import sys, time; sys.path.insert(0, %r)\n"
              "from tests.unit.test_worker import worker\n"
              "print(worker()._process.pid, flush=True); time.sleep(60)") % str(ROOT)
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                              env=dict(os.environ, PYTHONPATH=str(ROOT)))
    try:
        child = int(parent.stdout.readline())
    finally:
        parent.kill()
        parent.wait()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        # A killed worker stays a zombie of init only briefly; check its state.
        try:
            if open(f"/proc/{child}/stat").read().split()[2] == "Z":
                return
        except FileNotFoundError:
            return
        time.sleep(.05)
    raise AssertionError("worker outlived its proxy process")
