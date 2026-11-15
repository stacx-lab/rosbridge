"""A backend proxy whose native simulator runs in another interpreter. No ROS.

The world node keeps the proxy on its owner thread; the worker's main thread
owns every native call. Calls are sequential request/response pairs over a
socket, so worker stdout and stderr stay free for native logging.
"""
import dataclasses
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
from multiprocessing.connection import Connection

import numpy as np

# Attributes the world reads from a backend. Static ones change only on reset.
STATIC = ("default_task", "layout", "devices", "cameras", "period", "spectators")
DYNAMIC = ("time_s", "task", "seed", "steps", "max_steps", "done", "reward", "success")
# Exceptions whose type the world interprets; others become RuntimeError.
ERRORS = {error.__name__: error for error in (ValueError, KeyError, RuntimeError, NotImplementedError)}


class WorkerError(RuntimeError):
    kind = None  # the type name of the native exception it carries, if one


class Array:
    """An ndarray independent of the NumPy version on either side of the socket."""
    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype, shape, data):
        self.dtype, self.shape, self.data = dtype, shape, data

    def __reduce__(self):
        return Array, (self.dtype, self.shape, self.data)


def encode(value):
    """Replace NumPy arrays and scalars, whose pickles name version-specific modules."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return Array(array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return type(value)(encode(item) for item in value)
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{field.name: encode(getattr(value, field.name))
                                             for field in dataclasses.fields(value) if field.init})
    return value


def decode(value):
    if isinstance(value, Array):
        return np.frombuffer(value.data, dtype=np.dtype(value.dtype)).reshape(value.shape).copy()
    if isinstance(value, (list, tuple)):
        return type(value)(decode(item) for item in value)
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


def native_environment(simulator):
    """The native interpreter and environment of a simulator's manifest."""
    file = os.environ.get("ROSBRIDGE_NATIVE_ENVIRONMENT")
    if file:
        value = json.loads(Path(file).read_text())
        return value["python"], value["env"]
    from rosbridge.catalog import simulator as manifest
    from rosbridge.provisioning.environment import native_activated
    root, _, config = manifest(simulator)
    python, env = native_activated(root, config)
    for key in ("ROS_HOME", "ROS_LOG_DIR", "TMPDIR", "ROSBRIDGE_RUN_DIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    return str(python), env


def python_path(simulator):
    """rosbridge.core plus, for a cataloged simulator, its adapter and prepared upstream."""
    if simulator is None:
        return [str(Path(__file__).resolve().parents[2])]
    from rosbridge.catalog import simulator as manifest
    from rosbridge.provisioning.environment import native_path
    root, _, config = manifest(simulator)
    return native_path(root, config)


class WorkerBackend:
    """Subclasses name their `simulator`; the native class defaults to its manifest's
    `native_backend` ("module:Class") and the interpreter to its `native_environment`.
    """

    simulator = None
    target = None
    close_timeout_s = 60.

    def __init__(self, *, max_steps=None, terminate_on_success=False, python=None, env=None):
        self._cache, self._process, self._connection, self._unread = {}, None, None, False
        if python is None:
            python, env = native_environment(self.simulator)
        env = dict(os.environ if env is None else env)
        env["PYTHONPATH"] = os.pathsep.join(python_path(self.simulator) + (
            [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        target = self.target
        if target is None:
            from rosbridge.catalog import simulator as manifest
            target = manifest(self.simulator)[2]["native_backend"]
        parent, child = socket.socketpair()
        try:
            self._process = subprocess.Popen(
                [str(python), "-m", "rosbridge.core.worker_main", str(child.fileno()), target],
                env=env, stdin=subprocess.DEVNULL, pass_fds=(child.fileno(),))
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._connection = Connection(parent.detach())
        try:
            self._call("__init__", max_steps=max_steps, terminate_on_success=terminate_on_success)
        except BaseException:
            self._kill()
            raise

    def __getattr__(self, name):
        cache = self.__dict__.get("_cache", {})
        if name in cache:
            return cache[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._call("__setattr__", name, value)

    def _receive(self):
        status, payload, snapshot = decode(self._connection.recv())
        self._unread = False
        if snapshot:
            self._cache.update(snapshot)
        return status, payload

    def _call(self, method, *args, **kwargs):
        if self._connection is None:
            raise WorkerError("native worker is closed")
        try:
            if self._unread:
                # The previous call was interrupted awaiting its reply (a stop's KeyboardInterrupt during a step).
                # The worker completed it regardless, so its reply comes before this call's.
                self._receive()
            self._connection.send(encode((method, args, kwargs)))
            self._unread = True
            status, payload = self._receive()
        except (EOFError, OSError) as error:
            self._kill(5.)
            code = self._process.returncode if self._process else None
            raise WorkerError(f"native worker exited (exit {code})") from error
        if status == "ok":
            return payload
        kind, message, trace = payload
        if kind in ERRORS:
            raise ERRORS[kind](message)
        error = WorkerError(f"native {kind}: {message}\n{trace}")
        error.kind = kind
        raise error

    def configure_spectators(self, width=960, height=720, views=None):
        return self._call("configure_spectators", width, height, views)

    def reset(self, task, seed, randomize=True):
        return self._call("reset", task, seed, randomize)

    def state(self):
        return self._call("state")

    def step(self, action):
        return self._call("step", action)

    def intrinsics(self, camera):
        return self._call("intrinsics", camera)

    def render(self, camera):
        return self._call("render", camera)

    def render_depth(self, camera):
        return self._call("render_depth", camera)

    def render_spectator(self, name):
        return self._call("render_spectator", name)

    def spectator_intrinsics(self, name):
        return self._call("spectator_intrinsics", name)

    def camera_pose(self, camera):
        return self._call("camera_pose", camera)

    def native(self, method, *args, **kwargs):
        """Invoke another backend method in the worker, for tests and diagnostics."""
        return self._call(method, *args, **kwargs)

    def close(self):
        if self._connection is None:
            return
        try:
            self._call("close")
        except WorkerError:
            pass
        finally:
            self._kill(self.close_timeout_s)

    def _kill(self, timeout=0.):
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
