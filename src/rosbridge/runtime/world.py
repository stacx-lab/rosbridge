"""Main-thread episode execution, request admission and observation ordering."""
from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import threading

from rosbridge.core import CommandBuffer
from .pacing import Pacer
from .waits import WaitScheduler


@dataclass(frozen=True)
class ResetResult:
    success: bool
    message: str


class WorldRuntime:
    """Own an already-reset backend; publish(reset) runs on the same owner thread.

    submit() is the only cross-thread entry point: queued functions run on the
    owner thread between steps, so callback threads never touch native state.
    """
    def __init__(self, sim, mode, publish, **pacer_options):
        self.sim, self.publish = sim, publish
        self._owner = threading.get_ident()
        self._requests = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self.episode = -1
        self.waits = WaitScheduler()
        self.pacer = Pacer(mode, sim.period, **pacer_options)
        self._snapshot_reset()

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("native simulator access must stay on its owner thread")

    def _snapshot_reset(self):
        self.episode += 1
        self.time_s, self.steps = self.sim.time_s, self.sim.steps
        self._judged = False
        self.commands = CommandBuffer(self.sim.layout, self.sim.state()[0])
        self.pacer.reset(self.sim.period)

    def submit(self, request):
        """Run request() on the owner thread; the future carries its result."""
        future = Future()
        with self._lock:
            if self._closed:
                future.set_exception(RuntimeError("runtime is closed"))
            else:
                self._requests.put((request, future))
        return future

    def reset(self, task, seed, randomize):
        self._assert_owner()
        self.waits.interrupt(self.time_s, "interrupted by reset")
        previous = self._outcome("reset")
        try:
            self.sim.reset(task, seed, randomize)
        except Exception:
            return ResetResult(False, "reset failed")
        self._record_terminal(result=previous)
        self.commands.close()
        self._snapshot_reset()
        self.publish(True)
        return ResetResult(True, "episode reset")

    def drain(self, timeout=0.):
        self._assert_owner()
        try:
            item = self._requests.get(timeout=timeout) if timeout else self._requests.get_nowait()
        except queue.Empty:
            return
        # Bound admission work so a request flood cannot starve the physics loop.
        for index in range(64):
            request, future = item
            try:
                future.set_result(request())
            except Exception as error:
                future.set_exception(error)
            if index == 63:
                return
            try:
                item = self._requests.get_nowait()
            except queue.Empty:
                return

    def tick(self):
        self._assert_owner()
        if self.sim.done or not self.pacer.ready(self.waits.pending):
            return False
        self.sim.step(self.commands.action(self.time_s + self.sim.period))
        self.pacer.advanced()
        self.time_s, self.steps = self.sim.time_s, self.sim.steps
        self.publish(False)
        if self.sim.done:
            self.commands.close()
            self._record_terminal()
        # Publications precede completion, including a successful terminal tick.
        self.waits.settle(self.time_s, self.steps, self.sim.done)
        return True

    def _outcome(self, reason=None):
        limit = self.sim.max_steps
        # A backend may add its own measures of the episode (BEHAVIOR: the goal predicates that hold).
        extra = self.sim.outcome_extra() if hasattr(type(self.sim), "outcome_extra") else {}
        return {"episode": self.episode, "task": self.sim.task, "seed": self.sim.seed,
                "step": self.steps, "time_s": self.time_s, "reward": self.sim.reward,
                "success": bool(self.sim.success), "done": bool(self.sim.done),
                "reason": reason or ("success" if self.sim.success else "max_steps"
                           if limit and self.steps >= limit else "episode_done"), **extra}

    def _record_terminal(self, reason=None, *, result=None, error=None):
        if self._judged:
            return
        self._judged = True
        result = self._outcome(reason) if result is None else result
        if error:
            result = dict(result, error=error)
        folder = os.environ.get("ROSBRIDGE_RUN_DIR")
        if folder:
            with (Path(folder) / "episode-results.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        print("episode terminal: " + json.dumps(result), flush=True)

    def idle_s(self):
        return 0.05 if self.sim.done else self.pacer.wait_s(self.waits.pending)

    def close(self, error=None):
        """Stop: interrupt the pending wait, fail queued requests and record the open episode as stopped (with the
        error that stopped the world, if one did)."""
        self._assert_owner()
        with self._lock:
            self._closed = True
        self.waits.interrupt(self.time_s, "runtime stopped")
        while True:
            try:
                _, future = self._requests.get_nowait()
            except queue.Empty:
                break
            future.set_exception(RuntimeError("runtime stopped"))
        try:
            self._record_terminal("stopped", error=error)
        finally:
            self.commands.close()
            self.sim.close()
