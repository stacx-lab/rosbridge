"""Main-thread episode execution, request admission and observation ordering."""
from concurrent.futures import Future
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import sys
import threading
import traceback

from rosbridge.core import CommandBuffer
from .pacing import Pacer
from .waits import WaitScheduler

# A wait's reply names at most this many kinds of refused command (endpoint and reason), then counts the rest.
REFUSALS_LISTED = 5


class WorldRuntime:
    """Run the one episode a backend has loaded; publish() runs on the same owner thread.

    submit() is the only cross-thread entry point: queued functions run on the
    owner thread between steps, so callback threads never touch native state.
    stop_on_success: the backend was created to end the episode at its first success.
    """
    def __init__(self, sim, mode, publish, stop_on_success=False, **pacer_options):
        self.sim, self.publish, self.stop_on_success = sim, publish, stop_on_success
        self.ended = None  # once the episode has ended, why, as wait replies state it
        self._owner = threading.get_ident()
        self._requests = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self._judged = False
        self.time_s, self.steps = sim.time_s, sim.steps
        self.commands = CommandBuffer(sim.layout, sim.state()[0])
        self.waits = WaitScheduler()
        self.pacer = Pacer(mode, sim.period, **pacer_options)
        self.refused = {}  # (endpoint, reason) -> commands refused since the last wait reply

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("native simulator access must stay on its owner thread")

    def submit(self, request):
        """Run request() on the owner thread; the future carries its result."""
        future = Future()
        with self._lock:
            if self._closed:
                future.set_exception(RuntimeError("runtime is closed"))
            else:
                self._requests.put((request, future))
        return future

    def refuse(self, endpoint, reason):
        """A device refused a command published on endpoint; the next wait's reply names it. Owner thread."""
        self._assert_owner()
        key = (endpoint, str(reason))
        self.refused[key] = self.refused.get(key, 0) + 1

    def reply(self, result):
        """A wait's result with the commands refused since the previous reply added to its message: a client
        publishing commands on topics learns of a refusal no other way. Owner thread."""
        self._assert_owner()
        if not self.refused:
            return result
        refused, self.refused = list(self.refused.items()), {}
        named = [f"{endpoint}: {reason}" + (f" ({count} commands)" if count > 1 else "")
                 for (endpoint, reason), count in refused[:REFUSALS_LISTED]]
        rest = sum(count for _, count in refused[REFUSALS_LISTED:])
        if rest:
            named.append(f"and {rest} more")
        return replace(result, message=f"{result.message}; refused since the previous wait reply: " + "; ".join(named))

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
        if self.ended or not self.pacer.ready(self.waits.pending):
            return False
        action = self.commands.action(self.time_s + self.sim.period)
        try:
            self.sim.step(action)
        except Exception as failure:
            # A step that fails, as BEHAVIOR's physics can when the robot is driven into furniture, ends the episode;
            # the world goes on answering.
            kind = getattr(failure, "kind", None) or type(failure).__name__  # a worker's: the native exception's
            print("episode ended by an error:", file=sys.stderr, flush=True)
            traceback.print_exc()
            self._end(f"the simulation failed ({kind})", "error", f"{type(failure).__name__}: {failure}"[:500])
        else:
            self.pacer.advanced()
            self.time_s, self.steps = self.sim.time_s, self.sim.steps
            self.publish()
            if self.sim.done:
                # A backend ends an episode only at its first success, if created to stop there, or at its step limit.
                self._end("the task succeeded" if self.stop_on_success and self.sim.success
                          else "the time limit was reached")
        # Publications precede completion, including a successful terminal tick.
        self.waits.settle(self.time_s, self.steps, self.ended)
        return True

    def _end(self, cause, reason=None, error=None):
        self.commands.close()
        self.ended = f"the episode ended because {cause}"
        self._record_terminal(reason, error=error)

    def _outcome(self, reason=None):
        return {"task": self.sim.task, "seed": self.sim.seed,
                "step": self.steps, "time_s": self.time_s, "reward": self.sim.reward,
                "success": bool(self.sim.success), "done": self.ended is not None,
                "reason": reason or ("success" if self.sim.success else "max_steps")}

    def _record_terminal(self, reason=None, *, error=None):
        if self._judged:
            return
        self._judged = True
        result = self._outcome(reason)
        if error is None and hasattr(type(self.sim), "measures"):
            # The backend's own measures (BEHAVIOR: q_score). A world stopped by an error records none, since the
            # error may have broken the backend; a measure that fails is the record's error.
            try:
                result.update(self.sim.measures())
            except Exception as failure:
                traceback.print_exc()
                error = f"measures: {type(failure).__name__}: {failure}"[:500]
        if error:
            result["error"] = error
        folder = os.environ.get("ROSBRIDGE_RUN_DIR")
        if folder:
            with (Path(folder) / "episode-results.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        print("episode terminal: " + json.dumps(result), flush=True)

    def idle_s(self):
        return 0.05 if self.ended else self.pacer.wait_s(self.waits.pending)

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
