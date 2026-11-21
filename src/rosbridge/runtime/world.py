"""Main-thread episode execution, request admission and observation ordering."""
from concurrent.futures import Future
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import traceback

from rosbridge.core import CommandBuffer
from .pacing import Pacer
from .waits import WaitScheduler

# Why a wait with time left ends at once: a refusal its reply must name. The rest of the wait is never stepped, so a
# stepped run never charges it.
REFUSED = "ended early by a refused command"

# A wait's reply names at most this many kinds of refused command (endpoint and reason), then counts the rest.
REFUSALS_LISTED = 5
# What wait replies say of an episode's end, by the record's reason; a cause the backend names follows.
ENDS = {"success": "the task succeeded", "failure": "the task failed", "timeout": "the time limit was reached"}


class WorldRuntime:
    """Run the one episode a backend has loaded; publish() runs on the same owner thread.

    submit() is the only cross-thread entry point: queued functions run on the
    owner thread between steps, so callback threads never touch native state.
    early_stop: the backend was created to end the episode where its simulator's evaluator does, such as at a
    success or a task failure; otherwise only its step limit ends it.
    """
    def __init__(self, sim, mode, publish, early_stop=False, **pacer_options):
        self.sim, self.publish, self.early_stop = sim, publish, early_stop
        self.ended = None  # once the episode has ended, why, as wait replies state it
        self._owner = threading.get_ident()
        self._requests = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self._judged = False
        self.time_s, self.steps = sim.time_s, sim.steps
        # The step at which the backend first reported success: where an early-stop run ends at a success, so a
        # run without early stop also records it.
        self.first_success_step = None
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
        """A device refused a command published on endpoint: the pending wait ends now, and its reply, or the next
        wait's, names the refusal. Owner thread."""
        self._assert_owner()
        key = (endpoint, str(reason))
        self.refused[key] = self.refused.get(key, 0) + 1
        self.waits.interrupt(self.time_s, REFUSED)

    def wait(self, duration_s, complete):
        """Admit a world wait; complete receives its WaitResult. One admitted while a refusal awaits its reply ends at
        once, as refuse ends a pending one. Owner thread."""
        self._assert_owner()
        self.waits.admit(duration_s, self.time_s, self.steps, self.sim.period, self.ended, complete,
                         interrupted=REFUSED if self.refused else None)

    def hold(self, group):
        """Stop a command group where it is, as its device's stop does: a base at zero velocity, a pose group that
        takes only pose targets at its measured tip pose, any other group at its measured positions (a gripper
        keeps its applied target instead). Owner thread."""
        self._assert_owner()
        spec = next(g for g in self.sim.layout.groups if g.name == group)
        if spec.velocity:
            values = [0.] * len(spec.indices)
        elif spec.tip and not spec.joint_targets:
            values = self.sim.tip_pose(group)
        else:
            values = self.sim.state()[0][list(spec.indices)]
        self.commands.stop(group, values)

    def budget(self):
        """The episode budget in seconds of world time, Kinex's [used_s, max_s]: in stepped mode the control steps
        taken and the step limit that ends the episode (inf without one). Realtime has no budget, since world time
        passes whether or not the agent spends it: NaN for both. Owner thread."""
        self._assert_owner()
        if self.pacer.mode != "stepped":
            return math.nan, math.nan
        limit = getattr(self.sim, "max_steps", None)
        return self.steps * self.sim.period, limit * self.sim.period if limit else math.inf

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
        poses = self.commands.poses()  # pose groups' tip targets for the same step
        try:
            if poses:
                self.sim.step(action, poses)
            else:
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
            if self.sim.success and self.first_success_step is None:
                self.first_success_step = self.steps
            self.publish()
            if self.sim.done:
                reason, cause = self._stop()
                self._end(ENDS.get(reason, f"the simulator ended it ({reason})") + (f" ({cause})" if cause else ""),
                          reason, cause=cause)
        # Publications precede completion, including a successful terminal tick.
        self.waits.settle(self.time_s, self.steps, self.ended)
        return True

    def _stop(self):
        """Why a backend that reports done ended the episode: the stop it names, a success in an early-stop run,
        or its step limit."""
        stop = getattr(self.sim, "stop", None)
        if stop:
            reason, cause = stop
            return reason, cause
        return ("success" if self.early_stop and self.sim.success else "timeout"), None

    def _end(self, why, reason, error=None, cause=None):
        self.commands.close()
        self.ended = f"the episode ended because {why}"
        self._record_terminal(reason, error=error, cause=cause)

    def _outcome(self, reason):
        return {"task": self.sim.task, "seed": self.sim.seed,
                "step": self.steps, "time_s": self.time_s, "reward": self.sim.reward,
                "success": bool(self.sim.success), "first_success_step": self.first_success_step,
                "done": self.ended is not None, "early_stop": self.early_stop, "reason": reason}

    def _record_terminal(self, reason, *, error=None, cause=None):
        if self._judged:
            return
        self._judged = True
        result = self._outcome(reason)
        if cause is not None:
            result["cause"] = cause
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
