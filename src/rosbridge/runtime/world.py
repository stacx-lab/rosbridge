"""Main-thread episode execution, request admission and observation ordering."""
from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import queue
import threading

from rosbridge.core import CommandBuffer
from .pacing import Pacer
from .waits import WaitScheduler


# Per-episode counts of device commands the world clipped to a limit or refused (malformed, or the episode is done);
# the episode record carries them. A topic publisher gets no reply, so a refusal (device, reason) is also stated in the
# reply of the next world wait (WorldRuntime.reporting).
COUNTERS = ("commands_clipped", "commands_refused")
# How a wait states why the episode ended, per the reason its record gives (WorldRuntime.end_cause).
END_CAUSES = {"success": "the goal was reached", "max_steps": "the time limit was reached",
              "episode_done": "the simulator ended it"}
# A wait reply names at most this many kinds of refusal (device and reason), then counts the rest.
REFUSALS_LISTED = 5
REFUSED = "refused, so not executed"  # how a wait reply introduces them


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
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.refusals = {}  # (device, reason) -> [count, world time of the first]; until a wait reply states them
        self.commands = CommandBuffer(self.sim.layout, self.sim.state()[0][:len(self.sim.layout.joint_names)])
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
            if self.sim.done:
                # The backend lost the episode in the attempt (BEHAVIOR reloads its one scene in place). Its outcome was
                # measured before; record it, or the episode would go unjudged although it ended normally.
                self.commands.close()
                self._record_terminal(result=dict(previous, reason="reset_failed"))
            return ResetResult(False, "reset failed")
        self._record_terminal(result=previous)
        self.commands.close()
        self._snapshot_reset()
        self.publish(True)
        return ResetResult(True, "episode reset")

    def count(self, name):
        """Count an event of this episode (COUNTERS) for its record; owner thread."""
        self._assert_owner()
        self.counters[name] += 1

    def refuse(self, device, reason):
        """A device refused a command (device: its name and operation, e.g. right_arm.joint_target): count it for the
        record and keep it for the next wait's reply; owner thread."""
        self.count("commands_refused")
        entry = self.refusals.setdefault((device, str(reason)), [0, self.time_s])
        entry[0] += 1

    def refusal_note(self):
        """The refusals not yet stated, as a wait reply states them (empty if none); clears them."""
        if not self.refusals:
            return ""
        items, self.refusals = list(self.refusals.items()), {}
        parts = [f"{device}: {reason} (" + (f"at world time {first:.2f} s" if count == 1 else
                                            f"{count} commands, the first at world time {first:.2f} s") + ")"
                 for (device, reason), (count, first) in items[:REFUSALS_LISTED]]
        rest = sum(count for _, (count, _) in items[REFUSALS_LISTED:])
        if rest:
            parts.append(f"and {rest} more")
        return f"{REFUSED}: " + "; ".join(parts)

    def reporting(self, complete):
        """complete() for a wait whose reply also states the commands refused since the previous reply; owner thread."""
        def report(result):
            note = self.refusal_note()
            complete(replace(result, message=f"{result.message}; {note}") if note else result)
        return report

    def _end_reason(self):
        limit = self.sim.max_steps
        return "success" if self.sim.success else "max_steps" if limit and self.steps >= limit else "episode_done"

    def end_cause(self):
        """Why the episode ended, as a wait states it (None while it runs); stated only once it is over."""
        return END_CAUSES[self._end_reason()] if self.sim.done else None

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
        self.waits.settle(self.time_s, self.steps, self.end_cause())
        return True

    def _outcome(self, reason=None):
        # A backend may add its own measures of the episode (BEHAVIOR: the goal predicates that hold).
        extra = self.sim.outcome_extra() if hasattr(type(self.sim), "outcome_extra") else {}
        return {"episode": self.episode, "task": self.sim.task, "seed": self.sim.seed,
                "step": self.steps, "time_s": self.time_s, "reward": self.sim.reward,
                "success": bool(self.sim.success), "done": bool(self.sim.done),
                "reason": reason or self._end_reason(), **self.counters, **extra}

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
