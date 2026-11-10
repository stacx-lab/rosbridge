"""Main-thread episode execution, request admission and observation ordering."""
from concurrent.futures import Future
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import threading
import traceback

from rosbridge.core import CommandBuffer
from .pacing import Pacer
from .waits import WaitScheduler

# A wait's reply names at most this many kinds of refused command (endpoint and reason), then counts the rest.
REFUSALS_LISTED = 5
# What a wait reply at or after the end says ended the episode, by the termination the world applied
# (WorldRuntime._termination). Only "success" is an evaluator result: the first success of a run launched to stop on
# success, which discloses it by ending before its time limit, as its operator chose. The others are the world's own
# conditions: at the step limit the cause is the limit whether or not the goal holds. "invalid_state" is a backend's
# fault (its robot's state became invalid), stated with the backend's own words.
END_CAUSES = {"success": "the goal was reached", "max_steps": "the time limit was reached",
              "episode_done": "the simulator ended it", "invalid_state": "the robot's state became invalid"}
# Every wait reply states a tilting base's tilt beyond this many degrees from upright (the backend's base_attitude),
# after the end notice. Measured natively on the R1Pro (putting_dishes_away_after_cleaning, seed 0): on open
# floor its base stays within 4.1 degrees in full-speed drives; drives square into furniture tilt it 0.6 to 9.6, the
# same drives 5.6 degrees off square prop it at 30 to 50 or tip it over, to rest at 82 to 90.
TILTED_DEG = 15.


class WorldRuntime:
    """Run the one episode a backend has loaded; publish() runs on the same owner thread.

    submit() is the only cross-thread entry point: queued functions run on the
    owner thread between steps, so callback threads never touch native state.
    stop_on_success: the run was launched to end at the first success (the backend's
    terminate_on_success), which the world needs to tell that end from the backend's own.
    """
    def __init__(self, sim, mode, publish, stop_on_success=False, **pacer_options):
        self.sim, self.publish = sim, publish
        self.stop_on_success, self._ended = stop_on_success, None
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
        """A wait's result with the base's tilt, when it is large (tilt_notice), a backend's notice of its robot's
        state (state_notice: BEHAVIOR's joints far beyond their ranges) and the commands refused since the previous
        reply added to its message, after its first words and the end notice: a client publishing commands on topics
        learns of a refusal no other way, and a planar view of the base never showed it tipped. Owner thread."""
        self._assert_owner()
        notices = [notice for notice in (self.tilt_notice(), getattr(self.sim, "state_notice", None)) if notice]
        if self.refused:
            refused, self.refused = list(self.refused.items()), {}
            named = [f"{endpoint}: {reason}" + (f" ({count} commands)" if count > 1 else "")
                     for (endpoint, reason), count in refused[:REFUSALS_LISTED]]
            rest = sum(count for _, count in refused[REFUSALS_LISTED:])
            if rest:
                named.append(f"and {rest} more")
            notices.append("refused since the previous wait reply: " + "; ".join(named))
        return replace(result, message="; ".join([result.message, *notices])) if notices else result

    def tilt_deg(self):
        """The base's tilt from upright in degrees, for a backend whose base tilts (base_attitude); else None."""
        attitude = getattr(self.sim, "base_attitude", None)
        tilt = math.degrees(float(attitude["tilt"])) if attitude else math.nan
        return tilt if math.isfinite(tilt) else None

    def tilt_notice(self):
        """How a wait reply states a base tilted beyond TILTED_DEG from upright, with its roll and pitch; else None."""
        tilt = self.tilt_deg()
        if tilt is None or tilt <= TILTED_DEG:
            return None
        attitude = self.sim.base_attitude
        return (f"the base is tilted {tilt:.1f} degrees from upright (roll {math.degrees(attitude['roll']):.1f}, "
                f"pitch {math.degrees(attitude['pitch']):.1f})")

    def admit_wait(self, duration_s, complete):
        """Admit a world wait of duration_s; complete receives its result, which at or after the end of the episode
        says so (end_cause). Owner thread."""
        self._assert_owner()
        self.waits.admit(duration_s, self.time_s, self.steps, self.sim.period, self.end_cause(), complete)

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
        if self.sim.done:
            # Recorded before the last observation is published, so that a failure to publish it cannot lose the
            # record (an invalid robot state ends an episode with a robot that may not render).
            self.commands.close()
            self._record_terminal()
        self.publish()
        # Publications precede completion, including a successful terminal tick.
        self.waits.settle(self.time_s, self.steps, self.end_cause())
        return True

    def _fault(self):
        """Why the backend found its robot's state invalid, which ended the episode (a backend's fault); else None."""
        return getattr(self.sim, "fault", None) or None

    def _termination(self):
        """What ended the episode: the first success of a run launched to stop on success, else the backend's fault,
        else the step limit, else the backend itself. Not the record's reason, which says "success" whenever the goal
        holds at the end, also at the step limit of a run that goes on after success: that is the evaluator's
        result."""
        limit = self.sim.max_steps
        if self.stop_on_success and self.sim.success:
            return "success"
        if self._fault():
            return "invalid_state"
        return "max_steps" if limit and self.steps >= limit else "episode_done"

    def end_cause(self):
        """None while the episode runs. Once it has ended, what ended it as a wait reply states it (END_CAUSES, with
        the backend's fault), or True for an end stated without a cause. Fixed at the end, as the record is. Owner
        thread."""
        self._assert_owner()
        if self._ended is None and self.sim.done:
            termination = self._termination()
            cause = END_CAUSES.get(termination, True)
            self._ended = f"{cause} ({self._fault()})" if termination == "invalid_state" else cause
        return self._ended

    def _outcome(self, reason=None):
        limit = self.sim.max_steps
        result = {"task": self.sim.task, "seed": self.sim.seed,
                  "step": self.steps, "time_s": self.time_s, "reward": self.sim.reward,
                  "success": bool(self.sim.success), "done": bool(self.sim.done),
                  "reason": reason or ("success" if self.sim.success else "invalid_state" if self._fault()
                             else "max_steps" if limit and self.steps >= limit else "episode_done")}
        if self._fault():
            # A valid end of the episode (its record keeps the backend's measures), not an error that stopped the world.
            result["fault"] = str(self._fault())
        return result

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
