"""Shared simulation-time waits. Called only by the physics owner, with no ROS imports."""
from dataclasses import dataclass
import math


def ended(time_s):
    """How a wait states that the episode has ended: the world time at the end, which a worker compares with the
    episode's time limit to tell an early end (a stop on success) from running out of time; never the outcome."""
    return f"the episode ended at world time {time_s:.2f} s"


@dataclass(frozen=True)
class WaitResult:
    success: bool
    message: str
    waited_s: float
    time_s: float


@dataclass
class PendingWait:
    start_s: float
    end_step: int
    complete: object


class WaitScheduler:
    def __init__(self):
        self._pending = []

    @property
    def pending(self):
        return bool(self._pending)

    def admit(self, duration_s, time_s, step, period, done, complete):
        if not math.isfinite(duration_s) or duration_s < 0:
            complete(WaitResult(False, "duration_s must be finite and nonnegative", 0., time_s))
        elif duration_s == 0:
            complete(WaitResult(True, f"ok; {ended(time_s)}" if done else "ok", 0., time_s))
        elif done:
            complete(WaitResult(False, ended(time_s), 0., time_s))
        else:
            # Integer deadlines avoid accumulated floating-point drift. Move one ULP
            # toward zero so e.g. 0.3 / 0.1 does not accidentally request an extra tick.
            ticks = duration_s / period
            if not math.isfinite(ticks):
                complete(WaitResult(False, "duration_s is too large", 0., time_s))
                return
            count = max(1, math.ceil(math.nextafter(ticks, 0.)))
            self._pending.append(PendingWait(time_s, step + count, complete))

    def settle(self, time_s, step, done=False):
        remaining = []
        for wait in self._pending:
            reached = step >= wait.end_step
            if reached or done:
                message = ("ok" if reached else "episode ended during wait") + (f"; {ended(time_s)}" if done else "")
                wait.complete(WaitResult(reached, message, max(0., time_s - wait.start_s), time_s))
            else:
                remaining.append(wait)
        self._pending = remaining

    def cancel(self, complete, time_s):
        for wait in list(self._pending):
            if wait.complete is complete:
                self._pending.remove(wait)
                wait.complete(WaitResult(False, "cancelled", max(0., time_s - wait.start_s), time_s))
                return True
        return False

    def interrupt(self, time_s, message):
        pending, self._pending = self._pending, []
        for wait in pending:
            wait.complete(WaitResult(False, message, max(0., time_s - wait.start_s), time_s))
