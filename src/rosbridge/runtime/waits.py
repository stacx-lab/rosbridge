"""Shared simulation-time waits. Called only by the physics owner, with no ROS imports."""
from dataclasses import dataclass
import math


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
            complete(WaitResult(True, "ok", 0., time_s))
        elif done:
            complete(WaitResult(False, "episode is done; call reset", 0., time_s))
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
                wait.complete(WaitResult(reached, "ok" if reached else "episode ended during wait",
                                         max(0., time_s - wait.start_s), time_s))
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
