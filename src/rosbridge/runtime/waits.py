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
    until: tuple = ()
    expire_step: float = math.inf
    expire: object = None


def control_steps(seconds, period):
    """The whole control steps that cover seconds; inf beyond counting. Integer deadlines avoid accumulated
    floating-point drift, and one ULP toward zero keeps 0.3 / 0.1 from asking an extra step."""
    ticks = seconds / period
    return max(1, math.ceil(math.nextafter(ticks, 0.))) if 0 < ticks < math.inf else ticks


class WaitScheduler:
    def __init__(self):
        self._pending = []

    @property
    def pending(self):
        return bool(self._pending)

    def admit(self, duration_s, time_s, step, period, ended, complete, interrupted=None, until=(), timeout_s=0.,
              expire=None):
        """ended: None while the episode runs, then why it ended, which every valid reply adds. interrupted: why a
        wait with anything left to wait for ends at once, as interrupt ends pending ones; a zero wait, an
        observation barrier, still succeeds. until: predicates, such as goals having finished; the wait ends once
        duration_s has passed and all hold. With a positive timeout_s, expire() is called that long into the wait
        if some do not hold yet, and makes them."""
        if not math.isfinite(duration_s) or duration_s < 0:
            complete(WaitResult(False, "duration_s must be finite and nonnegative", 0., time_s))
        elif duration_s == 0 and all(done() for done in until):
            complete(WaitResult(True, f"ok; {ended}" if ended else "ok", 0., time_s))
        elif ended:
            complete(WaitResult(False, f"episode is done; {ended}", 0., time_s))
        elif interrupted:
            complete(WaitResult(False, interrupted, 0., time_s))
        elif control_steps(duration_s, period) == math.inf:
            complete(WaitResult(False, "duration_s is too large", 0., time_s))
        else:
            self._pending.append(PendingWait(
                time_s, step + control_steps(duration_s, period), complete, tuple(until),
                step + control_steps(timeout_s, period) if timeout_s else math.inf, expire))

    def settle(self, time_s, step, ended=None):
        """ended: as for admit; the end completes every pending wait."""
        remaining = []
        for wait in self._pending:
            if step >= wait.expire_step and not all(done() for done in wait.until):
                wait.expire()
            reached = step >= wait.end_step and all(done() for done in wait.until)
            if reached or ended:
                message = "ok" if reached else "episode ended during wait"
                wait.complete(WaitResult(reached, f"{message}; {ended}" if ended else message,
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
        """End every pending wait now, unsuccessfully, with message."""
        pending, self._pending = self._pending, []
        for wait in pending:
            wait.complete(WaitResult(False, message, max(0., time_s - wait.start_s), time_s))
