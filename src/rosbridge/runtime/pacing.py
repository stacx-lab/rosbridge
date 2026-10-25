"""Wall-clock pacing; fixed simulation periods are never changed or skipped."""
import math
import time

MODES = ("realtime", "stepped")


class Pacer:
    def __init__(self, mode, period, clock=time.monotonic):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self._clock = mode, clock
        self.reset(period)

    def reset(self, period):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("control period must be finite and positive")
        self.period = period
        self.next_tick = self._clock()

    def ready(self, pending=False):
        return self._clock() >= self.next_tick if self.mode == "realtime" else pending

    def advanced(self):
        self.next_tick += self.period

    def wait_s(self, pending=False, idle=0.05):
        if self.mode == "realtime":
            return min(idle, max(0., self.next_tick - self._clock()))
        return 0. if pending else idle
