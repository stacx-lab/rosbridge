"""Latest-target command buffer, written and sampled on the simulator owner thread. No ROS."""

import bisect

import numpy as np

# A base stops this long in world time after its last velocity command (Kinex locomotion watchdog).
VELOCITY_TIMEOUT_S = 0.5
# Targets this close outside a limit (rad or m) are clipped to it, not refused: interpolation rounding (0.2 - 1e-17) and
# measured positions a joint's soft limit lets past its bound (-0.0005 rad) are valid targets.
LIMIT_TOLERANCE = 1e-3


class CommandError(ValueError):
    pass


class Trajectory:
    """Piecewise-linear joint targets; starts from the target held on arrival."""

    def __init__(self, start_s, start_q, times, points):
        if times[0] > 0:
            times, points = [0.0, *times], [start_q, *points]
        self.start_s, self.times, self.points = start_s, list(times), [np.asarray(p) for p in points]

    @property
    def end_s(self):
        return self.start_s + self.times[-1]

    def sample(self, t):
        dt = t - self.start_s
        if dt <= self.times[0]:
            return self.points[0]
        if dt >= self.times[-1]:
            return self.points[-1]
        i = bisect.bisect_right(self.times, dt) - 1
        a = (dt - self.times[i]) / (self.times[i + 1] - self.times[i])
        return self.points[i] + a * (self.points[i + 1] - self.points[i])


class CommandBuffer:
    """Retain the last valid target per command group of the backend layout.

    Velocity groups start at rest and return to rest VELOCITY_TIMEOUT_S after their last command.
    """

    def __init__(self, layout, initial):
        self.layout = layout
        self._groups = {g.name: g for g in layout.groups}
        self._target = self._vector(initial, len(layout.joint_names), "initial action").copy()
        self._trajectory = dict.fromkeys(self._groups)
        self._expiry = {}
        for group in layout.groups:
            if group.velocity:
                self._target[list(group.indices)] = 0.
        self._open = True

    def stop(self, group, values):
        # Measured positions can fall outside command limits.
        self._target[list(self._groups[group].indices)] = values
        self._trajectory[group] = None
        self._expiry.pop(group, None)

    def close(self):
        self._open = False

    @staticmethod
    def _vector(values, size, what):
        try:
            v = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise CommandError(f"{what} must contain numbers") from None
        if v.shape != (size,):
            raise CommandError(f"{what} needs {size} values, got {v.size}")
        if not np.all(np.isfinite(v)):
            raise CommandError(f"{what} has non-finite values")
        return v

    def _group(self, name, values):
        if not self._open:
            raise CommandError("episode is done")
        if name not in self._groups:
            raise CommandError(f"unknown command group: {name}")
        group = self._groups[name]
        q = self._vector(values, len(group.indices), f"{name} target")
        lo, hi = np.asarray(group.lower), np.asarray(group.upper)
        bad = np.flatnonzero((q < lo - LIMIT_TOLERANCE) | (q > hi + LIMIT_TOLERANCE))
        if bad.size:
            i = bad[0]
            joint = self.layout.joint_names[group.indices[i]]
            raise CommandError(f"{joint}={q[i]:.4f} outside [{lo[i]:.4f}, {hi[i]:.4f}]")
        return np.clip(q, lo, hi)

    def set_group(self, name, values, time_s=None):
        """time_s: world time of a velocity command, which expires VELOCITY_TIMEOUT_S later."""
        q = self._group(name, values)
        if self._groups[name].velocity:
            if time_s is None:
                raise CommandError(f"{name} velocity needs its command time")
            self._expiry[name] = time_s + VELOCITY_TIMEOUT_S
        self._target[list(self._groups[name].indices)] = q
        self._trajectory[name] = None

    def set_trajectory(self, name, start_s, times, points):
        """times: seconds from start_s, strictly increasing and >= 0."""
        if name in self._groups and self._groups[name].velocity:
            raise CommandError(f"{name} accepts velocities, not trajectories")
        if not np.isfinite(start_s) or not np.isfinite(times).all():
            raise CommandError("trajectory times must be finite")
        if not times or len(times) != len(points):
            raise CommandError("trajectory needs one time per point and at least one point")
        if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise CommandError("trajectory times must be >= 0 and strictly increasing")
        points = [self._group(name, p).copy() for p in points]
        current = self._trajectory[name]
        start_q = (current.sample(start_s) if current is not None
                   else self._target[list(self._groups[name].indices)].copy())
        self._trajectory[name] = Trajectory(start_s, start_q, times, points)

    def action(self, t):
        """Native action for the step that ends at sim time t."""
        for name, trajectory in self._trajectory.items():
            if trajectory is None:
                continue
            self._target[list(self._groups[name].indices)] = trajectory.sample(t)
            if t >= trajectory.end_s:
                self._trajectory[name] = None
        for name, expiry in list(self._expiry.items()):
            if t > expiry + 1e-9:
                self._target[list(self._groups[name].indices)] = 0.
                del self._expiry[name]
        return self._target.copy()
