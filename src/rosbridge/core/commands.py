"""Latest-target command buffer, written and sampled on the simulator owner thread. No ROS."""

import bisect

import numpy as np

# A base stops this long in world time after its last velocity command.
VELOCITY_TIMEOUT_S = 0.5
# Targets this close outside a limit (rad or m) are held at it, not refused. Measured positions can sit slightly past
# their limits, so a client that sends a measured posture back, or interpolates and rounds past a limit, sends a valid
# target.
LIMIT_TOLERANCE = 1e-3


class CommandError(ValueError):
    pass


def linear(a, b, s):
    return a + s * (b - a)


def slerp(a, b, s):
    """Unit quaternions [x, y, z, w] from a to b along the shorter arc."""
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 1 - 1e-9:
        q = linear(a, b, s)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1 - s) * angle) * a + np.sin(s * angle) * b) / np.sin(angle)
    return q / np.linalg.norm(q)


def pose_blend(a, b, s):
    """Poses [x, y, z, qx, qy, qz, qw]: positions linearly, rotations by slerp."""
    return np.r_[linear(a[:3], b[:3], s), slerp(a[3:], b[3:], s)]


class Trajectory:
    """Piecewise joint targets or tip poses, interpolated by blend; starts from the target held on arrival."""

    def __init__(self, start_s, start_q, times, points, blend=linear):
        if times[0] > 0:
            times, points = [0.0, *times], [start_q, *points]
        self.start_s, self.times, self.points = start_s, list(times), [np.asarray(p) for p in points]
        self.blend = blend

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
        return self.blend(self.points[i], self.points[i + 1], a)


class CommandBuffer:
    """Retain the last valid target per command group of the backend layout.

    Velocity groups start at rest and return to rest VELOCITY_TIMEOUT_S after their last command. Pose groups hold
    a tip pose target, None until their first command; poses() gives each with the action. A group with a tip
    and joint targets is a joint group.
    """

    def __init__(self, layout, initial):
        self.layout = layout
        self._groups = {g.name: g for g in layout.groups}
        self._target = self._vector(initial, len(layout.joint_names), "initial action").copy()
        self._trajectory = dict.fromkeys(self._groups)
        self._expiry = {}
        self._pose = {group.name: None for group in layout.groups if group.tip and not group.joint_targets}
        for group in layout.groups:
            if group.velocity:
                self._target[list(group.indices)] = 0.
        self._open = True

    def stop(self, group, values):
        """Hold values: measured positions, which can fall outside command limits, or a pose group's measured tip
        pose. A pose group never commanded keeps no target: the simulator holds it as it was. A gripper (a scalar
        group) keeps its applied target instead, as Kinex's grip contract requires, since its measured opening
        would release a held object; it starts at the reset opening, so a stop never closes it."""
        if group in self._pose:
            if self._pose[group] is not None or self._trajectory[group] is not None:
                self._pose[group] = self.unit_pose(values, f"{group} hold")
        elif not self._groups[group].scalar:
            self._target[list(self._groups[group].indices)] = values
        self._trajectory[group] = None
        self._expiry.pop(group, None)

    def take(self, group, values):
        """Take values as a group's command in place of the one it has: joints, a gripper's target, a base's
        velocity or a pose group's tip pose, as a backend's goal commands them."""
        if group in self._pose:
            self._pose[group] = self.unit_pose(values, f"{group} command")
        else:
            self._target[list(self._groups[group].indices)] = values
        self._trajectory[group] = None
        self._expiry.pop(group, None)

    def poses(self):
        """Each pose group's tip pose target [x, y, z, qx, qy, qz, qw] in its device base, or None before its first
        command, as the last action() sampled it."""
        return {name: None if pose is None else pose.copy() for name, pose in self._pose.items()}

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

    @classmethod
    def unit_pose(cls, value, what):
        pose = cls._vector(value, 7, what)
        norm = np.linalg.norm(pose[3:])
        if norm < 1e-6:
            raise CommandError(f"{what} needs a unit quaternion")
        return np.r_[pose[:3], pose[3:] / norm]

    @staticmethod
    def _times(start_s, times, points):
        if not np.isfinite(start_s) or not np.isfinite(times).all():
            raise CommandError("trajectory times must be finite")
        if not times or len(times) != len(points):
            raise CommandError("trajectory needs one time per point and at least one point")
        if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise CommandError("trajectory times must be >= 0 and strictly increasing")

    def _open_group(self, name):
        if not self._open:
            raise CommandError("episode is done")
        if name not in self._groups:
            raise CommandError(f"unknown command group: {name}")
        return self._groups[name]

    def _group(self, name, values):
        group = self._open_group(name)
        if name in self._pose:
            raise CommandError(f"{name} takes {group.tip} pose targets, not joint targets")
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
        self._times(start_s, times, points)
        points = [self._group(name, p).copy() for p in points]
        self._trajectory[name] = Trajectory(start_s, self._held(name, start_s), times, points)

    def _held(self, name, t):
        """A joint group's target at time t: its trajectory's, or the one it holds."""
        current = self._trajectory[name]
        return current.sample(t) if current is not None else self._target[list(self._groups[name].indices)].copy()

    def _unit_poses(self, name, start_s, times, poses):
        self._times(start_s, times, poses)
        return [self.unit_pose(pose, f"{name} {self._groups[name].tip} pose") for pose in poses]

    def set_pose_trajectory(self, name, start_s, times, poses, measured):
        """Tip poses [x, y, z, qx, qy, qz, qw] in the device base at times, seconds from start_s, strictly
        increasing and >= 0. It starts from the pose target held on arrival; before a group's first command, from
        measured(), its measured tip pose."""
        group = self._open_group(name)
        if name not in self._pose:
            raise CommandError(f"{name} takes joint targets, not pose targets")
        points = self._unit_poses(name, start_s, times, poses)
        current = self._trajectory[name]
        start = (current.sample(start_s) if current is not None else self._pose[name] if self._pose[name] is not None
                 else self.unit_pose(measured(), f"{name} measured {group.tip} pose"))
        self._trajectory[name] = Trajectory(start_s, start, times, points, pose_blend)

    def action(self, t):
        """Native action for the step that ends at sim time t; poses() holds the pose groups' targets for it."""
        for name, trajectory in self._trajectory.items():
            if trajectory is None:
                continue
            if name in self._pose:
                self._pose[name] = trajectory.sample(t)
            else:
                self._target[list(self._groups[name].indices)] = trajectory.sample(t)
            if t >= trajectory.end_s:
                self._trajectory[name] = None
        for name, expiry in list(self._expiry.items()):
            if t > expiry + 1e-9:
                self._target[list(self._groups[name].indices)] = 0.
                del self._expiry[name]
        return self._target.copy()
