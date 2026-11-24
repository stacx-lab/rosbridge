"""Tip goals a joint group steps toward at its own pace, judged on the simulator owner thread. No ROS."""
import numpy as np

from .commands import slerp

# Why a goal ended: the error codes of control_msgs/action/FollowJointTrajectory.
SUCCESSFUL, INVALID_GOAL, PATH_TOLERANCE_VIOLATED, GOAL_TOLERANCE_VIOLATED = 0, -1, -4, -5
# The fastest a tip is stepped, m/s and rad/s: 4 cm and 0.13 rad a 0.04 s step, uni-robot-agent's standard profile.
SPEED = np.array([1., 3.3])
# How near its last pose a tip has reached it, m and rad, unless the goal says: uni-robot-agent's precision profile.
TOLERANCE = np.array([.005, .05])
# World seconds a goal has beyond its nominal end, unless it says.
SETTLE_S = 1.
# A tip that gains too little on its pose over this many steps has stalled.
WINDOW = 4


def offset(a, b):
    """Distance (m) and angle (rad) between poses [x, y, z, qx, qy, qz, qw] with unit quaternions."""
    return np.array([np.linalg.norm(b[:3] - a[:3]), 2 * np.arccos(min(1., abs(a[3:] @ b[3:])))])


class Motion:
    """A tip's goal: poses it passes in turn, each step commanding a pose at most one step from the measured one, so
    a blocked tip is never driven far past where it is.

    times are the poses' nominal times from the start: a segment given more time than SPEED needs is stepped that
    much slower, and one given less, or none, goes at SPEED and nominally takes what that needs. The goal ends when
    the tip is within tolerance of the last pose, when it stalls, or settle_s after its nominal end.
    """

    def __init__(self, pose, time_s, period, poses, times, tolerance=(0., 0.), settle_s=0.):
        self.pose, self.points, self.index, self.gaps = np.asarray(pose), poses, 0, []
        self.tolerance = np.where(np.asarray(tolerance) > 0, tolerance, TOLERANCE)
        self.start_s, self.deadline_s = time_s, time_s + (settle_s or SETTLE_S)
        self.steps, previous, before = [], self.pose, 0.
        for point, time in zip(poses, times):
            needed = max(offset(previous, point) / SPEED)
            given = max(time - before, needed)
            self.steps.append(SPEED * period * (needed / given if given else 1.))
            self.deadline_s += given
            previous, before = point, time

    def reference(self):
        """The pose to command for the next step."""
        point, (reach, turn) = self.points[self.index], self.steps[self.index]
        distance, angle = offset(self.pose, point)
        return np.r_[self.pose[:3] + (point[:3] - self.pose[:3]) * min(1., reach / max(distance, 1e-9)),
                     slerp(self.pose[3:], point[3:], min(1., turn / max(angle, 1e-9)))]

    def observe(self, pose, time_s):
        """Take the pose measured after a step: (code, reason) once the goal ends, None while it goes on."""
        self.pose = np.asarray(pose)
        gap = offset(self.pose, self.points[self.index])
        if np.all(gap <= self.tolerance):
            if self.index == len(self.points) - 1:
                return SUCCESSFUL, "reached"
            self.index, self.gaps = self.index + 1, []
        elif time_s > self.deadline_s - 1e-9:
            return GOAL_TOLERANCE_VIOLATED, "not reached in time"
        else:
            self.gaps = [*self.gaps[1 - WINDOW:], gap]
            gained = self.gaps[0] - np.min(self.gaps[1:], axis=0) if len(self.gaps) == WINDOW else np.inf
            if np.all(gained <= np.minimum(.1 * self.tolerance, .5 * self.steps[self.index])):
                return PATH_TOLERANCE_VIOLATED, "stalled"

    def report(self, reason, time_s):
        distance, angle = offset(self.pose, self.points[-1])
        return f"{reason}: {distance:.3f} m and {angle:.3f} rad from the goal after {time_s - self.start_s:.2f} s"
