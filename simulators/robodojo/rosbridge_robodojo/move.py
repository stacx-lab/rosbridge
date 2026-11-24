"""The arms' move action: goals of flange poses an arm steps toward at its own pace. No ROS and no Isaac: a goal,
its feedback and its result are the fields of the action's messages."""
import numpy as np

from rosbridge.core.commands import CommandBuffer, slerp

TYPE = "control_msgs/action/FollowJointTrajectory"
# Why a goal ended: the action's error codes.
SUCCESSFUL, INVALID_GOAL, PATH_TOLERANCE_VIOLATED, GOAL_TOLERANCE_VIOLATED = 0, -1, -4, -5
# The action's components of a pose that a tolerance names: its distance and its angle.
TRANSLATION, ROTATION = 4, 5
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


def seconds(duration):
    return duration["sec"] + duration["nanosec"] * 1e-9


def motion(goal, tip, frame, pose, time_s, period):
    """The Motion of a goal, the fields of the action's Goal, for the tip measured at pose in frame. A ValueError
    says why it is none."""
    path = goal["multi_dof_trajectory"]
    if path["header"]["frame_id"] not in ("", frame) or path["joint_names"] != [tip] or not path["points"] or any(
            len(point["transforms"]) != 1 for point in path["points"]):
        raise ValueError(f"a goal is a multi_dof_trajectory of {tip} poses in {frame}")
    poses = [CommandBuffer.unit_pose([*(transform["translation"][axis] for axis in "xyz"),
                                      *(transform["rotation"][axis] for axis in "xyzw")], "goal pose")
             for transform in (point["transforms"][0] for point in path["points"])]
    tolerance = {item["component"]: item["position"] for item in goal["component_goal_tolerance"]}
    return Motion(pose, time_s, period, poses, [seconds(point["time_from_start"]) for point in path["points"]],
                  (tolerance.get(TRANSLATION, 0.), tolerance.get(ROTATION, 0.)), seconds(goal["goal_time_tolerance"]))


def transform(pose):
    x, y, z, qx, qy, qz, qw = map(float, pose)
    return {"transforms": [{"translation": {"x": x, "y": y, "z": z}, "rotation": {"x": qx, "y": qy, "z": qz, "w": qw}}]}


def feedback(tip, frame, time_s, target, pose):
    """The action's Feedback for a step: the pose it commands and the measured one."""
    nanoseconds = round(time_s * 1e9)
    return {"header": {"stamp": {"sec": nanoseconds // 10**9, "nanosec": nanoseconds % 10**9}, "frame_id": frame},
            "multi_dof_joint_names": [tip], "multi_dof_desired": transform(target), "multi_dof_actual": transform(pose)}


def ended(code, text, commands):
    """A goal that has ended, as a backend reports it: the action's Result and what it leaves its arm commanded."""
    return {"status": "succeeded" if code == SUCCESSFUL else "aborted", "feedback": None, "commands": commands,
            "result": {"error_code": code, "error_string": text}}
