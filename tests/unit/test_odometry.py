"""The base's odometry message (rosbridge_runtime.conversions.odometry): planar for a base that cannot tilt, in full
(z, roll, pitch and their rates) for one whose backend keeps base_attitude."""
import math

import numpy as np
import pytest

pytest.importorskip("nav_msgs.msg")
from rosbridge_runtime import conversions as conv  # noqa: E402


def matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def turn(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def test_a_planar_base_publishes_x_y_yaw_and_vx_vy_wz_only():
    message = conv.odometry(conv.stamp(2.5), "base", (1., -2., .7), (.3, -.1, .2))
    pose, twist = message.pose.pose, message.twist.twist
    assert (message.header.frame_id, message.child_frame_id) == ("base/odom", "base/base")
    assert (pose.position.x, pose.position.y, pose.position.z) == (1., -2., 0.)
    assert (pose.orientation.x, pose.orientation.y) == (0., 0.)
    assert (pose.orientation.z, pose.orientation.w) == (math.sin(.35), math.cos(.35))
    assert (twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z) == (
        .3, -.1, 0., 0., 0., .2)


def test_a_tilting_base_publishes_its_full_pose_and_twist():
    attitude = {"z": .12, "roll": .5, "pitch": -.3, "vz": .04, "wx": .7, "wy": -.6, "tilt": 0.}
    message = conv.odometry(conv.stamp(2.5), "base", (1., -2., 2.4), (.3, -.1, .2), attitude)
    pose, twist = message.pose.pose, message.twist.twist
    assert (pose.position.x, pose.position.y, pose.position.z) == (1., -2., .12)
    q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    assert sum(v * v for v in q) == pytest.approx(1.)
    assert matrix(q) == pytest.approx(turn("z", 2.4) @ turn("y", -.3) @ turn("x", .5))  # ZYX, as an IMU reports
    assert (twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z) == (
        .3, -.1, .04, .7, -.6, .2)
