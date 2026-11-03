"""Finite ROS-only exercise of the R1Pro's base, torso, arm, Sharpa hand and head camera; not a task policy."""
import json
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge.clients import WorldClockClient
from rosbridge_runtime.devices import FRAME_QOS, STREAM_QOS

JOINTS = ("torso", "left_arm", "right_hand")
# Flexion of the index, middle and ring fingers (MCP, PIP, DIP) in the hand's joint order.
CURL = {i: .6 for finger in (5, 9, 13) for i in (finger, finger + 2, finger + 3)}


class ControlDemo(Node):
    def __init__(self):
        super().__init__("behavior_dex_control_demo")
        self.last = {}
        for device in JOINTS:
            self.create_subscription(JointState, f"{device}/joint_state",
                                     lambda m, d=device: self.last.update({d: m}), STREAM_QOS)
        self.create_subscription(Odometry, "base/odometry", lambda m: self.last.update(base=m), STREAM_QOS)
        self.joint_target = {d: self.create_publisher(JointTrajectory, f"{d}/joint_target", 1) for d in JOINTS}
        self.velocity = self.create_publisher(Twist, "base/velocity", 1)
        self.world = WorldClockClient(self)

    def wait(self, predicate, timeout=900.):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=.02)
        raise RuntimeError("timed out waiting for the BEHAVIOR dex graph")

    def spin(self, seconds=.1):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.01)

    def advance(self, seconds):
        result = self.world.wait(seconds, timeout_s=600.)
        if not result.success:
            raise RuntimeError(result.message)
        self.spin()

    def move(self, device, offsets, start=None):
        start = list(self.last[device].position) if start is None else start
        goal = [q + offsets.get(i, 0.) for i, q in enumerate(start)]
        self.joint_target[device].publish(JointTrajectory(
            joint_names=list(self.last[device].name),
            points=[JointTrajectoryPoint(positions=goal, time_from_start=Duration(sec=1))]))
        self.spin()
        self.advance(2.)
        return max(abs(a - b) for a, b in zip(self.last[device].position, goal))

    def turn(self, seconds, rate=.5):
        # The base stops 0.5 s of world time after the last twist, so keep sending.
        twist = Twist()
        twist.angular.z = rate
        for _ in range(round(seconds / .25)):
            self.velocity.publish(twist)
            self.spin()
            self.advance(.25)
        self.advance(1.)
        return self.last["base"].pose.pose.orientation.z, abs(self.last["base"].twist.twist.angular.z)

    def frame(self):
        received = {"image": [], "depth": []}
        subscriptions = [self.create_subscription(Image, f"head_camera/{name}",
                                                  lambda m, n=name: received[n].append(m), FRAME_QOS)
                         for name in received]
        try:
            self.wait(lambda: received["image"] and received["depth"], timeout=120.)
        finally:
            for subscription in subscriptions:
                self.destroy_subscription(subscription)
        image, depth = received["image"][0], received["depth"][0]
        if (image.encoding, depth.encoding) != ("rgb8", "16UC1") or depth.header != image.header:
            raise AssertionError("invalid head RGB-D frame")
        return image.width, image.height

    def run(self):
        self.wait(lambda: all(p.get_subscription_count() for p in (*self.joint_target.values(), self.velocity)))
        self.advance(.2)
        self.wait(lambda: all(k in self.last for k in (*JOINTS, "base")))
        errors = {device: self.move(device, {0: .1}) for device in ("torso", "left_arm")}
        # From the open hand at zero: measured positions can sit a hair below a lower limit of
        # zero, and a target outside the limits is rejected.
        hand_error = self.move("right_hand", CURL, start=[0.] * len(self.last["right_hand"].position))
        if max(errors.values()) > .05 or hand_error > .2:
            raise AssertionError(f"joint targets were not reached: {errors=}, {hand_error=}")
        turned, residual = self.turn(1.)
        if turned < .05 or residual > .05:
            raise AssertionError(f"base did not turn and stop: {turned=}, {residual=}")
        size = self.frame()
        print(json.dumps({"result": "PASS", "joint_errors_rad": errors, "hand_error_rad": hand_error,
                          "base_odometry_qz": turned, "head_image_size": size}), flush=True)


def main():
    rclpy.init()
    node = ControlDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
