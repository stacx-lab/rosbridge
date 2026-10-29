"""Finite ROS-only exercise of the R1Pro base, torso, arms, grippers and head camera; not a task policy."""
import json
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge.clients import WorldClockClient
from rosbridge_runtime.devices import FRAME_QOS, STREAM_QOS

JOINTS = ("torso", "left_arm", "right_arm")
GRIPPERS = ("left_gripper", "right_gripper")


class ControlDemo(Node):
    def __init__(self):
        super().__init__("behavior_control_demo")
        self.last = {}
        for device in (*JOINTS, *GRIPPERS):
            self.create_subscription(JointState, f"{device}/joint_state",
                                     lambda m, d=device: self.last.update({d: m}), STREAM_QOS)
        self.create_subscription(Odometry, "base/odometry", lambda m: self.last.update(base=m), STREAM_QOS)
        self.joint_target = {d: self.create_publisher(JointTrajectory, f"{d}/joint_target", 1) for d in JOINTS}
        self.grip_target = {d: self.create_publisher(Float64, f"{d}/target", 1) for d in GRIPPERS}
        self.velocity = self.create_publisher(Twist, "base/velocity", 1)
        self.world = WorldClockClient(self)

    def wait(self, predicate, timeout=900.):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=.02)
        raise RuntimeError("timed out waiting for the BEHAVIOR graph")

    def spin(self, seconds=.1):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.01)

    def advance(self, seconds):
        result = self.world.wait(seconds, timeout_s=600.)
        if not result.success:
            raise RuntimeError(result.message)
        self.spin()

    def move(self, device, offset):
        start = list(self.last[device].position)
        goal = [start[0] + offset, *start[1:]]
        self.joint_target[device].publish(JointTrajectory(
            joint_names=list(self.last[device].name),
            points=[JointTrajectoryPoint(positions=goal, time_from_start=Duration(sec=1))]))
        self.spin()
        self.advance(2.)
        return abs(self.last[device].position[0] - goal[0])

    def grip(self, target):
        for publisher in self.grip_target.values():
            publisher.publish(Float64(data=target))
        self.spin()
        self.advance(1.)
        return [self.last[d].position[0] for d in GRIPPERS]

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
        self.wait(lambda: all(p.get_subscription_count() for p in (
            *self.joint_target.values(), *self.grip_target.values(), self.velocity)))
        self.advance(.2)
        self.wait(lambda: all(k in self.last for k in (*JOINTS, *GRIPPERS, "base")))
        errors = {device: self.move(device, .1) for device in JOINTS}
        if max(errors.values()) > .05:
            raise AssertionError(f"joint targets were not reached: {errors}")
        closed, opened = self.grip(-1.), self.grip(1.)
        if max(closed) > -.8 or min(opened) < .8:
            raise AssertionError(f"grippers did not close and open: {closed=}, {opened=}")
        turned, residual = self.turn(1.)
        if turned < .05 or residual > .05:
            raise AssertionError(f"base did not turn and stop: {turned=}, {residual=}")
        size = self.frame()
        print(json.dumps({"result": "PASS", "joint_errors_rad": errors,
                          "grippers": {"closed": closed, "opened": opened},
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
