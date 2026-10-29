"""Finite ROS-only exercise of the G1's walking base, arms, Sharpa hands and head camera; not a task policy."""
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

JOINTS = ("left_arm", "right_hand")


class ControlDemo(Node):
    def __init__(self):
        super().__init__("behavior_g1_control_demo")
        self.last = {}
        for device in JOINTS:
            self.create_subscription(JointState, f"{device}/joint_state",
                                     lambda m, d=device: self.last.update({d: m}), STREAM_QOS)
        self.create_subscription(Odometry, "base/odometry", lambda m: self.last.update(base=m), STREAM_QOS)
        self.joint_target = {d: self.create_publisher(JointTrajectory, f"{d}/joint_target", 1) for d in JOINTS}
        self.velocity = self.create_publisher(Twist, "base/velocity", 1)
        self.world = WorldClockClient(self)

    def wait(self, predicate, timeout=1200.):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=.02)
        raise RuntimeError("timed out waiting for the BEHAVIOR G1 graph")

    def spin(self, seconds=.1):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.01)

    def advance(self, seconds):
        result = self.world.wait(seconds, timeout_s=600.)
        if not result.success:
            raise RuntimeError(result.message)
        self.spin()

    def drive(self, seconds, **axes):
        # The base stops 0.5 s of world time after the last twist, so keep sending.
        twist = Twist()
        twist.linear.x, twist.angular.z = axes.get("vx", 0.), axes.get("wz", 0.)
        for _ in range(round(seconds / .25)):
            self.velocity.publish(twist)
            self.spin()
            self.advance(.25)
        self.advance(1.)
        pose = self.last["base"].pose.pose
        return pose.position.x, pose.orientation.z

    def move(self, device, offsets):
        start = list(self.last[device].position)
        goal = [q + offsets.get(i, 0.) for i, q in enumerate(start)]
        self.joint_target[device].publish(JointTrajectory(
            joint_names=list(self.last[device].name),
            points=[JointTrajectoryPoint(positions=goal, time_from_start=Duration(sec=1))]))
        self.spin()
        self.advance(2.)
        return max(abs(a - b) for a, b in zip(self.last[device].position, goal))

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
        walked, _ = self.drive(4., vx=.3)
        _, turned = self.drive(2., wz=.4)
        if walked < .3 or turned < .1:
            raise AssertionError(f"the G1 did not walk and turn: {walked=}, {turned=}")
        arm_error = self.move("left_arm", {3: .4})
        hand_error = self.move("right_hand", {i: .6 for i in range(5, 22)})
        if arm_error > .1 or hand_error > .3:
            raise AssertionError(f"joint targets were not reached: {arm_error=}, {hand_error=}")
        size = self.frame()
        print(json.dumps({"result": "PASS", "walked_m": walked, "base_odometry_qz": turned,
                          "arm_error_rad": arm_error, "hand_error_rad": hand_error,
                          "head_image_size": size}), flush=True)


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
