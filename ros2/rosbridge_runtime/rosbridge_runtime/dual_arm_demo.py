"""Finite ROS-only control acceptance for bimanual rigs; no simulator imports or policy.

Simulators with left/right arm and gripper devices run it with their observation
camera. Joint names come from measured joint states, so native names are kept.
"""
import json
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge.clients import WorldClockClient
from rosbridge_runtime.devices import FRAME_QOS, STREAM_QOS


class ControlDemo(Node):
    def __init__(self, name, camera, depth):
        super().__init__(name)
        self.camera, self.depth, self.last = camera, depth, {}
        self.info_count = 0
        self.create_subscription(Clock, "/clock", lambda msg: self.last.update(clock=msg), STREAM_QOS)
        for device in ("left_arm", "right_arm", "right_gripper"):
            self.create_subscription(JointState, f"{device}/joint_state",
                                     lambda msg, d=device: self.last.update({d: msg}), STREAM_QOS)
        self.create_subscription(CameraInfo, f"{camera}/camera_info", self.on_info, STREAM_QOS)
        self.targets = {side: self.create_publisher(JointTrajectory, f"{side}_arm/joint_target", 1)
                        for side in ("left", "right")}
        self.gripper = self.create_publisher(Float64, "right_gripper/target", 1)
        self.world = WorldClockClient(self)

    def on_info(self, msg):
        self.info_count += 1

    def frame(self):
        """The camera's current frame as a Kinex reader takes it: subscribe, then the first
        image and, for RGB-D, the equally stamped depth."""
        received = {"image": [], "depth": []}
        names = ("image", "depth") if self.depth else ("image",)
        subscriptions = [self.create_subscription(Image, f"{self.camera}/{name}",
                                                  lambda msg, name=name: received[name].append(msg),
                                                  FRAME_QOS) for name in names]

        def pair():
            depths = {(m.header.stamp.sec, m.header.stamp.nanosec): m for m in received["depth"]}
            for image in received["image"]:
                stamp = image.header.stamp.sec, image.header.stamp.nanosec
                if not self.depth or stamp in depths:
                    return image, depths.get(stamp)
        try:
            self.wait(pair)
            return pair()
        finally:
            for subscription in subscriptions:
                self.destroy_subscription(subscription)

    def wait(self, predicate, timeout=30):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.02)
        raise RuntimeError("timed out waiting for ROS graph or control result")

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def advance(self, duration_s=2.04):
        result = self.world.wait(duration_s)
        if not result.success:
            raise RuntimeError(f"world wait failed: {result.message}")
        self.wait(lambda: self.last.get("clock") and self.sim_time() + 1e-9 >= result.time_s)
        self.spin(0.1)  # allow observation callbacks to drain

    def positions(self, side):
        return list(self.last[f"{side}_arm"].position)

    def target(self, side, positions):
        self.targets[side].publish(JointTrajectory(
            joint_names=list(self.last[f"{side}_arm"].name),
            points=[JointTrajectoryPoint(positions=positions, time_from_start=Duration(sec=1))]))

    def grip_position(self):
        return self.last["right_gripper"].position[0]

    def sim_time(self):
        stamp = self.last["clock"].clock
        return stamp.sec + stamp.nanosec * 1e-9

    def run(self):
        # Backends may build scenes or start renderers for minutes before serving devices.
        self.wait(lambda: all(pub.get_subscription_count() for pub in (*self.targets.values(), self.gripper)),
                  timeout=900)
        self.wait(lambda: all(key in self.last for key in ("left_arm", "right_arm", "right_gripper", "clock")))
        # Settle before taking a reference position or measuring throughput.
        self.advance(0.34)
        opened = self.grip_position()
        start = self.positions("left")
        goal = [start[0] + 0.2, *start[1:]]
        self.target("left", goal)
        self.gripper.publish(Float64(data=0.0))
        self.spin(0.1)
        wall_start, sim_start = time.monotonic(), self.sim_time()
        infos_start = self.info_count
        self.advance()
        elapsed_wall, elapsed_sim = time.monotonic() - wall_start, self.sim_time() - sim_start
        if elapsed_sim <= 0 or abs(self.positions("left")[0] - goal[0]) > 0.03:
            raise AssertionError("left arm did not reach its target with an advancing clock")
        # Fingers stop on each other before a squeezing target, so compare with the open state.
        if self.grip_position() > opened - 0.5:
            raise AssertionError("right gripper did not close")
        if self.info_count <= infos_start:
            raise AssertionError("top camera did not publish during motion")
        image, depth = self.frame()
        if image.encoding != "rgb8" or len(image.data) != image.width * image.height * 3:
            raise AssertionError("invalid camera image")
        if depth is not None and (depth.encoding != "16UC1" or depth.header != image.header
                                  or (depth.width, depth.height) != (image.width, image.height)):
            raise AssertionError("invalid registered depth image")
        rtf = elapsed_sim / elapsed_wall
        self.target("left", [99.0] * 6)
        self.spin(0.1)
        self.advance(0.17)
        if abs(self.positions("left")[0] - goal[0]) > 0.03:
            raise AssertionError("an invalid target changed the held target")
        right = self.positions("right")
        right_goal = [right[0] + 0.15, *right[1:]]
        self.target("right", right_goal)
        self.target("left", start)
        self.gripper.publish(Float64(data=1.0))
        self.spin(0.1)
        self.advance()
        if abs(self.positions("right")[0] - right_goal[0]) > 0.03:
            raise AssertionError("right arm trajectory did not reach its endpoint")
        if abs(self.positions("left")[0] - start[0]) > 0.03 or self.grip_position() < 0.8:
            raise AssertionError("return motion or gripper opening failed")
        print(json.dumps({"result": "PASS",
                          "control_test": "arm trajectories, gripper cycle, target rejection, "
                                          + ("RGB-D frame" if self.depth else "RGB frame"),
                          "real_time_factor": round(rtf, 3),
                          "measured_sim_seconds": round(elapsed_sim, 3),
                          "measured_wall_seconds": round(elapsed_wall, 3),
                          "image_size": [image.width, image.height],
                          "camera_info_count": self.info_count}), flush=True)


def main(name, camera, depth=True):
    rclpy.init()
    node = ControlDemo(name, camera, depth)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
