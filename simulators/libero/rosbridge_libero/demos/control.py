"""Finite ROS-only exercise of the arm, gripper and camera endpoints; not a task policy."""
import json
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge.clients import WorldClockClient
from rosbridge_runtime.devices import STREAM_QOS

CAMERAS = ('camera_agentview', 'camera_wrist')


class ControlDemo(Node):
    def __init__(self):
        super().__init__('libero_control_demo')
        self.last, self.image_count = {}, dict.fromkeys(CAMERAS, 0)
        for device in ('arm', 'gripper'):
            self.create_subscription(JointState, f'{device}/joint_state',
                                     lambda m, d=device: self.last.update({d: m}), STREAM_QOS)
        for camera in CAMERAS:
            self.create_subscription(Image, f'{camera}/image', lambda m, c=camera: self.on_image(c, m),
                                     qos_profile_sensor_data)
        self.joint_target = self.create_publisher(JointTrajectory, 'arm/joint_target', 1)
        self.gripper_target = self.create_publisher(Float64, 'gripper/target', 1)
        self.world = WorldClockClient(self)

    def on_image(self, camera, msg):
        if msg.encoding != 'rgb8' or len(msg.data) != msg.width * msg.height * 3:
            raise AssertionError(f'invalid {camera} RGB image')
        self.image_count[camera] += 1
        self.last[camera] = msg

    def wait(self, predicate, timeout=120.):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=.02)
        raise RuntimeError('timed out waiting for the LIBERO graph')

    def spin(self, seconds=.1):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.01)

    def advance(self, seconds):
        # Works in both modes. Stepped admits ticks; realtime observes elapsed sim time.
        result = self.world.wait(seconds)
        if not result.success:
            raise RuntimeError(result.message)
        self.spin()

    def move(self, positions):
        self.joint_target.publish(JointTrajectory(joint_names=list(self.last['arm'].name), points=[
            JointTrajectoryPoint(positions=positions, time_from_start=Duration(sec=1))]))
        self.spin()
        self.advance(1.2)
        return abs(self.last['arm'].position[0] - positions[0])

    def grip(self, target):
        self.gripper_target.publish(Float64(data=target))
        self.spin()
        self.advance(1.)
        return self.last['gripper'].position[0]

    def run(self):
        self.wait(lambda: self.joint_target.get_subscription_count() and self.gripper_target.get_subscription_count())
        self.advance(.2)
        self.wait(lambda: 'arm' in self.last and 'gripper' in self.last)
        start = list(self.last['arm'].position)
        joint_error = self.move([start[0] + .1, *start[1:]])
        if joint_error > .04:
            raise AssertionError(f'joint target was not reached: error={joint_error}')
        if self.move(start) > .04:
            raise AssertionError('joint trajectory did not return to its start')
        closed, opened = self.grip(0.), self.grip(.04)
        if opened <= closed + .01:
            raise AssertionError('gripper did not open after closing')
        if not all(self.image_count.values()):
            raise AssertionError('missing observation camera stream')
        print(json.dumps({'result': 'PASS', 'joint_error_rad': joint_error,
                          'finger_joint1_m': {'closed': closed, 'opened': opened},
                          'image_count': self.image_count}), flush=True)


def main():
    rclpy.init()
    node = ControlDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
