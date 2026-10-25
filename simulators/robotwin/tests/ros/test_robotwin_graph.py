"""The Kinex-facing RoboTwin graph: native joint names, normalized grippers, RGB-D and spectators."""
import numpy as np
import pytest
from builtin_interfaces.msg import Duration
from kinex_interfaces.srv import CaptureRGBD
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds

# Scene construction runs upstream's expert dry run; allow minutes to start or reset.
STARTUP_S = RESET_S = 600.


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_robotwin_graph_serves_the_kinex_devices(tmp_path, mode):
    graph = Graph(tmp_path, backend="robotwin", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        for kind, topic, key in ((JointState, "left_arm/joint_state", "arm"),
                                 (JointState, "left_gripper/joint_state", "gripper"),
                                 (CameraInfo, "head_camera/camera_info", "info"),
                                 (Image, "world/spectator/third_person/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        joint_target = node.create_publisher(JointTrajectory, "left_arm/joint_target", 1)
        gripper_target = node.create_publisher(Float64, "left_gripper/target", 1)
        capture = node.create_client(CaptureRGBD, "left_camera/capture")
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: joint_target.get_subscription_count() and gripper_target.get_subscription_count()
                    and capture.service_is_ready() and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("arm", "gripper", "info", "spectator")))
        assert last["arm"].name == [f"fl_joint{i}" for i in range(1, 7)]
        assert last["gripper"].name == ["left_gripper"]
        assert (last["info"].width, last["info"].height, last["info"].header.frame_id) == (
            320, 240, "head_camera/base")
        assert (last["spectator"].width, last["spectator"].height) == (960, 720)
        target = list(last["arm"].position)
        target[0] += .1
        joint_target.publish(JointTrajectory(joint_names=list(last["arm"].name), points=[
            JointTrajectoryPoint(positions=target, time_from_start=Duration(nanosec=500_000_000))]))
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert abs(last["arm"].position[0] - target[0]) < .03
        assert graph.call(stop, Trigger.Request()).success
        opened = last["gripper"].position[0]
        gripper_target.publish(Float64(data=0.))
        graph.spin(.1)
        assert graph.wait(1.5).success
        graph.spin(.1)
        assert last["gripper"].position[0] < opened - .5
        response = graph.call(capture, CaptureRGBD.Request())
        image, depth = response.image, response.depth_image
        assert image.header.frame_id == depth.header.frame_id == "left_camera/base"
        assert (depth.encoding, depth.width, depth.height) == ("16UC1", 320, 240)
        assert (np.frombuffer(bytes(depth.data), np.uint16) > 0).mean() > .9
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before)
        assert graph.reset(RESET_S).success
    finally:
        graph.close()
