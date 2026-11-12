"""The Kinex-facing BiDex-100 graph: arms, Sharpa hands, RGB-D cameras and spectators on the world node."""
import numpy as np
import pytest
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds


def trajectory(names, positions):
    return JointTrajectory(joint_names=list(names), points=[
        JointTrajectoryPoint(positions=list(positions), time_from_start=Duration(nanosec=500_000_000))])


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_bidex_graph_serves_arms_hands_and_rgbd_cameras(tmp_path, mode):
    graph = Graph(tmp_path, backend="bidex", mode=mode, spectators=True)
    node, last = graph.node, {}
    try:
        for kind, topic, key in ((JointState, "left_arm/joint_state", "arm"),
                                 (JointState, "right_hand/joint_state", "hand"),
                                 (CameraInfo, "head_camera/camera_info", "info"),
                                 (Image, "world/spectator/third_person/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        arm_target = node.create_publisher(JointTrajectory, "left_arm/joint_target", 1)
        hand_target = node.create_publisher(JointTrajectory, "right_hand/joint_target", 1)
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: arm_target.get_subscription_count() and hand_target.get_subscription_count()
                    and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("arm", "hand", "info", "spectator")))
        assert last["arm"].name == [f"left_fr3v2_joint{i}" for i in range(1, 8)]
        assert len(last["hand"].name) == 22 and last["hand"].name[0] == "right_thumb_CMC_FE"
        assert (last["info"].width, last["info"].height, last["info"].header.frame_id) == (
            480, 360, "head_camera/base")
        assert (last["spectator"].width, last["spectator"].height) == (960, 720)
        # Measured Sharpa joints can read past their limits, so the hand target is an in-limit posture.
        arm, hand = list(last["arm"].position), [0.] * 22
        arm[0] += .1
        hand[5] = .8  # index-finger MCP flexion of the open hand
        arm_target.publish(trajectory(last["arm"].name, arm))
        hand_target.publish(trajectory(last["hand"].name, hand))
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert abs(last["arm"].position[0] - arm[0]) < .02
        assert abs(last["hand"].position[5] - hand[5]) < .05
        assert graph.call(stop, Trigger.Request()).success
        image, depth = graph.frame("left_camera", depth=True)
        assert image.header.frame_id == "left_camera/base" and (image.width, image.height) == (480, 360)
        assert depth.header == image.header and depth.encoding == "16UC1"
        assert (np.frombuffer(bytes(depth.data), np.uint16) > 0).mean() > .9
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before)
    finally:
        graph.close()
