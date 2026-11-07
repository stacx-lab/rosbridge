"""The Kinex-facing BEHAVIOR graph: a Python 3.12 world node proxying OmniGibson in a worker."""
import numpy as np
import psutil
import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import FRAME_QOS, STREAM_QOS
from tests.ros.support import Graph, seconds

# OmniGibson boots Isaac Sim and loads the house and the instance before serving.
STARTUP_S = 1800.


def workers(graph):
    """Native worker processes of this graph, identified by its leased ROS domain."""
    found = []
    for process in psutil.process_iter(["cmdline"]):
        try:
            if ("rosbridge.core.worker_main" in " ".join(process.info["cmdline"] or ())
                    and process.environ().get("ROS_DOMAIN_ID") == str(graph.lease.domain)):
                found.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


@pytest.mark.parametrize("mode", ["stepped"])
def test_behavior_graph_serves_the_r1pro_from_a_native_worker(tmp_path, mode):
    graph = Graph(tmp_path, backend="behavior", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "behavior-native" in native[0].exe()
        for kind, topic, key in ((JointState, "torso/joint_state", "torso"),
                                 (JointState, "left_arm/joint_state", "arm"),
                                 (JointState, "right_gripper/joint_state", "gripper"),
                                 (Odometry, "base/odometry", "base"),
                                 (CameraInfo, "head_camera/camera_info", "head"),
                                 (CameraInfo, "left_camera/camera_info", "wrist"),
                                 (Image, "world/spectator/front_left/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        joint_target = node.create_publisher(JointTrajectory, "left_arm/joint_target", 1)
        gripper_target = node.create_publisher(Float64, "right_gripper/target", 1)
        velocity = node.create_publisher(Twist, "base/velocity", 1)
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: joint_target.get_subscription_count() and gripper_target.get_subscription_count()
                    and velocity.get_subscription_count() and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("torso", "arm", "gripper", "base", "head", "wrist", "spectator")))
        assert last["arm"].name == [f"left_arm_joint{i}" for i in range(1, 8)]
        assert last["gripper"].name == ["right_gripper"] and last["gripper"].position[0] > .9  # starts open
        # The evaluator's full-resolution cameras: a 720x720 head and 480x480 wrists.
        assert (last["head"].width, last["head"].height, last["head"].k[2]) == (720, 720, 359.5)
        assert (last["wrist"].width, last["wrist"].height, last["wrist"].k[2]) == (480, 480, 239.5)
        assert last["head"].header.frame_id == "head_camera/base"
        target = list(last["arm"].position)
        target[3] -= .2
        joint_target.publish(JointTrajectory(joint_names=list(last["arm"].name), points=[
            JointTrajectoryPoint(positions=target, time_from_start=Duration(nanosec=500_000_000))]))
        gripper_target.publish(Float64(data=-1.))
        twist = Twist()
        twist.linear.x = .5
        velocity.publish(twist)
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert abs(last["arm"].position[3] - target[3]) < .03
        assert last["gripper"].position[0] < -.9
        assert last["base"].pose.pose.position.x > .1 and last["base"].child_frame_id == "base/base"
        assert graph.call(stop, Trigger.Request()).success
        image, depth = graph.frame("head_camera", depth=True, timeout=120.)
        assert (image.width, image.height, depth.encoding) == (720, 720, "16UC1")
        assert np.frombuffer(bytes(image.data), np.uint8).std() > 1
        assert (np.frombuffer(bytes(depth.data), np.uint16) > 0).mean() > .9
        # A stream renders once per step: a frame of every step reaches a steady subscriber.
        stamps = []
        stream = node.create_subscription(Image, "head_camera/image",
                                          lambda msg: stamps.append(seconds(msg.header.stamp)), FRAME_QOS)
        graph.spin(1.)
        assert graph.wait(.5).success
        graph.spin(.5)
        assert stamps == sorted(stamps) and len(set(stamps)) >= 10
        node.destroy_subscription(stream)
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before, 60.)
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)
