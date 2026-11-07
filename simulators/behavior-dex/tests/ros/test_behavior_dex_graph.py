"""The Kinex-facing BEHAVIOR graph with the R1Pro and Sharpa hands, proxying OmniGibson in a worker."""
import numpy as np
import psutil
import pytest
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds

# OmniGibson boots Isaac Sim and loads the house and the instance before serving.
STARTUP_S = 1800.


def workers(graph):
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
def test_behavior_dex_graph_serves_the_hands_and_wrist_cameras(tmp_path, mode):
    graph = Graph(tmp_path, backend="behavior-dex", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "behavior-dex-native" in native[0].exe()
        for kind, topic, key in ((JointState, "left_hand/joint_state", "hand"),
                                 (Odometry, "base/odometry", "base"),
                                 (CameraInfo, "left_camera/camera_info", "wrist"),
                                 (Image, "world/spectator/front_left/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        hand_target = node.create_publisher(JointTrajectory, "left_hand/joint_target", 1)
        graph.until(lambda: hand_target.get_subscription_count())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("hand", "base", "wrist", "spectator")))
        assert len(last["hand"].name) == 22 and last["hand"].name[0] == "left_thumb_CMC_FE"
        assert np.abs(last["hand"].position).max() < .05  # starts open and flat
        assert (last["wrist"].width, last["wrist"].height) == (480, 480)
        # Open and flat is zero, within every limit; measured positions can sit a hair below a
        # lower limit of zero, which a target may not.
        target = [0.] * 22
        for i in (5, 7, 8):  # curl the index finger
            target[i] += .6
        hand_target.publish(JointTrajectory(joint_names=list(last["hand"].name), points=[
            JointTrajectoryPoint(positions=target, time_from_start=Duration(nanosec=500_000_000))]))
        graph.spin(.1)
        assert graph.wait(1.5).success
        graph.spin(.1)
        np.testing.assert_allclose([last["hand"].position[i] for i in (5, 7, 8)],
                                   [target[i] for i in (5, 7, 8)], atol=.1)
        image, depth = graph.frame("left_camera", depth=True, timeout=120.)
        assert (image.width, image.height, depth.encoding) == (480, 480, "16UC1")
        assert np.frombuffer(bytes(image.data), np.uint8).std() > 1
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before, 60.)
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)
