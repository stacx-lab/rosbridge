"""The Kinex-facing HandsOn graph: a Python 3.12 world node proxying Isaac Lab in a worker."""
import numpy as np
import psutil
import pytest
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds

# Isaac Sim boots, converts the task's MJCF scene and settles the recording origin before serving.
STARTUP_S = 1200.
# A step and a spectator sample each cost path-traced seconds of wall time.
WAIT_S = 600.


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


def trajectory(names, positions, nanosec=500_000_000):
    return JointTrajectory(joint_names=list(names), points=[
        JointTrajectoryPoint(positions=list(positions), time_from_start=Duration(nanosec=nanosec))])


@pytest.mark.parametrize("mode", ["stepped"])
def test_handson_graph_serves_the_kinex_devices_from_a_native_worker(tmp_path, monkeypatch, mode):
    # The graph is stopped by signal, before the worker removes its episode's files; pytest prunes these.
    monkeypatch.setenv("ROSBRIDGE_RUN_DIR", str(tmp_path))
    graph = Graph(tmp_path, backend="handson", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "handson-native" in native[0].exe()
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
        assert graph.wait(.2, WAIT_S).success
        graph.until(lambda: all(k in last for k in ("arm", "hand", "info", "spectator")))
        assert last["arm"].name == [f"joint{i}_left" for i in range(1, 8)]
        assert last["hand"].name[:4] == [f"right_finger1_joint{j}" for j in range(1, 5)]
        assert (last["info"].width, last["info"].height, last["info"].header.frame_id) == (
            854, 480, "head_camera/base")
        assert (last["spectator"].width, last["spectator"].height) == (960, 720)
        arm, hand = list(last["arm"].position), list(last["hand"].position)
        arm[0] -= .1  # lifts the left hand: joint1 swings it vertically
        hand[4] += .5  # index finger flexion
        arm_target.publish(trajectory(last["arm"].name, arm))
        hand_target.publish(trajectory(last["hand"].name, hand))
        graph.spin(.1)
        assert graph.wait(1., WAIT_S).success
        graph.spin(.1)
        assert abs(last["arm"].position[0] - arm[0]) < .03
        assert abs(last["hand"].position[4] - hand[4]) < .1
        assert graph.call(stop, Trigger.Request()).success
        image, _ = graph.frame("left_camera", timeout=WAIT_S)
        assert image.header.frame_id == "left_camera/base" and (image.width, image.height) == (848, 480)
        assert np.frombuffer(bytes(image.data), np.uint8).std() > 1
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2, WAIT_S).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before, WAIT_S)
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)
