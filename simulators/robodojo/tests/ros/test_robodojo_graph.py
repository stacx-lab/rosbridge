"""The Kinex-facing RoboDojo graph: a Python 3.12 world node proxying Isaac Sim 5.1 in a worker."""
import numpy as np
import psutil
import pytest
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds

# Isaac Sim boots, compiles shaders and loads the layout before serving.
STARTUP_S = 1200.


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
def test_robodojo_graph_serves_the_kinex_devices_from_a_native_worker(tmp_path, mode):
    graph = Graph(tmp_path, backend="robodojo", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "robodojo-native" in native[0].exe()
        for kind, topic, key in ((JointState, "left_arm/joint_state", "arm"),
                                 (JointState, "right_arm/joint_state", "right"),
                                 (JointState, "left_gripper/joint_state", "gripper"),
                                 (CameraInfo, "head_camera/camera_info", "info"),
                                 (Image, "world/spectator/third_person/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        joint_target = node.create_publisher(JointTrajectory, "left_arm/joint_target", 1)
        gripper_target = node.create_publisher(Float64, "left_gripper/target", 1)
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: joint_target.get_subscription_count() and gripper_target.get_subscription_count()
                    and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("arm", "right", "gripper", "info", "spectator")))
        # Both arms keep the native joint names; devices disambiguate them.
        assert last["arm"].name == last["right"].name == [f"joint{i}" for i in range(1, 7)]
        assert last["gripper"].name == ["left_gripper"]
        assert (last["info"].width, last["info"].height, last["info"].header.frame_id) == (
            640, 480, "head_camera/base")
        assert last["info"].k[2] == 319.5 and (last["spectator"].width, last["spectator"].height) == (960, 720)
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
        assert graph.wait(1.).success
        graph.spin(.1)
        assert last["gripper"].position[0] < opened - .5
        image, _ = graph.frame("left_camera", timeout=60.)
        assert image.header.frame_id == "left_camera/base" and (image.width, image.height) == (640, 480)
        assert np.frombuffer(bytes(image.data), np.uint8).std() > 1
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before, 60.)
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)
