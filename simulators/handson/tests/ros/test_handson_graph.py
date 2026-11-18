"""The HandsOn graph: a Python 3.12 world node proxying Isaac Lab in a worker."""
import json

import numpy as np
import psutil
import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, Transform, Vector3
from custom_interfaces.srv import Json
from sensor_msgs.msg import CameraInfo, Image, JointState, MultiDOFJointState
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from trajectory_msgs.msg import (JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)

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


def pose(state):
    """A pose_state's palm [x, y, z, qx, qy, qz, qw]."""
    t, q = state.transforms[0].translation, state.transforms[0].rotation
    return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]


def palm(target):
    """A left palm pose target [x, y, z, qx, qy, qz, qw] in left_arm/base, reached in 0.5 s."""
    x, y, z, qx, qy, qz, qw = target
    return MultiDOFJointTrajectory(header=Header(frame_id="left_arm/base"), joint_names=["palm"], points=[
        MultiDOFJointTrajectoryPoint(transforms=[Transform(translation=Vector3(x=x, y=y, z=z),
                                                           rotation=Quaternion(x=qx, y=qy, z=qz, w=qw))],
                                     time_from_start=Duration(nanosec=500_000_000))])


@pytest.mark.parametrize("mode", ["stepped"])
def test_handson_graph_serves_its_devices_from_a_native_worker(tmp_path, monkeypatch, mode):
    # The graph is stopped by signal, before the worker removes its episode's files; pytest prunes these.
    monkeypatch.setenv("ROSBRIDGE_RUN_DIR", str(tmp_path))
    graph = Graph(tmp_path, backend="handson", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "handson-native" in native[0].exe()
        for kind, topic, key in ((JointState, "left_arm/joint_state", "arm"),
                                 (MultiDOFJointState, "left_arm/pose_state", "palm"),
                                 (JointState, "right_hand/joint_state", "hand"),
                                 (CameraInfo, "head_camera/camera_info", "info"),
                                 (Image, "world/spectator/third_person/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        arm_target = node.create_publisher(MultiDOFJointTrajectory, "left_arm/pose_target", 1)
        hand_target = node.create_publisher(JointTrajectory, "right_hand/joint_target", 1)
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: arm_target.get_subscription_count() and hand_target.get_subscription_count()
                    and stop.service_is_ready())
        assert graph.wait(.2, WAIT_S).success
        graph.until(lambda: all(k in last for k in ("arm", "palm", "hand", "info", "spectator")))
        assert last["arm"].name == [f"joint{i}_left" for i in range(1, 8)]
        assert last["hand"].name[:4] == [f"right_finger1_joint{j}" for j in range(1, 5)]
        assert (last["info"].width, last["info"].height, last["info"].header.frame_id) == (
            854, 480, "head_camera/base")
        # The head camera's lens, in ROS pixel centers.
        assert last["info"].distortion_model == "plumb_bob" and last["info"].d[0] == pytest.approx(.00646656)
        assert last["info"].k[2] == pytest.approx(431.267, abs=1e-3)
        # The world node serves that lens: its descriptor, and rays through it.
        geometry, unproject = (node.create_client(Json, f"head_camera/{name}") for name in ("geometry", "unproject"))
        graph.until(lambda: geometry.service_is_ready() and unproject.service_is_ready())
        stream = json.loads(graph.call(geometry, Json.Request(request="{}")).response)["streams"]["color"]
        assert stream["frame"] == "head_camera/base" and stream["distortion"]["model"] == "opencv_brown_conrady"
        assert stream["distortion"]["k1"] == pytest.approx(last["info"].d[0])
        assert stream["projection"]["cx"] == pytest.approx(last["info"].k[2])
        corner = json.loads(graph.call(unproject, Json.Request(request=json.dumps(
            {"stream": "color", "calibration_id": stream["calibration_id"], "pixels": [[0., 0.]]}))).response)["rays"][0]
        pinhole = [-stream["projection"]["cx"] / stream["projection"]["fx"],
                   -stream["projection"]["cy"] / stream["projection"]["fy"]]
        assert 1. < np.hypot(*np.subtract(corner[:2], pinhole)) * stream["projection"]["fx"] < 10.  # pixels
        assert not any(name.startswith("/left_camera/") for name, _ in node.get_service_names_and_types())
        assert (last["spectator"].width, last["spectator"].height) == (960, 720)
        assert last["palm"].header.frame_id == "left_arm/base" and list(last["palm"].joint_names) == ["palm"]
        goal, hand = pose(last["palm"]), list(last["hand"].position)
        goal[1] -= .05  # lifts the left palm 5 cm: world up is -y in left_arm/base
        hand[4] += .5  # index finger flexion
        arm_target.publish(palm(goal))  # upstream's own IK moves the arm
        hand_target.publish(trajectory(last["hand"].name, hand))
        graph.spin(.1)
        assert graph.wait(1., WAIT_S).success
        graph.spin(.1)
        assert np.linalg.norm(np.subtract(pose(last["palm"])[:3], goal[:3])) < .01
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
