"""The Kinex-facing BEHAVIOR G1 graph: a world node proxying OmniGibson and GR00T in a worker."""
import psutil
import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph

STARTUP_S, RESET_S = 1800., 600.


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
def test_behavior_g1_graph_walks_and_moves_the_hands(tmp_path, mode):
    graph = Graph(tmp_path, backend="behavior-g1", mode=mode, spectators=True, startup_s=STARTUP_S)
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "behavior-g1-native" in native[0].exe()
        for kind, topic, key in ((JointState, "torso/joint_state", "torso"),
                                 (JointState, "right_hand/joint_state", "hand"),
                                 (Odometry, "base/odometry", "base"),
                                 (CameraInfo, "head_camera/camera_info", "head")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        hand_target = node.create_publisher(JointTrajectory, "right_hand/joint_target", 1)
        velocity = node.create_publisher(Twist, "base/velocity", 1)
        graph.until(lambda: hand_target.get_subscription_count() and velocity.get_subscription_count())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("torso", "hand", "base", "head")))
        assert last["torso"].name == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        assert len(last["hand"].name) == 22 and last["hand"].name[0] == "right_thumb_CMC_FE"
        assert (last["head"].width, last["head"].height, last["head"].k[2]) == (720, 720, 359.5)
        twist = Twist()
        twist.linear.x = .3
        for _ in range(8):  # the base stops 0.5 s after the last twist
            velocity.publish(twist)
            graph.spin(.05)
            assert graph.wait(.25).success
        graph.spin(.1)
        assert last["base"].pose.pose.position.x > .2
        goal = list(last["hand"].position)
        goal[5:9] = .8, 0., .8, .6  # index finger
        hand_target.publish(JointTrajectory(joint_names=list(last["hand"].name), points=[
            JointTrajectoryPoint(positions=goal, time_from_start=Duration(nanosec=500_000_000))]))
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert max(abs(a - b) for a, b in zip(last["hand"].position, goal)) < .1
        image, depth = graph.frame("head_camera", depth=True, timeout=120.)
        assert (image.width, image.height, depth.encoding) == (720, 720, "16UC1")
        assert graph.reset(RESET_S).success
        graph.spin(.5)
        assert abs(last["base"].pose.pose.position.x) < 1e-3
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)
