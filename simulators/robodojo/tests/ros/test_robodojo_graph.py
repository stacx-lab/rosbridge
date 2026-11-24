"""The RoboDojo graph: a Python 3.12 world node proxying Isaac Sim 5.1 in a worker."""
import numpy as np
import psutil
import pytest
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Transform, Vector3
from rclpy.action import ActionClient
from sensor_msgs.msg import CameraInfo, Image, JointState, MultiDOFJointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import (JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, ended, execute, seconds

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
def test_robodojo_graph_serves_its_devices_from_a_native_worker(tmp_path, mode):
    # The depth configuration serves everything the standard devices do, and registered depth; one Isaac boot.
    graph = Graph(tmp_path, backend="robodojo", mode=mode, spectators=True, startup_s=STARTUP_S,
                  configuration="depth")
    node, last = graph.node, {}
    try:
        native = workers(graph)
        assert len(native) == 1 and "robodojo-native" in native[0].exe()
        for kind, topic, key in ((JointState, "left_arm/joint_state", "arm"),
                                 (JointState, "right_arm/joint_state", "right"),
                                 (JointState, "left_gripper/joint_state", "gripper"),
                                 (CameraInfo, "head_camera/camera_info", "info"),
                                 (MultiDOFJointState, "left_arm/pose_state", "flange"),
                                 (Image, "world/spectator/third_person/image_raw", "spectator")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        joint_target = node.create_publisher(JointTrajectory, "left_arm/joint_target", 1)
        gripper_target = node.create_publisher(Float64, "left_gripper/target", 1)
        stop = node.create_client(Trigger, "left_arm/stop")
        graph.until(lambda: joint_target.get_subscription_count()
                    and gripper_target.get_subscription_count() and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ("arm", "right", "gripper", "flange", "info", "spectator")))
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
        assert (last["flange"].header.frame_id, list(last["flange"].joint_names)) == ("left_arm/base", ["flange"])
        assert graph.call(stop, Trigger.Request()).success
        opened = last["gripper"].position[0]
        gripper_target.publish(Float64(data=0.))
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert last["gripper"].position[0] < opened - .5
        image, depth = graph.frame("left_camera", depth=True, timeout=60.)
        assert image.header.frame_id == "left_camera/base" and (image.width, image.height) == (640, 480)
        assert np.frombuffer(bytes(image.data), np.uint8).std() > 1
        # Registered depth: same grid, stamp and optical frame; 16UC1 millimetres.
        assert (depth.encoding, depth.width, depth.height, depth.step) == ("16UC1", 640, 480, 1280)
        assert depth.header == image.header
        millimetres = np.frombuffer(bytes(depth.data), np.uint16)
        assert (millimetres > 0).mean() > .99 and 50 < np.median(millimetres) < 2000
        before = seconds(last["spectator"].header.stamp)
        assert graph.wait(.2).success
        graph.until(lambda: seconds(last["spectator"].header.stamp) > before, 60.)
    finally:
        graph.close()
    assert not any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in native)


def test_robodojo_arms_take_move_goals_in_the_move_configuration(tmp_path):
    graph = Graph(tmp_path, backend="robodojo", mode="stepped", startup_s=STARTUP_S, configuration="move")
    node, last = graph.node, {}
    try:
        for key, arm in (("flange", "left_arm"), ("right_flange", "right_arm")):
            node.create_subscription(MultiDOFJointState, f"{arm}/pose_state", lambda msg, k=key: last.update({k: msg}),
                                     STREAM_QOS)
        assert not node.count_subscribers("left_arm/joint_target")  # move goals are the arms' only command
        # Both arms at once by move goals: each flange lifts at the arm's pace, and one wait ends with both.
        moves = [ActionClient(node, FollowJointTrajectory, f"{arm}/move") for arm in ("left_arm", "right_arm")]
        graph.until(lambda: all(move.server_is_ready() for move in moves) and "right_flange" in last)
        graph.spin(.1)

        def raised(key, dz):
            pose = last[key].transforms[0]
            lifted = Transform(translation=Vector3(x=pose.translation.x, y=pose.translation.y,
                                                   z=pose.translation.z + dz), rotation=pose.rotation)
            return lifted.translation.z, FollowJointTrajectory.Goal(multi_dof_trajectory=MultiDOFJointTrajectory(
                joint_names=["flange"], points=[MultiDOFJointTrajectoryPoint(transforms=[lifted])]))

        heights, goals = zip(*(raised(key, dz) for key, dz in (("flange", .06), ("right_flange", .04))))
        handles = [execute(graph, move, goal) for move, goal in zip(moves, goals)]
        both = graph.wait(0., timeout=120., goals=handles, goal_timeout=3.)
        assert both.success and list(both.finished) == [True, True] and both.waited_s < .5
        for handle in handles:
            status, code, text = ended(graph, handle)
            assert (status, code) == (GoalStatus.STATUS_SUCCEEDED, 0) and text.startswith("reached: "), text
        graph.spin(.1)
        for key, height in zip(("flange", "right_flange"), heights):
            assert abs(last[key].transforms[0].translation.z - height) < .03
        # A goal beyond reach is aborted where upstream's IK has no next step, with the reason.
        beyond = execute(graph, moves[0], raised("flange", 3.)[1])
        assert list(graph.wait(0., timeout=120., goals=[beyond]).finished) == [True]
        status, code, text = ended(graph, beyond)
        assert (status, code) == (GoalStatus.STATUS_ABORTED, -1) and " IK " in text, text
    finally:
        graph.close()
