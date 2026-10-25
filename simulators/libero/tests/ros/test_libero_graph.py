"""The Kinex-facing LIBERO graph: arm and gripper targets, camera frames and world waits."""
import pytest
from builtin_interfaces.msg import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds


@pytest.mark.parametrize('mode', ['stepped', 'realtime'])
def test_libero_graph_serves_the_kinex_devices(tmp_path, mode):
    graph = Graph(tmp_path, backend='libero', mode=mode)
    node, last = graph.node, {}
    try:
        for kind, topic, key, qos in (
            (JointState, 'arm/joint_state', 'arm', STREAM_QOS),
            (JointState, 'gripper/joint_state', 'gripper', STREAM_QOS),
            (Image, 'camera_agentview/image', 'image', qos_profile_sensor_data),
        ):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), qos)
        joint_target = node.create_publisher(JointTrajectory, 'arm/joint_target', 1)
        gripper_target = node.create_publisher(Float64, 'gripper/target', 1)
        stop = node.create_client(Trigger, 'arm/stop')
        graph.until(lambda: joint_target.get_subscription_count() and gripper_target.get_subscription_count()
                    and stop.service_is_ready())
        assert graph.wait(.2).success
        graph.until(lambda: all(k in last for k in ('arm', 'gripper', 'image')))
        assert last['arm'].name == [f'robot0_joint{i}' for i in range(1, 8)]
        assert last['gripper'].name == ['finger_joint1']
        assert (last['image'].width, last['image'].height, last['image'].header.frame_id) == (
            256, 256, 'camera_agentview/base')
        target = list(last['arm'].position)
        target[0] += .08
        joint_target.publish(JointTrajectory(joint_names=list(last['arm'].name), points=[
            JointTrajectoryPoint(positions=target, time_from_start=Duration(nanosec=500_000_000))]))
        graph.spin(.1)
        assert graph.wait(1.).success
        graph.spin(.1)
        assert abs(last['arm'].position[0] - target[0]) < .04
        assert graph.call(stop, Trigger.Request()).success
        for goal, reached in ((0., lambda q: q < .01), (.04, lambda q: q > .03)):
            gripper_target.publish(Float64(data=goal))
            graph.spin(.1)
            assert graph.wait(1.5).success
            graph.spin(.1)
            assert reached(last['gripper'].position[0]), last['gripper'].position[0]
        image, _ = graph.frame('camera_wrist')
        assert image.header.frame_id == 'camera_wrist/base' and len(image.data) == 256 * 256 * 3
        if mode == 'stepped':  # a paused world still yields its current frame
            assert seconds(image.header.stamp) == seconds(graph.last['clock'].clock)
        assert graph.reset().success
    finally:
        graph.close()
