"""The Kinex-facing RoboCasa graph: base velocity and odometry, torso, arm, gripper and RGB-D."""
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds


def test_robocasa_graph_serves_the_mobile_kinex_devices(tmp_path):
    graph = Graph(tmp_path, backend='robocasa', mode='stepped', startup_s=120.)
    node, last = graph.node, {}
    try:
        for kind, topic, key in ((JointState, 'arm/joint_state', 'arm'),
                                 (JointState, 'torso/joint_state', 'torso'),
                                 (JointState, 'gripper/joint_state', 'gripper'),
                                 (Odometry, 'base/odometry', 'base')):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        torso = node.create_publisher(JointTrajectory, 'torso/joint_target', 1)
        gripper = node.create_publisher(Float64, 'gripper/target', 1)
        velocity = node.create_publisher(Twist, 'base/velocity', 1)
        stop = node.create_client(Trigger, 'base/stop')
        graph.until(lambda: torso.get_subscription_count() and gripper.get_subscription_count()
                    and velocity.get_subscription_count() and stop.service_is_ready())
        assert graph.wait(.1).success
        graph.until(lambda: all(k in last for k in ('arm', 'torso', 'gripper', 'base')))
        assert last['arm'].name == [f'robot0_joint{i}' for i in range(1, 8)]
        assert last['torso'].name == ['mobilebase0_joint_torso_height']
        assert last['gripper'].name == ['gripper']
        assert (last['base'].header.frame_id, last['base'].child_frame_id) == ('base/odom', 'base/base')

        def moved(duration):
            graph.spin(.1)
            assert graph.wait(duration).success
            graph.spin(.1)

        torso.publish(JointTrajectory(joint_names=['mobilebase0_joint_torso_height'], points=[
            JointTrajectoryPoint(positions=[.15], time_from_start=Duration(nanosec=500_000_000))]))
        gripper.publish(Float64(data=1.))
        moved(1.)
        assert abs(last['torso'].position[0] - .15) < .02 and last['gripper'].position[0] == 1.
        # A twist drives the base until 0.5 s of world time after the last one.
        twist = Twist()
        twist.angular.z = .5
        velocity.publish(twist)
        moved(.3)
        assert last['base'].twist.twist.angular.z > .3
        moved(1.)
        yaw = last['base'].pose.pose.orientation.z
        assert abs(last['base'].twist.twist.angular.z) < .05 and yaw > .05
        velocity.publish(twist)
        moved(.2)
        assert graph.call(stop, Trigger.Request()).success
        moved(.3)
        assert abs(last['base'].twist.twist.angular.z) < .05
        twist.linear.z = 1.  # undeclared axes are rejected, so the base stays at rest
        velocity.publish(twist)
        moved(.3)
        assert abs(last['base'].twist.twist.angular.z) < .05
        image, depth = graph.frame('camera_wrist', depth=True)  # the paused world's current frame
        assert image.header.frame_id == 'camera_wrist/base' and depth.encoding == '16UC1'
        assert len(image.data) == 256 * 256 * 3
        assert seconds(image.header.stamp) == seconds(graph.last['clock'].clock)
        assert graph.reset(120.).success
    finally:
        graph.close()
