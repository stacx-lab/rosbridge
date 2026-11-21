"""Actuator endpoints validate, replace and stop without ROS actions."""
import math
import pytest
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState, MultiDOFJointState
from std_msgs.msg import Float64, Header
from std_srvs.srv import Trigger
from geometry_msgs.msg import Quaternion, Transform, Twist, Vector3
from trajectory_msgs.msg import (JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)
from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph


def trajectory(positions, names=('slide', 'rotate'), **point):
    return JointTrajectory(joint_names=list(names), points=[
        JointTrajectoryPoint(positions=positions, time_from_start=Duration(sec=1), **point)])


def endpoints(graph, device, command, message):
    states = []
    graph.node.create_subscription(JointState, f'{device}/joint_state', states.append, STREAM_QOS)
    publisher = graph.node.create_publisher(message, f'{device}/{command}', 10)
    stop = graph.node.create_client(Trigger, f'{device}/stop')
    graph.until(lambda: stop.service_is_ready() and publisher.get_subscription_count() and states)
    return states, publisher, stop


def send(graph, publisher, message):
    # Commands are admitted asynchronously; let the graph receive one before waiting.
    publisher.publish(message)
    graph.spin(.2)


def test_arm_target_validates_replaces_and_stops(tmp_path):
    graph = Graph(tmp_path)
    try:
        states, publisher, stop = endpoints(graph, 'axes', 'joint_target', JointTrajectory)
        position = lambda: [dict(zip(states[-1].name, states[-1].position))[name] for name in ('slide', 'rotate')]
        send(graph, publisher, trajectory([.2, .1]))
        assert graph.wait(1.5).success
        graph.until(lambda: position() == pytest.approx([.2, .1]))
        # Invalid replacements are refused and keep the active reference; the next wait's reply names them.
        for invalid in (trajectory([math.nan, .2]), trajectory([.2, .1], names=('slide',)),
                        trajectory([.2, .1], velocities=[0., 0.]), trajectory([9., .1]),
                        JointTrajectory(joint_names=['slide', 'rotate'])):
            send(graph, publisher, invalid)
        result = graph.wait(.5)  # ends at once: the refusals await this reply
        assert not result.success and result.waited_s == 0.
        assert result.message.startswith('ended early by a refused command; refused since the previous wait reply: ')
        for reason in ('axes target has non-finite values', 'trajectory must name each of',
                       'only position trajectories are supported', 'slide=9.0000 outside',
                       'trajectory needs one time per point and at least one point'):
            assert f'/axes/joint_target: {reason}' in result.message
        graph.spin(.1)
        assert position() == pytest.approx([.2, .1])
        send(graph, publisher, trajectory([.3, .2]))
        assert graph.wait(.3).success
        assert graph.call(stop, Trigger.Request()).success
        assert graph.wait(1.5).success
        graph.spin(.1)
        held = position()
        assert .2 < held[0] < .3 and .1 < held[1] < .2
    finally:
        graph.close()


def poses(*targets, frame='reach/base', tip='tip', **point):
    """A tip trajectory of (x, y, quaternion xyzw) targets one second apart."""
    return MultiDOFJointTrajectory(header=Header(frame_id=frame), joint_names=[tip], points=[
        MultiDOFJointTrajectoryPoint(transforms=[Transform(translation=Vector3(x=x, y=y),
                                                           rotation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))],
                                     time_from_start=Duration(sec=i + 1), **point)
        for i, (x, y, q) in enumerate(targets)])


def test_pose_target_moves_the_tip_validates_and_stops(tmp_path):
    graph = Graph(tmp_path, backend='pose_toy')
    try:
        palms = []
        graph.node.create_subscription(MultiDOFJointState, 'reach/pose_state', palms.append, STREAM_QOS)
        states, publisher, stop = endpoints(graph, 'reach', 'pose_target', MultiDOFJointTrajectory)
        position = lambda: list(states[-1].position)
        assert graph.wait(.3).success
        graph.spin(.1)
        assert position() == pytest.approx([.5, -.5])  # uncommanded, the simulator holds the arm
        # pose_state reports the measured tip in the device base, stamped with joint_state.
        palm = palms[-1]
        assert (palm.header.frame_id, list(palm.joint_names), palm.header.stamp) == (
            'reach/base', ['tip'], states[-1].header.stamp)
        assert (palm.transforms[0].translation.x, palm.transforms[0].translation.y) == (.5, -.5)
        upright = (0., 0., 0., 1.)
        send(graph, publisher, poses((1., 0., upright), (1.5, .5, (0., 0., 0., 2.))))
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx([1., 0.])  # interpolated from the measured tip
        for invalid in (poses((0., 0., upright), frame='world'), poses((0., 0., upright), tip='palm'),
                        poses((0., 0., (0., 0., 0., 0.))), poses((math.nan, 0., upright)),
                        poses((0., 0., upright), velocities=[Twist()])):
            send(graph, publisher, invalid)
        result = graph.wait(1.)
        assert not result.success and result.message.startswith('ended early by a refused command; ')
        for reason in ('pose targets must be expressed in reach/base', 'pose targets must name the tip link tip once',
                       'reach tip pose needs a unit quaternion', 'reach tip pose has non-finite values',
                       'only pose trajectories are supported'):
            assert f'/reach/pose_target: {reason}' in result.message
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx([1.5, .5])  # the active trajectory went on
        send(graph, publisher, poses((-1., -1., upright)))
        assert graph.wait(.5).success
        assert graph.call(stop, Trigger.Request()).success
        graph.spin(.1)
        held = position()
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx(held) and -1. < held[0] < 1.5  # stop holds the measured tip
    finally:
        graph.close()


def test_an_arm_taking_joint_and_pose_targets_follows_the_latest_and_stops_on_its_joints(tmp_path):
    graph = Graph(tmp_path, backend='joint_pose_toy')
    try:
        states, poser, stop = endpoints(graph, 'reach', 'pose_target', MultiDOFJointTrajectory)
        joints = graph.node.create_publisher(JointTrajectory, 'reach/joint_target', 10)
        graph.until(joints.get_subscription_count)
        position = lambda: list(states[-1].position)
        upright = (0., 0., 0., 1.)
        send(graph, joints, trajectory([1., 0.], names=('x', 'y')))
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx([1., 0.])
        send(graph, poser, poses((1., 1., upright)))  # takes over from the measured tip
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx([1., 1.])
        send(graph, joints, trajectory([0., 0.], names=('x', 'y')))  # and back, from the measured joints
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx([0., 0.])
        send(graph, joints, trajectory([3., 0.], names=('x', 'y')))
        assert '/reach/joint_target: x=3.0000 outside [-2.0000, 2.0000]' in graph.wait(.1).message
        # A pose the simulator cannot execute stops the arm where it is; the next reply names why.
        send(graph, poser, poses((2., 0., upright)))
        result = graph.wait(1.)
        assert not result.success and result.waited_s < 1. and '/reach/pose_target: at t=' in result.message
        assert result.message.endswith('the tip pose is beyond the 1.5 m reach')
        graph.spin(.1)
        held = position()
        assert 0. < held[0] <= 1.5 and held[1] == pytest.approx(0.)
        assert graph.wait(.5).message == 'ok'
        graph.spin(.1)
        assert position() == pytest.approx(held)
        send(graph, poser, poses((1., -1., upright)))
        assert graph.wait(.5).success
        assert graph.call(stop, Trigger.Request()).success
        graph.spin(.1)
        stopped = position()
        assert graph.wait(1.).success
        graph.spin(.1)
        assert position() == pytest.approx(stopped)  # a stop holds the measured joints
    finally:
        graph.close()


def test_gripper_target_persists_and_stop_keeps_it(tmp_path):
    graph = Graph(tmp_path)
    try:
        states, publisher, stop = endpoints(graph, 'finger', 'target', Float64)
        send(graph, publisher, Float64(data=-1.))
        assert graph.wait(.1).success
        graph.until(lambda: states[-1].position[0] == -1.)
        assert graph.wait(2.).success
        graph.spin(.1)
        assert states[-1].position[0] == -1.
        for value in (math.nan, math.inf, -2., 2.):
            send(graph, publisher, Float64(data=value))
        result = graph.wait(.1)
        assert not result.success and '/finger/target: finger target has non-finite values (2 commands)' in result.message
        assert graph.wait(0.).message == 'ok'  # named once
        graph.spin(.1)
        assert states[-1].position[0] == -1.
        stopped = graph.call(stop, Trigger.Request())
        assert stopped.success and stopped.message == 'keeping the grip target'
        assert graph.wait(.5).success
        graph.spin(.1)
        assert states[-1].position[0] == -1.  # stop keeps the grip target
    finally:
        graph.close()


@pytest.mark.parametrize('mode', ['stepped', 'realtime'])
def test_terminal_episode_rejects_commands_and_stays_stoppable(tmp_path, mode):
    graph = Graph(tmp_path, mode=mode, max_steps=10)
    try:
        states, publisher, stop = endpoints(graph, 'finger', 'target', Float64)
        # Realtime may already have terminated while the clients discovered each other.
        result = graph.wait(3.)
        assert not result.success
        assert 'episode' in result.message or 'terminal' in result.message
        send(graph, publisher, Float64(data=1.))
        graph.spin(.3)
        assert graph.owner.poll() is None
        assert states[-1].position[0] == 0.
        assert graph.call(stop, Trigger.Request()).success
    finally:
        graph.close()
