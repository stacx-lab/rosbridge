"""Actuator endpoints validate, replace and stop without ROS actions."""
import math
import pytest
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
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
        # Invalid replacements are ignored and keep the active reference.
        for invalid in (trajectory([math.nan, .2]), trajectory([.2, .1], names=('slide',)),
                        trajectory([.2, .1], velocities=[0., 0.]), trajectory([9., .1])):
            send(graph, publisher, invalid)
        assert graph.wait(.5).success
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


def test_gripper_target_persists_until_stop_holds_it(tmp_path):
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
        assert graph.wait(.1).success
        graph.spin(.1)
        assert states[-1].position[0] == -1.
        assert graph.call(stop, Trigger.Request()).success
        assert graph.wait(.5).success
        graph.spin(.1)
        assert states[-1].position[0] == -1.  # stop holds the measured position
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
