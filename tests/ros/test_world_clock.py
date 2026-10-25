"""World services and device endpoints against a synthetic backend."""
import time

import pytest
from kinex_interfaces.srv import Wait
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from rosbridge.clients import WorldClockClient
from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds


@pytest.fixture
def stepped(tmp_path):
    graph = Graph(tmp_path)
    yield graph
    graph.close()


def ticks(graph):
    return round(seconds(graph.last["clock"].clock) / .1)


def test_stepped_pause_validation_and_rounding(stepped):
    graph = stepped
    assert graph.reset().success
    result = graph.wait(0.)
    assert result.success and result.waited_s == 0 and result.time_s == 0
    pub = graph.node.create_publisher(Float64, "finger/target", 1)
    graph.until(lambda: pub.get_subscription_count() > 0)
    pub.publish(Float64(data=-.5))
    graph.spin(.3)
    assert ticks(graph) == 0  # commands do not step the world
    for duration in (-1., float("nan"), float("inf")):
        assert not graph.wait(duration).success
    result = graph.wait(.21)
    assert result.success and result.waited_s == pytest.approx(.3)
    graph.until(lambda: ticks(graph) == 3)
    graph.spin(.3)
    assert ticks(graph) == 3 and seconds(graph.last["clock"].clock) == pytest.approx(.3)
    assert WorldClockClient(graph.node).wait(.1, timeout_s=5.).success
    assert graph.node.count_publishers("/clock") == 1


def test_waits_are_sequential_and_reset_or_stop_ends_them(stepped):
    graph = stepped
    assert set(Wait.Request.get_fields_and_field_types()) == {"duration_s"}
    pending = graph.wait_client.call_async(Wait.Request(duration_s=1e8))
    graph.spin(.2)
    assert "sequential" in graph.wait(0.).message
    assert graph.reset().success
    graph.until(pending.done)
    assert not pending.result().success and "reset" in pending.result().message
    graph.spin(.1)
    assert ticks(graph) == 0
    pending = graph.wait_client.call_async(Wait.Request(duration_s=1e8))
    graph.spin(.2)
    assert graph.call(graph.stop_client, Trigger.Request()).success
    graph.until(pending.done)
    assert not pending.result().success and "cancelled" in pending.result().message
    graph.spin(.1)
    stopped = ticks(graph)
    graph.spin(.2)
    assert ticks(graph) == stopped  # nothing is left advancing the stepped world
    assert graph.wait(0.).success
    assert graph.call(graph.stop_client, Trigger.Request()).success


def test_realtime_runs_autonomously(tmp_path):
    graph = Graph(tmp_path, mode="realtime")
    try:
        assert graph.reset().success
        graph.spin(.3)
        before = ticks(graph)
        graph.spin(.3)
        assert ticks(graph) > before
        start = time.monotonic()
        result = graph.wait(.5)
        assert result.success and result.waited_s == pytest.approx(.5)
        assert time.monotonic() - start < 2.
    finally:
        graph.close()


def test_terminal_tick_and_runtime_unavailable(tmp_path):
    graph = Graph(tmp_path, max_steps=2)
    try:
        result = graph.wait(.2)
        assert result.success and result.waited_s == pytest.approx(.2)
        assert not graph.wait(.1).success
        assert graph.reset().success
        result = graph.wait(.3)
        assert not result.success and result.waited_s == pytest.approx(.2)
        graph.stop(graph.owner)
        graph.until(lambda: not graph.wait_client.service_is_ready(), 15.)
        with pytest.raises(TimeoutError, match="unavailable"):
            WorldClockClient(graph.node).wait(.1, timeout_s=.2)
    finally:
        graph.close()


def test_wall_timeout_leaves_the_admitted_wait_to_stop(stepped):
    graph = stepped
    with pytest.raises(TimeoutError, match="remains pending"):
        WorldClockClient(graph.node).wait(100000., timeout_s=.2)
    graph.spin(.1)
    before = ticks(graph)
    graph.spin(.1)
    assert ticks(graph) > before
    assert graph.call(graph.stop_client, Trigger.Request()).success
    assert graph.reset().success


def test_runtime_loss_interrupts_an_inflight_wait(tmp_path):
    graph = Graph(tmp_path, mode="realtime")
    try:
        future = graph.wait_client.call_async(Wait.Request(duration_s=1000.))
        graph.spin(.2)
        assert not future.done()
        graph.stop(graph.owner)
        # Graceful stop resolves waits before DDS teardown when delivery is possible.
        # A destroyed participant cannot promise a response; client wall timeouts bound it.
        graph.until(lambda: not graph.wait_client.service_is_ready())
        if future.done():
            assert not future.result().success
    finally:
        graph.close()


def test_device_endpoints_share_the_world_node(stepped):
    graph = stepped
    states = []
    graph.node.create_subscription(JointState, "finger/joint_state", states.append, STREAM_QOS)
    pub = graph.node.create_publisher(Float64, "finger/target", 1)
    graph.until(lambda: pub.get_subscription_count() > 0)
    subscriptions = graph.node.get_subscriptions_info_by_topic("/finger/target")
    assert {s.node_name for s in subscriptions} == {"world"}
    pub.publish(Float64(data=-.5))
    graph.spin(.1)
    assert graph.wait(.1).success
    graph.until(lambda: bool(states) and list(states[-1].position) == [-.5])
    assert states[-1].name == ["finger"]


def test_no_task_or_judge_topics_and_reset_payload(stepped):
    graph = stepped
    graph.spin(.2)
    topics = {name for name, _ in graph.node.get_topic_names_and_types()}
    assert not {"/task/status", "/task/instruction", "/joint_states", "/action"} & topics
    result = graph.reset()
    assert result.success and "toy" not in result.message
