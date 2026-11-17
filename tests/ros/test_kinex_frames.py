"""Frame ownership, measured URDF links and reset behavior in a native ROS graph."""
import asyncio
import math
import os
import sys
import uuid

import pytest

if os.environ.get("KINEX_PYTHON"):
    sys.path.insert(0, os.environ["KINEX_PYTHON"])
pytest.importorskip("kinex.node")
from kinex.communication import ambient, communication, qos
from kinex.host.device import DeviceHost
from kinex_interfaces import srv as kinex_srv

# Kinex's frame host serves its LookupTransform; rosbridge's own copy of kinex_interfaces
# has only Wait. Source a Kinex workspace's .kinex/python/ros/install to run these.
pytestmark = pytest.mark.skipif(not hasattr(kinex_srv, "LookupTransform"),
                                reason="needs Kinex's kinex_interfaces with LookupTransform")


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.01)


def operation(endpoint, kind, transport="service", direction="input"):
    return {"endpoint": endpoint, "type": kind, "transport": transport, "direction": direction,
            "qos": {"depth": 1, "durability": "transient_local"}}


def test_urdf_links_and_kinex_frames_without_simulator(tmp_path):
    from builtin_interfaces.msg import Time as Stamp
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from tf2_ros import StaticTransformBroadcaster, TransformException
    from kinex.common.frames import transform
    from kinex.host.frame_graph import SERVICE_HISTORY_S
    from kinex.lib.frames import lookup_transform

    arm_entry = tmp_path / "links.py"
    arm_entry.write_text("from pathlib import Path\nfrom kinex.node import LinkPublisher\n"
                         "def create(profile, context):\n    return LinkPublisher(profile, context, Path(__file__).parent)\n")
    (tmp_path / "arm.urdf").write_text('''<robot name="arm"><link name="base"/><link name="rotor"/><link name="tip"/>
<joint name="hinge" type="revolute"><parent link="base"/><child link="rotor"/>
<axis xyz="0 0 1"/><limit lower="-3.141592653589793" upper="3.141592653589793"/></joint>
<joint name="end" type="fixed"><parent link="rotor"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>''')
    model = {"format": "urdf", "path": "arm.urdf", "base_link": "base", "tip_links": ["tip"]}

    async def run():
        name = "arm_" + uuid.uuid4().hex
        prefix = "/" + name
        state_op = operation(prefix + "/joints", "sensor_msgs/msg/JointState", "topic", "output")
        clock_op = operation(prefix + "/time", "rosgraph_msgs/msg/Clock", "topic", "output")
        runtime = {"domain_id": 202, "ready_timeout_s": 5}
        arm = {"name": name, "type": "joint", "runtime": {**runtime, "nodes": [{"entry": str(arm_entry)}]},
               "operations": {"joint_state": state_op}, "model": model, "home": None, "limits": "strict",
               "frames": {"base": {"parent": "world:base", "translation_m": [2, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
                          "tool": {"parent": name + ":tip", "translation_m": [0, 0, .2], "rotation_xyzw": [0, 0, 0, 1]}}}
        hosts = []
        async with communication({}, domain_id=202) as backend:
            clock = backend.node.create_publisher(Clock, clock_op["endpoint"], qos(clock_op))
            states = backend.node.create_publisher(JointState, state_op["endpoint"], qos(state_op))
            clock.publish(Clock(clock=Stamp(sec=10)))
            sample = JointState(name=["hinge"], position=[math.pi / 2])
            sample.header.stamp = Stamp(sec=10)
            states.publish(sample)
            try:
                # Load the frame graph after the arm, while no clock or joint
                # samples are advancing. Durable TF must preserve measured time.
                arm_host, graph_host = DeviceHost(), DeviceHost()
                hosts.append(arm_host)
                await arm_host.load(arm, clock_op)
                hosts.append(graph_host)
                await graph_host.run("$frames", {"clock": clock_op, "endpoint": prefix + "/lookup", "domain_id": 202})
                graph = graph_host.io.extra_nodes[0]
                await until(lambda: graph.stamp == 10_000_000_000 and
                            graph.buffer.can_transform("world/base", name + "/tool", Time()))
                relation = await graph_host.io._request(graph_host.io.frame_service, { "target_frame": "world/base", "source_frame": name + "/tool"})
                p = relation.transform.transform.translation
                assert [p.x, p.y, p.z] == pytest.approx([2, 1, .2], abs=1e-9)
                assert relation.transform.header.stamp.sec == 10
                assert graph.buffer.history_s == SERVICE_HISTORY_S
                # A failed lookup waits its whole timeout, then names the cause.
                with pytest.raises(RuntimeError, match=f"timed out after 0.03 s: missing_{name} is not on TF, nor is any frame of missing_{name}"):
                    await graph_host.io._request(graph_host.io.frame_service, { "target_frame": "world/base", "source_frame": "missing_" + name, "wait_s": .03}, timeout_s=1)
                with pytest.raises(RuntimeError, match=f"{name}/nope is not on TF; {name} has {name}/base, "):
                    await graph_host.io._request(graph_host.io.frame_service, { "target_frame": "world/base", "source_frame": name + "/nope"})
                # A frame that appears late within the wait still resolves.
                late = asyncio.create_task(graph_host.io._request(graph_host.io.frame_service, {
                    "target_frame": "world/base", "source_frame": name + "/late", "wait_s": 3.}, timeout_s=4))
                await asyncio.sleep(1.5)
                StaticTransformBroadcaster(backend.node).sendTransform(transform(name + "/base", name + "/late", [0, 0, 1, 0, 0, 0, 1]))
                p = (await late).transform.transform.translation
                assert [p.x, p.y, p.z] == pytest.approx([2, 0, 1], abs=1e-9)
                # Live frames retain measurements outside command limits.
                sample.position = [3.5]
                sample.header.stamp = Stamp(sec=11)
                clock.publish(Clock(clock=Stamp(sec=11)))
                states.publish(sample)
                await until(lambda: graph.buffer.can_transform("world/base", name + "/tip", Time(seconds=11)))
                measured = graph.buffer.lookup_transform("world/base", name + "/tip", Time(seconds=11))
                assert measured.transform.translation.x == pytest.approx(2 + math.cos(3.5))
                # Reset invalidates history, retains mounts and admits the reset state.
                with ambient(graph_host.io):  # the task keeps the scope it starts in
                    pending = asyncio.create_task(lookup_transform("world/base", name + "/tip", Stamp(sec=12), 2.))
                await until(lambda: bool(graph.pending))
                clock.publish(Clock(clock=Stamp(sec=1)))
                with pytest.raises(RuntimeError, match="world clock reset"):
                    await pending
                await until(lambda: graph.stamp == 1_000_000_000)
                with pytest.raises(TransformException):
                    graph.buffer.lookup_transform("world/base", name + "/tip", Time(seconds=11))
                sample.position, sample.header.stamp = [0.], Stamp(sec=1)
                states.publish(sample)
                await until(lambda: graph.buffer.can_transform("world/base", name + "/tool", Time(seconds=1)))
                measured = graph.buffer.lookup_transform("world/base", name + "/tool", Time(seconds=1))
                assert measured.transform.translation.x == pytest.approx(3.)
                assert measured.transform.translation.z == pytest.approx(.2)
                assert not backend.node.count_publishers("/clock")  # Test clock has its own endpoint.
            finally:
                for host in reversed(hosts):
                    await host.run("unload", {})
    asyncio.run(run())


def test_delayed_frames_cannot_restore_pre_reset_history():
    from builtin_interfaces.msg import Time as Stamp
    from rclpy.time import Time
    from tf2_ros import TransformException
    from kinex.host.frame_graph import FrameBuffer
    from kinex.common.frames import transform

    buffer = FrameBuffer()
    buffer.set_transform_static(transform("world/base", "arm/base", [2, 0, 0, 0, 0, 0, 1]), "kinex")
    buffer.observe_clock(10_000_000_000)
    old = transform("arm/base", "arm/flange", [1, 0, 0, 0, 0, 0, 1], Stamp(sec=10))
    buffer.set_transform(old, "kinex")
    buffer.observe_clock(1_000_000_000)
    buffer.set_transform(old, "delayed")
    with pytest.raises(TransformException):
        buffer.lookup_transform("world/base", "arm/flange", Time())
    buffer.set_transform(transform("arm/base", "arm/flange", [0, 1, 0, 0, 0, 0, 1], Stamp(sec=1)), "kinex")
    buffer.observe_clock(10_000_000_000)
    current = buffer.lookup_transform("world/base", "arm/flange", Time())
    assert current.header.stamp.sec == 1
    assert current.transform.translation.x == pytest.approx(2.)
    buffer.set_transform(transform("arm/base", "arm/flange", [0, 2, 0, 0, 0, 0, 1], Stamp(sec=11)), "kinex")
    assert not buffer.can_transform("world/base", "arm/flange", Time(seconds=11))
    buffer.observe_clock(11_000_000_000)
    assert buffer.lookup_transform("world/base", "arm/flange", Time(seconds=11)).transform.translation.y == pytest.approx(2.)


def test_failed_lookups_name_their_cause():
    from builtin_interfaces.msg import Time as Stamp
    from rclpy.time import Time
    from tf2_ros import TransformException
    from kinex.host.frame_graph import FrameBuffer
    from kinex.common.frames import transform

    buffer = FrameBuffer()
    buffer.set_transform_static(transform("world/base", "arm/base", [1, 0, 0, 0, 0, 0, 1]), "kinex")
    # The camera's mount is uncalibrated, so no edge joins it to the world.
    buffer.set_transform_static(transform("camera/base", "camera/optical", [0, 0, 0, 0, 0, 0, 1]), "kinex")
    buffer.observe_clock(130_000_000_000)
    for sec in range(100, 131):
        buffer.set_transform(transform("arm/base", "arm/flange", [0, 0, 1, 0, 0, 0, 1], Stamp(sec=sec)), "kinex")

    def explain(source, sec=0):
        return buffer.explain("world/base", source, Time(seconds=sec))
    assert explain("arm/tool0") == "arm/tool0 is not on TF; arm has arm/base, arm/flange"
    assert explain("gripper/tcp") == "gripper/tcp is not on TF, nor is any frame of gripper"
    assert explain("camera/optical") == (
        "world/base and camera/optical are not connected: world/base is under world/base and "
        "camera/optical under camera/base; a mount of unknown calibration publishes no edge")
    stale = explain("arm/flange", 90)
    assert "extrapolation into the past" in stale and stale.endswith("Kinex keeps 10 s of world-time TF history.")
    assert buffer.timed_out("world/base", "arm/tool0", Time(), 5) == (
        "transform lookup timed out after 5 s: arm/tool0 is not on TF; arm has arm/base, arm/flange")
    # A direct lookup raises tf2's own exception type with the same cause.
    with pytest.raises(TransformException, match="extrapolation into the past.*Kinex keeps 10 s"):
        buffer.lookup_transform("world/base", "arm/flange", Time(seconds=90))
    with pytest.raises(TransformException, match="^arm/tool0 is not on TF; arm has "):
        buffer.lookup_transform("world/base", "arm/tool0", Time())
    buffer.device("arm", False)
    assert explain("arm/flange") == "arm is not loaded"


def test_the_frame_service_keeps_longer_history_than_a_scope():
    from builtin_interfaces.msg import Time as Stamp
    from rclpy.time import Time
    from kinex.host.frame_graph import SCOPE_HISTORY_S, SERVICE_HISTORY_S, FrameBuffer
    from kinex.common.frames import transform

    buffers = FrameBuffer(), FrameBuffer(SERVICE_HISTORY_S)
    for buffer in buffers:
        buffer.set_transform_static(transform("world/base", "arm/base", [0, 0, 0, 0, 0, 0, 1]), "kinex")
        buffer.observe_clock(130_000_000_000)
        for sec in range(131):
            buffer.set_transform(transform("arm/base", "arm/flange", [sec, 0, 0, 0, 0, 0, 1], Stamp(sec=sec)), "kinex")
    old = Time(seconds=130 - (SCOPE_HISTORY_S + SERVICE_HISTORY_S) / 2)
    assert [buffer.can_transform("world/base", "arm/flange", old) for buffer in buffers] == [False, True]


def test_unloaded_static_frames_cannot_leak_from_tf2_cache():
    from kinex.host.frame_graph import FrameBuffer
    from kinex.common.frames import transform
    from rclpy.time import Time
    from tf2_ros import TransformException
    buffer = FrameBuffer()
    for parent, child in [('world/base', 'arm/base'), ('arm/base', 'gripper/base'), ('gripper/base', 'gripper/tcp')]:
        buffer.set_transform_static(transform(parent, child, [0, 0, 1, 0, 0, 0, 1]), 'test')
    assert buffer.lookup_transform('world/base', 'gripper/tcp', Time()).transform.translation.z == 3
    buffer.device('arm', False)
    assert not buffer.can_transform('world/base', 'gripper/tcp', Time())
    with pytest.raises(TransformException):
        buffer.lookup_transform('world/base', 'gripper/tcp', Time())
    buffer.device('arm', True)
    assert not buffer.can_transform('world/base', 'gripper/tcp', Time())
    buffer.set_transform_static(transform('world/base', 'arm/base', [0, 0, 2, 0, 0, 0, 1]), 'test')
    assert buffer.lookup_transform('world/base', 'gripper/tcp', Time()).transform.translation.z == 4


def test_static_publication_roots_unknown_edges_and_stereo_frames():
    from unittest.mock import patch
    from kinex.common.frames import publish_profile, ros_frame
    from tf2_ros import Buffer, TransformException
    from rclpy.time import Time

    frames = {
        'base': {'parent': None, 'translation_m': [100, 100, 100], 'rotation_xyzw': [0, 0, 0, 1]},
        'color': {'parent': 'camera:base', 'translation_m': [0, 0, 0], 'rotation_xyzw': [0, 0, 0, 1]},
        'depth': {'parent': 'camera:color', 'translation_m': [.05, 0, 0], 'rotation_xyzw': [0, 0, 0, 1]},
        'unknown': {'parent': 'camera:base', 'translation_m': None, 'rotation_xyzw': [0, 0, 0, 1]},
        'independent': {'parent': None, 'translation_m': [1, 2, 3], 'rotation_xyzw': [0, 0, 0, 1]},
    }
    with patch('tf2_ros.StaticTransformBroadcaster') as broadcaster:
        publish_profile(None, {'name': 'camera', 'frames': frames})
    edges = broadcaster.return_value.sendTransform.call_args.args[0]
    assert {edge.child_frame_id for edge in edges} == {'camera/color', 'camera/depth'}
    buffer = Buffer()
    for edge in edges:
        buffer.set_transform_static(edge, 'test')
    tf = buffer.lookup_transform(ros_frame('camera:color'), ros_frame('camera:depth'), Time())
    assert tf.transform.translation.x == pytest.approx(.05)
    assert ros_frame('camera/base') == 'camera/base'
    for missing in ('camera/unknown', 'camera/independent'):
        with pytest.raises(TransformException):
            buffer.lookup_transform('camera/base', missing, Time())
