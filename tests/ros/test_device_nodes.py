"""Owned implementations and borrowed interfaces have independent lifecycles."""
import asyncio
import copy
import os
from pathlib import Path
import sys
import uuid

import pytest
import yaml

if os.environ.get("KINEX_PYTHON"):
    sys.path.insert(0, os.environ["KINEX_PYTHON"])
pytest.importorskip("kinex.host.device")
from kinex.communication import communication, qos
from kinex.host.device import DeviceHost


def stream(endpoint):
    return {"transport": "topic", "direction": "output", "endpoint": endpoint,
            "type": "std_msgs/msg/Float64", "qos": {"durability": "transient_local"}}


def profile(name, operations, nodes=()):
    return {"name": name, "operations": operations,
            "runtime": {"domain_id": 201, "nodes": list(nodes),
                        "ready_timeout_s": 5}}


def test_device_can_own_multiple_nodes(tmp_path):
    source = '''from rclpy.node import Node
from std_msgs.msg import Float64
from kinex.communication import qos

def first(profile, context):
    return create(profile, context, "first", 1.)

def second(profile, context):
    return create(profile, context, "second", 2.)

def create(profile, context, name, value):
    node = Node(name, context=context)
    op = profile["operations"][name]
    publisher = node.create_publisher(Float64, op["endpoint"], qos(op))
    publisher.publish(Float64(data=value))
    return node
'''
    entry = tmp_path / "nodes.py"
    entry.write_text(source)

    async def run():
        prefix = "/test_" + uuid.uuid4().hex
        host = DeviceHost()
        definition = profile("pair", {name: stream(prefix + "/" + name) for name in ("first", "second")},
                             [{"entry": str(entry), "factory": name} for name in ("first", "second")])
        try:
            await host.load(definition)
            assert {node.get_name() for node in host.io.extra_nodes} == {"first", "second"}
            for name, value in (("first", 1.), ("second", 2.)):
                assert (await host.io.subscribe("pair." + name).next(timeout_s=3)).data == value
        finally:
            await host.run("unload", {})
        assert host.io is None
    asyncio.run(run())


def test_two_devices_can_borrow_endpoints_from_one_node():
    from std_msgs.msg import Float64

    async def run():
        prefix = "/test_" + uuid.uuid4().hex
        hosts = []
        async with communication({}, domain_id=201) as backend:
            operations = [stream(prefix + "/a"), stream(prefix + "/b")]
            publishers = [backend.node.create_publisher(Float64, op["endpoint"], qos(op)) for op in operations]
            command = {**stream(prefix + "/command"), "direction": "input"}
            received = []
            backend.node.create_subscription(Float64, command["endpoint"], received.append, qos(command))
            try:
                for index, operation in enumerate(operations):
                    host = DeviceHost()
                    hosts.append(host)
                    await host.load(profile("device" + str(index), {"value": operation, "command": command}))
                    assert host.io.extra_nodes == []
                await hosts.pop().run("unload", {})
                assert backend.context.ok()
                publishers[0].publish(Float64(data=1.2345678901234567))
                message = await hosts[0].io.subscribe("device0.value").next(timeout_s=3)
                assert message.data == 1.2345678901234567
                await hosts[0].io.publish("device0.command", {"data": 9.5})
                async with asyncio.timeout(3):
                    while not received:
                        await asyncio.sleep(.02)
                assert received[-1].data == 9.5
            finally:
                for host in reversed(hosts):
                    await host.run("unload", {})
            assert backend.context.ok()
    asyncio.run(run())


def test_catalog_camera_borrows_rosbridge_endpoints_and_checks_readiness():
    from kinex_interfaces.srv import Capture
    from sensor_msgs.msg import CameraInfo, Image

    profile = Path(__file__).resolve().parents[2] / "catalog/robots/libero/libero-panda-tabletop/devices/camera_agentview/profile.yaml"
    definition = yaml.safe_load(profile.read_text())
    assert definition["runtime"]["nodes"] == []
    definition["runtime"].update(domain_id=201, ready_timeout_s=.15)
    operations = definition["operations"]

    async def run():
        absent = DeviceHost()
        try:
            with pytest.raises(TimeoutError):
                await absent.load(copy.deepcopy(definition))
        finally:
            await absent.run("unload", {})
        async with communication({}, domain_id=201) as backend:
            image = Image(width=1, height=1, encoding="rgb8", step=3, data=[1, 2, 3])
            image.header.frame_id = "camera_agentview/base"
            publisher = backend.node.create_publisher(Image, operations["image"]["endpoint"], qos(operations["image"]))
            backend.node.create_publisher(CameraInfo, operations["camera_info"]["endpoint"], qos(operations["camera_info"]))

            def capture(request, response):
                response.image = image
                return response

            backend.node.create_service(Capture, operations["capture"]["endpoint"], capture)
            definition["runtime"]["ready_timeout_s"] = 5
            host = DeviceHost()
            try:
                await host.load(definition)
                observation = host.io.subscribe("camera_agentview.image")
                async with asyncio.timeout(3):
                    while not publisher.get_subscription_count():
                        await asyncio.sleep(.02)
                publisher.publish(image)
                assert list((await observation.next(timeout_s=3)).data) == [1, 2, 3]
                response = await host.io.request("camera_agentview.capture")
                assert response.image.header.frame_id == "camera_agentview/base"
            finally:
                await host.run("unload", {})
            assert backend.context.ok()
    asyncio.run(run())
