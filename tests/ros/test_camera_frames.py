"""A paused world's camera reaches readers that subscribe from a fresh process."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from rosbridge_runtime.devices import FRAME_QOS, KEEPALIVE_S, Camera
from tests.ros.support import Graph, seconds

READERS = 10


def resting_camera(subscribers=1):
    """A camera of a world resting at 1 s; records renders, publications and owner requests."""
    camera = Camera.__new__(Camera)
    renders, published, requests = [], [], []

    def render(source):
        renders.append(source)
        return np.zeros((2, 4, 3), np.uint8)
    camera.spec, camera.frame = SimpleNamespace(name="camera", source="top"), "camera/base"
    camera.node = SimpleNamespace(sim=SimpleNamespace(render=render),
                                  world=SimpleNamespace(time_s=1., submit=requests.append))
    camera.image = SimpleNamespace(get_subscription_count=lambda: subscribers,
                                   publish=published.append)
    camera.latest = camera.latest_depth = camera.depth = None
    camera.sent_s, camera.sent_stamp, camera.queued = time.monotonic(), None, False
    return camera, renders, published, requests


def test_a_resting_world_resends_its_frame_once_queued_and_without_rendering_again():
    camera, renders, published, requests = resting_camera()
    camera.send_current()
    camera.keepalive()
    assert requests == []  # it has just sent
    camera.sent_s -= KEEPALIVE_S
    camera.keepalive()
    camera.keepalive()
    assert requests == [camera.resend]
    requests.pop()()
    assert len(published) == 2 and published[1] is published[0]
    assert renders == ["top"] and not camera.queued


def test_a_running_world_keeps_its_camera_rate_and_an_unsubscribed_camera_sends_nothing():
    camera, renders, published, requests = resting_camera()
    camera.send_current()
    camera.node.world.time_s, camera.sent_s = 1.1, camera.sent_s - KEEPALIVE_S
    camera.keepalive()
    requests.pop()()
    assert len(published) == 1 and renders == ["top"]
    idle, _, _, requests = resting_camera(subscribers=0)
    idle.sent_s -= KEEPALIVE_S
    idle.keepalive()
    assert requests == []


def fresh_frame(domain, camera, timeout=3.):
    """The first image and equally stamped depth a new participant receives; None after timeout.

    The reader subscribes before its participant has discovered the camera.
    """
    context = Context()
    rclpy.init(context=context, domain_id=domain)
    executor = SingleThreadedExecutor(context=context)
    node = Node("fresh_reader", context=context)
    received = {"image": [], "depth": []}
    try:
        for name, messages in received.items():
            node.create_subscription(Image, f"{camera}/{name}", messages.append, FRAME_QOS)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            depths = {seconds(message.header.stamp): message for message in received["depth"]}
            for image in received["image"]:
                if seconds(image.header.stamp) in depths:
                    return image, depths[seconds(image.header.stamp)]
            rclpy.spin_once(node, executor=executor, timeout_sec=.01)
        return None
    finally:
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()


@pytest.mark.parametrize("configuration", [None, "depth"])
def test_a_configuration_selects_the_devices_the_world_serves(tmp_path, configuration):
    graph = Graph(tmp_path, backend="depth_toy", configuration=configuration)
    try:
        def topics():
            return {name for name, _ in graph.node.get_topic_names_and_types()}
        graph.until(lambda: {"/camera/image", "/camera/camera_info"} <= topics())
        assert ("/camera/depth" in topics()) == (configuration == "depth")
    finally:
        graph.close()


def test_every_fresh_reader_of_a_paused_camera_receives_its_current_frame(tmp_path):
    graph = Graph(tmp_path, backend="camera_toy")
    try:
        # The world advances while no reader is subscribed: a frame from before a wait is stale.
        for advance in (.3, .2):
            now = graph.wait(advance).time_s
            for reader in range(READERS):
                frame = fresh_frame(graph.lease.domain, "camera")
                assert frame is not None, f"fresh reader {reader} received no frame"
                assert [seconds(image.header.stamp) for image in frame] == [pytest.approx(now)] * 2
    finally:
        graph.close()
