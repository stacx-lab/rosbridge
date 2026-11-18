"""A lensed camera's geometry, unproject and project services on the world node."""
import json

import numpy as np
import pytest
from custom_interfaces.srv import Json
from sensor_msgs.msg import CameraInfo

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph


def request(graph, client, **fields):
    reply = graph.call(client, Json.Request(request=json.dumps(fields)))
    return reply, json.loads(reply.response) if reply.success else None


def test_a_lensed_camera_serves_its_geometry_and_converts_through_the_lens(tmp_path):
    graph = Graph(tmp_path, backend="lens_toy")
    try:
        clients = {name: graph.node.create_client(Json, f"camera/{name}")
                   for name in ("geometry", "unproject", "project")}
        info = []
        graph.node.create_subscription(CameraInfo, "camera/camera_info", info.append, STREAM_QOS)
        graph.until(lambda: info and all(client.service_is_ready() for client in clients.values()))
        reply, geometry = request(graph, clients["geometry"])
        assert reply.success and list(geometry["streams"]) == ["color"]
        stream = geometry["streams"]["color"]
        assert (stream["width"], stream["height"], stream["frame"]) == (64, 48, "camera/base")
        assert stream["projection"] == {"model": "pinhole", "fx": 40., "fy": 40., "cx": 31.5, "cy": 23.5, "skew": 0}
        # camera_info states the same lens.
        lens = [stream["distortion"][key] for key in ("k1", "k2", "p1", "p2", "k3")]
        assert stream["distortion"]["model"] == "opencv_brown_conrady" and list(info[-1].d) == pytest.approx(lens)
        calibration = stream["calibration_id"]
        pixels = [[31.5, 23.5], [0., 0.], [63., 47.]]
        reply, rays = request(graph, clients["unproject"], stream="color", calibration_id=calibration, pixels=pixels)
        assert reply.success and rays["rays"][0] == pytest.approx([0., 0., 1.])
        points = [[2. * x, 2. * y, 2.] for x, y, _ in rays["rays"]] + [[0., 0., -1.]]
        reply, projected = request(graph, clients["project"], stream="color", calibration_id=calibration, points=points)
        assert reply.success
        np.testing.assert_allclose(projected["pixels"][:3], pixels, atol=1e-8)
        assert projected["pixels"][3] is None  # behind the camera
        # A failure is a reply that names its cause, and the services keep answering.
        reply, _ = request(graph, clients["unproject"], stream="color", calibration_id="stale", pixels=pixels)
        assert not reply.success and reply.message == (
            "ValueError: stale or unknown camera calibration; refresh geometry")
        reply, _ = request(graph, clients["unproject"], stream="color")
        assert not reply.success and reply.message.startswith("TypeError: ")
        assert not graph.call(clients["geometry"], Json.Request(request="not json")).success
        assert request(graph, clients["geometry"])[1] == geometry  # one calibration for the run
    finally:
        graph.close()


def test_an_undistorted_camera_serves_no_geometry_services(tmp_path):
    graph = Graph(tmp_path, backend="camera_toy")
    try:
        graph.frame("camera")  # the camera is discovered
        services = dict(graph.node.get_service_names_and_types())
        assert "/world/wait" in services
        assert not any(name.startswith("/camera/") for name in services)
    finally:
        graph.close()
