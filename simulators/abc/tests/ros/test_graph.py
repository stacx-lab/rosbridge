"""ABC observations and spectator streams on the world node; motion is tested by the demo."""
import pytest
import numpy as np
from kinex_interfaces.srv import CaptureRGBD
from sensor_msgs.msg import CameraInfo, Image, JointState

from rosbridge_runtime.devices import STREAM_QOS
from tests.ros.support import Graph, seconds


def test_abc_observations_without_spectators(tmp_path):
    graph = Graph(tmp_path, backend="abc")
    node, last = graph.node, {}
    try:
        for kind, topic, key in ((JointState, "left_arm/joint_state", "joints"),
                                 (CameraInfo, "camera_top/camera_info", "info")):
            node.create_subscription(kind, topic, lambda msg, k=key: last.update({k: msg}), STREAM_QOS)
        capture = node.create_client(CaptureRGBD, "camera_top/capture")
        assert graph.reset().success
        assert graph.wait(.34).success
        graph.until(lambda: all(k in last for k in ("joints", "info")) and capture.service_is_ready())
        assert last["joints"].name == [f"left_joint{i}" for i in range(1, 7)]
        # Full-size frames are checked through the reliable capture service; best-effort
        # streams of large images may drop fragments on a loaded host.
        response, info = graph.call(capture, CaptureRGBD.Request()), last["info"]
        image, depth = response.image, response.depth_image
        assert image.encoding == "rgb8" and len(image.data) == image.width * image.height * 3
        assert (image.width, image.height) == (info.width, info.height) == (640, 480)
        assert image.header.frame_id == info.header.frame_id == "camera_top/base"
        # Registered depth: same grid, stamp and optical frame; 16UC1 millimetres.
        assert (depth.encoding, depth.width, depth.height, depth.step) == ("16UC1", 640, 480, 1280)
        assert depth.header == image.header
        millimetres = np.frombuffer(bytes(depth.data), np.uint16)
        assert (millimetres > 0).mean() > .9 and 200 < np.median(millimetres[millimetres > 0]) < 2000
        topics = dict(node.get_topic_names_and_types())
        assert topics["/camera_top/depth"] == ["sensor_msgs/msg/Image"]
        assert info.k[0] > 0 and info.k[2] == (info.width - 1) / 2
        topics = {name for name, _ in node.get_topic_names_and_types()}
        assert not any(name.startswith("/world/spectator/") for name in topics)
    finally:
        graph.close()


@pytest.mark.parametrize('mode', ['stepped', 'realtime'])
def test_world_publishes_spectator_streams(tmp_path, mode):
    graph = Graph(tmp_path, backend='abc', mode=mode, spectators=True)
    node, frames, infos = graph.node, {}, {}
    try:
        for view in ('third_person', 'wrist', 'right_wrist'):
            node.create_subscription(Image, f'world/spectator/{view}/image_raw',
                                     lambda msg, view=view: frames.update({view: msg}), STREAM_QOS)
            node.create_subscription(CameraInfo, f'world/spectator/{view}/camera_info',
                                     lambda msg, view=view: infos.update({view: msg}), STREAM_QOS)
        graph.until(lambda: len(frames) == len(infos) == 3)
        for view, image in frames.items():
            assert (image.width, image.height, image.encoding) == (960, 720, 'rgb8')
            assert len(image.data) == 960 * 720 * 3
            assert image.header.frame_id == infos[view].header.frame_id == f'spectator_{view}'
        for topic in ('world/spectator/third_person/image_raw', 'camera_top/camera_info', 'left_arm/joint_state'):
            assert {e.node_name for e in node.get_publishers_info_by_topic(f'/{topic}')} == {'world'}
        if mode == 'stepped':
            assert all(seconds(m.header.stamp) == 0 for m in frames.values())
            graph.spin(.2)
            assert graph.wait(0.).waited_s == 0
        before = seconds(frames['third_person'].header.stamp)
        assert graph.wait(.34).success
        graph.until(lambda: all(seconds(m.header.stamp) > before for m in frames.values()))
        if mode == 'stepped':
            stamps = {k: seconds(v.header.stamp) for k, v in frames.items()}
            graph.spin(.3)
            assert stamps == {k: seconds(v.header.stamp) for k, v in frames.items()}
        assert graph.reset().success
        if mode == 'stepped':
            graph.until(lambda: all(seconds(m.header.stamp) == 0 for m in frames.values()))
    finally:
        graph.close()
