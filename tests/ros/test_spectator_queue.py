"""A slow DDS writer cannot hold the native thread or grow a frame backlog."""
import threading
from types import SimpleNamespace

import numpy as np
from rosbridge_runtime.devices import Camera
from rosbridge_runtime.publishing import Spectator
from rosbridge_runtime import conversions as conv


def frame(*_):
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_camera_reuses_one_capture_per_world_tick():
    captures = []
    camera = Camera.__new__(Camera)
    camera.spec, camera.frame, camera.latest = SimpleNamespace(source='top'), 'camera_top/base', None
    camera.node = SimpleNamespace(sim=SimpleNamespace(render=lambda source: captures.append(source) or frame()))
    image = camera.render(conv.stamp(1.))
    for _ in range(10):
        assert camera.render(conv.stamp(1.)) is image
    assert captures == ['top'] and image.header.frame_id == 'camera_top/base'
    camera.render(conv.stamp(1.1))
    assert len(captures) == 2


def test_slow_writer_keeps_only_latest_pending_frame():
    spectator = Spectator.__new__(Spectator)
    spectator.view, spectator.frame = SimpleNamespace(name='overview', width=2, height=2), 'spectator_overview'
    spectator._lock, spectator._pending = threading.Lock(), None
    entered, release = threading.Event(), threading.Event()
    published = []

    def publish(image):
        if not entered.is_set():
            entered.set()
            assert release.wait(5.)
        published.append(conv.seconds(image.header.stamp))

    spectator.info = SimpleNamespace(publish=lambda _: None)
    spectator.image = SimpleNamespace(publish=publish)
    sim = SimpleNamespace(spectator_intrinsics=lambda _: (1., 1., .5, .5), render_spectator=frame)
    spectator.submit(conv.stamp(0.), sim)
    writer = threading.Thread(target=spectator._flush)
    writer.start()
    try:
        assert entered.wait(5.)
        # A blocked DDS write must not prevent the owner from submitting frames.
        for t in range(1, 101):
            spectator.submit(conv.stamp(float(t)), sim)
        assert conv.seconds(spectator._pending[1].header.stamp) == 100.
    finally:
        release.set()
        writer.join(timeout=5.)
    assert not writer.is_alive()
    spectator._flush()
    assert published == [0., 100.]
