"""Owner-thread observation publication: clock, joint states, cameras and spectator streams."""
import threading

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from rosbridge.core.devices import SampleSchedule
from . import conversions as conv
from .devices import STREAM_QOS, Camera

CAMERA_RATE_HZ = 30.


class Spectator:
    """The owner thread renders; a timer publishes the latest frame so reliable writes never block physics."""

    def __init__(self, node, view):
        self.view, self.frame = view, f'spectator_{view.name}'
        self.image = node.create_publisher(Image, f'world/spectator/{view.name}/image_raw', STREAM_QOS)
        self.info = node.create_publisher(CameraInfo, f'world/spectator/{view.name}/camera_info', STREAM_QOS)
        self._lock = threading.Lock()
        self._pending = None
        node.create_timer(.005, self._flush)

    def submit(self, stamp, sim):
        info = conv.camera_info(stamp, self.frame, self.view.width, self.view.height,
                                *sim.spectator_intrinsics(self.view.name))
        image = conv.image(stamp, self.frame, sim.render_spectator(self.view.name))
        with self._lock:
            self._pending = (info, image)

    def _flush(self):
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self.info.publish(pending[0])
            self.image.publish(pending[1])


class StatusPublisher:
    """The backend's status texts ("<device>/<name>" -> text), latched, published when they change and on reset."""

    def __init__(self, node, keys):
        self.publishers = {key: node.create_publisher(String, key, STREAM_QOS) for key in keys}
        self.last = {}

    def publish(self, status, reset=False):
        for key, publisher in self.publishers.items():
            value = status.get(key, "")
            if reset or self.last.get(key) != value:
                publisher.publish(String(data=value))
                self.last[key] = value


class ObservationPublisher:
    def __init__(self, node, spectator_rate_hz, views=None):
        self.node, self.sim = node, node.sim
        self.clock = node.create_publisher(Clock, '/clock', STREAM_QOS)
        self.status = StatusPublisher(node, sorted(getattr(self.sim, 'status', None) or {}))
        self.cameras = SampleSchedule(CAMERA_RATE_HZ)
        self.spectator_schedule = SampleSchedule(spectator_rate_hz)
        self.spectators = [Spectator(node, view) for view in (self.sim.spectators if views is None else views)]

    def publish(self, reset=False):
        time_s = self.node.world.time_s
        stamp = conv.stamp(time_s)
        positions, velocities = self.sim.state()
        self.clock.publish(conv.clock(stamp))
        self.status.publish(getattr(self.sim, 'status', None) or {}, reset)
        cameras = self.cameras.due(time_s, reset)
        for device in self.node.devices:
            if not isinstance(device, Camera):
                device.publish(stamp, positions, velocities)
                continue
            if reset:
                device.reset()
            if cameras:
                device.publish(stamp)
        if self.spectators and self.spectator_schedule.due(time_s, reset):
            for spectator in self.spectators:
                spectator.submit(stamp, self.sim)
