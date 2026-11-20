"""Owner-thread observation publication: clock, episode budget, joint states, cameras and spectator streams."""
import threading

from custom_interfaces.msg import Budget
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

from rosbridge.core.devices import SampleSchedule, camera_rate
from . import conversions as conv
from .devices import STREAM_QOS, Camera


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


class ObservationPublisher:
    def __init__(self, node, spectator_rate_hz):
        self.node, self.sim = node, node.sim
        self.clock = node.create_publisher(Clock, '/clock', STREAM_QOS)
        self.budget = node.create_publisher(Budget, 'world/budget', STREAM_QOS)  # latched, as wait replies give it
        self.cameras = SampleSchedule(camera_rate(self.sim))
        self.spectator_schedule = SampleSchedule(spectator_rate_hz)
        self.spectators = [Spectator(node, view) for view in self.sim.spectators]

    def publish(self):
        time_s = self.node.world.time_s
        stamp = conv.stamp(time_s)
        positions, velocities = self.sim.state()
        self.clock.publish(conv.clock(stamp))
        used_s, max_s = self.node.world.budget()
        self.budget.publish(Budget(used_s=used_s, max_s=max_s))
        cameras = self.cameras.due(time_s)
        for device in self.node.devices:
            if not isinstance(device, Camera):
                device.publish(stamp, positions, velocities)
            elif cameras:
                device.publish(stamp)
        if self.spectators and self.spectator_schedule.due(time_s):
            for spectator in self.spectators:
                spectator.submit(stamp, self.sim)
