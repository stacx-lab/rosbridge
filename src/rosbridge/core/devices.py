"""Simulator-neutral device inventory, spectator views and observation sampling."""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class DeviceSpec:
    """A device on the world node; its endpoints are named `<name>/...` and its kind is
    its Kinex device type."""
    name: str
    kind: str
    source: str

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', self.name) or self.name == 'world':
            raise ValueError(f'invalid device name: {self.name}')
        if self.kind not in ('joint', 'gripper', 'locomotion', 'rgb', 'rgbd'):
            raise ValueError(f'unsupported device kind: {self.kind}')


def device_inventory(layout, cameras, depth=()):
    """Joint, gripper and locomotion devices are named by command group; cameras maps device
    name to native camera.

    Cameras named in depth are RGB-D devices; their backend implements render_depth.
    """
    if not set(depth) <= set(cameras):
        raise ValueError('depth cameras must be cameras')
    devices = tuple(DeviceSpec(g.name, 'gripper' if g.scalar else 'locomotion' if g.velocity else 'joint',
                               g.name) for g in layout.groups)
    devices += tuple(DeviceSpec(name, 'rgbd' if name in depth else 'rgb', source)
                     for name, source in cameras.items())
    if len({d.name for d in devices}) != len(devices):
        raise ValueError('device names must be unique')
    return devices


@dataclass(frozen=True)
class SpectatorView:
    name: str
    source: str
    width: int = 960
    height: int = 720

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', self.name):
            raise ValueError(f'invalid spectator view name: {self.name}')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('spectator dimensions must be positive')


class SampleSchedule:
    """At most one sample per control boundary; skip missed sampling deadlines."""
    def __init__(self, rate):
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('sample rate must be finite and nonnegative')
        self.rate = rate
        self.next_s = 0.

    def due(self, time_s, reset=False):
        if reset:
            self.next_s = time_s
        if not self.rate or time_s + 1e-9 < self.next_s:
            return False
        intervals = max(1, math.floor((time_s - self.next_s + 1e-9) * self.rate) + 1)
        self.next_s += intervals / self.rate
        return True
