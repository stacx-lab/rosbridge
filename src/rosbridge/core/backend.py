"""Backend contract. Importing it requires neither ROS nor a simulator."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from .devices import DeviceSpec, SpectatorView


@dataclass(frozen=True)
class CommandGroup:
    """A named vector or scalar endpoint mapped into the native action.

    A velocity group is a mobile base: it commands body velocities in its own
    units and expires to zero; its feedback is odometry, not a hold target.
    """

    name: str
    indices: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    scalar: bool = False
    velocity: bool = False


@dataclass(frozen=True)
class ActionLayout:
    """Action and feedback channel order, partitioned into named groups.

    Commands are positions in feedback units, so a measured state is a valid
    hold target. Joint names are unique within a group; separate devices may
    share native names such as joint1. A velocity group's channels are the
    planar odometry pose (x, y, yaw) and body twist (vx, vy, wz).

    measured names joints a group reports in its joint state but never
    commands, as (group, joint names) pairs; their positions and velocities
    follow the action channels in state(), in this order. A command may name
    them too; they are ignored.

    clip: targets outside a group's limits are clipped to them (and the
    clipping reported) instead of refused.
    """

    joint_names: tuple[str, ...]
    groups: tuple[CommandGroup, ...]
    measured: tuple[tuple[str, tuple[str, ...]], ...] = ()
    clip: bool = False

    def __post_init__(self):
        if not self.joint_names or not all(self.joint_names):
            raise ValueError("joint names must be nonempty")
        if len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("command group names must be unique")
        indices = []
        for group in self.groups:
            if (not group.name or not group.indices
                    or len(group.indices) != len(group.lower)
                    or len(group.indices) != len(group.upper)
                    or (group.scalar and len(group.indices) != 1)
                    or (group.velocity and (group.scalar or len(group.indices) != 3))
                    or not all(0 <= i < len(self.joint_names) for i in group.indices)):
                raise ValueError(f"invalid command group: {group.name}")
            if len({self.joint_names[i] for i in group.indices}) != len(group.indices):
                raise ValueError(f"joint names must be unique within {group.name}")
            lo, hi = np.asarray(group.lower), np.asarray(group.upper)
            if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
                raise ValueError(f"invalid limits: {group.name}")
            indices.extend(group.indices)
        if sorted(indices) != list(range(len(self.joint_names))):
            raise ValueError("groups must partition the action indices exactly once")
        groups = {g.name: g for g in self.groups}
        for name, joints in self.measured:
            group = groups.get(name)
            if group is None or group.velocity or not joints or not all(joints):
                raise ValueError(f"invalid measured joints: {name}")
            reported = [self.joint_names[i] for i in group.indices] + list(joints)
            if len(set(reported)) != len(reported):
                raise ValueError(f"joint names must be unique within {name}")
        if len({name for name, _ in self.measured}) != len(self.measured):
            raise ValueError("measured joints must name each group once")

    def joint_state(self, group):
        """A group's reported joint names and their state() indices: its action channels, then its measured
        joints."""
        command = next(g for g in self.groups if g.name == group)
        names, indices = [self.joint_names[i] for i in command.indices], list(command.indices)
        start = len(self.joint_names)
        for name, joints in self.measured:
            if name == group:
                names += joints
                indices += range(start, start + len(joints))
            start += len(joints)
        return names, indices


class SimulatorBackend(Protocol):
    """Created, reset, stepped, rendered and closed only on the main thread.

    Layout, device and camera names stay fixed across resets. Failed resets
    preserve the previous episode. Every step advances exactly one `period`.
    A backend whose native command is not a position, such as a gripper
    direction, converts the position target inside step().

    Optionally, `status` maps "<device>/<name>" to a short text the world
    publishes as std_msgs/String on that topic whenever it changes (latched):
    states a joint position cannot carry, such as whether a hand holds an
    object. Its keys stay fixed across resets.
    """

    default_task: str
    layout: ActionLayout
    devices: tuple[DeviceSpec, ...]
    cameras: tuple[str, ...]
    period: float
    time_s: float
    task: str
    seed: int
    steps: int
    max_steps: int | None
    done: bool
    reward: float
    success: bool
    spectators: tuple[SpectatorView, ...]

    def __init__(self, *, max_steps, terminate_on_success): ...
    def reset(self, task: str, seed: int, randomize: bool = True) -> None: ...
    def state(self) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, action: np.ndarray) -> None: ...
    # A camera's image size and pinhole: width, height, fx, fy, cx, cy (ROS pixel centers).
    def intrinsics(self, camera: str) -> tuple: ...
    def render(self, camera: str) -> np.ndarray: ...
    # RGB-D cameras only: float32 (H, W) metres along the optical axis; 0 or nonfinite is invalid.
    def render_depth(self, camera: str) -> np.ndarray: ...
    def configure_spectators(self, width: int, height: int) -> None: ...
    def render_spectator(self, name: str) -> np.ndarray: ...
    def spectator_intrinsics(self, name: str) -> tuple: ...
    def close(self) -> None: ...
