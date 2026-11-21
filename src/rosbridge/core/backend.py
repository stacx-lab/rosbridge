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
    A pose group names the tip link of a joint device that takes pose targets:
    the tip's pose in the device base, [x, y, z, qx, qy, qz, qw], which the
    simulator's own Cartesian control executes. Its feedback stays joint positions,
    and its limits bound nothing it is commanded. A tip with joint_targets is a
    joint group that also takes its tip's poses: the backend's joint_path solves a
    whole pose path into joints when it arrives.
    """

    name: str
    indices: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    scalar: bool = False
    velocity: bool = False
    tip: str | None = None
    joint_targets: bool = False


@dataclass(frozen=True)
class ActionLayout:
    """Action and feedback channel order, partitioned into named groups.

    Commands are positions in feedback units, so a measured state is a valid
    hold target. Joint names are unique within a group; separate devices may
    share native names such as joint1. A velocity group's channels are the
    planar odometry pose (x, y, yaw) and body twist (vx, vy, wz). A pose group's
    channels carry no command: its tip pose target travels in step's poses.
    """

    joint_names: tuple[str, ...]
    groups: tuple[CommandGroup, ...]

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
                    or (group.tip is not None and (not group.tip or group.scalar or group.velocity))
                    or (group.joint_targets and group.tip is None)
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


class SimulatorBackend(Protocol):
    """Created, reset, stepped, rendered and closed only on the main thread.

    A run is one episode: the world resets the backend once, to load the launch's
    task, seed and scene options, then steps it. Every step advances exactly one
    `period`.

    A simulator that selects scenes in its own terms, such as RoboCasa's kitchen
    layout and style ids, names them in its manifest's scene_options; a launch
    passes those it gives to reset as keyword arguments, which the backend checks
    and defaults. The seed selects everything the scene options do not.

    A simulator that serves alternative device sets, such as RoboDojo's RGB-D
    cameras, names them in its manifest's configurations; a launch that selects
    one creates the backend with configuration=NAME, and the backend serves that
    configuration's devices, as its rigs' configurations/NAME describe them. A
    backend without configurations is never given one.

    A backend whose native command is not a position, such as a gripper
    direction, converts the position target inside step().

    A backend sets done at its step limit, if it has one, and, when created with
    early_stop, also where its simulator's own evaluator ends an episode, such as
    a success or a task failure. It then names that stop in `stop`, a pair of the
    record's reason ("success" or "failure") and a native cause or None; `stop`
    stays None otherwise, so done without it is the step limit ("timeout"). A
    backend whose only early stop is success may leave `stop` unset: done with
    success then names it.

    A backend whose layout has pose groups is stepped as step(action, poses):
    poses maps each pose group to its tip's target pose in the device base,
    [x, y, z, qx, qy, qz, qw], or to None until the group's first command. Its
    tip_pose(group) returns the measured tip pose there, which a stop holds and
    a group's first trajectory starts from.

    A backend with a joint group that takes tip poses has
    joint_path(group, joints, poses): the joints for the group's tip pose at each
    control period of a path that starts at joints, or a ValueError saying why the
    path has none. The path is then refused whole, before the group moves.

    Optionally, camera_rate_hz sets how many times per simulated second a
    subscribed camera renders (default 30), for a simulator whose renders are
    expensive. A new subscriber still gets a frame at once.

    Optionally, distortion maps a camera to the OpenCV Brown-Conrady lens
    (k1, k2, p1, p2, k3) its images render through, as its evaluator renders
    them; a camera it does not name renders as an undistorted pinhole. The
    camera's CameraInfo carries it, the world node serves its geometry,
    unproject and project operations (rosbridge.core.lens), and its rig's
    profile states the same lens (tools/robots/rigs.py: camera).

    Optionally, measures() returns the backend's own measures of the episode as
    JSON fields, such as BEHAVIOR's q_score; the episode record adds them.
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
    stop: tuple[str, str | None] | None
    spectators: tuple[SpectatorView, ...]

    def __init__(self, *, max_steps, early_stop): ...
    def reset(self, task: str, seed: int, randomize: bool = True, **scene) -> None: ...
    def state(self) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, action: np.ndarray) -> None: ...
    # A camera's image size and pinhole: width, height, fx, fy, cx, cy (ROS pixel centers).
    def intrinsics(self, camera: str) -> tuple: ...
    def render(self, camera: str) -> np.ndarray: ...
    # RGB-D cameras only: float32 (H, W) metres along the optical axis; 0 or nonfinite is invalid.
    def render_depth(self, camera: str) -> np.ndarray: ...
    # Spectator views at width x height; with `views`, only the named ones (the others cost nothing).
    def configure_spectators(self, width: int, height: int, views: list[str] | None = None) -> None: ...
    # One RGB frame of every configured view at the current state, by name: a backend renders them together.
    def render_spectators(self) -> dict[str, np.ndarray]: ...
    def spectator_intrinsics(self, name: str) -> tuple: ...
    def close(self) -> None: ...
