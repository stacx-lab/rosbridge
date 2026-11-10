"""BEHAVIOR-1K activities on an embodiment in OmniGibson; runs in the native worker.

OmniGibson needs Isaac Sim 5.1 on Python 3.11, so the world node proxies this class through
rosbridge.core.worker. The episode follows the official evaluator (environment.py); an
embodiment maps rosbridge channels to native actions (r1pro.py, r1pro_sharpa.py, g1_sharpa.py).
"""
import math
import secrets
import sys
from typing import NamedTuple

import numpy as np

from rosbridge.catalog import resolve
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.project import project_root
from .assets import check_dex_task, check_g1_task, check_task
from .environment import Episode, boot, rotation, values
from .rendering import delivered_time, render_for_capture, render_settings

OPTICAL = np.diag([1., -1., -1.])  # USD cameras look along -z with +y up; ROS optical is +z, +y down
OVERVIEW_FOV_Y = 65.
# Evaluator rendering (the R1Pro embodiments): physics steps without rendering, and a camera read
# renders once after a step, as OmniGibson's env.step renders once per step, with OmniGibson's
# own anti-aliasing. Without a render at the previous step the renderer has no recent history,
# so a read renders OmniGibson's num_rerenders_on_reset; light tasks add the evaluator's
# re-renders after its light sync.
RERENDERS = 3
# On-demand rendering (the G1): the renderer's temporal history stays usable across captures
# this close in world time: as cameras move between them, two renders clean a capture. After a
# longer gap (a reset, or cameras unrendered since moving), sixteen bring the path tracer to
# the same noise.
HISTORY_S, CUT_DELIVERIES = .25, 16
# Overview spectators keep these bearings from the robot's heading (radians counterclockwise).
OVERVIEWS = {"front_left": np.pi / 4, "rear_left": 3 * np.pi / 4}
FOLLOW_S = 1.  # the overviews turn after the robot this slowly, so gait sway does not shake them
FOLLOW_GAP_S = .25  # after a longer pause in the overviews, as after a cut, they face the heading
WALL_MARGIN = .2  # an overview stays this far (m) in front of what it would otherwise stand behind
# The robot's state, checked after every step (BehaviorSim._check_state). The numbers rest on 1,460 native scenarios
# with both R1Pro embodiments in two scenes: releases from rest tilted 30 to 150 degrees in 8 directions and 7
# postures, drives into furniture until it tips, bursts and joint targets on a lying robot, and pushes.
# - Notice: a joint more than JOINT_NOTICE beyond its range (rad, or m for a prismatic joint) is stated in every wait
#   reply while it lasts; the episode goes on. Pressed against a counter the R1Pro's arm and torso joints stay within
#   3e-4 rad of their limits and the Sharpa fingers within 0.18; of the 1,234 scenarios that started from a
#   reachable state, 9 went beyond 0.5.
# - End (reason invalid_state): a joint or base value that is not finite, at once; a revolute joint more than
#   REVOLUTE_END beyond its range; a prismatic joint more than its own range width beyond its range. A fall is not a
#   failed state and must not end the episode: the largest excess without divergence was 4.97 rad (a squatting robot
#   fell on its side with a wrist wedged under the 250 kg base, which wound the 18 N m forearm roll joint past its
#   stop for 3 s; it unwound by itself). Every measured runaway read 10 rad or more beyond, or not finite (15 of 15;
#   21 to 1334 rad in recorded agent episodes), so 2 pi ends none of the reachable scenarios and all the runaways.
# - The base's six virtual joints (x, y and z in m, rx, ry and rz in rad, each limited to [-1000, 1000]) are checked
#   for finite values only: those limits bound no motion of the robot (x and y stayed within 7.3 m, rx and ry within
#   6.3 rad, in falls and long drives), and only a state that is already not finite passes them.
JOINT_NOTICE = .5
REVOLUTE_END = 2 * math.pi


def inside(offset, contains, step):
    """The fraction of the horizontal offset (2-vector, m) from a point that stays where contains(fraction) holds,
    sampled every `step` metres from the point outward, less WALL_MARGIN; 1 if it holds all the way."""
    length = float(np.linalg.norm(offset))
    for fraction in np.arange(step, length + step, step) / max(length, 1e-9):
        if not contains(min(fraction, 1.)):
            return max(.01, (fraction * length - step - WALL_MARGIN) / length)
    return 1.


class Overview(NamedTuple):
    """An embodiment's overview spectators: `distance` m out horizontally from the base, with
    their eye and the point they look at `eye` and `target` m above the base's height at the
    episode start."""
    distance: float
    eye: float
    target: float


def checked_ranges(robot):
    """What BehaviorSim._check_state checks the robot's joints against: their names (DOF order), lower and upper
    limits, how far beyond its range each may go before the episode ends (REVOLUTE_END, or a prismatic joint's range
    width), and whether its range is checked at all (not for a holonomic base's virtual joints)."""
    names = list(robot.dof_names_ordered)
    lower, upper = values(robot.joint_lower_limits), values(robot.joint_upper_limits)
    prismatic = np.array([robot.joints[name].joint_type == "PrismaticJoint" for name in names])
    virtual = set(values(robot.base_idx).astype(int).tolist()) if robot.is_holonomic_base else set()
    ranged = np.array([index not in virtual for index in range(len(names))])
    return names, lower, upper, np.where(prismatic, upper - lower, REVOLUTE_END), ranged


def pose_matrix(position, quaternion_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation(quaternion_xyzw), values(position)
    return matrix


class BehaviorSim:
    simulator = "behavior"
    default_task = "turning_on_radio"
    check_task = staticmethod(check_task)

    def __init__(self, embodiment, *, max_steps=None, terminate_on_success=False):
        self.embodiment = embodiment
        self.requested_max_steps = self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success
        self.period = 1. / embodiment.frequencies["action_frequency"]
        self.evaluator_rendering = embodiment.rendering == "evaluator"
        self.episode = self.task = None
        self.spectators, self._spectators, self._mounts = (), {}, {}
        self._frames, self._spectator_frames, self._fresh, self._renders = {}, {}, False, 0
        # The base's attitude after the latest reset or step (Episode.attitude) for an embodiment whose base tilts,
        # else None; the world node publishes it with the odometry and states a large tilt in its wait replies.
        self.base_attitude = None
        # The robot's state after the latest step (_check_state): fault, why it became invalid, which ended the
        # episode; state_notice, a joint beyond its range by more than JOINT_NOTICE while it is valid. When the state
        # cannot even be read (frozen), observations are the last valid step's: its state and frames.
        self.fault = self.state_notice = None
        self._frozen, self._observed, self._held = False, None, {}

    @property
    def cameras(self):
        return tuple(self.embodiment.cameras)

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in self.cameras}, depth=self.cameras)

    def configure_spectators(self, width=960, height=720, views=None):
        # Overviews that follow the base, and the head camera's view; only the selected ones get render products.
        self.spectators = spectator_views((*((name, "base") for name in OVERVIEWS), ("head", "head_camera")),
                                          width, height, views)

    # Episode lifecycle ------------------------------------------------------------------

    def reset(self, task, seed, randomize=True):
        """Load public test instance seed % 20; randomize has no effect.

        Isaac hosts one environment per process, so a reset reloads the live one. A failure
        after the reload starts leaves the episode done until a reset succeeds.
        """
        native = resolve(self.simulator, task=task)[2]["native_task"]
        self.check_task(project_root() / ".runtime" / self.simulator, native)
        if self.episode is not None and native != self.task:
            raise ValueError("a BEHAVIOR worker serves one task; start another run for another task")
        if self.episode is None:
            boot(self.simulator)
            self.task = native
            self.episode = Episode(native, self.embodiment, self.requested_max_steps)
            self.layout = self.embodiment.layout(self.episode.robot)
            self.max_steps = self.episode.max_steps
            self._ranges = checked_ranges(self.episode.robot)
        self.done = True
        self.episode.load(seed)
        # The evaluator's TaskMetric takes the goal options' satisfaction here, once the instance has loaded and
        # before the policy acts: its partial credit counts only literals the episode newly satisfies.
        self.initial_goal_options = self.episode.env.task.get_goal_option_satisfaction(0)
        self.embodiment.reset(self.episode)
        self.episode.origin = self.episode.base_pose()
        self.seed, self.steps, self.done = seed, 0, False
        self.fault = self.state_notice = None
        self._frozen, self._observed, self._held = False, None, {}
        self._observe()
        self.success = bool(self.episode.env.task.success[0])
        self.reward, self._frames, self._spectator_frames = float(self.success), {}, {}
        self._rendered_s, self._fresh, self._followed = None, False, None
        if self.spectators and not self._spectators:
            self._open_spectators()
        if self.evaluator_rendering:
            # The episode's env.reset rendered the reset state, so its frames are current.
            self._fresh, self._rendered_s = True, 0.
            if self._spectators:
                # The spectators cut to the new pose: settle the path tracer there, as an
                # on-demand capture does after a cut.
                self._place(self._spectators)
                self._renders += render_for_capture(CUT_DELIVERIES, dlaa=False)
            if self.episode.lights:
                # The evaluator re-renders after the light sync so the first frames show it.
                self._renders += render_for_capture(RERENDERS, dlaa=False)

    @property
    def time_s(self):
        return self.steps * self.period

    def state(self):
        if self._frozen and self._observed is not None:
            return self._observed  # the robot's state is unreadable since the fault: the last valid one
        self._observed = self.embodiment.state(self.episode)
        return self._observed

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is done")
        try:
            self.success = self.episode.step(self.embodiment.action(self.episode, np.asarray(action, float)))
        except Exception as error:
            # A physics failure can raise inside OmniGibson's step (its assertion on the base's NaN orientation, after
            # PhysX's "Illegal BroadPhaseUpdateData"): with the robot's state invalid, that ends the episode, not the
            # world. Any other error is raised as before.
            try:
                fault = self._check_state()
            except Exception:
                fault = None
            if fault is None:
                raise
            self._fail(f"{fault}; the physics step failed ({type(error).__name__})")
        else:
            fault = self._check_state()
            if fault is not None:
                self._fail(fault)
        if not self._frozen:
            self._observe()
        self.steps, self._frames, self._spectator_frames, self._fresh = self.steps + 1, {}, {}, False
        self.reward = float(self.success)
        self.done = (self.steps >= self.max_steps or (self.terminate_on_success and self.success)
                     or self.fault is not None)
        if self.evaluator_rendering and self._spectators and not self._frozen:
            # While recording, render every step as the evaluator does, so spectator captures
            # keep the renderer's history.
            self._render()

    def _check_state(self):
        """The robot's state after a step: why it is invalid, which ends the episode, or None. Invalid is a joint
        position or velocity or the base's pose that is not finite (observations then freeze), a revolute joint more
        than REVOLUTE_END or a prismatic one more than its range width beyond its range. A valid state keeps
        state_notice, the joints more than JOINT_NOTICE beyond their ranges, else None."""
        names, lower, upper, end, ranged = self._ranges
        robot = self.episode.robot
        q, dq = values(robot.get_joint_positions()), values(robot.get_joint_velocities())
        self.state_notice = None
        broken = [names[i] for i in np.flatnonzero(~(np.isfinite(q) & np.isfinite(dq)))]
        if broken:
            self._frozen = True
            return f"the state of {', '.join(broken[:3])}{' and others' if len(broken) > 3 else ''} is not finite"
        try:
            position, frame = self.episode.base_pose()
            finite = bool(np.isfinite(position).all() and np.isfinite(frame).all())
        except AssertionError:  # OmniGibson asserts that the base's orientation is a unit quaternion
            finite = False
        if not finite:
            self._frozen = True
            return "the base's pose is not finite"
        excess = np.where(ranged, np.maximum(lower - q, q - upper), -np.inf)

        def beyond(i):
            return f"{names[i]} is at {q[i]:.3f}, {excess[i]:.3f} beyond its range [{lower[i]:.3f}, {upper[i]:.3f}]"
        ended = np.flatnonzero(excess > end)
        if ended.size:
            return beyond(ended[np.argmax(excess[ended] - end[ended])])
        noticed = np.flatnonzero(excess > JOINT_NOTICE)
        if noticed.size:
            self.state_notice = beyond(noticed[np.argmax(excess[noticed])]) + (
                f" ({noticed.size} joints are more than {JOINT_NOTICE:g} beyond their ranges)" if noticed.size > 1
                else "")
        return None

    def _fail(self, fault):
        self.fault = fault
        print(f"episode ended at step {self.steps + 1}: the robot's state became invalid: {fault}", file=sys.stderr,
              flush=True)

    def _observe(self):
        if getattr(self.embodiment, "attitude", False):
            self.base_attitude = self.episode.attitude()

    def measures(self):
        """The evaluator's partial credit, TaskMetric's q_score: 1 on success, else the best grounded goal
        option's fraction of literals that hold now and did not at the episode start."""
        from omnigibson.metrics.task_metric import compute_q_score

        now = self.episode.env.task.get_goal_option_satisfaction(0)
        return {"q_score": float(compute_q_score(self.success, now, self.initial_goal_options))}

    # Cameras ----------------------------------------------------------------------------

    def _sensor(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        return self.episode.robot.sensors[self.embodiment.cameras[camera]]

    def intrinsics(self, camera):
        """Isaac's symmetric pinhole with square pixels: the optical axis through the image
        center, in ROS pixel-center coordinates."""
        sensor = self._sensor(camera)
        width, height = sensor.image_width, sensor.image_height
        fx = float(sensor.focal_length) * width / float(sensor.horizontal_aperture)
        return width, height, fx, fx, (width - 1) / 2., (height - 1) / 2.

    def _camera_frame(self, camera):
        """A camera's USD frame (4x4) in the world, from its link's physics pose. A camera prim's
        own pose updates only when Kit renders, and physics steps without rendering."""
        if camera not in self._mounts:
            from pxr import UsdGeom

            sensor = self._sensor(camera)
            link = next(link for link in self.episode.robot.links.values()
                        if str(sensor.prim_path).startswith(str(link.prim_path) + "/"))
            local, _ = UsdGeom.XformCache().ComputeRelativeTransform(sensor.prim, link.prim)
            self._mounts[camera] = link, np.asarray(local, dtype=float).T
        link, local = self._mounts[camera]
        return pose_matrix(*link.get_position_orientation()) @ local

    def camera_pose(self, camera):
        """Optical-frame pose (4x4) in the world frame."""
        matrix = self._camera_frame(camera)
        matrix[:3, :3] = matrix[:3, :3] @ OPTICAL
        return matrix

    def _frame(self, camera):
        """RGB and z-depth of the current state; one render serves every camera until the next step."""
        if self._frozen:
            # Nothing renders a robot whose state is unreadable: the camera's last frame, else a blank one.
            sensor = self._sensor(camera)
            shape = (sensor.image_height, sensor.image_width)
            return self._held.get(camera) or (np.zeros((*shape, 3), np.uint8), np.zeros(shape, np.float32))
        if camera not in self._frames:
            sensor = self._sensor(camera)
            if not self._frames and not self._fresh:
                self._render()
            observation, _ = sensor.get_obs()
            rgb, depth = observation["rgb"], observation["depth_linear"]
            rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            shape = (sensor.image_height, sensor.image_width)
            if rgb.dtype != np.uint8 or rgb.shape[:2] != shape or values(depth).shape != shape:
                raise RuntimeError(f"{camera} produced no valid RGB-D frame")
            self._frames[camera] = self._held[camera] = (np.ascontiguousarray(rgb[..., :3]),
                                                         values(depth).astype(np.float32))
        return self._frames[camera]

    def _render(self):
        """Render until every camera holds the current state."""
        since = None if self._rendered_s is None else self.time_s - self._rendered_s
        if self.evaluator_rendering:
            deliveries = 1 if since is not None and since <= 1.5 * self.period else RERENDERS
            if self.episode.lights:
                deliveries += RERENDERS
        else:
            deliveries = 2 if since is not None and since <= HISTORY_S else CUT_DELIVERIES
        self._renders += render_for_capture(deliveries, dlaa=not self.evaluator_rendering)
        self._rendered_s, self._fresh = self.time_s, self.evaluator_rendering

    def render(self, camera):
        return self._frame(camera)[0]

    def render_depth(self, camera):
        """Metres along the optical axis (OmniGibson's depth_linear)."""
        return self._frame(camera)[1]

    def link_pose(self, link):
        """A robot link's world pose (4x4), for geometry checks."""
        return pose_matrix(*self.episode.robot.links[link].get_position_orientation())

    def object_joint(self, name, joint):
        """A scene object's joint position (rad or m), for physics checks."""
        state = self.episode.env.scene.object_registry("name", name).joints[joint].get_state()
        return float(values(state[0])[0])

    def drive_object_joint(self, name, joint, position=None, effort=0.):
        """For physics checks: put a scene object's joint at a position, at rest, if one is given, and hold a constant
        effort (N m or N) on it until the next call."""
        target = self.episode.env.scene.object_registry("name", name).joints[joint]
        if position is not None:
            target.set_pos(position)
            target.set_vel(0.)
        target.set_effort(effort)

    # Spectators: separate USD cameras and render products; agent cameras stay untouched.

    def _open_spectators(self):
        import omni.replicator.core as rep
        import omnigibson as og
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        stage, root = get_current_stage(), "/RosbridgeSpectator_" + secrets.token_hex(8)
        with og.sim.editing_usd():
            for view in self.spectators:
                path = f"{root}/{view.name}"
                camera = UsdGeom.Camera.Define(stage, path)
                if view.source == "base":
                    aperture = 20.
                    camera.GetVerticalApertureAttr().Set(aperture)
                    camera.GetHorizontalApertureAttr().Set(aperture * view.width / view.height)
                    camera.GetFocalLengthAttr().Set(aperture / (2 * np.tan(np.deg2rad(OVERVIEW_FOV_Y) / 2)))
                    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100))
                else:
                    source = UsdGeom.Camera.Get(stage, str(self._sensor(view.source).prim_path))
                    for attribute in ("projection", "focalLength", "horizontalAperture", "verticalAperture",
                                      "horizontalApertureOffset", "verticalApertureOffset", "clippingRange"):
                        value = source.GetPrim().GetAttribute(attribute).Get()
                        camera.GetPrim().GetAttribute(attribute).Set(value)
                product = rep.create.render_product(path, (view.width, view.height), force_new=True)
                rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                rgb.attach(product)
                self._spectators[view.name] = (view, camera, UsdGeom.Xformable(camera).AddTransformOp(), rgb)

    def _overview(self, bearing):
        """An overview camera (4x4) on `bearing` from the base's heading, pulled in front of
        anything between it and the robot."""
        from omnigibson.utils.sampling_utils import raytest

        view, robot = self.embodiment.overview, self.episode.robot
        position, quaternion = robot.base_footprint_link.get_position_orientation()
        target = np.r_[values(position)[:2], self.episode.origin[0][2] + view.target]
        angle = bearing + self._heading(quaternion)
        offset = np.array([view.distance * np.cos(angle), view.distance * np.sin(angle), view.eye - view.target])
        hit = raytest(target.tolist(), (target + offset).tolist(),
                      ignore_bodies=[link.prim_path for link in robot.links.values()])
        if hit["hit"]:
            offset *= max(0.01, float(hit["distance"]) - WALL_MARGIN) / np.linalg.norm(offset)
        # The ray passes windows and gaps in walls (an overview filmed a room through a window from outside the
        # house): keep the overview inside the robot's room, by the scene's room segmentation.
        rooms = self.episode.env.scene.seg_map
        room = rooms.get_room_instance_by_point(target[:2])  # None on a room boundary, such as a doorway
        if room is not None:
            def in_room(fraction):
                return rooms.get_room_instance_by_point(target[:2] + fraction * offset[:2]) == room
            offset *= inside(offset[:2], in_room, float(rooms.map_resolution) / 2)
        backward = offset / np.linalg.norm(offset)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = np.column_stack((right, np.cross(backward, right), backward)), target + offset
        return matrix

    def _heading(self, quaternion):
        """The base's yaw, smoothed while the overviews stream."""
        frame = rotation(quaternion)
        yaw = np.arctan2(frame[1, 0], frame[0, 0])
        if self._followed is not None and self.time_s - self._followed[0] <= FOLLOW_GAP_S:
            since, previous = self._followed
            turn = (yaw - previous + np.pi) % (2 * np.pi) - np.pi
            yaw = previous + turn * (1 - np.exp((since - self.time_s) / FOLLOW_S))
        self._followed = self.time_s, yaw
        return yaw

    def _place(self, names):
        import omnigibson as og
        from pxr import Gf

        for name in names:
            view, _, transform, _ = self._spectators[name]
            matrix = self._overview(OVERVIEWS[name]) if view.source == "base" else self._camera_frame(view.source)
            with og.sim.editing_usd():
                transform.Set(Gf.Matrix4d(matrix.T.tolist()))

    def render_spectator(self, name):
        if self._frozen:
            view = self._spectators[name][0]
            held = self._held.get(("spectator", name))
            return held if held is not None else np.zeros((view.height, view.width, 3), np.uint8)
        if not self.evaluator_rendering:
            self._place([name])
            self._render()
            frame = np.ascontiguousarray(np.asarray(self._spectators[name][3].get_data())[..., :3])
            self._held[("spectator", name)] = frame
            return frame
        if name not in self._spectator_frames:
            # The first view of a sample places every view, and one render serves them all.
            self._place(self._spectators)
            self._render()
            self._spectator_frames = {view: np.ascontiguousarray(np.asarray(rgb.get_data())[..., :3])
                                      for view, (_, _, _, rgb) in self._spectators.items()}
            self._held.update({("spectator", view): frame for view, frame in self._spectator_frames.items()})
        return self._spectator_frames[name]

    # Diagnostics for tests --------------------------------------------------------------

    def render_count(self):
        """Renders this backend requested; OmniGibson's own (as in env.reset) are not counted."""
        return self._renders

    def delivered_time(self):
        """World time, in Kit's reference clock, of the frame the cameras last delivered."""
        return delivered_time()

    def render_settings(self):
        return render_settings()

    def spectator_intrinsics(self, name):
        view, camera = self._spectators[name][:2]
        fx = camera.GetFocalLengthAttr().Get() * view.width / camera.GetHorizontalApertureAttr().Get()
        return fx, fx, (view.width - 1) / 2., (view.height - 1) / 2.

    def close(self):
        # Render products share the native graph; they end with Kit.
        self._spectators = {}
        if self.episode is not None:
            import omnigibson as og

            self.episode = None
            og.shutdown()


class R1ProSim(BehaviorSim):
    def __init__(self, **options):
        from .r1pro import R1Pro

        super().__init__(R1Pro(), **options)


class G1SharpaSim(BehaviorSim):
    simulator = "behavior-g1"
    check_task = staticmethod(check_g1_task)

    def __init__(self, **options):
        from .g1_sharpa import G1Sharpa

        super().__init__(G1Sharpa(), **options)


class R1ProSharpaSim(BehaviorSim):
    simulator = "behavior-dex"
    check_task = staticmethod(check_dex_task)

    def __init__(self, **options):
        from .r1pro_sharpa import R1ProSharpa

        super().__init__(R1ProSharpa(), **options)
