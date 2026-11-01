"""BEHAVIOR-1K activities on an embodiment in OmniGibson; runs in the native worker.

OmniGibson needs Isaac Sim 5.1 on Python 3.11, so the world node proxies this class through
rosbridge.core.worker. The episode follows the official evaluator (environment.py); an
embodiment maps rosbridge channels to native actions (r1pro.py, r1pro_sharpa.py, g1_sharpa.py).
"""
import os
import secrets
from typing import NamedTuple

import numpy as np

from rosbridge.catalog import resolve
from rosbridge.core.devices import SpectatorView, device_inventory
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
HISTORY_S, CUT_DELIVERIES = .25, int(os.environ.get("ROSBRIDGE_BEHAVIOR_CUT_RENDERS", 16))
# (After a gap, depth is exact from the first render; RGB keeps converging: mean absolute difference from a 48-render
# image, 0-255 scale, 6.0 after 1 render, 2.9 after 4, 1.2 after 16; a render costs about 20 ms, so a capture after a
# gap costs 0.5-0.9 s at 16. ROSBRIDGE_BEHAVIOR_CUT_RENDERS trades that noise for time.)
# Overview spectators keep these bearings from the robot's heading (radians counterclockwise).
OVERVIEWS = {"front_left": np.pi / 4, "rear_left": 3 * np.pi / 4}
FOLLOW_S = 1.  # the overviews turn after the robot this slowly, so gait sway does not shake them
FOLLOW_GAP_S = .25  # after a longer pause in the overviews, as after a cut, they face the heading


class Overview(NamedTuple):
    """An embodiment's overview spectators: `distance` m out horizontally from the base, with
    their eye and the point they look at `eye` and `target` m above the base's height at the
    episode start."""
    distance: float
    eye: float
    target: float


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

    @property
    def cameras(self):
        return tuple(self.embodiment.cameras)

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in self.cameras}, depth=self.cameras)

    def configure_spectators(self, width=960, height=720):
        # Overviews that follow the base, and the head camera's view.
        self.spectators = (*(SpectatorView(name, "base", width, height) for name in OVERVIEWS),
                           SpectatorView("head", "head_camera", width, height))

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
        self.done = True
        self.episode.load(seed)
        self.embodiment.reset(self.episode)
        self.episode.origin = self.episode.base_pose()
        self.seed, self.steps, self.done = seed, 0, False
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
        return self.embodiment.state(self.episode)

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is done")
        self.success = self.episode.step(self.embodiment.action(self.episode, np.asarray(action, float)))
        self.steps, self._frames, self._spectator_frames, self._fresh = self.steps + 1, {}, {}, False
        self.reward = float(self.success)
        self.done = self.steps >= self.max_steps or (self.terminate_on_success and self.success)
        if self.evaluator_rendering and self._spectators:
            # While recording, render every step as the evaluator does, so spectator captures
            # keep the renderer's history.
            self._render()

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
            self._frames[camera] = (np.ascontiguousarray(rgb[..., :3]), values(depth).astype(np.float32))
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
            offset *= max(0.01, float(hit["distance"]) - 0.2) / np.linalg.norm(offset)
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
        if not self.evaluator_rendering:
            self._place([name])
            self._render()
            return np.ascontiguousarray(np.asarray(self._spectators[name][3].get_data())[..., :3])
        if name not in self._spectator_frames:
            # The first view of a sample places every view, and one render serves them all.
            self._place(self._spectators)
            self._render()
            self._spectator_frames = {view: np.ascontiguousarray(np.asarray(rgb.get_data())[..., :3])
                                      for view, (_, _, _, rgb) in self._spectators.items()}
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
