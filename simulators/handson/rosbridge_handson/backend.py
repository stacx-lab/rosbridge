"""HandsOn tasks on the bimanual ROKAE AR5 rig with Wuji hands; runs in the native worker.

Isaac Sim 5.1 needs Python 3.11, so the world node proxies this class through
rosbridge.core.worker. A reset replays the PhysX branch of upstream's official
evaluator (handson/evaluation/native.py:run_episode): the frozen task contract, the
seeded setting, native construction, reset, the stable-origin settle gate and the
seeded instruction, written to PROMPT.json in the run directory. Every 1/30 s step
passes the hands' absolute joint targets through the native action path and replaces
only its palm-pose IK with the arms' absolute joint targets, so native hand slewing,
arm step limits and success and failure predicates apply.

Cameras render through upstream's three-view capture, with two contract choices: the
default camera profile for every task, the head camera undistorted (its calibrated K
kept), and nominal mounts without the per-seed jitter.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views

SIDES = ("right", "left")
CAMERAS = {"head_camera": "main", "left_camera": "wrist_left", "right_camera": "wrist_right"}
SPECTATORS = (("third_person", "head_camera"), ("left_wrist", "left_camera"),
              ("right_wrist", "right_camera"))
OPENGL_TO_OPTICAL = np.diag([1., -1., -1., 1.])
# Each path-traced pass renders every enabled render product completely; frames reach an
# annotator a pass or more later. Upstream's capture renders six passes per view.
RENDER_PASSES, SPECTATOR_PASSES = 6, 2


def values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def world_pose(position, quaternion):
    """A 4x4 world pose from an Isaac Lab position and quaternion."""
    from handson.evaluation.cameras import pose_matrix
    from handson.sim.isaaclab.compat import quat_isaaclab_to_wxyz
    return pose_matrix(values(position), quat_isaaclab_to_wxyz(values(quaternion)))


def hydra_texture(product):
    """A render product's HydraTexture, as upstream's role_render_products finds it."""
    from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager
    manager = ViewportManager()
    return manager._hydra_textures.get(manager._context, product).hydra_texture


def lens(spec):
    """A contract camera's width, height, fx, fy, cx, cy: its native K, with the image's pixel centers."""
    (fx, _, cx), (_, fy, cy), _ = spec["K_effective"]
    width, height = spec["native_width_height"]
    return width, height, float(fx), float(fy), float(cx), float(cy)


def contract_cameras():
    """The default camera profile for every task, with the head camera's distortion zeroed."""
    from handson.evaluation import camera_contracts, camera_randomization
    from handson.sim.isaaclab import calibrated_cameras
    contract = camera_contracts.current()  # the default profile while none is active
    main = contract["cameras"]["main"]
    main["D_effective"] = [0.] * len(main["D_effective"])
    camera_contracts._active.set(contract)  # authors the lens and checks it in render_probe
    profile = calibrated_cameras.profile  # checked by verify_environment
    calibrated_cameras.profile = lambda: {**profile(), "runtime_D": [0.] * len(profile()["runtime_D"])}
    sample = camera_randomization.sample

    def nominal(*args):
        draw = sample(*args)
        for camera in draw["cameras"].values():
            camera.update(translation_m=[0.] * 3, quaternion_wxyz=[1., 0., 0., 0.], roll_pitch_yaw_deg=[0.] * 3)
        return {**draw, "enabled": False}
    camera_randomization.sample = nominal
    return contract


class HandsOnSim:
    default_task = "t24_place_ball_in_bowl"
    period = 1 / 30

    def __init__(self, *, max_steps=None, terminate_on_success=False):
        self.requested_max_steps = self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success
        self.cameras = tuple(CAMERAS)
        self.env = self.task = self.failure = self._tracker = self._frames = self._temporary = None
        self.spectators, self._spectators = (), {}

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in CAMERAS})

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    # Episode lifecycle ------------------------------------------------------------------

    def reset(self, task, seed, randomize=True):
        """The seeded native episode, as the official evaluator builds it; randomize has no effect.

        Isaac hosts one episode per process. The settle gate may reject a seed; the error ends the launch.
        """
        if self.env is not None:
            raise ValueError("a HandsOn worker serves one episode; start another run for another")
        import torch  # noqa: F401  torch's libgomp needs static TLS that Kit's plugins would exhaust
        from handson.evaluation import backends, native, setting_runtime, task_contracts
        from handson.evaluation.cameras import install_launcher, install_native_camera_creation
        from handson.evaluation.goals import bind
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        from handson.tasks.registry import TASKS
        from rosbridge.core.gpu import single_gpu

        self.task, self.seed = resolve("handson", task=task)[2]["native_task"], seed
        # Upstream's receipts and PROMPT.json stay with a managed run; tests use a temporary directory.
        run = os.environ.get("ROSBRIDGE_RUN_DIR")
        if not run:
            self._temporary = tempfile.TemporaryDirectory(prefix="rosbridge-handson-")
            run = self._temporary.name
        self._out = out = Path(run) / "handson"
        out.mkdir(parents=True)
        self._adapter = TASKS[self.task].evaluation_adapter()
        contract = task_contracts.configure(self.task)  # replaces every HANDSON_* variable
        backends.configure("physx")
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.update(HANDSON_BACKGROUND_MODE="off", HANDSON_ISAACLAB_CACHE=str(out / "scene-cache"),
                          HANDSON_ISAACLAB_DEFER_CAMERA_RENDER="0",
                          HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS="1",
                          HANDSON_ISAACSIM_RENDERER_GPU=str(single_gpu()))
        self._contract = contract_cameras()
        install_launcher("physx")
        install_native_camera_creation(HandsOnABCIsaacLabEnv)
        self._adapter.prepare_cameras()
        task_contracts.install_native_trace(self.task, out)
        self._adapter.prepare(out)
        selection = setting_runtime.prepare(self.task, seed, out)
        setting_runtime.configure_solver(selection)
        reset_seed = selection["seeds"]["reset"] if selection else seed
        self.env, self._wrapped, _ = native.construct(self.task, reset_seed, selection=selection)
        env = self.env
        if not env._cameras:
            env._init_recording_cameras()
        observation, info = env.reset(seed=reset_seed)
        task_contracts.initialize(env, self._wrapped, observation, info, contract, out)
        (out / "PROMPT.json").write_text(json.dumps(bind(env, self.task, seed), indent=2) + "\n")
        if self._adapter.spec.force_scoring:  # t10 scores support, lift and hold; force is diagnostic
            from handson.evaluation.carton import install
            self._tracker = install(env, out / "CARTON_RAW_FORCE_000")
        budgets = json.loads(Path(native.__file__).with_name("budgets.json").read_text())
        self.period = float(env._control_dt)
        self.max_steps = self.requested_max_steps or int(budgets[self.task]["steps"])
        env._time_limit = self.max_steps * self.period
        env.hz = 0  # the world paces wall time; upstream would sleep to 30 Hz
        self._index()
        # Arms take absolute joint targets in place of the palm-pose IK; the rest of the native path stays.
        self._arm = {side: env._joint_targets[side][:, env._arm_joint_ids[side]].clone() for side in SIDES}
        env._compute_arm_joint_target = lambda side, robot, pose, commanded: (self._arm[side], "rosbridge")
        self.steps, self.done, self.success, self.reward = 0, False, False, 0.
        if self.spectators:
            self._open_spectators()

    def _index(self):
        names, groups, self._ids, self._slices = [], [], {}, {}
        for side in SIDES:
            robot = self.env._robots[side]
            joints, limits = list(robot.joint_names), values(robot.data.joint_pos_limits[0])
            self._ids[side], self._slices[side] = [], []
            for part, ids in (("arm", self.env._arm_joint_ids[side]), ("hand", self.env._hand_joint_ids[side])):
                ids, offset = [int(i) for i in ids], len(names)
                names += [joints[i] for i in ids]
                self._ids[side] += ids
                self._slices[side].append(slice(offset, len(names)))
                groups.append(CommandGroup(f"{side}_{part}", tuple(range(offset, len(names))),
                                           tuple(limits[ids, 0]), tuple(limits[ids, 1])))
        self.layout = ActionLayout(tuple(names), tuple(groups))

    @property
    def time_s(self):
        return self.steps * self.period

    def state(self):
        """Measured arm and hand joints in radians, right side first."""
        robots = [self.env._robots[side] for side in SIDES]
        return tuple(np.concatenate([values(getattr(robot.data, field)[0])[self._ids[side]]
                                     for side, robot in zip(SIDES, robots)])
                     for field in ("joint_pos", "joint_vel"))

    def step(self, action):
        import torch
        if self.done:
            raise RuntimeError("episode is done")
        action, command = np.asarray(action, dtype=float), {}
        for side in SIDES:
            robot, (arm, hand) = self.env._robots[side], (action[part] for part in self._slices[side])
            self._arm[side] = torch.tensor([arm], device=robot.device, dtype=robot.data.joint_pos.dtype)
            command[side] = np.r_[np.zeros(7), hand]  # a zero palm pose is "not commanded"
        _, _, terminated, truncated, info = self.env.step(command)
        self.steps += 1
        reason = self._adapter.terminal_reason(dict(terminated=terminated, truncated=truncated, info=info),
                                               env=self.env)
        contact = reason not in (None, "native_terminated", "native_truncated")
        if self.failure is None and (info.get("failed") or contact):
            # Upstream ends the trial at a failure, so it can no longer succeed.
            self.failure = reason if contact else "native_failed"
        self.success = bool(info.get("succeed")) and self.failure is None
        self.reward = float(self.success)
        self.done = self.steps >= self.max_steps or (self.terminate_on_success and self.success)

    def measures(self):
        return {"failure": self.failure}

    def link_pose(self, side, link):
        """A robot body's world pose (rotation, translation), for geometry checks."""
        robot = self.env._robots[side]
        state = robot.data.body_state_w[0, list(robot.body_names).index(link)]
        matrix = world_pose(state[:3], state[3:7])
        return matrix[:3, :3], matrix[:3, 3]

    # Cameras ----------------------------------------------------------------------------

    def _camera(self, camera):
        from handson.evaluation.cameras import SENSORS
        return self.env._cameras[SENSORS[CAMERAS[camera]]]

    def intrinsics(self, camera):
        return lens(self._contract["cameras"][CAMERAS[camera]])

    def camera_pose(self, camera):
        """The rendered optical-frame pose (position, quaternion wxyz) in the world."""
        from scipy.spatial.transform import Rotation
        positions, quaternions = self._camera(camera)._view.get_world_poses()  # OpenGL camera axes
        matrix = world_pose(positions[0], quaternions[0]) @ OPENGL_TO_OPTICAL
        return matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def render(self, camera):
        if camera not in CAMERAS:
            raise KeyError(camera)
        if self._frames is None or self._frames[0] != self.steps:
            # One native capture renders all three views; it never steps physics.
            from handson.evaluation.cameras import render_probe
            from handson.evaluation.native import capture
            self._frames = self.steps, render_probe(self.env, {"task": self.task}, self._out, 0, self.seed,
                                                    "rosbridge", capture, save_png=False)
        return np.ascontiguousarray(self._frames[1][CAMERAS[camera]])

    # Spectators: separate USD cameras and render products at the observation mounts.

    def _open_spectators(self):
        import omni.replicator.core as rep
        from pxr import UsdGeom
        for view in self.spectators:
            source = UsdGeom.Camera(self._camera(view.source)._sensor_prims[0].GetPrim())
            path = f"/World/RosbridgeSpectators/{view.name}"
            camera = UsdGeom.Camera.Define(self.env._stage, path)
            aperture = source.GetHorizontalApertureAttr().Get()
            camera.GetHorizontalApertureAttr().Set(aperture)
            camera.GetVerticalApertureAttr().Set(aperture * view.height / view.width)
            camera.GetFocalLengthAttr().Set(source.GetFocalLengthAttr().Get())
            camera.GetClippingRangeAttr().Set(source.GetClippingRangeAttr().Get())
            product = rep.create.render_product(path, (view.width, view.height), force_new=True)
            # A spectator renders only during its own capture. Attached while disabled, its annotator
            # keeps its render var across toggles (upstream's prepare_toggle_safe_annotators).
            texture = hydra_texture(product.path)
            texture.set_updates_enabled(False)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach(product)
            self._spectators[view.name] = (view, camera, UsdGeom.Xformable(camera).AddTransformOp(),
                                           texture, rgb)

    def _mount(self, camera):
        """A view's OpenGL world pose, as upstream's capture places it."""
        from handson.eef52 import measured_palm_pose
        from handson.evaluation.cameras import pose_matrix
        spec = self._contract["cameras"][CAMERAS[camera]]
        if camera == "head_camera":
            return np.asarray(spec["nominal_world_pose_opengl"])
        pose, _ = measured_palm_pose(self.env, camera.removesuffix("_camera"))
        return pose_matrix(pose[:3], pose[3:]) @ np.asarray(spec["nominal_camera_to_palm_opengl"])

    def render_spectator(self, name):
        from handson.evaluation.cameras import role_render_products
        from pxr import Gf
        view, _, transform, texture, rgb = self._spectators[name]
        transform.Set(Gf.Matrix4d(self._mount(view.source).T.tolist()))
        # Only this view renders. Wrist views hide the wrist's own obstructions, as upstream's capture does.
        roles = role_render_products(self.env._cameras).values()
        for role in roles:
            role.set_updates_enabled(False)
        texture.set_updates_enabled(True)
        self.env._set_wrist_obstructions_hidden(view.source != "head_camera")
        try:
            for passes in range(1, RENDER_PASSES + 1):
                self.env._sim.render()
                frame = np.asarray(rgb.get_data())
                if passes >= SPECTATOR_PASSES and frame.shape[:2] == (view.height, view.width):
                    break
        finally:
            self.env._set_wrist_obstructions_hidden(False)
            texture.set_updates_enabled(False)
            for role in roles:
                role.set_updates_enabled(True)
        return np.ascontiguousarray(frame[..., :3])

    def spectator_intrinsics(self, name):
        view, camera = self._spectators[name][:2]
        fx = camera.GetFocalLengthAttr().Get() * view.width / camera.GetHorizontalApertureAttr().Get()
        return fx, fx, (view.width - 1) / 2., (view.height - 1) / 2.

    def close(self):
        self._spectators = {}
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        # Closing the SimulationApp ends this process, so files go first: the episode's converted
        # scene (about 0.5 GB) is needed only while it runs.
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        elif self.env is not None:
            shutil.rmtree(self._out / "scene-cache", ignore_errors=True)
        if self.env is not None:
            self._wrapped.close()  # also closes the SimulationApp the environment launched
            self.env = self._wrapped = None
