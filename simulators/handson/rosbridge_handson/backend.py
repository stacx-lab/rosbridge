"""HandsOn tasks on the bimanual ROKAE AR5 rig with Wuji hands; runs in the native worker.

Isaac Sim 5.1 needs Python 3.11, so the world node proxies this class through
rosbridge.core.worker. A launch's seed k names the task's k-th native seed that a
pre-screen accepted (seeds.json, written by tools/handson_seeds.py). A reset replays the
PhysX branch of upstream's official evaluator (handson/evaluation/native.py:run_episode)
at that native seed: the frozen task contract, the seeded setting, native construction,
reset, the stable-origin settle gate and the seeded instruction, written to PROMPT.json
in the run directory. Every 1/30 s step passes upstream's own command: each arm's palm pose
target, which upstream's EEF52 actions command (the <side>_palm_link world pose), and the
hand's absolute joint targets. The task's own arm IK, palm-pose limits and guards, hand
slewing and success and failure predicates apply; an arm not yet commanded is upstream's idle.
An early-stop run ends where the evaluator ends a trial (evaluation/rollout.py): at its
first success, or at a task failure the native environment or the evaluator reports.

Cameras render through upstream's three-view capture as the evaluator renders them: the
task's camera profile, with the head camera's calibrated lens, and the episode's drawn mount
offsets (rosbridge_handson.cameras).
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
from .cameras import CAMERAS, OPENGL_TO_OPTICAL, camera_seed, contract, distortion, lens, offsets
from .seeds import screened_seed

SIDES = ("right", "left")
SPECTATORS = (("third_person", "head_camera"), ("left_wrist", "left_camera"),
              ("right_wrist", "right_camera"))
# Each path-traced pass renders every enabled render product completely; frames reach an
# annotator a pass or more later. Upstream's capture renders six passes per view.
RENDER_PASSES, SPECTATOR_PASSES = 6, 2
# Native info entries whose names contain these explain a failure (failure_reason, ycb01_failure_reason, ...).
FAILURE_INFO = ("fail", "reason")


class SettleGateRejected(RuntimeError):
    """Upstream's stable-origin settle gate rejected a reset. Deterministic for a task and native seed."""


def values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def world_pose(position, quaternion):
    """A 4x4 world pose from an Isaac Lab position and quaternion."""
    from handson.evaluation.cameras import pose_matrix
    from handson.sim.isaaclab.compat import quat_isaaclab_to_wxyz
    return pose_matrix(values(position), quat_isaaclab_to_wxyz(values(quaternion)))


def matrix(pose):
    """The 4x4 transform of a pose [x, y, z, qx, qy, qz, qw]."""
    from scipy.spatial.transform import Rotation
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    return result


def pose(transform):
    """[x, y, z, qx, qy, qz, qw] of a 4x4 transform."""
    from scipy.spatial.transform import Rotation
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]


def hydra_texture(product):
    """A render product's HydraTexture, as upstream's role_render_products finds it."""
    from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager
    manager = ViewportManager()
    return manager._hydra_textures.get(manager._context, product).hydra_texture


def activate_cameras(task):
    """The task's camera contract, active for the episode."""
    from handson.evaluation import camera_contracts
    value = contract(task)
    camera_contracts._active.set(value)  # authors the lens and checks it in render_probe, as use(task) does
    return value


def failure_cause(cause, info):
    """A task failure's cause and the native info of its step that explains it: set scalar entries whose names
    contain fail or reason, such as "native_failed: ycb01_failure_reason=ball_or_hand_touched_gate"."""
    detail = [f"{key}={str(value)[:200]}" for key, value in info.items()
              if key != "failed" and any(word in key for word in FAILURE_INFO)
              and isinstance(value, (str, bool, int, float)) and value is not False and value != ""]
    return cause + (": " + ", ".join(detail) if detail else "")


def native_stop(returned, reason, timed_out):
    """Where the evaluator ends a trial after a step (rollout.py): None to go on, else the record's reason and
    cause. A step that succeeds is a success, as the evaluator scores its info["succeed"], also when it fails."""
    info = returned["info"]
    if info.get("succeed"):
        return "success", None
    if reason not in (None, "native_terminated", "native_truncated"):
        return "failure", reason  # an evaluator rule, such as t11's direct hand contact
    if info.get("failed"):
        return "failure", failure_cause("native_failed", info)
    if reason == "native_terminated" and not timed_out:
        # t17 publishes the frame's failed over the latched one, so its trial can end with neither flag.
        return "failure", failure_cause("native_terminated", info)
    return None


class HandsOnSim:
    default_task = "t24_place_ball_in_bowl"
    period = 1 / 30
    # A three-view capture costs seconds, so a subscribed camera renders twice per simulated second, not at 30 Hz.
    camera_rate_hz = 2.

    def __init__(self, *, max_steps=None, early_stop=False):
        self.requested_max_steps = self.max_steps = max_steps
        self.early_stop = early_stop
        self.cameras = tuple(CAMERAS)
        self.env = self.task = self.stop = self.native_seed = self._tracker = self._frames = self._temporary = None
        self.spectators, self._spectators = (), {}

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in CAMERAS})

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    # Episode lifecycle ------------------------------------------------------------------

    def reset(self, task, seed, randomize=True):
        """The episode at the task's seed-th screened native seed, as the official evaluator builds it; randomize
        has no effect. Isaac hosts one episode per process.
        """
        task = resolve("handson", task=task)[2]["native_task"]
        native_seed = screened_seed(task, seed)
        try:
            self._reset(task, seed, native_seed)
        except SettleGateRejected as error:
            # The screen accepted this native seed, so the gate decides differently on this machine.
            raise SettleGateRejected(f"seed {seed} names native seed {native_seed}, which the pre-screen accepted, "
                                     f"but the settle gate rejected it here: {error}") from error

    def _reset(self, task, seed, native_seed):
        """The native episode at native_seed, recorded as seed; tools/handson_seeds.py screens native seeds with it.

        The settle gate may reject the native seed: SettleGateRejected.
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

        self.task, self.seed, self.native_seed = resolve("handson", task=task)[2]["native_task"], seed, native_seed
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
        self._contract = activate_cameras(self.task)
        lenses = {camera: distortion(self._contract["cameras"][role]) for camera, role in CAMERAS.items()}
        self.distortion = {camera: d for camera, d in lenses.items() if d is not None}
        install_launcher("physx")
        install_native_camera_creation(HandsOnABCIsaacLabEnv)
        self._adapter.prepare_cameras()
        task_contracts.install_native_trace(self.task, out)
        self._adapter.prepare(out)
        selection = setting_runtime.prepare(self.task, native_seed, out)
        setting_runtime.configure_solver(selection)
        reset_seed = selection["seeds"]["reset"] if selection else native_seed
        # The episode's camera draw, as run_episode seeds render_probe.
        self._camera_seed = selection["seeds"]["camera"] if selection else native_seed
        assert self._camera_seed == camera_seed(self.task, native_seed)
        self._offsets = offsets(self.task, self._camera_seed)
        self.env, self._wrapped, _ = native.construct(self.task, reset_seed, selection=selection)
        env = self.env
        if not env._cameras:
            env._init_recording_cameras()
        observation, info = env.reset(seed=reset_seed)
        try:
            task_contracts.initialize(env, self._wrapped, observation, info, contract, out)
        except Exception as error:
            settle = out / "TASK_BOUNDARY_SETTLE.json"
            if type(error).__name__ == "RecordingOriginStabilityError" or (
                    settle.is_file() and not json.loads(settle.read_text())["passed"]):
                # Upstream's stable-origin gate rejected this reset; its evaluator reports the rejection and
                # substitutes no other seed. It is deterministic for a task and native seed.
                raise SettleGateRejected(f"{self.task} native seed {native_seed}: {str(error)[:300]}") from error
            raise
        (out / "PROMPT.json").write_text(json.dumps(bind(env, self.task, native_seed), indent=2) + "\n")
        if self._adapter.spec.force_scoring:  # t10 scores support, lift and hold; force is diagnostic
            from handson.evaluation.carton import install
            self._tracker = install(env, out / "CARTON_RAW_FORCE_000")
        budgets = json.loads(Path(native.__file__).with_name("budgets.json").read_text())
        self.period = float(env._control_dt)
        self.max_steps = self.requested_max_steps or int(budgets[self.task]["steps"])
        env._time_limit = self.max_steps * self.period
        env.hz = 0  # the world paces wall time; upstream would sleep to 30 Hz
        self._index()
        # Arm pose targets are palm poses in the arm base (link0), which Isaac places in the world here.
        self._base = {side: self._body(side, f"link0_{side}") for side in SIDES}
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
                                           tuple(limits[ids, 0]), tuple(limits[ids, 1]),
                                           tip="palm" if part == "arm" else None))
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

    def step(self, action, poses):
        """action: the hands' joint targets; poses: each arm's palm pose target in its base, or None."""
        if self.done:
            raise RuntimeError("episode is done")
        action, command = np.asarray(action, dtype=float), {}
        for side in SIDES:
            target = poses[f"{side}_arm"]
            palm = np.zeros(7)  # upstream's "not commanded"
            if target is not None:
                world = pose(self._base[side] @ matrix(target))
                # Upstream's commands are xyz and quaternion wxyz with w >= 0, as EEF52 decodes a rotation vector.
                palm = np.r_[world[:3], np.copysign(1., world[6]) * world[[6, 3, 4, 5]]]
            command[side] = np.r_[palm, action[self._slices[side][1]]]
        _, _, terminated, truncated, info = self.env.step(command)
        self.steps += 1
        self.success = bool(info.get("succeed"))
        self.reward = float(self.success)
        if self.early_stop:
            returned = dict(terminated=terminated, truncated=truncated, info=info)
            self.stop = native_stop(returned, self._adapter.terminal_reason(returned, env=self.env),
                                    self.steps >= self.max_steps)
        self.done = self.steps >= self.max_steps or self.stop is not None

    def measures(self):
        return {"native_seed": self.native_seed}

    def tip_pose(self, group):
        """An arm's measured palm pose [x, y, z, qx, qy, qz, qw] in its base."""
        from handson.eef52 import measured_palm_pose
        from handson.evaluation.cameras import pose_matrix
        side = group.removesuffix("_arm")
        palm, _ = measured_palm_pose(self.env, side)
        return pose(np.linalg.inv(self._base[side]) @ pose_matrix(palm[:3], palm[3:]))

    def held_arms(self):
        """The arms whose targets upstream's control path never asks for in this task, for checks: it applies the
        current targets once, without a physics step."""
        raw, asked = getattr(self.env, "_raw", self.env), set()  # t22's evaluator steps a view of its environment
        compute = raw._compute_arm_joint_target

        def probe(side, *args):
            asked.add(side)
            return compute(side, *args)
        raw._compute_arm_joint_target = probe
        try:
            raw._apply_joint_targets()
        finally:
            del raw._compute_arm_joint_target
        return [side for side in SIDES if side not in asked]

    def _body(self, side, link):
        robot = self.env._robots[side]
        state = robot.data.body_state_w[0, list(robot.body_names).index(link)]
        return world_pose(state[:3], state[3:7])

    def link_pose(self, side, link):
        """A robot body's world pose (rotation, translation), for geometry checks."""
        transform = self._body(side, link)
        return transform[:3, :3], transform[:3, 3]

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
            # It places each camera at its drawn mount, as at the episode's camera seed.
            self._frames = self.steps, render_probe(self.env, {"task": self.task}, self._out, 0, self._camera_seed,
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
        """A view's OpenGL world pose, as upstream's capture places it: nominal mount and drawn offset."""
        from handson.eef52 import measured_palm_pose
        from handson.evaluation.cameras import pose_matrix
        spec = self._contract["cameras"][CAMERAS[camera]]
        if camera == "head_camera":
            return np.asarray(spec["nominal_world_pose_opengl"]) @ self._offsets[camera]
        pose, _ = measured_palm_pose(self.env, camera.removesuffix("_camera"))
        return (pose_matrix(pose[:3], pose[3:]) @ np.asarray(spec["nominal_camera_to_palm_opengl"])
                @ self._offsets[camera])

    def render_spectators(self):
        """Each view in passes of its own: sharing them between views measured no faster here."""
        return {name: self._render_spectator(name) for name in self._spectators}

    def _render_spectator(self, name):
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
