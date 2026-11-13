"""HandsOn tasks on the bimanual ROKAE AR5 rig with Wuji hands; runs in the native worker.

Isaac Sim 5.1 needs Python 3.11, so the world node proxies this class through
rosbridge.core.worker. A reset replays the PhysX branch of upstream's official
evaluator (handson/evaluation/native.py:run_episode): the frozen task contract, the
seeded setting, native construction, reset, the stable-origin settle gate and the
seeded instruction, written to PROMPT.json in the run directory. Every 1/30 s step
passes the hands' absolute joint targets through the native action path and replaces
only its palm-pose IK with the arms' absolute joint targets, so native hand slewing,
arm step limits and success and failure predicates apply. The palm pose of the native
action is the one each arm target commands, so native logic that reads the commanded
palm sees what an upstream policy's command would show it. Where upstream's control path
would send its IK another palm pose, a task's palm limiter (eval_flex02's hand-above-cloth
floor) or a native guard that rewrites the commanded palm (t10's carton lift guard), the arm
takes the joints nearest its target that put the palm there.

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
# Checklist rows that say a task has none.
NO_CHECKLIST = {"event_success_unavailable", "condition_panel_error"}
# Native info entries whose names contain these explain a failure (failure_reason, ycb01_failure_reason, ...).
FAILURE_INFO = ("fail", "reason")


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


def judge(failure, info, reason):
    """The trial's failure and success after a step, as upstream's evaluator scores the step that ends it.

    A failure latches at the first step whose native info says failed or whose adapter names a contact rule
    (t11), unless that step also succeeds: upstream ends the trial there and scores its success.
    """
    succeed = bool(info.get("succeed"))
    contact = reason not in (None, "native_terminated", "native_truncated")
    if failure is None and not succeed and (info.get("failed") or contact):
        failure = reason if contact else "native_failed"
    return failure, succeed and failure is None


def failure_detail(info):
    """The native info that explains a failure at its step: set scalar entries whose names contain fail or reason."""
    return {key: value[:200] if isinstance(value, str) else value for key, value in info.items()
            if any(word in key for word in FAILURE_INFO) and isinstance(value, (str, bool, int, float)) and value}


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


def pose_matrix_wxyz(pose):
    """A 4x4 pose from xyz and a wxyz quaternion."""
    w, x, y, z = np.asarray(pose[3:7], dtype=float) / np.linalg.norm(pose[3:7])
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = pose[:3]
    return matrix


def rotation_vector(matrix):
    """A rotation matrix's axis times its angle."""
    vee = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]) / 2
    sine = np.linalg.norm(vee)
    angle = np.arctan2(sine, (np.trace(matrix) - 1) / 2)
    return vee * (angle / sine) if sine > 1e-12 else vee


def reach(forward, target, seeds, lower, upper, tolerance=1e-6, iterations=100, step=1e-7):
    """Joints within the limits at which forward(joints), a 4x4 pose, is the 4x4 target within tolerance (metres,
    radians), by damped least squares from each seed in turn, so near the first seed that gets there; a seed already
    there is returned as it is. If none gets there, the joints that came nearest."""
    def error(joints):
        pose = forward(joints)
        return pose, np.r_[target[:3, 3] - pose[:3, 3], rotation_vector(target[:3, :3] @ pose[:3, :3].T)]

    nearest = None
    for seed in seeds:
        joints, damping = np.asarray(seed, dtype=float), 1e-2
        pose, residual = error(joints)
        for _ in range(iterations):
            if np.linalg.norm(residual[:3]) <= tolerance and np.linalg.norm(residual[3:]) <= tolerance:
                return joints
            jacobian = np.empty((6, joints.size))
            for j in range(joints.size):
                moved = forward(joints + step * np.eye(joints.size)[j])
                jacobian[:, j] = np.r_[moved[:3, 3] - pose[:3, 3],
                                       rotation_vector(moved[:3, :3] @ pose[:3, :3].T)] / step
            trial = np.clip(joints + jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping ** 2 * np.eye(6), residual), lower, upper)
            trial_pose, trial_residual = error(trial)
            if np.linalg.norm(trial_residual) < np.linalg.norm(residual):  # Levenberg-Marquardt damping
                joints, pose, residual, damping = trial, trial_pose, trial_residual, max(damping / 10, 1e-4)
            elif (damping := damping * 10) > 1e2:
                break
        if nearest is None or np.linalg.norm(residual) < nearest[0]:
            nearest = np.linalg.norm(residual), joints
    return nearest[1]


def limits_palm(env):
    """Whether upstream's control path passes a commanded palm through the task's _limit_arm_pose_target before its
    IK: with the Haorui IK, or where a task asks for PhysX contact limits (bimanual_assembly_env.py:4157-4161). The
    frozen Core classes ask for Haorui only and have no contact limits (core_box_base.py:1568-1569 and copies)."""
    contact = getattr(env, "_physx_contact_arm_target_limits_enabled", None)
    return env._arm_ik_backend == "haorui" or bool(contact and contact())


class Palm:
    """An arm's palm pose in the world at given arm joints: the pose of upstream's <side>_palm_link that its EEF52
    action commands, as xyz in metres and a wxyz quaternion.

    The kinematics are upstream's robot MJCF, the model Isaac imports for the environment's hand. The reset places its
    frame where the live palm and arm joints put it; the arms are fixed-base.
    """

    def __init__(self, env, side):
        import mujoco
        from handson.eef52 import measured_palm_pose
        from handson.evaluation.cameras import pose_matrix
        from handson.sim.isaaclab.bimanual_assembly_scene import ROBOT_MJCF_BY_HAND
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_MJCF_BY_HAND[env.hand_type][side]))
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(f"joint{i}_{side}").qposadr[0] for i in range(1, 8)]
        self.lower, self.upper = self.model.jnt_range[[self.model.joint(f"joint{i}_{side}").id
                                                       for i in range(1, 8)]].T
        self.body = self.model.body(f"{side}_palm_link").id
        pose, _ = measured_palm_pose(env, side)
        joints = values(env._robots[side].data.joint_pos[0])[env._arm_joint_ids[side]]
        self.world = pose_matrix(pose[:3], pose[3:]) @ np.linalg.inv(self.model_pose(joints))

    def model_pose(self, joints):
        """The palm's 4x4 pose in the MJCF's frame."""
        import mujoco
        self.data.qpos[self.joints] = joints
        mujoco.mj_kinematics(self.model, self.data)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = self.data.xmat[self.body].reshape(3, 3), self.data.xpos[self.body]
        return matrix

    def __call__(self, joints):
        from handson.sim.isaaclab.bimanual_assembly_env import _matrix_to_quat_wxyz
        matrix = self.world @ self.model_pose(joints)
        return np.r_[matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])]

    def reach(self, pose, seeds):
        """Arm joints within the joint limits that put the palm at pose, near the first seed that gets there, else
        the joints that come nearest."""
        target = np.linalg.solve(self.world, pose_matrix_wxyz(pose))
        return reach(self.model_pose, target, seeds, self.lower, self.upper)


class HandsOnSim:
    default_task = "t24_place_ball_in_bowl"
    period = 1 / 30

    def __init__(self, *, max_steps=None, terminate_on_success=False, terminate_on_failure=None, partial_credit=None):
        self.requested_max_steps = self.max_steps = max_steps
        # A run created to stop on success ends at its first outcome, success or failure, as upstream's trial does;
        # one created to stop on failure ends at a failure only (upstream's evaluator also ends the trial there) and
        # is judged on its final state otherwise.
        # Stop on failure is HandsOn's own option (rosbridge passes no other): ROSBRIDGE_HANDSON_STOP_ON_FAILURE=1.
        if terminate_on_failure is None:
            terminate_on_failure = os.environ.get("ROSBRIDGE_HANDSON_STOP_ON_FAILURE") == "1"
        self.terminate_on_success, self.terminate_on_failure = terminate_on_success, bool(terminate_on_failure)
        self.failure_step = None
        # Optional partial credit from upstream's success checklist; off unless ROSBRIDGE_HANDSON_PARTIAL_CREDIT=1.
        if partial_credit is None:
            partial_credit = os.environ.get("ROSBRIDGE_HANDSON_PARTIAL_CREDIT") == "1"
        self.partial_credit, self._initial, self._frozen = bool(partial_credit), None, None
        self.failure_detail = None
        # The palm each arm target commands, and the steps whose arm target upstream's palm limits or guards changed.
        self._commanded, self.guarded, self._guarded_at = dict.fromkeys(SIDES), dict.fromkeys(SIDES, 0), {}
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
        # Arms take absolute joint targets in place of the palm-pose IK; the rest of the native path stays, and the
        # native action's palm pose is the one the target commands.
        self._arm = {side: env._joint_targets[side][:, env._arm_joint_ids[side]].clone() for side in SIDES}
        self._palm = {side: Palm(env, side) for side in SIDES}
        self._limits_palm = limits_palm(env)
        driven = set()  # the arms whose targets upstream's control path applies

        def target(side, robot, pose, commanded):
            driven.add(side)
            return self._arm_target(side, robot, pose, commanded), "rosbridge"
        env._compute_arm_joint_target = target
        # A single-arm task's control path may hold the left arm and hand at home without asking for a target
        # (bimanual_assembly_env.py:3641-3662). Applying the current targets once, without a physics step, shows
        # which arms it drives; the others' devices refuse commands.
        env._apply_joint_targets()
        self._index(inactive=set(SIDES) - driven)
        self.steps, self.done, self.success, self.reward = 0, False, False, 0.
        if self.spectators:
            self._open_spectators()

    def _arm_target(self, side, robot, pose, commanded):
        """The arm's drive target in place of upstream's IK: the commanded joints, unless upstream's path would give
        its IK another palm than they command. That palm is the task's _limit_arm_pose_target of the commanded one
        where upstream applies it (eval_flex02 lifts it so the hand stays above the cloth), else the native command
        when native logic rewrote it after the action arrived (t10 pins its height until both hands grip). The arm
        then takes the joints nearest its target that put the palm there (else nearest its previous target), or, if
        none do, the joints that come nearest, as upstream's IK steps toward a palm it cannot reach."""
        target, palm = self._arm[side], self._commanded[side]
        if not commanded or palm is None:  # nothing commanded yet: upstream's IK idles too
            return target
        env = self.env
        if self._limits_palm:
            pose = env._limit_arm_pose_target(side, robot, pose)
        elif np.array_equal(pose, palm):
            return target
        joints = values(target)[0]
        previous = values(env._joint_targets[side])[0, [int(i) for i in env._arm_joint_ids[side]]]
        reached = self._palm[side].reach(pose, (joints, previous))
        if np.array_equal(reached, joints):
            return target
        if self._guarded_at.get(side) != self.steps:  # a native guard may re-solve the targets within a step
            if not self.guarded[side]:
                print(f"[rosbridge] {side} arm: upstream's palm {'limits' if self._limits_palm else 'guard'} changed "
                      f"its target at step {self.steps + 1}", flush=True)
            self._guarded_at[side] = self.steps
            self.guarded[side] += 1
        import torch
        return torch.tensor(reached[None], device=robot.device, dtype=robot.data.joint_pos.dtype)

    def _index(self, inactive=()):
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
                                           tuple(limits[ids, 0]), tuple(limits[ids, 1]), inactive=(
                                               "upstream's single-arm control holds it at home"
                                               if side in inactive else "")))
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
            self._arm[side] = torch.tensor(arm[None], device=robot.device, dtype=robot.data.joint_pos.dtype)
            # Native logic reads the commanded palm (t16's hinge opens only while it rises); IK does not see it.
            command[side] = np.r_[self._palm[side](arm), hand]
            self._commanded[side] = command[side][:7]
        _, _, terminated, truncated, info = self.env.step(command)
        self.steps += 1
        reason = self._adapter.terminal_reason(dict(terminated=terminated, truncated=truncated, info=info),
                                               env=self.env)
        self._settle(info, reason)

    def _settle(self, info, reason):
        """Score the step just taken, end the episode where the run stops, and read the checklist if asked."""
        failed_before = self.failure is not None
        # Upstream ends the trial at a failure, so it can no longer succeed.
        self.failure, self.success = judge(self.failure, info, reason)
        self.reward = float(self.success)
        self.done = self.steps >= self.max_steps or (
            self.terminate_on_success and (self.success or self.failure is not None)) or (
            self.terminate_on_failure and self.failure is not None)
        if self.failure is not None and not failed_before:
            self.failure_detail, self.failure_step = failure_detail(info), self.steps
        if self.partial_credit:
            if self._initial is None:
                self._initial = self._checklist()
            if self.failure is not None and not failed_before:
                self._frozen = self._checklist()

    def end_reason(self):
        """A run created to stop on success or on failure ends at a failure, as upstream's evaluator does."""
        if (self.terminate_on_success or self.terminate_on_failure) and self.failure is not None:
            return "failure", self.failure
        return None

    def _checklist(self):
        """Upstream's success checklist (its teleoperation panel's rows) as {key: ok}."""
        return {row["key"]: bool(row["ok"]) for row in self.env._success_condition_rows()
                if row["key"] not in NO_CHECKLIST}

    def measures(self):
        """The latched failure, its native detail and the step at which it latched; the steps whose arm target
        upstream's palm limits or guards changed, if any; with partial credit on, the checklist at the trial's end and
        its q_score, as BEHAVIOR's TaskMetric computes it over one goal option, the checklist.

        The trial ends at the failure step if a failure latched, else at the last step. q_score is 1 on success, else
        the fraction of all conditions that hold at the trial's end and did not after the first step (0 without any).
        """
        result = {"failure": self.failure}
        if self.failure is not None:
            result["failure_detail"], result["failure_step"] = self.failure_detail, self.failure_step
        if any(self.guarded.values()):
            result["arm_guard_steps"] = dict(self.guarded)
        if not self.partial_credit:
            return result
        conditions = self._frozen if self._frozen is not None else self._checklist() if self.steps else None
        initial = self._initial or {}
        newly = sum(ok and not initial.get(key, True) for key, ok in (conditions or {}).items())
        score = 1. if self.success else newly / len(conditions) if conditions else 0.
        return dict(result, conditions=conditions, conditions_initial=self._initial, q_score=score)

    def link_pose(self, side, link):
        """A robot body's world pose (rotation, translation), for geometry checks."""
        robot = self.env._robots[side]
        state = robot.data.body_state_w[0, list(robot.body_names).index(link)]
        matrix = world_pose(state[:3], state[3:7])
        return matrix[:3, :3], matrix[:3, 3]

    def commanded_palm(self, side):
        """The palm pose upstream last took as commanded for an arm (rotation, translation), for geometry checks."""
        from handson.evaluation.cameras import pose_matrix
        pose = values(getattr(self.env, f"_{side}_tcp_pose"))
        matrix = pose_matrix(pose[:3], pose[3:7])
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
