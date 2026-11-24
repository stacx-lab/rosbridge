"""RoboDojo tasks on the native Isaac Sim dual ARX X5 rig; runs in the native worker.

Isaac Sim 5.1 needs Python 3.11, so the world node proxies this class through
rosbridge.core.worker. Every 0.04 s frame passes absolute arm joints and the
native normalized gripper value to upstream's EvalEnv.take_action, which
interpolates the arm over its ten 250 Hz substeps. Each step of an arm's move
goal becomes joints through the cuRobo IK upstream's ee_pose action uses,
checked as uni-robot-agent's RoboDojo adapter checks it. Resets load
upstream's saved evaluation layouts. Ported from RoboEnv's RoboDojo adapter.
"""
import os
import secrets
import sys

import numpy as np

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.project import project_root
from .assets import check_task

CAMERAS = {"head_camera": "cam_head", "left_camera": "cam_left_wrist",
           "right_camera": "cam_right_wrist"}
SIDES = ("left", "right")
SPECTATORS = (("third_person", "head_camera"), ("left_wrist", "left_camera"),
              ("right_wrist", "right_camera"))
UNSTABLE_RETRIES = 3
# An arm's goals are poses of its flange, link6: upstream's ee_link and its cuRobo tool frame.
TIP = "flange"
# A pose solution's joint path from the measured joints is checked against cuRobo's self and table constraints at
# this spacing, as uni-robot-agent's RoboDojo adapter checks it.
PATH_SAMPLE_RAD = .025
# The most a goal's step may ask of a joint from where it is, uni-robot-agent's guard against a change of posture.
# Beyond a frame's native travel (0.2 rad) the actuator saturates, and the next step starts from where the arm got.
JUMP_RAD = 1.
_APP = None  # Isaac allows one SimulationApp per process.


def runtime():
    return project_root() / ".runtime/robodojo"


def boot():
    """Start headless Isaac Sim once, before any RoboDojo env.* or task.* import."""
    global _APP
    if _APP is not None:
        return _APP
    # torch's bundled libgomp needs static TLS that Kit's plugins would otherwise exhaust.
    import torch  # noqa: F401
    root = str(runtime())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # EvalEnv keys resume state and temporary video directories by this ID.
    os.environ.setdefault("ROBODOJO_RUN_ID", f"rosbridge-{secrets.token_hex(8)}")
    from rosbridge.core.gpu import single_gpu
    renderer = single_gpu()
    from isaaclab.app import AppLauncher

    class SingleGPU(AppLauncher):
        def _config_resolution(self, launcher_args):
            super()._config_resolution(launcher_args)
            # AppLauncher would otherwise use the CUDA ordinal as the Vulkan index.
            self._sim_app_config.update(active_gpu=renderer, physics_gpu=0, multi_gpu=False,
                                        max_gpu_count=1)

    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    argv, sys.argv = sys.argv, sys.argv[:1]  # AppLauncher parses sys.argv
    try:
        _APP = SingleGPU(device="cuda:0", headless=True, enable_cameras=True, kit_args=(
            "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
            + zero_delay_kit_args())).app
    finally:
        sys.argv = argv
    return _APP


class NoModelClient:
    """Stands in for XPolicyLab's policy client: devices are the only control surface."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def checks(spec):
    """A reward check, or an "or" list of them, as words: is_lift(label=t, z_threshold=0.01)."""
    if isinstance(spec, tuple):
        name, arguments = spec
        return f"{name}({', '.join(f'{key}={value}' for key, value in arguments.items())})"
    return " or ".join(checks(item) for item in spec)


def violations(manager):
    """The constraints an episode exceeded: a query (or triggered query) that held more often than allowed."""
    queries = [(query, aim, count) for query, aim, count in manager.query_list[0]]
    queries += [(query, aim, count) for _, query, _, _, aim, count in manager.trigger_query_list[0]]
    found = [f"{' and '.join(checks(c) for c in query)} held on {count} step{'s' * (count != 1)}, at most {aim}"
             for query, aim, count in queries if count > aim]
    return "; ".join(found)


def values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def nearest_turn(q, previous, lower, upper):
    """q with each joint moved by whole turns toward previous where its limits allow."""
    tau = 2 * np.pi
    turns = np.clip(np.round((previous - q) / tau), np.ceil((lower - q) / tau), np.floor((upper - q) / tau))
    return q + tau * np.where(np.isfinite(turns), turns, 0.)


def path(previous, q):
    """The joint samples from previous (excluded) to q at most PATH_SAMPLE_RAD apart."""
    count = max(1, int(np.ceil(np.max(np.abs(q - previous)) / PATH_SAMPLE_RAD)))
    return np.linspace(previous, q, count + 1)[1:]


def choose(candidates, previous, lower, upper, step_limit, path_free, names):
    """The candidate nearest previous that a step can take: within [lower, upper] (a whole turn nearer previous
    where the limits allow), moving no joint more than step_limit from previous, along a joint path path_free(rows)
    accepts. Returns (joints, None), or (None, why the nearest candidate cannot be taken; None without one)."""
    nearest = None
    turned = [nearest_turn(np.asarray(q, dtype=float), previous, lower, upper) for q in candidates]
    for q in sorted(turned, key=lambda q: np.abs(q - previous).max()):
        steps = np.abs(q - previous)
        i = int(np.argmax(steps))
        if np.any(q < lower) or np.any(q > upper):
            why = "lies outside the joint limits"
        elif steps[i] > step_limit:
            why = f"moves {names[i]} {q[i] - previous[i]:+.2f} rad in one step, a change of posture"
        elif not path_free(path(previous, q)):
            why = "has a joint path that collides with the arm itself or the table"
        else:
            return q, None
        nearest = nearest or why
    return None, nearest


def config(task, early_stop, env_cfg_type="arx_x5", depth=False):
    """Upstream src/eval_client/main.py configuration for one environment; with depth, every camera also
    captures upstream's distance_to_image_plane, as env_cfg/camera/camera_config.yml shows commented out."""
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from task.RoboDojo import task_registry
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, env_cfg_type + ".yml"))
    eval_cfg.update(task_name=task, num_envs=1, device_id=0, eval_batch=False,
                    policy_name="rosbridge", additional_info="", seed=0,
                    physx_monitor_enabled=False, early_stop=early_stop)
    run = os.environ["ROBODOJO_RUN_ID"]
    deploy_cfg = {"policy_name": "rosbridge", "port": 0, "host": "localhost", "protocol": "ws",
                  "policy_server_url": "ws://localhost:0", "evaluation_id": run,
                  "trial_id": f"{task}-{run}", "action_case_id": f"{task}_case",
                  "repeat_index": None}
    sections = {name: load_yaml(os.path.join(ENV_CONFIG_PATH, name, eval_cfg["config"][name] + ".yml"))
                for name in ("sim", "scene", "camera", "robot")}
    benchmark = os.path.join(ROOT_DIR, "task", BENCHMARK, "config")
    cfg = OmegaConf.create({**sections,
                            "task_env": load_yaml(task_registry.task_config_path(benchmark, task)),
                            "eval_cfg": eval_cfg, "deploy_cfg": deploy_cfg})
    OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(cfg, "eval_cfg.num_envs", 1, force_add=True)
    cfg = process_randomization(cfg)
    cfg, eval_num = process_config(cfg, task_name=task)
    eval_cfg["eval_num"] = eval_num
    for camera in CAMERAS.values() if depth else ():
        # After process_config, which may load a task's own camera configuration.
        OmegaConf.update(cfg, f"camera.annotator.{camera}.distance_to_image_plane_capture",
                         {"type": "distance_to_image_plane", "device": "cpu"}, force_add=True)
    OmegaConf.update(cfg, "camera.default_frequency",
                     eval_cfg["observation"].get("collect_freq", 0), force_add=True)
    cfg.sim.seed = [0]
    cfg.eval_cfg.config_name = env_cfg_type
    for robot in cfg.robot.robots:
        # Rounded source quaternions such as [0.707, 0, 0, 0.707] are not unit length;
        # PhysX and URDF readers would normalize them differently.
        quaternion = np.asarray(robot["default_root_rot"], dtype=float)
        robot["default_root_rot"] = (quaternion / np.linalg.norm(quaternion)).tolist()
    return cfg


class RoboDojoSim:
    default_task = "pack_objects_into_box"
    width, height = 640, 480

    def __init__(self, *, max_steps=None, early_stop=False, configuration=None):
        if configuration not in (None, "depth", "move"):
            raise ValueError(f"RoboDojo has no configuration {configuration!r}")
        self.depth = configuration == "depth"  # RGB-D cameras
        self.move = configuration == "move"  # the arms take flange goals in place of joint targets
        self._captures = None  # each camera's annotator outputs at the current step; None until it is rendered
        self.requested_max_steps = max_steps
        self.max_steps = max_steps
        self.early_stop = early_stop
        self.stop = None
        self.cameras = tuple(CAMERAS)
        self.env = None
        self.task = None
        self.spectators, self._spectators = (), {}

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in CAMERAS},
                                depth=tuple(CAMERAS) if self.depth else ())

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    # Episode lifecycle ------------------------------------------------------------------

    def reset(self, task, seed, randomize=True):
        """Load a saved native layout, selected by the seed; randomize has no effect.

        Isaac hosts one environment per process, so a reset reloads the live env. A
        failure after the reload starts leaves the episode done until a reset succeeds.
        """
        spec = resolve("robodojo", task=task)[2]
        native = spec["native_task"]
        check_task(runtime(), native)
        if self.env is not None and native != self.task:
            raise ValueError("a RoboDojo worker serves one task; start another run for another task")
        if self.env is None:
            boot()
            import src.eval_client.eval_env as eval_env
            self.task = native
            self.env = eval_env.create_eval_env(config(native, self.early_stop, depth=self.depth), _APP,
                                                model_client=NoModelClient())
            self.interval = int(self.env.obs_manager.collect_interval)
            self.physics_dt = float(self.env.dt)
            self.period = self.interval * self.physics_dt
            self.native_step_limit = int(self.env.step_lim)
        layouts = list(getattr(self.env.seed_manager, "seed_list", None)
                       or sorted(getattr(self.env.seed_manager, "seed_info", {})))
        if not layouts:
            raise ValueError(f"{native} has no saved evaluation layouts")
        self.done, self._captures = True, None
        layout = int(layouts[int(seed) % len(layouts)])
        from utils.cluttered_generator import UnStableError
        for attempt in range(UNSTABLE_RETRIES):
            self.env.env_seeds = [layout]
            self.env.reward_manager.diagnostic_observer = None
            try:
                self.env.reset(seed=[layout])
                break
            except UnStableError:
                # Reloading a layout after another one occasionally fails the native settle
                # check; the same layout settles on a retry, so the seed keeps its layout.
                if attempt == UNSTABLE_RETRIES - 1:
                    raise
        if not np.isclose(self.env.sim.physics_dt, self.physics_dt, rtol=0, atol=1e-12):
            raise ValueError("native physics timestep differs from the control clock")
        for hook in ("run_reward", "get_score"):
            if callable(getattr(self.env, hook, None)):
                getattr(self.env, hook)()
        self.seed, self.layout_id = seed, layout
        # The layout's instruction pair as upstream's policy receives it: the observation's instruction and its
        # extra_info's detailed instruction, a description of the task and its scoring.
        manager = self.env.obs_manager
        self.description = {"instruction": manager.instruction[0],
                            "detailed_instruction": manager.detailed_instruction[0]}
        self.max_steps = self.requested_max_steps or self.native_step_limit
        # EvalEnv checks its own counter in take_action; keep both limits equal.
        self.env.step_lim = self.max_steps
        self._index()
        self.steps, self.done, self.stop = 0, False, None
        self._score()
        if self.spectators and not self._spectators:
            self._open_spectators()

    def _index(self):
        manager = self.env.robot_manager
        names, groups, self._actuators, self._limits = [], [], {}, {}
        for side in SIDES:
            robot = manager.get_robot_by_arm_name(f"{side}_arm")
            if robot.type != "target":
                raise ValueError("native support arms are not devices")
            key = manager.robot_key[manager.robot_list.index(robot)]
            arm = list(robot.arm_joint_indices)
            limits = values(key.data.joint_pos_limits[0])[arm]
            self._actuators[side] = (robot, key, arm, list(robot.gripper_joint_indices))
            # One frame's joint travel at the native velocity limit (upstream's 5 rad/s actuators), for the rig.
            step = float(np.min(values(key.data.joint_vel_limits[0])[arm])) * self.period
            self._limits[side] = (limits[:, 0], limits[:, 1], step)
            offset = len(names)
            names += list(robot.arm_joints_name) + [f"{side}_gripper"]
            # Joint targets as upstream's arm_joint_state, or flange goals through upstream's cuRobo IK.
            groups.append(CommandGroup(f"{side}_arm", tuple(range(offset, offset + len(arm))), tuple(limits[:, 0]),
                                       tuple(limits[:, 1]), tip=TIP, joint_targets=True, move=self.move))
            groups.append(CommandGroup(f"{side}_gripper", (offset + len(arm),), (0.,), (1.,), scalar=True))
        self.layout = ActionLayout(tuple(names), tuple(groups))

    @property
    def time_s(self):
        return self.steps * self.period

    def _fraction(self, robot, primary):
        low, high = robot.gripper_scale
        fraction = (primary - low) / (high - low)
        return fraction if robot.gripper_move["sign"] == 1 else 1 - fraction

    def state(self):
        """Arm joints in radians; grippers as the native normalized opening of the primary finger."""
        positions, velocities = [], []
        for side in SIDES:
            robot, key, arm, fingers = self._actuators[side]
            qpos, qvel = values(key.data.joint_pos[0]), values(key.data.joint_vel[0])
            low, high = robot.gripper_scale
            sign = 1 if robot.gripper_move["sign"] == 1 else -1
            positions += [*qpos[arm], self._fraction(robot, qpos[fingers[0]])]
            velocities += [*qvel[arm], sign * qvel[fingers[0]] / (high - low)]
        return np.asarray(positions), np.asarray(velocities)

    def step(self, action):
        import torch
        if self.done:
            raise RuntimeError("episode is done")
        action = np.asarray(action, dtype=float)
        manager, native = self.env.robot_manager, {}
        for side, offset in zip(SIDES, (0, 7)):
            robot, key, arm, fingers = self._actuators[side]
            joints, grip = action[offset:offset + len(arm)], float(action[offset + len(arm)])
            for name, target, indices in ((robot.arm_name, list(joints), arm),
                                          (robot.gripper_name, [grip], fingers)):
                command = manager.process_name(name)
                native[command] = list(target)
                positions = list(target)
                if indices is fingers:
                    low, high = robot.gripper_scale
                    primary = low + (grip if robot.gripper_move["sign"] == 1 else 1 - grip) * (high - low)
                    _, multiplier, offset_m = robot.gripper_move["mimic"]
                    positions = [primary, primary * multiplier + offset_m]
                # Hold targets so a stopped device keeps them; EvalEnv interpolates from here.
                tensor = torch.tensor([positions], device=key.device, dtype=key.data.joint_pos.dtype)
                key.set_joint_position_target(tensor, joint_ids=indices)
                key.set_joint_velocity_target(torch.zeros_like(tensor), joint_ids=indices)
                manager.control_manager.prev_control[0][command] = {"position": positions, "velocity": 0}
        self.env.take_action(native)
        self.steps, self._captures = self.steps + 1, None
        self._score()
        if self.early_stop and self.env.end_flag[0]:
            # EvalEnv ends at its completion, or at a failed constraint (patch 0001 keeps both for early_stop).
            # A failure before the step limit, or one that names a constraint at it; else the limit ended it.
            found = violations(self.env.reward_manager)
            if self.success:
                self.stop = ("success", None)
            elif found or self.steps < self.max_steps:
                self.stop = ("failure", found or "a task constraint failed")
        self.done = bool(self.env.end_flag[0] or self.steps >= self.max_steps)

    # Pose targets -----------------------------------------------------------------------

    def _link(self, side, link):
        """A robot link's pose in the environment frame as a 4x4 transform."""
        robot = self._actuators[side][0]
        pose = values(self.env.robot_manager.get_link_pose(robot, link, [0], is_relative=True)[0])
        import transforms3d as t3d
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = t3d.quaternions.quat2mat(pose[3:7] / np.linalg.norm(pose[3:7])), pose[:3]
        return matrix

    def tip_pose(self, group):
        """The measured flange (link6) pose in <arm>/base, [x, y, z, qx, qy, qz, qw]."""
        import transforms3d as t3d
        side = group.removesuffix("_arm")
        relative = np.linalg.inv(self._link(side, "base_link")) @ self._link(side, "link6")
        w, x, y, z = t3d.quaternions.mat2quat(relative[:3, :3])
        return np.r_[relative[:3, 3], x, y, z, w]

    def _native_pose(self, side, pose):
        """A flange pose in <arm>/base as upstream's <side>_ee_pose: the link6 pose in the environment frame,
        [x, y, z, qw, qx, qy, qz], as its observation reports it."""
        import transforms3d as t3d
        x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
        relative = np.eye(4)
        relative[:3, :3], relative[:3, 3] = t3d.quaternions.quat2mat([w, x, y, z]), pose[:3]
        world = self._link(side, "base_link") @ relative
        return np.r_[world[:3, 3], t3d.quaternions.mat2quat(world[:3, :3])]

    def joint_step(self, group, pose):
        """The joints for the next step of a goal: cuRobo's IK solves the flange pose from the measured joints, and
        its solution nearest them must lie within the joint limits and JUMP_RAD of them, along a joint path clear
        of cuRobo's self and table constraints. A ValueError says why there is none."""
        side = group.removesuffix("_arm")
        robot, key, arm, _ = self._actuators[side]
        planner = self.env.robot_manager.ik_solver[robot.robot_name]
        lower, upper, _ = self._limits[side]
        joints = values(key.data.joint_pos[0])[arm]
        target = self._native_pose(side, np.asarray(pose, dtype=float))
        q, why = choose(self._candidates(planner, robot, target, joints), joints, lower, upper, JUMP_RAD,
                        lambda rows: self._path_free(planner, rows), list(robot.arm_joints_name))
        if q is None:
            raise ValueError(f"the nearest IK solution {why}" if why else
                             "cuRobo's IK has no solution clear of the arm itself and the table")
        return q

    def _candidates(self, planner, robot, target, current):
        """cuRobo's successful IK solutions for an ee_pose target, built as upstream's
        CuroboPlanner.solve_ik_to_joint builds its goal, from the joints current."""
        from utils.transformer import calculate_target_pose
        goal = calculate_target_pose(robot.entity_origin_pose, planner.robot_origin_pose, list(target))
        position, rotation = planner._trans_from_world_to_base(
            np.asarray(planner.robot_origin_pose, dtype=np.float32), np.asarray(goal, dtype=np.float32))
        position = position + np.asarray(planner.frame_bias, dtype=position.dtype)
        result = planner.ik_solver.solve_pose(goal_tool_poses=planner._build_goal_pose(position, rotation),
                                              current_state=planner._build_joint_state(current), return_seeds=32)
        solution = result.js_solution
        names = list(solution.joint_names)
        joints = values(solution.position).reshape(-1, len(names))[:, [names.index(n) for n in
                                                                        planner.active_joints_name]]
        success = values(result.success).reshape(-1) > 0
        return [q for q, ok in zip(joints, success) if ok]

    @staticmethod
    def _path_free(planner, rows):
        """Whether cuRobo's IK constraints (self collision, the table, joint limits) hold at every joint row, in
        batches of the IK's 32 seeds so its rollout keeps its shape."""
        import torch
        solver = planner.ik_solver
        for start in range(0, len(rows), 32):
            chunk = np.stack([planner._build_cspace_joint_values(q) for q in rows[start:start + 32]])
            padded = np.concatenate([chunk, np.repeat(chunk[-1:], 32 - len(chunk), axis=0)])
            action = torch.as_tensor(padded, device=planner.device_cfg.device, dtype=torch.float32)[:, None, :]
            feasible = solver.metrics_rollout.compute_metrics_from_action(action).costs_and_constraints.get_feasible(
                include_all_hybrid=False, sum_horizon=True)
            if not (feasible if isinstance(feasible, bool) else bool(values(feasible).reshape(32, -1)[:len(chunk)].all())):
                return False
        return True

    def _score(self):
        self.success = bool(self.env.completed[0] and self.env.success[0])
        # Upstream's episode score (EvalEnv.run_eval): 1 for a success; a failed graded task
        # keeps its native 0..100 score; other failures score 0.
        graded = getattr(self.env.reward_manager, "_gated_score_lst", None)
        raw = 100. if self.success else float(graded[0]) if graded else 0.
        self.reward = min(1., max(0., raw / 100.))

    def link_pose(self, side, link):
        """A robot link's world pose (rotation, translation), for geometry checks."""
        robot = self._actuators[side][0]
        pose = values(self.env.robot_manager.get_link_pose(robot, link, [0])[0])  # x y z qw qx qy qz
        w, x, y, z = pose[3:7] / np.linalg.norm(pose[3:7])
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return rotation, pose[:3]

    # Cameras ----------------------------------------------------------------------------

    def _camera(self, name):
        manager = self.env.camera_manager
        return manager.cameras[0][manager.camera_names[0].index(CAMERAS[name])]

    def intrinsics(self, camera):
        """Isaac's symmetric pinhole: optical axis through the image center, ROS pixel centers."""
        native = self._camera(camera)
        width, height = native.get_resolution()
        fx = float(native.get_focal_length()) * width / float(native.get_horizontal_aperture())
        return width, height, fx, fx, (width - 1) / 2., (height - 1) / 2.

    def camera_pose(self, camera):
        """Optical-frame pose (pos, quat_wxyz) in the world frame."""
        position, quaternion = self._camera(camera).get_world_pose(camera_axes="ros")
        return values(position), values(quaternion)

    def _render(self):
        """Render the current step once: one native render fills every camera's and every spectator view's
        buffers, so the first read of a step places the spectator cameras and renders, and the others reuse it."""
        if self._captures is not None:
            return
        from pxr import Gf
        for view, transform, _ in self._spectators.values():
            position, q = self._camera(view.source).get_world_pose(camera_axes="usd")
            w, x, y, z = values(q)
            matrix = np.eye(4)
            matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
            matrix[:3, 3] = values(position)
            transform.Set(Gf.Matrix4d(matrix.T.tolist()))
        self.env.obs_manager.render_for_capture()
        self._captures = {}

    def _capture(self, camera, annotator, dtype):
        """A camera's annotator output at the current step."""
        if camera not in CAMERAS:
            raise KeyError(camera)
        self._render()
        if camera not in self._captures:
            # RoboDojo owns tiled render products and their annotators.
            index = self.env.camera_manager.cameras[0].index(self._camera(camera))
            self._captures[camera] = self.env.capture_manager.step(env_ids=[0], cam_ids=[index])[0]
        frame = np.asarray(self._captures[camera][annotator][0]["data"])
        if frame.shape[:2] != (self.height, self.width) or frame.dtype != dtype:
            raise RuntimeError(f"{camera} produced no valid {annotator} frame")
        return frame

    def render(self, camera):
        return np.ascontiguousarray(self._capture(camera, "rgb", np.uint8)[..., :3])

    def render_depth(self, camera):
        """Metres along the optical axis: upstream's distance_to_image_plane capture."""
        if not self.depth:
            raise KeyError(camera)
        return np.ascontiguousarray(self._capture(camera, "distance_to_image_plane", np.float32)[..., 0])

    # Spectators: separate USD cameras and render products that follow the observation mounts.

    def _open_spectators(self):
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        stage = get_current_stage()
        root = "/RosbridgeSpectator_" + secrets.token_hex(8)
        for view in self.spectators:
            source = UsdGeom.Camera.Get(stage, str(self._camera(view.source).prim_path))
            camera = UsdGeom.Camera.Define(stage, f"{root}/{view.name}")
            for attribute in ("projection", "focalLength", "horizontalAperture", "verticalAperture",
                              "horizontalApertureOffset", "verticalApertureOffset", "clippingRange"):
                camera.GetPrim().GetAttribute(attribute).Set(source.GetPrim().GetAttribute(attribute).Get())
            product = rep.create.render_product(f"{root}/{view.name}", (view.width, view.height),
                                                force_new=True)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach(product)
            self._spectators[view.name] = (view, UsdGeom.Xformable(camera).AddTransformOp(), rgb)

    def render_spectators(self):
        self._render()
        return {name: np.ascontiguousarray(np.asarray(rgb.get_data())[..., :3])
                for name, (_, _, rgb) in self._spectators.items()}

    def spectator_intrinsics(self, name):
        view = self._spectators[name][0]
        source = self._camera(view.source)
        fx = float(source.get_focal_length()) * view.width / float(source.get_horizontal_aperture())
        return fx, fx, (view.width - 1) / 2., (view.height - 1) / 2.

    def close(self):
        global _APP
        # Render products share the native graph; they end with the environment.
        self._spectators = {}
        if self.env is not None:
            self.env.close()
            self.env = None
        if _APP is not None:
            _APP.close()
            _APP = None
