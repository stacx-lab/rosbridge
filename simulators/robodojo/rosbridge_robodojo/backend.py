"""RoboDojo tasks on the native Isaac Sim dual ARX X5 rig; runs in the native worker.

Isaac Sim 5.1 needs Python 3.11, so the world node proxies this class through
rosbridge.core.worker. Every 0.04 s frame passes absolute arm joints and the
native normalized gripper value to upstream's EvalEnv.take_action, which
interpolates the arm over its ten 250 Hz substeps. Resets load upstream's saved
evaluation layouts. Ported from RoboEnv's RoboDojo adapter.
"""
import os
import secrets
import sys

import numpy as np

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import SpectatorView, device_inventory
from rosbridge.project import project_root
from .assets import check_task

CAMERAS = {"head_camera": "cam_head", "left_camera": "cam_left_wrist",
           "right_camera": "cam_right_wrist"}
SIDES = ("left", "right")
SPECTATORS = (("third_person", "head_camera"), ("left_wrist", "left_camera"),
              ("right_wrist", "right_camera"))
UNSTABLE_RETRIES = 3
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


def values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def config(task, stop_on_success, env_cfg_type="arx_x5"):
    """Upstream src/eval_client/main.py configuration for one environment."""
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from task.RoboDojo import task_registry
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, env_cfg_type + ".yml"))
    eval_cfg.update(task_name=task, num_envs=1, device_id=0, eval_batch=False,
                    policy_name="rosbridge", additional_info="", seed=0,
                    physx_monitor_enabled=False, stop_on_success=stop_on_success)
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
    OmegaConf.update(cfg, "camera.default_frequency",
                     eval_cfg["observation"].get("collect_freq", 0), force_add=True)
    cfg.sim.seed = [0]
    cfg.eval_cfg.config_name = env_cfg_type
    for robot in cfg.robot.robots:
        # Rounded source quaternions such as [0.707, 0, 0, 0.707] are not unit length;
        # PhysX and URDF readers would normalize them differently.
        quaternion = np.asarray(robot["default_root_rot"], dtype=float)
        robot["default_root_rot"] = (quaternion / np.linalg.norm(quaternion)).tolist()
        if robot.get("type", "target") == "target":
            robot["need_planner"] = False
    return cfg


class RoboDojoSim:
    default_task = "pack_objects_into_box"
    width, height = 640, 480

    def __init__(self, *, max_steps=None, terminate_on_success=False):
        self.requested_max_steps = max_steps
        self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success
        self.cameras = tuple(CAMERAS)
        self.env = None
        self.task = None
        self.spectators, self._spectators = (), {}

    @property
    def devices(self):
        return device_inventory(self.layout, {name: name for name in CAMERAS})

    def configure_spectators(self, width=960, height=720):
        self.spectators = tuple(SpectatorView(role, source, width, height) for role, source in SPECTATORS)

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
            self.env = eval_env.create_eval_env(config(native, self.terminate_on_success), _APP,
                                                model_client=NoModelClient())
            self.interval = int(self.env.obs_manager.collect_interval)
            self.physics_dt = float(self.env.dt)
            self.period = self.interval * self.physics_dt
            self.native_step_limit = int(self.env.step_lim)
        layouts = list(getattr(self.env.seed_manager, "seed_list", None)
                       or sorted(getattr(self.env.seed_manager, "seed_info", {})))
        if not layouts:
            raise ValueError(f"{native} has no saved evaluation layouts")
        self.done = True
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
        self.max_steps = self.requested_max_steps or self.native_step_limit
        # EvalEnv checks its own counter in take_action; keep both limits equal.
        self.env.step_lim = self.max_steps
        self._index()
        self.steps, self.done = 0, False
        self._score()
        if self.spectators and not self._spectators:
            self._open_spectators()

    def _index(self):
        manager = self.env.robot_manager
        names, groups, self._actuators = [], [], {}
        for side in SIDES:
            robot = manager.get_robot_by_arm_name(f"{side}_arm")
            if robot.type != "target":
                raise ValueError("native support arms are not devices")
            key = manager.robot_key[manager.robot_list.index(robot)]
            arm = list(robot.arm_joint_indices)
            limits = values(key.data.joint_pos_limits[0])[arm]
            self._actuators[side] = (robot, key, arm, list(robot.gripper_joint_indices))
            offset = len(names)
            names += list(robot.arm_joints_name) + [f"{side}_gripper"]
            groups.append(CommandGroup(f"{side}_arm", tuple(range(offset, offset + len(arm))),
                                       tuple(limits[:, 0]), tuple(limits[:, 1])))
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
        self.steps += 1
        self._score()
        self.done = bool(self.env.end_flag[0] or self.steps >= self.max_steps
                         or (self.terminate_on_success and self.success))

    def _score(self):
        self.success = bool(self.env.completed[0] and self.env.success[0])
        # Graded tasks keep the native 0..100 score; others are binary.
        graded = getattr(self.env.reward_manager, "_gated_score_lst", None)
        raw = float(graded[0]) if graded else 100. * self.success
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

    def render(self, camera):
        if camera not in CAMERAS:
            raise KeyError(camera)
        self.env.obs_manager.render_for_capture()
        # RoboDojo owns tiled render products and their RGB annotators.
        index = self.env.camera_manager.cameras[0].index(self._camera(camera))
        frame = np.asarray(self.env.capture_manager.step(env_ids=[0], cam_ids=[index])[0]["rgb"][0]["data"])
        if frame.shape[:2] != (self.height, self.width) or frame.dtype != np.uint8:
            raise RuntimeError(f"{camera} produced no valid RGB frame")
        return np.ascontiguousarray(frame[..., :3])

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

    def render_spectator(self, name):
        from pxr import Gf
        view, transform, rgb = self._spectators[name]
        position, q = self._camera(view.source).get_world_pose(camera_axes="usd")
        w, x, y, z = values(q)
        matrix = np.eye(4)
        matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        matrix[:3, 3] = values(position)
        transform.Set(Gf.Matrix4d(matrix.T.tolist()))
        self.env.obs_manager.render_for_capture()
        return np.ascontiguousarray(np.asarray(rgb.get_data())[..., :3])

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
