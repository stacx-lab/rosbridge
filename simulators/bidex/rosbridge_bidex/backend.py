"""BiDex-100 on dual Franka FR3v2 or Astribot S1 arms with Sharpa hands; no ROS dependencies.

A reset replays one episode of upstream's leaderboard evaluator (bidex/eval/run_leaderboard_eval.py
and vector_rollout.py:_start_next_episode): the ID task with a recorded working pose from the
evaluation H5, or the __ood task with its scene, layout, distractors and object scales, then a 1 s
settle with the robot pinned. OOD cameras keep their nominal mounts. Every 1/25 s step passes the 58
absolute joint targets through BiDexEnv's joint mode and its latched subtask tracking; the episode
record adds the evaluator's progress score.
"""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.core.mujoco_rendering import refresh, render_depth
from rosbridge.project import project_root
from .assets import check_task

EPISODES, OOD_SEED = 25, 2  # the official evaluator's episodes per eval seed and first OOD eval seed
CAMERAS = {"head_camera": "head", "left_camera": "left_wrist", "right_camera": "right_wrist"}
SPECTATORS = (("third_person", "head"), ("left_wrist", "left_wrist"), ("right_wrist", "right_wrist"))
# BiDexEnv's right-first packing: arm_joint_position (14), then dex_joint_position (44).
GROUPS = {"right_arm": "right_franka_jnt_ids", "left_arm": "left_franka_jnt_ids",
          "right_hand": "right_sharpa_jnt_ids", "left_hand": "left_sharpa_jnt_ids"}
# MuJoCo cameras look along -z with y up; ROS optical frames look along +z with y down.
OPTICAL = np.diag([1., -1., -1.])


def episode(seed, ood):
    """The official evaluator's (eval seed, episode) for a seed: 0-49 replay its rollouts."""
    eval_seed, index = divmod(int(seed), EPISODES)
    return eval_seed + OOD_SEED * ood, index


class BiDexSim:
    default_task = "Generalization/put_apple_in_white_bowl"
    period = 1 / 25

    def __init__(self, *, max_steps=None, terminate_on_success=False):
        self.requested_max_steps = self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success
        self.cameras = tuple(CAMERAS.values())
        self.env = self._loaded = self._spectator = None
        self.spectators = ()

    @property
    def devices(self):
        return device_inventory(self.layout, CAMERAS, depth=tuple(CAMERAS))

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    def reset(self, task, seed, randomize=True):
        """The seeded evaluator episode; randomize has no effect."""
        native = resolve("bidex", task=task)[2]["native_task"]
        check_task(project_root() / ".runtime/bidex", native)
        candidate = BiDexSim(max_steps=self.requested_max_steps,
                             terminate_on_success=self.terminate_on_success)
        candidate.spectators = self.spectators
        try:
            candidate._initialize(native, seed)
            if hasattr(self, "layout") and candidate.layout != self.layout:
                raise ValueError("reset cannot change the action layout")
        except BaseException:
            candidate.close()
            raise
        self.close()
        self.__dict__.update(candidate.__dict__)

    def _initialize(self, native, seed):
        from bidex import BiDexEnv
        from bidex.eval import ood, run_leaderboard_eval as evaluator
        from bidex.eval.scoring import subtask_weights_from_task

        suite, name = native.split("/")
        variant = name.endswith("__ood")
        name = name.removesuffix("__ood")
        eval_seed, index = episode(seed, variant)
        self.task, self.seed, key = native, seed, (suite, name, eval_seed)
        self._loaded, _, _ = evaluator.load_task_for_domain(
            suite, name, "out_of_domain" if variant else "in_domain", eval_seed,
            evaluator.stable_task_hash(suite, name), episode=index if variant else None)
        self._loaded._ood_camera_plan = None  # cameras keep the rig's nominal mounts
        self.env = env = BiDexEnv(self._loaded, control_mode="joint", cameras=list(self.cameras),
                                  seed=ood.episode_seed(*key, index if variant else 0, "environment"))
        env.reset(seed=ood.episode_seed(*key, index, "environment"))
        if variant:
            ood.resample_distractors(env, env.task, ood.episode_rng(*key, index, "placement"))
        else:
            poses = evaluator.load_collection_init_states(suite, name)
            if not poses.size:
                raise RuntimeError(f"no recorded ID robot poses for {suite}/{name}")
            draw = ood.episode_rng(*key, index, "id.collection_init").integers(0, poses.shape[0])
            evaluator.restore_state_from_collection(env, poses[int(draw)])
        env.settle(1.0)
        self._weights = subtask_weights_from_task(self._loaded)
        self.max_steps = self.requested_max_steps or evaluator.max_steps_for_task(suite, name)
        self._index()
        env._init_renderer()
        if env.renderer is None:
            raise RuntimeError("BiDex-100 cannot render its cameras")
        if self.spectators:
            from rosbridge.core.mujoco_rendering import SpectatorRenderer
            self._spectator = SpectatorRenderer(env.model, env.data, self.spectators,
                                                options=env._scene_option)
            for view in self.spectators:
                self._spectator.capture(view.name)
        self.steps, self.done, self.success, self.reward = 0, False, False, 0.
        self._instruction = env._current_language()
        print(f"[bidex] {native} seed {seed} (eval seed {eval_seed}, episode {index}): "
              f"{self._instruction}", flush=True)

    def _index(self):
        model, groups, joints = self.env.model, [], []
        for name, attribute in GROUPS.items():
            ids = list(getattr(self.env, attribute))
            if len(ids) != (7 if name.endswith("arm") else 22):
                raise ValueError(f"{self.task} lacks the {name} joints")
            groups.append(CommandGroup(name, tuple(range(len(joints), len(joints) + len(ids))),
                                       tuple(model.jnt_range[ids, 0]), tuple(model.jnt_range[ids, 1])))
            joints += ids
        self._qpos, self._qvel = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.layout = ActionLayout(tuple(model.joint(j).name for j in joints), tuple(groups))

    @property
    def time_s(self):
        return self.steps * self.period

    def state(self):
        """Arm and hand joints in radians, right side first, as BiDexEnv packs them."""
        return self.env.data.qpos[self._qpos].copy(), self.env.data.qvel[self._qvel].copy()

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is done")
        action = np.asarray(action, dtype=float)
        # The evaluator's physics and subtask tracking, without rendering the policy's images.
        reward, _, info = self.env.step_physics_only(
            {"arm_joint_position": action[:14], "dex_joint_position": action[14:]})
        refresh(self.env.model, self.env.data)
        self.steps += 1
        self.reward, self.success = float(reward), bool(info["success"])
        self.done = bool(self.steps >= self.max_steps or info["timed_out"]
                         or (self.terminate_on_success and self.success))
        instruction = self.env._current_language()  # Instruction_Following tasks switch it
        if instruction != self._instruction:
            self._instruction = instruction
            print(f"[bidex] at {self.time_s:.2f} s the instruction becomes: {instruction}", flush=True)

    def measures(self):
        """The evaluator's progress score, 0-100: the weights of the subtasks credited now."""
        from bidex.eval.scoring import episode_score
        return {"score": episode_score(self.env._subtask_tracker.status()["subtasks"],
                                       self._weights)["score"]}

    # Cameras ----------------------------------------------------------------------------

    def _scene(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        env = self.env
        env.renderer.update_scene(env.data, camera=env.camera_ids[camera],
                                  scene_option=env._scene_option)
        return env.renderer

    def render(self, camera):
        """(H, W, 3) uint8 RGB, as BiDexEnv renders its observations."""
        return self._scene(camera).render().copy()

    def render_depth(self, camera):
        return render_depth(self._scene(camera), self.env.model)

    def intrinsics(self, camera):
        env = self.env
        w, h = env.img_width, env.img_height
        focal = (h / 2) / np.tan(np.deg2rad(env.model.cam_fovy[env.camera_ids[camera]]) / 2)
        return w, h, float(focal), float(focal), (w - 1) / 2, (h - 1) / 2

    def camera_pose(self, camera):
        """Optical-frame pose (pos, quat_wxyz) in the world frame."""
        cid, data = self.env.camera_ids[camera], self.env.data
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (data.cam_xmat[cid].reshape(3, 3) @ OPTICAL).flatten())
        return data.cam_xpos[cid].copy(), quat

    def render_spectator(self, name):
        return self._spectator.capture(name)

    def spectator_intrinsics(self, name):
        return self._spectator.intrinsics(name)

    def close(self):
        if self._spectator is not None:
            self._spectator.close()
            self._spectator = None
        if self.env is not None:
            self.env.close()
            self.env = None
        # The OOD scene upstream generates beside the LIBERO scenes (run_leaderboard_eval._close_eval_env).
        scene = Path(getattr(self._loaded, "scene", ""))
        if scene.name.startswith("ood_scene_"):
            scene.unlink(missing_ok=True)
        self._loaded = None
