"""One ABC episode: upstream env, state, scoring, and camera rendering. No ROS."""

import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.core.mujoco_rendering import refresh
from .assets import require_assets
from .scoring import supplemental_evaluator

SIDES = ("left", "right")
GRIPPER_SCALE = 0.0475  # upstream _GRIPPER_CTRL_MAX: finger joint meters per policy unit
# MuJoCo cameras look along -z with y up; ROS optical frames look along +z with y down.
OPTICAL = np.diag([1.0, -1.0, -1.0])


def world_seed(seed):
    """Upstream's evaluation seed for world `seed`: abc_minimal's SimEvalConfig.seed + world index
    (eval_policy.py: rollout_worlds)."""
    from abc_minimal.config import SimEvalConfig
    return SimEvalConfig.seed + int(seed)


def joint_names():
    names = []
    for side in SIDES:
        names += [f"{side}_joint{i}" for i in range(1, 7)] + [f"{side}_gripper"]
    return names


def rollout_stop(task_eval):
    """Where abc_minimal's evaluator ends a rollout early (eval_policy.py: rollout_over): a task that reports
    ever_failed, such as ball_tray_balancing, at its failure; any other at its first success."""
    if "ever_failed" in task_eval:
        if not task_eval["ever_failed"]:
            return None
        detail = [name for name, value in (("ball_dropped", task_eval.get("ball_dropped")),
                                           ("ball_left_tray", task_eval.get("inside_tray") is False)) if value]
        return "failure", ", ".join(detail) or "ever_failed"
    return ("success", None) if task_eval.get("ever_success") else None


class AbcSim:
    default_task = "put_plastic_bottles_in_bin"

    def __init__(self, *, cameras=("top", "left", "right"), width=640, height=480,
                 max_steps=None, early_stop=False):
        self.cameras = tuple(cameras)
        self.width, self.height = width, height
        self.max_steps = max_steps
        self.early_stop = early_stop
        self.env = self._renderer = None
        self._spectator = self._supplemental = self.stop = None
        self._task_eval = {}
        self.spectators = ()

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views((("third_person", "top"), ("wrist", "left"), ("right_wrist", "right")),
                                          width, height, views)

    @property
    def devices(self):
        cameras = {f"camera_{camera}": camera for camera in self.cameras}
        return device_inventory(self.layout, cameras, depth=tuple(cameras))

    def reset(self, task, seed, randomize=True):
        """Reset as upstream's evaluator resets world `seed`, with its world seed."""
        import abc_sim

        spec = abc_sim.get_task_spec(task)
        native = world_seed(seed)
        require_assets(Path(abc_sim.__file__).resolve().parents[1], spec.name, spec.env_task)
        env = abc_sim.make_env(
            task=spec.name,
            render_cameras=False,
            max_episode_steps=self.max_steps,
            terminate_on_success=False,  # early stop follows the evaluator's rollout_over instead
            # inhand_transfer samples its object when the env is built.
            **({"seed": native} if spec.env_task == "inhand_transfer" else {}),
        )
        candidate = AbcSim(cameras=self.cameras, width=self.width, height=self.height,
                           max_steps=self.max_steps, early_stop=self.early_stop)
        candidate.spectators = self.spectators
        candidate.env, candidate.task, candidate.seed, candidate.world_seed = env, spec.name, seed, native
        try:
            env.reset(seed=native, randomize=randomize)
            missing = [c for c in self.cameras
                       if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, c) < 0]
            if missing:
                raise ValueError(f"unknown cameras: {missing}")
            candidate.period = float(env.model.opt.timestep) * env._control_decimation
            candidate.steps, candidate.done, candidate.stop = 0, False, None
            candidate._index(env.model)
            if hasattr(self, "layout") and candidate.layout != self.layout:
                raise ValueError("reset cannot change the action layout")
            # Upstream lacks evaluators for four tasks; judge-only rules fill them in.
            candidate._supplemental = supplemental_evaluator(spec.name, env.model)
            candidate._refresh()
            candidate._evaluate()
            if candidate.spectators:
                from rosbridge.core.mujoco_rendering import SpectatorRenderer
                candidate._spectator = SpectatorRenderer(env.model, env.data, candidate.spectators)
                # Validate capture before replacing a live episode.
                for view in candidate.spectators:
                    candidate._spectator.capture(view.name)
        except BaseException:
            candidate.close()
            raise
        # Adopt only after every initialization stage succeeds; a failed reset
        # leaves the old environment, renderer, indices and score intact.
        self.close()
        self.__dict__.update(candidate.__dict__)

    def measures(self):
        return {"world_seed": self.world_seed}

    def _index(self, model):
        def joint(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"model has no joint {name}")
            return jid

        dofs, limits = [], {}
        for side in SIDES:
            arm = [joint(f"{side}_joint{i}") for i in range(1, 7)]
            dofs += [model.jnt_dofadr[j] for j in arm]
            dofs.append(model.jnt_dofadr[joint(f"{side}_left_finger")])
            lo = [model.jnt_range[j][0] if model.jnt_limited[j] else -math.inf for j in arm]
            hi = [model.jnt_range[j][1] if model.jnt_limited[j] else math.inf for j in arm]
            limits[side] = (np.array(lo), np.array(hi))
        self._dofs, self.limits = np.array(dofs), limits
        groups = []
        for side, offset in (("left", 0), ("right", 7)):
            lo, hi = limits[side]
            groups.append(CommandGroup(f"{side}_arm", tuple(range(offset, offset + 6)),
                                       tuple(lo), tuple(hi)))
            groups.append(CommandGroup(f"{side}_gripper", (offset + 6,), (0.0,), (1.0,),
                                       scalar=True))
        self.layout = ActionLayout(tuple(joint_names()), tuple(groups))
        self._camera_ids = {
            c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in self.cameras
        }

    def _score(self, info):
        info = info or {}
        self._task_eval = info
        self.reward = float(info.get("reward", 0.0))
        self.success = bool(info.get("success", False))

    def _evaluate(self, info=None):
        if self._supplemental is not None:
            self._score(self._supplemental.evaluate(self.env.data))
        elif info is not None and "task_eval" in info:
            self._score(info["task_eval"])
        elif info is None:
            result = self.env.evaluate_task()
            self._score(result.to_info(squeeze=True) if result is not None else None)

    def _refresh(self):
        refresh(self.env.model, self.env.data)

    @property
    def time_s(self):
        return float(self.env.data.time)

    def state(self):
        """14-D positions (policy space) and velocities in the same units."""
        pos = np.asarray(self.env.get_state(), dtype=float)
        vel = np.asarray(self.env.data.qvel[self._dofs], dtype=float)
        vel[[6, 13]] /= GRIPPER_SCALE
        return pos, vel

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is done")
        _, _, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32), render_obs=False
        )
        self.steps += 1
        self._refresh()
        self._evaluate(info)
        if self.early_stop:
            # Upstream lacks an evaluator for the supplemental tasks; their judge-only success ends them.
            self.stop = (rollout_stop(self._task_eval) if self._supplemental is None else
                         ("success", None) if self.success else None)
        self.done = bool(terminated or truncated or self.stop)

    def camera_pose(self, camera):
        """Optical-frame pose (pos, quat_wxyz) in the world frame."""
        cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise KeyError(camera)
        d = self.env.data
        rot = d.cam_xmat[cid].reshape(3, 3) @ OPTICAL
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        return d.cam_xpos[cid].copy(), quat

    def intrinsics(self, camera):
        """width, height, fx, fy, cx, cy in pixels."""
        m, cid = self.env.model, self._camera_ids[camera]
        if m.cam_sensorsize[cid].any():
            raise NotImplementedError("cameras with sensorsize are not supported")
        w, h = self.width, self.height
        fy = (h / 2) / math.tan(math.radians(m.cam_fovy[cid]) / 2)
        return w, h, fy, fy, (w - 1) / 2, (h - 1) / 2

    def _scene(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.env.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.env.data, camera=camera)
        return self._renderer

    def render(self, camera):
        """(H, W, 3) uint8 RGB."""
        return self._scene(camera).render().copy()

    def render_depth(self, camera):
        """(H, W) float32 metres along the optical axis; 0 where nothing is within the clip range."""
        from rosbridge.core.mujoco_rendering import render_depth
        return render_depth(self._scene(camera), self.env.model)

    def close(self):
        if self._spectator is not None:
            self._spectator.close()
            self._spectator = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def render_spectator(self, name):
        return self._spectator.capture(name)

    def spectator_intrinsics(self, name):
        return self._spectator.intrinsics(name)
