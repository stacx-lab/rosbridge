# RoboTwin

RoboTwin 2.0 runs its 50 evaluation tasks on the ALOHA AgileX dual-arm robot in
SAPIEN, with the clean-background `demo_clean` configuration. The integration
follows RoboEnv's RoboTwin device runtime. Upstream is pinned to
`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`; see the
[design and CLI guide](README.md) for shared lifecycle and timing contracts.

```bash
rosbridge prepare robotwin
rosbridge tasks list --simulator robotwin
rosbridge launch robotwin/handover-block-aloha-agilex --execution-mode stepped
rosbridge setup robotwin/handover-block-aloha-agilex --directory ../my-robotwin-workspace
```

`robotwin` selects `handover_block`. Task IDs keep RoboEnv's embodiment suffix,
for example `robotwin/stack-blocks-two-aloha-agilex`; native names such as
`stack_blocks_two` also resolve.

## Seeds

Seed k runs the scene upstream's evaluation runs as its episode k. Upstream
(`scripts/eval_policy_xpolicylab.py`) tries the native seeds 100000 and up in order,
and evaluates a seed only once its expert check passes:

- its scripted expert solves the scene (`plan_success` and `check_success`);
- the scene is stable;
- setup raises no error.

`simulators/robotwin/seeds.json` lists those seeds per task. The episode record names
the one in use as `native_seed`, and a seed past the table is refused.

`tools/robotwin_seeds.py` writes the table with upstream's own code:

- **Tree:** the pinned revision unpatched, with this checkout's assets.
- **Arguments and planner:** upstream's task arguments, and cuRobo v0.7.8, which
  aloha-agilex's configuration plans with.
- **Edits:** upstream's edits to mplib and SAPIEN (`scripts/_install.sh`).

cuRobo needs CUDA torch, so the screen runs in its own `robotwin-screen` environment;
the simulator's `robotwin` environment stays CPU-only.

Screen on idle GPUs, one task per GPU. cuRobo plans under a 10 s wall-clock budget, so
on a loaded GPU verdicts shift both ways: a screen at two tasks per busy GPU differed
from an idle one in 18 of the 50 tasks. Even on idle GPUs, a borderline
seed can flip between runs: about one seed in twenty to thirty in the hardest tasks
(`put_object_cabinet`, `scan_object`), as it would in upstream's own evaluation. The
table is one such run of upstream's check. The screen resumes an
interrupted run, and a larger `--count` extends the table:

```bash
pixi run --frozen -e robotwin-screen install-curobo   # once
pixi run --frozen -e robotwin-screen python tools/robotwin_seeds.py --gpus 0,1,2,4
```

## Environment, source and assets

The `robotwin` Pixi environment is Python 3.12 with RoboEnv's pins (SAPIEN
3.0.0b1, mplib 0.2.1, toppra 0.6.9, CPU torch 2.4.1, NumPy 1.26.4); scipy 1.13.1
and open3d 0.19 replace releases without Python 3.12 wheels. cuRobo is not
installed: the expert dry run plans with mplib.

`patches/robotwin/0001-runtime.patch` is RoboEnv's runtime patch without its
unused YAM hunks: an explicit SAPIEN render device, optional cuRobo with mplib
planning, verified grasp candidates behind an embodiment flag, and planner effort
and seed from `ROBOENV_PLANNER_*`.

Preparation mounts the pinned `TianxingChen/RoboTwin2.0` packs
(`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`): `embodiments.zip` and
`objects.zip`, each SHA-256 pinned. Randomized background textures are excluded.
With `ROSBRIDGE_ASSET_SEED` pointing at RoboEnv's asset cache, the extracted packs
are verified against their archives (every member's size and CRC) and moved into
the store instead of being downloaded.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `left_gripper`, `right_gripper` | `<gripper>/joint_state`, `<gripper>/target`, `<gripper>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/depth`, `<camera>/camera_info` |

Arm targets name the native joints `fl_joint1..6` (left) and `fr_joint1..6`
(right) in radians: the arm entries of upstream's 14-D `qpos` action. The native
URDF limits are ±10 rad placeholders. Gripper targets are upstream's normalized
gripper value from 0 (closed) to 1 (open); the finger drive target is
`gripper_scale` `-0.01 + 0.055 v` metres, so a closed command squeezes past the
finger stop. `joint_state` reports the same normalization of the measured finger,
which reads about 0.18 when the fingers touch. Each arm `stop` holds the measured
position; a gripper `stop` keeps its applied grip target, so a held object stays squeezed.

Cameras are D435 models at 320×240 (fx = fy ≈ 358.64 px, principal point at the
image center). RGB is upstream's ray-traced render; registered depth comes from a
raster twin of each camera at pixel centers, as in RoboEnv, with the same stamp as
its image. The rig `robotwin/robotwin-aloha-agilex` holds
the arm chains from `arx5_description_isaac.urdf`, the measured mounts, the TCP
(upstream's tcp pose, 0.12 m ahead of each wrist flange) and the camera mounts.

## Timing and episodes

Physics runs at 250 Hz. One control frame is ten substeps, 0.04 seconds: RoboEnv's
absolute-joint practice protocol. Each frame sets both arms' native joint drive
targets once, and calls the native normalized `set_gripper` before every substep,
keeping its saturation and per-call rate limit. Upstream's own `take_action`
retimes each action with TOPP over a variable number of substeps; that timing is
not reproduced, as in RoboEnv.

The episode starts as upstream evaluation does: `setup_demo` for the native seed, the
expert dry run (`play_once`, whose planning may fail without consequence), then
`setup_demo` again in evaluation mode. It takes about a minute. Success is
upstream's `check_success`, evaluated on every substep from the first action on, as
upstream's `take_action` does, and latched; the episode
horizon is the native `step_lim` unless a maximum is given. `--early-stop` ends the
episode at the first success, as the evaluator does; RoboTwin has no failure termination.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` follow
the head and wrist cameras at 960×720.

```bash
pixi run --frozen -e robotwin smoke                      # native: timing, gripper, reset, rendering
ROSBRIDGE_ALL_TASKS=1 pixi run --frozen -e robotwin smoke # every task loads and steps (~35 min)
pixi run --frozen -e robotwin test-ros
rosbridge demo robotwin
```

`tests/kinex/native/test_robotwin.py` (with `KINEX_PYTHON`) checks the exported rig's arm FK,
finger links, TCP and camera mounts against SAPIEN.
