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
rosbridge setup robotwin/robotwin-aloha-agilex --directory ../my-robotwin-workspace
```

`robotwin` selects `handover_block`. Task IDs keep RoboEnv's embodiment suffix,
for example `robotwin/stack-blocks-two-aloha-agilex`; native names such as
`stack_blocks_two` also resolve. Four scenes are unstable at seed 0 upstream, so
their catalog entries carry RoboEnv's default seed (`dump_bin_bigbin` 5,
`pick_diverse_bottles` 1, `shake_bottle` and `shake_bottle_horizontally` 2), used
when `--seed` is omitted.

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
| `world` | `/clock`, `world/wait`, `world/stop`, `world/reset`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `left_gripper`, `right_gripper` | `<gripper>/joint_state`, `<gripper>/target`, `<gripper>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/depth`, `<camera>/camera_info`, `<camera>/capture` |

Arm targets name the native joints `fl_joint1..6` (left) and `fr_joint1..6`
(right) in radians: the arm entries of upstream's 14-D `qpos` action. The native
URDF limits are ±10 rad placeholders. Gripper targets are upstream's normalized
gripper value from 0 (closed) to 1 (open); the finger drive target is
`gripper_scale` `-0.01 + 0.055 v` metres, so a closed command squeezes past the
finger stop. `joint_state` reports the same normalization of the measured finger,
which reads about 0.18 when the fingers touch. Each `stop` holds the measured
position.

Cameras are D435 models at 320×240 (fx = fy ≈ 358.64 px, principal point at the
image center). RGB is upstream's ray-traced render; registered depth comes from a
raster twin of each camera at pixel centers, as in RoboEnv, and `capture` is
`kinex_interfaces/srv/CaptureRGBD`. The rig `robotwin/robotwin-aloha-agilex` holds
the arm chains from `arx5_description_isaac.urdf`, the measured mounts, the TCP
(upstream's tcp pose, 0.12 m ahead of each wrist flange) and the camera mounts.

## Timing and episodes

Physics runs at 250 Hz. One control frame is ten substeps, 0.04 seconds: RoboEnv's
absolute-joint practice protocol. Each frame sets both arms' native joint drive
targets once, and calls the native normalized `set_gripper` before every substep,
keeping its saturation and per-call rate limit. Upstream's own `take_action`
retimes each action with TOPP over a variable number of substeps; that timing is
not reproduced, as in RoboEnv.

Reset follows upstream evaluation: `setup_demo` for the seed, the expert dry run
(`play_once`, whose planning may fail without consequence), then `setup_demo` again
in evaluation mode. It takes about a minute. Success is upstream's `check_success`,
evaluated on every substep and latched; the episode horizon is the native
`step_lim` unless a maximum is given. A failed reset keeps the previous episode.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` follow
the head and wrist cameras at 960×720.

```bash
pixi run -e robotwin smoke                      # native: timing, gripper, reset, rendering
ROSBRIDGE_ALL_TASKS=1 pixi run -e robotwin smoke # every task loads and steps (~35 min)
pixi run -e robotwin test-ros
rosbridge demo robotwin
```

`test_kinex_geometry.py` (with `KINEX_PYTHON`) checks the exported rig's arm FK,
finger links, TCP and camera mounts against SAPIEN.
