# RoboDojo

RoboDojo runs its 42 simulation tasks on the dual ARX X5 rig in Isaac Sim 5.1,
with upstream's saved evaluation layouts. The integration follows RoboEnv's
RoboDojo device runtime. Upstream is pinned to
`cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c`, with its Isaac Lab fork
(`yuechen0614/IsaacLab@afca7b09`) and cuRobo fork (`yuechen0614/curobo@895c6517`)
pinned as nested sources. See the [design and CLI guide](README.md) for shared
lifecycle and timing contracts.

```bash
rosbridge prepare robodojo
rosbridge tasks list --simulator robodojo
rosbridge launch robodojo/stack-bowls --execution-mode stepped --timeout 1800
rosbridge setup robodojo/robodojo-arx-x5 --directory ../my-robodojo-workspace
```

`robodojo` selects `pack_objects_into_box`. The twelve `*_random` variants are not
cataloged. Three competition tasks (`imitate_sorting_sequence`, `make_kong`,
`play_tic_tac_toe`) add a Franka support arm that the native environment drives;
it is not a device.

## Native worker

Isaac Sim 5.1 publishes Python 3.11 wheels only, while ROS Jazzy from RoboStack is
Python 3.12. The world node therefore runs in the `robodojo` environment (the
shared ROS environment) and proxies `rosbridge_robodojo.backend.RoboDojoSim`,
which runs on the main thread of a worker process in `robodojo-native` (Python
3.11, Isaac Sim 5.1, CUDA 12.8 torch 2.7, RoboEnv's pins). Calls are sequential
over a private socket, so the ROS clock, waits and device semantics are the same
as for in-process simulators; a dead worker stops the graph. The worker sees
only one GPU, the first in `CUDA_VISIBLE_DEVICES` (default 0), and renders on its
Vulkan device: on IOMMU hosts Kit otherwise validates peer-to-peer access between
every GPU, holding each for minutes. It imports torch before Kit starts, since
torch's bundled OpenMP needs static TLS that Kit's plugins would otherwise use up.

Isaac starts, compiles shaders and builds the scene before the first reset: allow
several minutes (the default launch timeout is 900 s; the first launch on a host
may need `--timeout 1800`). A worker serves the one task it was launched with.

## Source and assets

Preparation exports RoboDojo with both forks into `.runtime/robodojo`, applies
RoboEnv's four patches (success no longer ends an episode unless requested; an
injected model client replaces the policy server; predicate diagnostics hooks;
Isaac Lab's livestream dependency becomes an extra), and installs the forks as
editable packages into `robodojo-native`.

Assets come from the `RoboDojo-Benchmark/RoboDojo` dataset at
`0a709fe7863f6c869a8d141676585a883fce4270`: 15,365 files (41 GB) under
`Assets/`, each pinned by LFS SHA-256 or git blob SHA-1 in the dataset listing.
With `ROSBRIDGE_ASSET_SEED` pointing at RoboEnv's asset cache, every file is
verified and the tree is moved into the store; otherwise files are downloaded
once at a polite request rate.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/reset`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `left_gripper`, `right_gripper` | `<gripper>/joint_state`, `<gripper>/target`, `<gripper>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/camera_info` |

Both arms keep upstream's joint names `joint1..joint6` (radians; native limits
±10 rad for joints 1–5 and ±3.14 for joint 6); the device disambiguates them.
Arm targets are upstream's `<side>_arm_joint_state` action entries. Gripper
targets are upstream's normalized `<side>_ee_joint_state` value from 0 (closed) to
1 (open), mapped by `gripper_scale` to finger targets `-0.01 + 0.054 v` metres;
`joint_state` reports the same normalization of the measured primary finger
(about 0.185 when the fingers touch). Each `stop` holds the measured position.

Cameras are RGB only, as upstream renders them: the head camera (Gemini 345Lg
model, fx ≈ 288.13 px) and two D435 wrist cameras (fx ≈ 397.04 px), all 640×480,
rendered through RoboDojo's tiled render products. Isaac's symmetric frustum puts
the principal point at the image center, (319.5, 239.5) in ROS pixel coordinates.
The rig `robodojo/robodojo-arx-x5` holds the X5 chains from `X5A.urdf`, mounts
at (∓0.3, −0.45, 0.765) with a 90° yaw, RoboEnv's TCP (the pad center, 0.15 m
ahead of `link6`) and the measured camera mounts.

## Timing and episodes

Physics runs at 250 Hz and one control frame is ten substeps, 0.04 seconds. Each
frame holds the commanded targets and calls upstream's `EvalEnv.take_action` with
the joint action dictionary, which interpolates the arms and fingers within the
frame, as RoboEnv does. Success is native completion and success; graded tasks
report upstream's 0..100 score as a reward in [0, 1]. The horizon is the native
`step_lim` unless a maximum is given.

Reset loads saved layout `ids[seed % len(ids)]`. Reloading a layout after another
occasionally fails upstream's settle check; the same layout is retried up to three
times, so a seed keeps its layout. Isaac hosts one environment per process, so the
reload happens in the live environment: a reset that fails after reloading starts
leaves the episode done until a reset succeeds.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` copy the
optics of the head and wrist cameras at 960×720 and follow them.

```bash
pixi run -e robodojo-native smoke                       # native checks via worker processes
ROSBRIDGE_ALL_TASKS=1 pixi run -e robodojo-native smoke  # every task boots Isaac (hours)
pixi run -e robodojo test-ros
rosbridge demo robodojo --timeout 1800
```

The native checks run the backend in worker subprocesses because Isaac's shutdown
ends its process. With `KINEX_PYTHON`, they also check the exported rig's arm FK,
finger links and camera mounts against Isaac.
