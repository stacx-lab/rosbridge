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
rosbridge setup robodojo/stack-bowls --directory ../my-robodojo-workspace
rosbridge setup robodojo/stack-bowls --config depth --directory ../my-robodojo-depth-workspace
rosbridge launch robodojo/stack-bowls --config depth --execution-mode stepped --timeout 1800
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

Isaac starts, compiles shaders and builds the scene before the episode starts: allow
several minutes (the default launch timeout is 900 s; the first launch on a host
may need `--timeout 1800`). A worker serves the one task it was launched with.

## Source and assets

Preparation exports RoboDojo with both forks into `.runtime/robodojo`, applies
RoboEnv's four patches (success and a failed constraint end an episode only with early stop; an
injected model client replaces the policy server; predicate diagnostics hooks;
Isaac Lab's livestream dependency becomes an extra), and installs the forks as
editable packages into `robodojo-native`. A fifth patch lets upstream's cuRobo planner read its
`curobo_tmp.yml` template, with `${ASSETS_PATH}` replaced. Upstream's install script writes that
same file as `curobo.yml` (`utils/update_embodiment_config_path.py`), but rosbridge never writes
into the pinned asset store.

Assets come from the `RoboDojo-Benchmark/RoboDojo` dataset at
`0a709fe7863f6c869a8d141676585a883fce4270`: 15,365 files (41 GB) under
`Assets/`, each pinned by LFS SHA-256 or git blob SHA-1 in the dataset listing.
With `ROSBRIDGE_ASSET_SEED` pointing at RoboEnv's asset cache, every file is
verified and the tree is moved into the store; otherwise files are downloaded
once at a polite request rate.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/pose_target`, `<arm>/pose_state`, `<arm>/stop` |
| `left_gripper`, `right_gripper` | `<gripper>/joint_state`, `<gripper>/target`, `<gripper>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/camera_info`; `<camera>/depth` with `--config depth` |

Both arms keep upstream's joint names `joint1..joint6` (radians; native limits
±10 rad for joints 1–5 and ±3.14 for joint 6); the device disambiguates them.
Arm targets are upstream's `<side>_arm_joint_state` action entries. Gripper
targets are upstream's normalized `<side>_ee_joint_state` value from 0 (closed) to
1 (open), mapped by `gripper_scale` to finger targets `-0.01 + 0.054 v` metres;
`joint_state` reports the same normalization of the measured primary finger
(about 0.185 when the fingers touch). Each arm `stop` holds the measured position; a
gripper `stop` keeps its applied grip target, so a held object stays squeezed.

Cameras are RGB, as upstream's evaluation renders them, or RGB-D in the depth
configuration below. They are the head camera (Gemini 345Lg model, fx ≈ 288.13 px) and
two D435 wrist cameras (fx ≈ 397.04 px), all 640×480,
rendered through RoboDojo's tiled render products. Isaac's symmetric frustum puts
the principal point at the image center, (319.5, 239.5) in ROS pixel coordinates.
The rig `robodojo/robodojo-arx-x5` holds the X5 chains from `X5A.urdf`, mounts
at (∓0.3, −0.45, 0.765) with a 90° yaw, RoboEnv's TCP (the pad center, 0.15 m
ahead of `link6`) and the measured camera mounts.

### Pose targets

Each arm also takes `pose_target`: timed waypoints for its `flange` in `<arm>/base`. The flange
is `link6`, upstream's `ee_link` and cuRobo's tool frame, and `pose_state` reports its measured
pose. The flange reaches each waypoint at its time along a straight line, with rotation by
slerp. Nothing plans a path around objects, so a single far waypoint is a straight move, and it
needs enough time for every joint to stay within one frame's travel. The latest command of either kind replaces the active one: a `joint_target` after a pose
command starts from the measured joints, and a `stop` holds them.

Upstream's own `ee_pose` action solves a pose with cuRobo IK and, when the solve fails, leaves
the arm on its previous target without saying so. The backend solves a pose the same way, then
executes the result as upstream's joint action:

1. Each frame, upstream's `RobotManager.solve_ik` solves that frame's pose from the measured
   joints. It is the cuRobo `InverseKinematics` the `ee_pose` action uses: 32 seeds, self and
   table collision, 1 mm and 0.02 rad. An unchanged pose keeps its solution.
2. The solution must lie within the joint limits, taking a whole turn nearer the commanded joints
   where the limits allow. It must move no joint more than 0.2 rad from the commanded joints: the
   X5 actuators' 5 rad/s `velocity_limit_sim` over the 0.04 s frame. And its joint path, sampled
   every 0.025 rad, must pass cuRobo's self and table constraints. If it fails any of these, the
   next of cuRobo's 32 ranked solutions that passes is taken. These checks follow
   uni-robot-agent's RoboDojo adapter, with the native velocity limit in place of its 1 rad
   jump limit.
3. If no solution passes, the arm keeps its commanded joints for that frame and the world then
   stops it at its measured joints. The pending `world/wait` ends there, with `success` false and
   a message naming the pose target, the time and the reason, so the rest of the wait costs no
   budget.

The joints then go through `EvalEnv.take_action` exactly as joint targets do, so evaluation and
scoring are unchanged. A pose is tracked only as precisely as upstream's IK tolerances, 1 mm and
0.02 rad. To serve pose targets, rosbridge builds upstream's cuRobo planner for the arms, as
upstream's evaluation does by default (`need_planner`). RoboEnv's adapter, whose arms took joints
only, skipped it. `robodojo-native` carries cuRobo's `[cu12]` extra (`cuda-core`), as upstream's
install does, because the planner compiles its kernels through it.

### Depth configuration

`--config depth` makes all three cameras RGB-D, through upstream's own depth capture.
Every camera gets the `distance_to_image_plane` annotator, which upstream's
`env_cfg/camera/camera_config.yml` carries commented out. Depth is metres along the optical
axis, rendered with each image. Each camera serves it as `<camera>/depth`: 16UC1 millimetres
registered to the image, with the same stamp and frame. The profiles state the cameras'
clipping ranges as valid ranges:

- head camera: 0.005–10 m;
- wrist cameras: 0.001–10 m. Their 0.0001 m near clip is below the millimetre encoding.

0 marks a missing sample. The rig's `configurations/depth/profiles.yaml` holds these profile
changes, so setup and launch must name the same configuration. Upstream's policy observations
are unchanged: `vision.depth` stays off.

## Timing and episodes

Physics runs at 250 Hz and one control frame is ten substeps, 0.04 seconds. Each
frame holds the commanded targets and calls upstream's `EvalEnv.take_action` with
the joint action dictionary, which interpolates the arms and fingers within the
frame, as RoboEnv does. Success is native completion and success. The reward is
upstream's episode score (`EvalEnv.run_eval`): 1 for a success; a failed graded task
keeps upstream's 0..100 score, as a reward in [0, 1); other failures score 0. The horizon is the native
`step_lim` unless a maximum is given. `--early-stop` ends the episode where `EvalEnv`
does: at its completion, or when a task constraint fails (a `query` or `trigger_query`
that held on more steps than its task allows, such as `push_T`'s lifted T), with that
constraint as the cause. Without it a failed constraint keeps success false and the
episode runs to its horizon.

The episode loads saved layout `ids[seed % len(ids)]`, retried up to three times
when it fails upstream's settle check, so a seed keeps its layout.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` copy the
optics of the head and wrist cameras at 960×720 and follow them.

```bash
pixi run --frozen -e robodojo-native smoke                       # native checks via worker processes
ROSBRIDGE_ALL_TASKS=1 pixi run --frozen -e robodojo-native smoke  # every task boots Isaac (hours)
pixi run --frozen -e robodojo test-ros
rosbridge demo robodojo --timeout 1800
```

The native checks run the backend in worker subprocesses because Isaac's shutdown
ends its process. `tests/kinex/native/test_robodojo.py` (with `KINEX_PYTHON`) checks the
exported rig's arm FK, finger links and camera mounts against Isaac.
