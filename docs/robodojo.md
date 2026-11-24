# RoboDojo

RoboDojo runs its 42 simulation tasks on the dual ARX X5 rig in Isaac Sim 5.1,
with upstream's saved evaluation layouts. The integration follows RoboEnv's
RoboDojo device runtime. Upstream is pinned to
`c3af1ac6ad7302c0c09d17f4b050fd3a5c68c921`, with its Isaac Lab fork
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
rosbridge launch robodojo/stack-bowls --config move --execution-mode stepped --timeout 1800
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
`1127bda94f172d757386945af3eb3a6754d1a4e6`: 15,365 files (41 GB) under
`Assets/`, each pinned by LFS SHA-256 or git blob SHA-1 in the dataset listing.
With `ROSBRIDGE_ASSET_SEED=DIR`, a depot laid out as `DIR/robodojo/<revision>/Assets` is
verified file by file and moved into the store; otherwise files are downloaded
once at a polite request rate. A seed of hard links to an earlier revision's files, with the
changed files replaced, bumps the pin without copying.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/pose_state`, `<arm>/stop`; `<arm>/move` in place of `<arm>/joint_target` with `--config move` |
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

The flange is `link6`, upstream's `ee_link` and cuRobo's tool frame, and `pose_state` reports its
measured pose in `<arm>/base`.

### Move configuration

`--config move` gives each arm one command, `<arm>/move`, in place of `joint_target`. It is a
`control_msgs/action/FollowJointTrajectory`: a goal's `multi_dof_trajectory` gives the `flange` poses,
in `<arm>/base`, that the arm passes in turn at its own pace, so a caller chooses no times. It is the
stepped, closed-loop counterpart of uni-robot-agent's `move` tool, and nothing plans a path around
objects. The action is this backend's own (`rosbridge_robodojo.move`): it advances each goal inside its
step, and the world only carries the goal, its feedback and its result (see
[Actions](adding-a-simulator.md#actions)):

1. Each 0.04 s frame commands a pose at most one step from the measured one, 4 cm and 0.13 rad at the
   device pace of 1 m/s and 3.3 rad/s, so a blocked flange is never driven far past where it is.
2. cuRobo's `InverseKinematics`, the solver upstream's `ee_pose` action uses (32 seeds, self and table
   collision, 1 mm and 0.02 rad), solves that pose from the measured joints (`RoboDojoSim.joint_step`).
   Of its solutions, the one nearest the measured joints is taken that lies within the joint limits, a
   whole turn nearer where they allow, asks no joint for more than 1 rad, and has a joint path, sampled
   every 0.025 rad, that passes cuRobo's self and table constraints. These checks follow
   uni-robot-agent's RoboDojo adapter, with the nearest solution in place of cuRobo's first. A joint
   asked for more than a frame's 0.2 rad (the X5 actuators' 5 rad/s `velocity_limit_sim`) saturates, and
   the next frame goes on from where the arm is.
3. A point's `time_from_start` is its nominal time: zero goes at the device pace, a later time
   proportionally slower.
4. The goal succeeds once the flange is within `component_goal_tolerance` of the last pose (`TRANSLATION`
   in metres, `ROTATION` in radians; 0.005 m and 0.05 rad when omitted).
5. It is aborted, the arm holding its measured joints, when the flange stalls (`error_code` -4: over four
   frames it gains less than a tenth of the tolerance), when it is not there `goal_time_tolerance` (1 s
   when zero) after its nominal end (-5), or when a frame has no IK solution (-1). `error_string` says
   why and how far from the goal it ended. A cancel, a newer goal, a stop or a wait's `goal_timeout_s`
   ends it the same way (-1, with that reason).

A goal advances no time. Send both arms their goals, then give the goals' ids to one `world/wait`: it ends
on the frame the last goal finishes, so the wait costs no more budget than the motion, and its
`goal_timeout_s` cancels a goal that takes longer. A goal is the world's from its acceptance; a wait that names
any other is refused.

The joints run through `EvalEnv.take_action`, as joint targets do, so evaluation and scoring are
unchanged. To solve them, rosbridge builds upstream's cuRobo planner for the arms, as upstream's
evaluation does by default (`need_planner`). RoboEnv's adapter, whose arms took joints only, skipped it.
`robodojo-native` carries cuRobo's `[cu12]` extra (`cuda-core`), as upstream's install does, because the
planner compiles its kernels through it.

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

**Instruction.** The episode's instructions are upstream's, drawn for its layout, so a seed
keeps them too. Upstream's policy receives two, and the
[task description](devices.md#task-description) holds both: `instruction`, a sentence in each
observation, and `detailed_instruction`, the task's description and scoring rubric in its
`extra_info`.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` copy the
optics of the head and wrist cameras at 960×720 and follow them. One native render fills every
camera's and every spectator view's buffers, so a step renders once however many of them are read:
a frame of the three views takes about 86 ms, where a render for each took 220 ms.

```bash
pixi run --frozen -e robodojo-native smoke                       # native checks via worker processes
ROSBRIDGE_ALL_TASKS=1 pixi run --frozen -e robodojo-native smoke  # every task boots Isaac (hours)
pixi run --frozen -e robodojo test-ros
rosbridge demo robodojo --timeout 1800
```

The native checks run the backend in worker subprocesses because Isaac's shutdown
ends its process. `tests/kinex/native/test_robodojo.py` (with `KINEX_PYTHON`) checks the
exported rig's arm FK, finger links and camera mounts against Isaac.
