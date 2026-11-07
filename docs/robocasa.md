# RoboCasa

RoboCasa runs in its own Python 3.12 / ROS Jazzy environment with the official
RoboCasa365 kitchens, tasks, success predicates and PandaOmron mobile manipulator: a
Franka Panda on a lift torso on an Omron holonomic base. The catalog holds all 365
tasks (65 atomic, 300 composite); 50 are upstream's evaluation tasks (`TARGET_TASKS`):

```bash
rosbridge prepare robocasa
rosbridge tasks list --simulator robocasa
rosbridge launch robocasa/pick-place-counter-to-cabinet --execution-mode stepped --record ../ros_robocasa/run
```

`robocasa` selects `robocasa/pick-place-counter-to-cabinet`; task IDs are the
kebab-cased native class names, which also resolve as bare names. Upstream RoboCasa
is pinned to `456174f62b89b8fca99eaaf33949c29fec9cfc2a` and robosuite to
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c` (1.5.2), exported into the same
generated tree. Three patches make object models load from a private temporary
file instead of writing into the shared asset cache, make counter placement
regions deterministic per seed, and accept GPU-UUID CUDA masks in robosuite's EGL
check. The environment pins NumPy 2.2.5 and MuJoCo 3.3.1, which RoboCasa asserts;
its learning and teleoperation extras (torch, LeRobot) are not installed.

## Assets

Every task samples kitchens from the fixture, texture and generative-texture
archives and objects from the objaverse, lightwheel and aigen registries, so one
preparation mounts all six upstream archives (about 23 GB) for every task. They are
cached once under `.runtime/assets/robocasa` and linked into
`.runtime/robocasa/robocasa/models/assets`, beside the files upstream tracks there;
ABC mounts the objaverse and lightwheel packs from the same cache. Upstream
publishes no checksums. With `ROSBRIDGE_ASSET_SEED=DIR`, extracted archives at
`DIR/robocasa/<root>/` (`textures`, `generative_textures`, `fixtures`, `objaverse`,
`aigen_objs`, `lightwheel`) are verified file by file against RoboEnv's sha256
receipts `DIR/robocasa/<archive>.json` and moved into the cache:

```bash
ROSBRIDGE_ASSET_SEED=../RoboEnv/.cache/roboenv/assets rosbridge prepare robocasa
```

## Graph

| Device | Kinex type | Interfaces |
| --- | --- | --- |
| `world` | world | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `base` | locomotion | `base/velocity`, `base/odometry`, `base/stop` |
| `torso` | joint | `torso/joint_state`, `torso/joint_target`, `torso/stop` |
| `arm` | joint | `arm/joint_state`, `arm/joint_target`, `arm/stop` |
| `gripper` | gripper | `gripper/joint_state`, `gripper/target`, `gripper/stop` |
| `camera_left`, `camera_right`, `camera_wrist` | rgbd | `<camera>/{image,depth,camera_info}` |

The cameras are upstream's `robot0_agentview_left`, `robot0_agentview_right` (on
the torso lift) and `robot0_eye_in_hand`. The rig's frames chain the robot:
`torso` mounts on `base:base`, the arm on `torso:top` and the gripper and wrist
camera on `arm:flange`. The base is an unlocalized root; nothing relates it to
`world`.

## Base, torso, arm and gripper

All parts use upstream's controllers at its 20 Hz control rate. RoboEnv's mapping
is kept: absolute `JOINT_POSITION` goals for the arm and torso, and the native base
and gripper commands.

- `base/velocity` (`geometry_msgs/msg/Twist`) carries upstream's normalized base
  action: `linear.x`, `linear.y` and `angular.z` in [-1, 1], in the body frame (the
  native kitchen controller already rotates them by the base's heading). Other
  fields must be zero. As Kinex locomotion requires, the base stops 0.5 s of world
  time after the last twist, so re-send to keep moving; `base/stop` commands zero.
  `base/odometry` (`nav_msgs/msg/Odometry`) reports the measured body twist in m/s
  and rad/s at the chassis origin (21 cm ahead of the yaw axis) in `base/base`, and
  the planar pose relative to the episode start in `base/odom`.
- `torso/joint_target` and `arm/joint_target` take position-only
  `JointTrajectory` points for `mobilebase0_joint_torso_height` (metres, 0 to 0.34)
  and `robot0_joint1..7` (radians), as for the other simulators. Each episode
  starts at upstream's stowed posture with its 0.02 rad initialization noise and
  the torso at 0. Upstream settles every new kitchen with zero actions before the
  episode starts; the joint parts take native delta input during that settling, so
  the posture holds, and absolute input afterwards.
- `gripper/target` (`std_msgs/msg/Float64`) is robosuite's integrated gripper
  command from -1 (open) to +1 (closed). Every control step the backend moves it by
  the native 0.2 toward the target while that gets strictly closer, so -1 and +1 are
  reached exactly within 10 steps; a native policy's gripper action `a` maps to the
  target `sign(a)`. `gripper/joint_state` reports this command, not the measured
  fingers: the URDF fingers mimic the position their actuators are commanded to,
  so a closed grip may be stalled on an object.

Each `stop` holds the measured joint positions, the gripper's integrated command
(keeping any grip), or zero base velocity.

## Episode timing and tasks

Each action interval advances 0.05 simulation seconds (25 MuJoCo substeps). The
seed fully determines the kitchen layout and style, objects, placements and
instruction; building the kitchen takes about 10 s. Tasks marked
`split = "target"` in `tasks.toml` use upstream's evaluation split, as
`create_env(split="target")` does: held-out object instances in kitchens 1–10 with
matching layout and style. The other tasks sample all kitchens and objects. The
default horizon is upstream's dataset horizon for the task (for example 750 steps
for `PickPlaceCounterToCabinet`), or none when upstream has no dataset; `--max-steps`
overrides it. Success is Kitchen's native predicate, evaluated once per step. The
native language instruction is printed to the run log when the episode starts;
rosbridge does not publish it.

## Cameras, recording and demo

Observation cameras are 256×256 RGB-D in frames `<camera>/base` (vertical FOV 60°
for the agentview cameras and 75° for the wrist), with principal point
`(width-1)/2, (height-1)/2`. RGB keeps robosuite's antialiased rendering; depth
comes from a separate single-sample renderer at pixel centres, in millimetres along
the optical axis. The near clip plane scales with each kitchen's extent (about
2–3 cm). With `--record`, spectator views render `robot0_agentview_left` and
`robot0_eye_in_hand` into `third_person.mp4` and `wrist.mp4`.

```bash
rosbridge demo robocasa --record ../ros_robocasa/demo
rosbridge demo robocasa/prepare-veggie-dip --execution-mode realtime --seed 2
```

The finite ROS-only demo moves the arm and torso, closes and opens the gripper,
turns the base and checks that it stops, and checks all three cameras. It reports
measurements as JSON and shuts down the graph.

Installed sources live in `.runtime/robocasa`, assets in `.runtime/assets/robocasa`,
and the ROS workspace in `.workspaces/robocasa`. The rig is rebuilt with
`pixi run -e robocasa python tools/robots/build_robocasa.py`.

## Checks

```bash
pixi run -e robocasa smoke      # representative tasks, controls, rendering and rig geometry
pixi run -e robocasa test-ros
```

Native checks cover an atomic and a composite evaluation task and a task without a
dataset horizon, not all 365. Rig geometry compares Kinex's FK through base, torso
and arm with MuJoCo and needs `KINEX_PYTHON`.
