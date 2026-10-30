# BEHAVIOR

BEHAVIOR-1K's 100 household activities of the 2026 challenge run in OmniGibson on
Isaac Sim 5.1, as two simulators that share one adapter
(`simulators/behavior/rosbridge_behavior`):

| Simulator | Robot | Devices |
|---|---|---|
| `behavior` | BEHAVIOR's R1Pro, as the official evaluator configures it | `base`, `torso`, `left_arm`, `right_arm`, `left_gripper`, `right_gripper`, RGB-D `head_camera`, `left_camera`, `right_camera` |
| `behavior-g1` | Unitree G1 (29 DoF) with two SharpaWave hands, walking on GR00T WholeBodyControl | `base`, `torso`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, RGB-D `head_camera` |

The activity, scene, task instance and judging are the same in both; only the
embodiment differs. Upstream is pinned to
`a8247a8cc1633fe1ca0cc66aa07243d46c64f155`, as RoboEnv pins it. See the
[design and CLI guide](README.md) for shared lifecycle and timing contracts.

```bash
ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES rosbridge prepare behavior
rosbridge tasks list --simulator behavior
rosbridge launch behavior/turning-on-radio --execution-mode stepped --timeout 1800
rosbridge setup behavior/behavior-r1pro --directory ../my-behavior-workspace

ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES rosbridge prepare behavior-g1
rosbridge launch behavior-g1/turning-on-radio --execution-mode stepped --timeout 1800
rosbridge setup behavior-g1/behavior-g1-sharpa --directory ../my-g1-workspace
```

Task IDs are kebab-cased activity names. Both simulators catalog the same
activities, so a bare name such as `turning_on_radio` is ambiguous: select
`behavior/...` or `behavior-g1/...`.

## Episodes and judging

Each episode follows the official evaluator (`OmniGibson/omnigibson/eval`) for one
environment. The activity loads in its scene with the rooms the evaluator loads
(`B100_task_misc.csv`), with its three disabled transition rules (toggleable
machines, mixing tools and cooking systems) and light synchronization for
`turning_out_all_lights_before_sleep`. The seed selects public test instance
`301 + seed % 20` and seeds Python, NumPy, Torch and Warp. Reset restores the
instance's objects and the robot's start pose, settles 25 physics steps holding
task entities still, and restarts the clock, as the evaluator's `load_batch` does.

Success is the activity's BDDL goal predicate, evaluated every step; the reward is
1 while it holds. `--stop-on-success` ends the episode at the first success, as the
evaluator's termination does. The default horizon is the evaluator's timeout, 1.5
average human demonstrations (`task.jsonl`), for example 3224 steps (107 s) for
`turning_on_radio`. The G1 steps at 50 Hz instead of the evaluator's 30 Hz and gets the
same time, 5374 steps. The world node's `max_steps` parameter overrides the horizon.

Isaac hosts one environment per process: a worker serves the one activity it was
launched with, and a reset reloads the instance in the live environment (seconds).
A reset that fails after the reload starts leaves the episode done until a reset
succeeds.

## Native worker

Isaac Sim 5.1 publishes Python 3.11 wheels only, so the world node runs in the
shared ROS environment (`behavior`, `behavior-g1`) and proxies the backend in a
worker in `behavior-native` or `behavior-g1-native`: Isaac Sim 5.1, CUDA 12.8 torch
2.7 and OmniGibson's requirements at RoboEnv's resolved versions. Each environment
installs the generated tree's OmniGibson and BDDL editable, so the two simulators
never share an installation. The worker simulates and renders on one GPU, the first
in `CUDA_VISIBLE_DEVICES` (default 0), and hides the others: on IOMMU hosts Kit
otherwise validates peer-to-peer access between every GPU, holding each for minutes.
`CUDA_VISIBLE_DEVICES=3 rosbridge launch ...` selects GPU 3. Booting Kit and loading a house takes several minutes (the first boot on a host
also compiles kernels); allow `--timeout 1800`. `OMNI_KIT_ACCEPT_EULA` defaults to
`YES`.

Preparation exports BEHAVIOR-1K into `.runtime/<simulator>` and applies RoboEnv's
two patches: OmniGibson's dataset-recording and LeRobot wrappers become optional
imports and extras, and `OMNIGIBSON_RENDER_GPU` selects the Vulkan renderer apart
from the CUDA physics device.

## Assets and license

The BEHAVIOR Data Bundle is encrypted and licensed separately from OmniGibson;
preparation requires `ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES`, which accepts the
BEHAVIOR Data Bundle End User License Agreement, and then installs the dataset key
with OmniGibson's own `download_key`. The three archives of
`behavior-1k/zipped-datasets` at `9f0d57d4…` are pinned by SHA-256 and size:
`behavior-1k-assets` 3.9.0 (31 GB), `omnigibson-robot-assets` 3.8.2 and
`2026-challenge-task-instances`, about 36 GB extracted (downloading also needs
space for the largest archive). They are cached once under
`.runtime/assets/behavior` and linked into each simulator's data root,
`.runtime/<simulator>/datasets`, which OmniGibson reads. Every activity shares them, so
one preparation serves all activities; a changed pin, generated source or G1 importer
requires preparing again.

With `ROSBRIDGE_ASSET_SEED=DIR`, RoboEnv's extracted depot
(`DIR/behavior/datasets/<dataset>`) is verified file by file against RoboEnv's
receipts (`DIR/behavior/<dataset>.json`, which must name the pinned archive) and
moved into the cache; the key is copied:

```bash
ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES ROSBRIDGE_ASSET_SEED=../RoboEnv/.cache/roboenv/assets \
    rosbridge prepare behavior
```

## R1Pro

The R1Pro runs with the evaluator's `eval/r1pro.yaml` and settings: action
frequency 30 Hz over 120 Hz physics, assisted grasping, a 250 kg base so contact
with the house does not shove it, and the head camera's horizontal aperture of 40.
Commands are the evaluator's action vector:

- `base/velocity` carries the normalized base action (`linear.x`, `linear.y`,
  `angular.z` in [-1, 1]) that its holonomic controller scales to 0.75 m/s and
  1 rad/s in the base frame. The base stops 0.5 s of world time after the last
  twist; `base/odometry` reports the measured body twist and the planar pose of
  `base_link` relative to the episode start.
- `torso` (`torso_joint1..4`) and `left_arm`, `right_arm` (`<side>_arm_joint1..7`)
  take absolute joint positions in radians within the joint limits.
- `<side>_gripper/target` is the smooth gripper command from -1 (closed) to 1
  (open): both fingers target `0.025 (v + 1)` m. An object between closing fingers
  is held by assisted grasping. `joint_state` reports the command the mean finger
  opening reaches.

Each `stop` holds the measured joints or the commanded gripper opening, or commands
zero base velocity. Cameras are the evaluator's full-resolution RGB-D set: the
720×720 head camera (fx = 306) and 480×480 wrist cameras (fx ≈ 388.66), with
OmniGibson's `depth_linear` along the optical axis and the principal point at the
image center, `(width-1)/2` in ROS pixel coordinates.

The rig `behavior/behavior-r1pro` chains `torso` from the unlocalized base
(`base_link`), the arms and head camera from `torso:top` (`torso_link4`), and each
gripper and wrist camera from its arm's flange (`<side>_arm_link7`). Its TCP is the
native eef frame between the fingertips.

## G1 with Sharpa hands

`behavior-g1` replaces the R1Pro with a Unitree G1 29-DoF (rev 1.0) carrying two
SharpaWave hands, composed from pinned sources: `unitree_ros`
(`deeb607f`, BSD-3) for the G1 and `sharpa-urdf-usd-xml` (`0d19cac6`, Apache-2.0)
for the flange-mounted hands, which replace the G1's rubber hands at their palm
mounts (41.5 mm along the wrist, palms toward the body's midline, the index finger
on the thumb side). Preparation composes the URDF and imports it once with
OmniGibson's own import functions into the asset cache, keyed by the sources and the
adapter; it skips the importer's canonical-frame shift, which with Isaac Sim 5.1's
URDF importer moves the pelvis geometry but not its joints, so the pelvis is the
robot's root. Joint drives are force drives with GR00T's motor gains for the body
and Sharpa's Isaac calibration (stiffness, damping, armature, friction) for the
hands. The G1 wears its URDF colours (the importer would bind white), and the hands
wear SharpaWave's textured high-fidelity visuals on the same links; the flanges and
wrists keep their URDF meshes. Collision geometry is the URDF's.

The legs and waist run GR00T WholeBodyControl's decoupled lower body
(`songlin/decoupled_wbc` at `5b304e4d`, as RoboEnv's SIMPLE G1 pins it, with its
metric-command patch; the ONNX balance and walking policies are under the NVIDIA
Open Model License): every 20 ms it reads the body joints, the pelvis orientation
and body-frame angular velocity, and the commands, and sets the 15 lower-body
position targets. Physics runs at GR00T's 200 Hz. Commands:

- `base/velocity` is a body twist in SI units within GR00T's navigation limits
  (`vx`, `vy` in ±0.5 m/s, `wz` in ±0.5 rad/s). Tracking is approximate: a 0.3 m/s
  command walks about 0.2 m/s. The twist expires 0.5 s after the last one, as for
  every base.
- `torso/joint_target` sets the waist (yaw, roll, pitch); its torso orientation is
  GR00T's torso command, which the lower body tracks while it balances.
- `left_arm`, `right_arm` (7 joints) and `left_hand`, `right_hand` (22 joints) take
  absolute joint positions on position drives; grasping is physical.

OmniGibson makes the links below a robot's moving joints weightless, as ideal gravity
compensation. The G1 needs its weight to walk, so gravity acts on the pelvis, legs
and torso; the arms and hands keep OmniGibson's compensation, as RoboEnv's SIMPLE G1
compensates its arms, and track their targets without sagging (their own weight does
not load the torso). Each reset places the pelvis at standing height above the
instance's R1Pro start pose and runs GR00T's standing initialization outside the
episode, as RoboEnv's SIMPLE G1 does, until the base is still (within 6 s). The
policies were trained for a G1 with Dex3 hands in MuJoCo; here they balance the
heavier Sharpa hands in PhysX. A fall is not recovered; reset the world.

The head camera is an RGB-D camera on the G1's D435 mount with the R1Pro head
camera's optics (720×720, fx = 306). The rig `behavior-g1/behavior-g1-sharpa` chains
`torso` from the base (the pelvis), the arms and head camera from `torso:top`
(`torso_link`) and each hand from its arm's flange (`<side>_wrist_yaw_link`); hands
are `joint` devices without a tip link.

## Timing, cameras and recording

The R1Pro steps at 30 Hz over 120 Hz physics, the G1 at 50 Hz over 200 Hz. Physics
steps exactly as the evaluator's, but without rendering every period: cameras render
on demand, until the renderer delivers two frames of the current state (the first
after a change carries the renderer's temporal history). The real-time path tracer
needs that history: when cameras were last rendered more than 0.25 s of world time
earlier (after a reset, or when a stream starts), a capture renders 16 frames, about a
second, to reach the noise of streamed captures. Cameras anti-alias with DLAA at native
resolution instead of OmniGibson's DLSS Performance upscaling, which cuts the
frame-to-frame noise of on-demand captures by a third. A house's physics dominates the
step time; stepping is slower than real time.

With `--record`, spectators `front_left`, `rear_left` and `head` (the head camera's
optics) render at 960×720. The two overviews stand 45° and 135° to the left of the
robot's heading and keep those bearings. They turn after the robot with a one-second
lag, so gait sway does not shake them, and an overview moves in only when something
stands between it and the robot. For the R1Pro they keep the distance and heights of
RoboEnv's BEHAVIOR overview: 2.6 m out, the eye 2.4 m above the floor and aimed at
0.9 m. For the G1 they stand 1.8 m out, the eye 1.4 m above the floor and aimed at
0.8 m, so the whole body and the hands stay in view.

```bash
pixi run -e behavior-native smoke        # R1Pro: controls, cameras, reset, rig geometry
pixi run -e behavior-g1-native smoke     # G1: standing, walking, turning, arms, hands, rig geometry
pixi run -e behavior test-ros
pixi run -e behavior-g1 test-ros
rosbridge demo behavior --timeout 1800
rosbridge demo behavior-g1 --timeout 1800
```

Rigs are rebuilt with `tools/robots/build_behavior.py` and
`tools/robots/build_behavior_g1.py` in the native environments. Native checks boot
one activity (`ROSBRIDGE_ALL_TASKS=1` loads every R1Pro activity); with
`KINEX_PYTHON`, they compare the exported rigs' FK with OmniGibson.
