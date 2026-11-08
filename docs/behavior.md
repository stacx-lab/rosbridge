# BEHAVIOR

BEHAVIOR-1K's 100 household activities of the 2026 challenge run in OmniGibson on
Isaac Sim 5.1, as three simulators that share one adapter
(`simulators/behavior/rosbridge_behavior`):

| Simulator | Robot | Devices |
|---|---|---|
| `behavior` | BEHAVIOR's R1Pro, as the official evaluator configures it | `base`, `torso`, `left_arm`, `right_arm`, `left_gripper`, `right_gripper`, RGB-D `head_camera`, `left_camera`, `right_camera` |
| `behavior-g1` | Unitree G1 (29 DoF) with two SharpaWave hands, walking on GR00T WholeBodyControl | `base`, `torso`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, RGB-D `head_camera` |
| `behavior-dex` | BEHAVIOR's R1Pro with two SharpaWave hands in place of its grippers | `base`, `torso`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, RGB-D `head_camera`, `left_camera`, `right_camera` |

The activity, scene, task instance and judging are the same in all three; only the
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

ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES rosbridge prepare behavior-dex
rosbridge launch behavior-dex/turning-on-radio --execution-mode stepped --timeout 1800
rosbridge setup behavior-dex/behavior-r1pro-sharpa --directory ../my-dex-workspace
```

Task IDs are kebab-cased activity names. The simulators catalog the same
activities, so a bare name such as `turning_on_radio` is ambiguous: select
`behavior/...`, `behavior-g1/...` or `behavior-dex/...`.

## Episodes and judging

Each episode follows the official evaluator (`OmniGibson/omnigibson/eval`) for one
environment. The activity loads in its scene with the rooms the evaluator loads
(`B100_task_misc.csv`), with its three disabled transition rules (toggleable
machines, mixing tools and cooking systems) and light synchronization for
`turning_out_all_lights_before_sleep`. The seed selects public test instance
`301 + seed % 20` and seeds Python, NumPy, Torch and Warp. Loading it restores the
instance's objects and the robot's start pose and settles 25 physics steps holding
task entities still, as the evaluator's `load_batch` does. OmniGibson loads every door of
the scene but the floors of the loaded rooms only, so a door can open onto no floor: a door
with a loaded room on one side of the scene's room segmentation and none on the other is
closed and locked. No such door is part of an activity.

Success is the activity's BDDL goal predicate, evaluated every step; the reward is
1 while it holds. `--stop-on-success` ends the episode at the first success, as the
evaluator's termination does. The default horizon is the evaluator's timeout, 1.5
average human demonstrations (`task.jsonl`), for example 3224 steps (107 s) for
`turning_on_radio`. `behavior-dex` keeps the R1Pro's 30 Hz and horizon. The G1 steps at
50 Hz instead of the evaluator's 30 Hz and gets the same time, 5374 steps. The world node's `max_steps` parameter overrides the horizon.

Episode records add the evaluator's partial credit, `q_score`, as its `TaskMetric`
computes it: 1 on success, else the best grounded goal option's fraction of literals
that hold at the end and did not when the instance loaded. Quantified goals ground to
many options that repeat the same literals (`assembling_gift_baskets`: 331,776
options of 16 literals, 64 distinct), so OmniGibson is patched to evaluate each
distinct literal once (`patches/behavior/0003-goal-option-literals.patch`).

Isaac hosts one environment per process: a worker serves the one activity and
instance it was launched with.

## Native worker

Isaac Sim 5.1 publishes Python 3.11 wheels only, so the world node runs in the
shared ROS environment (`behavior`, `behavior-g1`, `behavior-dex`) and proxies the
backend in a worker in `behavior-native`, `behavior-g1-native` or
`behavior-dex-native`: Isaac Sim 5.1, CUDA 12.8 torch 2.7 and OmniGibson's
requirements at RoboEnv's resolved versions. Each environment installs the generated
tree's OmniGibson and BDDL editable, so the simulators never share an installation. The worker simulates and renders on one GPU, the first
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
one preparation serves all activities; a changed pin, generated source or robot
importer requires preparing again.

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
not load the torso). Loading places the pelvis at standing height above the
instance's R1Pro start pose and runs GR00T's standing initialization outside the
episode, as RoboEnv's SIMPLE G1 does, until the base is still (within 6 s). The
policies were trained for a G1 with Dex3 hands in MuJoCo; here they balance the
heavier Sharpa hands in PhysX. A fall is not recovered.

The head camera is an RGB-D camera on the G1's D435 mount with the R1Pro head
camera's optics (720×720, fx = 306). The rig `behavior-g1/behavior-g1-sharpa` chains
`torso` from the base (the pelvis), the arms and head camera from `torso:top`
(`torso_link`) and each hand from its arm's flange (`<side>_wrist_yaw_link`); hands
are `joint` devices without a tip link.

## R1Pro with Sharpa hands

`behavior-dex` is the evaluator's R1Pro with each parallel gripper replaced by a
SharpaWave hand from the G1's pinned `sharpa-urdf-usd-xml` source (one clone serves
both). Everything else is the R1Pro's: its base, torso and arm controllers and
commands, the 250 kg base, the head camera, and the wrist cameras' optics,
resolution and sensor names (the wrist cameras move to the backs of the hands).

- Each hand's flange sits on `<side>_arm_link7` at the gripper's mount face, the top
  of the gripper body: 29.5 mm behind link7's axis and 84 mm below its origin. The
  fingers continue the gripper's approach axis; with the arms at zero, hanging at the
  sides, the palms face the robot's midline and the thumbs lead forward.
- Each wrist camera, the R1Pro's RealSense with its optics, resolution and sensor
  name, sits on the back of its hand: 100 mm behind the palm's axis at the wrist,
  looking along the fingers turned 10° toward the palm side. The open hand, fingertips
  included, fills the bottom sixth of the image and the scene beyond the fingertips the
  rest. At the R1Pro's own pose a Sharpa hand covers the lens from 2 cm. Objects held
  against the palm are behind the hand from there. Collisions between each camera and
  the palm and thumb, and between link7 and the wrist and palm, are disabled. The
  R1Pro keeps the evaluator's self-collisions; within each hand the pairs that Sharpa's
  own model filters (the palm and each finger's base) do not collide.
- With the hands held up in front of the body to work, for example the wrists 0.4 m
  ahead of the base and 0.97 m high with the fingers forward, the head camera sees
  both hands in the lower half of its image (the evaluator's torso posture and head
  optics); the native checks drive two such postures and require every fingertip and
  palm in view and unoccluded. With the arms raised, the wrist cameras look out over
  the scene past the fingertips.
- `left_hand`, `right_hand` (22 joints each, as on the G1) take absolute joint
  positions on Sharpa's calibrated position drives and start open and flat (all
  zero); the torso starts in the evaluator's posture. Grasping is physical, by the
  fingers' contacts: there is no assisted-grasp constraint. The fingertips' elastomer
  pads count as finger links, so a press with a fingertip toggles a switch (OmniGibson
  counts only finger links' contacts, and a fingertip touches through its pad).

Preparation imports both hands with OmniGibson's URDF importer (convex hulls of
Sharpa's meshes, Sharpa's drive calibration and textured high-fidelity visuals) at
their final poses, then grafts them with Sdf onto the R1Pro's shipped USD in place of
the grippers, so the R1Pro's body, collision hulls and holonomic base stay exactly as
shipped. The model is cached like the G1's, keyed by the sources, the robot assets and
the importer. OmniGibson's gravity compensation makes the arms and hands weightless,
so the heavier hands (1.3 kg each, the gripper and camera 0.5 kg) do not load the
arms. The rig `behavior-dex/behavior-r1pro-sharpa` is the R1Pro rig with each hand a
`joint` device without a tip link on its arm's flange (`<side>_arm_link7`), and each
wrist camera mounted from that flange at its place on the back of the hand.

## Timing, cameras and recording

The R1Pro and `behavior-dex` step at 30 Hz over 120 Hz physics, the G1 at 50 Hz over
200 Hz. Physics steps exactly as the evaluator's but never renders, so it does not
depend on whether anyone reads a camera; reads render afterwards. Zero-delay update
ordering makes each render deliver the state it rendered, so frames always show the
current state (the evaluator's frames lag its physics by one step).

- The R1Pro embodiments render as the evaluator does: once per step, with OmniGibson's
  own anti-aliasing (DLSS Performance). The first camera read after a step renders
  once, and that render serves every camera until the next step; steps nobody reads
  never render. Without a render at the previous step the renderer has no recent
  history, so a read renders three times, as OmniGibson re-renders after a reset;
  `turning_out_all_lights_before_sleep` adds the evaluator's three re-renders after
  its light sync. The first frames come from the load's own renders. In a house, a
  step takes about 40 ms unread and 85 ms with every camera read.
- The G1 renders on demand: a capture renders until the renderer delivers two frames
  of the current state (the first after a change carries the renderer's temporal
  history of an older state). When cameras were last rendered more than 0.25 s of
  world time earlier (after loading, or when a stream starts), a capture renders 16
  frames, about a second, to reach the noise of streamed captures. It anti-aliases with
  DLAA at native resolution instead of DLSS Performance upscaling, which cuts the
  frame-to-frame noise of on-demand captures by a third.

A house's physics dominates the step time; stepping is slower than real time.

With `--record`, spectators `front_left`, `rear_left` and `head` (the head camera's
optics) render at 960×720. Every rendered view costs stepping time; `--record-views`
and `--record-size` (see the [README](README.md)) make a recorded run faster. The two
overviews stand 45° and 135° to the left of the robot's heading and keep those
bearings. They turn after the robot with a one-second lag, so gait sway does not
shake them. An overview moves in when something stands between it and the robot, or
when it would leave the robot's room by the scene's room segmentation (a ray test
alone let it film through windows from outside the house). For the R1Pro they keep
the distance and heights of RoboEnv's BEHAVIOR overview: 2.6 m out, the eye 2.4 m
above the floor and aimed at 0.9 m. For the G1 they stand 1.8 m out, the eye 1.4 m
above the floor and aimed at 0.8 m, so the whole body and the hands stay in view.
With the R1Pro embodiments, recording renders every step, so the spectators keep the
renderer's history; loading settles them at their start pose with 16 renders, and each
spectator sample renders once for every recorded view.

```bash
pixi run -e behavior-native smoke        # R1Pro: controls, cameras, reset, rig geometry
pixi run -e behavior-g1-native smoke     # G1: standing, walking, turning, arms, hands, rig geometry
pixi run -e behavior-dex-native smoke    # R1Pro with hands: cameras, mount, hands, rig geometry
pixi run -e behavior test-ros
pixi run -e behavior-g1 test-ros
pixi run -e behavior-dex test-ros
rosbridge demo behavior --timeout 1800
rosbridge demo behavior-g1 --timeout 1800
rosbridge demo behavior-dex --timeout 1800
```

Rigs are rebuilt with `tools/robots/build_behavior.py`,
`tools/robots/build_behavior_g1.py` and `tools/robots/build_behavior_dex.py` in the
native environments. Native checks boot
one activity (`ROSBRIDGE_ALL_TASKS=1` loads every R1Pro activity); with
`KINEX_PYTHON`, they compare the exported rigs' FK with OmniGibson.
