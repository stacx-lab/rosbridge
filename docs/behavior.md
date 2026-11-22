# BEHAVIOR

BEHAVIOR-1K's 100 household activities of the 2026 challenge run in OmniGibson on
Isaac Sim 5.1 with BEHAVIOR's R1Pro, as the official evaluator configures it: devices
`base`, `torso`, `left_arm`, `right_arm`, `left_gripper`, `right_gripper` and RGB-D
`head_camera`, `left_camera` and `right_camera`. Upstream is pinned to
`a8247a8cc1633fe1ca0cc66aa07243d46c64f155`, as RoboEnv pins it. See the
[design and CLI guide](README.md) for shared lifecycle and timing contracts, and [BEHAVIOR:
known environment issues](behavior-known-issues.md) for what is known to be wrong or
surprising in the activities, and which of it rosbridge changes.

```bash
ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES rosbridge prepare behavior
rosbridge tasks list --simulator behavior
rosbridge launch behavior/turning-on-radio --execution-mode stepped --timeout 1800
rosbridge setup behavior/turning-on-radio --directory ../my-behavior-workspace
```

Task IDs are kebab-cased activity names, such as `behavior/turning-on-radio`.

## Episodes and judging

Each episode follows the official evaluator (`OmniGibson/omnigibson/eval`) for one
environment. The activity loads in its scene with the rooms the evaluator loads
(`B100_task_misc.csv`), with its three disabled transition rules (toggleable
machines, mixing tools and cooking systems) and light synchronization for
`turning_out_all_lights_before_sleep`. Python, NumPy, Torch and Warp are seeded once with
the evaluator's seed 0 before the environment is built, as `omnigibson/eval/eval.py` does;
the seed selects public test instance `301 + seed % 20`. Loading it restores the
instance's objects and the robot's start pose and settles 25 physics steps holding
task entities still, as the evaluator's `load_batch` does; `env.reset` then places the
robot at the instance's presampled pose again, as there. OmniGibson loads every door of
the scene but the floors of the loaded rooms only, so a door can open onto no floor: a door
with a loaded room on one side of the scene's room segmentation and none on the other is
closed and locked. No such door is part of an activity. BEHAVIOR's sampler places a floor's
particles wherever the floor is traversable, and a floor can reach past a locked door, so
loading removes the visual particles that the room segmentation places in a room that is
not loaded. The episode record names them in `particles_removed_out_of_loaded_rooms`. Among
the public instances, these are `sweeping_garage`'s two at seed 0 and one at seed 12, outside
its garage door.

Upstream's evaluator gives its policy no text, only the activity's task id
(`eval_utils.TASK_NAMES_TO_INDICES`), so the [task description](devices.md#task-description)'s
`instruction` is the activity's name, such as `turning_on_radio`.

Success is the activity's BDDL goal predicate, evaluated every step; the reward is
1 while it holds. `--early-stop` ends the episode at the first success, as the
evaluator's termination does; BEHAVIOR has no failure termination. The default horizon is
the evaluator's timeout, 1.5 average human demonstrations (`task.jsonl`), for example 3224
for `turning_on_radio`: upstream's `Timeout` checks the step count before counting the
step, so the episode ends on the step after it, step 3225 (107.5 s). The world node's `max_steps` parameter overrides the
horizon.

Episode records add the evaluator's partial credit, `q_score`, as its `TaskMetric`
computes it: 1 on success, else the best grounded goal option's fraction of literals
that hold at the end and did not when the instance loaded. Quantified goals ground to
many options that repeat the same literals (`assembling_gift_baskets`: 331,776
options of 16 literals, 64 distinct), so OmniGibson is patched to evaluate each
distinct literal once (`patches/behavior/0003-goal-option-literals.patch`).

Two of the rules OmniGibson judges by are patched:

| Patch | OmniGibson | Patched |
|---|---|---|
| `0005-slicer-blade.patch` | A sliceable object is halved, and a half diced, when it touches any part of any slicer while any slicer of the scene is active. A slicer turns inactive after it touches whole food, and active again 2 s after it last touched whole food. | An object is cut by a slicer that is active, touches it and reaches it with its blade: the box of its slicer meta link, BEHAVIOR's cutting volume, where that link is at the step, 5 mm larger on every side. The slicer that cut turns inactive, and active again 2 s after it last touched whole or cut food. A slicer without such a meta link cuts with its whole body. |
| `0006-projection-remover-margin.patch` | A particle remover by projection takes a visual particle whose centre lies inside its volume. | It takes one whose centre lies within 1 cm of its volume. |

Both change what is judged, so on the activities they decide, results do not compare with
the evaluator's ([known issues](behavior-known-issues.md), 5.1 and 5.2), and neither do
`sweeping_garage`'s at the two seeds whose particles loading removes (6.1).

Unpatched, a knife's handle cuts what it touches (`make_pizza` takes its dough and a
pepperoni whole), an inactive knife cuts while another is active, and a knife lying on
the halves it cut dices them. A blade cuts at the first contact at any speed; there is
no speed condition. The slicer rule decides the nine activities with a slicer and food
it cuts, such as `chopping_wood`, `halve_an_egg` and `make_pizza`. `vacuuming_floors`'
vacuum is BEHAVIOR's only particle remover by projection. OmniGibson sets its volume
below the vacuum's meta link, ending 1.1 mm under the floor it stands on, so unpatched it
removes floor dust only while lifted 1.2 to 21.5 mm or leaning by a degree or more.

Isaac hosts one environment per process: a worker serves the one activity and
instance it was launched with.

## Native worker

Isaac Sim 5.1 publishes Python 3.11 wheels only, so the world node runs in the
shared ROS environment (`behavior`) and proxies the backend in a worker in
`behavior-native`: Isaac Sim 5.1, CUDA 12.8 torch 2.7 and OmniGibson's requirements at
RoboEnv's resolved versions, with the generated tree's OmniGibson and BDDL installed
editable. The worker simulates and renders on one GPU, the first
in `CUDA_VISIBLE_DEVICES` (default 0), and hides the others: on IOMMU hosts Kit
otherwise validates peer-to-peer access between every GPU, holding each for minutes.
`CUDA_VISIBLE_DEVICES=3 rosbridge launch ...` selects GPU 3. Booting Kit and loading a house takes several minutes (the first boot on a host
also compiles kernels); allow `--timeout 1800`. `OMNI_KIT_ACCEPT_EULA` defaults to
`YES`.

Preparation exports BEHAVIOR-1K into `.runtime/behavior` and applies RoboEnv's
two patches: OmniGibson's dataset-recording and LeRobot wrappers become optional
imports and extras, and `OMNIGIBSON_RENDER_GPU` selects the Vulkan renderer apart
from the CUDA physics device.

OmniGibson's macro particle systems updated the physics handles, which rebuilds every
physics view, after each particle they added or removed, twice per physical particle
removed. `patches/behavior/0004-particle-handle-updates.patch` updates them once per
call that adds or removes physical particles, and once per call for visual particles
removed too: removing any prim drops the articulation views' handles, and through a
dropped handle a joint read fails and a joint or pose write does nothing. In
`make_microwave_popcorn` the cooking rule's 268 particles took 165 s of one step and
take 33 s, most of it creating their prims; removing 200 dust particles took 31 s and
takes 1 s.

## Assets and license

The BEHAVIOR Data Bundle is encrypted and licensed separately from OmniGibson;
preparation requires `ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES`, which accepts the
BEHAVIOR Data Bundle End User License Agreement, and then installs the dataset key
with OmniGibson's own `download_key`. The three archives of
`behavior-1k/zipped-datasets` at `9f0d57d4…` are pinned by SHA-256 and size:
`behavior-1k-assets` 3.9.0 (31 GB), `omnigibson-robot-assets` 3.8.2 and
`2026-challenge-task-instances`, about 36 GB extracted (downloading also needs
space for the largest archive). They are cached once under
`.runtime/assets/behavior` and linked into the data root, `.runtime/behavior/datasets`,
which OmniGibson reads. Every activity shares them, so one preparation serves all
activities; a changed pin or generated source requires preparing again.

With `ROSBRIDGE_ASSET_SEED=DIR`, an extracted depot
(`DIR/behavior/datasets/<dataset>`) is verified file by file against its
receipts (`DIR/behavior/<dataset>.json`, which must name the pinned archive) and
moved into the cache; the key is copied:

```bash
ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE=YES ROSBRIDGE_ASSET_SEED=DIR rosbridge prepare behavior
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

Each `stop` holds the measured joints, keeps a gripper's applied grip target (keeping any
grasp), or commands zero base velocity. Cameras are the evaluator's full-resolution RGB-D set: the
720×720 head camera (fx = 306) and 480×480 wrist cameras (fx ≈ 388.66), with
OmniGibson's `depth_linear` along the optical axis and the principal point at the
image center, `(width-1)/2` in ROS pixel coordinates.

The rig `behavior/behavior-r1pro` chains `torso` from the unlocalized base
(`base_link`), the arms and head camera from `torso:top` (`torso_link4`), and each
gripper and wrist camera from its arm's flange (`<side>_arm_link7`). Its TCP is the
native eef frame between the fingertips.

## Timing, cameras and recording

The R1Pro steps at 30 Hz over 120 Hz physics. Each step is the evaluator's `env.step`: physics, then one render of every camera
with OmniGibson's own anti-aliasing (DLSS Performance), whether or not anyone reads a
camera. A camera read returns the latest frame and neither renders nor steps. OmniGibson
delivers a render's frame at the next render, so a frame shows the state one step before
its stamp, as the evaluator's observations do; the load's re-renders make the first
frames show the loaded instance. `turning_out_all_lights_before_sleep` re-renders three
times after each light sync, as the evaluator does.

A house's physics dominates the step time; stepping is slower than real time.

With `--record`, spectators `front_left`, `rear_left` and `head` (the head camera's
optics) render at 960×720. Every rendered view costs stepping time; `--record-views`
and `--record-size` (see the [README](README.md)) make a recorded run faster. The two
overviews stand 45° and 135° to the left of the robot's heading and keep those
bearings. They turn after the robot with a one-second lag, so gait sway does not
shake them. An overview moves in when something stands between it and the robot, or
when it would leave the robot's room by the scene's room segmentation (a ray test
alone let it film through windows from outside the house). They keep the distance and
heights of RoboEnv's BEHAVIOR overview: 2.6 m out, the eye 2.4 m above the floor and
aimed at 0.9 m.
The spectators render with every step, and loading settles them at their start pose with
16 renders; a spectator sample reads the latest frame.

```bash
pixi run --frozen -e behavior-native smoke   # controls, cameras, reset, slicer and vacuum rules
pixi run --frozen -e behavior test-ros
rosbridge demo behavior --timeout 1800
```

The rig is rebuilt with `tools/robots/build_behavior.py` in the native environment.
Native checks boot
one activity (`ROSBRIDGE_ALL_TASKS=1` loads every R1Pro activity).
`tests/kinex/native/test_behavior.py` (with `KINEX_PYTHON`) compares the exported rig's FK
with OmniGibson.
