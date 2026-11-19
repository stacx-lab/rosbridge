# RoboCasa

RoboCasa runs in its own Python 3.12 / ROS Jazzy environment with the official
RoboCasa365 kitchens, tasks, success predicates and PandaOmron mobile manipulator: a
Franka Panda on a lift torso on an Omron holonomic base. The catalog holds all 365
tasks (65 atomic, 300 composite), including upstream's 300 training and 50 evaluation
tasks; a run's flags select the kitchen and objects (see
[Scenes](#scenes-split-layout-and-style)):

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

| Device | Type | Interfaces |
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
  fields must be zero. The base stops 0.5 s of world
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

Each action interval advances 0.05 simulation seconds (25 MuJoCo substeps). The task,
seed and scene options (see [Scenes](#scenes-split-layout-and-style)) determine the
episode; building the kitchen takes about 10 s. The default horizon is upstream's dataset horizon for the task (for example 750 steps
for `PickPlaceCounterToCabinet`), or none when upstream has no dataset; the world
node's `max_steps` parameter overrides it. Success is Kitchen's native predicate,
evaluated once per step; `--early-stop` ends the episode at its first success, and
RoboCasa has no failure termination. The kitchen and the native language instruction
are printed to the run log when the episode starts; rosbridge does not publish the
instruction.

## Scenes: split, layout and style

A run selects its scene as upstream does, with launch flags for upstream's own arguments:

| Flag | Values | Default | Upstream argument | Selects |
|---|---|---|---|---|
| `--split` | `pretrain`, `target`, `all` | none | `create_env(split=...)` | the kitchens the seed picks from and the object instances (below) |
| `--layout-id` | 1–60 | the seed picks | `layout_ids` | the floor plan: walls, counters, cabinets and where each appliance stands (`kitchen_layouts/*/layoutNNN.yaml`) |
| `--style-id` | 1–60 | the seed picks | `style_ids` | the materials and appliance models: cabinet, counter, floor and wall textures, which sink, stove, fridge and so on (`kitchen_styles/*/styleNNN.yaml`) |

The split is upstream's `create_env` preset, passed to the kitchen unchanged:

| `--split` | Kitchens the seed picks from | Objects (`obj_instance_split`) |
|---|---|---|
| `target` | the 10 pairs `(i, i)`, i in 1–10 | `target`: per category, the held-out last instances (up to 5) |
| `pretrain` | layouts 11–60 × styles 11–60 (2,500) | `pretrain`: the instances before those |
| `all` | layouts 1–60 × styles 1–60 | every instance |
| none | as `all`, upstream's `create_env(split=None)` | every instance |

The seed then works exactly as upstream's: it seeds the kitchen's own generator, which picks
the kitchen from the split's, the fixtures' initial state, object categories and instances
within the split, their placements, the robot's base pose and the instruction.
`--split target --seed 3` builds the same episode as upstream's
`create_env(task, split="target", seed=3)`; rosbridge also seeds Python's and NumPy's process
generators, which upstream leaves as they are. A layout or style id given overrides the
split's kitchens, as upstream's `collect_demos.py --layout/--style` does: an id not given
ranges over the split's group (1–10 for `target`, 11–60 for `pretrain`, 1–60 otherwise), and
the objects stay the split's. The seed picks only among the kitchens a task allows; an id the
task excludes fails to launch with an error naming the ids it excludes. Each episode record
states `split`, `layout_id` and `style_id`, the kitchen the seed picked or the ids pinned.

```bash
rosbridge launch robocasa/close-fridge --split target --seed 2                         # a seed-picked target kitchen
rosbridge launch robocasa/close-fridge --split target --layout-id 4 --style-id 4 --seed 2
```

### Training and evaluation splits

Upstream's benchmark trains on 300 tasks in its training kitchens and evaluates on 50 tasks
in 10 held-out kitchens, with held-out objects. rosbridge ships no datasets; the split is the
scene a run launches:

| | Train: pretraining (`pretrain300`) | Test: target (`TARGET_TASKS`) |
|---|---|---|
| Tasks | 300: 65 atomic and 235 composite | 50: 18 atomic-seen, 16 composite-seen, 16 composite-unseen |
| Flag | `--split pretrain` | `--split target` |
| Kitchens | layouts 11–60 × styles 11–60 (2,500) | the 10 pairs `(i, i)`, i in 1–10 |
| Objects | pretraining instances | held-out target instances |
| Human data | 100 demonstrations per task (`pretrain_human300`) | 500 demonstrations per task (`target50`) |

Upstream evaluates each task on 50 randomly sampled scenarios and reports the average binary
success; launching `--split target` with 50 seeds samples kitchens the same way, though the
seeds upstream's evaluation scripts use are not published. `--early-stop` ends each at its
first success.

The 34 seen test tasks are also training tasks, and the 16 unseen ones are not. Layouts and
styles 1–10, and target objects, are the test scene: a training run should never use them.
All 350 tasks are in the catalog, which also holds 49 composite tasks in neither split. The
300 training tasks are upstream's registry, `PRETRAINING_TASKS["pretrain300"]`:

```bash
pixi run --frozen -e robocasa python -c "from robocasa.utils.dataset_registry import PRETRAINING_TASKS; print(*PRETRAINING_TASKS['pretrain300'], sep='\n')" | grep -v '^WARNING'
```

The 50 test tasks (`TARGET_TASKS`, native names, which also resolve as task IDs):

| Group | Tasks |
|---|---|
| Atomic, seen (18) | CloseBlenderLid, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug, NavigateKitchen, OpenCabinet, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter, SlideDishwasherRack, TurnOffStove, TurnOnElectricKettle, TurnOnMicrowave, TurnOnSinkFaucet |
| Composite, seen (16) | DeliverStraw, GetToastedBread, KettleBoiling, LoadDishwasher, PackIdenticalLunches, PreSoakPan, PrepareCoffee, RinseSinkBasin, ScrubCuttingBoard, SearingMeat, SetUpCuttingStation, StackBowlsCabinet, SteamInMicrowave, StirVegetables, StoreLeftoversInBowl, WashLettuce |
| Composite, unseen (16) | ArrangeBreadBasket, ArrangeTea, BreadSelection, CategorizeCondiments, CuttingToolSelection, GarnishPancake, GatherTableware, HeatKebabSandwich, MakeIceLemonade, PanTransfer, PortionHotDogs, RecycleBottlesByType, SeparateFreezerRack, WaffleReheat, WashFruitColander, WeighIngredients |

Nine test tasks exclude some test kitchens, so with `--split target` the seed picks only
among the ones they allow, as upstream's does:

| Test task | Test kitchens `(i, i)` it builds |
|---|---|
| DeliverStraw, GetToastedBread, StoreLeftoversInBowl, ArrangeBreadBasket, GarnishPancake, PortionHotDogs | 2, 4, 7, 8, 9, 10 |
| RecycleBottlesByType | 2, 4, 7, 9, 10 |
| SeparateFreezerRack | 2, 7, 9 |
| WaffleReheat | all but 9 |

Likewise 116 training tasks exclude some training kitchens (RetrieveIceTray builds 450 of
the 2,500), and 30 of them exclude layout or style 11.

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
rosbridge launch robocasa/close-fridge --split target --layout-id 4 --style-id 4 --seed 0
```

The finite ROS-only demo moves the arm and torso, closes and opens the gripper,
turns the base and checks that it stops, and checks all three cameras. It reports
measurements as JSON and shuts down the graph.

Installed sources live in `.runtime/robocasa`, assets in `.runtime/assets/robocasa`,
and the ROS workspace in `.workspaces/robocasa`. The rig is rebuilt with
`pixi run --frozen -e robocasa python tools/robots/build_robocasa.py`.

## Checks

```bash
pixi run --frozen -e robocasa smoke      # representative tasks, controls and rendering
pixi run --frozen -e robocasa test-ros
```

Native checks cover an atomic and a composite evaluation task and a task without a
dataset horizon, not all 365; that a split and seed build the same episode as upstream's
`create_env` for the target and pretraining splits; kitchen ids overriding a split; and
invalid scene options and excluded kitchens. `tests/kinex/native/test_robocasa.py` (with `KINEX_PYTHON`)
compares the exported rig's FK through base, torso and arm with MuJoCo.
