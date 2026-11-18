# HandsOn

HandsOn runs its 66 dexterous-manipulation tasks on two ROKAE AR5 arms with Wuji hands
in Isaac Sim 5.1 and Isaac Lab, as its official PhysX evaluator builds them: Core-50
(`t01`–`t50`) and Transfer-16 (`eval_comb01..06`, `eval_gen01..07`, `eval_flex01..03`).
Upstream is pinned to `c9598b53fc2a01355e74837de397bd7b2b1ab602`, with Isaac Lab v2.3.2
(`isaac-sim/IsaacLab@37ddf626`) as a nested source. See the [design and CLI
guide](README.md) for shared lifecycle and timing contracts, and [HandsOn: known
environment issues](handson-known-issues.md) for what is known to be wrong or surprising in
the tasks, and which of it rosbridge corrects.

```bash
rosbridge prepare handson
rosbridge tasks list --simulator handson
rosbridge launch handson/t12-load-and-start-microwave --execution-mode stepped --early-stop --seed 3 --timeout 1800
rosbridge setup handson/t12-load-and-start-microwave --seed 3 --directory ../my-handson-workspace
```

`handson` selects `t24_place_ball_in_bowl`. Task IDs are the native names with
hyphens, such as `handson/eval-flex02-fold-towel-twice`.

## Native worker, source and assets

The world node runs in the `handson` environment (the shared ROS environment) and proxies
`rosbridge_handson.backend.HandsOnSim`, which runs on the main thread of a worker in
`handson-native`: Python 3.11, Isaac Sim 5.1, CUDA 12.8 torch 2.7 and Isaac Lab's direct
requirements, the pins `robodojo-native` shares, plus HandsOn's own. As for RoboDojo, the
worker sees one GPU, the first in `CUDA_VISIBLE_DEVICES`, and renders on its Vulkan device.

HandsOn bundles its robot, task and rendering assets in git, so nothing is downloaded:
preparation clones upstream (6.5 GB with history), exports it into `.runtime/handson`
(4.2 GB), installs `handson` and Isaac Lab's `isaaclab` package editable without
dependencies, links three of Isaac Sim's MDL modules beside upstream's archived materials (see
[Upstream corrections](#upstream-corrections)), and checks every task's budget, upstream's
SHA-256 pins of the bundled robot and rendering assets, and those links.

## Upstream corrections

Preparation corrects three upstream defects whose correction leaves the scored task
bit-identical. Every other upstream defect stays as the evaluator has it; [known
issues](handson-known-issues.md) gives the rule, the evidence and the rest.
- **Kit's present thread** (`patches/handson/0001`): Isaac Lab's headless app turns it off under
  the extension's pre-Kit-107 name, `omni.renderer.core`, so every world, idle or not, kept
  presenting about 300 frames a second, each a GPU interrupt; with several worlds per GPU a
  host can lock up. The patch uses `omni.kit.renderer.core`.
- **Materials:** three archived MDL files import Isaac Sim's `OmniGlass_Opacity`,
  `OmniPBR_ClearCoat` and `OmniPBRBase` by relative name, which MDL resolves only beside the
  importing file, so glass and some plastics in `t15`, `t27`, `t29`, `t30` and `t37` rendered
  opaque red. Preparation links the three modules there from the pinned Isaac Sim, and the
  asset check requires the links; nothing NVIDIA ships is copied into the repository.
- **`eval_gen06_spray_shoe`'s spray cone** (`patches/handson/0002`): setup hid the cone and its
  same-named child mesh, but showing a spray un-hides only the cone, so no spray was ever
  visible. The patch hides only the prim the spray toggles.

## Episodes

A launch builds the episode as upstream's `handson/evaluation/native.py:run_episode` does:
- **Seed:** the launch's seed `k`, from 0 to 9, names the task's `k`-th native seed that a
  pre-screen accepted (see [Seeds and the settle gate](#seeds-and-the-settle-gate)). For 16
  Core tasks the native seed selects a setting and derives the reset seed.
- **Task contract:** the task's frozen contract replaces every `HANDSON_*` variable.
- **Scene:** construction converts the task's MJCF scene into the run directory (about
  0.5 GB, removed when the run stops).
- **Settle gate:** after the reset, a stable-origin gate settles the scene. The pre-screen
  kept only native seeds it accepts.
- **Instruction:** the seeded instruction (canonical or paraphrased) is written with
  upstream's other receipts to `handson/PROMPT.json` in the run's directory
  (`.cache/rosbridge/runs/<id>/`). Give it to the agent; it is not published on ROS.

Booting Isaac, converting the scene and settling take minutes (about 2.5 on an L40S); allow
`--timeout 1800`. A worker serves the one episode it was launched with.

Control runs at 30 Hz: four 1/120 s PhysX substeps per step, eight 1/240 s for
`t37_scrub_bottle`. Each step also records the evaluator's per-substep PhysX ground-truth
trace, which its settle gate and contact checkers read. A step costs hundreds of milliseconds
of wall time (0.4–0.9 s on a shared L40S), so a realtime run lags the clock; prefer
`stepped`. The horizon is upstream's evaluation budget (`budgets.json`, three times the 75th
percentile of demonstration length; the `eval_flex` tasks keep their native durations), from
222 steps (`t19`) to 7200 (`eval_flex01`).

Success is upstream's `succeed` at each step. The evaluator ends a trial at its first
outcome (`evaluation/rollout.py`), and so does a run launched with `--early-stop`: at a
success, or at a task failure, whose cause the wait reply and the record name:
- `native_failed`, with the native info that explains it, such as
  `native_failed: ycb01_failure_reason=ball_or_hand_touched_gate`: 38 tasks have failure
  predicates, from a dropped object or a forbidden contact to a step order violated or a
  physics check failed;
- `native_c12_direct_hand_contact` for `t11_line_up_blocks`, the evaluator's own rule;
- `native_terminated` when the native environment ends a trial without either flag, as
  `t17_open_latched_door` can.

A step that succeeds and fails at once is a success, as the evaluator scores it. The other
27 tasks can only succeed or time out. Without `--early-stop` the episode runs to its horizon
and reports no failure; `success` is the native `succeed` of its last step, and some states
make it unreachable without failing, even with `--early-stop`: in `t02` a cable seated before
the plug, in `t20` the left hand touching the pen first, in `t31` an interrupted brew, and in
`t45` and `eval_comb06` a lost left-hand contact. Tasks whose success does not latch can also lose a success before the
horizon. `t10_lift_carton` scores support, lift and hold, recording grip force as a
diagnostic, as the evaluator does.

## Seeds and the settle gate

Every task has ten seeds, 0–9. Seed `k` names the task's `k`-th native seed in
`simulators/handson/seeds.json`, the table of a pre-screen. The pre-screen tried each task's
native seeds in the order 0, 1, 2, … and kept the first ten that it accepts: upstream's
settle gate accepts the reset, and every task body the native environment exposes ends the
gate within 2 cm of where the reset put it. The gate checks only that the scene comes to
rest. At native seed 0, for example, it accepts `t09_insert_peg` although its socket tray
slides 6 cm while settling; across the gate, bodies that stay in place move at most a few
millimetres. Both checks are deterministic, so a kept seed starts on every
launch as it did in the screen. The table also records why each native seed before the last
kept one was rejected: `settle_gate`, or `scene` with the bodies that moved. A seed outside
0–9 fails the launch, as does a native seed that the gate rejects on another machine; neither
is replaced. The episode record's `seed` is the launch's seed and `native_seed` the native one.

A native seed samples the layout, the table height and, where a task has one, its goal. `t24`
always asks for the tennis ball and `t26` for the apple: only the constructor overrides of
upstream's archived paper trials choose their targets, and rosbridge does not use those
trials. For the 16 tasks in upstream's `settings.json`, the native seed selects a setting and
derives the reset seed. The instruction is fixed by task, native seed and goal: canonical for
about one seed in five, else one of 32 paraphrases. Give the agent only `PROMPT.json`'s
`prompt`. Its `canonical_prompt` (the unparaphrased instruction), `goal` and
`native_goal_readback` must not reach the agent; `eval_gen05`'s readback also gives the
knobs' initial angles. Every receipt in `<run>/handson/` is the judge's, not the agent's:

| Receipt | Holds |
|---|---|
| `PROMPT.json` | The instruction, its draw and the goal |
| `ORIGIN_PARAMETERS.json`, `ORIGIN_IMPLEMENTATION.json`, `TRACE_IMPLEMENTATION.json`, `CHECKER_LAYOUT_PATCH.json` | The settle gate's parameters and pinned implementations |
| `SETTING_SELECTION.json` (16 tasks) | The setting, such as `t38`'s initial and target knob angles, and every derived seed |
| `TASK_BOUNDARY.json`, `TASK_BOUNDARY_SETTLE.json` (`t01`, `t15`, `t38`) | Every object pose and joint position, and the native task info |
| `CARTON_RAW_FORCE_000/` (`t10`) | Raw grip force |
| `AS019_SCENE_BRIDGE.json` (`t22`) | The scene bridge |

The settle gate is upstream's, per task contract: a frozen stable-origin gate for 34 Core
tasks, a checker-boundary gate for `t01`, `t15` and `t38`, and the current gate with its
defaults for Transfer-16. Thirteen tasks have no settle gate: `t05`, `t12`, `t13`, `t20`, `t21`,
`t30`, `t31`, `t32`, `t37`, `t42`, `t43`, `t45` and `t46`. It rejects a seed whose reset
drifts more than 5 mm or 3° while settling, penetrates, or does not come to rest in time.

The screen reset 698 native seeds and rejected 38, in 14 tasks; the other 52 tasks keep native
seeds 0–9. The same native seeds fail across tasks: 4 in 12 tasks, 6 in 10 and 8 in 7. `t23`
and `t27` reject five seeds each (`t27` keeps native seeds up to 14), `eval_gen07` four, and
`t09`'s native seed 0 is the only rejection of the scene check. Upstream rejects reset
candidates too: in its archive of 45,000 accepted paper trials, 4.8 % of candidates were
rejected, most for `t27`, `t23` and `t07`, and its campaign replaced each with the next
candidate. Its public evaluator replaces none.

The table is the output of `tools/handson_seeds.py`:

```bash
pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \
    python tools/handson_seeds.py --gpus 0,1,2,3,4,5,6,7 --per-gpu 2
```

It resets each candidate once in a process of its own, as a launch does but without ROS, and
rewrites the table after every verdict. A rerun screens only what is missing, so it resumes an
interrupted screen; `--count` extends every task to more seeds and `--tasks` narrows it. A
reset whose error is not a verdict runs once more, and a second error leaves its task
incomplete. Each GPU starts one reset at a time, and only while it has `--min-free-gb` (22 GB)
free. Every run's detail goes to `seeds.json.log.jsonl`, which is not committed.

Upstream's leaderboard (`docs/leaderboard.md` at the pin) also ranks two prompted agents on a
palm-and-finger interface like rosbridge's, GPT-6 and Fable 5.2, over ten fixed trials per
task with a budget of model calls rather than steps. Its trials are not these seeds, so the
two sets of results are not directly comparable.


## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `right_arm`, `left_arm` | `<arm>/joint_state`, `<arm>/pose_target`, `<arm>/pose_state`, `<arm>/stop` |
| `right_hand`, `left_hand` | `<hand>/joint_state`, `<hand>/joint_target`, `<hand>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/camera_info` |

Arms are commanded as upstream's EEF52 actions command them: by the pose of the palm,
`<side>_palm_link`, the arm's tip link `palm`. A `pose_target` is a
`MultiDOFJointTrajectory` of palm poses in `<side>_arm/base` (the arm's `link0_<side>`,
mounted in `world:base`, which is Isaac's world), timed from receipt. rosbridge interpolates
it at 30 Hz, positions linearly and rotations by slerp, and each step hands upstream the
palm's world pose with the hand's targets, unchanged. Everything after that is upstream's:
- **IK:** the task's own arm IK, Isaac Lab's damped least squares (λ 0.04) by default, one
  iteration per step from the measured joints, so a distant pose takes several steps, and
  the redundant elbow is the IK's, not the client's;
- **palm limits and guards** that rewrite the commanded palm before the IK, such as
  `eval_flex02`'s, `t10`'s lift guard, and the tracker-jump rejection and 0.78 m reach of
  `t21` and `eval_comb03`;
- **task mechanisms** that read it, such as the laptop hinge of `t16` and `eval_comb02`,
  whose brake releases only while the commanded right palm rises with the hand on the
  display, and `t41`'s arm-rate bound;
- **joint limits and drives:** the task contract's arm settings and stiff PD drives, which
  hold a free arm but neither compensate a load nor pass through objects.

Upstream keeps frozen copies of its base environment for some Core tasks, each with the arm
IK and control path its results were recorded on: `core_box_base` (`t03`, `t14`,
`t24`–`t26`), `core_pitcher_base` (`t04`, `t19`, `t44`, `t50`), `core_peg_base` (`t09`) and
`core_photograph_base` (`t23`). `t22` wraps the shared IK; the other 54 tasks use it as is.
Every task solves the full palm pose by damped least squares. A pose may be reached only in
part. `pose_state` reports the measured palm pose in
`<side>_arm/base` with `joint_state`'s stamp, the pose TF gives `<side>_arm/palm`. An arm
not yet commanded is upstream's idle: it holds its reset posture. A stop holds the measured
palm. Upstream treats a zero palm pose as no new command, so a commanded arm never returns
to idle.

Single-arm tasks `eval_comb01`, `eval_gen01`, `eval_gen03`, `eval_gen06` and `eval_gen07`
hold the left arm and hand at home and never ask for their targets
(`bimanual_assembly_env.py:_apply_joint_targets`). Their workspaces from `rosbridge setup`
declare no commands on those two devices, only their state, so a client that goes by the
profiles, as Kinex does, has no command to send. The other single-arm tasks drive both arms upstream: those built on the frozen bases,
which predate the hold, and the five Core tasks upstream exempts from it (`t11`, `t29`,
`t33`, `t34`, `t47`).

Hands are `joint` devices of 20 joints, `<side>_finger1_joint1`..`<side>_finger5_joint4`.
The order is finger-major from the thumb, the order of EEF52's hand20 values. Targets are
absolute joint positions in radians that pass through the native hand path, including its
target slewing. The rig `handson/handson-rokae-wuji`:
- holds the arm chains and hand trees from upstream's robot MJCF;
- ends each arm at `palm` (`<side>_palm_link`), the pose EEF52 and `pose_target` command;
- mounts each hand and wrist camera on its arm's `palm`.

Cameras are RGB and render through upstream's three-view RTX capture: its exposure per
sensor, and wrist views that hide the wrist's own cables. One capture renders all three
views at a camera sample, six path-traced passes each (about 4 s, 14 s on a shared L40S); it
never steps physics. A subscribed camera therefore samples at 2 Hz of simulation time, not
30 Hz. A new subscriber still gets the current frame at once, so a read (subscribe, take
one frame, unsubscribe) renders the current step whenever it reads. The cameras are the
evaluator's for the task and seed:
- **Head camera:** the Gemini 336's calibration (854×480, fx 463.39, fy 463.00, an
  off-center principal point) and lens, OpenCV's Brown-Conrady model (k1, k2, p1, p2, k3).
  Its profile states the lens (`opencv_brown_conrady`) and field of view, and the world node
  serves `head_camera/geometry`, `unproject` and `project` (`rosbridge_interfaces/srv/Json`)
  through that lens; images stay as rendered.
- **Wrist cameras:** undistorted pinholes of 848×480 on the palms (fx 272.14 left, 267.01
  right), or both 269.56 in `t05` and `t15`, whose camera profile is their own.
- **Principal points:** Isaac's lens places upstream's K with pixel corners at integers, so
  each profile states K's principal point less half a pixel, in ROS pixel centers: (431.27,
  239.42) for the head camera, the image center for the wrists.
- **Mounts:** each episode draws an offset per camera from its camera seed (at most 2 mm and
  0.3° per axis), as the evaluator's capture does: the setting's camera seed for the 16 tasks
  with a setting, else the native seed. The head camera is fixed in the world and each wrist
  camera on its palm, both at their nominal mount composed with that offset.

`rosbridge setup handson/TASK --seed K` exports the workspace of that task and seed: the rig
with each camera's intrinsics and drawn mount, and in the five held tasks the left arm and
hand without commands (`rosbridge_handson.workspace`, from upstream's camera contracts
without Isaac; it needs `rosbridge prepare handson`). The catalog rig itself holds the
default profile at nominal mounts.

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` follow the
head and wrist cameras at 960×720 with their horizontal field of view. Each renders alone,
two path-traced passes per sample (about 3 s), so `--record-views` and `--record-rate` set
much of a recorded run's cost.

```bash
pixi run --frozen -e handson-native smoke                       # native checks via worker processes
ROSBRIDGE_ALL_TASKS=1 pixi run --frozen -e handson-native smoke  # every task boots Isaac (hours)
pixi run --frozen -e handson test-ros
rosbridge demo handson --timeout 1800
pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \
    python tools/robots/build_handson.py                # rewrites the catalog rig
```

The native checks drive both palms by pose and check, per task, that the exported workspace
commands exactly the arms upstream drives, and that each camera's image lines up with the
geometry its profile states. `tests/kinex/native/test_handson.py` (with `KINEX_PYTHON`)
compares the exported workspace of `t24` at seed 0 (palm FK, the arm base mounts, fingertips
and drawn camera mounts) with Isaac.
