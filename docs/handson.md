# HandsOn

HandsOn runs its 66 dexterous-manipulation tasks on two ROKAE AR5 arms with Wuji hands
in Isaac Sim 5.1 and Isaac Lab, as its official PhysX evaluator builds them: Core-50
(`t01`–`t50`) and Transfer-16 (`eval_comb01..06`, `eval_gen01..07`, `eval_flex01..03`).
Upstream is pinned to `fee28452fa50d3ee906445f1717ce2449cc83798`, with Isaac Lab v2.3.2
(`isaac-sim/IsaacLab@37ddf626`) as a nested source. See the [design and CLI
guide](README.md) for shared lifecycle and timing contracts.

```bash
rosbridge prepare handson
rosbridge tasks list --simulator handson
rosbridge launch handson/t12-load-and-start-microwave --execution-mode stepped --timeout 1800
rosbridge setup handson/handson-rokae-wuji --directory ../my-handson-workspace
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
dependencies, and checks every task's budget and upstream's SHA-256 pins of the bundled
robot and rendering assets.

## Episodes

A launch builds the episode as upstream's `handson/evaluation/native.py:run_episode` does:
- **Task contract:** the task's frozen contract replaces every `HANDSON_*` variable.
- **Seed:** for 16 Core tasks the seed selects a setting and derives the reset seed.
- **Scene:** construction converts the task's MJCF scene into the run directory (about
  0.5 GB, removed when the run stops).
- **Settle gate:** after the reset, a stable-origin gate settles the scene. It can reject a
  seed (see [Seeds and the settle gate](#seeds-and-the-settle-gate)); the launch then fails
  with exit status 3, and another seed is needed.
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

Success is upstream's `succeed`. A native failure latches at its first step, and no success
can follow it:
- a task's failure predicate (`info["failed"]`);
- for `t11_line_up_blocks`, a direct hand contact, the evaluator's own rule.

From then on `success` stays false, and the episode record's `failure` names the cause
(`native_failed` or `native_c12_direct_hand_contact`), and `failure_detail` the native info
of that step that explains it: its set entries whose names contain `fail` or `reason`, such
as `{"failed": true, "ycb01_failure_reason": "ball_or_hand_touched_gate"}`. Wait replies
never carry it. A step that succeeds and fails at once is a success, as upstream scores it.
Upstream ends a trial at its first success or failure: its protocol is `--execution-mode
stepped --stop-on-success`. A run launched to stop on success ends at its first outcome. At a
failure its wait replies add `; the episode ended because the task failed`, and the record
says reason `failure`, with the native cause in `cause` and `failure`. Without the flag the
failure latches silently and the episode runs to its horizon. Tasks whose success does not
latch (all but `t07`, `t20`, `t22`, `t33`, `t34`, `t47`, `eval_comb04`, `t18`, `t49`,
`eval_gen02`, `eval_flex01` and `eval_flex03`) can also lose a success before the horizon,
since their holds restart when a condition breaks. `t10_lift_carton` scores support, lift and
hold, recording grip force as a diagnostic, as the evaluator does.

27 tasks have no native failure and can only succeed or time out: `t02`–`t06`, `t08`,
`t09`, `t12`, `t19`, `t21`, `t23`–`t26`, `t31`, `t32`, `t42`–`t45`, `eval_comb01`,
`eval_comb03`, `eval_comb04`, `eval_gen01`, `eval_gen03`, `eval_gen07` and
`eval_flex03`. Some states make success impossible without failing: in `t02` a cable seated
before the plug, in `t20` the left hand touching the pen first, in `t31` an interrupted
brew, and in `t45` the left hand's contact lost.

With `ROSBRIDGE_HANDSON_STOP_ON_FAILURE=1` in the launch's environment, a run judged on its
final state (no `--stop-on-success`) still ends at a native failure, as upstream's evaluator
ends the trial there: waits reply that the episode ended because the task failed, and the
record has reason `failure`, the native cause and its detail. A success does not end it.
Every record with a failure also has `failure_step`, the step at which it latched.

With `ROSBRIDGE_HANDSON_PARTIAL_CREDIT=1` in the launch's environment, the record also adds
upstream's success checklist, the rows of its teleoperation success panel
(`_success_condition_rows()`): `conditions` at the trial's end (the failure step, if a
failure latched, else the last step), `conditions_initial` after the first step, and
`q_score`, as BEHAVIOR's `TaskMetric` computes it with the checklist as its one goal option:
1 on success, else the fraction of all conditions that hold at the end and did not at the
start (0 without a checklist). Unlike BEHAVIOR, the start is after the first step, once the
native checker has evaluated the scene. Reward and success stay binary, and the checklist is
read at most three times per episode. Off, nothing is read. The score is graded for 50 tasks,
about three levels for `t02`, `t12`, `t19`, `t22`, `t23`, `t24`, `t26`, `t33`, `t40`, `t43` and
`t47`, and binary for `t18`, `t29`, `t35`, `t49` and `eval_gen02`, whose checklist is a single
row.

## Seeds and the settle gate

For a task without a setting, the seed is the native reset seed: it samples the layout, the
table height and, where a task has one, its goal. `t24` always asks for the tennis ball and
`t26` for the apple, since only the paper's archived trials override their targets. For the
16 tasks in upstream's `settings.json`, the seed selects a setting and derives the reset seed.
The instruction is fixed by task, seed and goal: canonical for about one seed in five, else one
of 32 paraphrases. Give the agent only `PROMPT.json`'s `prompt`. Its `canonical_prompt` (the
unparaphrased instruction), `goal` and `native_goal_readback` must not reach the agent;
`eval_gen05`'s readback also gives the knobs' initial angles. Every receipt in `<run>/handson/`
is the judge's, not the agent's:

| Receipt | Holds |
|---|---|
| `PROMPT.json` | The instruction, its draw and the goal |
| `ORIGIN_PARAMETERS.json`, `ORIGIN_IMPLEMENTATION.json`, `TRACE_IMPLEMENTATION.json`, `CHECKER_LAYOUT_PATCH.json` | The settle gate's parameters and pinned implementations |
| `SETTING_SELECTION.json` (16 tasks) | The setting, such as `t38`'s initial and target knob angles, and every derived seed |
| `TASK_BOUNDARY.json`, `TASK_BOUNDARY_SETTLE.json` (`t01`, `t15`, `t38`) | Every object pose and joint position, and the native task info |
| `CARTON_RAW_FORCE_000/` (`t10`) | Raw grip force |
| `AS019_SCENE_BRIDGE.json` (`t22`) | The scene bridge |
| `PAPER_TRIAL.json` (paper trials) | The archived trial: seeds, goal, prompt |

The settle gate is upstream's, per task contract: a frozen stable-origin gate for 34 Core
tasks, a checker-boundary gate for `t01`, `t15` and `t38`, and the current gate with its
defaults for Transfer-16. Thirteen tasks have no settle gate: `t05`, `t12`, `t13`, `t20`, `t21`,
`t30`, `t31`, `t32`, `t37`, `t42`, `t43`, `t45` and `t46`. It rejects a seed whose reset
drifts more than 5 mm or 3° while settling, penetrates, or does not come to rest in time. The
rejection is deterministic. Reset raises `SeedRejected`, and the launch fails with exit status
3 and, with `--json`, `"cause": "seed_rejected"`. Rosbridge does not retry with another seed,
which would change which scene a seed names.

In upstream's archive of 45,000 accepted paper trials, 4.8 % of reset candidates were rejected:
43 % for `t27`, 25 % for `t23`, 20 % for `t07`, 10–14 % for `t04`, `t09`, `t14`, `t19`, `t24`,
`t25` and `t26`, a few percent for `t11`, `t36`, `t40` and `t48`, and none for the other
Core tasks. Many rejections are the robot's, not the task's: the seeded reset of the left arm
drifts, so the same seeds fail across these tasks. Of seeds 0–9, 4, 6 and 8 fail `t23`, `t24`
and `t27` alike; `t27` also rejects 1 and 3. Pre-screen seeds before launching many runs:

```bash
pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \
    python tools/handson_seeds.py --tasks t23_photograph_logo t24_place_ball_in_bowl \
    --seeds 0-9 --gpus 0 --output seeds.json
```

It boots each task and seed once, as a launch does but without ROS, on the given GPUs, and
writes which seeds the gate accepts.

Upstream also archives the inputs of the 45,000 accepted paper trials, 100 per Core-50 task
for each of nine models (`dm05`, `dp`, `fastwam`, `hy`, `lingbot`, `motus`, `openwam`, `pi05`,
`spatial`). With `ROSBRIDGE_HANDSON_PAPER_TRIALS=<model>` in the launch's environment, the seed
names one of them, as upstream's `--paper-trial N --model <model>` does:

```bash
ROSBRIDGE_HANDSON_PAPER_TRIALS=pi05 rosbridge launch handson/t24-place-ball-in-bowl --seed 7 \
    --execution-mode stepped --stop-on-success
```

Seed `N` (0–99) restores trial `N`: its accepted reset seed, its goal, the constructor and reset
overrides that set it, its archived instruction and its setting. This restores the paper's goal
diversity: `t24` asks for each of its three balls and `t26` for each of its four fruits. The
archive holds accepted resets, so the settle gate normally passes them, although upstream does
not certify that today's simulator matches the paper's. The record's `seed` stays `N`; the
receipt `PAPER_TRIAL.json` holds the trial. Transfer-16 tasks have no archive, and their launch
fails with a message saying so.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `right_arm`, `left_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `right_hand`, `left_hand` | `<hand>/joint_state`, `<hand>/joint_target`, `<hand>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/camera_info` |

Arms keep upstream's joint names `joint1_<side>`..`joint7_<side>` (radians, native
limits). An arm target is absolute joint positions and replaces only the native palm-pose
IK of HandsOn's EEF52 action. The native controller still applies the task contract's arm
settings and its stiff PD drives. They hold a free arm on its target, but they neither
compensate a load nor pass through objects, so an arm that carries or presses on something
settles short of its joint target. Examples:
- `joint1_<side>` swings a hand vertically, positive raising the right hand and negative the
  left, since the arms mirror each other. Lowering it alone presses the hand into the table:
  in `t24`, −0.2 rad on `joint1_right` reaches 61%.
- In `t12`, the right arm's `joint1_right` and `joint4_right` reach 60–85% of a 0.1 rad step
  either way.

HandsOn's palm-pose IK hides this, because it starts each step from the measured joints. A
joint-target controller should close that loop on `joint_state` or on the palm pose itself.

Hands are `joint` devices of 20 joints, `<side>_finger1_joint1`..`<side>_finger5_joint4`.
The order is finger-major from the thumb, the order of EEF52's hand20 values. Targets are
absolute joint positions in radians that pass through the native hand path, including its
target slewing. The rig `handson/handson-rokae-wuji`:
- holds the arm chains and hand trees from upstream's robot MJCF;
- ends each arm at `palm` (`<side>_palm_link`), the pose EEF52 commands, so Kinex IK maps
  palm poses to arm joints;
- mounts each hand on its arm's `palm`.

Cameras are RGB and render through upstream's three-view RTX capture: its exposure per
sensor, and wrist views that hide the wrist's own cables. One capture renders all three
views at a camera sample, six path-traced passes each (about 14 s on a shared L40S); it never
steps physics. Two choices keep the images on the Kinex
contract, where a profile is an undistorted pinhole at a fixed mount:
- **Head camera:** the calibrated K of the Gemini 336 (854×480, fx 463.39, fy 463.00, an
  off-center principal point (431.77, 239.92)), with its lens distortion zeroed.
- **Wrist cameras:** 848×480 on the palms (fx 272.14 left, 267.01 right). Every task uses
  this default camera profile; `t05` and `t15` would use 269.56 in the paper.
- **Mounts:** every camera renders at its nominal mount, without the evaluator's per-seed
  jitter (±2 mm, ±0.3°).

## Recording and checks

With `--record`, spectators `third_person`, `left_wrist` and `right_wrist` follow the
head and wrist cameras at 960×720 with their horizontal field of view. Each renders alone,
two path-traced passes per sample (about 3 s), so `--record-views` and `--record-rate` set
much of a recorded run's cost.

```bash
pixi run -e handson-native smoke                       # native checks via worker processes
ROSBRIDGE_ALL_TASKS=1 pixi run -e handson-native smoke  # every task boots Isaac (hours)
pixi run -e handson test-ros
rosbridge demo handson --timeout 1800
pixi run -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \
    python tools/robots/build_handson.py                # rewrites the catalog rig
```

With `KINEX_PYTHON`, the native checks also compare the exported rig's palm FK,
fingertips and camera mounts with Isaac.
