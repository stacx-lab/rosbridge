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
  seed; the launch then fails with its error, as upstream reports it, and another seed is needed.
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

Success is upstream's `succeed`. Upstream ends a trial at its first success or failure.
Here `--stop-on-success` ends the episode at the first success. A native failure does not
end the episode, but no success can follow it:
- a task's failure predicate;
- for `t11_line_up_blocks`, a direct hand contact.

From then on `success` stays false, and the episode record's `failure` names the cause.
`t10_lift_carton` scores support, lift and hold, recording grip force as a diagnostic, as
the evaluator does.

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
