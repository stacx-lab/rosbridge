# BiDex-100

BiDex-100 runs 106 bimanual dexterous tasks in MuJoCo, in five suites: Memory, Long_Horizon,
Generalization, Instruction_Following and Limited_Demonstrations.
- **Robots:** 96 tasks use two Franka FR3v2 arms, and 10 Generalization tasks use an
  Astribot S1. Both carry Sharpa Wave hands.
- **ID and OOD:** every task is cataloged twice, as its in-domain (ID) task and as its
  out-of-domain (OOD) twin `-ood`. These are the two domains of the official leaderboard.
- **Pins:** upstream is pinned to `5d077559b0c648bd1c4f32261a8032785740cb01`, and its Hugging
  Face assets to dataset revision `ca846b94d51302387cc6a32e124b1e38834d1974`.

See the [design and CLI guide](README.md) for shared lifecycle and timing contracts.

```bash
rosbridge prepare bidex
rosbridge tasks list --simulator bidex
rosbridge launch bidex/pour-ice-between-two-cups-ood --execution-mode stepped --seed 7
rosbridge setup bidex/bidex-fr3v2-sharpa-tabletop --directory ../my-bidex-workspace
```

`bidex` selects `Generalization/put_apple_in_white_bowl`. A task ID is the native name with
hyphens, such as `bidex/put-apple-in-white-bowl`. Its OOD twin adds `-ood`
(`Generalization/put_apple_in_white_bowl__ood`).

## Source and assets

The simulator runs in the world node's `bidex` environment: Python 3.12, MuJoCo 3.8 and
upstream's simulation requirements. Preparation clones upstream, exports it into
`.runtime/bidex`, and installs `bidex` editable without dependencies.

**Downloads.** Preparation downloads four of BiDex's asset packs from the pinned dataset
revision, each verified by its LFS SHA-256: `libero`, `articraft_objects`, `assembly_objects`
and `astribot_descriptions`. Together they are 0.4 GB, or 0.95 GB extracted.

**Evaluation data.** The evaluator's `eval_dataset.h5` is a Git LFS object. It holds the
recorded ID start poses and the demonstration lengths behind the horizons.
- Where git-lfs is installed, the export carries the object itself.
- Otherwise preparation fetches it by its pinned digest.

**RoboCasa assets.** RoboCasa's objects and textures come from the robocasa simulator's
cache, which `rosbridge prepare robocasa` downloads or adopts from a seed. That cache is
identical to BiDex's 5.4 GB `robocasa_assets` pack in every file BiDex uses, including the
hashed textures of its OOD manifest, so that pack is not downloaded.

**Mounts.** Upstream writes temporary MJCF beside every object it loads and beside the OOD
scene. The mounts are therefore real directories down to each object, with links below them,
so the shared caches are never written. Preparation also recreates the alias links of
upstream's `setup_assets_links.sh` (assembly parts, threading rods, tiered baskets).

## Episodes

A launch replays one episode of upstream's leaderboard evaluator
(`bidex/eval/run_leaderboard_eval.py`, `vector_rollout.py:_start_next_episode`).

**Seeds.** A seed `s` is the evaluator's episode `s % 25` of eval seed `s // 25`, with 2
added to the eval seed for OOD. Seeds 0–49 of a task and of its `-ood` twin therefore replay
the official 50 ID and 50 OOD rollouts: ID eval seeds 0 and 1, OOD eval seeds 2 and 3, 25
episodes each.

**ID.** The base YAML samples the objects in their regions. The robot starts from a recorded
working pose drawn from `eval_dataset.h5`.

**OOD.** The `__ood` YAML applies:
- layout shifts, left-right mirroring in some tasks, and rescaled articulated objects;
- floor, wall and table textures and randomized key and fill lighting;
- distractors, placed per episode.

The robot starts at its default pose.

**Settle.** A 1 s settle with the robot pinned then lets the objects come to rest. It is not
part of the episode's time.

**Cameras.** Cameras keep their nominal mounts in every episode. This is a deliberate
departure from upstream: on the leaderboard, 30 of every 50 OOD episodes move the head
and/or wrist cameras by up to 1.8 cm and 6°. With nominal mounts the rig's camera frames
stay exact.

**Instruction.** The instruction is printed to the run log with the evaluator's eval seed
and episode. Give it to the agent; it is not published on ROS. Ten Instruction_Following
tasks switch their instruction during the episode, and the log names each switch with its
world time.

**Timing.** Control runs at 25 Hz: twenty 2 ms MuJoCo substeps per step. A step costs about
10 ms of wall time, and a reset about 3.5 s. The horizon is the evaluator's: the task's
`max_steps`, otherwise 1.2 × the 90th percentile of its demonstration lengths. The task's
`max_seconds` also ends the episode.

**Success and score.** Success is the evaluator's final-frame verdict: every subtask has
latched, every `must_hold` subtask still holds, and no `latch_violation` guard was violated.
The reward is 1 on success. Upstream ends an episode at its first success; here
`--stop-on-success` does. The episode record adds `score`, the benchmark's 0–100 progress
score: the weights of the subtasks credited so far.

## Devices

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/spectator/*` while recording |
| `right_arm`, `left_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `right_hand`, `left_hand` | `<hand>/joint_state`, `<hand>/joint_target`, `<hand>/stop` |
| `head_camera`, `left_camera`, `right_camera` | `<camera>/image`, `<camera>/depth`, `<camera>/camera_info` |

The devices split BiDex's 58-D joint action in its own order:
1. `right_arm`, `left_arm`: `arm_joint_position`;
2. `right_hand`, `left_hand`: `dex_joint_position`.

Targets are absolute joint positions in radians, through the native position servos.

**Arms.**
- Joint names: `<side>_fr3v2_joint1..7` on the FR3v2 and `astribot_arm_<side>_joint_1..7` on
  the Astribot. Each arm's tip link `flange` is the body that carries its hand and wrist
  camera.
- The Franka servos are stiff.
- The Astribot's are soft (kp 100–500, no gravity compensation). An Astribot arm holding a
  pose sags up to about 0.12 rad, so close the loop on `joint_state`. The control demo runs
  on the default FR3v2 task.
- The Astribot's torso and head are not devices. Their servos hold the initial pose, which
  moves them by less than 1 mm, and the rig bakes that pose into its mounts.

**Hands.**
- Each hand has 22 joints, from the thumb to the little finger: `<side>_thumb_CMC_FE`,
  `<side>_index_MCP_FE`, `<side>_pinky_DIP`, and so on.
- MuJoCo's joint limits are soft and the finger servos are weak, so measured joints can
  read past their ranges:
  - by 7 mrad on a recorded start pose;
  - by up to about 0.1 rad when fingers press into each other.
- A target more than 0.001 outside a range is refused. Clip hand targets to the profile
  limits rather than sending `joint_state` back.

**Cameras.**
- RGB-D at 480×360, as the evaluator renders them (0.75 × 640×480). Each has an 80°
  vertical field of view: fx = fy = 214.52 and centre (239.5, 179.5).
- The FR3v2 head camera is on the right arm's base; the Astribot's is in its head.
- The wrist cameras are on the flanges.
- Depth is in metres along the optical axis. The near plane scales with each scene's
  extent.

| Rig | Robot and scene | Tasks |
|---|---|---:|
| `bidex/bidex-fr3v2-sharpa-tabletop` | FR3v2 on the LIBERO tabletop | 76 |
| `bidex/bidex-fr3v2-sharpa-dexjoco` | FR3v2 at the dexjoco table (the shelf and tray tasks) | 20 |
| `bidex/bidex-astribot-s1-sharpa` | Astribot S1 at the dexjoco table | 10 |

The two FR3v2 rigs differ only in the robot's base pose.

## Recording and checks

With `--record`, the spectators `third_person`, `left_wrist` and `right_wrist` follow the
head and wrist cameras at 960×720.

```bash
pixi run -e bidex smoke
pixi run -e bidex test-ros
rosbridge demo bidex
pixi run -e bidex python tools/robots/build_bidex.py   # rewrites the three catalog rigs
```

With `KINEX_PYTHON`, the native checks also compare each rig's arm FK, hand mounts,
fingertips and camera mounts with MuJoCo, on ID and OOD episodes.
