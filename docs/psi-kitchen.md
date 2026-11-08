# PSI kitchen

`psi-kitchen` is PSI-Bench's kitchen: the [RoboCasa](robocasa.md) simulator (same RoboCasa365 and robosuite pins
and patches, same PandaOmron devices, cameras and timing) with a few additions for benchmark tasks. It has its own
Pixi environment and shares RoboCasa's asset cache.

```bash
rosbridge prepare psi-kitchen
PSI_SCENE=3 rosbridge launch psi-kitchen/open-drawer --seed 0 --execution-mode stepped --stop-on-success --detach
```

## Additions

- **Scene.** `PSI_SCENE=k` builds RoboCasa365 target kitchen `k` (layout `k`, style `k`, 1 to 10) with target object
  instances. Unset, the seed draws one of the ten as upstream does.
- **Time limit.** `PSI_TIME_LIMIT_S` ends the episode after that much robot time (rounded up to whole 0.05 s
  periods); 0 or unset keeps the dataset horizon.
- **Arm stiffness.** The arm's joint-position controller takes upstream's arm gain (kp 150, its OSC_POSE gain,
  instead of robosuite's joint default of 50). Both controllers scale by the mass matrix, so the hand is as stiff as
  upstream's, and demonstrations recorded under upstream's Cartesian controller replay as joint targets.
- **Evidence.** The build writes `episode-0/` in the run directory: `meta.json` (task, scene, seed, instruction,
  built layout and style, time limit, row layout), `model.xml` (the compiled scene) and `steps.f64`, one float64
  row `[time_s, action(12), qpos]` for the start and after every control step. Nothing is recorded live.
- **Re-rendering.** `python -m rosbridge_psi_kitchen.render EPISODE_DIR [--camera NAME] [--out FILE]` rebuilds
  the scene from `model.xml`, sets each logged state and encodes an MP4 (at most about 300 frames); nothing is
  simulated.

`python -m rosbridge_psi_kitchen.describe TASK SEED SCENE [SCENE ...]` builds each scene in turn without starting a
graph and prints one JSON line per scene: its time limit and instruction, or the `error` that kept it from building.
Tasks are in `simulators/psi-kitchen/tasks.toml`: all of RoboCasa365 (65 atomic, 300 composite), each built in the
target kitchens. The 48 composites without an upstream horizon (no dataset) get `horizon = 2250` steps there, the
median of upstream's composite horizons. 109 tasks exclude some of the ten kitchens (the task's
`EXCLUDE_LAYOUTS`/`EXCLUDE_STYLES`: no oven, no reachable freezer, no dining counter or island, known placement
problems), and the build says so; at seed 0, 25 more fail to place their fixtures or objects in 54 (task, kitchen)
pairs. Every task builds in at least one kitchen, 240 in all ten. The robot rig is `psi-kitchen/panda-omron`, a copy of RoboCasa's described in robot
terms.

## Demonstrations

`python -m rosbridge_psi_kitchen.dataset TASK --out DIR [--kitchens 1-5] [--size 128] [--jobs N]` turns RoboCasa365's
target-split human demonstrations of a task (500 teleoperated episodes over the ten target kitchens, downloaded once
into `--cache`) into the robot's own terms. Each episode from the chosen kitchens is restored from its recorded
states into a psi-kitchen robot (`PsiKitchenSim.settle`, the live robot's controllers and device readings):

- **state** (12): arm joints, gripper command, torso, base odometry (x, y, yaw from the episode start), as the
  devices report them;
- **images**: the three cameras at 256 × 256, resized with `cv2.INTER_AREA` to `--size`;
- **action** (12, the backend's step vector): the recorded command in the devices' terms. The arm target is the
  joints for the hand pose upstream's Cartesian controller aimed at: the current hand pose moved by the recorded
  command (in base mode, the previous goal moved by it), as joint angles by mass-weighted Newton steps from the
  recorded joints. The torso target is its position plus the recorded torso command; the gripper target is the
  recorded direction; the base velocity is the recorded one.

An episode is kept only if its commands, replayed open-loop from its first state, do the task again. `DIR` gets raw
arrays (`camera_{left,right,wrist}.u8` `(N, S, S, 3)`, `state.f32` and `action.f32` `(N, 12)`, `episode_end.i32`),
`episodes.jsonl` (rows, kitchen, instruction) and `meta.json` (layouts, units, counts); no simulator state, model or
object name. The smoke test checks that a restored episode reads and renders as the live robot did, and that a
Cartesian command becomes the joints for the hand pose it commands.

## Checks

```bash
pixi run -e psi-kitchen smoke   # scene, evidence, re-rendering matches the live camera, time limit,
                                # demonstrations restored as the live robot reads and sees them, commands aimed
                                # at the hand pose they command
```
