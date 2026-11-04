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
- **No reset.** The launch builds the kitchen once. The backend declares `resettable = False`, so the world node
  offers no `world/reset` and the rig's world device has no `reset` operation. A new attempt is a new launch.
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

## Checks

```bash
pixi run -e psi-kitchen smoke   # scene, evidence, no reset, re-rendering matches the live camera, time limit
```
