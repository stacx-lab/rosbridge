# PSI kitchen

`psi-kitchen` is PSI-Bench's kitchen: the [RoboCasa](robocasa.md) simulator (same RoboCasa365 and robosuite pins
and patches, same PandaOmron devices, cameras and timing) with three additions for benchmark tasks. It has its own
Pixi environment and shares RoboCasa's asset cache.

```bash
rosbridge prepare psi-kitchen
rosbridge launch psi-kitchen/open-drawer --seed 0 --execution-mode stepped --stop-on-success --detach
```

## Additions

- **Evidence.** Each built episode writes `episode-<n>/` in the run directory: `meta.json` (task, seed,
  instruction, step limit, row layout), `model.xml` (the compiled scene) and `steps.f64`, one float64 row
  `[time_s, action(12), qpos]` for the start and after every control step. Nothing is recorded live.
- **Re-rendering.** `python -m rosbridge_psi_kitchen.render EPISODE_DIR [--camera NAME] [--out FILE]` rebuilds
  the scene from `model.xml`, sets each logged state and encodes an MP4 (at most about 300 frames); nothing is
  simulated.
- **Reset guard.** `world/reset` fails ("reset failed", episode unchanged) unless `PSI_ALLOW_RESET=1`; the
  launch build always runs.
- **Step limit.** `PSI_MAX_STEPS` overrides the dataset horizon (0 or unset keeps it).

`python -m rosbridge_psi_kitchen.describe TASK SEED` prints a variant's task, instruction and step limit as JSON
without starting a graph. Tasks are in `simulators/psi-kitchen/tasks.toml`; the robot rig is
`psi-kitchen/panda-omron`, a copy of RoboCasa's described in robot terms.

## Checks

```bash
pixi run -e psi-kitchen smoke   # evidence, reset guard, re-rendering matches the live camera, step limit
```
