# Adding a simulator

This is the procedure for bringing an upstream simulator into rosbridge, from its pinned source to its
tests and docs. Every integration under `simulators/` follows it:

- `simulators/bidex` is a compact in-process example;
- `simulators/robodojo` runs its backend in a native worker;
- `simulators/libero` declares a configuration.

The contracts these steps rely on are in [Design and usage](README.md) and [Devices](devices.md).

## Principles

- **Upstream's behaviour, unchanged.** Run upstream's own environment, controllers, physics and
  evaluator. Change them only through a patch in `patches/<sim>/`, and only when the host or the
  device contract needs it. Document every departure on the simulator's page.
- **Native units and names.** Devices keep upstream's joint names, control units and command
  semantics: a normalized gripper stays normalized. Device types are Kinex's: `joint`, `gripper`,
  `locomotion`, `rgb` and `rgbd`.
- **Judge as upstream's evaluator judges.** Success, reward, step limit, early termination and
  episode selection follow upstream's evaluation script, not its environment's defaults (see
  [Scoring parity](#scoring-parity)).
- **No surprise downloads.** Preparation downloads and verifies assets once.
  `ROSBRIDGE_ASSET_SEED` adopts an existing verified depot instead, and launch never downloads.
  Check free disk space before large installs.

## 1. Source

The manifest pins the source, and preparation clones it into `third_party/<sim>`. It then exports
the pinned tree into `.runtime/<sim>` and applies the patches listed in `patches/<sim>/series`, in
order. Pin submodules as `[[nested_sources]]`, each with its own path, URL, commit and editable
`install` directories.

```toml
name = "mysim"
source = "third_party/mysim"
source_url = "https://github.com/org/mysim.git"
source_revision = "<full commit>"
patches = "patches/mysim"
```

Keep each patch minimal and explain it at its top. A change to the source or a patch means removing
`.runtime/<sim>` before preparing again (see [Sources, caches and extension](README.md#sources-caches-and-extension)).

## 2. Environment

1. **Feature:** add a Pixi feature `<sim>` with the dependencies upstream pins (its requirements or
   lock file), and an environment of the same name. Put the environment in the `jazzy` solve group
   when it runs the world node itself.
2. **Tasks:** give the feature the three tasks every simulator has:
   - `build = "python -m rosbridge.provisioning.workspace --simulator <sim>"`;
   - `test-ros`, which sources `.workspaces/<sim>/install` and runs `simulators/<sim>/tests/ros`;
   - `smoke`, which runs `simulators/<sim>/tests/native`.
3. **Unit tests:** add `simulators/<sim>/tests/unit` to the default `test` task.
4. **Different Python:** when upstream needs a Python that ROS Jazzy lacks, as Isaac Sim's 3.11 does:
   - add a native environment `<sim>-native` with `no-default-feature = true`;
   - name it in the manifest as `native_environment`, with `native_backend = "module:Class"`;
   - register a `WorkerBackend` subclass as the backend (`simulators/robodojo/rosbridge_robodojo/worker.py`).

   The world node stays in the ROS environment, and the worker runs the native class on its own main
   thread.
5. **Rebuilds:** editing `pixi.toml` or `ros2/rosbridge_runtime` marks every simulator's ROS build
   stale. Run `rosbridge doctor <sim>` for each, then `rosbridge prepare <sim>`; nothing downloads.
   Prepare refuses environments in use by live runs.

## 3. Package

```
simulators/<sim>/
  simulator.toml               the manifest (step 4)
  tasks.toml                   the task catalog (step 5)
  package.xml, setup.py, setup.cfg, resource/rosbridge_<sim>
  rosbridge_<sim>/
    backend.py                 the SimulatorBackend (step 9)
    assets.py                  prepare(runtime, task) and check_task(runtime, task) (step 7)
    demos/control.py           the control demo
  tests/unit/  tests/native/  tests/ros/
```

`setup.py` is an ament_python package with two entry points:

- `"rosbridge.backends": ["<sim> = rosbridge_<sim>.backend:Class"]`, or the worker class;
- `"console_scripts": ["control_demo = rosbridge_<sim>.demos.control:main"]`, the finite ROS-only exercise
  that `rosbridge demo <sim>` runs.

## 4. Manifest

| Key | Meaning |
|---|---|
| `default_task`, `task_catalog`, `python_path` | the task `<sim>` alone selects, the catalog file, and the package directory |
| `environment`, `ros_package` | the Pixi environment of the world node and the ROS package that holds the backend |
| `install` | the generated tree's directories installed editable without dependencies (default `["."]`; `[]` imports it in place) |
| `asset_preparer`, `asset_checker` | `module:function` hooks (step 7) |
| `native_environment`, `native_backend` | a native worker (step 2) |
| `[devices]` | the backend's device inventory by kind (step 8) |
| `scene_options` | upstream's episode controls besides the seed, which launch takes as flags and passes to `reset` (step 6) |
| `workspace` | a setup hook for profile changes by task and seed ([HandsOn](handson.md), [BiDex-100](bidex.md)); it runs in the CLI's interpreter, which has PyYAML, NumPy and SciPy (`rosbridge[cli]`) |
| `[configurations.NAME]` | alternative device sets that `--config` selects ([Devices](devices.md#robot-catalog-and-workspace-setup)) |

## 5. Task catalog

`tasks.toml` lists `[[tasks]]` entries with:

- `id = "<sim>/<slug>"`;
- `native_task`, upstream's name;
- `robot`, the catalog rig;
- any fields the backend reads, such as a suite or configuration name.

Catalog every task upstream evaluates. An optional `seed` sets a task's default launch seed. Prefer
none: when upstream defines the episodes its evaluation runs, the backend maps seeds onto them
instead (see [Scoring parity](#scoring-parity)).

## 6. Episode selection and randomization

Each simulator chooses and randomizes its episodes in its own terms. Find how upstream's evaluation
controls them, and expose exactly those controls, under upstream's names, values, defaults and
semantics. The controls may be a seed, a scene split, a layout, instance or trial index, a
randomization switch, or anything else. Don't rename them, merge them or invent controls of rosbridge's
own: whatever upstream leaves to chance stays upstream's chance.

- **Seed.** When upstream takes a seed, `--seed` passes it to `reset(task, seed, randomize)`, and the
  backend seeds through upstream's own seeding. When upstream's evaluation defines which episodes it
  runs, seed k selects its k-th (see [Scoring parity](#scoring-parity)).
- **Other controls.** Each control upstream takes besides a seed is a scene option:
  1. Declare it in the manifest's `scene_options`.
  2. Add it as a keyword-only argument of the backend's `reset`, defaulting to upstream's default.
  3. Add a launch flag for it to `SCENE_FLAGS` and the `launch`/`demo` parser in
     `src/rosbridge/cli/main.py`, with help that names the simulator and upstream's meaning.

  Launch refuses options a simulator does not declare, and the backend checks their values.
- **Tests.**
  - A native test builds upstream's own environment with the same controls and compares the
    episode it builds with the backend's.
  - A unit test checks that `scene_options` equals the backend's `reset` keywords.

## 7. Assets

- **Preparer:** `prepare(runtime, task)` fetches each pinned archive into the shared `AssetStore`
  (`rosbridge.assets`), verifies it against its pin, and links it into `.runtime/<sim>`.
- **Checker:** `check_task(runtime, task)` raises before an environment is built when something
  is missing. It never downloads.
- **Seeding:** where a verified copy already exists on the store's filesystem, adopt it with
  `ROSBRIDGE_ASSET_SEED` rather than downloading again.

## 8. Devices

Devices follow Kinex's device contract: its device types, operation names and types, and profile
schema. rosbridge serves them unchanged (see [Devices](devices.md) and [Kinex](kinex.md)). Map
upstream's robot and cameras onto that contract.

- **One inventory.** The backend's `devices`, the manifest's `[devices]` table and every catalog rig
  of the simulator must list the same devices; the export test compares them.
- **New types.** A device type the contract lacks needs that contract, and its endpoints in the world
  node (`ros2/rosbridge_runtime`), before a simulator uses it.
- **Several options.** Where upstream offers more than one way to expose a device, ask the user which
  one is the default before building it. Examples are a joint or an end-effector controller, or RGB or
  RGB-D cameras. Offer the others as configurations, which `--config` selects
  ([Devices](devices.md#robot-catalog-and-workspace-setup)).

## 9. Backend

Implement `SimulatorBackend` (`src/rosbridge/core/backend.py`). Its docstring is the contract.

- **Action layout.** `layout` is an `ActionLayout` of `CommandGroup`s.
  - Joint and gripper groups command positions in feedback units, so a measured state is a valid
    hold target.
  - A velocity group is a mobile base.
  - A pose group names the tip link of a joint device. Its targets arrive in `step(action, poses)`,
    and `tip_pose(group)` measures them.
  - A joint group with a tip and `joint_targets` reports its tip's pose, `tip_pose(group)`.
  - A group with `targets=False` serves no target topic: only the backend's [actions](#actions)
    command it.
- **Devices.** `devices` comes from `device_inventory(layout, cameras, depth=...)`. Each camera
  provides:
  - `intrinsics(camera)` (width, height, fx, fy, cx, cy in ROS pixel centres);
  - `render(camera)`;
  - `render_depth(camera)` for RGB-D, as float32 metres along the optical axis.
- **Episode.** One `reset(task, seed, randomize, **scene)` loads the run's one episode, and each
  `step` advances exactly one `period`.
  - The seed fully determines the episode.
  - A reset that fails keeps the live episode: build a candidate, and adopt it only once it has loaded.
  - Native state is touched only on the owner thread.
- **Task description.** After the reset, `description` holds the instructions upstream's evaluator
  gives its policy for this episode: `{"instruction": ...}`, read from upstream as its evaluation
  reads it, plus any further text it gives under its own name (RoboDojo's `detailed_instruction`).
  The world node logs it and `rosbridge describe` reads it back
  ([Task description](devices.md#task-description)); the backend neither prints nor writes it.
- **Ending.** `done` at the step limit; `success` and `reward` every step.
  - Created with `early_stop`, the backend ends where upstream's evaluator ends an episode, and names
    that end in `stop`.
  - `measures()` adds the backend's own fields to the episode record, such as a native seed or a
    graded score.
- **Optional.**
  - spectators (`configure_spectators`, `render_spectators`, `spectator_intrinsics`): one call returns a frame
    of every view, so a renderer that draws all its views in one pass renders once;
  - `camera_rate_hz`;
  - `distortion` for lensed cameras;
  - `configuration=NAME` for declared configurations.

### Actions

A command that runs over several steps and ends by itself, such as RoboDojo's `move`, is a ROS action
the backend defines and executes. The world knows no action by name: it serves each one the backend
lists, carries its goals, feedback and results, and lets waits name its goals.

- **Declare.** `actions` lists an `Action(device, name, type)` per operation; the world serves it as
  `<device>/<name>`, of any installed action type.
- **Start and end.** `start(operation, goal)` takes a goal as the fields of the type's `Goal`, in place
  of the operation's active one. `end(operation, reason)` ends the active one at once: a cancel, a
  stop, a newer command, a wait's `goal_timeout_s` or the episode's end. Neither raises for the goal's
  sake; a goal the backend cannot take ends as it starts, with the reason in its result.
- **Advance.** Goals advance in `step`, one `period` each as everything does, and a goal's commands
  replace the action's for the groups it commands.
- **Report.** `goals[operation]` holds the goal's `status` (`None`, then `"succeeded"` or `"aborted"`),
  its `result` and latest `feedback` as the fields of the type's messages, and `commands`: by command
  group, what the goal commands it, or leaves it commanded once ended. The world takes these as the
  groups' commands, so a group serves targets and goals side by side, each replacing the other, and
  a stop ends the goals that command its group.
- **Commanded only by goals.** A group with `targets=False` serves no target topic.
- **Native worker.** `start`, `end`, `actions` and `goals` cross to a `WorkerBackend` as they are.
- **Profile.** `rigs.action(profile, name, type, description)` adds the operation to the device's
  profile; the description is the backend's to write.
- **Tests.** The lifecycle is covered generically (`tests/unit/test_goals.py`, `tests/ros/test_motion.py`,
  on the toy backend's own action). Test what the action does: its executor without a simulator, as
  `simulators/robodojo/tests/unit/test_robodojo_move.py` does, and a goal in the graph test.

## Scoring parity

A launch's record should equal what upstream's evaluation would score for the same episode. Read
upstream's evaluation script, not only its environment, and match each item:

1. **Success.** Use the predicate the evaluator counts, read at the same point.
   - LIBERO's success is `env.step`'s `done`, before rosbridge's extra `sim.forward()`.
   - RoboCasa's gym wrapper evaluates `_check_success()` a second time after the step's state update.
   - A predicate with side effects, such as a timer counted per call, must be called as often as
     upstream calls it.
2. **Score.** Report upstream's per-episode score as the reward. For example, RoboDojo scores a
   success 1 and grades only failures.
3. **Step limit.** Use the evaluator's horizon, with the off-by-one included. BEHAVIOR's `Timeout`
   ends on the step after its limit.
4. **Early termination.** `--early-stop` ends where the evaluator ends: at its success, and at the
   task failures it reports.
5. **Episode selection.** Seed k is the episode upstream's evaluation runs as its k-th:
   - LIBERO: initial state k;
   - ABC: world k's seed;
   - BEHAVIOR: test instance k;
   - RoboTwin: the k-th seed its expert check accepts. When that needs a screen, write a tool and a
     committed table: `tools/robotwin_seeds.py`, `tools/handson_seeds.py`.
6. **Start.** Run the evaluator's own settle and initialization steps, with its seeding.

Prove the items with native tests that drive an independent upstream environment the same way and
compare state, reward and success. Document whatever stays different on the simulator's page.

## 10. Catalog rig

`tools/robots/build_<sim>.py` measures the rig from the live simulator. It writes
`catalog/robots/<sim>/<rig>/` through `tools/robots/rigs.py`:

- `workspace.yaml`;
- each device's `profile.yaml`, built with `rigs.joint`, `rigs.gripper`, `rigs.camera`, `rigs.world` and,
  for a backend's actions, `rigs.action`;
- kinematic URDFs.

Rigs of one simulator share files through symlinks into a common directory, as LIBERO's do. A
configuration's profile changes go to `configurations/NAME/profiles.yaml`, through
`write_rig(..., configurations=...)`. Never hand-edit generated profiles; change the builder and
rerun it.

## 11. Tests

- **Unit.** Add `simulators/<sim>/tests/unit` for catalog and manifest checks. The rig export test
  (`tests/unit/test_robot_catalog.py`) and Kinex's schema and readiness test (`tests/kinex/unit`)
  cover every rig, in every declared configuration, automatically.
- **Native.**
  - Parity with upstream: actions, timing and the scoring items above.
  - Every task loads, describes itself (`description["instruction"]`) and steps. Guard a slow sweep
    with `ROSBRIDGE_ALL_TASKS=1`.
  - Rendering and spectator isolation.
  - A failed reset keeps the episode.
- **ROS.** A graph test through `tests.ros.support.Graph(backend="<sim>")`: device endpoints, waits,
  stops and cameras.
- **Geometry.** `tests/kinex/native/test_<sim>.py`: the exported rig's FK, mounts and camera frames
  against the simulator.
- **Demo.** `rosbridge demo <sim>`.

## 12. Docs

- **Page:** add `docs/<sim>.md`, covering:
  - source, environment and assets;
  - devices;
  - timing and episodes, including the scoring parity and any departures, and where the
    instruction comes from;
  - recording and checks.
- **Links:** link the page from `README.md` and `docs/README.md`.
- **Lists:** add the rig to the table in `docs/devices.md`, and the commands to
  [Verification](README.md#verification).
- **Known issues:** when upstream defects are documented rather than fixed, keep them on a known-issues
  page, as BEHAVIOR and HandsOn do.
