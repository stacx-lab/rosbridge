# Design and usage

rosbridge exposes public robotics simulators, currently ABC, LIBERO, RoboTwin,
RoboDojo, RoboCasa and BEHAVIOR, as managed ROS 2 graphs that speak Kinex's device contract. It uses native ROS topics and
services, with no HTTP or WebSocket bridge. The command-line environment stays
separate from simulator environments.

## Setup and lifecycle

Linux x86-64, Git, uv, Pixi, and an EGL-capable graphics driver are required.
ROS Jazzy and Python 3.12 come from the locked RoboStack/Pixi environment; a system
ROS installation is not required. Every graph serves Kinex's world service
`kinex_interfaces/srv/Wait`; rosbridge builds it from its own identical copy in
`ros2/kinex_interfaces`, so preparation needs no Kinex checkout.

```bash
bash tools/bootstrap.sh
source .venv/bin/activate
rosbridge prepare abc
rosbridge tasks list --simulator abc
rosbridge launch abc/put-plastic-bottles-in-bin --execution-mode stepped --detach
rosbridge status RUN_ID
rosbridge exec RUN_ID -- ros2 node list
rosbridge exec RUN_ID -- python controller.py
rosbridge stop RUN_ID
```

`prepare SIMULATOR` synchronizes the pinned source (and any pinned nested sources,
such as RoboDojo's Isaac Lab and cuRobo forks), prepares assets for every cataloged
task, builds the ROS overlay with Kinex's interfaces, and checks readiness. `--task TASK` restricts asset preparation to a single task. Repeat
preparation reuses source and assets. `doctor SIMULATOR [--task TASK]` checks
readiness without installation or downloads. Preparation refuses to modify
environments used by active managed runs.

`launch` accepts `SIMULATOR/TASK`, an unambiguous native task name or task slug,
or a simulator's default-task alias. It takes `--execution-mode`, `--seed`,
`--stop-on-success`, `--domain-id`, `--timeout`, `--record DIR` and
`--record-type`. It requires preparation; launch never downloads assets.
Launch remains in the foreground unless `--detach` is supplied. Ctrl-C stops the
complete graph. Detached launch returns once the world node, its clock and wait
service are discoverable and, when recording, the recorder holds every initial
frame. Startup failures stop the graph and retain logs.

Each run leases a separate ROS domain; `--domain-id` selects one explicitly, and
leases coordinate this user's local rosbridge runs. `exec` supplies the selected
environment, ROS overlay and domain while preserving arguments and exit status.
External controllers must enable `use_sim_time` themselves. Runs retain logs and
metadata under `.cache/rosbridge/runs/<id>/`, including each run's ROS logs and
temporary files. The graph, recorder, probes and `exec` commands share that run's
`ROS_HOME`, `ROS_LOG_DIR` and `TMPDIR`. `status` and `stop` can omit the run ID when
exactly one live run exists. Process identity checks prevent signalling reused PIDs.

Management commands accept `--project` and `--json`.

## Runtime architecture

| Layer | Responsibility |
|---|---|
| `src/rosbridge/cli`, `catalog`, `provisioning`, `runs`, `recording` | Discovery, source/assets, environments, run supervision, recording |
| `src/rosbridge/core`, `runtime` | Backend/device contracts, commands, stepping, wait scheduling |
| `ros2/rosbridge_runtime` | The world node, device endpoints, observation publication, shared launch files |
| `simulators/<sim>` | Native backend, asset preparation, catalog, demo and tests |
| `tools/robots` | Rig builders that measure Kinex profiles and URDFs from each simulator |
| `catalog/robots` | Kinex workspace skeletons; see [Kinex integration](kinex.md) |
| `third_party/<sim>`, `patches/<sim>` | Pinned upstream source and explicit patch series |

A run has one process containing a `world` node that hosts every device's
endpoints: `<device>/...` for each arm, gripper and observation camera, plus the
world clock, waits, stop and, when recording, spectator streams. A simulator
whose native stack cannot share the ROS interpreter declares `native_environment`
and `native_backend` in its manifest: the world node then holds a proxy
(`rosbridge.core.worker`) and the backend runs on the main thread of a worker
process in that environment. Calls are sequential over a private socket; arrays
cross it independently of the NumPy versions on either side, and a dead worker
terminates the graph like an executor failure. World owns
the backend, stepping and clock; native judging stays internal. Kinex owns
frame publication and composition outside the simulator.

The main thread exclusively creates, loads, steps, renders and destroys native
objects. ROS callbacks queue functions that the main thread runs between steps, so
they never read native state. Wait handlers await futures, so pending waits cannot
exhaust executor workers. State and camera publication happen on the owner thread
before a step's waits complete. Spectator frames are handed to a timer that
publishes only the latest one, so reliable DDS writes cannot block physics. DDS
delivery to remote subscribers remains asynchronous; a wait response is not an
acknowledgement of remote image delivery. An executor callback failure terminates
the graph instead of leaving an unresponsive simulation running.

## Timing and public interfaces

`--execution-mode realtime` is the default: world continually steps with monotonic
wall-clock pacing. `stepped` advances only while an admitted `world/wait` needs more
simulation time. Commands, zero waits, discovery, and camera reads do not step.
Both modes retain the backend's native control period and physics decimation.
Rendering can make realtime slower than wall time; it never changes simulated dt.

`world/wait` uses `kinex_interfaces/srv/Wait`: `duration_s` in, then `success`,
`message`, `time_s` and actual `waited_s` out. Positive durations round up to whole
native control steps; zero is immediate; negative, nonfinite or unrepresentable
waits fail. Waits are sequential: a wait requested while another is pending is
rejected, and `world/stop` (`std_srvs/srv/Trigger`) cancels the pending one. A run
is one episode, of the task and seed it was launched with. There is no reset: to
start over, stop the run and launch another. Termination completes a wait due on
that tick and fails an unfinished one. A wait admitted before a client times out
stays pending until it completes or `world/stop` cancels it; abrupt process loss may
prevent a response, so use wall timeouts.

```python
from rosbridge.clients import WorldClockClient
world = WorldClockClient(node)
result = world.wait(1.0, timeout_s=30.0)  # node not already attached to an executor
assert result.success, result.message
# In a callback with an executor already spinning:
result = await world.wait_async(1.0)
```

The ROS graph inventory remains fixed during a run. Native evaluator results are
available to the operator through `rosbridge judge RUN_ID`; task instructions and
evaluator state are not published.

Arms accept `trajectory_msgs/msg/JointTrajectory` on `<arm>/joint_target`:
position-only points over every arm joint, strictly increasing times ending within
120 seconds, and a start that is now or later. Joint names are unique within a
device; two arms may keep the same native names. Grippers accept a
`std_msgs/msg/Float64` position on `<gripper>/target`, in each simulator's native
command units (LIBERO's finger metres; the normalized 0..1 opening of ABC, RoboTwin
and RoboDojo; RoboCasa's integrated -1..1 command; BEHAVIOR's smooth -1..1 R1Pro command).
Mobile bases (Kinex
`locomotion`) accept a body-frame `geometry_msgs/msg/Twist` on `<base>/velocity`,
held for 0.5 s of world time after the last one, and publish `nav_msgs/msg/Odometry`
on `<base>/odometry`. Invalid targets are logged
and rejected; a target within 0.001 (rad or m) of a limit, such as a measured
position slightly past it, is held at the limit. The latest valid target is held
until replaced or stopped. `<device>/stop` holds the measured position; a base's stop
commands zero velocity. Each command
subscription is created before its device's stop service. Cameras publish RGB8
`Image` on `<camera>/image` and `CameraInfo`; frames are `<camera>/base`. RGB-D
cameras (Kinex `rgbd`) add registered `<camera>/depth` (16UC1 millimetres along the
optical axis, 0 invalid) with the same stamp as its image. Frames are reliable and
render only for subscribers: once per camera sample, and at once for each new
subscriber, so subscribing reads the current frame even while a stepped world is
paused, without advancing time. There is no capture service. All observations
use simulation timestamps and ROS optical frame conventions, with the principal
point at `(width-1)/2, (height-1)/2`.

## Spectator streams and managed recording

Spectators exist only for recording: `launch --record DIR` enables them at 10
simulation-time FPS and 960×720. They use separate render targets and never resize
observation cameras. World publishes each view at
`world/spectator/<view>/{image_raw,camera_info}` with reliable, depth-one,
transient-local QoS: initial frames are available to matching late subscribers,
and slow consumers do not accumulate an unbounded queue. Frames can be dropped;
this is a live inspection stream, not lossless recording. Sampling includes the
initial state and follows native control boundaries, skipping missed sample
deadlines. Paused stepped worlds produce no repeated frames.

```bash
rosbridge launch abc/pouring --execution-mode stepped --detach --record ./recordings/pouring
rosbridge launch abc --record ./recordings/bag --record-type bag
rosbridge launch abc --record ./recordings/both --record-type both
rosbridge launch abc --record ./recordings/wrist --record-views wrist --record-size 640x480 --record-rate 5
```

`--record-type` accepts `mp4` (default), `bag`, or `both`. By default every view the
simulator offers is recorded at 960×720, 10 frames per world second, H.264 CRF 20.
`--record-views NAME,...` records only the named views; the simulator sets up no
others, so they cost no rendering. `--record-size WxH` (16×16 to 3840×2160),
`--record-rate HZ` (at most 50) and `--record-quality CRF` (0–51, lower is sharper)
change the rest. Each of these options requires `--record`.
Paths are relative to the invoking terminal, including with `--project`. The output
directory must be new or empty; existing recordings are never overwritten. The CLI
prints its absolute path and includes `recording_dir` in JSON output. `status`
includes current recording metadata. Recording also works with `demo`.

MP4 produces one H.264 file per view, such as `third_person.mp4`. Each view has
camera calibration, a frame/timestamp journal and an encoder log beside its video.
Bag mode writes MCAP under `DIR/bag/`, containing all world spectator images,
calibration topics and the clock. Both mode writes both representations.
RoboStack supplies rosbag2; Pixi supplies FFmpeg.

The recorder is a separate ROS process in the run's domain. Launch waits for
calibration, clock, and the initial frame of every spectator view before allowing
any physics steps. `world/wait` still controls stepped execution; recording itself
never advances simulation. MP4 timing follows simulation timestamps: paused
thinking time adds no footage, and missing sampling intervals hold the last frame.
Odd image dimensions are padded to even dimensions for H.264. Each view is one
file, `<view>.mp4`. Bags preserve the original simulation stamps.

Ctrl-C in foreground mode or `rosbridge stop RUN_ID` stops the producer, drains the
recorder and finalizes files. Encoder workers drain concurrently under one shutdown
deadline. Recording metadata is written only when it changes. A recorder failure
fails the run and stops the simulation. Forced termination may leave partial output.
`recording.json` records completion, errors, received frames, encoder-queue drops
and each view's file (`segments`); check it before treating a recording as complete.
The bounded live ROS pipeline can drop frames before the recorder sees them, so
recording is not a lossless capture guarantee.

## Sources, caches and extension

Simulator manifests declare source URL and immutable commit. Preparation clones
into `third_party/` without requiring a parent Git repository, and refuses sources
with tracked changes. It exports the pinned tree and applies ordered patches in
`.runtime/`. Source or patch fingerprint changes require removing that simulator's
generated source before repreparing; assets stay cached. Export/patch failures
never publish a partial tree.

Downloads use file locks, atomic extraction and upstream checksums where supplied.
`ROSBRIDGE_ASSET_SEED=DIR` adopts an existing, already extracted asset depot laid
out like RoboEnv's (`DIR/<simulator>/...`): each file is verified against its pin
and then renamed into the store, so it must share the store's filesystem. A seeded
preparation never downloads; a missing or mismatched file is an error.
Packages without upstream checksums record an observed digest without claiming
verification. Assets default to `.runtime/assets`; `ROSBRIDGE_ASSET_ROOT`
overrides that location. The CLI's managed Python lives in `.runtime/python`.
Both are shared by all jobs in this checkout. CLI-managed tool caches, temporary
files and ROS logs are checkout-local; HOME is unchanged.

There is no automatic pruning, log truncation or history retention limit.
`.cache/rosbridge` contains disposable run history, logs, temporary files and
rebuildable tool caches. Stop all managed runs and other commands using this
checkout before removing it; to remove one finished run's history, delete
`.cache/rosbridge/runs/RUN_ID` after that run and its `exec` commands stop. Keep
`.runtime`, `.venv`, `.pixi`, `.workspaces` and `third_party`; they contain the
prepared installation. Recordings stay in the directory passed to `--record`.

To add a simulator, implement `SimulatorBackend` (`src/rosbridge/core/backend.py`)
and register a lazy `rosbridge.backends` entry point (a `WorkerBackend` subclass
for a native worker). Supply an `ActionLayout` whose commands are positions in
feedback units (or, for a base's velocity group, body velocities whose feedback is
odometry), a `device_inventory` naming its devices by Kinex device type (`joint`,
`gripper`, `locomotion`, `rgb`, `rgbd`), each camera's image size and intrinsics
(`intrinsics(camera)`: width, height, fx, fy, cx, cy) and `SpectatorView` settings, and
declare the inventory in the manifest's `[devices]` table. A manifest's `install`
lists the generated tree's directories installed editable without dependencies
(default `["."]`; `[]` imports the tree in place).
Native methods run only on the owner thread. New device kinds require an explicit
contract and endpoint implementation. Provide a manifest, task catalog, independent
Pixi environment, asset preparer and checker, demo, a catalog rig built by
`tools/robots`, and unit/native/geometry/ROS tests.

## Verification

```bash
pixi run test
pixi run -e abc smoke
pixi run -e abc test-ros
pixi run -e libero test-ros
pixi run -e robotwin smoke && pixi run -e robotwin test-ros
pixi run -e robodojo-native smoke && pixi run -e robodojo test-ros
pixi run -e robocasa smoke && pixi run -e robocasa test-ros
pixi run -e behavior-native smoke && pixi run -e behavior test-ros
pixi run -e behavior-g1-native smoke && pixi run -e behavior-g1 test-ros
pixi run -e behavior-dex-native smoke && pixi run -e behavior-dex test-ros
pixi run -e abc python -m pytest -q tests/integration
rosbridge demo abc
rosbridge demo libero --execution-mode realtime
rosbridge demo robotwin
rosbridge demo robodojo
rosbridge demo robocasa
rosbridge demo behavior --timeout 1800
rosbridge demo behavior-g1 --timeout 1800
rosbridge demo behavior-dex --timeout 1800
```

Kinex-backed checks (rig geometry against each simulator, Kinex client tests)
need `KINEX_PYTHON=../kinex2.0/packages/core/assets/python`; without it they are
skipped while the remaining tests still run. The Kinex schema and readiness check
of every rig also needs `KINEX_PYTHON_PROJECT`, one prepared `.kinex/python`
directory, because Kinex provisions a multi-gigabyte environment per workspace.
The Kinex frame-host tests (`tests/ros/test_kinex_frames.py`) also need Kinex's full
`kinex_interfaces`, which rosbridge does not build: source that directory's
`ros/install/local_setup.bash` after the simulator overlay, or they are skipped.

Unit tests cover the command buffer, scheduling, the robot catalog, source
preparation and caches. Native checks compare actions and timing with upstream and
verify spectator state/image isolation. ROS checks verify the device endpoints,
waits, stops and spectator streams in both modes on the synthetic backend,
ABC, LIBERO, RoboTwin and RoboCasa, and RoboDojo and BEHAVIOR through their native workers.
Integration checks exercise actual CLI runs, two domains, failure
cleanup, foreground/detached shutdown and MP4/MCAP recording. Skipped checks are
not validation evidence. See [ABC](abc.md), [LIBERO](libero.md), [RoboTwin](robotwin.md),
[RoboDojo](robodojo.md), [RoboCasa](robocasa.md) and [BEHAVIOR](behavior.md) for device maps.
