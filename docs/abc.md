# ABC simulator

ABC runs the native MuJoCo YAM dual-arm tasks from
`https://github.com/amazon-far/abc`, pinned at
`d0832d12651d1b260a652861a14648dc5f3660c7`. See the
[design and CLI guide](README.md) for shared lifecycle and timing contracts.

## Preparation and task selection

```bash
bash tools/bootstrap.sh
source .venv/bin/activate
rosbridge prepare abc --task put-plastic-bottles-in-bin
rosbridge launch abc --execution-mode stepped
# Prepare all 33 cataloged tasks:
rosbridge prepare abc
rosbridge tasks list --simulator abc
rosbridge launch abc/put-markers-in-top-drawer
```

The default task is `abc/put-plastic-bottles-in-bin`. Unambiguous native names such
as `put_plastic_bottles_in_bin` also work.

Preparation installs simulator dependencies only; it does not install upstream
policy/training stacks or download model checkpoints. It downloads the shared YAM
model and selected task assets. Full preparation also includes the larger RoboCasa
object packs for in-hand transfer. Repeated preparation reuses verified cached
packages. Launch never downloads. `MUJOCO_GL=egl` is set in the runtime environment.

`patches/abc/0001-complete-task-asset-map.patch` fills missing drawer/marker and
building-block mappings in upstream's preparation script. It applies only to the
generated runtime, not the pinned checkout, and does not alter physics or scoring.
`0002-preserve-mounted-asset-paths.patch` retains logical compiler asset paths when
randomizers rebuild XML. This keeps relative sibling mesh references valid across
separately cached asset packs, including the clutter task's crate meshes.
`0003-unified-arm-mounts.patch`, ported from RoboEnv, raises both arm bases from
z=0.75 to the 0.76 used by every other scene in six scene files (ten tasks), so all
33 tasks share the `abc/abc-yam` rig. It moves the robots by 1 cm; objects, physics
parameters and scoring are unchanged.

## Nodes and controls

The simulator hosts these devices on one `world` node:

| Device | Interfaces |
|---|---|
| `world` | `/clock`, `world/wait`, `world/stop`, `world/reset`, `world/spectator/*` while recording |
| `left_arm`, `right_arm` | `<arm>/joint_state`, `<arm>/joint_target`, `<arm>/stop` |
| `left_gripper`, `right_gripper` | `<gripper>/joint_state`, `<gripper>/target`, `<gripper>/stop` |
| `camera_top`, `camera_left`, `camera_right` | `<camera>/image`, `<camera>/depth`, `<camera>/camera_info`, `<camera>/capture` |

Every task runs on the Kinex rig `abc/abc-yam`; `rosbridge setup abc/abc-yam`
exports its workspace (arm URDFs measured from the MuJoCo model, mounts at
(0.2525, ±0.31, 0.76), wrist cameras on the `link_6` flanges).

Arm targets are `trajectory_msgs/msg/JointTrajectory` messages over
`<side>_joint1..6` in radians within native model bounds: the arm part of
upstream's 14-D policy action. Gripper targets are `std_msgs/msg/Float64` in
`[0,1]`, closed to open, upstream's policy gripper value with its 0.0475-metre
finger scale; `joint_state` reports the same normalized `<side>_gripper`, and the
rig's gripper URDF drives both finger links from it. Invalid targets are logged
and leave the held target unchanged. Each `stop` (`std_srvs/srv/Trigger`) holds
the measured position.

The cameras are Kinex `rgbd` devices, as in RoboEnv: `<camera>/depth` publishes
16UC1 millimetres along the optical axis, registered to the image (0 where nothing
lies within MuJoCo's clip range), and `capture` is `kinex_interfaces/srv/CaptureRGBD`.

```bash
rosbridge exec RUN_ID -- ros2 topic pub --once /left_gripper/target \
  std_msgs/msg/Float64 '{data: 0.0}'
rosbridge exec RUN_ID -- ros2 service call /world/wait \
  kinex_interfaces/srv/Wait '{duration_s: 1.0}'
```

In stepped mode, follow commands with a positive wait to advance physics. Use a
spinning controller and wait for subscription discovery before publishing.
A stepped wait renders camera frames back to back; best-effort subscribers of the
640×480 image streams may lose such bursts, so use `capture` for reliable frames.

## Native behavior and defaults

One ABC step is one native `env.step`, currently 17 physics substeps × 0.002 seconds
= 0.034 seconds (~29.41 Hz). Timing is read from the initialized environment.
A one-second wait therefore advances 1.020 simulation seconds in stepped mode.
Both modes preserve physics, randomization, action limits and native evaluators.
Upstream has no evaluator for `put_markers_in_{top,middle,bottom}_drawer` and
`build_wood_block_tower`; rosbridge judges them with RoboEnv's rules (all five
markers inside the closed drawer and released, or a released five-level tower,
held for one second). These rules only read state.

| Setting | Default |
|---|---|
| Execution, seed, randomize | realtime, 0, true |
| Maximum steps, stop on success | unlimited, false |
| Observation cameras | top, left, right; 640×480; 30 Hz while subscribed |
| Spectators, with `--record` only | third_person, wrist, right_wrist; 960×720; 10 Hz |

Rates throttle publication at native control boundaries and never change control
frequency. Reset preserves the graph and restarts native time.

The spectator roles use top, left-wrist, and right-wrist native camera poses.
They render live state into a dedicated MuJoCo GL context and framebuffer, without
changing `model.vis`, native dynamics, or the observation renderer's resolution.
World publishes each at `world/spectator/<view>/image_raw` with its calibration.
Spectator setup and capture occur on the simulation owner thread. High-resolution
rendering consumes time, so realtime throughput depends on the host and settings.
Blender rendering and MoveIt are outside this integration.
Use `--record DIR` for managed MP4 recording, or `--record-type bag|both` for MCAP
or both formats. Recording starts with initial spectator frames and finalizes on
Ctrl-C or `rosbridge stop RUN_ID`; see the shared guide for output details.

## Checks and example controller

```bash
pixi run -e abc test
pixi run -e abc smoke
pixi run -e abc test-ros
rosbridge demo abc
rosbridge demo abc --execution-mode realtime
```

The finite demo checks arm trajectories and return, gripper close/open,
invalid-target rejection, an RGB-D capture and advancing simulation time. It
reports measured realtime factor; this is an actuation check, not a task-success
benchmark. Native tests additionally compare a wrapped rollout to upstream and
check that 960×720 spectator capture leaves physics and observation images intact,
and `test_kinex_geometry.py` checks the exported rig's arm FK, finger mimics, TCP
and camera mounts against MuJoCo on patched and unpatched scenes.
All 33 cataloged tasks also have a native load-and-step check, which validates
actual XML/mesh references beyond cached asset inventories. ROS tests check
observations, initial and advancing spectator streams and reset in stepped and
realtime modes.
