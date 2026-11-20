# Devices and robot catalog

Every device of a run is a set of endpoints on the single `world` node. They use
standard ROS types, `std_srvs/srv/Trigger` stops and two types of `custom_interfaces`
(`ros2/custom_interfaces`): `Wait`, the world service, and `Json`, the free-form
service of a lensed camera's geometry. A ROS 2 program on the run's domain operates the
devices with nothing else: in the run's environment (`rosbridge exec`), or anywhere
`custom_interfaces` is built. Each device is described by a profile of the
robot catalog (its endpoints, types, QoS, units, limits, frames and URDF), which
`rosbridge setup` exports. The device types, operation names and profile format are
Kinex's, so a Kinex workspace uses an export as it is; see [Kinex](kinex.md).

## Device contract

| Device type | Operations (endpoint, type) |
|---|---|
| `world` | `time` (`/clock`, Clock), `wait` (`/world/wait`, Wait), `budget` (`/world/budget`, Float64MultiArray), `stop` (`/world/stop`) |
| `joint` (arm, torso) | `joint_state` (`/<arm>/joint_state`, JointState), `joint_target` (`/<arm>/joint_target`, JointTrajectory), `stop` |
| `joint` commanded by its tip pose (HandsOn's arms, LIBERO's arm with `--config osc`) | `joint_state`, `pose_target` (`/<arm>/pose_target`, MultiDOFJointTrajectory in `<arm>/base`), `pose_state` (`/<arm>/pose_state`, MultiDOFJointState), `stop` |
| `joint` commanded by joints or its tip pose (RoboDojo's arms) | `joint_state`, `joint_target`, `pose_target`, `pose_state`, `stop`; the latest command of either kind replaces the active one, and a stop holds the measured joints |
| `gripper` | `joint_state`, `target` (`/<gripper>/target`, Float64 position), `stop` |
| `locomotion` (base) | `velocity` (`/<base>/velocity`, Twist), `odometry` (`/<base>/odometry`, Odometry), `stop` |
| `rgb` (camera) | `image` (`/<camera>/image`), `camera_info`, in frame `<camera>/base` |
| `rgbd` (camera) | `image`, `depth` (`/<camera>/depth`, 16UC1 mm, same stamp as its image), `camera_info` |

A gripper's `target` is the URDF position of its `grip` joint, in the simulator's
native command units. LIBERO's grip joint is `finger_joint1` in metres. ABC,
RoboTwin and RoboDojo keep their policies' normalized opening (0 closed, 1 open):
the gripper URDF has a geometry-less virtual drive joint `<side>_gripper` with
limits [0, 1], and the finger joints mimic it with the native affine scale
(`multiplier`, `offset`). Policies trained on the native action spaces therefore
command arms with absolute native joint positions and grippers with the same
normalized value they were trained on. RoboCasa's grip joint `gripper` is robosuite's
integrated command, limits [-1, 1] with `open: -1` and `closed: 1`, which its finger
joints mimic from their actuator ranges. BEHAVIOR's R1Pro grip joint `<side>_gripper`
is the evaluator's smooth command, limits [-1, 1] with `open: 1` and `closed: -1`.
HandsOn's Wuji hands are `joint` devices of 20 joints on their arm's `palm`, the tip link
of each ROKAE arm and the frame HandsOn's palm-pose actions command. BiDex-100's Sharpa hands are `joint` devices of 22 joints on their arm's
`flange`; as BiDex's evaluator does, rosbridge passes their targets and the arms' to MuJoCo's
actuators unchanged, which bound them.

A base is a `locomotion` device in the single mode `drive`: `velocity`
declares the commandable axes in native units (RoboCasa and BEHAVIOR's R1Pro:
normalized `vx`, `vy`, `wz`), the base stops 0.5 s of world time after the last twist, and `odometry`
reports the twist in `<base>/base` with a pose in `<base>/odom` relative to the
episode start. Its `frames.base` is an unlocalized root; a mobile rig chains its
other devices from it (`torso` on `base:base`, the arm on `torso:top`).

World waits are sequential: one requested while another is pending is rejected, and
`world/stop` cancels the pending wait. The latched `world/budget` topic reports the
episode budget after each step: `data` is `[used_s, max_s]`, seconds of world time, the
control steps taken and the step limit times the control period (`max_s` inf without a
limit). Only a stepped run has a budget; a realtime run publishes `[NaN, NaN]`. Realtime observes progress; stepped mode
advances only for a pending wait; shutdown interrupts it. Commands are
validated on the simulator owner thread; invalid ones are logged and keep the held
target. A device's `stop` holds the measured position, a gripper's keeps its applied grip
target so a held object stays gripped (Kinex's grip contract), and a base's commands zero
velocity; each stop is created after its command subscription, so a client that
discovered `stop` reaches the device.
Joint states and CameraInfo use transient-local reliable QoS. Images are reliable
and volatile and render only for subscribers: once per camera sample (30 Hz of
simulation time, or the simulator's own rate, 2 Hz for HandsOn), and at once for each
new subscriber, so a reader that subscribes, takes one frame and unsubscribes gets
the current frame even in a paused stepped world.

Camera profiles declare pinhole intrinsics in ROS pixel centers. A camera rendered
as an undistorted pinhole declares `distortion: {model: none}`; its FOV and rays
follow from the projection. A camera its evaluator renders through a lens
(HandsOn's head camera) declares the lens as OpenCV's Brown-Conrady model
(`opencv_brown_conrady`: k1, k2, p1, p2, k3) and its FOV, and the world node serves
three `custom_interfaces/srv/Json` services through that lens; images stay as rendered:

| Service | JSON request | JSON reply |
|---|---|---|
| `<camera>/geometry` | `{}` | `{streams: {color: descriptor}}`: width, height, projection, distortion, the optical `frame` and a `calibration_id` |
| `<camera>/unproject` | `{stream, calibration_id, pixels: [[u, v], ...]}` | `{rays: [[x, y, 1], ...]}` in the optical frame |
| `<camera>/project` | `{stream, calibration_id, points: [[X, Y, Z], ...]}`, Z > 0 | `{pixels: [[u, v], ...]}` |

The request is the `request` string and the reply the `response` string; a failure sets
`success` false and names its cause in `message`. A conversion names the `calibration_id`
that `geometry` returned, takes at most 8192 coordinates and returns null for an entry it
cannot convert. `camera_info` mirrors the profile, its plumb_bob D included, and is not a
geometry source. `fps` is null because frames
render on demand rather than at a fixed rate.

The world node publishes state and measurements without TF. Frames are data in the
profiles: a device's `frames` give its base, and a gripper's `tcp`, in a parent frame
written `device:frame`, and the URDF of a joint or gripper device (`model`) gives its
links from `joint_state`. A client composes them itself. The frame IDs on the graph name
these frames as `<device>/<frame>`: `<camera>/base` on images, `<base>/odom` and
`<base>/base` on odometry, `<arm>/base` for pose targets.

## Robot catalog and workspace setup

`rosbridge robots list` lists every rig. Builders in `tools/robots` measure each
rig from its simulator (`build_abc.py`, `build_robotwin.py`, `build_robodojo.py`,
`build_robocasa.py`, `build_behavior.py`, `build_handson.py`, `build_bidex.py`);
the native [Kinex checks](kinex.md#compatibility-checks) compare the exported profiles
with the simulator.

| Robot ID | Devices | Catalog tasks |
|---|---|---:|
| `abc/abc-yam` | `left_arm`, `right_arm` (YAM, `<side>_joint1..6`), normalized grippers, RGB-D `camera_top`, `camera_left`, `camera_right` | 33 |
| `robotwin/robotwin-aloha-agilex` | `left_arm` (`fl_joint1..6`), `right_arm` (`fr_joint1..6`), normalized grippers, RGB-D `head_camera`, `left_camera`, `right_camera` | 50 |
| `robodojo/robodojo-arx-x5` | `left_arm`, `right_arm` (ARX X5, both `joint1..6`; joint or `flange` pose targets), normalized grippers, RGB (RGB-D with `--config depth`) `head_camera`, `left_camera`, `right_camera` | 42 |
| `robocasa/robocasa-panda-omron` | `base` (Omron, locomotion), `torso` (lift), `arm` (Panda, `robot0_joint1..7`), integrated-command `gripper`, RGB-D `camera_left`, `camera_right`, `camera_wrist` | 365 |
| `behavior/behavior-r1pro` | `base` (holonomic, locomotion), `torso` (`torso_joint1..4`), `left_arm`, `right_arm` (`<side>_arm_joint1..7`), smooth grippers, RGB-D `head_camera` (720²), `left_camera`, `right_camera` (480²) | 100 |
| `handson/handson-rokae-wuji` | `right_arm`, `left_arm` (ROKAE AR5, `joint1..7_<side>`, tip `palm`), `right_hand`, `left_hand` (Wuji, `<side>_finger1..5_joint1..4`), RGB `head_camera` (854×480), `left_camera`, `right_camera` (848×480, on the palms) | 66 |
| `bidex/bidex-fr3v2-sharpa-tabletop`, `bidex/bidex-fr3v2-sharpa-dexjoco` | `right_arm`, `left_arm` (Franka FR3v2, `<side>_fr3v2_joint1..7`), `right_hand`, `left_hand` (Sharpa Wave, 22 joints), RGB-D `head_camera`, `left_camera`, `right_camera` (480×360, on the flanges); the two differ in the base pose | 152, 40 |
| `bidex/bidex-astribot-s1-sharpa` | `right_arm`, `left_arm` (Astribot S1, `astribot_arm_<side>_joint_1..7`), `right_hand`, `left_hand` (Sharpa Wave, 22 joints), RGB-D `head_camera` (in the head), `left_camera`, `right_camera` (480×360, on the flanges) | 20 |

The LIBERO rigs are the same Panda arm and gripper
with distinct fixed mounts and camera calibration. Each rig directory is a complete
workspace skeleton (`workspace.yaml` and `devices/<name>/profile.yaml`) whose
profiles carry its calibration. Files shared by all rigs (URDFs, the gripper,
wrist-camera and world profiles, and the `osc` configuration that
`tools/robots/build_libero.py` writes) are symlinks into `libero/panda`; export
copies their content.

| Robot ID | Catalog tasks | Example task |
|---|---:|---|
| `libero/libero-panda-tabletop` | 20 | `libero/libero-spatial-01` |
| `libero/libero-panda-floor` | 10 | `libero/libero-object-01` |
| `libero/libero-panda-kitchen` | 50 | `libero/libero-10-03` |
| `libero/libero-panda-living-room` | 32 | `libero/libero-10-01` |
| `libero/libero-panda-study` | 18 | `libero/libero-10-06` |

```sh
rosbridge robots list
rosbridge tasks list --simulator libero --robot libero/libero-panda-tabletop
rosbridge setup libero/libero-spatial-01 --directory ../my-libero-workspace --domain-id 83
rosbridge launch libero/libero-spatial-01 --execution-mode stepped --domain-id 83
```

`rosbridge setup` takes a task as `launch` does, with `--seed` and `--config`. It exports the
workspace that a launch of that task, seed and configuration serves: the task's rig in that
configuration, with the profile changes its simulator makes for that task and seed. HandsOn
changes its cameras' intrinsics and mounts by task and seed, and BiDex-100 moves cameras in its
`-ood` tasks (see their pages). The seed defaults as launch's does. Setup is offline: it exports
unloaded profiles on the selected ROS domain into `.kinex/` (`workspace.yaml`, `devices/` and
`kinex.json`); it launches nothing, and it refuses an existing workspace. Export a workspace
for each task, seed and configuration whose devices differ, and launch with the same ones.

A rig may also ship Kinex workspace content beside its profiles: a `lib/` function library
(`kinex.lib.<module>`), `programs/` and `skills/`. Setup copies these into `.kinex/`, and it
refuses a workspace that already has any of them. No rig ships any yet.

A simulator that serves alternative device sets declares each one in its manifest, as
`[configurations.NAME]` with a `description` and the device kinds it changes (`devices`).
RoboDojo's `depth` makes its cameras RGB-D. Every rig of that simulator holds
`configurations/NAME/`:

- an optional `profiles.yaml`, the configuration's profile changes by device, as JSON merge
  patches that the rig builder writes;
- optional `lib/`, `programs/` and `skills/`, whose files are added over the rig's.

The backend is created with `configuration=NAME`. The catalog tests export every rig in every
configuration its simulator declares and check the device kinds against the manifest.

## Episodes and judging

Task instructions and judging are not device operations or ROS topics. Supply
the agent objective in the user prompt; the operator reads native outcomes with
`rosbridge judge RUN_ID --json`.

A run is one episode; the world offers no reset, so starting over is the operator's:
stop the run and launch another. An episode ending at a step limit, a failed simulator
step or, with `--early-stop`, where the simulator's evaluator ends it (a success, or a
task failure with its cause) pauses physics, and wait replies from then on say which.
The graph remains available for images and clock; new commands are rejected. The
episode's native outcome is appended once, when it ends or the run stops, to
`episode-results.jsonl` in rosbridge's run directory; a simulator may add its own measures, such as BEHAVIOR's partial credit `q_score`
or RoboCasa's `split` and kitchen `layout_id` and `style_id`, except after a failed step. An interrupted run is marked stopped, not a successful terminal
episode. Judge records are separate from the agent workspace; this separation is not
an OS security sandbox.
