# Kinex integration

rosbridge speaks Kinex's device contract directly: every device of a run is a set
of endpoints on the single `world` node, using standard ROS types,
`std_srvs/srv/Trigger` stops and Kinex's `kinex_interfaces/srv/Wait` world service.
Exported Kinex profiles point straight at those endpoints, so Kinex borrows the
simulator instead of relaying it; only the URDF link publishers (`links.py`) of the
joint and gripper devices run in Kinex. rosbridge builds `Wait` from its own copy
(`ros2/kinex_interfaces`, byte-identical to Kinex's), so builds need no Kinex checkout.

## Device contract

| Device type | Operations (endpoint, type) |
|---|---|
| `world` | `time` (`/clock`, Clock), `wait` (`/world/wait`, Wait), `stop` (`/world/stop`) |
| `joint` (arm, torso) | `joint_state` (`/<arm>/joint_state`, JointState), `joint_target` (`/<arm>/joint_target`, JointTrajectory), `stop` |
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
is the evaluator's smooth command, limits [-1, 1] with `open: 1` and `closed: -1`. The
Sharpa hands, on the G1 and on the R1Pro of `behavior-dex`, are `joint` devices: 22
commanded joints each and no tip link.

A base is a Kinex `locomotion` device in the single mode `drive`: `velocity`
declares the commandable axes in native units (RoboCasa and BEHAVIOR's R1Pro:
normalized `vx`, `vy`, `wz`, also with Sharpa hands; the BEHAVIOR G1: `vx`, `vy`, `wz` in m/s and rad/s within
GR00T's navigation limits), the base stops 0.5 s of world time after the last twist, and `odometry`
reports the twist in `<base>/base` with a pose in `<base>/odom` relative to the
episode start. Its `frames.base` is an unlocalized root; a mobile rig chains its
other devices from it (`torso` on `base:base`, the arm on `torso:top`).

World waits are sequential: one requested while another is pending is rejected, and
`world/stop` cancels the pending wait. Realtime observes progress; stepped mode
advances only for a pending wait; shutdown interrupts it. Commands are
validated on the simulator owner thread; invalid ones are logged and keep the held
target. A device's `stop` holds the measured position (a base commands zero
velocity) and is created after its command subscription, so a Kinex client that
discovered `stop` reaches the device.
Joint states and CameraInfo use transient-local reliable QoS. Images are reliable
and volatile and render only for subscribers: once per camera sample, and at once
for each new subscriber, so a Kinex read (`helpers.camera_frame`, snapshots) gets
the current frame even in a paused stepped world.

Camera profiles declare undistorted pinhole intrinsics (`distortion: {model: none}`),
so Kinex derives FOV and computes rays locally; cameras serve no geometry
services. `camera_info` mirrors the profile and is not a geometry source. `fps` is
null because frames render on demand rather than at a fixed rate.

The simulator publishes state and measurements without TF. Kinex computes FK/IK
from the arm URDF, publishes calibrated mounts and fixed frames such as
`gripper/tcp`, and owns tf2 lookup through `kinex.helpers.lookup_transform`.

## Robot catalog and workspace setup

`rosbridge robots list` lists every rig. Builders in `tools/robots` measure each
rig from its simulator (`build_abc.py`, `build_robotwin.py`, `build_robodojo.py`,
`build_robocasa.py`, `build_behavior.py`, `build_behavior_g1.py`, `build_behavior_dex.py`);
native geometry tests check the exported profiles with Kinex's own FK against the
simulator.

| Robot ID | Devices | Catalog tasks |
|---|---|---:|
| `abc/abc-yam` | `left_arm`, `right_arm` (YAM, `<side>_joint1..6`), normalized grippers, RGB-D `camera_top`, `camera_left`, `camera_right` | 33 |
| `robotwin/robotwin-aloha-agilex` | `left_arm` (`fl_joint1..6`), `right_arm` (`fr_joint1..6`), normalized grippers, RGB-D `head_camera`, `left_camera`, `right_camera` | 50 |
| `robodojo/robodojo-arx-x5` | `left_arm`, `right_arm` (ARX X5, both `joint1..6`), normalized grippers, RGB `head_camera`, `left_camera`, `right_camera` | 42 |
| `robocasa/robocasa-panda-omron` | `base` (Omron, locomotion), `torso` (lift), `arm` (Panda, `robot0_joint1..7`), integrated-command `gripper`, RGB-D `camera_left`, `camera_right`, `camera_wrist` | 365 |
| `behavior/behavior-r1pro` | `base` (holonomic, locomotion), `torso` (`torso_joint1..4`), `left_arm`, `right_arm` (`<side>_arm_joint1..7`), smooth grippers, RGB-D `head_camera` (720²), `left_camera`, `right_camera` (480²) | 100 |
| `behavior-g1/behavior-g1-sharpa` | `base` (legged, locomotion), `torso` (waist), `left_arm`, `right_arm` (G1, 7 joints), `left_hand`, `right_hand` (Sharpa Wave, 22 joints), RGB-D `head_camera` | 100 |
| `behavior-dex/behavior-r1pro-sharpa` | `base` (holonomic, locomotion), `torso` (`torso_joint1..4`), `left_arm`, `right_arm` (`<side>_arm_joint1..7`), `left_hand`, `right_hand` (Sharpa Wave, 22 joints, in place of the grippers), RGB-D `head_camera` (720²), `left_camera`, `right_camera` (480², on the backs of the hands) | 100 |

The LIBERO rigs are the same Panda arm and gripper
with distinct fixed mounts and camera calibration. Each rig directory is a complete
Kinex workspace skeleton (`workspace.yaml` and `devices/<name>/profile.yaml`) whose
profiles carry its calibration. Files shared by all rigs (URDFs, `links.py`, and the
gripper, wrist-camera and world profiles) are symlinks into `libero/panda`; export
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
rosbridge setup libero/libero-panda-tabletop --directory ../my-libero-workspace --domain-id 83
rosbridge launch libero/libero-spatial-01 --execution-mode stepped --domain-id 83
```

Setup is offline: it exports unloaded profiles on the selected ROS domain and
writes `.kinex/kinex.json` with the `ros-jazzy` provider; it neither starts Kinex
nor launches a simulator, and it refuses an existing workspace. Prepare Kinex
Python separately; it builds `kinex_interfaces` and installs ROS clients in its own
environment. A workspace belongs to one rig: launch tasks whose robot ID matches
it, and export another workspace for another rig. Each simulator's native
`test_kinex_geometry.py` checks its rigs' profile frames against the simulator.

## Episodes and judging

Task instructions and judging are not device operations or ROS topics. Supply
the agent objective in the user prompt; the operator reads native outcomes with
`rosbridge judge RUN_ID --json`.

A run is one episode; the world offers no reset, so starting over is the operator's:
stop the run and launch another. An episode ending on success (with
`--stop-on-success`), a step limit or a failed simulator step pauses physics, and
wait replies from then on say which. The graph remains available for images and
clock; new commands are rejected. The episode's native outcome is appended once, when
it ends or the run stops, to `episode-results.jsonl` in rosbridge's run directory; a
simulator may add its own measures, such as BEHAVIOR's partial credit `q_score`, except
after a failed step. An interrupted run is marked stopped, not a successful terminal
episode. Judge records are separate from the agent workspace; this separation is not
an OS security sandbox.
