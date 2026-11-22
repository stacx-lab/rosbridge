# Using rosbridge with Kinex

rosbridge does not depend on Kinex: its graphs, profiles and tests stand on their own
(see [Devices and robot catalog](devices.md)). Its device types, operation names and
profile format are Kinex's, so a workspace that `rosbridge setup` exports is a Kinex
workspace: `.kinex/workspace.yaml`, `.kinex/devices/<name>/profile.yaml` and
`.kinex/kinex.json` with the `ros-jazzy` provider, beside the episode's
`.kinex/task_description.json` ([Task description](devices.md#task-description)). Setup neither
starts Kinex nor prepares its Python environment.

```sh
rosbridge setup libero/libero-spatial-01 --directory ../my-libero-workspace --domain-id 83
rosbridge launch libero/libero-spatial-01 --execution-mode stepped --domain-id 83
```

Every operation of a profile is an endpoint of rosbridge's world node, so Kinex borrows the
simulator instead of relaying it. What Kinex adds runs in Kinex:

- the links of each joint and gripper device's URDF, which Kinex publishes on TF from
  `joint_state` while the device is loaded;
- its frame host, which publishes the mounts and fixed frames of the profiles' `frames`,
  such as `gripper/tcp`, and answers frame lookups;
- FK and IK from the URDFs, and the rays of undistorted pinhole cameras from their profiles.

## Interface types

The world's `wait` is `custom_interfaces/srv/Wait` and a lensed camera's `geometry`,
`unproject` and `project` are `custom_interfaces/srv/Json`. rosbridge and Kinex each build an
identical copy of the package, from `ros2/custom_interfaces` here and from Kinex's own assets
there, so the types are the same on the wire and neither project needs the other to build or
to call them. The package also holds `LookupTransform`, the type of Kinex's frame service;
rosbridge carries it to keep the two copies the same and does not serve it. The camera
operations take and return the JSON of Kinex's camera geometry operations.

## Compatibility checks

Every check that needs Kinex is in `tests/kinex`, apart from rosbridge's own tests, which
run without it. They take Kinex Python from
`KINEX_PYTHON=../kinex2.0/packages/core/assets/python` and are skipped without it.

| Directory | Checks | Runs in |
|---|---|---|
| `tests/kinex/unit` | `custom_interfaces` is the same package here and in Kinex; Kinex's profile schema and readiness plan accept every rig | the default environment, next to a Kinex checkout |
| `tests/kinex/ros` | Kinex's ROS communication and device host on a rosbridge overlay; `workspace.mjs`, an end-to-end program against a launched LIBERO run | a sourced simulator overlay |
| `tests/kinex/native` | `test_<simulator>.py`: the exported rig's FK, mounts and camera frames, through Kinex's kinematics, against the simulator | that simulator's native environment |

```bash
export KINEX_PYTHON=$PWD/../kinex2.0/packages/core/assets/python
pixi run --frozen python -m pytest -q tests/kinex/unit
pixi run --frozen -e abc bash -c \
    'source .workspaces/abc/install/local_setup.bash && python -m pytest -q tests/kinex/ros'
pixi run --frozen -e abc python -m pytest -q tests/kinex/native/test_abc.py
pixi run --frozen -e handson-native env PYTHONPATH=src:simulators/handson:.runtime/handson \
    python -m pytest -q -p no:cacheprovider tests/kinex/native/test_handson.py
```

LIBERO, RoboTwin, RoboCasa and BiDex-100 run as ABC does, and RoboDojo and BEHAVIOR as
HandsOn does, in `robodojo-native` and `behavior-native`. The schema and readiness check also
needs Node, a built Kinex core and `KINEX_PYTHON_PROJECT`, one prepared `.kinex/python`
directory, because Kinex provisions a multi-gigabyte environment per workspace.
