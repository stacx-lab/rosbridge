# Using rosbridge with Kinex

rosbridge does not depend on Kinex: its graphs, profiles and tests stand on their own
(see [Devices and robot catalog](devices.md)). Its device types, operation names and
profile format are Kinex's, so a workspace that `rosbridge setup` exports is a Kinex
workspace: `.kinex/workspace.yaml`, `.kinex/devices/<name>/profile.yaml` and
`.kinex/kinex.json` with the `ros-jazzy` provider. Setup neither starts Kinex nor prepares
its Python environment.

```sh
rosbridge setup libero/libero-spatial-01 --directory ../my-libero-workspace --domain-id 83
rosbridge launch libero/libero-spatial-01 --execution-mode stepped --domain-id 83
```

Every operation of a profile is an endpoint of rosbridge's world node, so Kinex borrows the
simulator instead of relaying it. What Kinex adds runs in Kinex:

- the `links.py` of each joint and gripper device, which publishes its URDF links on TF from
  `joint_state`;
- its frame host, which publishes the mounts and fixed frames of the profiles' `frames`,
  such as `gripper/tcp`, and answers frame lookups;
- FK and IK from the URDFs, and the rays of undistorted pinhole cameras from their profiles.

## Interface types

The world's `wait` is `rosbridge_interfaces/srv/Wait` and a lensed camera's `geometry`,
`unproject` and `project` are `rosbridge_interfaces/srv/Json`. Both have the request and
reply of Kinex's `kinex_interfaces` types of the same names, and the camera operations take
and return the JSON of Kinex's camera geometry operations, but they are rosbridge's own
types. To call them, Kinex needs:

- `rosbridge_interfaces` in its Python environment: build `ros2/rosbridge_interfaces` there,
  or source a simulator overlay (`.workspaces/<simulator>/install/local_setup.bash`);
- to treat `rosbridge_interfaces/srv/Json` as it treats its own `Json`, a JSON object in
  `request` and one in `response`, and `rosbridge_interfaces/srv/Wait` as its own `Wait`,
  whose abandoned wait stops the world.

## Compatibility checks

Every check that needs Kinex is in `tests/kinex`, apart from rosbridge's own tests, which
run without it. They take Kinex Python from
`KINEX_PYTHON=../kinex2.0/packages/core/assets/python` and are skipped without it.

| Directory | Checks | Runs in |
|---|---|---|
| `tests/kinex/unit` | `rosbridge_interfaces` keeps the definitions of Kinex's `Wait` and `Json`; Kinex's profile schema and readiness plan accept every rig | the default environment, next to a Kinex checkout |
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
