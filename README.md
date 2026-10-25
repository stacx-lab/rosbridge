# rosbridge

Run public robotics simulators as managed ROS 2 graphs that speak Kinex's device
contract. One `world` node exposes every device's topics and services; world
coordinates native stepping and, when recording, spectator streams. ABC, LIBERO,
RoboTwin, RoboDojo and RoboCasa have separate native environments managed with
RoboStack and Pixi; RoboDojo's Isaac Sim 5.1 runs in a Python 3.11 worker beside the
ROS node.

```bash
bash tools/bootstrap.sh
source .venv/bin/activate
rosbridge prepare abc
rosbridge launch abc --execution-mode stepped --record ./recordings/abc
```

- [Design and CLI guide](docs/README.md)
- [ABC devices, assets and examples](docs/abc.md)
- [LIBERO devices, controllers and examples](docs/libero.md)
- [RoboTwin devices, assets and examples](docs/robotwin.md)
- [RoboDojo devices, native worker and examples](docs/robodojo.md)
- [RoboCasa mobile manipulator, assets and examples](docs/robocasa.md)
- [Kinex device contract and robot catalog](docs/kinex.md)
