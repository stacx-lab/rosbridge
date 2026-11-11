# rosbridge

Run public robotics simulators as managed ROS 2 graphs that speak Kinex's device
contract. One `world` node exposes every device's topics and services; world
coordinates native stepping and, when recording, spectator streams. ABC, LIBERO,
RoboTwin, RoboDojo, RoboCasa, BEHAVIOR and HandsOn have separate native environments managed
with RoboStack and Pixi; Isaac Sim 5.1 (RoboDojo, BEHAVIOR's OmniGibson, HandsOn) runs in a
Python 3.11 worker beside the ROS node.

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
- [BEHAVIOR household activities with the R1Pro, the R1Pro with Sharpa hands, or a G1 with Sharpa hands](docs/behavior.md)
- [HandsOn dexterous manipulation with two ROKAE arms and Wuji hands](docs/handson.md)
- [Kinex device contract and robot catalog](docs/kinex.md)
