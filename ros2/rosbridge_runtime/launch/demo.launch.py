"""The world graph plus a backend's control demo; the graph exits with the demo."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD = ("backend", "task", "seed", "mode", "spectator_rate_hz", "spectator_width", "spectator_height",
         "spectator_select", "terminate_on_success")


def generate_launch_description():
    world = Path(get_package_share_directory("rosbridge_runtime")) / "launch/sim.launch.py"
    controller = Node(package=LaunchConfiguration("demo"), executable="control_demo", output="screen",
                      parameters=[{"use_sim_time": True}])

    def finish(event, context):
        if event.returncode:
            raise RuntimeError(f"control demo failed (exit {event.returncode})")
        return [EmitEvent(event=Shutdown(reason="control demo passed"))]

    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value="") for name in (*WORLD, "demo")),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(world)),
                                 launch_arguments={name: LaunchConfiguration(name) for name in WORLD}.items()),
        RegisterEventHandler(OnProcessExit(target_action=controller, on_exit=finish)),
        controller,
    ])
