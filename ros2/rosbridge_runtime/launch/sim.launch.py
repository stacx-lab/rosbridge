"""The world graph for one backend; omitted arguments keep the node's defaults."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TYPES = {"backend": str, "task": str, "seed": int, "mode": str, "spectator_rate_hz": float,
         "terminate_on_success": lambda value: {"true": True, "false": False}[value.lower()]}


def world(context):
    values = {name: LaunchConfiguration(name).perform(context) for name in TYPES}
    parameters = {name: convert(values[name]) for name, convert in TYPES.items() if values[name]}
    node = Node(package="rosbridge_runtime", executable="sim_node", name="world", output="screen",
                parameters=[parameters])

    def exited(event, context):
        if not context.is_shutdown:
            raise RuntimeError(f"world process exited unexpectedly (exit {event.returncode})")
    return [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=exited)), node]


def generate_launch_description():
    return LaunchDescription([*(DeclareLaunchArgument(name, default_value="") for name in TYPES),
                              OpaqueFunction(function=world)])
