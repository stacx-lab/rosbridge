cmake_minimum_required(VERSION 3.10)
project(custom_interfaces)
# The interfaces of a graph's devices that are not standard ROS types: the world's Wait service and Json, the
# free-form service of operations whose description defines their fields. LookupTransform is not served here:
# it keeps the package the same as the one Kinex builds, whose frame service uses it.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "srv/Json.srv" "srv/Wait.srv" "srv/LookupTransform.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces unique_identifier_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
