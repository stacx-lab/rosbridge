cmake_minimum_required(VERSION 3.10)
project(rosbridge_interfaces)
# The world's Wait service: the one interface of a rosbridge graph that is not a standard ROS type.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "srv/Wait.srv")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
