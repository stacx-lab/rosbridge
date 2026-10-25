cmake_minimum_required(VERSION 3.10)
project(kinex_interfaces)
# rosbridge's copy of Kinex's world Wait service, so preparation needs no Kinex checkout.
# srv/Wait.srv must stay byte-identical to Kinex's packages/core/assets/ros/kinex_interfaces:
# ROS 2 matches services by type name and definition hash.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "srv/Wait.srv")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
