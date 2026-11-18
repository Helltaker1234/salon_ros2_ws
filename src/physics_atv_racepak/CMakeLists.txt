cmake_minimum_required(VERSION 3.8)
project(physics_atv_racepak)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RpControls.msg"
  "msg/RpShockSensors.msg"
  "msg/RpWheelEncoders.msg"
  DEPENDENCIES std_msgs
)

install(PROGRAMS
  scripts/racepak_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
