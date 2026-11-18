cmake_minimum_required(VERSION 3.8)
project(physics_atv_visual_mapping)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install ROS node scripts
install(PROGRAMS
  scripts/ros/dino_localmapping.py
  scripts/ros/cloud_repub.py
  scripts/ros/dino_vlad_image_click_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config and data
install(DIRECTORY config data
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
