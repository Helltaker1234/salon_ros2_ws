cmake_minimum_required(VERSION 3.8)
project(context_adaptation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install node scripts
install(PROGRAMS
  scripts/dino_costmap_gp_speed_input_output.py
  scripts/cost_publisher.py
  scripts/quadruped_cost_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install assets and launch
install(DIRECTORY assets launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
