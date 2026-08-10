cmake_minimum_required(VERSION 3.8)
project(soislip_core)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Python and install Python scripts
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(soislip_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Create C++ executables
add_executable(gridmap_feature_extractor_node src/gridmap_feature_extractor.cpp)
ament_target_dependencies(gridmap_feature_extractor_node
  rclcpp
  soislip_interfaces
  std_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
  Eigen3
)
  
add_executable(robot_experience_collector_node src/robot_experience_collector.cpp)
ament_target_dependencies(robot_experience_collector_node
  rclcpp
  soislip_interfaces
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(slip_prediction_manager src/slip_prediction_manager.cpp)
ament_target_dependencies(slip_prediction_manager
  rclcpp
  soislip_interfaces
  std_msgs
  geometry_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
)

add_executable(camera_odom_republisher src/camera_odom_republisher.cpp)
ament_target_dependencies(camera_odom_republisher
  rclcpp
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install C++ executables
install(TARGETS
  gridmap_feature_extractor_node
  robot_experience_collector_node
  slip_prediction_manager
  camera_odom_republisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install reusable Python module code.
ament_python_install_package(soinn_py)

# Install Python executables
install(PROGRAMS
  soislip_core/latent_feature_extractor_node.py
  soislip_core/sample_recorder_node.py
  soislip_core/soinn_training_node.py
  soislip_core/soinn_prediction_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
