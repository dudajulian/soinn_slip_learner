cmake_minimum_required(VERSION 3.8)
project(soinn_slip_learner)

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
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
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

set(srv_files
  "srv/GetCellFeatures.srv"
  "srv/GetMapFeatures.srv"
  "srv/PredictBatch.srv"
)

# Generate ROS interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Create C++ executables
add_executable(gridmap_feature_extractor_node src/gridmap_feature_extractor.cpp)
ament_target_dependencies(gridmap_feature_extractor_node
  rclcpp
  std_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
  Eigen3
)
target_link_libraries(gridmap_feature_extractor_node ${cpp_typesupport_target})
  
add_executable(robot_experience_collector_node src/robot_experience_collector.cpp)
ament_target_dependencies(robot_experience_collector_node
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(robot_experience_collector_node ${cpp_typesupport_target})

add_executable(slip_prediction_manager src/slip_prediction_manager.cpp)
ament_target_dependencies(slip_prediction_manager
  rclcpp
  std_msgs
  geometry_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
)
target_link_libraries(slip_prediction_manager ${cpp_typesupport_target})



# Install C++ executables
install(TARGETS
  gridmap_feature_extractor_node
  robot_experience_collector_node
  slip_prediction_manager
  DESTINATION lib/${PROJECT_NAME}
)

# Install reusable Python module code.
ament_python_install_package(soinn_py)

# Install Python executables
install(PROGRAMS
  soinn_slip_learner/latent_feature_extractor_node.py
  soinn_slip_learner/soinn_training_node.py
  soinn_slip_learner/soinn_prediction_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install other directories
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
  install(DIRECTORY
    models
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
