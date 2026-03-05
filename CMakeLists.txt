cmake_minimum_required(VERSION 3.8)
project(soinn_slip_learner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
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

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetCellFeatures.srv"
  "srv/GetMapFeatures.srv"
  "srv/PredictBatch.srv"
  DEPENDENCIES std_msgs geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

add_executable(gridmap_feature_extractor_node src/gridmap_feature_extractor.cpp)
ament_target_dependencies(gridmap_feature_extractor_node
  rclcpp
  std_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
  Eigen3
)

add_executable(robot_experience_collector_node src/robot_experience_collector.cpp)
ament_target_dependencies(robot_experience_collector_node
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

rosidl_target_interfaces(robot_experience_collector_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

rosidl_target_interfaces(gridmap_feature_extractor_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(slip_prediction_manager src/slip_prediction_manager.cpp)
ament_target_dependencies(slip_prediction_manager
  rclcpp
  std_msgs
  geometry_msgs
)

rosidl_target_interfaces(slip_prediction_manager
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS
  gridmap_feature_extractor_node
  robot_experience_collector_node
  slip_prediction_manager
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  soinn_slip_learner/latent_feature_extractor_node.py
  soinn_slip_learner/soinn_training_node.py
  soinn_slip_learner/soinn_prediction_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
