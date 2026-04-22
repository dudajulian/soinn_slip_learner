cmake_minimum_required(VERSION 3.8)
project(soislip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/SOINNSample.msg"
)

set(srv_files
  "srv/GetCellFeatures.srv"
  "srv/GetMapFeatures.srv"
  "srv/PredictBatch.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
