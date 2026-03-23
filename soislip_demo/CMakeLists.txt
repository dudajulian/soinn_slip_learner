cmake_minimum_required(VERSION 3.8)
project(soislip_demo)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/startup.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  resources
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
