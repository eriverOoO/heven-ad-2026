cmake_minimum_required(VERSION 3.8)
project(ad_morai_interfaces)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BridgeStatistics.msg"
  "msg/Collision.msg"
  "msg/CollisionArray.msg"
  "msg/CtrlCmd.msg"
  "msg/EgoVehicleStatus.msg"
  "msg/GpsGga.msg"
  "msg/GpsRmc.msg"
  "msg/ImuPacket.msg"
  "msg/RawPacket.msg"
  "msg/SensorTiming.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(interface_contract test/test_interface_contract.py)
endif()

ament_package()
