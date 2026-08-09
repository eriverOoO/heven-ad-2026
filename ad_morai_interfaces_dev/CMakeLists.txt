cmake_minimum_required(VERSION 3.8)
project(ad_morai_interfaces_dev)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraBoundingBox.msg"
  "msg/CameraBoundingBoxArray.msg"
  "msg/EgoVehicleStatus.msg"
  "msg/EgoGhostCmd.msg"
  "msg/IntersectionControl.msg"
  "msg/Lamps.msg"
  "msg/Lidar2D.msg"
  "msg/MultiEgoSetting.msg"
  "msg/MultiEgoVehicle.msg"
  "msg/NpcGhostCmd.msg"
  "msg/NpcGhostInfo.msg"
  "msg/ObjectStatus.msg"
  "msg/ObjectStatusArray.msg"
  "msg/TrafficLightStatus.msg"
  "msg/VehicleCollision.msg"
  "msg/VehicleCollisionArray.msg"
  "msg/VehicleCollisionObject.msg"
  "msg/IntersectionStatus.msg"
  "msg/SimulatorCommand.msg"
  "msg/SaveSensorData.msg"
  "msg/ScenarioLoad.msg"
  "msg/SensorPosControl.msg"
  "msg/TrafficLightControl.msg"
  "srv/MoraiGrpcCall.srv"
  "srv/MoraiSyncMode.srv"
  "srv/MoraiStep.srv"
  DEPENDENCIES geometry_msgs std_msgs)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(interface_contract test/test_dev_interface_contract.py)
endif()

ament_package()
