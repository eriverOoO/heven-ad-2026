cmake_minimum_required(VERSION 3.8)
project(ad_planner)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

option(
  AD_PLANNER_ENABLE_NAV2_MPPI
  "Build the optional Nav2 MPPI FollowPath action proxy"
  OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ad_control REQUIRED)
find_package(ad_interfaces REQUIRED)
find_package(ad_morai_interfaces REQUIRED)
find_package(behaviortree_cpp_v3 3.8.7 EXACT REQUIRED CONFIG)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

if(NOT TARGET behaviortree_cpp_v3::behaviortree_cpp_v3)
  message(FATAL_ERROR
    "behaviortree_cpp_v3 3.8.7 was found, but required imported target "
    "behaviortree_cpp_v3::behaviortree_cpp_v3 is unavailable")
endif()

add_library(ad_planner_core
  src/io/data_loader.cpp
  src/io/route_corridor_loader.cpp
  src/local_planning/common/future_road_risk.cpp
  src/local_planning/common/local_motion_frame.cpp
  src/local_planning/common/local_motion_runtime.cpp
  src/local_planning/common/local_motion_timing.cpp
  src/local_planning/common/local_motion_validation.cpp
  src/local_planning/common/occupancy.cpp
  src/local_planning/common/occupancy_grid_reprojector.cpp
  src/local_planning/common/prediction_admission.cpp
  src/local_planning/common/road_corridor_grid.cpp
  src/local_planning/dwa/dwa.cpp
  src/local_planning/dwa/dwa_backend.cpp
  src/local_planning/frenet/frenet_geometry.cpp
  src/local_planning/frenet/frenet_lattice.cpp
  src/local_planning/local_motion_factory.cpp
  src/local_planning/mppi_nav2/mppi_follow_path_state.cpp
  src/local_planning/mppi_nav2/mppi_nav2_backend.cpp
  src/visualization/path_tracking_markers.cpp
  src/visualization/planner_visualization.cpp
  src/visualization/route_markers.cpp)
target_compile_features(ad_planner_core PUBLIC cxx_std_17)
target_include_directories(ad_planner_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ad_planner_core PUBLIC ad_control::ad_control_core yaml-cpp)
ament_target_dependencies(
  ad_planner_core PUBLIC
  builtin_interfaces geometry_msgs nav_msgs rclcpp std_msgs
  visualization_msgs yaml_cpp_vendor)

add_library(ad_planner_bt
  src/behavior/bt_nodes.cpp
  src/behavior/planner_supervisor.cpp)
target_compile_features(ad_planner_bt PUBLIC cxx_std_17)
target_include_directories(ad_planner_bt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ad_planner_bt PUBLIC
  ad_planner_core
  behaviortree_cpp_v3::behaviortree_cpp_v3)

add_executable(ad_planner_node
  src/planner/main.cpp
  src/planner/planner_node.cpp
  src/planner/planner_ros_interfaces.cpp)
target_compile_features(ad_planner_node PUBLIC cxx_std_17)
target_link_libraries(ad_planner_node ad_planner_bt)
ament_target_dependencies(ad_planner_node
  ament_index_cpp
  ad_interfaces
  ad_morai_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  std_srvs
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs)

add_executable(ad_occupancy_grid_reprojector_node
  src/local_planning/common/occupancy_grid_reprojector_main.cpp
  src/local_planning/common/occupancy_grid_reprojector_node.cpp)
target_compile_features(
  ad_occupancy_grid_reprojector_node PUBLIC cxx_std_17)
target_include_directories(
  ad_occupancy_grid_reprojector_node PRIVATE
  include
  src/local_planning/common)
target_link_libraries(
  ad_occupancy_grid_reprojector_node ad_planner_core)
ament_target_dependencies(
  ad_occupancy_grid_reprojector_node
  geometry_msgs
  nav_msgs
  rclcpp
  tf2_ros)

add_executable(ad_road_corridor_mask_node
  src/local_planning/common/road_corridor_mask_main.cpp
  src/local_planning/common/road_corridor_mask_node.cpp)
target_compile_features(
  ad_road_corridor_mask_node PUBLIC cxx_std_17)
target_include_directories(
  ad_road_corridor_mask_node PRIVATE include)
target_link_libraries(
  ad_road_corridor_mask_node ad_planner_core)
ament_target_dependencies(
  ad_road_corridor_mask_node
  ad_interfaces
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros)

if(AD_PLANNER_ENABLE_NAV2_MPPI)
  find_package(nav2_msgs REQUIRED)
  find_package(rclcpp_action REQUIRED)

  add_executable(ad_planner_mppi_follow_path_proxy
    src/local_planning/mppi_nav2/mppi_follow_path_proxy.cpp
    src/local_planning/mppi_nav2/mppi_follow_path_proxy_main.cpp)
  target_compile_features(
    ad_planner_mppi_follow_path_proxy PUBLIC cxx_std_17)
  target_include_directories(
    ad_planner_mppi_follow_path_proxy PRIVATE include)
  target_link_libraries(
    ad_planner_mppi_follow_path_proxy ad_planner_core)
  ament_target_dependencies(
    ad_planner_mppi_follow_path_proxy
    nav2_msgs
    rclcpp
    rclcpp_action
    nav_msgs)
  install(
    TARGETS ad_planner_mppi_follow_path_proxy
    DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS ad_planner_core ad_planner_bt
  EXPORT export_ad_planner_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  TARGETS
    ad_planner_node
    ad_occupancy_grid_reprojector_node
    ad_road_corridor_mask_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY behavior_trees/ DESTINATION share/${PROJECT_NAME}/behavior_trees)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
ament_export_targets(export_ad_planner_core HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  ad_control
  behaviortree_cpp_v3
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  visualization_msgs
  yaml_cpp_vendor)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
  foreach(fixture path_csv path_space)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/${fixture}.txt"
      "${CMAKE_CURRENT_BINARY_DIR}/fixtures/${fixture}.txt"
      COPYONLY)
  endforeach()

  ament_add_gtest(test_data_loader test/test_data_loader.cpp)
  target_include_directories(test_data_loader PRIVATE include)
  target_link_libraries(test_data_loader ad_planner_core)

  ament_add_gtest(test_exact_stamp_pairer test/test_exact_stamp_pairer.cpp)
  target_include_directories(test_exact_stamp_pairer PRIVATE include)

  ament_add_gtest(test_route_corridor_loader test/test_route_corridor_loader.cpp)
  target_include_directories(test_route_corridor_loader PRIVATE include)
  target_link_libraries(test_route_corridor_loader ad_planner_core)

  ament_add_gtest(
    test_path_tracking_markers test/test_path_tracking_markers.cpp)
  target_include_directories(test_path_tracking_markers PRIVATE include)
  target_link_libraries(test_path_tracking_markers ad_planner_core)

  ament_add_gtest(test_route_markers test/test_route_markers.cpp)
  target_include_directories(test_route_markers PRIVATE include)
  target_link_libraries(test_route_markers ad_planner_core)

  foreach(controller occupancy dwa gear parameter_validation)
    ament_add_gtest(test_${controller} test/test_${controller}.cpp)
    target_include_directories(test_${controller} PRIVATE include)
    target_link_libraries(test_${controller} ad_planner_core)
  endforeach()

  ament_add_gtest(
    test_occupancy_grid_reprojector
    test/test_occupancy_grid_reprojector.cpp)
  target_include_directories(test_occupancy_grid_reprojector PRIVATE include)
  target_link_libraries(test_occupancy_grid_reprojector ad_planner_core)

  ament_add_gtest(test_local_motion_factory test/test_local_motion_factory.cpp)
  target_include_directories(test_local_motion_factory PRIVATE include)
  target_link_libraries(test_local_motion_factory ad_planner_core)

  foreach(local_motion_test local_motion_contract dwa_backend)
    ament_add_gtest(test_${local_motion_test} test/test_${local_motion_test}.cpp)
    target_include_directories(test_${local_motion_test} PRIVATE include)
    target_link_libraries(test_${local_motion_test} ad_planner_core)
  endforeach()

  ament_add_gtest(test_mppi_nav2_backend test/test_mppi_nav2_backend.cpp)
  target_include_directories(test_mppi_nav2_backend PRIVATE include)
  target_link_libraries(test_mppi_nav2_backend ad_planner_core)

  ament_add_gtest(
    test_mppi_follow_path_state
    test/test_mppi_follow_path_state.cpp)
  target_include_directories(test_mppi_follow_path_state PRIVATE include)
  target_link_libraries(test_mppi_follow_path_state ad_planner_core)
  ament_target_dependencies(test_mppi_follow_path_state nav_msgs)

  ament_add_gtest(test_frenet_geometry test/test_frenet_geometry.cpp)
  target_include_directories(test_frenet_geometry PRIVATE include)
  target_link_libraries(test_frenet_geometry ad_planner_core)

  ament_add_gtest(
    test_road_corridor_grid test/test_road_corridor_grid.cpp)
  target_include_directories(test_road_corridor_grid PRIVATE include)
  target_link_libraries(test_road_corridor_grid ad_planner_core)

  ament_add_gtest(test_frenet_lattice test/test_frenet_lattice.cpp)
  target_include_directories(test_frenet_lattice PRIVATE include)
  target_link_libraries(test_frenet_lattice ad_planner_core)

  ament_add_gtest(test_local_motion_frame test/test_local_motion_frame.cpp)
  target_include_directories(test_local_motion_frame PRIVATE include)
  target_link_libraries(test_local_motion_frame ad_planner_core)

  ament_add_gtest(test_local_motion_timing test/test_local_motion_timing.cpp)
  target_include_directories(test_local_motion_timing PRIVATE include)
  target_link_libraries(test_local_motion_timing ad_planner_core)

  ament_add_gtest(
    test_prediction_admission test/test_prediction_admission.cpp)
  target_include_directories(test_prediction_admission PRIVATE include)
  target_link_libraries(test_prediction_admission ad_planner_core)

  ament_add_gtest(
    test_future_road_risk test/test_future_road_risk.cpp)
  target_include_directories(test_future_road_risk PRIVATE include)
  target_link_libraries(test_future_road_risk ad_planner_core)

  ament_add_gtest(
    test_local_motion_validation test/test_local_motion_validation.cpp)
  target_include_directories(test_local_motion_validation PRIVATE include)
  target_link_libraries(test_local_motion_validation ad_planner_core)

  ament_add_gtest(
    test_local_motion_runtime test/test_local_motion_runtime.cpp)
  target_include_directories(test_local_motion_runtime PRIVATE include)
  target_link_libraries(test_local_motion_runtime ad_planner_core)

  ament_add_gtest(
    test_planner_visualization test/test_planner_visualization.cpp)
  target_include_directories(test_planner_visualization PRIVATE include)
  target_link_libraries(test_planner_visualization ad_planner_core)

  ament_add_gtest(test_behavior_tree test/test_behavior_tree.cpp)
  target_include_directories(test_behavior_tree PRIVATE include)
  target_compile_definitions(test_behavior_tree PRIVATE
    AD_PLANNER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(test_behavior_tree
    ad_planner_bt)

  ament_add_pytest_test(test_planner_launch test/test_planner_launch.py)
  ament_add_pytest_test(
    test_road_corridor_mask_launch
    test/test_road_corridor_mask_launch.py)
  ament_add_pytest_test(
    test_mppi_nav2_launch
    test/test_mppi_nav2_launch.py)
  ament_add_pytest_test(
    test_frenet_runtime_contract
    test/test_frenet_runtime_contract.py
    TIMEOUT 240
    ENV
      "RCUTILS_LOGGING_BUFFERED_STREAM=1"
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "ROS_LOCALHOST_ONLY=1")
  ament_add_pytest_test(
    test_occupancy_grid_reprojector_runtime
    test/test_occupancy_grid_reprojector_runtime.py
    TIMEOUT 180
    ENV
      "RCUTILS_LOGGING_BUFFERED_STREAM=1"
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "ROS_LOCALHOST_ONLY=1")
endif()

ament_package(
  CONFIG_EXTRAS_POST cmake/ad_planner_dependency_targets.cmake)
