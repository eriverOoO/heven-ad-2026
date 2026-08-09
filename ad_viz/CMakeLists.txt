cmake_minimum_required(VERSION 3.8)
project(ad_viz)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(ament_cmake REQUIRED)
find_package(ad_interfaces REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(ad_viz_marker_builder
  src/perception/marker_builder.cpp)
target_compile_features(ad_viz_marker_builder PUBLIC cxx_std_17)
target_include_directories(ad_viz_marker_builder PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(ad_viz_marker_builder
  ad_interfaces
  builtin_interfaces
  geometry_msgs
  std_msgs
  unique_identifier_msgs
  visualization_msgs)

add_library(ad_viz_perception_marker_node
  src/perception/perception_marker_node.cpp)
target_compile_features(ad_viz_perception_marker_node PUBLIC cxx_std_17)
target_include_directories(ad_viz_perception_marker_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(
  ad_viz_perception_marker_node ad_viz_marker_builder)
ament_target_dependencies(ad_viz_perception_marker_node
  ad_interfaces
  rclcpp
  std_msgs
  visualization_msgs)

add_executable(ad_perception_marker_node
  src/perception/main.cpp)
target_compile_features(ad_perception_marker_node PUBLIC cxx_std_17)
target_link_libraries(
  ad_perception_marker_node ad_viz_perception_marker_node)
ament_target_dependencies(ad_perception_marker_node rclcpp)

add_library(ad_viz_odometry_projection
  src/localization/odometry_projection.cpp)
target_compile_features(ad_viz_odometry_projection PUBLIC cxx_std_17)
target_include_directories(ad_viz_odometry_projection PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(ad_viz_odometry_projection nav_msgs)

add_library(ad_viz_odometry_ground_node
  src/localization/odometry_ground_node.cpp)
target_compile_features(ad_viz_odometry_ground_node PUBLIC cxx_std_17)
target_include_directories(ad_viz_odometry_ground_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(
  ad_viz_odometry_ground_node ad_viz_odometry_projection)
ament_target_dependencies(ad_viz_odometry_ground_node nav_msgs rclcpp)

add_executable(ad_localization_ground_odometry
  src/localization/main.cpp)
target_compile_features(ad_localization_ground_odometry PUBLIC cxx_std_17)
target_link_libraries(
  ad_localization_ground_odometry ad_viz_odometry_ground_node)
ament_target_dependencies(ad_localization_ground_odometry rclcpp)

add_library(ad_viz_route_elevation_projection
  src/localization/route_elevation_projection.cpp)
target_compile_features(ad_viz_route_elevation_projection PUBLIC cxx_std_17)
target_include_directories(ad_viz_route_elevation_projection PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(ad_viz_route_elevation_projection nav_msgs)

add_library(ad_viz_odometry_route_elevation_node
  src/localization/odometry_route_elevation_node.cpp)
target_compile_features(
  ad_viz_odometry_route_elevation_node PUBLIC cxx_std_17)
target_include_directories(ad_viz_odometry_route_elevation_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(
  ad_viz_odometry_route_elevation_node
  ad_viz_route_elevation_projection)
ament_target_dependencies(
  ad_viz_odometry_route_elevation_node nav_msgs rclcpp)

add_executable(ad_localization_route_elevation_odometry
  src/localization/route_elevation_main.cpp)
target_compile_features(
  ad_localization_route_elevation_odometry PUBLIC cxx_std_17)
target_link_libraries(
  ad_localization_route_elevation_odometry
  ad_viz_odometry_route_elevation_node)
ament_target_dependencies(ad_localization_route_elevation_odometry rclcpp)

add_library(ad_viz_visualization_tf
  src/localization/visualization_tf.cpp)
target_compile_features(ad_viz_visualization_tf PUBLIC cxx_std_17)
target_include_directories(ad_viz_visualization_tf PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(ad_viz_visualization_tf
  geometry_msgs
  nav_msgs
  tf2_msgs)

add_library(ad_viz_visualization_tf_relay_node
  src/localization/visualization_tf_relay_node.cpp)
target_compile_features(
  ad_viz_visualization_tf_relay_node PUBLIC cxx_std_17)
target_include_directories(ad_viz_visualization_tf_relay_node PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(
  ad_viz_visualization_tf_relay_node ad_viz_visualization_tf)
ament_target_dependencies(ad_viz_visualization_tf_relay_node
  nav_msgs
  rclcpp
  tf2_msgs)

add_executable(ad_visualization_tf_relay
  src/localization/visualization_tf_relay_main.cpp)
target_compile_features(ad_visualization_tf_relay PUBLIC cxx_std_17)
target_link_libraries(
  ad_visualization_tf_relay ad_viz_visualization_tf_relay_node)
ament_target_dependencies(ad_visualization_tf_relay rclcpp)

install(TARGETS
  ad_viz_marker_builder
  ad_viz_perception_marker_node
  ad_perception_marker_node
  ad_viz_odometry_projection
  ad_viz_odometry_ground_node
  ad_localization_ground_odometry
  ad_viz_route_elevation_projection
  ad_viz_odometry_route_elevation_node
  ad_localization_route_elevation_odometry
  ad_viz_visualization_tf
  ad_viz_visualization_tf_relay_node
  ad_visualization_tf_relay
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

if(BUILD_TESTING)
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_marker_builder test/test_marker_builder.cpp)
  target_link_libraries(
    test_marker_builder
    ad_viz_marker_builder
    ad_viz_perception_marker_node)
  ament_target_dependencies(
    test_marker_builder
    ad_interfaces
    rclcpp
    visualization_msgs)

  ament_add_gtest(
    test_odometry_projection test/test_odometry_projection.cpp)
  target_link_libraries(
    test_odometry_projection
    ad_viz_odometry_projection
    ad_viz_route_elevation_projection)
  ament_target_dependencies(test_odometry_projection nav_msgs)

  ament_add_gtest(
    test_visualization_tf test/test_visualization_tf.cpp)
  target_link_libraries(
    test_visualization_tf ad_viz_visualization_tf)
  ament_target_dependencies(test_visualization_tf
    geometry_msgs
    nav_msgs
    tf2_msgs)

  ament_add_pytest_test(
    visualization_launch test/test_visualization_launch.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1)
  add_launch_test(
    test/test_perception_marker_runtime.py
    TARGET test_perception_marker_runtime
    TIMEOUT 60
    ENV
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "RCUTILS_LOGGING_BUFFERED_STREAM=1"
      "ROS_DOMAIN_ID=97"
      "ROS_LOCALHOST_ONLY=1")
endif()

ament_export_include_directories(include)
ament_export_libraries(
  ad_viz_marker_builder
  ad_viz_perception_marker_node
  ad_viz_odometry_projection
  ad_viz_odometry_ground_node
  ad_viz_route_elevation_projection
  ad_viz_odometry_route_elevation_node
  ad_viz_visualization_tf
  ad_viz_visualization_tf_relay_node)
ament_export_dependencies(
  ad_interfaces
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  tf2_msgs
  unique_identifier_msgs
  visualization_msgs)
ament_package()
