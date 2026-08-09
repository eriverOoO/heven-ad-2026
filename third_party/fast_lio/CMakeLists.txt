cmake_minimum_required(VERSION 3.16)
project(fast_lio)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenMP REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "msg/Pose6D.msg")

add_library(fast_lio_core STATIC
  src/fastlio_node.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp)
target_compile_features(fast_lio_core PUBLIC cxx_std_17)
# Keep scan-to-map correspondence search parallel as in the proven 2025
# tunnel build.  Linking OpenMP alone does not enable the guarded loop in
# fastlio_node.cpp; these definitions are the upstream FAST-LIO switch.
target_compile_definitions(fast_lio_core PRIVATE MP_EN MP_PROC_NUM=3)
target_include_directories(fast_lio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_link_libraries(fast_lio_core Eigen3::Eigen ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
ament_target_dependencies(fast_lio_core
  rclcpp sensor_msgs geometry_msgs nav_msgs diagnostic_msgs std_srvs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions)
rosidl_target_interfaces(fast_lio_core ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(fastlio_mapping src/main.cpp)
target_link_libraries(fastlio_mapping fast_lio_core)
ament_target_dependencies(fastlio_mapping rclcpp)

install(TARGETS fast_lio_core fastlio_mapping DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/fast_lio include/IKFoM_toolkit include/ikd-Tree
  DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES LICENSE UPSTREAM.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_mode_policy test/test_mode_policy.cpp)
  target_link_libraries(test_mode_policy fast_lio_core)
  ament_add_gtest(test_pose_utils test/test_pose_utils.cpp)
  target_link_libraries(test_pose_utils fast_lio_core)
  ament_add_gtest(test_localization_health test/test_localization_health.cpp)
  target_link_libraries(test_localization_health fast_lio_core)
  ament_add_gtest(test_point_validation test/test_point_validation.cpp)
  target_link_libraries(test_point_validation fast_lio_core)
  ament_add_gtest(test_ikdtree_snapshot test/test_ikdtree_snapshot.cpp)
  target_link_libraries(test_ikdtree_snapshot fast_lio_core)
  ament_add_gtest(test_wheel_velocity_update test/test_wheel_velocity_update.cpp)
  target_link_libraries(test_wheel_velocity_update fast_lio_core)
endif()

ament_package()
