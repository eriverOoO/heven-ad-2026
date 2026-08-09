cmake_minimum_required(VERSION 3.8)
project(ad_localization)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(PkgConfig REQUIRED)
pkg_check_modules(PROJ REQUIRED IMPORTED_TARGET proj)
find_package(ament_cmake REQUIRED)
find_package(ad_morai_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(localization_adapter src/adapter/localization_adapter.cpp)
target_compile_features(localization_adapter PUBLIC cxx_std_17)
target_include_directories(localization_adapter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(localization_adapter PkgConfig::PROJ)
ament_target_dependencies(localization_adapter
  ad_morai_interfaces
  geometry_msgs
  nav_msgs
  sensor_msgs)

add_library(gnss_imu_localizer src/gnss_imu/gnss_imu_localizer.cpp)
target_compile_features(gnss_imu_localizer PUBLIC cxx_std_17)
target_include_directories(gnss_imu_localizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(gnss_imu_localizer
  geometry_msgs
  nav_msgs
  sensor_msgs)

add_library(localization_handoff src/gnss_shadow/localization_handoff.cpp)
target_compile_features(localization_handoff PUBLIC cxx_std_17)
target_include_directories(localization_handoff PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(localization_handoff
  geometry_msgs
  nav_msgs)

add_library(localization_manager src/manager/localization_manager.cpp)
target_compile_features(localization_manager PUBLIC cxx_std_17)
target_include_directories(localization_manager PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(localization_manager
  geometry_msgs
  nav_msgs)

add_library(imu_quaternion_encoder
  src/imu_quaternion_encoder/imu_quaternion_encoder.cpp)
target_compile_features(imu_quaternion_encoder PUBLIC cxx_std_17)
target_include_directories(imu_quaternion_encoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(imu_quaternion_encoder
  ad_morai_interfaces
  geometry_msgs
  nav_msgs
  sensor_msgs)

add_library(quaternion_wheel_gnss_ekf
  src/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf.cpp)
target_compile_features(quaternion_wheel_gnss_ekf PUBLIC cxx_std_17)
target_include_directories(quaternion_wheel_gnss_ekf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(quaternion_wheel_gnss_ekf
  geometry_msgs
  nav_msgs
  sensor_msgs)

add_library(quaternion_wheel_gnss_ekf_node_lib
  src/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf_node.cpp)
target_compile_features(quaternion_wheel_gnss_ekf_node_lib PUBLIC cxx_std_17)
target_include_directories(quaternion_wheel_gnss_ekf_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(quaternion_wheel_gnss_ekf_node_lib
  quaternion_wheel_gnss_ekf)
ament_target_dependencies(quaternion_wheel_gnss_ekf_node_lib
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_srvs)

add_executable(ad_localization_node
  src/adapter/localization_node.cpp
  src/adapter/main.cpp)
target_compile_features(ad_localization_node PRIVATE cxx_std_17)
target_link_libraries(ad_localization_node localization_adapter)
ament_target_dependencies(ad_localization_node
  ad_morai_interfaces
  diagnostic_msgs
  geometry_msgs
  lifecycle_msgs
  nav_msgs
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  std_srvs
  tf2
  tf2_ros)

add_executable(gnss_imu_localization_node
  src/gnss_imu/gnss_imu_localization_node.cpp
  src/gnss_imu/gnss_imu_main.cpp)
target_compile_features(gnss_imu_localization_node PRIVATE cxx_std_17)
target_link_libraries(gnss_imu_localization_node gnss_imu_localizer)
ament_target_dependencies(gnss_imu_localization_node
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_srvs
  tf2_ros)

add_executable(localization_handoff_node
  src/gnss_shadow/localization_handoff_node.cpp
  src/gnss_shadow/localization_handoff_main.cpp)
target_compile_features(localization_handoff_node PRIVATE cxx_std_17)
target_link_libraries(localization_handoff_node localization_handoff)
ament_target_dependencies(localization_handoff_node
  diagnostic_msgs
  geometry_msgs
  nav_msgs
  rclcpp
  tf2_ros)

add_executable(localization_manager_node
  src/manager/localization_manager_node.cpp
  src/manager/localization_manager_main.cpp)
target_compile_features(localization_manager_node PRIVATE cxx_std_17)
target_link_libraries(localization_manager_node localization_manager)
ament_target_dependencies(localization_manager_node
  nav_msgs
  rclcpp
  std_srvs
  tf2_ros)

add_executable(imu_quaternion_encoder_node
  src/imu_quaternion_encoder/imu_quaternion_encoder_node.cpp
  src/imu_quaternion_encoder/imu_quaternion_encoder_main.cpp)
target_compile_features(imu_quaternion_encoder_node PRIVATE cxx_std_17)
target_link_libraries(imu_quaternion_encoder_node imu_quaternion_encoder)
ament_target_dependencies(imu_quaternion_encoder_node
  ad_morai_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_srvs)

add_executable(quaternion_wheel_gnss_ekf_node
  src/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf_main.cpp)
target_compile_features(quaternion_wheel_gnss_ekf_node PRIVATE cxx_std_17)
target_link_libraries(quaternion_wheel_gnss_ekf_node
  quaternion_wheel_gnss_ekf_node_lib)
ament_target_dependencies(quaternion_wheel_gnss_ekf_node
  rclcpp)

install(TARGETS
  localization_adapter
  gnss_imu_localizer
  localization_handoff
  localization_manager
  imu_quaternion_encoder
  quaternion_wheel_gnss_ekf
  quaternion_wheel_gnss_ekf_node_lib
  ad_localization_node
  gnss_imu_localization_node
  localization_handoff_node
  localization_manager_node
  imu_quaternion_encoder_node
  quaternion_wheel_gnss_ekf_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch maps DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(kalman_filter_localization_core REQUIRED)
  ament_add_gtest(test_localization_adapter test/test_localization_adapter.cpp)
  ament_target_dependencies(test_localization_adapter
    ad_morai_interfaces
    geometry_msgs
    nav_msgs
    sensor_msgs)
  target_include_directories(test_localization_adapter PRIVATE include)
  target_link_libraries(test_localization_adapter localization_adapter)
  ament_add_gtest(test_gnss_imu_localizer test/test_gnss_imu_localizer.cpp)
  ament_target_dependencies(test_gnss_imu_localizer
    geometry_msgs
    nav_msgs
    sensor_msgs)
  target_include_directories(test_gnss_imu_localizer PRIVATE include)
  target_link_libraries(test_gnss_imu_localizer gnss_imu_localizer)
  ament_add_gtest(test_localization_handoff test/test_localization_handoff.cpp)
  ament_target_dependencies(test_localization_handoff
    geometry_msgs
    nav_msgs)
  target_include_directories(test_localization_handoff PRIVATE include)
  target_link_libraries(test_localization_handoff localization_handoff)
  ament_add_gtest(test_localization_manager test/test_localization_manager.cpp)
  ament_target_dependencies(test_localization_manager
    geometry_msgs
    nav_msgs)
  target_include_directories(test_localization_manager PRIVATE include)
  target_link_libraries(test_localization_manager localization_manager)
  ament_add_gtest(test_imu_quaternion_encoder
    test/test_imu_quaternion_encoder.cpp)
  ament_target_dependencies(test_imu_quaternion_encoder
    ad_morai_interfaces
    geometry_msgs
    nav_msgs
    sensor_msgs)
  target_include_directories(test_imu_quaternion_encoder PRIVATE include)
  target_link_libraries(test_imu_quaternion_encoder imu_quaternion_encoder)
  ament_add_gtest(test_quaternion_wheel_gnss_ekf
    test/test_quaternion_wheel_gnss_ekf.cpp)
  ament_target_dependencies(test_quaternion_wheel_gnss_ekf
    geometry_msgs
    nav_msgs
    sensor_msgs)
  target_include_directories(test_quaternion_wheel_gnss_ekf PRIVATE include)
  target_link_libraries(test_quaternion_wheel_gnss_ekf
    quaternion_wheel_gnss_ekf)
  ament_add_gtest(test_quaternion_wheel_gnss_ekf_node
    test/test_quaternion_wheel_gnss_ekf_node.cpp)
  ament_target_dependencies(test_quaternion_wheel_gnss_ekf_node
    geometry_msgs
    nav_msgs
    rclcpp
    sensor_msgs
    std_srvs)
  target_include_directories(test_quaternion_wheel_gnss_ekf_node PRIVATE include)
  target_link_libraries(test_quaternion_wheel_gnss_ekf_node
    quaternion_wheel_gnss_ekf_node_lib)
  ament_add_gtest(test_eskf_future_measurement
    test/test_eskf_future_measurement.cpp)
  target_link_libraries(test_eskf_future_measurement
    kalman_filter_localization_core::core)
  ament_add_pytest_test(localization_contract test/test_localization_contract.py
    TIMEOUT 60)
  ament_add_pytest_test(gnss_imu_profile test/test_gnss_imu_profile.py
    TIMEOUT 60)
  ament_add_pytest_test(eskf_profile test/test_eskf_profile.py
    TIMEOUT 60)
  ament_add_pytest_test(quaternion_wheel_gnss_ekf_profile
    test/test_quaternion_wheel_gnss_ekf_profile.py
    TIMEOUT 60)
  ament_add_pytest_test(source_layout test/test_source_layout.py
    TIMEOUT 60)
  ament_add_pytest_test(fastlio_profile test/test_fastlio_profile.py
    TIMEOUT 60)
  ament_add_pytest_test(hybrid_localization_profile
    test/test_hybrid_localization_profile.py
    TIMEOUT 60)
  ament_add_pytest_test(eskf_integration test/test_eskf_integration.py
    TIMEOUT 90)
endif()

ament_package()
