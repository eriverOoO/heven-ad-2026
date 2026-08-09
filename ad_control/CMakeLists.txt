cmake_minimum_required(VERSION 3.8)
project(ad_control)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(ament_cmake REQUIRED)
find_package(ad_morai_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

add_library(ad_control_core
  src/command/command_adapter.cpp
  src/command/curvature_command_adapter.cpp
  src/command/mppi_command_adapter.cpp
  src/lateral/path_tracking_controller_factory.cpp
  src/lateral/stanley.cpp
  src/longitudinal/pid.cpp
  src/path/route_progress_tracker.cpp
  src/path/route_speed_profile.cpp)
add_library(ad_control::ad_control_core ALIAS ad_control_core)
target_compile_features(ad_control_core PUBLIC cxx_std_17)
target_include_directories(ad_control_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(ad_control_core ad_morai_interfaces std_msgs)

install(TARGETS ad_control_core
  EXPORT export_ad_control_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_ad_control_core HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(ad_morai_interfaces std_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  foreach(test_name
      command_adapter
      curvature_command_adapter
      mppi_command_adapter
      path_tracking_controller_factory
      pid
      profile_stanley
      route_speed_profile
      stanley)
    ament_add_gtest(test_${test_name} test/test_${test_name}.cpp)
    target_link_libraries(test_${test_name} ad_control_core)
  endforeach()

endif()

ament_package()
