cmake_minimum_required(VERSION 3.8)
project(ad_lidar_perception)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

option(AD_WITH_AUTOWARE "Build Autoware perception adapters" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ad_interfaces REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

if(AD_WITH_AUTOWARE)
  find_package(autoware_perception_msgs 1.13.0 EXACT REQUIRED)
endif()

add_library(ad_lidar_perception_xyzirt_layout
  src/preprocessing/xyzirt_layout.cpp)
target_compile_features(ad_lidar_perception_xyzirt_layout PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_xyzirt_layout PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(ad_lidar_perception_xyzirt_layout sensor_msgs)

add_library(ad_lidar_perception_motion_deskewer
  src/preprocessing/motion_history.cpp
  src/preprocessing/motion_deskewer.cpp)
target_compile_features(ad_lidar_perception_motion_deskewer PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_motion_deskewer PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(ad_lidar_perception_motion_deskewer
  ad_lidar_perception_xyzirt_layout)
ament_target_dependencies(ad_lidar_perception_motion_deskewer
  builtin_interfaces
  sensor_msgs)

add_library(ad_lidar_perception_point_layout_converter
  src/preprocessing/point_layout_converter.cpp)
target_compile_features(
  ad_lidar_perception_point_layout_converter PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_point_layout_converter PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(ad_lidar_perception_point_layout_converter
  ad_lidar_perception_xyzirt_layout)
ament_target_dependencies(
  ad_lidar_perception_point_layout_converter sensor_msgs)

add_library(ad_lidar_perception_self_crop_filter
  src/preprocessing/self_crop_filter.cpp)
target_compile_features(
  ad_lidar_perception_self_crop_filter PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_self_crop_filter PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(ad_lidar_perception_self_crop_filter
  ad_lidar_perception_xyzirt_layout)
ament_target_dependencies(ad_lidar_perception_self_crop_filter sensor_msgs)

add_library(ad_lidar_perception_gravity_leveler
  src/preprocessing/gravity_leveler.cpp)
target_compile_features(
  ad_lidar_perception_gravity_leveler PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_gravity_leveler PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(ad_lidar_perception_gravity_leveler
  ad_lidar_perception_xyzirt_layout)
ament_target_dependencies(ad_lidar_perception_gravity_leveler sensor_msgs)

add_library(ad_lidar_perception_pointcloud_densifier
  src/preprocessing/pointcloud_densifier.cpp)
target_compile_features(
  ad_lidar_perception_pointcloud_densifier PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_pointcloud_densifier PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(
  ad_lidar_perception_pointcloud_densifier sensor_msgs)

add_library(ad_lidar_perception_finite_point_filter
  src/preprocessing/finite_point_filter.cpp)
target_compile_features(
  ad_lidar_perception_finite_point_filter PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_finite_point_filter PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
ament_target_dependencies(
  ad_lidar_perception_finite_point_filter sensor_msgs)

add_library(ad_lidar_perception_adaptive_cluster
  src/clustering/adaptive_euclidean_cluster.cpp)
target_compile_features(
  ad_lidar_perception_adaptive_cluster PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_adaptive_cluster PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_library(ad_lidar_perception_occupancy_grid
  src/occupancy_grid/grid_builder.cpp)
target_compile_features(ad_lidar_perception_occupancy_grid PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_occupancy_grid PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_library(ad_lidar_perception_dynamic_grid
  src/occupancy_grid/dynamic_grid_builder.cpp)
target_compile_features(ad_lidar_perception_dynamic_grid PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_dynamic_grid PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_library(ad_lidar_perception_grid_combiner
  src/occupancy_grid/grid_combiner.cpp)
target_compile_features(ad_lidar_perception_grid_combiner PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_grid_combiner PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_library(ad_lidar_perception_cv_predictor
  src/tracking/cv_predictor.cpp)
target_compile_features(ad_lidar_perception_cv_predictor PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_cv_predictor PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_library(ad_lidar_perception_imm_predictor
  src/tracking/imm_predictor.cpp)
target_compile_features(ad_lidar_perception_imm_predictor PUBLIC cxx_std_17)
target_include_directories(ad_lidar_perception_imm_predictor PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

if(AD_WITH_AUTOWARE)
  add_executable(ad_adaptive_euclidean_cluster_node
    src/clustering/main.cpp
    src/clustering/adaptive_euclidean_cluster_node.cpp)
  target_compile_features(ad_adaptive_euclidean_cluster_node PUBLIC cxx_std_17)
  target_link_libraries(
    ad_adaptive_euclidean_cluster_node ad_lidar_perception_adaptive_cluster)
  ament_target_dependencies(ad_adaptive_euclidean_cluster_node
    autoware_perception_msgs
    rclcpp
    sensor_msgs)

  add_library(ad_lidar_perception_autoware_prediction
    src/tracking/autoware_prediction_node.cpp)
  target_compile_features(
    ad_lidar_perception_autoware_prediction PUBLIC cxx_std_17)
  target_include_directories(
    ad_lidar_perception_autoware_prediction
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/tracking>"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(
    ad_lidar_perception_autoware_prediction
    ad_lidar_perception_cv_predictor
    ad_lidar_perception_imm_predictor)
  ament_target_dependencies(
    ad_lidar_perception_autoware_prediction
    ad_interfaces
    autoware_perception_msgs
    diagnostic_msgs
    geometry_msgs
    rclcpp)

  add_executable(ad_autoware_prediction_node
    src/tracking/main.cpp)
  target_compile_features(ad_autoware_prediction_node PUBLIC cxx_std_17)
  target_link_libraries(
    ad_autoware_prediction_node
    ad_lidar_perception_autoware_prediction)
  ament_target_dependencies(ad_autoware_prediction_node rclcpp)

  install(TARGETS
    ad_adaptive_euclidean_cluster_node
    ad_lidar_perception_autoware_prediction
    ad_autoware_prediction_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

add_executable(ad_lidar_perception_node
  src/occupancy_grid/main.cpp
  src/occupancy_grid/occupancy_grid_node.cpp)
target_compile_features(ad_lidar_perception_node PUBLIC cxx_std_17)
target_link_libraries(ad_lidar_perception_node ad_lidar_perception_occupancy_grid)
ament_target_dependencies(ad_lidar_perception_node
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_sensor_msgs)

add_executable(ad_dynamic_occupancy_grid_node
  src/occupancy_grid/dynamic_main.cpp
  src/occupancy_grid/dynamic_occupancy_grid_node.cpp)
target_compile_features(
  ad_dynamic_occupancy_grid_node PUBLIC cxx_std_17)
target_link_libraries(
  ad_dynamic_occupancy_grid_node ad_lidar_perception_dynamic_grid)
ament_target_dependencies(ad_dynamic_occupancy_grid_node
  ad_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  tf2
  tf2_geometry_msgs
  tf2_ros)

add_executable(ad_combined_occupancy_grid_node
  src/occupancy_grid/combined_main.cpp
  src/occupancy_grid/combined_occupancy_grid_node.cpp)
target_compile_features(
  ad_combined_occupancy_grid_node PUBLIC cxx_std_17)
target_link_libraries(
  ad_combined_occupancy_grid_node ad_lidar_perception_grid_combiner)
ament_target_dependencies(ad_combined_occupancy_grid_node
  nav_msgs
  rclcpp)

add_executable(ad_point_layout_adapter_node
  src/preprocessing/main.cpp
  src/preprocessing/point_layout_adapter_node.cpp)
target_compile_features(ad_point_layout_adapter_node PUBLIC cxx_std_17)
target_link_libraries(ad_point_layout_adapter_node
  ad_lidar_perception_point_layout_converter)
ament_target_dependencies(ad_point_layout_adapter_node
  rclcpp
  sensor_msgs)

add_executable(ad_self_crop_filter_node
  src/preprocessing/self_crop_filter_main.cpp
  src/preprocessing/self_crop_filter_node.cpp)
target_compile_features(ad_self_crop_filter_node PUBLIC cxx_std_17)
target_link_libraries(ad_self_crop_filter_node
  ad_lidar_perception_self_crop_filter)
ament_target_dependencies(ad_self_crop_filter_node
  rclcpp
  sensor_msgs
  tf2
  tf2_ros)

add_executable(ad_gravity_leveler_node
  src/preprocessing/gravity_leveler_main.cpp
  src/preprocessing/gravity_leveler_node.cpp)
target_compile_features(ad_gravity_leveler_node PUBLIC cxx_std_17)
target_link_libraries(ad_gravity_leveler_node
  ad_lidar_perception_gravity_leveler)
ament_target_dependencies(ad_gravity_leveler_node
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_ros)

add_executable(ad_finite_point_filter_node
  src/preprocessing/finite_point_filter_main.cpp
  src/preprocessing/finite_point_filter_node.cpp)
target_compile_features(ad_finite_point_filter_node PUBLIC cxx_std_17)
target_link_libraries(ad_finite_point_filter_node
  ad_lidar_perception_finite_point_filter)
ament_target_dependencies(ad_finite_point_filter_node
  rclcpp
  sensor_msgs)

add_executable(ad_pointcloud_densifier_node
  src/preprocessing/pointcloud_densifier_main.cpp
  src/preprocessing/pointcloud_densifier_node.cpp)
target_compile_features(ad_pointcloud_densifier_node PUBLIC cxx_std_17)
target_link_libraries(ad_pointcloud_densifier_node
  ad_lidar_perception_pointcloud_densifier)
ament_target_dependencies(ad_pointcloud_densifier_node
  builtin_interfaces
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_ros)

add_executable(ad_motion_deskew_node
  src/preprocessing/motion_deskew_main.cpp
  src/preprocessing/motion_deskew_node.cpp)
target_compile_features(ad_motion_deskew_node PUBLIC cxx_std_17)
target_link_libraries(ad_motion_deskew_node
  ad_lidar_perception_motion_deskewer)
ament_target_dependencies(ad_motion_deskew_node
  builtin_interfaces
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_ros)

install(TARGETS
  ad_lidar_perception_xyzirt_layout
  ad_lidar_perception_motion_deskewer
  ad_lidar_perception_point_layout_converter
  ad_lidar_perception_self_crop_filter
  ad_lidar_perception_gravity_leveler
  ad_lidar_perception_pointcloud_densifier
  ad_lidar_perception_finite_point_filter
  ad_lidar_perception_adaptive_cluster
  ad_lidar_perception_occupancy_grid
  ad_lidar_perception_dynamic_grid
  ad_lidar_perception_grid_combiner
  ad_lidar_perception_cv_predictor
  ad_lidar_perception_imm_predictor
  ad_lidar_perception_node
  ad_dynamic_occupancy_grid_node
  ad_combined_occupancy_grid_node
  ad_point_layout_adapter_node
  ad_self_crop_filter_node
  ad_gravity_leveler_node
  ad_finite_point_filter_node
  ad_pointcloud_densifier_node
  ad_motion_deskew_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  ../scripts/check_autoware_perception.py
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  ad_lidar_perception/centerpoint_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ad_centerpoint_detector)
install(PROGRAMS
  ad_lidar_perception/publish_morai_frames.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ad_publish_morai_frames)
install(PROGRAMS
  ad_lidar_perception/record_detected_objects.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ad_record_detected_objects)
install(PROGRAMS
  ad_lidar_perception/ab3dmot_tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ad_ab3dmot_tracker)
install(DIRECTORY
  ../tools/centerpoint_offline/
  DESTINATION share/${PROJECT_NAME}/centerpoint_offline
  PATTERN "__pycache__" EXCLUDE
  PATTERN "test_*.py" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_grid_builder test/test_grid_builder.cpp)
  target_link_libraries(test_grid_builder ad_lidar_perception_occupancy_grid)
  ament_add_gtest(
    test_exact_stamp_pairer test/test_exact_stamp_pairer.cpp)
  target_include_directories(test_exact_stamp_pairer PRIVATE include)
  ament_add_gtest(
    test_dynamic_grid_builder test/test_dynamic_grid_builder.cpp)
  target_link_libraries(
    test_dynamic_grid_builder ad_lidar_perception_dynamic_grid)
  ament_add_gtest(test_grid_combiner test/test_grid_combiner.cpp)
  target_link_libraries(test_grid_combiner ad_lidar_perception_grid_combiner)
  ament_add_gtest(
    test_point_layout_converter test/test_point_layout_converter.cpp)
  target_link_libraries(
    test_point_layout_converter ad_lidar_perception_point_layout_converter)
  ament_add_gtest(
    test_motion_deskewer test/test_motion_deskewer.cpp)
  target_link_libraries(
    test_motion_deskewer ad_lidar_perception_motion_deskewer)
  ament_add_gtest(
    test_self_crop_filter test/test_self_crop_filter.cpp)
  target_link_libraries(
    test_self_crop_filter ad_lidar_perception_self_crop_filter)
  ament_add_gtest(
    test_gravity_leveler test/test_gravity_leveler.cpp)
  target_link_libraries(
    test_gravity_leveler ad_lidar_perception_gravity_leveler)
  ament_add_gtest(
    test_pointcloud_densifier test/test_pointcloud_densifier.cpp)
  target_link_libraries(
    test_pointcloud_densifier ad_lidar_perception_pointcloud_densifier)
  ament_add_gtest(
    test_finite_point_filter test/test_finite_point_filter.cpp)
  target_link_libraries(
    test_finite_point_filter ad_lidar_perception_finite_point_filter)
  ament_add_gtest(test_cv_predictor test/test_cv_predictor.cpp)
  target_link_libraries(test_cv_predictor ad_lidar_perception_cv_predictor)
  ament_add_gtest(
    test_adaptive_euclidean_cluster
    test/test_adaptive_euclidean_cluster.cpp)
  target_link_libraries(
    test_adaptive_euclidean_cluster
    ad_lidar_perception_adaptive_cluster)
  ament_add_gtest(test_imm_predictor test/test_imm_predictor.cpp)
  target_link_libraries(test_imm_predictor ad_lidar_perception_imm_predictor)
  if(AD_WITH_AUTOWARE)
    ament_add_gtest(
      test_autoware_prediction_adapter
      test/test_autoware_prediction_adapter.cpp)
    target_include_directories(test_autoware_prediction_adapter PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/tracking")
    target_link_libraries(
      test_autoware_prediction_adapter
      ad_lidar_perception_autoware_prediction)
    ament_target_dependencies(
      test_autoware_prediction_adapter
      ad_interfaces
      autoware_perception_msgs
      diagnostic_msgs
      rclcpp)
  endif()
  ament_add_pytest_test(
    test_lidar_perception_launch test/test_lidar_perception_launch.py)
  ament_add_pytest_test(
    test_centerpoint_ros test/test_centerpoint_ros.py)
  ament_add_pytest_test(
    test_morai_replay test/test_morai_replay.py)
  ament_add_pytest_test(
    test_detection_recording test/test_detection_recording.py)
  ament_add_pytest_test(
    test_ab3dmot_geometry test/test_ab3dmot_geometry.py)
  ament_add_pytest_test(
    test_ab3dmot_core test/test_ab3dmot_core.py)
  ament_add_pytest_test(
    test_ab3dmot_ros test/test_ab3dmot_ros.py)
  ament_add_pytest_test(
    test_ab3dmot_tracker_node test/test_ab3dmot_tracker_node.py)
  ament_add_pytest_test(
    test_ab3dmot_association test/test_ab3dmot_association.py)
  ament_add_pytest_test(
    test_ab3dmot_association_metrics test/test_ab3dmot_association_metrics.py)
  ament_add_pytest_test(
    test_ab3dmot_hybrid_gate test/test_ab3dmot_hybrid_gate.py)
  ament_add_pytest_test(
    test_ab3dmot_ekf test/test_ab3dmot_ekf.py)
  ament_add_pytest_test(
    test_ab3dmot_heading test/test_ab3dmot_heading.py)
  ament_add_pytest_test(
    test_ab3dmot_imm test/test_ab3dmot_imm.py)
  ament_add_pytest_test(
    test_lidar_bag_replay_launch test/test_lidar_bag_replay_launch.py)
  ament_add_pytest_test(
    test_ground_segmentation_launch test/test_ground_segmentation_launch.py)
  ament_add_pytest_test(
    test_preprocessing_launch test/test_preprocessing_launch.py)
  ament_add_pytest_test(
    test_selection_config test/test_selection_config.py)
  ament_add_pytest_test(
    test_object_detection_launch test/test_object_detection_launch.py)
  ament_add_pytest_test(
    test_tracking_launch test/test_tracking_launch.py)
  ament_add_pytest_test(
    test_perception_visualization_launch
    test/test_perception_visualization_launch.py)
  ament_add_pytest_test(
    test_check_autoware_perception
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/tests/test_check_autoware_perception.py"
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")
  ament_add_pytest_test(
    test_autoware_pipeline_integration
    test/test_autoware_pipeline_integration.py
    TIMEOUT 300
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ENV
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "ROS_DOMAIN_ID=197"
      "ROS_LOCALHOST_ONLY=1"
      "AD_AUTOWARE_INTEGRATION_METRICS_FILE=${CMAKE_CURRENT_BINARY_DIR}/autoware_pipeline_metrics.json")
  add_launch_test(
    test/test_occupancy_layer_launch.py
    TARGET test_occupancy_layer_launch
    TIMEOUT 120
    ENV
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "RCUTILS_LOGGING_BUFFERED_STREAM=1"
      "ROS_DOMAIN_ID=93"
      "ROS_LOCALHOST_ONLY=1")
  add_launch_test(
    test/test_self_crop_runtime.py
    TARGET test_self_crop_runtime
    TIMEOUT 60
    ENV
      "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      "RCUTILS_LOGGING_BUFFERED_STREAM=1"
      "ROS_DOMAIN_ID=96"
      "ROS_LOCALHOST_ONLY=1")
  if(AD_WITH_AUTOWARE)
    add_launch_test(
      test/test_adaptive_cluster_runtime.py
      TARGET test_adaptive_cluster_runtime
      TIMEOUT 60
      ENV
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
        "RCUTILS_LOGGING_BUFFERED_STREAM=1"
        "ROS_DOMAIN_ID=95"
        "ROS_LOCALHOST_ONLY=1")
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(
  ad_lidar_perception_motion_deskewer
  ad_lidar_perception_point_layout_converter
  ad_lidar_perception_self_crop_filter
  ad_lidar_perception_gravity_leveler
  ad_lidar_perception_pointcloud_densifier
  ad_lidar_perception_xyzirt_layout
  ad_lidar_perception_finite_point_filter
  ad_lidar_perception_adaptive_cluster
  ad_lidar_perception_occupancy_grid
  ad_lidar_perception_dynamic_grid
  ad_lidar_perception_grid_combiner
  ad_lidar_perception_cv_predictor
  ad_lidar_perception_imm_predictor)
ament_export_dependencies(
  ad_interfaces
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_sensor_msgs)
ament_package()
