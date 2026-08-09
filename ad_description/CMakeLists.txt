cmake_minimum_required(VERSION 3.8)
project(ad_description)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DefaultBuildType.cmake")

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes urdf DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(description_contract test/test_description.py)
endif()

ament_package()
