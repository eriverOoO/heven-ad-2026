# Keep local and CI builds fast enough for real-time ROS nodes while preserving
# symbols for profiling and post-mortem debugging. Callers may still select any
# build type explicitly, for example with -DCMAKE_BUILD_TYPE=Debug.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the CMake build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()
