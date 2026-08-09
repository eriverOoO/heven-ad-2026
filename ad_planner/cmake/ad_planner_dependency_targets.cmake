# The Humble binary exposes two package-config entry points for BehaviorTree.CPP.
# The ament entry point defines the target required by this package, while the
# upstream entry point defines BT::behaviortree_cpp_v3.  Normalize either result
# for downstream consumers of the exported ad_planner targets.
if(NOT TARGET behaviortree_cpp_v3::behaviortree_cpp_v3)
  if(TARGET BT::behaviortree_cpp_v3)
    add_library(behaviortree_cpp_v3::behaviortree_cpp_v3 INTERFACE IMPORTED)
    set_target_properties(
      behaviortree_cpp_v3::behaviortree_cpp_v3
      PROPERTIES INTERFACE_LINK_LIBRARIES BT::behaviortree_cpp_v3)
  else()
    message(FATAL_ERROR
      "ad_planner requires the BehaviorTree.CPP v3 imported target")
  endif()
endif()
