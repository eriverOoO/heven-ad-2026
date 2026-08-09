#ifndef AD_PLANNER__BT_NODES_HPP_
#define AD_PLANNER__BT_NODES_HPP_

#include <behaviortree_cpp_v3/bt_factory.h>

#include <memory>
#include <string>
#include <vector>

#include "ad_planner/behavior/planner_context.hpp"

namespace ad_planner
{

struct BtNodeEnvironment
{
  PlannerContext & context;
  const PlannerConfig & config;
  PlannerRuntimeState & state;
};

using BtNodeEnvironmentPtr = std::shared_ptr<BtNodeEnvironment>;

struct PerceptionForwardCheck
{
  Pose2 pose;
  FootprintConfig footprint;
  double far_x_m{0.0};
};

PerceptionForwardCheck make_perception_forward_check(
  const PerceptionMissionConfig & config, double speed_mps,
  double minimum_far_x_m = 0.0);

void register_ad_bt_nodes(
  BT::BehaviorTreeFactory & factory, const BtNodeEnvironmentPtr & environment);
std::vector<std::string> ad_bt_node_ids();

}  // namespace ad_planner

#endif  // AD_PLANNER__BT_NODES_HPP_
