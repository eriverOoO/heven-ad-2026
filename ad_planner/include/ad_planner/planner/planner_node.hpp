#ifndef AD_PLANNER__PLANNER__PLANNER_NODE_HPP_
#define AD_PLANNER__PLANNER__PLANNER_NODE_HPP_

#include <memory>

#include "rclcpp/node.hpp"

namespace ad_planner
{

std::shared_ptr<rclcpp::Node> make_planner_node();

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNER__PLANNER_NODE_HPP_
