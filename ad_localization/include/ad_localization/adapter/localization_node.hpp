#ifndef AD_LOCALIZATION__ADAPTER__LOCALIZATION_NODE_HPP_
#define AD_LOCALIZATION__ADAPTER__LOCALIZATION_NODE_HPP_

#include <memory>

#include "rclcpp/node_options.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace ad_localization
{

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> make_localization_node(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__ADAPTER__LOCALIZATION_NODE_HPP_
