#ifndef AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_NODE_HPP_
#define AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_NODE_HPP_

#include <memory>

#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace ad_localization
{

std::shared_ptr<rclcpp::Node> make_localization_manager_node(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_NODE_HPP_
