#ifndef AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_NODE_HPP_
#define AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_NODE_HPP_

#include <memory>

#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace ad_localization
{

std::shared_ptr<rclcpp::Node> make_quaternion_wheel_gnss_ekf_node(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_NODE_HPP_
