#ifndef AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_NODE_HPP_
#define AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_NODE_HPP_

#include <memory>

#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace ad_localization
{

std::shared_ptr<rclcpp::Node> make_imu_quaternion_encoder_node(
  const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_NODE_HPP_
