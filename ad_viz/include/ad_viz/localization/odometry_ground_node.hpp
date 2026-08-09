#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace ad_viz::localization
{

class OdometryGroundNode final : public rclcpp::Node
{
public:
  explicit OdometryGroundNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
};

}  // namespace ad_viz::localization
