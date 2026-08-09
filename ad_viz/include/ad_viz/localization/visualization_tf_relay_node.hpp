#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <string>

namespace ad_viz::localization
{

class VisualizationTfRelayNode final : public rclcpp::Node
{
public:
  explicit VisualizationTfRelayNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_dynamic_tf(tf2_msgs::msg::TFMessage::ConstSharedPtr message);
  void on_static_tf(tf2_msgs::msg::TFMessage::ConstSharedPtr message);
  void on_route_odometry(nav_msgs::msg::Odometry::ConstSharedPtr odometry);

  std::string visual_parent_frame_;
  std::string visual_child_frame_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_publisher_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr static_publisher_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_subscription_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
};

}  // namespace ad_viz::localization
