#include "ad_viz/localization/visualization_tf_relay_node.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#include "ad_viz/localization/visualization_tf.hpp"

namespace ad_viz::localization
{

VisualizationTfRelayNode::VisualizationTfRelayNode(
  const rclcpp::NodeOptions & options)
: Node("ad_visualization_tf_relay", options)
{
  const std::string source_dynamic_topic = declare_parameter<std::string>(
    "source_dynamic_tf_topic", "/tf");
  const std::string source_static_topic = declare_parameter<std::string>(
    "source_static_tf_topic", "/tf_static");
  const std::string output_dynamic_topic = declare_parameter<std::string>(
    "output_dynamic_tf_topic", "/ad/viz/tf");
  const std::string output_static_topic = declare_parameter<std::string>(
    "output_static_tf_topic", "/ad/viz/tf_static");
  const std::string route_odometry_topic = declare_parameter<std::string>(
    "route_odometry_topic",
    "/ad/viz/localization/odometry_route_elevation");
  visual_parent_frame_ = declare_parameter<std::string>(
    "visual_parent_frame", "odom");
  visual_child_frame_ = declare_parameter<std::string>(
    "visual_child_frame", "base_link");

  const std::array<std::string, 5> topics{
    source_dynamic_topic, source_static_topic, output_dynamic_topic,
    output_static_topic, route_odometry_topic};
  if (std::any_of(
      topics.begin(), topics.end(),
      [](const std::string & topic) {return topic.empty();}) ||
    visual_parent_frame_.empty() || visual_child_frame_.empty() ||
    visual_parent_frame_ == visual_child_frame_)
  {
    throw std::invalid_argument(
            "visualization TF topics and frames must be nonempty and valid");
  }

  const auto dynamic_qos =
    rclcpp::QoS(rclcpp::KeepLast(100)).reliable().durability_volatile();
  const auto static_qos =
    rclcpp::QoS(rclcpp::KeepLast(100)).reliable().transient_local();
  const auto odometry_qos =
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

  dynamic_publisher_ = create_publisher<tf2_msgs::msg::TFMessage>(
    output_dynamic_topic, dynamic_qos);
  static_publisher_ = create_publisher<tf2_msgs::msg::TFMessage>(
    output_static_topic, static_qos);
  dynamic_subscription_ = create_subscription<tf2_msgs::msg::TFMessage>(
    source_dynamic_topic, dynamic_qos,
    [this](const tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
      on_dynamic_tf(message);
    });
  static_subscription_ = create_subscription<tf2_msgs::msg::TFMessage>(
    source_static_topic, static_qos,
    [this](const tf2_msgs::msg::TFMessage::ConstSharedPtr message) {
      on_static_tf(message);
    });
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    route_odometry_topic, odometry_qos,
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr odometry) {
      on_route_odometry(odometry);
    });
}

void VisualizationTfRelayNode::on_dynamic_tf(
  const tf2_msgs::msg::TFMessage::ConstSharedPtr message)
{
  auto filtered = without_transform_edge(
    *message, visual_parent_frame_, visual_child_frame_);
  if (!filtered.transforms.empty()) {
    dynamic_publisher_->publish(std::move(filtered));
  }
}

void VisualizationTfRelayNode::on_static_tf(
  const tf2_msgs::msg::TFMessage::ConstSharedPtr message)
{
  auto filtered = without_transform_edge(
    *message, visual_parent_frame_, visual_child_frame_);
  if (!filtered.transforms.empty()) {
    static_publisher_->publish(std::move(filtered));
  }
}

void VisualizationTfRelayNode::on_route_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr odometry)
{
  auto transform = visualization_transform_from_odometry(
    *odometry, visual_parent_frame_, visual_child_frame_);
  if (!transform) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Ignoring invalid route-elevation odometry for visualization TF");
    return;
  }
  tf2_msgs::msg::TFMessage message;
  message.transforms.push_back(std::move(*transform));
  dynamic_publisher_->publish(std::move(message));
}

}  // namespace ad_viz::localization
