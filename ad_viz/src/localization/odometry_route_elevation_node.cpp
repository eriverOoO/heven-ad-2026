#include "ad_viz/localization/odometry_route_elevation_node.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "ad_viz/localization/route_elevation_projection.hpp"

namespace ad_viz::localization
{

OdometryRouteElevationNode::OdometryRouteElevationNode(
  const rclcpp::NodeOptions & options)
: Node("ad_localization_route_elevation_odometry", options)
{
  const std::string input_odometry_topic = declare_parameter<std::string>(
    "input_odometry_topic", "/ad/localization/odometry");
  const std::string input_path_topic = declare_parameter<std::string>(
    "input_path_topic", "/ad/planner/path");
  const std::string output_odometry_topic = declare_parameter<std::string>(
    "output_odometry_topic",
    "/ad/viz/localization/odometry_route_elevation");
  expected_odometry_frame_ = declare_parameter<std::string>(
    "expected_odometry_frame", "odom");
  expected_path_frame_ = declare_parameter<std::string>(
    "expected_path_frame", "map");
  maximum_lateral_distance_m_ = declare_parameter<double>(
    "maximum_lateral_distance_m", 10.0);

  const std::string resolved_odometry_input =
    get_node_topics_interface()->resolve_topic_name(input_odometry_topic);
  const std::string resolved_path_input =
    get_node_topics_interface()->resolve_topic_name(input_path_topic);
  const std::string resolved_output =
    get_node_topics_interface()->resolve_topic_name(output_odometry_topic);
  if (input_odometry_topic.empty() || input_path_topic.empty() ||
    output_odometry_topic.empty() || expected_odometry_frame_.empty() ||
    expected_path_frame_.empty() ||
    resolved_odometry_input == resolved_path_input ||
    resolved_odometry_input == resolved_output ||
    resolved_path_input == resolved_output ||
    !std::isfinite(maximum_lateral_distance_m_) ||
    maximum_lateral_distance_m_ < 0.0)
  {
    throw std::invalid_argument(
            "route-elevation visualization topics, frames, and distance must be valid and distinct");
  }

  const auto odometry_qos =
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  const auto path_qos =
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  publisher_ = create_publisher<nav_msgs::msg::Odometry>(
    output_odometry_topic, odometry_qos);
  path_subscription_ = create_subscription<nav_msgs::msg::Path>(
    input_path_topic, path_qos,
    [this](const nav_msgs::msg::Path::ConstSharedPtr path) {on_path(path);});
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    input_odometry_topic, odometry_qos,
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr odometry) {
      on_odometry(odometry);
    });
}

void OdometryRouteElevationNode::on_path(
  const nav_msgs::msg::Path::ConstSharedPtr path)
{
  if (path->header.frame_id != expected_path_frame_ ||
    !valid_route_elevation_path(*path))
  {
    RCLCPP_WARN(
      get_logger(),
      "Ignoring invalid route-elevation path; retaining the last valid path");
    return;
  }
  std::lock_guard<std::mutex> lock(path_mutex_);
  path_ = path;
}

void OdometryRouteElevationNode::on_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr odometry)
{
  if (odometry->header.frame_id != expected_odometry_frame_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Ignoring route-elevation odometry with an unexpected frame");
    return;
  }

  nav_msgs::msg::Path::ConstSharedPtr path;
  {
    std::lock_guard<std::mutex> lock(path_mutex_);
    path = path_;
  }
  std::optional<nav_msgs::msg::Odometry> projected;
  if (path) {
    projected = project_odometry_to_route_elevation(
      *odometry, *path, maximum_lateral_distance_m_);
  }
  if (path && !projected) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Holding the last valid route elevation for visualization odometry");
  }
  publisher_->publish(elevation_hold_.apply(*odometry, projected));
}

}  // namespace ad_viz::localization
