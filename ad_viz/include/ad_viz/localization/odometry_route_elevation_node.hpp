#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <mutex>
#include <string>

#include "ad_viz/localization/route_elevation_projection.hpp"

namespace ad_viz::localization
{

class OdometryRouteElevationNode final : public rclcpp::Node
{
public:
  explicit OdometryRouteElevationNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_path(nav_msgs::msg::Path::ConstSharedPtr path);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr odometry);

  std::string expected_odometry_frame_;
  std::string expected_path_frame_;
  double maximum_lateral_distance_m_{10.0};
  std::mutex path_mutex_;
  nav_msgs::msg::Path::ConstSharedPtr path_;
  RouteElevationHold elevation_hold_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
};

}  // namespace ad_viz::localization
