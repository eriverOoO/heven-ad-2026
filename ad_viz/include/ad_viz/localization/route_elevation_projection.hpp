#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include <optional>

namespace ad_viz::localization
{

class RouteElevationHold final
{
public:
  nav_msgs::msg::Odometry apply(
    const nav_msgs::msg::Odometry & input,
    const std::optional<nav_msgs::msg::Odometry> & projected);

private:
  std::optional<double> last_valid_elevation_z_;
};

bool valid_route_elevation_path(const nav_msgs::msg::Path & path) noexcept;

std::optional<nav_msgs::msg::Odometry> project_odometry_to_route_elevation(
  const nav_msgs::msg::Odometry & input,
  const nav_msgs::msg::Path & path,
  double maximum_lateral_distance_m);

}  // namespace ad_viz::localization
