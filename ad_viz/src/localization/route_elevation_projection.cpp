#include "ad_viz/localization/route_elevation_projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ad_viz::localization
{
namespace
{

bool finite_position(const geometry_msgs::msg::Point & position) noexcept
{
  return std::isfinite(position.x) && std::isfinite(position.y) &&
         std::isfinite(position.z);
}

}  // namespace

nav_msgs::msg::Odometry RouteElevationHold::apply(
  const nav_msgs::msg::Odometry & input,
  const std::optional<nav_msgs::msg::Odometry> & projected)
{
  if (projected) {
    last_valid_elevation_z_ = projected->pose.pose.position.z;
    return *projected;
  }

  auto output = input;
  if (last_valid_elevation_z_) {
    output.pose.pose.position.z = *last_valid_elevation_z_;
  }
  return output;
}

bool valid_route_elevation_path(const nav_msgs::msg::Path & path) noexcept
{
  return !path.poses.empty() &&
         std::all_of(
    path.poses.begin(), path.poses.end(),
    [](const geometry_msgs::msg::PoseStamped & pose) {
      return finite_position(pose.pose.position);
    });
}

std::optional<nav_msgs::msg::Odometry> project_odometry_to_route_elevation(
  const nav_msgs::msg::Odometry & input,
  const nav_msgs::msg::Path & path,
  const double maximum_lateral_distance_m)
{
  const auto & input_position = input.pose.pose.position;
  if (!std::isfinite(input_position.x) || !std::isfinite(input_position.y) ||
    !std::isfinite(maximum_lateral_distance_m) ||
    maximum_lateral_distance_m < 0.0 || !valid_route_elevation_path(path))
  {
    return std::nullopt;
  }

  double best_distance_squared = std::numeric_limits<double>::infinity();
  double best_z = 0.0;
  if (path.poses.size() == 1U) {
    const auto & point = path.poses.front().pose.position;
    const double dx = input_position.x - point.x;
    const double dy = input_position.y - point.y;
    best_distance_squared = dx * dx + dy * dy;
    best_z = point.z;
  } else {
    for (std::size_t index = 0U; index + 1U < path.poses.size(); ++index) {
      const auto & start = path.poses[index].pose.position;
      const auto & end = path.poses[index + 1U].pose.position;
      const double dx = end.x - start.x;
      const double dy = end.y - start.y;
      const double length_squared = dx * dx + dy * dy;
      double interpolation = 0.0;
      if (length_squared > 0.0 && std::isfinite(length_squared)) {
        interpolation = std::clamp(
          ((input_position.x - start.x) * dx +
          (input_position.y - start.y) * dy) / length_squared,
          0.0, 1.0);
      }
      const double projected_x = start.x + interpolation * dx;
      const double projected_y = start.y + interpolation * dy;
      const double offset_x = input_position.x - projected_x;
      const double offset_y = input_position.y - projected_y;
      const double distance_squared =
        offset_x * offset_x + offset_y * offset_y;
      const double projected_z =
        start.z + interpolation * (end.z - start.z);
      if (!std::isfinite(distance_squared) || !std::isfinite(projected_z)) {
        return std::nullopt;
      }
      if (distance_squared < best_distance_squared) {
        best_distance_squared = distance_squared;
        best_z = projected_z;
      }
    }
  }

  const double maximum_distance_squared =
    maximum_lateral_distance_m * maximum_lateral_distance_m;
  if (!std::isfinite(best_distance_squared) ||
    !std::isfinite(maximum_distance_squared) ||
    best_distance_squared > maximum_distance_squared)
  {
    return std::nullopt;
  }

  auto output = input;
  output.pose.pose.position.z = best_z;
  return output;
}

}  // namespace ad_viz::localization
