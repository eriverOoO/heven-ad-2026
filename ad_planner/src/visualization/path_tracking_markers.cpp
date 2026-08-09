#include "ad_planner/visualization/path_tracking_markers.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_planner {
namespace {

constexpr double kRouteLineWidthM = 0.12;
constexpr double kCurvatureLineWidthM = 0.05;
constexpr double kCurvatureMinimumLengthM = 0.1;
constexpr double kCurvatureScaleM2 = 3.0;
constexpr double kCurvatureMaximumLengthM = 2.0;

bool finite_point(const ad_control::Point3 &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

std::vector<std::size_t> sampled_indices(std::size_t route_size,
                                         std::size_t sample_stride) {
  std::vector<std::size_t> indices;
  for (std::size_t index = 0U; index < route_size; index += sample_stride) {
    indices.push_back(index);
    if (sample_stride > std::numeric_limits<std::size_t>::max() - index) {
      break;
    }
  }
  if (route_size > 0U &&
      (indices.empty() || indices.back() != route_size - 1U)) {
    indices.push_back(route_size - 1U);
  }
  return indices;
}

std_msgs::msg::ColorRGBA speed_color(double speed_mps, double minimum_speed_mps,
                                     double maximum_speed_mps) {
  const double range = maximum_speed_mps - minimum_speed_mps;
  const double normalized =
      range > 1.0e-9
          ? std::clamp((speed_mps - minimum_speed_mps) / range, 0.0, 1.0)
          : 0.5;
  std_msgs::msg::ColorRGBA color;
  color.r = static_cast<float>(normalized);
  color.g = static_cast<float>(1.0 - std::abs(2.0 * normalized - 1.0));
  color.b = static_cast<float>(1.0 - normalized);
  color.a = 1.0F;
  return color;
}

geometry_msgs::msg::Point marker_point(const ad_control::Point3 &point) {
  geometry_msgs::msg::Point result;
  result.x = point.x;
  result.y = point.y;
  result.z = point.z;
  return result;
}

} // namespace

visualization_msgs::msg::MarkerArray make_route_profile_markers(
    const ad_control::Route &route,
    const ad_control::RouteSpeedProfile *profile, const std::string &frame_id,
    const builtin_interfaces::msg::Time &stamp, std::size_t sample_stride) {
  if (sample_stride == 0U) {
    throw std::invalid_argument("profile sample stride must be positive");
  }
  if (profile == nullptr) {
    return visualization_msgs::msg::MarkerArray{};
  }
  if (profile->speed_mps.size() != route.points.size() ||
      profile->curvature_inv_m.size() != route.points.size()) {
    throw std::invalid_argument("route and speed profile sizes must match");
  }

  double minimum_speed_mps = std::numeric_limits<double>::infinity();
  double maximum_speed_mps = -std::numeric_limits<double>::infinity();
  for (const double speed_mps : profile->speed_mps) {
    if (std::isfinite(speed_mps)) {
      minimum_speed_mps = std::min(minimum_speed_mps, speed_mps);
      maximum_speed_mps = std::max(maximum_speed_mps, speed_mps);
    }
  }

  visualization_msgs::msg::Marker route_line;
  route_line.header.frame_id = frame_id;
  route_line.header.stamp = stamp;
  route_line.ns = "path_tracking_profile";
  route_line.id = 0;
  route_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
  route_line.action = visualization_msgs::msg::Marker::ADD;
  route_line.pose.orientation.w = 1.0;
  route_line.scale.x = kRouteLineWidthM;
  route_line.color.a = 1.0F;

  visualization_msgs::msg::Marker curvature_lines;
  curvature_lines.header = route_line.header;
  curvature_lines.ns = "path_tracking_curvature";
  curvature_lines.id = 1;
  curvature_lines.type = visualization_msgs::msg::Marker::LINE_LIST;
  curvature_lines.action = visualization_msgs::msg::Marker::ADD;
  curvature_lines.pose.orientation.w = 1.0;
  curvature_lines.scale.x = kCurvatureLineWidthM;
  curvature_lines.color.r = 1.0F;
  curvature_lines.color.g = 1.0F;
  curvature_lines.color.a = 0.9F;

  for (const std::size_t index :
       sampled_indices(route.points.size(), sample_stride)) {
    const auto &point = route.points[index];
    const double speed_mps = profile->speed_mps[index];
    if (finite_point(point) && std::isfinite(speed_mps) &&
        std::isfinite(minimum_speed_mps) && std::isfinite(maximum_speed_mps)) {
      route_line.points.push_back(marker_point(point));
      route_line.colors.push_back(
          speed_color(speed_mps, minimum_speed_mps, maximum_speed_mps));
    }

    const double curvature_inv_m = profile->curvature_inv_m[index];
    if (!finite_point(point) || !std::isfinite(curvature_inv_m)) {
      continue;
    }
    const std::size_t previous = index == 0U ? index : index - 1U;
    const std::size_t next = std::min(index + 1U, route.points.size() - 1U);
    if (!finite_point(route.points[previous]) ||
        !finite_point(route.points[next])) {
      continue;
    }
    double tangent_x = route.points[next].x - route.points[previous].x;
    double tangent_y = route.points[next].y - route.points[previous].y;
    const double tangent_norm = std::hypot(tangent_x, tangent_y);
    if (tangent_norm > 1.0e-9) {
      tangent_x /= tangent_norm;
      tangent_y /= tangent_norm;
    } else {
      tangent_x = 1.0;
      tangent_y = 0.0;
    }
    const double direction = curvature_inv_m < 0.0 ? -1.0 : 1.0;
    const double length_m =
        std::clamp(std::abs(curvature_inv_m) * kCurvatureScaleM2,
                   kCurvatureMinimumLengthM, kCurvatureMaximumLengthM);
    auto start = marker_point(point);
    auto end = start;
    end.x += -tangent_y * direction * length_m;
    end.y += tangent_x * direction * length_m;
    curvature_lines.points.push_back(std::move(start));
    curvature_lines.points.push_back(std::move(end));
  }

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(std::move(route_line));
  markers.markers.push_back(std::move(curvature_lines));
  return markers;
}

} // namespace ad_planner
