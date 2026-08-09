#include "ad_control/path/route_progress_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ad_control
{
namespace
{
constexpr long double kPi = 3.14159265358979323846L;
constexpr long double kMaximumHeadingErrorRad = kPi / 2.0L;
constexpr long double kHeadingBranchAllowanceM = 5.0L;

bool finite_point(const Point3 & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

long double euclidean_distance(const Point3 & lhs, const Point3 & rhs)
{
  const long double dx =
    static_cast<long double>(lhs.x) - static_cast<long double>(rhs.x);
  const long double dy =
    static_cast<long double>(lhs.y) - static_cast<long double>(rhs.y);
  const long double dz =
    static_cast<long double>(lhs.z) - static_cast<long double>(rhs.z);
  return std::hypot(dx, dy, dz);
}

std::size_t nearest_index(
  const Route & route, const Point3 & position, std::size_t first, std::size_t last)
{
  std::size_t best = first;
  long double best_distance = euclidean_distance(position, route.points[first]);
  for (std::size_t index = first + 1; index <= last; ++index) {
    const long double candidate = euclidean_distance(position, route.points[index]);
    if (candidate < best_distance) {
      best = index;
      best_distance = candidate;
    }
  }
  return best;
}

long double route_heading(const Route & route, std::size_t index)
{
  const std::size_t terminal_index = route.points.size() - 1;
  const std::size_t first = index < terminal_index ? index : index - 1;
  const std::size_t second = index < terminal_index ? index + 1 : index;
  return std::atan2(
    static_cast<long double>(route.points[second].y) - route.points[first].y,
    static_cast<long double>(route.points[second].x) - route.points[first].x);
}

std::size_t direction_aware_nearest_index(
  const Route & route, const Point3 & position, double yaw_rad,
  std::size_t first, std::size_t last)
{
  const std::size_t spatial_best = nearest_index(route, position, first, last);
  const long double spatial_distance =
    euclidean_distance(position, route.points[spatial_best]);
  std::size_t directional_best = spatial_best;
  long double directional_distance = std::numeric_limits<long double>::infinity();
  for (std::size_t index = first; index <= last; ++index) {
    const long double heading_error = std::abs(std::remainder(
      route_heading(route, index) - static_cast<long double>(yaw_rad),
      2.0L * kPi));
    if (heading_error > kMaximumHeadingErrorRad) {
      continue;
    }
    const long double candidate = euclidean_distance(position, route.points[index]);
    if (candidate < directional_distance) {
      directional_best = index;
      directional_distance = candidate;
    }
  }
  if (directional_distance <= spatial_distance + kHeadingBranchAllowanceM) {
    return directional_best;
  }
  return spatial_best;
}

void validate_route(const Route & route)
{
  if (route.points.size() < 2) {
    throw std::invalid_argument("route requires at least two points");
  }
  bool any_distinct = false;
  for (std::size_t index = 0; index < route.points.size(); ++index) {
    if (!finite_point(route.points[index])) {
      throw std::invalid_argument("route contains a non-finite point");
    }
    if (index > 0 &&
      euclidean_distance(route.points[index - 1], route.points[index]) == 0.0L)
    {
      throw std::invalid_argument("route contains consecutive duplicate points");
    }
    any_distinct = any_distinct ||
      euclidean_distance(route.points.front(), route.points[index]) > 0.0L;
  }
  if (!any_distinct) {
    throw std::invalid_argument("route is fully degenerate");
  }
}

}  // namespace

RouteProgressTracker::RouteProgressTracker(
  Route route, std::size_t forward_window, std::size_t max_laps)
: route_(std::move(route)), forward_window_(forward_window), max_laps_(max_laps)
{
  validate_route(route_);
  if (forward_window_ == 0) {
    throw std::invalid_argument("forward window must be positive");
  }
  if (max_laps_ == 0) {
    throw std::invalid_argument("maximum laps must be positive");
  }
}

RouteProgressState RouteProgressTracker::update(const Point3 & position)
{
  if (!finite_point(position)) {
    throw std::invalid_argument("route progress position must be finite");
  }
  if (state_.terminal_full_brake) {
    return state_;
  }

  const std::size_t terminal_index = route_.points.size() - 1;
  if (!state_.initialized) {
    state_.target_index = nearest_index(route_, position, 0, terminal_index);
    state_.initialized = true;
    if (!route_.closed && state_.target_index == terminal_index) {
      state_.terminal_full_brake = true;
    }
    return state_;
  }

  if (route_.closed && state_.target_index == terminal_index) {
    const std::size_t wrapped_last = std::min(forward_window_ - 1, terminal_index);
    const std::size_t wrapped_best = nearest_index(route_, position, 0, wrapped_last);
    const long double terminal_distance =
      euclidean_distance(position, route_.points[terminal_index]);
    const long double wrapped_distance =
      euclidean_distance(position, route_.points[wrapped_best]);
    if (wrapped_distance < terminal_distance) {
      state_.target_index = wrapped_best;
      ++state_.lap_count;
      if (state_.lap_count >= max_laps_) {
        state_.terminal_full_brake = true;
      }
    }
    return state_;
  }

  const std::size_t remaining = terminal_index - state_.target_index;
  const std::size_t last =
    state_.target_index + std::min(forward_window_, remaining);
  state_.target_index = nearest_index(route_, position, state_.target_index, last);
  if (!route_.closed && state_.target_index == terminal_index) {
    state_.terminal_full_brake = true;
  }
  return state_;
}

RouteProgressState RouteProgressTracker::update(
  const Point3 & position, double yaw_rad)
{
  if (!finite_point(position) || !std::isfinite(yaw_rad)) {
    throw std::invalid_argument("route progress pose must be finite");
  }
  if (state_.terminal_full_brake) {
    return state_;
  }

  const std::size_t terminal_index = route_.points.size() - 1;
  if (!state_.initialized) {
    state_.target_index = direction_aware_nearest_index(
      route_, position, yaw_rad, 0, terminal_index);
    state_.initialized = true;
    if (!route_.closed && state_.target_index == terminal_index) {
      state_.terminal_full_brake = true;
    }
    return state_;
  }

  if (route_.closed && state_.target_index == terminal_index) {
    const std::size_t wrapped_last = std::min(forward_window_ - 1, terminal_index);
    const std::size_t wrapped_best = direction_aware_nearest_index(
      route_, position, yaw_rad, 0, wrapped_last);
    const long double terminal_distance =
      euclidean_distance(position, route_.points[terminal_index]);
    const long double wrapped_distance =
      euclidean_distance(position, route_.points[wrapped_best]);
    if (wrapped_distance < terminal_distance) {
      state_.target_index = wrapped_best;
      ++state_.lap_count;
      if (state_.lap_count >= max_laps_) {
        state_.terminal_full_brake = true;
      }
    }
    return state_;
  }

  const std::size_t remaining = terminal_index - state_.target_index;
  const std::size_t last =
    state_.target_index + std::min(forward_window_, remaining);
  state_.target_index = direction_aware_nearest_index(
    route_, position, yaw_rad, state_.target_index, last);
  if (!route_.closed && state_.target_index == terminal_index) {
    state_.terminal_full_brake = true;
  }
  return state_;
}

const RouteProgressState & RouteProgressTracker::state() const noexcept
{
  return state_;
}

}  // namespace ad_control
