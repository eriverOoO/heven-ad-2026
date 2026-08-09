#include "ad_control/path/route_speed_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ad_control
{
namespace
{
constexpr std::size_t kCurvatureWindowRadius = 5;
constexpr double kMinimumProfileSpeedMps = 5.0 / 3.6;
constexpr double kMaximumProfileSpeedMps = 60.0 / 3.6;
constexpr double kAccelerationMps2 = 2.0;
constexpr double kDecelerationMps2 = 2.5;
constexpr double kCurvatureLookaheadM = 1.0;

long double distance_xy(const Point3 & lhs, const Point3 & rhs)
{
  return std::hypot(
    static_cast<long double>(lhs.x) - rhs.x,
    static_cast<long double>(lhs.y) - rhs.y);
}

long double curvature(
  const Point3 & first, const Point3 & middle, const Point3 & last)
{
  const long double a = distance_xy(first, middle);
  const long double b = distance_xy(middle, last);
  const long double c = distance_xy(first, last);
  if (a * b * c <= 1.0e-9L) {
    return 0.0L;
  }
  const long double ab_x = static_cast<long double>(middle.x) - first.x;
  const long double ab_y = static_cast<long double>(middle.y) - first.y;
  const long double ac_x = static_cast<long double>(last.x) - first.x;
  const long double ac_y = static_cast<long double>(last.y) - first.y;
  const long double twice_area = std::abs(ab_x * ac_y - ab_y * ac_x);
  return 2.0L * twice_area / (a * b * c);
}

std::size_t offset_by_distance(
  const Route & route, std::size_t start, bool forward, double distance_m)
{
  const std::size_t count = route.points.size();
  std::size_t current = start;
  long double accumulated = 0.0L;
  for (std::size_t step = 0;
    step + 1 < count && accumulated < static_cast<long double>(distance_m);
    ++step)
  {
    std::size_t next = current;
    if (forward) {
      if (current + 1 < count) {
        next = current + 1;
      } else if (route.closed) {
        next = 0;
      } else {
        break;
      }
    } else {
      if (current > 0) {
        next = current - 1;
      } else if (route.closed) {
        next = count - 1;
      } else {
        break;
      }
    }
    accumulated += distance_xy(route.points[current], route.points[next]);
    current = next;
  }
  return current;
}

void validate_longitudinal_profile(const LongitudinalProfile & profile)
{
  if (!std::isfinite(profile.braking_delay_s) || profile.braking_delay_s < 0.0) {
    throw std::invalid_argument("profile braking delay must be finite and nonnegative");
  }
  if (profile.empty()) {
    return;
  }
  const std::size_t count = profile.speed_mps.size();
  if (count < 2 || profile.acceleration_mps2.size() != count ||
    profile.deceleration_mps2.size() != count)
  {
    throw std::invalid_argument(
            "longitudinal profile arrays must have the same size of at least two");
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(profile.speed_mps[index]) ||
      !std::isfinite(profile.acceleration_mps2[index]) ||
      !std::isfinite(profile.deceleration_mps2[index]) ||
      profile.speed_mps[index] < 0.0 ||
      profile.acceleration_mps2[index] <= 0.0 ||
      profile.deceleration_mps2[index] <= 0.0 ||
      (index > 0 && profile.speed_mps[index] <= profile.speed_mps[index - 1]))
    {
      throw std::invalid_argument(
              "longitudinal profile values must be finite, positive, and speed-sorted");
    }
  }
}

double interpolate(
  const std::vector<double> & speed_mps,
  const std::vector<double> & values,
  double query_speed_mps)
{
  if (query_speed_mps <= speed_mps.front()) {
    return values.front();
  }
  if (query_speed_mps >= speed_mps.back()) {
    return values.back();
  }
  const auto upper = std::upper_bound(
    speed_mps.begin(), speed_mps.end(), query_speed_mps);
  const std::size_t upper_index =
    static_cast<std::size_t>(upper - speed_mps.begin());
  const std::size_t lower_index = upper_index - 1;
  const double fraction =
    (query_speed_mps - speed_mps[lower_index]) /
    (speed_mps[upper_index] - speed_mps[lower_index]);
  return values[lower_index] +
         fraction * (values[upper_index] - values[lower_index]);
}

double reachable_speed(
  double adjacent_speed, double acceleration, long double distance,
  double maximum_speed_mps)
{
  const long double squared =
    static_cast<long double>(adjacent_speed) * adjacent_speed +
    2.0L * acceleration * distance;
  return static_cast<double>(std::min(
           std::sqrt(std::max(0.0L, squared)),
           static_cast<long double>(maximum_speed_mps)));
}

void apply_braking_delay(
  const Route & route,
  const std::vector<long double> & segment_length,
  double delay_distance_m,
  std::vector<double> & speed_cap)
{
  if (delay_distance_m <= 0.0 || speed_cap.empty()) {
    return;
  }
  const std::vector<double> original = speed_cap;
  const std::size_t count = speed_cap.size();
  for (std::size_t destination = 0; destination < count; ++destination) {
    std::size_t current = destination;
    long double accumulated = 0.0L;
    for (std::size_t step = 0;
      step + 1 < count &&
      accumulated < static_cast<long double>(delay_distance_m);
      ++step)
    {
      if (current == 0) {
        if (!route.closed) {
          break;
        }
        current = count - 1;
      } else {
        --current;
      }
      accumulated += segment_length[current];
      speed_cap[current] = std::min(speed_cap[current], original[destination]);
    }
  }
}

struct RouteProjection
{
  std::size_t index;
  long double distance_m;
};

RouteProjection nearest_route_projection(
  const Route & route, const Point3 & point)
{
  std::size_t nearest = 0;
  long double nearest_distance = std::numeric_limits<long double>::infinity();
  for (std::size_t index = 0; index < route.points.size(); ++index) {
    const long double distance = distance_xy(route.points[index], point);
    if (distance < nearest_distance) {
      nearest = index;
      nearest_distance = distance;
    }
  }
  return RouteProjection{nearest, nearest_distance};
}

void apply_speed_zones(
  const Route & route, const std::vector<RouteSpeedZone> & zones,
  std::vector<double> & speed_cap)
{
  for (const auto & zone : zones) {
    if (!std::isfinite(zone.start.x) || !std::isfinite(zone.start.y) ||
      !std::isfinite(zone.end.x) || !std::isfinite(zone.end.y) ||
      !std::isfinite(zone.maximum_speed_mps) || zone.maximum_speed_mps < 0.0 ||
      !std::isfinite(zone.maximum_projection_distance_m) ||
      zone.maximum_projection_distance_m <= 0.0)
    {
      throw std::invalid_argument(
              "route speed zone values and projection distance must be physically valid");
    }
    const auto start_projection = nearest_route_projection(route, zone.start);
    const auto end_projection = nearest_route_projection(route, zone.end);
    if (start_projection.distance_m > zone.maximum_projection_distance_m ||
      end_projection.distance_m > zone.maximum_projection_distance_m)
    {
      continue;
    }
    const std::size_t start = start_projection.index;
    const std::size_t end = end_projection.index;
    if (!route.closed && start > end) {
      throw std::invalid_argument(
              "open-route speed zone start must precede its end");
    }
    std::size_t index = start;
    while (true) {
      speed_cap[index] = std::min(speed_cap[index], zone.maximum_speed_mps);
      if (index == end) {
        break;
      }
      index = (index + 1) % route.points.size();
    }
  }
}
}  // namespace

double longitudinal_acceleration_limit(
  const RouteSpeedProfileConfig & config, double speed_mps, bool braking)
{
  const double configured =
    braking ? config.deceleration_mps2 : config.acceleration_mps2;
  if (config.longitudinal_profile.empty()) {
    return configured;
  }
  const auto & values = braking ?
    config.longitudinal_profile.deceleration_mps2 :
    config.longitudinal_profile.acceleration_mps2;
  return std::min(
    configured,
    interpolate(config.longitudinal_profile.speed_mps, values, speed_mps));
}

RouteSpeedProfile build_route_speed_profile(
  const Route & route, const RouteSpeedProfileConfig & config)
{
  if (!std::isfinite(config.minimum_speed_mps) || config.minimum_speed_mps < 0.0 ||
    !std::isfinite(config.maximum_speed_mps) ||
    config.maximum_speed_mps < config.minimum_speed_mps ||
    !std::isfinite(config.lateral_acceleration_mps2) ||
    config.lateral_acceleration_mps2 <= 0.0 ||
    !std::isfinite(config.acceleration_mps2) || config.acceleration_mps2 <= 0.0 ||
    !std::isfinite(config.deceleration_mps2) || config.deceleration_mps2 <= 0.0 ||
    config.curvature_window_radius == 0 ||
    !std::isfinite(config.curvature_lookahead_m) ||
    config.curvature_lookahead_m <= 0.0)
  {
    throw std::invalid_argument("route speed profile limits must be finite and physically valid");
  }
  validate_longitudinal_profile(config.longitudinal_profile);

  const std::size_t count = route.points.size();
  if (count == 0) {
    return RouteSpeedProfile{};
  }
  std::vector<long double> segment_length(count, 0.0L);
  for (std::size_t index = 0; index + 1 < count; ++index) {
    segment_length[index] = distance_xy(route.points[index], route.points[index + 1]);
  }
  if (route.closed) {
    segment_length.back() = distance_xy(route.points.back(), route.points.front());
  }

  std::vector<long double> raw_curvature(count, 0.0L);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t previous =
      offset_by_distance(route, index, false, config.curvature_lookahead_m);
    const std::size_t next =
      offset_by_distance(route, index, true, config.curvature_lookahead_m);
    if (previous != index && next != index && previous != next) {
      raw_curvature[index] = curvature(
        route.points[previous], route.points[index], route.points[next]);
    }
  }

  std::vector<double> smoothed_curvature(count, 0.0);
  std::vector<double> speed_cap(count, config.maximum_speed_mps);
  const bool closed_window_covers_route =
    route.closed && config.curvature_window_radius >= count / 2;
  long double closed_route_maximum_curvature = 0.0L;
  if (closed_window_covers_route) {
    for (const long double value : raw_curvature) {
      closed_route_maximum_curvature =
        std::max(closed_route_maximum_curvature, value);
    }
  }
  for (std::size_t index = 0; index < count; ++index) {
    long double worst = 0.0L;
    if (closed_window_covers_route) {
      worst = closed_route_maximum_curvature;
    } else if (route.closed) {
      worst = raw_curvature[index];
      for (std::size_t offset = 1;
        offset <= config.curvature_window_radius; ++offset)
      {
        const std::size_t distance_to_end = count - index;
        const std::size_t forward =
          offset < distance_to_end ? index + offset : offset - distance_to_end;
        const std::size_t backward =
          index >= offset ? index - offset : count - (offset - index);
        worst = std::max(worst, raw_curvature[forward]);
        worst = std::max(worst, raw_curvature[backward]);
      }
    } else {
      const std::size_t first =
        index - std::min(config.curvature_window_radius, index);
      const std::size_t last =
        index + std::min(config.curvature_window_radius, count - 1 - index);
      for (std::size_t candidate = first; candidate <= last; ++candidate) {
        worst = std::max(worst, raw_curvature[candidate]);
      }
    }
    smoothed_curvature[index] = static_cast<double>(worst);
    const long double curve_speed = std::sqrt(
      static_cast<long double>(config.lateral_acceleration_mps2) /
      std::max(worst, 1.0e-6L));
    speed_cap[index] = static_cast<double>(std::clamp(
        curve_speed, static_cast<long double>(config.minimum_speed_mps),
        static_cast<long double>(config.maximum_speed_mps)));
  }

  apply_speed_zones(route, config.speed_zones, speed_cap);

  if (!route.closed) {
    speed_cap.back() = 0.0;
  }
  if (!config.longitudinal_profile.empty()) {
    apply_braking_delay(
      route, segment_length,
      config.maximum_speed_mps *
      config.longitudinal_profile.braking_delay_s,
      speed_cap);
  }

  if (!route.closed) {
    for (std::size_t index = count - 1; index-- > 0; ) {
      speed_cap[index] = std::min(
        speed_cap[index],
        reachable_speed(
          speed_cap[index + 1],
          longitudinal_acceleration_limit(
            config, speed_cap[index + 1], true),
          segment_length[index], config.maximum_speed_mps));
    }
    for (std::size_t index = 1; index < count; ++index) {
      speed_cap[index] = std::min(
        speed_cap[index],
        reachable_speed(
          speed_cap[index - 1],
          longitudinal_acceleration_limit(
            config, speed_cap[index - 1], false),
          segment_length[index - 1], config.maximum_speed_mps));
    }
    return RouteSpeedProfile{
      std::move(speed_cap), std::move(smoothed_curvature),
      !config.longitudinal_profile.empty()};
  }

  std::vector<double> repeated(count * 3, config.maximum_speed_mps);
  for (std::size_t index = 0; index < repeated.size(); ++index) {
    repeated[index] = speed_cap[index % count];
  }
  for (std::size_t index = repeated.size() - 1; index-- > 0; ) {
    repeated[index] = std::min(
      repeated[index],
      reachable_speed(
        repeated[index + 1],
        longitudinal_acceleration_limit(
          config, repeated[index + 1], true),
        segment_length[index % count], config.maximum_speed_mps));
  }
  for (std::size_t index = 1; index < repeated.size(); ++index) {
    repeated[index] = std::min(
      repeated[index],
      reachable_speed(
        repeated[index - 1],
        longitudinal_acceleration_limit(
          config, repeated[index - 1], false),
        segment_length[(index - 1) % count], config.maximum_speed_mps));
  }
  return RouteSpeedProfile{
    std::vector<double>(repeated.begin() + count, repeated.begin() + 2 * count),
    std::move(smoothed_curvature),
    !config.longitudinal_profile.empty()};
}

std::vector<double> build_route_speed_profile(
  const Route & route,
  double lateral_acceleration_mps2)
{
  return build_route_speed_profile(
    route,
    RouteSpeedProfileConfig{
      kMinimumProfileSpeedMps,
      kMaximumProfileSpeedMps,
      lateral_acceleration_mps2,
      kAccelerationMps2,
      kDecelerationMps2,
      kCurvatureWindowRadius,
      kCurvatureLookaheadM,
      LongitudinalProfile{}}).speed_mps;
}

}  // namespace ad_control
