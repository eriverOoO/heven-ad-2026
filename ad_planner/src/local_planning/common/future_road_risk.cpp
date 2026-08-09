#include "ad_planner/local_planning/common/future_road_risk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_planner
{
namespace
{

constexpr double kGeometryEpsilon = 1.0e-9;

struct CorridorSegment
{
  Pose2 start;
  Pose2 end;
  double maximum_width_m{0.0};
};

bool finite(const double value)
{
  return std::isfinite(value);
}

void require_finite(const double value, const char * const name)
{
  if (!finite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double squared_distance(
  const double first_x, const double first_y,
  const double second_x, const double second_y)
{
  const double dx = first_x - second_x;
  const double dy = first_y - second_y;
  const double result = dx * dx + dy * dy;
  if (!finite(result)) {
    throw std::invalid_argument("future-road-risk distance overflowed");
  }
  return result;
}

double point_segment_squared_distance(
  const Pose2 & point, const Pose2 & start, const Pose2 & end)
{
  const double dx = end.x - start.x;
  const double dy = end.y - start.y;
  const double length_squared = dx * dx + dy * dy;
  if (!finite(length_squared) || !(length_squared > 0.0)) {
    throw std::invalid_argument(
            "future-road-risk segment geometry is invalid");
  }
  const double raw_ratio =
    ((point.x - start.x) * dx + (point.y - start.y) * dy) /
    length_squared;
  if (!finite(raw_ratio)) {
    throw std::invalid_argument(
            "future-road-risk projection is invalid");
  }
  const double ratio = std::clamp(raw_ratio, 0.0, 1.0);
  return squared_distance(
    point.x, point.y, start.x + ratio * dx, start.y + ratio * dy);
}

double orientation(
  const Pose2 & first, const Pose2 & second, const Pose2 & point)
{
  const double result =
    (second.x - first.x) * (point.y - first.y) -
    (second.y - first.y) * (point.x - first.x);
  if (!finite(result)) {
    throw std::invalid_argument(
            "future-road-risk orientation overflowed");
  }
  return result;
}

bool point_on_segment(
  const Pose2 & point, const Pose2 & start, const Pose2 & end)
{
  return std::abs(orientation(start, end, point)) <= kGeometryEpsilon &&
         point.x >= std::min(start.x, end.x) - kGeometryEpsilon &&
         point.x <= std::max(start.x, end.x) + kGeometryEpsilon &&
         point.y >= std::min(start.y, end.y) - kGeometryEpsilon &&
         point.y <= std::max(start.y, end.y) + kGeometryEpsilon;
}

bool segments_intersect(
  const Pose2 & first_start, const Pose2 & first_end,
  const Pose2 & second_start, const Pose2 & second_end)
{
  const double first_side_start =
    orientation(first_start, first_end, second_start);
  const double first_side_end =
    orientation(first_start, first_end, second_end);
  const double second_side_start =
    orientation(second_start, second_end, first_start);
  const double second_side_end =
    orientation(second_start, second_end, first_end);
  if (((first_side_start > kGeometryEpsilon &&
    first_side_end < -kGeometryEpsilon) ||
    (first_side_start < -kGeometryEpsilon &&
    first_side_end > kGeometryEpsilon)) &&
    ((second_side_start > kGeometryEpsilon &&
    second_side_end < -kGeometryEpsilon) ||
    (second_side_start < -kGeometryEpsilon &&
    second_side_end > kGeometryEpsilon)))
  {
    return true;
  }
  return point_on_segment(second_start, first_start, first_end) ||
         point_on_segment(second_end, first_start, first_end) ||
         point_on_segment(first_start, second_start, second_end) ||
         point_on_segment(first_end, second_start, second_end);
}

double segment_squared_distance(
  const Pose2 & first_start, const Pose2 & first_end,
  const Pose2 & second_start, const Pose2 & second_end)
{
  if (segments_intersect(
      first_start, first_end, second_start, second_end))
  {
    return 0.0;
  }
  return std::min(
    {point_segment_squared_distance(first_start, second_start, second_end),
      point_segment_squared_distance(first_end, second_start, second_end),
      point_segment_squared_distance(second_start, first_start, first_end),
      point_segment_squared_distance(second_end, first_start, first_end)});
}

std::vector<CorridorSegment> validate_corridor(
  const ReferenceCorridor & corridor,
  const FutureRoadRiskLimits & limits)
{
  if (corridor.frame_id.empty() || corridor.lanes.empty() ||
    corridor.primary_lane_index >= corridor.lanes.size())
  {
    throw std::invalid_argument(
            "future-road-risk corridor metadata is invalid");
  }
  std::vector<CorridorSegment> segments;
  for (const auto & lane : corridor.lanes) {
    if (lane.points.size() < 2U) {
      throw std::invalid_argument(
              "future-road-risk lane must have at least two points");
    }
    if (lane.points.size() - 1U >
      limits.maximum_corridor_segments - segments.size())
    {
      throw std::invalid_argument(
              "future-road-risk corridor exceeds the segment bound");
    }
    for (std::size_t index = 0U; index < lane.points.size(); ++index) {
      const auto & point = lane.points[index];
      require_finite(point.pose.x, "future-road-risk corridor x");
      require_finite(point.pose.y, "future-road-risk corridor y");
      require_finite(point.pose.yaw_rad, "future-road-risk corridor yaw");
      require_finite(point.route_s_m, "future-road-risk corridor station");
      require_finite(point.left_width_m, "future-road-risk left width");
      require_finite(point.right_width_m, "future-road-risk right width");
      if (!(point.left_width_m > 0.0) ||
        !(point.right_width_m > 0.0))
      {
        throw std::invalid_argument(
                "future-road-risk corridor widths must be positive");
      }
      if (index == 0U) {
        continue;
      }
      const auto & previous = lane.points[index - 1U];
      if (!(point.route_s_m > previous.route_s_m) ||
        squared_distance(
          point.pose.x, point.pose.y,
          previous.pose.x, previous.pose.y) <= kGeometryEpsilon)
      {
        throw std::invalid_argument(
                "future-road-risk corridor segment is invalid");
      }
      segments.push_back(
        CorridorSegment{
          previous.pose, point.pose,
          std::max(
            {previous.left_width_m, previous.right_width_m,
              point.left_width_m, point.right_width_m})});
    }
  }
  return segments;
}

double covariance_margin(
  const PredictedFootprint & footprint,
  const FutureRoadRiskLimits & limits)
{
  const double midpoint =
    0.5 * footprint.covariance_xx + 0.5 * footprint.covariance_yy;
  const double eigen_radius = std::hypot(
    0.5 * footprint.covariance_xx - 0.5 * footprint.covariance_yy,
    footprint.covariance_xy);
  const double maximum_eigenvalue = midpoint + eigen_radius;
  if (!finite(midpoint) || !finite(eigen_radius) ||
    midpoint - eigen_radius < -kGeometryEpsilon ||
    !finite(maximum_eigenvalue))
  {
    throw std::invalid_argument(
            "future-road-risk covariance must be positive semidefinite");
  }
  const double margin = limits.covariance_sigma *
    std::sqrt(std::max(0.0, maximum_eigenvalue));
  if (!finite(margin)) {
    throw std::invalid_argument(
            "future-road-risk covariance margin overflowed");
  }
  return std::max(limits.minimum_margin_m, margin);
}

double footprint_radius(
  const PredictedFootprint & footprint,
  const FutureRoadRiskLimits & limits)
{
  return std::hypot(0.5 * footprint.length_m, 0.5 * footprint.width_m) +
         covariance_margin(footprint, limits);
}

void validate_footprint(
  const PredictedFootprint & footprint,
  const FutureRoadRiskLimits & limits)
{
  require_finite(footprint.time_from_start_s, "future-road-risk time");
  require_finite(footprint.pose.x, "future-road-risk object x");
  require_finite(footprint.pose.y, "future-road-risk object y");
  require_finite(footprint.pose.yaw_rad, "future-road-risk object yaw");
  require_finite(footprint.length_m, "future-road-risk object length");
  require_finite(footprint.width_m, "future-road-risk object width");
  require_finite(footprint.covariance_xx, "future-road-risk covariance xx");
  require_finite(footprint.covariance_yy, "future-road-risk covariance yy");
  require_finite(footprint.covariance_xy, "future-road-risk covariance xy");
  if (!(footprint.length_m > 0.0) || !(footprint.width_m > 0.0) ||
    footprint.covariance_xx < 0.0 || footprint.covariance_yy < 0.0)
  {
    throw std::invalid_argument(
            "future-road-risk footprint dimensions are invalid");
  }
  (void)footprint_radius(footprint, limits);
}

PredictedFootprint interpolate(
  const PredictedFootprint & first, const PredictedFootprint & second,
  const double time_s)
{
  const double duration = second.time_from_start_s - first.time_from_start_s;
  if (!(duration > 0.0)) {
    throw std::invalid_argument(
            "future-road-risk prediction times must be increasing");
  }
  const double ratio = std::clamp(
    (time_s - first.time_from_start_s) / duration, 0.0, 1.0);
  const double yaw_delta = std::remainder(
    second.pose.yaw_rad - first.pose.yaw_rad,
    2.0 * std::acos(-1.0));
  return PredictedFootprint{
    time_s,
    Pose2{
      first.pose.x + ratio * (second.pose.x - first.pose.x),
      first.pose.y + ratio * (second.pose.y - first.pose.y),
      std::remainder(
        first.pose.yaw_rad + ratio * yaw_delta,
        2.0 * std::acos(-1.0))},
    first.length_m + ratio * (second.length_m - first.length_m),
    first.width_m + ratio * (second.width_m - first.width_m),
    first.covariance_xx +
    ratio * (second.covariance_xx - first.covariance_xx),
    first.covariance_yy +
    ratio * (second.covariance_yy - first.covariance_yy),
    first.covariance_xy +
    ratio * (second.covariance_xy - first.covariance_xy)};
}

bool point_intersects_corridor(
  const PredictedFootprint & footprint,
  const std::vector<CorridorSegment> & corridor,
  const FutureRoadRiskLimits & limits)
{
  const double radius = footprint_radius(footprint, limits);
  for (const auto & segment : corridor) {
    const double threshold = segment.maximum_width_m + radius;
    if (point_segment_squared_distance(
        footprint.pose, segment.start, segment.end) <=
      threshold * threshold)
    {
      return true;
    }
  }
  return false;
}

bool swept_intersects_corridor(
  const PredictedFootprint & first, const PredictedFootprint & second,
  const std::vector<CorridorSegment> & corridor,
  const FutureRoadRiskLimits & limits)
{
  const double radius = std::max(
    footprint_radius(first, limits),
    footprint_radius(second, limits));
  if (squared_distance(
      first.pose.x, first.pose.y, second.pose.x, second.pose.y) <=
    kGeometryEpsilon)
  {
    return point_intersects_corridor(first, corridor, limits) ||
           point_intersects_corridor(second, corridor, limits);
  }
  for (const auto & segment : corridor) {
    const double threshold = segment.maximum_width_m + radius;
    if (segment_squared_distance(
        first.pose, second.pose, segment.start, segment.end) <=
      threshold * threshold)
    {
      return true;
    }
  }
  return false;
}

}  // namespace

FutureRoadRiskState evaluate_future_road_risk(
  const ReferenceCorridor & corridor,
  const PredictedObjectSet & objects,
  const FutureRoadRiskLimits & limits)
{
  if (!finite(limits.maximum_horizon_s) ||
    !(limits.maximum_horizon_s > 0.0) ||
    limits.maximum_objects == 0U ||
    limits.maximum_footprints == 0U ||
    limits.maximum_corridor_segments == 0U ||
    !finite(limits.covariance_sigma) || limits.covariance_sigma < 0.0 ||
    !finite(limits.minimum_margin_m) || limits.minimum_margin_m < 0.0)
  {
    throw std::invalid_argument(
            "future-road-risk limits are invalid");
  }
  if (objects.size() > limits.maximum_objects) {
    throw std::invalid_argument(
            "future-road-risk object count exceeds the bound");
  }
  const auto segments = validate_corridor(corridor, limits);

  FutureRoadRiskState result;
  result.evaluated_object_count = objects.size();
  for (const auto & object : objects) {
    if (object.footprints.empty()) {
      throw std::invalid_argument(
              "future-road-risk object has no footprints");
    }
    if (object.footprints.size() >
      limits.maximum_footprints - result.evaluated_footprint_count)
    {
      throw std::invalid_argument(
              "future-road-risk footprint count exceeds the bound");
    }
    result.evaluated_footprint_count += object.footprints.size();
    for (std::size_t index = 0U; index < object.footprints.size(); ++index) {
      validate_footprint(object.footprints[index], limits);
      if (index > 0U &&
        !(object.footprints[index].time_from_start_s >
        object.footprints[index - 1U].time_from_start_s))
      {
        throw std::invalid_argument(
                "future-road-risk prediction times must be increasing");
      }
    }

    bool object_risky = false;
    double object_earliest_s = std::numeric_limits<double>::infinity();
    for (const auto & footprint : object.footprints) {
      if (footprint.time_from_start_s > 0.0 &&
        footprint.time_from_start_s <= limits.maximum_horizon_s &&
        point_intersects_corridor(footprint, segments, limits))
      {
        object_risky = true;
        object_earliest_s =
          std::min(object_earliest_s, footprint.time_from_start_s);
      }
    }
    for (std::size_t index = 1U;
      index < object.footprints.size(); ++index)
    {
      const auto & raw_first = object.footprints[index - 1U];
      const auto & raw_second = object.footprints[index];
      const double start_s = std::max(0.0, raw_first.time_from_start_s);
      const double end_s = std::min(
        limits.maximum_horizon_s, raw_second.time_from_start_s);
      if (!(end_s > start_s)) {
        continue;
      }
      const auto first = interpolate(raw_first, raw_second, start_s);
      const auto second = interpolate(raw_first, raw_second, end_s);
      validate_footprint(first, limits);
      validate_footprint(second, limits);
      if (swept_intersects_corridor(first, second, segments, limits)) {
        object_risky = true;
        object_earliest_s = std::min(object_earliest_s, start_s);
      }
    }
    if (object_risky) {
      result.risk = true;
      ++result.risky_object_count;
      if (!result.earliest_risk_time_s ||
        object_earliest_s < *result.earliest_risk_time_s)
      {
        result.earliest_risk_time_s = object_earliest_s;
      }
    }
  }
  return result;
}

}  // namespace ad_planner
