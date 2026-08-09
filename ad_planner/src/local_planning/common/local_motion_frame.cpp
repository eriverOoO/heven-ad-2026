#include "ad_planner/local_planning/common/local_motion_frame.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_planner
{
namespace
{

constexpr long double kPi = 3.141592653589793238462643383279502884L;

bool finite(const double value)
{
  return std::isfinite(value);
}

double checked_double(const long double value, const char * const name)
{
  if (!std::isfinite(value) || !std::isfinite(static_cast<double>(value))) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
  return static_cast<double>(value);
}

void require_finite(const double value, const char * const name)
{
  if (!finite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double checked_difference(const double lhs, const double rhs, const char * const name)
{
  return checked_double(static_cast<long double>(lhs) - static_cast<long double>(rhs), name);
}

double checked_sum(const double lhs, const double rhs, const char * const name)
{
  return checked_double(static_cast<long double>(lhs) + static_cast<long double>(rhs), name);
}

double wrap_yaw(const double yaw_rad)
{
  require_finite(yaw_rad, "yaw");
  return std::remainder(yaw_rad, static_cast<double>(2.0L * kPi));
}

void validate_pose(const Pose2 & pose)
{
  require_finite(pose.x, "pose x");
  require_finite(pose.y, "pose y");
  require_finite(pose.yaw_rad, "pose yaw");
}

void validate_transform(const FrameTransform2 & transform)
{
  require_finite(transform.x_m, "transform x");
  require_finite(transform.y_m, "transform y");
  require_finite(transform.yaw_rad, "transform yaw");
}

void validate_lane(const ReferenceLane & lane)
{
  if (lane.points.size() < 2U) {
    throw std::invalid_argument("reference lane must contain at least two points");
  }
  for (std::size_t index = 0U; index < lane.points.size(); ++index) {
    const auto & point = lane.points[index];
    validate_pose(point.pose);
    require_finite(point.route_s_m, "reference route station");
    require_finite(point.curvature_inv_m, "reference curvature");
    require_finite(point.left_width_m, "reference left width");
    require_finite(point.right_width_m, "reference right width");
    require_finite(point.speed_limit_mps, "reference speed limit");
    if (!(point.left_width_m > 0.0) || !(point.right_width_m > 0.0) ||
      point.speed_limit_mps < 0.0)
    {
      throw std::invalid_argument("reference lane has invalid dimensions or speed limit");
    }
    if (index == 0U) {
      continue;
    }
    const auto & previous = lane.points[index - 1U];
    const double route_delta = checked_difference(
      point.route_s_m, previous.route_s_m, "reference route station delta");
    if (!(route_delta > 0.0)) {
      throw std::invalid_argument("reference route station must be strictly increasing");
    }
    const double dx = checked_difference(point.pose.x, previous.pose.x, "reference segment x");
    const double dy = checked_difference(point.pose.y, previous.pose.y, "reference segment y");
    const double length = std::hypot(dx, dy);
    require_finite(length, "reference segment length");
    if (!(length > 0.0)) {
      throw std::invalid_argument("reference lane contains a zero-length geometry segment");
    }
  }
}

void validate_corridor(const ReferenceCorridor & corridor)
{
  if (corridor.frame_id.empty()) {
    throw std::invalid_argument("reference corridor frame must be named");
  }
  if (corridor.lanes.empty() || corridor.primary_lane_index >= corridor.lanes.size()) {
    throw std::invalid_argument("reference corridor primary lane index is invalid");
  }
  for (std::size_t lane_index = 0U; lane_index < corridor.lanes.size(); ++lane_index) {
    const auto & lane = corridor.lanes[lane_index];
    validate_lane(lane);
    const auto validate_adjacency = [&](const std::vector<std::size_t> & indices) {
        for (const std::size_t adjacent_index : indices) {
          if (adjacent_index >= corridor.lanes.size() || adjacent_index == lane_index) {
            throw std::invalid_argument("reference corridor adjacency index is invalid");
          }
        }
      };
    validate_adjacency(lane.left_lane_indices);
    validate_adjacency(lane.right_lane_indices);
  }
}

struct SegmentProjection
{
  double distance_m{0.0};
  double route_s_m{0.0};
  double lateral_distance_m{0.0};
  double heading_error_rad{0.0};
};

SegmentProjection project_on_segment(
  const ReferencePoint & first, const ReferencePoint & second, const Pose2 & ego_pose)
{
  const double dx = checked_difference(second.pose.x, first.pose.x, "primary segment x");
  const double dy = checked_difference(second.pose.y, first.pose.y, "primary segment y");
  const double length = std::hypot(dx, dy);
  require_finite(length, "primary segment length");
  if (!(length > 0.0)) {
    throw std::invalid_argument("primary lane contains a zero-length segment");
  }
  const double relative_x = checked_difference(ego_pose.x, first.pose.x, "ego relative x");
  const double relative_y = checked_difference(ego_pose.y, first.pose.y, "ego relative y");
  const double unit_x = dx / length;
  const double unit_y = dy / length;
  const double along = checked_double(
    static_cast<long double>(relative_x) * static_cast<long double>(unit_x) +
    static_cast<long double>(relative_y) * static_cast<long double>(unit_y), "primary projection");
  const double clamped_along = std::clamp(along, 0.0, length);
  const double lateral = checked_double(
    -static_cast<long double>(relative_x) * static_cast<long double>(unit_y) +
    static_cast<long double>(relative_y) * static_cast<long double>(unit_x), "primary lateral projection");
  const double distance = std::hypot(along - clamped_along, lateral);
  require_finite(distance, "primary projection distance");
  const double segment_heading = std::atan2(dy, dx);
  const double heading_error = std::abs(std::remainder(
      segment_heading - ego_pose.yaw_rad,
      static_cast<double>(2.0L * kPi)));
  require_finite(heading_error, "primary projection heading error");
  const double ratio = clamped_along / length;
  const double route_delta = checked_difference(
    second.route_s_m, first.route_s_m, "primary route station delta");
  const double route_s = checked_double(
    static_cast<long double>(first.route_s_m) +
    static_cast<long double>(ratio) * static_cast<long double>(route_delta),
    "projected route station");
  return SegmentProjection{
    distance, route_s, std::abs(lateral), heading_error};
}

std::pair<std::size_t, std::size_t> crop_indices(
  const ReferenceLane & lane, const double lower_station, const double upper_station)
{
  const auto & points = lane.points;
  if (upper_station < points.front().route_s_m) {
    return {0U, 1U};
  }
  if (lower_station > points.back().route_s_m) {
    return {points.size() - 2U, points.size() - 1U};
  }

  const auto first_at_or_after = [&](const double station) {
      return static_cast<std::size_t>(std::lower_bound(
          points.begin(), points.end(), station,
          [](const ReferencePoint & point, const double value) {
            return point.route_s_m < value;
          }) - points.begin());
    };
  const std::size_t lower_upper = first_at_or_after(lower_station);
  const std::size_t upper_upper = first_at_or_after(upper_station);
  const std::size_t start = lower_upper == 0U ? 0U : lower_upper - 1U;
  std::size_t end = upper_upper >= points.size() ? points.size() - 1U : upper_upper;
  if (end <= start) {
    end = start + 1U < points.size() ? start + 1U : start;
  }
  if (end == start) {
    return {points.size() - 2U, points.size() - 1U};
  }
  return {start, end};
}

}  // namespace

Pose2 transform_pose(const FrameTransform2 & transform, const Pose2 & pose)
{
  validate_transform(transform);
  validate_pose(pose);
  const double cosine = std::cos(transform.yaw_rad);
  const double sine = std::sin(transform.yaw_rad);
  require_finite(cosine, "transform cosine");
  require_finite(sine, "transform sine");
  Pose2 result;
  result.x = checked_double(
    static_cast<long double>(transform.x_m) +
    static_cast<long double>(cosine) * static_cast<long double>(pose.x) -
    static_cast<long double>(sine) * static_cast<long double>(pose.y), "transformed pose x");
  result.y = checked_double(
    static_cast<long double>(transform.y_m) +
    static_cast<long double>(sine) * static_cast<long double>(pose.x) +
    static_cast<long double>(cosine) * static_cast<long double>(pose.y), "transformed pose y");
  result.yaw_rad = wrap_yaw(checked_sum(transform.yaw_rad, pose.yaw_rad, "transformed pose yaw"));
  return result;
}

ReferenceCorridor transform_reference_corridor(
  const FrameTransform2 & transform, const ReferenceCorridor & corridor,
  const std::string & target_frame_id)
{
  validate_transform(transform);
  validate_corridor(corridor);
  if (target_frame_id.empty()) {
    throw std::invalid_argument("target frame must be named");
  }
  ReferenceCorridor result = corridor;
  result.frame_id = target_frame_id;
  for (auto & lane : result.lanes) {
    for (auto & point : lane.points) {
      point.pose = transform_pose(transform, point.pose);
    }
  }
  validate_corridor(result);
  return result;
}

OccupancyGrid transform_occupancy_grid_origin(
  const FrameTransform2 & transform, const OccupancyGrid & grid)
{
  OccupancyGrid result = grid;
  result.origin = transform_pose(transform, grid.origin);
  return result;
}

PrimaryRouteProjection project_primary_route(
  const ReferenceCorridor & corridor, const Pose2 & ego_pose)
{
  validate_corridor(corridor);
  validate_pose(ego_pose);
  const auto & primary_lane =
    corridor.lanes[corridor.primary_lane_index];
  SegmentProjection best;
  bool have_projection = false;
  constexpr double kForwardHeadingLimitRad =
    static_cast<double>(kPi * 0.5L);
  constexpr double kHeadingToleranceRad = 1.0e-12;
  for (std::size_t index = 1U; index < primary_lane.points.size(); ++index) {
    const SegmentProjection candidate = project_on_segment(
      primary_lane.points[index - 1U], primary_lane.points[index], ego_pose);
    if (candidate.heading_error_rad >
      kForwardHeadingLimitRad + kHeadingToleranceRad)
    {
      continue;
    }
    if (!have_projection ||
      candidate.distance_m < best.distance_m ||
      (candidate.distance_m == best.distance_m &&
      candidate.heading_error_rad < best.heading_error_rad))
    {
      best = candidate;
      have_projection = true;
    }
  }
  if (!have_projection) {
    throw std::invalid_argument(
            "ego pose has no forward-facing primary route segment");
  }
  return PrimaryRouteProjection{
    best.route_s_m, best.lateral_distance_m, best.heading_error_rad};
}

ReferenceCorridor window_reference_corridor(
  const ReferenceCorridor & corridor, const Pose2 & ego_pose,
  const double behind_m, const double ahead_m)
{
  validate_corridor(corridor);
  validate_pose(ego_pose);
  require_finite(behind_m, "window behind distance");
  require_finite(ahead_m, "window ahead distance");
  if (behind_m < 0.0 || ahead_m < 0.0) {
    throw std::invalid_argument("window distances must be nonnegative");
  }

  const auto best = project_primary_route(corridor, ego_pose);
  const double lower_station = checked_difference(
    best.route_s_m, behind_m, "window lower station");
  const double upper_station = checked_sum(
    best.route_s_m, ahead_m, "window upper station");
  if (lower_station > upper_station) {
    throw std::invalid_argument("window station interval is invalid");
  }

  ReferenceCorridor result = corridor;
  for (std::size_t lane_index = 0U; lane_index < corridor.lanes.size(); ++lane_index) {
    const auto & source_lane = corridor.lanes[lane_index];
    const auto [first, last] = crop_indices(source_lane, lower_station, upper_station);
    auto & target_points = result.lanes[lane_index].points;
    target_points.assign(
      source_lane.points.begin() + static_cast<std::ptrdiff_t>(first),
      source_lane.points.begin() + static_cast<std::ptrdiff_t>(last + 1U));
  }
  validate_corridor(result);
  return result;
}

}  // namespace ad_planner
