#include "ad_planner/local_planning/frenet/frenet_lattice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"
#include "ad_planner/local_planning/common/occupancy.hpp"

namespace ad_planner
{
namespace
{

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kGeometryTolerance = 1e-9;
constexpr double kScoreTieTolerance = 1e-12;
constexpr double kMaximumClearanceSearchM = 5.0;
constexpr std::size_t kMaximumSamplesPerCandidate = 10000U;

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

void require_positive(const double value, const char * const name)
{
  require_finite(value, name);
  if (!(value > 0.0)) {
    throw std::invalid_argument(std::string(name) + " must be greater than zero");
  }
}

void require_nonnegative(const double value, const char * const name)
{
  require_finite(value, name);
  if (value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must not be negative");
  }
}

double wrap_yaw(const double yaw_rad)
{
  require_finite(yaw_rad, "yaw");
  return std::remainder(yaw_rad, kTwoPi);
}

void validate_sample_array(
  const std::vector<double> & values, const char * const name,
  const bool strictly_positive)
{
  if (values.empty()) {
    throw std::invalid_argument(std::string(name) + " must not be empty");
  }
  for (const double value : values) {
    if (strictly_positive) {
      require_positive(value, name);
    } else {
      require_finite(value, name);
    }
  }
}

void validate_config(const FrenetLatticeConfig & config)
{
  validate_sample_array(config.lateral_targets_m, "lateral_targets_m", false);
  validate_sample_array(config.target_speeds_mps, "target_speeds_mps", false);
  for (const double speed : config.target_speeds_mps) {
    if (speed < 0.0) {
      throw std::invalid_argument("target_speeds_mps must not contain reverse speeds");
    }
  }
  validate_sample_array(config.durations_s, "durations_s", true);
  require_positive(config.sample_dt_s, "sample_dt_s");
  require_positive(config.maximum_curvature_inv_m, "maximum_curvature_inv_m");
  require_positive(config.maximum_acceleration_mps2, "maximum_acceleration_mps2");
  require_positive(
    config.maximum_lateral_acceleration_mps2,
    "maximum_lateral_acceleration_mps2");
  require_positive(config.maximum_jerk_mps3, "maximum_jerk_mps3");
  require_positive(config.maximum_lateral_transition_m, "maximum_lateral_transition_m");
  require_nonnegative(config.footprint_clearance_m, "footprint_clearance_m");
  if (config.occupied_threshold < 0 || config.occupied_threshold > 100) {
    throw std::invalid_argument("occupied_threshold must be between 0 and 100");
  }
  if (config.maximum_cells_to_check == 0U) {
    throw std::invalid_argument("maximum_cells_to_check must be greater than zero");
  }
  for (const auto & [value, name] : std::array<std::pair<double, const char *>, 6>{{
      {config.progress_weight, "progress_weight"},
      {config.clearance_weight, "clearance_weight"},
      {config.jerk_weight, "jerk_weight"},
      {config.lateral_offset_weight, "lateral_offset_weight"},
      {config.lane_change_weight, "lane_change_weight"},
      {config.continuity_weight, "continuity_weight"}}})
  {
    require_nonnegative(value, name);
  }
  if (config.maximum_candidates == 0U) {
    throw std::invalid_argument("maximum_candidates must be greater than zero");
  }
  for (const double duration_s : config.durations_s) {
    const double sample_count = std::ceil(duration_s / config.sample_dt_s) + 1.0;
    if (!finite(sample_count) ||
      sample_count > static_cast<double>(kMaximumSamplesPerCandidate))
    {
      throw std::invalid_argument("duration/sample_dt_s produces too many samples");
    }
  }
}

bool valid_pose(const Pose2 & pose)
{
  return finite(pose.x) && finite(pose.y) && finite(pose.yaw_rad);
}

void validate_constraints(const VehicleConstraints & constraints)
{
  require_positive(constraints.wheelbase_m, "wheelbase_m");
  require_positive(constraints.maximum_steering_rad, "maximum_steering_rad");
  if (!(constraints.maximum_steering_rad < 1.5707963267948966)) {
    throw std::invalid_argument("maximum_steering_rad must be below pi/2");
  }
  require_positive(constraints.maximum_speed_mps, "maximum_speed_mps");
  require_positive(constraints.maximum_acceleration_mps2, "maximum_acceleration_mps2");
  require_positive(constraints.maximum_deceleration_mps2, "maximum_deceleration_mps2");
  require_positive(
    constraints.maximum_lateral_acceleration_mps2,
    "maximum_lateral_acceleration_mps2");
  require_positive(constraints.maximum_jerk_mps3, "maximum_jerk_mps3");
  require_positive(constraints.footprint_front_m, "footprint_front_m");
  require_positive(constraints.footprint_rear_m, "footprint_rear_m");
  require_positive(constraints.footprint_half_width_m, "footprint_half_width_m");
}

void validate_prediction(const PredictedObjectSet & objects)
{
  for (const auto & object : objects) {
    if (object.footprints.empty()) {
      throw std::invalid_argument("predicted object must contain at least one footprint");
    }
    double previous_time_s = -std::numeric_limits<double>::infinity();
    for (const auto & footprint : object.footprints) {
      require_finite(footprint.time_from_start_s, "prediction time");
      if (!(footprint.time_from_start_s > previous_time_s)) {
        throw std::invalid_argument("prediction times must be strictly increasing");
      }
      if (!valid_pose(footprint.pose)) {
        throw std::invalid_argument("predicted object pose must be finite");
      }
      require_positive(footprint.length_m, "predicted object length");
      require_positive(footprint.width_m, "predicted object width");
      require_finite(footprint.covariance_xx, "prediction covariance_xx");
      require_finite(footprint.covariance_yy, "prediction covariance_yy");
      require_finite(footprint.covariance_xy, "prediction covariance_xy");
      const double covariance_midpoint =
        0.5 * footprint.covariance_xx + 0.5 * footprint.covariance_yy;
      const double covariance_radius = std::hypot(
        0.5 * footprint.covariance_xx - 0.5 * footprint.covariance_yy,
        footprint.covariance_xy);
      if (footprint.covariance_xx < 0.0 ||
        footprint.covariance_yy < 0.0 ||
        !std::isfinite(covariance_midpoint) ||
        !std::isfinite(covariance_radius) ||
        covariance_midpoint - covariance_radius < -1.0e-9)
      {
        throw std::invalid_argument(
                "prediction covariance must be positive semidefinite");
      }
      previous_time_s = footprint.time_from_start_s;
    }
  }
}

void validate_trajectory(const TimedTrajectory & trajectory, const std::string & frame_id)
{
  if (trajectory.frame_id != frame_id || trajectory.points.empty()) {
    throw std::invalid_argument("previous trajectory must be nonempty and share corridor frame");
  }
  double previous_time_s = -std::numeric_limits<double>::infinity();
  for (const auto & point : trajectory.points) {
    if (!valid_pose(point.pose) || !finite(point.speed_mps) || point.speed_mps < 0.0 ||
      !finite(point.curvature_inv_m) || !finite(point.time_from_start_s) ||
      point.time_from_start_s < 0.0 ||
      !(point.time_from_start_s > previous_time_s))
    {
      throw std::invalid_argument("previous trajectory is malformed");
    }
    previous_time_s = point.time_from_start_s;
  }
}

std::vector<std::size_t> candidate_lane_indices(const ReferenceCorridor & corridor)
{
  if (corridor.frame_id.empty() || corridor.lanes.empty() ||
    corridor.primary_lane_index >= corridor.lanes.size())
  {
    throw std::invalid_argument("reference corridor metadata is invalid");
  }

  const auto & primary = corridor.lanes[corridor.primary_lane_index];
  std::vector<std::size_t> result{corridor.primary_lane_index};
  std::set<std::size_t> seen_indices{corridor.primary_lane_index};
  auto append = [&](const std::vector<std::size_t> & indices) {
      for (const std::size_t index : indices) {
        if (index >= corridor.lanes.size() || index == corridor.primary_lane_index) {
          throw std::invalid_argument("reference corridor adjacency index is invalid");
        }
        if (!seen_indices.insert(index).second) {
          throw std::invalid_argument("reference corridor adjacency indices must be unique");
        }
        result.push_back(index);
      }
    };
  append(primary.left_lane_indices);
  append(primary.right_lane_indices);
  return result;
}

void validate_request(const LocalPlanningRequest & request)
{
  if (!valid_pose(request.ego.pose) || !finite(request.ego.speed_mps) ||
    request.ego.speed_mps < 0.0 || !finite(request.ego.yaw_rate_radps))
  {
    throw std::invalid_argument("ego state is invalid or reverse");
  }
  require_positive(request.dt_s, "request dt_s");
  if (!finite(request.previous_command.accel) ||
    !finite(request.previous_command.brake) ||
    !finite(request.previous_command.steering_rad))
  {
    throw std::invalid_argument("previous command is invalid");
  }
  validate_constraints(request.constraints);
  const auto grid_validation = validate_occupancy_grid(request.occupancy_grid);
  if (!grid_validation.valid) {
    throw std::invalid_argument(grid_validation.reason);
  }
  const auto lane_indices = candidate_lane_indices(request.reference_corridor);
  for (const std::size_t lane_index : lane_indices) {
    const auto & lane = request.reference_corridor.lanes[lane_index];
    if (lane.points.size() < 2U) {
      throw std::invalid_argument("reference lane must contain at least two points");
    }
    for (const auto & point : lane.points) {
      if (point.left_width_m < 0.0 || point.right_width_m < 0.0 ||
        point.speed_limit_mps < 0.0)
      {
        throw std::invalid_argument("reference lane constraints must not be negative");
      }
    }
    static_cast<void>(frenet_to_cartesian(
        lane, FrenetState{lane.points.front().route_s_m, 0.0, 0.0, 0.0, 0.0, 0.0},
        0.0));
  }
  static_cast<void>(project_to_frenet(
      request.reference_corridor.lanes[request.reference_corridor.primary_lane_index],
      request.ego));
  validate_prediction(request.predicted_objects);
  if (request.previous_trajectory) {
    validate_trajectory(*request.previous_trajectory, request.reference_corridor.frame_id);
  }
}

struct LaneSample
{
  double left_width_m{0.0};
  double right_width_m{0.0};
  double speed_limit_mps{0.0};
};

LaneSample sample_lane_constraints(const ReferenceLane & lane, const double route_s_m)
{
  if (route_s_m < lane.points.front().route_s_m - kGeometryTolerance ||
    route_s_m > lane.points.back().route_s_m + kGeometryTolerance)
  {
    throw std::out_of_range("candidate leaves reference lane");
  }
  const double clamped_s = std::clamp(
    route_s_m, lane.points.front().route_s_m, lane.points.back().route_s_m);
  const auto upper = std::upper_bound(
    lane.points.begin(), lane.points.end(), clamped_s,
    [](const double value, const ReferencePoint & point) {
      return value < point.route_s_m;
    });
  if (upper == lane.points.begin()) {
    return LaneSample{
      upper->left_width_m, upper->right_width_m, upper->speed_limit_mps};
  }
  if (upper == lane.points.end()) {
    const auto & point = lane.points.back();
    return LaneSample{point.left_width_m, point.right_width_m, point.speed_limit_mps};
  }
  const auto & next = *upper;
  const auto & previous = *(upper - 1);
  const double ratio =
    (clamped_s - previous.route_s_m) / (next.route_s_m - previous.route_s_m);
  return LaneSample{
    previous.left_width_m + ratio * (next.left_width_m - previous.left_width_m),
    previous.right_width_m + ratio * (next.right_width_m - previous.right_width_m),
    previous.speed_limit_mps + ratio * (next.speed_limit_mps - previous.speed_limit_mps)};
}

bool projection_is_endpoint_clamped(const ReferenceLane & lane, const FrenetState & state)
{
  const double span = lane.points.back().route_s_m - lane.points.front().route_s_m;
  const double tolerance = kGeometryTolerance * std::max(1.0, std::abs(span));
  return state.s_m <= lane.points.front().route_s_m + tolerance ||
         state.s_m >= lane.points.back().route_s_m - tolerance;
}

std::vector<double> sample_times(const double duration_s, const double dt_s)
{
  std::vector<double> times{0.0};
  times.reserve(static_cast<std::size_t>(std::ceil(duration_s / dt_s)) + 1U);
  const double tolerance =
    16.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::abs(duration_s));
  for (std::size_t index = 1U;; ++index) {
    const double time_s = static_cast<double>(index) * dt_s;
    if (!(time_s < duration_s - tolerance)) {
      break;
    }
    times.push_back(time_s);
  }
  times.push_back(duration_s);
  return times;
}

PredictedFootprint interpolate_prediction(
  const PredictedObject & object, const double time_s)
{
  const auto & footprints = object.footprints;
  if (time_s <= footprints.front().time_from_start_s) {
    return footprints.front();
  }
  if (time_s >= footprints.back().time_from_start_s) {
    return footprints.back();
  }
  const auto upper = std::upper_bound(
    footprints.begin(), footprints.end(), time_s,
    [](const double time, const PredictedFootprint & footprint) {
      return time < footprint.time_from_start_s;
    });
  const auto & next = *upper;
  const auto & previous = *(upper - 1);
  const double ratio =
    (time_s - previous.time_from_start_s) /
    (next.time_from_start_s - previous.time_from_start_s);
  const double yaw_delta = wrap_yaw(next.pose.yaw_rad - previous.pose.yaw_rad);
  PredictedFootprint result;
  result.time_from_start_s = time_s;
  result.pose.x = previous.pose.x + ratio * (next.pose.x - previous.pose.x);
  result.pose.y = previous.pose.y + ratio * (next.pose.y - previous.pose.y);
  result.pose.yaw_rad = wrap_yaw(previous.pose.yaw_rad + ratio * yaw_delta);
  result.length_m = previous.length_m + ratio * (next.length_m - previous.length_m);
  result.width_m = previous.width_m + ratio * (next.width_m - previous.width_m);
  result.covariance_xx =
    previous.covariance_xx + ratio * (next.covariance_xx - previous.covariance_xx);
  result.covariance_yy =
    previous.covariance_yy + ratio * (next.covariance_yy - previous.covariance_yy);
  result.covariance_xy =
    previous.covariance_xy + ratio * (next.covariance_xy - previous.covariance_xy);
  return result;
}

struct OrientedRectangle
{
  double center_x{0.0};
  double center_y{0.0};
  double yaw_rad{0.0};
  double half_length_m{0.0};
  double half_width_m{0.0};
};

bool overlaps_on_axis(
  const OrientedRectangle & first, const OrientedRectangle & second,
  const double axis_x, const double axis_y)
{
  const auto radius = [&](const OrientedRectangle & rectangle) {
      const double longitudinal_x = std::cos(rectangle.yaw_rad);
      const double longitudinal_y = std::sin(rectangle.yaw_rad);
      const double lateral_x = -longitudinal_y;
      const double lateral_y = longitudinal_x;
      return rectangle.half_length_m *
             std::abs(axis_x * longitudinal_x + axis_y * longitudinal_y) +
             rectangle.half_width_m *
             std::abs(axis_x * lateral_x + axis_y * lateral_y);
    };
  const double delta_x = second.center_x - first.center_x;
  const double delta_y = second.center_y - first.center_y;
  const double first_radius = radius(first);
  const double second_radius = radius(second);
  const double center_projection =
    std::abs(delta_x * axis_x + delta_y * axis_y);
  if (!finite(delta_x) || !finite(delta_y) || !finite(first_radius) ||
    !finite(second_radius) || !finite(center_projection))
  {
    return true;
  }
  return center_projection <= first_radius + second_radius + kGeometryTolerance;
}

bool rectangles_intersect(
  const OrientedRectangle & first, const OrientedRectangle & second)
{
  const std::array<double, 10> geometry{
    first.center_x, first.center_y, first.yaw_rad,
    first.half_length_m, first.half_width_m,
    second.center_x, second.center_y, second.yaw_rad,
    second.half_length_m, second.half_width_m};
  if (std::any_of(geometry.begin(), geometry.end(), [](const double value) {
      return !finite(value);
    }) ||
    first.half_length_m < 0.0 || first.half_width_m < 0.0 ||
    second.half_length_m < 0.0 || second.half_width_m < 0.0)
  {
    return true;
  }
  const std::array<double, 4> yaws{
    first.yaw_rad, first.yaw_rad + 1.5707963267948966,
    second.yaw_rad, second.yaw_rad + 1.5707963267948966};
  for (const double yaw : yaws) {
    if (!overlaps_on_axis(first, second, std::cos(yaw), std::sin(yaw))) {
      return false;
    }
  }
  return true;
}

OrientedRectangle ego_rectangle(
  const TimedTrajectoryPoint & point, const VehicleConstraints & constraints,
  const double clearance_m)
{
  const double center_offset_m =
    (constraints.footprint_front_m - constraints.footprint_rear_m) * 0.5;
  return OrientedRectangle{
    point.pose.x + center_offset_m * std::cos(point.pose.yaw_rad),
    point.pose.y + center_offset_m * std::sin(point.pose.yaw_rad),
    point.pose.yaw_rad,
    (constraints.footprint_front_m + constraints.footprint_rear_m) * 0.5 + clearance_m,
    constraints.footprint_half_width_m + clearance_m};
}

OrientedRectangle object_rectangle(
  const PredictedFootprint & footprint, const double clearance_m)
{
  const double covariance_midpoint =
    0.5 * footprint.covariance_xx + 0.5 * footprint.covariance_yy;
  const double covariance_radius = std::hypot(
    0.5 * footprint.covariance_xx - 0.5 * footprint.covariance_yy,
    footprint.covariance_xy);
  const double covariance_margin =
    2.0 * std::sqrt(
    std::max(0.0, covariance_midpoint + covariance_radius));
  const double x_margin = clearance_m + covariance_margin;
  const double y_margin = clearance_m + covariance_margin;
  return OrientedRectangle{
    footprint.pose.x, footprint.pose.y, footprint.pose.yaw_rad,
    footprint.length_m * 0.5 + x_margin,
    footprint.width_m * 0.5 + y_margin};
}

bool collides_with_prediction(
  const TimedTrajectoryPoint & point, const LocalPlanningRequest & request,
  const FrenetLatticeConfig & config)
{
  const auto ego = ego_rectangle(point, request.constraints, 0.0);
  for (const auto & object : request.predicted_objects) {
    const auto footprint = interpolate_prediction(object, point.time_from_start_s);
    if (rectangles_intersect(
        ego, object_rectangle(footprint, config.footprint_clearance_m)))
    {
      return true;
    }
  }
  return false;
}

double bounded_grid_clearance(
  const OccupancyGrid & grid, const Pose2 & pose, const std::int8_t occupied_threshold,
  const std::size_t maximum_cells_to_check)
{
  const auto center = world_to_cell(grid, Point3{pose.x, pose.y, 0.0});
  if (!center) {
    return 0.0;
  }
  const std::size_t maximum_radius_cells = static_cast<std::size_t>(
    std::ceil(kMaximumClearanceSearchM / grid.resolution));
  const std::size_t capacity_radius = static_cast<std::size_t>(
    std::floor((std::sqrt(static_cast<double>(maximum_cells_to_check)) - 1.0) * 0.5));
  const std::size_t radius = std::min(maximum_radius_cells, capacity_radius);
  double best_m = kMaximumClearanceSearchM;
  const std::size_t first_x = center->x > radius ? center->x - radius : 0U;
  const std::size_t first_y = center->y > radius ? center->y - radius : 0U;
  const std::size_t last_x = std::min(grid.width - 1U, center->x + radius);
  const std::size_t last_y = std::min(grid.height - 1U, center->y + radius);
  for (std::size_t y = first_y; y <= last_y; ++y) {
    for (std::size_t x = first_x; x <= last_x; ++x) {
      const std::int8_t value = grid.cells[y * grid.width + x];
      if (value >= 0 && value < occupied_threshold) {
        continue;
      }
      const double local_x = (static_cast<double>(x) + 0.5) * grid.resolution;
      const double local_y = (static_cast<double>(y) + 0.5) * grid.resolution;
      const double cosine = std::cos(grid.origin.yaw_rad);
      const double sine = std::sin(grid.origin.yaw_rad);
      const double world_x = grid.origin.x + cosine * local_x - sine * local_y;
      const double world_y = grid.origin.y + sine * local_x + cosine * local_y;
      best_m = std::min(best_m, std::hypot(world_x - pose.x, world_y - pose.y));
    }
  }
  return best_m;
}

TimedTrajectoryPoint interpolate_trajectory_point(
  const TimedTrajectory & trajectory, const double time_s)
{
  if (time_s <= trajectory.points.front().time_from_start_s) {
    return trajectory.points.front();
  }
  if (time_s >= trajectory.points.back().time_from_start_s) {
    return trajectory.points.back();
  }
  const auto upper = std::upper_bound(
    trajectory.points.begin(), trajectory.points.end(), time_s,
    [](const double time, const TimedTrajectoryPoint & point) {
      return time < point.time_from_start_s;
    });
  const auto & next = *upper;
  const auto & previous = *(upper - 1);
  const double ratio =
    (time_s - previous.time_from_start_s) /
    (next.time_from_start_s - previous.time_from_start_s);
  const double yaw_delta = wrap_yaw(next.pose.yaw_rad - previous.pose.yaw_rad);
  TimedTrajectoryPoint result;
  result.time_from_start_s = time_s;
  result.pose.x = previous.pose.x + ratio * (next.pose.x - previous.pose.x);
  result.pose.y = previous.pose.y + ratio * (next.pose.y - previous.pose.y);
  result.pose.yaw_rad = wrap_yaw(previous.pose.yaw_rad + ratio * yaw_delta);
  result.speed_mps = previous.speed_mps + ratio * (next.speed_mps - previous.speed_mps);
  result.curvature_inv_m =
    previous.curvature_inv_m +
    ratio * (next.curvature_inv_m - previous.curvature_inv_m);
  return result;
}

double continuity_cost(
  const TimedTrajectory & candidate, const std::optional<TimedTrajectory> & previous)
{
  if (!previous) {
    return 0.0;
  }
  double cost = 0.0;
  for (const auto & point : candidate.points) {
    const auto reference = interpolate_trajectory_point(*previous, point.time_from_start_s);
    const double dx = point.pose.x - reference.pose.x;
    const double dy = point.pose.y - reference.pose.y;
    const double dyaw = wrap_yaw(point.pose.yaw_rad - reference.pose.yaw_rad);
    cost += dx * dx + dy * dy + 0.1 * dyaw * dyaw;
  }
  return cost / static_cast<double>(candidate.points.size());
}

struct Candidate
{
  TimedTrajectory trajectory;
  double total_cost{0.0};
  double progress_cost{0.0};
  double clearance_cost{0.0};
  double jerk_cost{0.0};
  double lateral_offset_cost{0.0};
  double lane_change_cost{0.0};
  double continuity_cost{0.0};
};

std::optional<Candidate> generate_candidate(
  const LocalPlanningRequest & request, const FrenetLatticeConfig & config,
  const std::size_t lane_index, const double duration_s,
  const double target_speed_mps, const double lateral_target_m)
{
  const auto & lane = request.reference_corridor.lanes[lane_index];
  const bool lane_change = lane_index != request.reference_corridor.primary_lane_index;
  const FrenetState initial = project_to_frenet(lane, request.ego);
  if (lane_change &&
    (projection_is_endpoint_clamped(lane, initial) ||
    std::abs(initial.d_m) > config.maximum_lateral_transition_m))
  {
    return std::nullopt;
  }

  const QuarticPolynomial longitudinal(
    initial.s_m, initial.s_dot_mps, initial.s_ddot_mps2,
    target_speed_mps, 0.0, duration_s);
  const QuinticPolynomial lateral(
    initial.d_m, initial.d_dot_mps, initial.d_ddot_mps2,
    lateral_target_m, 0.0, 0.0, duration_s);

  const double curvature_limit = std::min(
    config.maximum_curvature_inv_m,
    std::tan(request.constraints.maximum_steering_rad) /
    request.constraints.wheelbase_m);
  const double lateral_acceleration_limit = std::min(
    config.maximum_lateral_acceleration_mps2,
    request.constraints.maximum_lateral_acceleration_mps2);
  const double jerk_limit = std::min(
    config.maximum_jerk_mps3, request.constraints.maximum_jerk_mps3);
  const FootprintConfig footprint{
    (request.constraints.footprint_front_m + request.constraints.footprint_rear_m) * 0.5,
    request.constraints.footprint_half_width_m,
    config.footprint_clearance_m,
    config.occupied_threshold,
    config.maximum_cells_to_check,
    (request.constraints.footprint_front_m - request.constraints.footprint_rear_m) * 0.5};

  Candidate candidate;
  candidate.trajectory.frame_id = request.reference_corridor.frame_id;
  const auto times = sample_times(duration_s, config.sample_dt_s);
  candidate.trajectory.points.reserve(times.size());
  double jerk_squared_sum = 0.0;
  double minimum_clearance_m = kMaximumClearanceSearchM;
  double terminal_s_m = initial.s_m;

  for (const double time_s : times) {
    FrenetState state;
    state.s_m = longitudinal.position(time_s);
    state.s_dot_mps = longitudinal.velocity(time_s);
    state.s_ddot_mps2 = longitudinal.acceleration(time_s);
    state.d_m = lateral.position(time_s);
    state.d_dot_mps = lateral.velocity(time_s);
    state.d_ddot_mps2 = lateral.acceleration(time_s);
    const double longitudinal_jerk = longitudinal.jerk(time_s);
    const double lateral_jerk = lateral.jerk(time_s);
    const std::array<double, 8> values{
      state.s_m, state.s_dot_mps, state.s_ddot_mps2,
      state.d_m, state.d_dot_mps, state.d_ddot_mps2,
      longitudinal_jerk, lateral_jerk};
    if (std::any_of(values.begin(), values.end(), [](const double value) {
        return !finite(value);
      }))
    {
      return std::nullopt;
    }
    if (state.s_dot_mps < -kGeometryTolerance) {
      return std::nullopt;
    }
    if (state.s_ddot_mps2 >
      std::min(
        config.maximum_acceleration_mps2,
        request.constraints.maximum_acceleration_mps2) + kGeometryTolerance ||
      state.s_ddot_mps2 <
      -std::min(
        config.maximum_acceleration_mps2,
        request.constraints.maximum_deceleration_mps2) - kGeometryTolerance)
    {
      return std::nullopt;
    }
    const double total_jerk = std::hypot(longitudinal_jerk, lateral_jerk);
    if (!finite(total_jerk) || total_jerk > jerk_limit + kGeometryTolerance) {
      return std::nullopt;
    }
    jerk_squared_sum += total_jerk * total_jerk;

    const LaneSample lane_sample = sample_lane_constraints(lane, state.s_m);
    const double lateral_margin =
      request.constraints.footprint_half_width_m + config.footprint_clearance_m;
    if (state.d_m + lateral_margin > lane_sample.left_width_m + kGeometryTolerance ||
      -state.d_m + lateral_margin > lane_sample.right_width_m + kGeometryTolerance)
    {
      return std::nullopt;
    }
    auto point = frenet_to_cartesian(lane, state, time_s);
    if (point.speed_mps > request.constraints.maximum_speed_mps + kGeometryTolerance ||
      point.speed_mps > lane_sample.speed_limit_mps + kGeometryTolerance)
    {
      return std::nullopt;
    }
    if (!footprint_is_safe(request.occupancy_grid, point.pose, footprint) ||
      collides_with_prediction(point, request, config))
    {
      return std::nullopt;
    }
    if (config.clearance_weight > 0.0) {
      minimum_clearance_m = std::min(
        minimum_clearance_m,
        bounded_grid_clearance(
          request.occupancy_grid, point.pose, config.occupied_threshold,
          config.maximum_cells_to_check));
    }
    terminal_s_m = state.s_m;
    candidate.trajectory.points.push_back(point);
  }

  if (candidate.trajectory.points.empty()) {
    return std::nullopt;
  }

  if (candidate.trajectory.points.size() > 1U) {
    for (std::size_t index = 1U; index < candidate.trajectory.points.size(); ++index) {
      auto & current = candidate.trajectory.points[index];
      const auto & previous = candidate.trajectory.points[index - 1U];
      const double distance_m =
        std::hypot(current.pose.x - previous.pose.x, current.pose.y - previous.pose.y);
      const double dt_s = current.time_from_start_s - previous.time_from_start_s;
      if (!finite(distance_m) || !finite(dt_s) || !(dt_s > 0.0)) {
        return std::nullopt;
      }
      double curvature = current.curvature_inv_m;
      if (distance_m > kGeometryTolerance) {
        curvature =
          wrap_yaw(current.pose.yaw_rad - previous.pose.yaw_rad) / distance_m;
      } else if (
        current.speed_mps > kGeometryTolerance ||
        previous.speed_mps > kGeometryTolerance)
      {
        return std::nullopt;
      }
      if (!finite(curvature) ||
        std::abs(curvature) > curvature_limit + kGeometryTolerance ||
        current.speed_mps * current.speed_mps * std::abs(curvature) >
        lateral_acceleration_limit + kGeometryTolerance)
      {
        return std::nullopt;
      }
      current.curvature_inv_m = curvature;
      const double measured_acceleration =
        (current.speed_mps - previous.speed_mps) / dt_s;
      if (measured_acceleration >
        std::min(
          config.maximum_acceleration_mps2,
          request.constraints.maximum_acceleration_mps2) + kGeometryTolerance ||
        measured_acceleration <
        -std::min(
          config.maximum_acceleration_mps2,
          request.constraints.maximum_deceleration_mps2) - kGeometryTolerance)
      {
        return std::nullopt;
      }
      if (index > 1U) {
        const double previous_dt_s =
          previous.time_from_start_s -
          candidate.trajectory.points[index - 2U].time_from_start_s;
        const double previous_acceleration =
          (previous.speed_mps - candidate.trajectory.points[index - 2U].speed_mps) /
          previous_dt_s;
        const double measured_jerk =
          (measured_acceleration - previous_acceleration) /
          (0.5 * (dt_s + previous_dt_s));
        if (!finite(measured_jerk) ||
          std::abs(measured_jerk) > jerk_limit + kGeometryTolerance)
        {
          return std::nullopt;
        }
      }
    }
    candidate.trajectory.points.front().curvature_inv_m =
      candidate.trajectory.points[1U].curvature_inv_m;
    if (candidate.trajectory.points.front().speed_mps *
      candidate.trajectory.points.front().speed_mps *
      std::abs(candidate.trajectory.points.front().curvature_inv_m) >
      lateral_acceleration_limit + kGeometryTolerance)
    {
      return std::nullopt;
    }
  }

  candidate.progress_cost = -(terminal_s_m - initial.s_m);
  candidate.clearance_cost =
    config.clearance_weight > 0.0 ? -minimum_clearance_m : 0.0;
  candidate.jerk_cost =
    jerk_squared_sum / static_cast<double>(candidate.trajectory.points.size());
  candidate.lateral_offset_cost = std::abs(lateral_target_m);
  candidate.lane_change_cost = lane_change ? 1.0 : 0.0;
  candidate.continuity_cost =
    continuity_cost(candidate.trajectory, request.previous_trajectory);
  candidate.total_cost =
    config.progress_weight * candidate.progress_cost +
    config.clearance_weight * candidate.clearance_cost +
    config.jerk_weight * candidate.jerk_cost +
    config.lateral_offset_weight * candidate.lateral_offset_cost +
    config.lane_change_weight * candidate.lane_change_cost +
    config.continuity_weight * candidate.continuity_cost;
  if (!finite(candidate.total_cost)) {
    return std::nullopt;
  }
  return candidate;
}

}  // namespace

FrenetLatticeBackend::FrenetLatticeBackend(FrenetLatticeConfig config)
: config_(std::move(config))
{
  validate_config(config_);
}

LocalPlanningResult FrenetLatticeBackend::plan(const LocalPlanningRequest & request)
{
  LocalPlanningResult result;
  try {
    validate_request(request);
    const auto lane_indices = candidate_lane_indices(request.reference_corridor);
    std::optional<Candidate> best_candidate;
    std::size_t generated_candidates = 0U;

    for (const std::size_t lane_index : lane_indices) {
      for (const double duration_s : config_.durations_s) {
        for (const double target_speed_mps : config_.target_speeds_mps) {
          for (const double lateral_target_m : config_.lateral_targets_m) {
            if (generated_candidates >= config_.maximum_candidates) {
              break;
            }
            ++generated_candidates;
            std::optional<Candidate> candidate;
            try {
              candidate = generate_candidate(
                request, config_, lane_index, duration_s,
                target_speed_mps, lateral_target_m);
            } catch (const std::invalid_argument &) {
              candidate = std::nullopt;
            } catch (const std::domain_error &) {
              candidate = std::nullopt;
            } catch (const std::out_of_range &) {
              candidate = std::nullopt;
            } catch (const std::overflow_error &) {
              candidate = std::nullopt;
            }
            if (!candidate) {
              continue;
            }
            result.candidate_trajectories.push_back(candidate->trajectory);
            if (!best_candidate ||
              candidate->total_cost <
              best_candidate->total_cost - kScoreTieTolerance)
            {
              best_candidate = std::move(candidate);
            }
          }
          if (generated_candidates >= config_.maximum_candidates) {
            break;
          }
        }
        if (generated_candidates >= config_.maximum_candidates) {
          break;
        }
      }
      if (generated_candidates >= config_.maximum_candidates) {
        break;
      }
    }

    if (!best_candidate) {
      result.candidate_trajectories.clear();
      result.reason = "no valid candidate";
      return result;
    }

    result.valid = true;
    result.reason = "ok";
    result.trajectory = best_candidate->trajectory;
    const std::size_t controllable_index =
      result.trajectory.points.size() > 1U ? 1U : 0U;
    result.desired_speed_mps =
      result.trajectory.points[controllable_index].speed_mps;
    result.desired_curvature_inv_m =
      result.trajectory.points[controllable_index].curvature_inv_m;
    result.costs = {
      PlannerCost{"progress", best_candidate->progress_cost},
      PlannerCost{"clearance", best_candidate->clearance_cost},
      PlannerCost{"jerk", best_candidate->jerk_cost},
      PlannerCost{"lateral_offset", best_candidate->lateral_offset_cost},
      PlannerCost{"lane_change", best_candidate->lane_change_cost},
      PlannerCost{"continuity", best_candidate->continuity_cost},
      PlannerCost{"total", best_candidate->total_cost}};
    return result;
  } catch (const std::invalid_argument & error) {
    result.reason = error.what();
    result.trajectory = TimedTrajectory{};
    result.candidate_trajectories.clear();
    result.direct_command.reset();
    return result;
  } catch (const std::domain_error & error) {
    result.reason = error.what();
    result.trajectory = TimedTrajectory{};
    result.candidate_trajectories.clear();
    result.direct_command.reset();
    return result;
  } catch (const std::out_of_range & error) {
    result.reason = error.what();
    result.trajectory = TimedTrajectory{};
    result.candidate_trajectories.clear();
    result.direct_command.reset();
    return result;
  } catch (const std::overflow_error & error) {
    result.reason = error.what();
    result.trajectory = TimedTrajectory{};
    result.candidate_trajectories.clear();
    result.direct_command.reset();
    return result;
  }
}

}  // namespace ad_planner
