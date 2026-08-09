#include "ad_planner/local_planning/dwa/dwa.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ad_planner
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kMaximumPredictionTimelinePoints = 10'000U;
constexpr std::size_t kMaximumPredictionTimelineBoxes = 100'000U;
constexpr std::size_t kMaximumPredictionRoadCellChecks = 1'000'000U;
constexpr std::size_t kMaximumPredictionCollisionChecks = 1'000'000U;
constexpr std::size_t kMaximumPredictionSweepSamples = 10'000U;

bool finite_pose(const Pose2 & pose)
{
  return std::isfinite(pose.x) && std::isfinite(pose.y) &&
         std::isfinite(pose.yaw_rad);
}

bool finite_point(const Point3 & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

double move_toward(double value, double target, double maximum_change)
{
  return value < target ? std::min(target, value + maximum_change) :
         std::max(target, value - maximum_change);
}

void validate_sampling_range(double minimum, double maximum, double step)
{
  if (minimum == maximum) {
    return;
  }
  if (!(minimum + step > minimum) || !(maximum - step < maximum)) {
    throw std::invalid_argument(
            "DWA sampling step cannot advance configured range");
  }
  const long double interval_count =
    (static_cast<long double>(maximum) - static_cast<long double>(minimum)) /
    static_cast<long double>(step);
  if (!std::isfinite(interval_count) || interval_count < 0.0L ||
    interval_count > static_cast<long double>(
      std::numeric_limits<std::size_t>::max() - 1U))
  {
    throw std::invalid_argument("DWA sampling count overflows");
  }
}

std::size_t simulation_steps(double horizon, double dt)
{
  const long double count = std::ceil(
    static_cast<long double>(horizon) /
    static_cast<long double>(dt));
  if (!std::isfinite(count) || count < 1.0L ||
    count >
    static_cast<long double>(std::numeric_limits<std::size_t>::max()))
  {
    throw std::invalid_argument("DWA simulation step count overflows");
  }
  return static_cast<std::size_t>(count);
}

std::optional<std::size_t> prediction_braking_steps(
  double maximum_speed_mps, const DwaConfig & config)
{
  const long double braking_duration_s =
    static_cast<long double>(maximum_speed_mps) /
    static_cast<long double>(config.emergency_deceleration_mps2) +
    static_cast<long double>(config.dt);
  const long double count = std::ceil(
    braking_duration_s / static_cast<long double>(config.dt));
  if (!std::isfinite(count) || count < 1.0L ||
    count >= static_cast<long double>(kMaximumPredictionTimelinePoints))
  {
    return std::nullopt;
  }
  return static_cast<std::size_t>(count);
}

std::vector<double> samples(double minimum, double maximum, double step)
{
  std::vector<double> result;
  for (double value = minimum;; ) {
    result.push_back(value);
    if (value >= maximum) {
      break;
    }
    const double next = std::min(maximum, value + step);
    if (!(next > value)) {
      break;
    }
    value = next;
  }
  return result;
}

void include_sample(
  std::vector<double> & values, double value, double minimum,
  double maximum)
{
  if (value < minimum || value > maximum) {
    return;
  }
  const auto duplicate =
    std::find_if(
    values.begin(), values.end(), [value](double existing) {
      return std::abs(existing - value) <=
      8.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(existing), std::abs(value)});
    });
  if (duplicate == values.end()) {
    values.push_back(value);
    std::sort(values.begin(), values.end());
  }
}

std::vector<Point3> unsafe_cell_centers(
  const OccupancyGrid & grid,
  std::int8_t occupied_threshold)
{
  std::vector<Point3> result;
  const double cosine = std::cos(grid.origin.yaw_rad);
  const double sine = std::sin(grid.origin.yaw_rad);
  for (std::size_t y = 0; y < grid.height; ++y) {
    for (std::size_t x = 0; x < grid.width; ++x) {
      const auto value = grid.cells[y * grid.width + x];
      if (value >= 0 && value < occupied_threshold) {
        continue;
      }
      const double local_x = (static_cast<double>(x) + 0.5) * grid.resolution;
      const double local_y = (static_cast<double>(y) + 0.5) * grid.resolution;
      result.push_back(
        Point3{grid.origin.x + cosine * local_x - sine * local_y,
          grid.origin.y + sine * local_x + cosine * local_y,
          0.0});
    }
  }
  return result;
}

double nearest_clearance(
  const std::vector<Point3> & unsafe, const Pose2 & pose,
  double maximum)
{
  double result = maximum;
  for (const auto & point : unsafe) {
    result = std::min(result, std::hypot(point.x - pose.x, point.y - pose.y));
  }
  return result;
}

double point_segment_distance(
  const Pose2 & point, const Point3 & first,
  const Point3 & second)
{
  const long double dx =
    static_cast<long double>(second.x) - static_cast<long double>(first.x);
  const long double dy =
    static_cast<long double>(second.y) - static_cast<long double>(first.y);
  const long double length = std::hypot(dx, dy);
  if (length <= static_cast<long double>(
      std::numeric_limits<double>::epsilon()))
  {
    return std::hypot(point.x - first.x, point.y - first.y);
  }
  const long double unit_x = dx / length;
  const long double unit_y = dy / length;
  const long double point_dx =
    static_cast<long double>(point.x) - static_cast<long double>(first.x);
  const long double point_dy =
    static_cast<long double>(point.y) - static_cast<long double>(first.y);
  const long double projection_distance = std::clamp(
    point_dx * unit_x + point_dy * unit_y, 0.0L, length);
  const long double distance = std::hypot(
    point_dx - projection_distance * unit_x,
    point_dy - projection_distance * unit_y);
  return static_cast<double>(std::min(
           distance,
           static_cast<long double>(std::numeric_limits<double>::max())));
}

double path_distance(
  const Pose2 & pose,
  const std::vector<Point3> & reference_path)
{
  if (reference_path.size() == 1U) {
    return std::hypot(
      pose.x - reference_path.front().x,
      pose.y - reference_path.front().y);
  }
  double result = std::numeric_limits<double>::infinity();
  for (std::size_t index = 1U; index < reference_path.size(); ++index) {
    result = std::min(
      result,
      point_segment_distance(
        pose, reference_path[index - 1U],
        reference_path[index]));
  }
  return result;
}

bool lateral_acceleration_is_safe(
  double speed_mps, double steering_rad,
  const DwaConfig & config)
{
  const double lateral_acceleration =
    speed_mps * speed_mps * std::tan(steering_rad) / config.wheelbase_m;
  return std::isfinite(lateral_acceleration) &&
         std::abs(lateral_acceleration) <=
         config.maximum_lateral_acceleration_mps2;
}

bool footprint_is_inside_grid(
  const OccupancyGrid & grid, const Pose2 & pose,
  const FootprintConfig & footprint)
{
  const long double grid_cosine = std::cos(grid.origin.yaw_rad);
  const long double grid_sine = std::sin(grid.origin.yaw_rad);
  const long double pose_cosine = std::cos(pose.yaw_rad);
  const long double pose_sine = std::sin(pose.yaw_rad);
  const long double center_x =
    static_cast<long double>(pose.x) +
    static_cast<long double>(footprint.center_offset_x_m) * pose_cosine;
  const long double center_y =
    static_cast<long double>(pose.y) +
    static_cast<long double>(footprint.center_offset_x_m) * pose_sine;
  const long double half_length =
    static_cast<long double>(footprint.half_length_m) +
    static_cast<long double>(footprint.clearance_m);
  const long double half_width =
    static_cast<long double>(footprint.half_width_m) +
    static_cast<long double>(footprint.clearance_m);
  const long double map_width =
    static_cast<long double>(grid.width) *
    static_cast<long double>(grid.resolution);
  const long double map_height =
    static_cast<long double>(grid.height) *
    static_cast<long double>(grid.resolution);
  for (const int length_sign : {-1, 1}) {
    for (const int width_sign : {-1, 1}) {
      const long double longitudinal =
        static_cast<long double>(length_sign) * half_length;
      const long double lateral =
        static_cast<long double>(width_sign) * half_width;
      const long double world_x =
        center_x + longitudinal * pose_cosine - lateral * pose_sine;
      const long double world_y =
        center_y + longitudinal * pose_sine + lateral * pose_cosine;
      const long double dx =
        world_x - static_cast<long double>(grid.origin.x);
      const long double dy =
        world_y - static_cast<long double>(grid.origin.y);
      const long double grid_x = grid_cosine * dx + grid_sine * dy;
      const long double grid_y = -grid_sine * dx + grid_cosine * dy;
      if (grid_x <= 0.0L || grid_y <= 0.0L ||
        grid_x >= map_width || grid_y >= map_height)
      {
        return false;
      }
    }
  }
  return true;
}

struct FootprintCellBounds
{
  std::size_t first_x{0U};
  std::size_t last_x{0U};
  std::size_t first_y{0U};
  std::size_t last_y{0U};
};

std::optional<FootprintCellBounds> footprint_cell_bounds(
  const OccupancyGrid & grid, const Pose2 & pose,
  const FootprintConfig & footprint)
{
  if (!finite_pose(pose) ||
    !std::isfinite(footprint.half_length_m) ||
    !std::isfinite(footprint.half_width_m) ||
    !std::isfinite(footprint.clearance_m) ||
    !std::isfinite(footprint.center_offset_x_m) ||
    footprint.half_length_m < 0.0 ||
    footprint.half_width_m < 0.0 ||
    footprint.clearance_m < 0.0 ||
    footprint.maximum_cells_to_check == 0U)
  {
    return std::nullopt;
  }

  const long double resolution = grid.resolution;
  const long double grid_yaw = grid.origin.yaw_rad;
  const long double relative_yaw =
    static_cast<long double>(pose.yaw_rad) - grid_yaw;
  const long double length_axis_x = std::cos(relative_yaw);
  const long double length_axis_y = std::sin(relative_yaw);
  const long double width_axis_x = -length_axis_y;
  const long double width_axis_y = length_axis_x;
  const long double grid_cosine = std::cos(grid_yaw);
  const long double grid_sine = std::sin(grid_yaw);
  const long double dx =
    static_cast<long double>(pose.x) - grid.origin.x;
  const long double dy =
    static_cast<long double>(pose.y) - grid.origin.y;
  const long double center_offset = footprint.center_offset_x_m;
  const long double center_x =
    grid_cosine * dx + grid_sine * dy +
    center_offset * length_axis_x;
  const long double center_y =
    -grid_sine * dx + grid_cosine * dy +
    center_offset * length_axis_y;
  const long double half_length =
    static_cast<long double>(footprint.half_length_m) +
    footprint.clearance_m;
  const long double half_width =
    static_cast<long double>(footprint.half_width_m) +
    footprint.clearance_m;
  const long double extent_x =
    half_length * std::abs(length_axis_x) +
    half_width * std::abs(width_axis_x);
  const long double extent_y =
    half_length * std::abs(length_axis_y) +
    half_width * std::abs(width_axis_y);
  const long double minimum_x = center_x - extent_x;
  const long double maximum_x = center_x + extent_x;
  const long double minimum_y = center_y - extent_y;
  const long double maximum_y = center_y + extent_y;
  const long double map_width =
    static_cast<long double>(grid.width) * resolution;
  const long double map_height =
    static_cast<long double>(grid.height) * resolution;
  const std::array<long double, 16U> geometry{
    resolution, relative_yaw, length_axis_x, length_axis_y,
    width_axis_x, width_axis_y, center_x, center_y,
    half_length, half_width, extent_x, extent_y,
    minimum_x, maximum_x, minimum_y, maximum_y};
  if (!std::all_of(
      geometry.begin(), geometry.end(),
      [](const long double value) {return std::isfinite(value);}) ||
    !std::isfinite(map_width) || !std::isfinite(map_height) ||
    minimum_x <= 0.0L || minimum_y <= 0.0L ||
    maximum_x >= map_width || maximum_y >= map_height)
  {
    return std::nullopt;
  }

  const long double minimum_grid_x = std::floor(
    std::nextafter(
      minimum_x / resolution,
      -std::numeric_limits<long double>::infinity()));
  const long double maximum_grid_x = std::floor(
    std::nextafter(
      maximum_x / resolution,
      std::numeric_limits<long double>::infinity()));
  const long double minimum_grid_y = std::floor(
    std::nextafter(
      minimum_y / resolution,
      -std::numeric_limits<long double>::infinity()));
  const long double maximum_grid_y = std::floor(
    std::nextafter(
      maximum_y / resolution,
      std::numeric_limits<long double>::infinity()));
  if (!std::isfinite(minimum_grid_x) ||
    !std::isfinite(maximum_grid_x) ||
    !std::isfinite(minimum_grid_y) ||
    !std::isfinite(maximum_grid_y) ||
    minimum_grid_x < 0.0L || minimum_grid_y < 0.0L ||
    maximum_grid_x >= static_cast<long double>(grid.width) ||
    maximum_grid_y >= static_cast<long double>(grid.height))
  {
    return std::nullopt;
  }
  const auto first_x = static_cast<std::size_t>(minimum_grid_x);
  const auto last_x = static_cast<std::size_t>(maximum_grid_x);
  const auto first_y = static_cast<std::size_t>(minimum_grid_y);
  const auto last_y = static_cast<std::size_t>(maximum_grid_y);
  const std::size_t columns = last_x - first_x + 1U;
  const std::size_t rows = last_y - first_y + 1U;
  if (rows == 0U ||
    columns > footprint.maximum_cells_to_check / rows ||
    columns * rows > footprint.maximum_cells_to_check)
  {
    return std::nullopt;
  }
  return FootprintCellBounds{first_x, last_x, first_y, last_y};
}

class UnsafeCellIntegral
{
public:
  UnsafeCellIntegral(
    const OccupancyGrid & grid, const std::int8_t occupied_threshold)
  : width_(grid.width), height_(grid.height), stride_(grid.width + 1U)
  {
    if (!validate_occupancy_grid(grid).valid ||
      occupied_threshold < 0 || occupied_threshold > 100 ||
      grid.width == std::numeric_limits<std::size_t>::max() ||
      grid.height == std::numeric_limits<std::size_t>::max() ||
      stride_ > std::numeric_limits<std::size_t>::max() /
      (grid.height + 1U))
    {
      throw std::invalid_argument(
              "DWA unsafe-cell integral geometry is invalid");
    }
    const std::size_t entry_count = stride_ * (grid.height + 1U);
    if (entry_count > prefix_.max_size()) {
      throw std::invalid_argument(
              "DWA unsafe-cell integral allocation overflows");
    }
    prefix_.assign(entry_count, 0U);
    for (std::size_t y = 0U; y < grid.height; ++y) {
      std::size_t row_unsafe = 0U;
      for (std::size_t x = 0U; x < grid.width; ++x) {
        const std::int8_t value = grid.cells[y * grid.width + x];
        if (value < 0 || value >= occupied_threshold) {
          ++row_unsafe;
        }
        prefix_[(y + 1U) * stride_ + x + 1U] =
          prefix_[y * stride_ + x + 1U] + row_unsafe;
      }
    }
  }

  bool contains_unsafe(const FootprintCellBounds & bounds) const
  {
    if (bounds.first_x > bounds.last_x ||
      bounds.first_y > bounds.last_y ||
      bounds.last_x >= width_ || bounds.last_y >= height_)
    {
      return true;
    }
    const std::size_t left = bounds.first_x;
    const std::size_t right = bounds.last_x + 1U;
    const std::size_t top = bounds.first_y;
    const std::size_t bottom = bounds.last_y + 1U;
    const std::size_t count =
      prefix_[bottom * stride_ + right] -
      prefix_[top * stride_ + right] -
      prefix_[bottom * stride_ + left] +
      prefix_[top * stride_ + left];
    return count > 0U;
  }

private:
  std::size_t width_{0U};
  std::size_t height_{0U};
  std::size_t stride_{0U};
  std::vector<std::size_t> prefix_;
};

bool cached_footprint_is_safe(
  const OccupancyGrid & grid, const Pose2 & pose,
  const FootprintConfig & footprint,
  const UnsafeCellIntegral & unsafe_integral)
{
  const auto bounds = footprint_cell_bounds(grid, pose, footprint);
  if (bounds.has_value() && !unsafe_integral.contains_unsafe(*bounds)) {
    return true;
  }
  return footprint_is_safe(grid, pose, footprint);
}

bool swept_pose_is_safe(
  const OccupancyGrid & grid, const Pose2 & from, const Pose2 & to,
  const FootprintConfig & footprint,
  const UnsafeCellIntegral & unsafe_integral)
{
  if (!finite_pose(from) || !finite_pose(to)) {
    return false;
  }
  const long double translation = std::hypot(
    static_cast<long double>(to.x) - static_cast<long double>(from.x),
    static_cast<long double>(to.y) - static_cast<long double>(from.y));
  const long double footprint_radius = std::hypot(
    std::abs(static_cast<long double>(footprint.center_offset_x_m)) +
    static_cast<long double>(footprint.half_length_m) +
    static_cast<long double>(footprint.clearance_m),
    static_cast<long double>(footprint.half_width_m) +
    static_cast<long double>(footprint.clearance_m));
  const long double rotation =
    footprint_radius * std::abs(
    static_cast<long double>(to.yaw_rad) -
    static_cast<long double>(from.yaw_rad));
  const long double maximum_sweep_step =
    static_cast<long double>(grid.resolution);
  const long double raw_checks =
    std::ceil((translation + rotation) / maximum_sweep_step);
  constexpr std::size_t kMaximumSweepChecks = 1'000'000U;
  if (!std::isfinite(raw_checks) || raw_checks < 0.0L ||
    raw_checks > static_cast<long double>(kMaximumSweepChecks))
  {
    return false;
  }
  const std::size_t checks =
    std::max<std::size_t>(1U, static_cast<std::size_t>(raw_checks));
  for (std::size_t index = 1U; index <= checks; ++index) {
    const double ratio =
      static_cast<double>(index) / static_cast<double>(checks);
    const Pose2 interpolated{
      from.x + ratio * (to.x - from.x),
      from.y + ratio * (to.y - from.y),
      from.yaw_rad + ratio * (to.yaw_rad - from.yaw_rad)};
    // The summed-area broad phase proves an empty footprint AABB in O(1).
    // Any non-empty, malformed, oversized, or boundary case falls back to the
    // exact polygon/cell test, preserving the original fail-closed result.
    if (!cached_footprint_is_safe(
        grid, interpolated, footprint, unsafe_integral))
    {
      return false;
    }
  }
  return true;
}

bool same_grid_geometry(
  const OccupancyGrid & lhs, const OccupancyGrid & rhs)
{
  return lhs.width == rhs.width &&
         lhs.height == rhs.height &&
         lhs.resolution == rhs.resolution &&
         lhs.origin.x == rhs.origin.x &&
         lhs.origin.y == rhs.origin.y &&
         lhs.origin.yaw_rad == rhs.origin.yaw_rad;
}

struct OrientedBox
{
  double center_x;
  double center_y;
  double yaw_rad;
  double half_length;
  double half_width;
};

bool finite_box(const OrientedBox & box)
{
  return std::isfinite(box.center_x) &&
         std::isfinite(box.center_y) &&
         std::isfinite(box.yaw_rad) &&
         std::isfinite(box.half_length) &&
         std::isfinite(box.half_width) &&
         box.half_length >= 0.0 &&
         box.half_width >= 0.0;
}

OrientedBox ego_box(
  const Pose2 & pose, const FootprintConfig & footprint)
{
  const double cosine = std::cos(pose.yaw_rad);
  const double sine = std::sin(pose.yaw_rad);
  return OrientedBox{
    pose.x + footprint.center_offset_x_m * cosine,
    pose.y + footprint.center_offset_x_m * sine,
    pose.yaw_rad,
    footprint.half_length_m + footprint.clearance_m,
    footprint.half_width_m + footprint.clearance_m};
}

bool boxes_intersect(const OrientedBox & first, const OrientedBox & second)
{
  if (!finite_box(first) || !finite_box(second)) {
    return true;
  }
  const double first_cosine = std::cos(first.yaw_rad);
  const double first_sine = std::sin(first.yaw_rad);
  const double second_cosine = std::cos(second.yaw_rad);
  const double second_sine = std::sin(second.yaw_rad);
  const std::array<std::array<double, 2>, 4> axes{{
    {first_cosine, first_sine},
    {-first_sine, first_cosine},
    {second_cosine, second_sine},
    {-second_sine, second_cosine},
  }};
  const long double center_dx =
    static_cast<long double>(second.center_x) -
    static_cast<long double>(first.center_x);
  const long double center_dy =
    static_cast<long double>(second.center_y) -
    static_cast<long double>(first.center_y);
  for (const auto & axis : axes) {
    const long double axis_x = axis[0];
    const long double axis_y = axis[1];
    const long double first_longitudinal_projection =
      std::abs(
      axis_x * static_cast<long double>(first_cosine) +
      axis_y * static_cast<long double>(first_sine));
    const long double first_lateral_projection =
      std::abs(
      axis_x * static_cast<long double>(-first_sine) +
      axis_y * static_cast<long double>(first_cosine));
    const long double second_longitudinal_projection =
      std::abs(
      axis_x * static_cast<long double>(second_cosine) +
      axis_y * static_cast<long double>(second_sine));
    const long double second_lateral_projection =
      std::abs(
      axis_x * static_cast<long double>(-second_sine) +
      axis_y * static_cast<long double>(second_cosine));
    const long double first_radius =
      static_cast<long double>(first.half_length) *
      first_longitudinal_projection +
      static_cast<long double>(first.half_width) *
      first_lateral_projection;
    const long double second_radius =
      static_cast<long double>(second.half_length) *
      second_longitudinal_projection +
      static_cast<long double>(second.half_width) *
      second_lateral_projection;
    const long double center_distance =
      std::abs(center_dx * axis_x + center_dy * axis_y);
    if (!std::isfinite(first_radius) || !std::isfinite(second_radius) ||
      !std::isfinite(center_distance))
    {
      return true;
    }
    const long double combined_radius = first_radius + second_radius;
    if (!std::isfinite(combined_radius)) {
      return true;
    }
    const long double scale = std::max(
      {1.0L, center_distance, combined_radius});
    const long double tolerance =
      32.0L * std::numeric_limits<double>::epsilon() * scale;
    if (center_distance > combined_radius + tolerance) {
      return false;
    }
  }
  return true;
}

bool predicted_objects_are_valid(const PredictedObjectSet & objects)
{
  for (const auto & object : objects) {
    if (object.footprints.empty()) {
      return false;
    }
    double previous_time = -std::numeric_limits<double>::infinity();
    for (const auto & footprint : object.footprints) {
      const double covariance_midpoint =
        0.5 * footprint.covariance_xx + 0.5 * footprint.covariance_yy;
      const double covariance_eigen_radius = std::hypot(
        0.5 * footprint.covariance_xx - 0.5 * footprint.covariance_yy,
        footprint.covariance_xy);
      if (!std::isfinite(footprint.time_from_start_s) ||
        !finite_pose(footprint.pose) ||
        !std::isfinite(footprint.length_m) ||
        !std::isfinite(footprint.width_m) ||
        !std::isfinite(footprint.covariance_xx) ||
        !std::isfinite(footprint.covariance_yy) ||
        !std::isfinite(footprint.covariance_xy) ||
        footprint.length_m <= 0.0 ||
        footprint.width_m <= 0.0 ||
        footprint.covariance_xx < 0.0 ||
        footprint.covariance_yy < 0.0 ||
        !std::isfinite(covariance_midpoint) ||
        !std::isfinite(covariance_eigen_radius) ||
        covariance_midpoint - covariance_eigen_radius < -1.0e-9 ||
        !(footprint.time_from_start_s > previous_time))
      {
        return false;
      }
      previous_time = footprint.time_from_start_s;
    }
  }
  return true;
}

std::optional<double> interpolate_finite(
  double first, double second, long double ratio)
{
  const long double result =
    static_cast<long double>(first) +
    ratio * (
    static_cast<long double>(second) -
    static_cast<long double>(first));
  if (!std::isfinite(result) ||
    result > static_cast<long double>(std::numeric_limits<double>::max()) ||
    result < -static_cast<long double>(std::numeric_limits<double>::max()))
  {
    return std::nullopt;
  }
  return static_cast<double>(result);
}

std::optional<OrientedBox> predicted_box_at(
  const PredictedObject & object, double time_s,
  const DwaConfig & config)
{
  if (!std::isfinite(time_s) || object.footprints.empty()) {
    return std::nullopt;
  }
  const double time_tolerance =
    64.0 * std::numeric_limits<double>::epsilon() *
    std::max(
    {1.0, std::abs(time_s),
      std::abs(object.footprints.front().time_from_start_s),
      std::abs(object.footprints.back().time_from_start_s)});
  if (time_s < object.footprints.front().time_from_start_s - time_tolerance ||
    time_s > object.footprints.back().time_from_start_s + time_tolerance)
  {
    return std::nullopt;
  }
  time_s = std::clamp(
    time_s,
    object.footprints.front().time_from_start_s,
    object.footprints.back().time_from_start_s);
  const PredictedFootprint * first = &object.footprints.front();
  const PredictedFootprint * second = first;
  long double ratio = 0.0L;
  if (time_s == object.footprints.back().time_from_start_s) {
    first = &object.footprints.back();
    second = first;
  } else if (time_s > object.footprints.front().time_from_start_s) {
    const auto upper = std::upper_bound(
      object.footprints.begin(), object.footprints.end(), time_s,
      [](double time, const PredictedFootprint & footprint) {
        return time < footprint.time_from_start_s;
      });
    second = &*upper;
    first = &*(upper - 1);
    ratio =
      (static_cast<long double>(time_s) -
      static_cast<long double>(first->time_from_start_s)) /
      (static_cast<long double>(second->time_from_start_s) -
      static_cast<long double>(first->time_from_start_s));
  }

  const auto x = interpolate_finite(first->pose.x, second->pose.x, ratio);
  const auto y = interpolate_finite(first->pose.y, second->pose.y, ratio);
  const auto length =
    interpolate_finite(first->length_m, second->length_m, ratio);
  const auto width =
    interpolate_finite(first->width_m, second->width_m, ratio);
  const auto covariance_xx = interpolate_finite(
    first->covariance_xx, second->covariance_xx, ratio);
  const auto covariance_yy = interpolate_finite(
    first->covariance_yy, second->covariance_yy, ratio);
  const auto covariance_xy = interpolate_finite(
    first->covariance_xy, second->covariance_xy, ratio);
  const long double yaw_delta = std::remainder(
    static_cast<long double>(second->pose.yaw_rad) -
    static_cast<long double>(first->pose.yaw_rad),
    2.0L * static_cast<long double>(kPi));
  const long double yaw_value =
    static_cast<long double>(first->pose.yaw_rad) + ratio * yaw_delta;
  if (!x || !y || !length || !width || !covariance_xx ||
    !covariance_yy || !covariance_xy || !std::isfinite(yaw_value) ||
    yaw_value > static_cast<long double>(std::numeric_limits<double>::max()) ||
    yaw_value < -static_cast<long double>(std::numeric_limits<double>::max()))
  {
    return std::nullopt;
  }
  const double covariance_midpoint =
    0.5 * *covariance_xx + 0.5 * *covariance_yy;
  const double covariance_eigen_radius = std::hypot(
    0.5 * *covariance_xx - 0.5 * *covariance_yy, *covariance_xy);
  const double maximum_covariance_eigenvalue =
    covariance_midpoint + covariance_eigen_radius;
  if (!std::isfinite(maximum_covariance_eigenvalue) ||
    covariance_midpoint - covariance_eigen_radius < -1.0e-9)
  {
    return std::nullopt;
  }
  const double covariance_radius =
    std::sqrt(std::max(0.0, maximum_covariance_eigenvalue));
  const double covariance_margin =
    config.prediction_covariance_sigma * covariance_radius;
  const double margin =
    std::max(config.prediction_minimum_margin_m, covariance_margin);
  const OrientedBox result{
    *x, *y, static_cast<double>(yaw_value),
    0.5 * *length + margin,
    0.5 * *width + margin};
  if (!finite_box(result)) {
    return std::nullopt;
  }
  return result;
}

bool box_overlaps_drivable_road(
  const OrientedBox & box, const OccupancyGrid & drivable_mask,
  std::size_t & remaining_cell_checks)
{
  if (!finite_box(box)) {
    return true;
  }
  const long double box_cosine = std::cos(box.yaw_rad);
  const long double box_sine = std::sin(box.yaw_rad);
  const long double grid_cosine = std::cos(drivable_mask.origin.yaw_rad);
  const long double grid_sine = std::sin(drivable_mask.origin.yaw_rad);
  long double minimum_grid_x = std::numeric_limits<long double>::infinity();
  long double maximum_grid_x = -std::numeric_limits<long double>::infinity();
  long double minimum_grid_y = std::numeric_limits<long double>::infinity();
  long double maximum_grid_y = -std::numeric_limits<long double>::infinity();
  for (const int length_sign : {-1, 1}) {
    for (const int width_sign : {-1, 1}) {
      const long double longitudinal =
        static_cast<long double>(length_sign) * box.half_length;
      const long double lateral =
        static_cast<long double>(width_sign) * box.half_width;
      const long double world_x =
        static_cast<long double>(box.center_x) +
        longitudinal * box_cosine - lateral * box_sine;
      const long double world_y =
        static_cast<long double>(box.center_y) +
        longitudinal * box_sine + lateral * box_cosine;
      const long double dx =
        world_x - static_cast<long double>(drivable_mask.origin.x);
      const long double dy =
        world_y - static_cast<long double>(drivable_mask.origin.y);
      const long double grid_x = grid_cosine * dx + grid_sine * dy;
      const long double grid_y = -grid_sine * dx + grid_cosine * dy;
      minimum_grid_x = std::min(minimum_grid_x, grid_x);
      maximum_grid_x = std::max(maximum_grid_x, grid_x);
      minimum_grid_y = std::min(minimum_grid_y, grid_y);
      maximum_grid_y = std::max(maximum_grid_y, grid_y);
    }
  }
  const long double map_width =
    static_cast<long double>(drivable_mask.width) *
    drivable_mask.resolution;
  const long double map_height =
    static_cast<long double>(drivable_mask.height) *
    drivable_mask.resolution;
  if (!std::isfinite(minimum_grid_x) || !std::isfinite(maximum_grid_x) ||
    !std::isfinite(minimum_grid_y) || !std::isfinite(maximum_grid_y) ||
    !std::isfinite(map_width) || !std::isfinite(map_height))
  {
    return true;
  }
  if (maximum_grid_x < 0.0L || maximum_grid_y < 0.0L ||
    minimum_grid_x > map_width || minimum_grid_y > map_height)
  {
    return false;
  }

  const long double resolution = drivable_mask.resolution;
  const auto bounded_cell_index = [resolution](
    long double coordinate, std::size_t count)
    {
      const long double clamped = std::clamp(
        coordinate, 0.0L,
        static_cast<long double>(count) * resolution);
      const long double raw_index = std::floor(clamped / resolution);
      if (raw_index <= 0.0L) {
        return std::size_t{0U};
      }
      if (raw_index >= static_cast<long double>(count)) {
        return count - 1U;
      }
      return static_cast<std::size_t>(raw_index);
    };
  const std::size_t minimum_x =
    bounded_cell_index(minimum_grid_x, drivable_mask.width);
  const std::size_t maximum_x =
    bounded_cell_index(maximum_grid_x, drivable_mask.width);
  const std::size_t minimum_y =
    bounded_cell_index(minimum_grid_y, drivable_mask.height);
  const std::size_t maximum_y =
    bounded_cell_index(maximum_grid_y, drivable_mask.height);
  const std::size_t columns = maximum_x - minimum_x + 1U;
  const std::size_t rows = maximum_y - minimum_y + 1U;
  if (columns > remaining_cell_checks / rows) {
    remaining_cell_checks = 0U;
    return true;
  }
  remaining_cell_checks -= columns * rows;
  const double cell_half_extent = 0.5 * drivable_mask.resolution;
  for (std::size_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::size_t x = minimum_x; x <= maximum_x; ++x) {
      if (drivable_mask.cells[y * drivable_mask.width + x] != 0) {
        continue;
      }
      const double local_x =
        (static_cast<double>(x) + 0.5) * drivable_mask.resolution;
      const double local_y =
        (static_cast<double>(y) + 0.5) * drivable_mask.resolution;
      const OrientedBox cell{
        drivable_mask.origin.x +
        std::cos(drivable_mask.origin.yaw_rad) * local_x -
        std::sin(drivable_mask.origin.yaw_rad) * local_y,
        drivable_mask.origin.y +
        std::sin(drivable_mask.origin.yaw_rad) * local_x +
        std::cos(drivable_mask.origin.yaw_rad) * local_y,
        drivable_mask.origin.yaw_rad,
        cell_half_extent,
        cell_half_extent};
      if (boxes_intersect(box, cell)) {
        return true;
      }
    }
  }
  return false;
}

bool prediction_horizon_is_covered(
  const PredictedObjectSet & objects, const double required_horizon_s)
{
  if (!std::isfinite(required_horizon_s) || required_horizon_s < 0.0) {
    return false;
  }
  for (const auto & object : objects) {
    if (object.footprints.empty()) {
      return false;
    }
    const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(
      {1.0, std::abs(required_horizon_s),
        std::abs(object.footprints.front().time_from_start_s),
        std::abs(object.footprints.back().time_from_start_s)});
    if (object.footprints.front().time_from_start_s > tolerance ||
      object.footprints.back().time_from_start_s <
      required_horizon_s - tolerance)
    {
      return false;
    }
  }
  return true;
}

std::optional<bool> prediction_touches_drivable_road(
  const PredictedObject & object,
  const OccupancyGrid & drivable_mask,
  const DwaConfig & config,
  const double required_horizon_s,
  std::size_t & remaining_cell_checks,
  std::size_t & remaining_samples)
{
  if (object.footprints.size() < 2U) {
    return std::nullopt;
  }
  for (std::size_t index = 1U; index < object.footprints.size(); ++index) {
    const double segment_start_s = std::max(
      0.0, object.footprints[index - 1U].time_from_start_s);
    const double segment_end_s = std::min(
      required_horizon_s, object.footprints[index].time_from_start_s);
    if (segment_end_s < segment_start_s) {
      continue;
    }
    const auto start_box = predicted_box_at(object, segment_start_s, config);
    const auto end_box = predicted_box_at(object, segment_end_s, config);
    if (!start_box || !end_box) {
      return std::nullopt;
    }
    const double center_motion = std::hypot(
      end_box->center_x - start_box->center_x,
      end_box->center_y - start_box->center_y);
    const double footprint_radius = std::max(
      std::hypot(start_box->half_length, start_box->half_width),
      std::hypot(end_box->half_length, end_box->half_width));
    const double rotation_motion =
      footprint_radius * std::abs(
      std::remainder(end_box->yaw_rad - start_box->yaw_rad, 2.0 * kPi));
    const double raw_samples = std::ceil(
      (center_motion + rotation_motion) / drivable_mask.resolution);
    if (!std::isfinite(raw_samples) || raw_samples < 0.0 ||
      raw_samples >
      static_cast<double>(kMaximumPredictionSweepSamples))
    {
      return std::nullopt;
    }
    const std::size_t samples = std::max<std::size_t>(
      1U, static_cast<std::size_t>(raw_samples));
    if (samples == std::numeric_limits<std::size_t>::max() ||
      samples + 1U > remaining_samples)
    {
      return std::nullopt;
    }
    remaining_samples -= samples + 1U;
    for (std::size_t sample = 0U; sample <= samples; ++sample) {
      const double ratio =
        static_cast<double>(sample) / static_cast<double>(samples);
      const double time_s =
        segment_start_s + ratio * (segment_end_s - segment_start_s);
      const auto box = predicted_box_at(object, time_s, config);
      if (!box) {
        return std::nullopt;
      }
      if (box_overlaps_drivable_road(
          *box, drivable_mask, remaining_cell_checks))
      {
        return true;
      }
    }
  }
  return false;
}

std::optional<PredictedObjectSet> road_relevant_predictions(
  const PredictedObjectSet & objects,
  const OccupancyGrid & drivable_mask,
  const DwaConfig & config,
  const double required_horizon_s)
{
  PredictedObjectSet relevant;
  relevant.reserve(objects.size());
  std::size_t remaining_cell_checks = kMaximumPredictionRoadCellChecks;
  std::size_t remaining_samples = kMaximumPredictionTimelineBoxes;
  for (const auto & object : objects) {
    const auto touches = prediction_touches_drivable_road(
      object, drivable_mask, config, required_horizon_s,
      remaining_cell_checks, remaining_samples);
    if (!touches) {
      return std::nullopt;
    }
    if (*touches) {
      relevant.push_back(object);
    }
  }
  return relevant;
}

Pose2 interpolate_pose(
  const Pose2 & start, const Pose2 & end, const double ratio)
{
  const double yaw_delta =
    std::remainder(end.yaw_rad - start.yaw_rad, 2.0 * kPi);
  return Pose2{
    start.x + ratio * (end.x - start.x),
    start.y + ratio * (end.y - start.y),
    start.yaw_rad + ratio * yaw_delta};
}

bool predicted_sweep_is_safe(
  const Pose2 & ego_start, const Pose2 & ego_end,
  const double start_time_s, const double end_time_s,
  const PredictedObjectSet & objects,
  const DwaConfig & config,
  const double maximum_sweep_step_m,
  std::size_t & remaining_collision_checks)
{
  if (!finite_pose(ego_start) || !finite_pose(ego_end) ||
    !std::isfinite(start_time_s) || !std::isfinite(end_time_s) ||
    end_time_s < start_time_s || !std::isfinite(maximum_sweep_step_m) ||
    maximum_sweep_step_m <= 0.0)
  {
    return false;
  }
  const OrientedBox ego_start_box = ego_box(ego_start, config.footprint);
  const OrientedBox ego_end_box = ego_box(ego_end, config.footprint);
  double ego_sweep_motion = std::hypot(
    ego_end_box.center_x - ego_start_box.center_x,
    ego_end_box.center_y - ego_start_box.center_y);
  const double ego_radius = std::max(
    std::hypot(ego_start_box.half_length, ego_start_box.half_width),
    std::hypot(ego_end_box.half_length, ego_end_box.half_width));
  ego_sweep_motion += ego_radius * std::abs(
    std::remainder(
      ego_end_box.yaw_rad - ego_start_box.yaw_rad, 2.0 * kPi));
  double maximum_object_motion = 0.0;
  for (const auto & object : objects) {
    const auto object_start = predicted_box_at(object, start_time_s, config);
    const auto object_end = predicted_box_at(object, end_time_s, config);
    if (!object_start || !object_end) {
      return false;
    }
    const double object_radius = std::max(
      std::hypot(object_start->half_length, object_start->half_width),
      std::hypot(object_end->half_length, object_end->half_width));
    const double object_motion = std::hypot(
      object_end->center_x - object_start->center_x,
      object_end->center_y - object_start->center_y) +
      object_radius * std::abs(
      std::remainder(
        object_end->yaw_rad - object_start->yaw_rad, 2.0 * kPi));
    maximum_object_motion =
      std::max(maximum_object_motion, object_motion);
  }
  const double maximum_relative_motion =
    ego_sweep_motion + maximum_object_motion;
  const double raw_samples =
    std::ceil(maximum_relative_motion / maximum_sweep_step_m);
  if (!std::isfinite(raw_samples) || raw_samples < 0.0 ||
    raw_samples > static_cast<double>(kMaximumPredictionSweepSamples))
  {
    return false;
  }
  const std::size_t samples = std::max<std::size_t>(
    1U, static_cast<std::size_t>(raw_samples));
  if (samples == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t samples_with_endpoints = samples + 1U;
  if (!objects.empty() &&
    samples_with_endpoints >
    remaining_collision_checks / objects.size())
  {
    remaining_collision_checks = 0U;
    return false;
  }
  remaining_collision_checks -= samples_with_endpoints * objects.size();
  for (std::size_t sample = 0U; sample <= samples; ++sample) {
    const double ratio =
      static_cast<double>(sample) / static_cast<double>(samples);
    const Pose2 ego = interpolate_pose(ego_start, ego_end, ratio);
    const OrientedBox vehicle = ego_box(ego, config.footprint);
    if (!finite_box(vehicle)) {
      return false;
    }
    const double time_s =
      start_time_s + ratio * (end_time_s - start_time_s);
    for (const auto & object : objects) {
      const auto prediction = predicted_box_at(object, time_s, config);
      if (!prediction || boxes_intersect(vehicle, *prediction)) {
        return false;
      }
    }
  }
  return true;
}
} // namespace

DwaController::DwaController(DwaConfig config)
: base_config_(config), config_(config), pid_(config_.pid)
{
  const double values[] = {config_.min_speed_mps,
    config_.max_speed_mps,
    config_.speed_step_mps,
    config_.min_steer_rad,
    config_.sampled_max_steer_rad,
    config_.steer_step_rad,
    config_.dt,
    config_.horizon_s,
    config_.wheelbase_m,
    config_.max_steer_rad,
    config_.control_period_s,
    config_.dynamic_window_time_s,
    config_.maximum_acceleration_mps2,
    config_.maximum_deceleration_mps2,
    config_.emergency_deceleration_mps2,
    config_.initial_inflation_escape_s,
    config_.maximum_steering_rate_radps,
    config_.maximum_lateral_acceleration_mps2,
    config_.clearance_saturation_m,
    config_.maximum_path_distance_m,
    config_.prediction_covariance_sigma,
    config_.prediction_minimum_margin_m,
    config_.progress_weight,
    config_.goal_weight,
    config_.heading_weight,
    config_.clearance_weight,
    config_.smoothness_weight,
    config_.path_distance_weight,
    config_.speed_weight};
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("DWA configuration must be finite");
    }
  }
  if (config_.min_speed_mps < 0.0 ||
    config_.max_speed_mps < config_.min_speed_mps ||
    config_.speed_step_mps <= 0.0 ||
    config_.sampled_max_steer_rad < config_.min_steer_rad ||
    config_.steer_step_rad <= 0.0 || config_.dt <= 0.0 ||
    config_.horizon_s <= 0.0 || config_.wheelbase_m <= 0.0 ||
    config_.max_steer_rad <= 0.0 || config_.control_period_s <= 0.0 ||
    config_.dynamic_window_time_s <= 0.0 ||
    config_.maximum_acceleration_mps2 <= 0.0 ||
    config_.maximum_deceleration_mps2 <= 0.0 ||
    config_.emergency_deceleration_mps2 <= 0.0 ||
    config_.initial_inflation_escape_s <= 0.0 ||
    config_.initial_inflation_escape_s > config_.horizon_s ||
    config_.maximum_steering_rate_radps <= 0.0 ||
    config_.maximum_lateral_acceleration_mps2 <= 0.0 ||
    config_.clearance_saturation_m <= 0.0 ||
    config_.maximum_path_distance_m <= 0.0 ||
    config_.prediction_covariance_sigma < 0.0 ||
    config_.prediction_minimum_margin_m < 0.0 ||
    config_.min_steer_rad < -config_.max_steer_rad ||
    config_.sampled_max_steer_rad > config_.max_steer_rad ||
    config_.progress_weight < 0.0 || config_.goal_weight < 0.0 ||
    config_.heading_weight < 0.0 || config_.clearance_weight < 0.0 ||
    config_.smoothness_weight < 0.0 || config_.path_distance_weight < 0.0 ||
    config_.speed_weight < 0.0)
  {
    throw std::invalid_argument("DWA configuration has invalid ranges");
  }
  if (!std::isfinite(config_.footprint.half_length_m) ||
    !std::isfinite(config_.footprint.half_width_m) ||
    !std::isfinite(config_.footprint.clearance_m) ||
    !std::isfinite(config_.footprint.center_offset_x_m) ||
    config_.footprint.half_length_m < 0.0 ||
    config_.footprint.half_width_m < 0.0 ||
    config_.footprint.clearance_m < 0.0 ||
    config_.footprint.occupied_threshold < 0 ||
    config_.footprint.occupied_threshold > 100 ||
    config_.footprint.maximum_cells_to_check == 0)
  {
    throw std::invalid_argument("DWA footprint configuration is invalid");
  }
  validate_sampling_range(
    config_.min_speed_mps, config_.max_speed_mps,
    config_.speed_step_mps);
  validate_sampling_range(
    config_.min_steer_rad, config_.sampled_max_steer_rad,
    config_.steer_step_rad);
  static_cast<void>(simulation_steps(config_.horizon_s, config_.dt));
  static_cast<void>(simulation_steps(
    config_.initial_inflation_escape_s, config_.dt));
  static_cast<void>(simulation_steps(
    config_.max_speed_mps / config_.emergency_deceleration_mps2 + config_.dt,
    config_.dt));
}

void DwaController::apply_vehicle_constraints(
  const VehicleConstraints & constraints)
{
  const double values[] = {
    constraints.wheelbase_m,
    constraints.maximum_steering_rad,
    constraints.maximum_speed_mps,
    constraints.maximum_acceleration_mps2,
    constraints.maximum_deceleration_mps2,
    constraints.maximum_lateral_acceleration_mps2,
    constraints.maximum_jerk_mps3,
    constraints.footprint_front_m,
    constraints.footprint_rear_m,
    constraints.footprint_half_width_m};
  if (std::any_of(
      std::begin(values), std::end(values),
      [](const double value) {
        return !std::isfinite(value) || value <= 0.0;
      }))
  {
    throw std::invalid_argument(
            "DWA vehicle constraints must be finite and positive");
  }

  config_ = base_config_;
  config_.wheelbase_m = constraints.wheelbase_m;
  config_.max_steer_rad = std::min(
    config_.max_steer_rad, constraints.maximum_steering_rad);
  config_.sampled_max_steer_rad = std::min(
    config_.sampled_max_steer_rad, config_.max_steer_rad);
  config_.min_steer_rad = std::max(
    config_.min_steer_rad, -config_.max_steer_rad);
  config_.max_speed_mps = std::min(
    config_.max_speed_mps, constraints.maximum_speed_mps);
  config_.maximum_acceleration_mps2 = std::min(
    config_.maximum_acceleration_mps2,
    constraints.maximum_acceleration_mps2);
  config_.maximum_deceleration_mps2 = std::min(
    config_.maximum_deceleration_mps2,
    constraints.maximum_deceleration_mps2);
  config_.emergency_deceleration_mps2 = std::min(
    config_.emergency_deceleration_mps2,
    constraints.maximum_deceleration_mps2);
  config_.maximum_lateral_acceleration_mps2 = std::min(
    config_.maximum_lateral_acceleration_mps2,
    constraints.maximum_lateral_acceleration_mps2);
  config_.footprint.half_length_m =
    (constraints.footprint_front_m + constraints.footprint_rear_m) * 0.5;
  config_.footprint.center_offset_x_m =
    (constraints.footprint_front_m - constraints.footprint_rear_m) * 0.5;
  config_.footprint.half_width_m = constraints.footprint_half_width_m;

  if (config_.max_speed_mps < config_.min_speed_mps ||
    config_.sampled_max_steer_rad < config_.min_steer_rad)
  {
    throw std::invalid_argument(
            "DWA vehicle constraints exclude the configured sampling range");
  }
}

ControllerResult DwaController::plan(
  const OccupancyGrid & grid,
  const Pose2 & pose, const Point3 & target,
  double current_speed_mps,
  double previous_steering_rad,
  int behavior_id, int gear_id)
{
  return plan_with_reference(
    grid, pose, target, {}, current_speed_mps,
    previous_steering_rad, behavior_id, gear_id, nullptr, nullptr);
}

ControllerResult DwaController::plan_with_reference(
  const OccupancyGrid & grid, const Pose2 & pose, const Point3 & target,
  const std::vector<Point3> & reference_path, double current_speed_mps,
  double previous_steering_rad, int behavior_id, int gear_id,
  const OccupancyGrid * drivable_mask,
  const PredictedObjectSet * predicted_objects)
{
  candidate_trajectories_.clear();
  if (!validate_occupancy_grid(grid).valid || !finite_pose(pose) ||
    !finite_point(target) || !std::isfinite(current_speed_mps) ||
    current_speed_mps < 0.0 || !std::isfinite(previous_steering_rad) ||
    std::any_of(
      reference_path.begin(), reference_path.end(),
      [](const Point3 & point) {return !finite_point(point);}))
  {
    return ControllerResult{};
  }
  if (predicted_objects != nullptr && !predicted_objects->empty() &&
    drivable_mask == nullptr)
  {
    ControllerResult result;
    result.reason = "predicted objects require drivable mask";
    return result;
  }
  if (predicted_objects != nullptr &&
    !predicted_objects_are_valid(*predicted_objects))
  {
    ControllerResult result;
    result.reason = "malformed predicted objects";
    return result;
  }
  auto lethal_footprint = config_.footprint;
  lethal_footprint.occupied_threshold = 100;
  const UnsafeCellIntegral occupied_integral{
    grid, config_.footprint.occupied_threshold};
  const UnsafeCellIntegral lethal_integral{
    grid, lethal_footprint.occupied_threshold};
  std::optional<UnsafeCellIntegral> drivable_integral;
  if (drivable_mask != nullptr) {
    if (!validate_occupancy_grid(*drivable_mask).valid ||
      !same_grid_geometry(grid, *drivable_mask))
    {
      ControllerResult result;
      result.reason = "drivable mask geometry mismatch";
      return result;
    }
    drivable_integral.emplace(
      *drivable_mask, config_.footprint.occupied_threshold);
    if (!cached_footprint_is_safe(
        *drivable_mask, pose, config_.footprint, *drivable_integral))
    {
      ControllerResult result;
      result.reason = "initial footprint outside drivable mask";
      return result;
    }
  }
  if (!footprint_is_inside_grid(grid, pose, config_.footprint)) {
    ControllerResult result;
    result.reason = "initial footprint is unsafe";
    return result;
  }
  const std::size_t steps = simulation_steps(config_.horizon_s, config_.dt);
  PredictedObjectSet relevant_predictions;
  if (predicted_objects != nullptr && !predicted_objects->empty()) {
    const double maximum_prediction_speed =
      std::max(config_.max_speed_mps, current_speed_mps);
    const auto maximum_braking_steps =
      prediction_braking_steps(maximum_prediction_speed, config_);
    if (!maximum_braking_steps ||
      steps >= kMaximumPredictionTimelinePoints ||
      *maximum_braking_steps >
      kMaximumPredictionTimelinePoints - 1U - steps)
    {
      ControllerResult result;
      result.reason = "prediction timeline overflows";
      return result;
    }
    const double required_prediction_horizon_s =
      static_cast<double>(steps + *maximum_braking_steps) * config_.dt;
    if (!std::isfinite(required_prediction_horizon_s) ||
      !prediction_horizon_is_covered(
        *predicted_objects, required_prediction_horizon_s))
    {
      double minimum_available_horizon_s =
        std::numeric_limits<double>::infinity();
      for (const auto & object : *predicted_objects) {
        if (!object.footprints.empty()) {
          minimum_available_horizon_s = std::min(
            minimum_available_horizon_s,
            object.footprints.back().time_from_start_s);
        }
      }
      std::ostringstream reason;
      reason << "prediction horizon is shorter than rollout and braking"
             << " (required=" << required_prediction_horizon_s
             << " s, available=" << minimum_available_horizon_s << " s)";
      ControllerResult result;
      result.reason = reason.str();
      return result;
    }
    auto filtered = road_relevant_predictions(
      *predicted_objects, *drivable_mask, config_,
      required_prediction_horizon_s);
    if (!filtered) {
      ControllerResult result;
      result.reason = "malformed predicted objects";
      return result;
    }
    relevant_predictions = std::move(*filtered);
    std::size_t initial_collision_checks =
      kMaximumPredictionCollisionChecks;
    if (!predicted_sweep_is_safe(
        pose, pose, 0.0, 0.0, relevant_predictions, config_,
        drivable_mask->resolution, initial_collision_checks))
    {
      ControllerResult result;
      result.reason = "initial footprint intersects predicted object";
      return result;
    }
  }

  std::vector<Point3> effective_reference = reference_path;
  if (effective_reference.empty()) {
    effective_reference = {Point3{pose.x, pose.y, 0.0}, target};
  } else if (effective_reference.size() == 1U) {
    effective_reference.insert(
      effective_reference.begin(), Point3{pose.x, pose.y, 0.0});
  }
  const auto unsafe =
    unsafe_cell_centers(grid, config_.footprint.occupied_threshold);
  const bool initial_inflation_overlap =
    !cached_footprint_is_safe(
    grid, pose, config_.footprint, occupied_integral);
  if (!cached_footprint_is_safe(
      grid, pose, lethal_footprint, lethal_integral))
  {
    ControllerResult result;
    result.reason = "initial footprint is unsafe";
    return result;
  }
  const std::size_t inflation_escape_steps = simulation_steps(
    config_.initial_inflation_escape_s, config_.dt);
  const double reachable_min_speed =
    std::max(
    config_.min_speed_mps,
    current_speed_mps - config_.maximum_deceleration_mps2 *
    config_.dynamic_window_time_s);
  double reachable_max_speed =
    std::min(
    config_.max_speed_mps,
    current_speed_mps + config_.maximum_acceleration_mps2 *
    config_.dynamic_window_time_s);
  reachable_max_speed = std::max(reachable_min_speed, reachable_max_speed);

  // Preview steering intents reachable over the same dynamic-window interval
  // used for speed.  The command returned below is still rate-limited over one
  // control tick.  Restricting the preview itself to one 50 ms command step
  // prevents an Ackermann vehicle from seeing a viable hard-turn escape after
  // it has stopped in front of an obstacle: every failed cycle resets the
  // command to zero before a larger steering intent can be explored.
  double reachable_min_steering =
    std::max(
    config_.min_steer_rad,
    previous_steering_rad - config_.maximum_steering_rate_radps *
    config_.dynamic_window_time_s);
  double reachable_max_steering =
    std::min(
    config_.sampled_max_steer_rad,
    previous_steering_rad + config_.maximum_steering_rate_radps *
    config_.dynamic_window_time_s);
  const double lateral_limit_speed =
    std::max(current_speed_mps, reachable_max_speed);
  if (lateral_limit_speed > 0.0) {
    const double lateral_steering_limit = std::atan(
      config_.maximum_lateral_acceleration_mps2 * config_.wheelbase_m /
      (lateral_limit_speed * lateral_limit_speed));
    reachable_min_steering =
      std::max(reachable_min_steering, -lateral_steering_limit);
    reachable_max_steering =
      std::min(reachable_max_steering, lateral_steering_limit);
  }
  if (reachable_min_steering > reachable_max_steering) {
    const double nearest_configured =
      std::clamp(
      previous_steering_rad, config_.min_steer_rad,
      config_.sampled_max_steer_rad);
    reachable_min_steering = nearest_configured;
    reachable_max_steering = nearest_configured;
  }
  const auto speed_samples =
    samples(reachable_min_speed, reachable_max_speed, config_.speed_step_mps);
  auto steering_samples = samples(
    reachable_min_steering, reachable_max_steering, config_.steer_step_rad);
  // Stepping from a clipped lateral-acceleration boundary does not generally
  // land on zero or on the current steering angle.  Omitting those candidates
  // can make an otherwise free, straight high-speed stop falsely inadmissible.
  include_sample(
    steering_samples, 0.0, reachable_min_steering,
    reachable_max_steering);
  include_sample(
    steering_samples, previous_steering_rad,
    reachable_min_steering, reachable_max_steering);

  bool found = false;
  bool found_forward_progress = false;
  long double best_score = -std::numeric_limits<long double>::infinity();
  double best_speed = 0.0;
  double best_steering = 0.0;
  Trajectory best_trajectory;
  const long double initial_goal_distance = std::hypot(
    static_cast<long double>(target.x) - static_cast<long double>(pose.x),
    static_cast<long double>(target.y) - static_cast<long double>(pose.y));

  for (const double target_speed : speed_samples) {
    for (const double target_steering : steering_samples) {
      Pose2 simulated = pose;
      double simulated_speed = current_speed_mps;
      double simulated_steering = previous_steering_rad;
      Trajectory trajectory;
      bool safe = true;
      std::size_t remaining_prediction_collision_checks =
        kMaximumPredictionCollisionChecks;
      bool escaping_initial_inflation = initial_inflation_overlap;
      double minimum_clearance = config_.clearance_saturation_m;
      long double path_distance_sum = 0.0L;
      for (std::size_t step = 0; step < steps; ++step) {
        const Pose2 previous_pose = simulated;
        const double previous_speed = simulated_speed;
        const double previous_steering = simulated_steering;
        const double speed_change =
          target_speed >= simulated_speed ?
          config_.maximum_acceleration_mps2 * config_.dt :
          config_.maximum_deceleration_mps2 * config_.dt;
        simulated_speed =
          move_toward(simulated_speed, target_speed, speed_change);
        simulated_steering =
          move_toward(
          simulated_steering, target_steering,
          config_.maximum_steering_rate_radps * config_.dt);
        const double integration_speed =
          0.5 * (previous_speed + simulated_speed);
        const double integration_steering =
          0.5 * (previous_steering + simulated_steering);
        const double maximum_step_speed =
          std::max(previous_speed, simulated_speed);
        const double maximum_step_steering =
          std::abs(previous_steering) >= std::abs(simulated_steering) ?
          previous_steering :
          simulated_steering;
        if (!lateral_acceleration_is_safe(
            maximum_step_speed,
            maximum_step_steering, config_))
        {
          safe = false;
          break;
        }
        simulated.x +=
          integration_speed * std::cos(simulated.yaw_rad) * config_.dt;
        simulated.y +=
          integration_speed * std::sin(simulated.yaw_rad) * config_.dt;
        simulated.yaw_rad += integration_speed / config_.wheelbase_m *
          std::tan(integration_steering) * config_.dt;
        if (!finite_pose(simulated)) {
          safe = false;
          break;
        }
        trajectory.poses.push_back(simulated);
        if (!swept_pose_is_safe(
            grid, previous_pose, simulated,
            escaping_initial_inflation ?
            lethal_footprint : config_.footprint,
            escaping_initial_inflation ?
            lethal_integral : occupied_integral))
        {
          safe = false;
          break;
        }
        if (drivable_mask != nullptr &&
          !swept_pose_is_safe(
            *drivable_mask, previous_pose, simulated, config_.footprint,
            *drivable_integral))
        {
          safe = false;
          break;
        }
        if (!relevant_predictions.empty() &&
          !predicted_sweep_is_safe(
            previous_pose, simulated,
            static_cast<double>(step) * config_.dt,
            static_cast<double>(step + 1U) * config_.dt,
            relevant_predictions, config_,
            drivable_mask->resolution,
            remaining_prediction_collision_checks))
        {
          safe = false;
          break;
        }
        if (escaping_initial_inflation) {
          escaping_initial_inflation =
            !cached_footprint_is_safe(
            grid, simulated, config_.footprint, occupied_integral);
          if (escaping_initial_inflation &&
            step + 1U >= inflation_escape_steps)
          {
            safe = false;
            break;
          }
        }
        const double reference_distance =
          path_distance(simulated, effective_reference);
        if (reference_distance > config_.maximum_path_distance_m) {
          safe = false;
          break;
        }
        minimum_clearance =
          std::min(
          minimum_clearance,
          nearest_clearance(
            unsafe, simulated,
            config_.clearance_saturation_m));
        path_distance_sum += reference_distance;
      }

      // Keep comfortable target-speed changes separate from the measured
      // emergency braking envelope. Using the 20%-brake deceleration here
      // makes a 58.5 km/h vehicle reserve enough distance before an obstacle
      // rejects every otherwise safe avoidance candidate far too early.
      //
      // A candidate remains admissible only when its selected curvature can
      // be brought to a complete emergency stop without entering occupied or
      // unknown space.
      Pose2 braking_pose = simulated;
      double braking_speed = simulated_speed;
      double braking_steering = simulated_steering;
      std::size_t braking_time_index = steps;
      while (safe && braking_speed > 0.0) {
        const Pose2 previous_braking_pose = braking_pose;
        const double next_braking_speed =
          std::max(
          0.0, braking_speed -
          config_.emergency_deceleration_mps2 * config_.dt);
        const double integration_speed =
          0.5 * (braking_speed + next_braking_speed);
        const double next_braking_steering =
          move_toward(
          braking_steering, 0.0,
          config_.maximum_steering_rate_radps * config_.dt);
        const double integration_steering =
          0.5 * (braking_steering + next_braking_steering);
        braking_pose.x +=
          integration_speed * std::cos(braking_pose.yaw_rad) * config_.dt;
        braking_pose.y +=
          integration_speed * std::sin(braking_pose.yaw_rad) * config_.dt;
        braking_pose.yaw_rad += integration_speed / config_.wheelbase_m *
          std::tan(integration_steering) * config_.dt;
        braking_speed = next_braking_speed;
        braking_steering = next_braking_steering;
        ++braking_time_index;
        if (!finite_pose(braking_pose) ||
          !swept_pose_is_safe(
            grid, previous_braking_pose, braking_pose, config_.footprint,
            occupied_integral) ||
          (drivable_mask != nullptr &&
          !swept_pose_is_safe(
            *drivable_mask, previous_braking_pose, braking_pose,
            config_.footprint, *drivable_integral)) ||
          path_distance(braking_pose, effective_reference) >
          config_.maximum_path_distance_m)
        {
          safe = false;
        }
        if (safe && !relevant_predictions.empty() &&
          !predicted_sweep_is_safe(
            previous_braking_pose, braking_pose,
            static_cast<double>(braking_time_index - 1U) * config_.dt,
            static_cast<double>(braking_time_index) * config_.dt,
            relevant_predictions, config_,
            drivable_mask->resolution,
            remaining_prediction_collision_checks))
        {
          safe = false;
        }
      }

      if (safe && !trajectory.poses.empty()) {
        candidate_trajectories_.push_back(trajectory);
        const long double goal_distance =
          std::hypot(
          static_cast<long double>(target.x) -
          static_cast<long double>(simulated.x),
          static_cast<long double>(target.y) -
          static_cast<long double>(simulated.y));
        const long double desired_heading =
          std::atan2(
          static_cast<long double>(target.y) -
          static_cast<long double>(simulated.y),
          static_cast<long double>(target.x) -
          static_cast<long double>(simulated.x));
        const long double mean_path_distance =
          path_distance_sum /
          static_cast<long double>(trajectory.poses.size());
        const long double normalized_clearance =
          static_cast<long double>(minimum_clearance) /
          static_cast<long double>(config_.clearance_saturation_m);
        const long double normalized_speed =
          config_.max_speed_mps > 0.0 ?
          static_cast<long double>(target_speed / config_.max_speed_mps) :
          0.0L;
        const long double goal_progress =
          initial_goal_distance - goal_distance;
        const long double score =
          static_cast<long double>(config_.progress_weight) *
          goal_progress -
          static_cast<long double>(config_.goal_weight) * goal_distance -
          static_cast<long double>(config_.heading_weight) *
          std::abs(
          std::remainder(
            desired_heading -
            static_cast<long double>(simulated.yaw_rad),
            2.0L * static_cast<long double>(kPi))) +
          static_cast<long double>(config_.clearance_weight) *
          normalized_clearance -
          static_cast<long double>(config_.smoothness_weight) *
          std::abs(
          static_cast<long double>(target_steering) -
          previous_steering_rad) -
          static_cast<long double>(config_.path_distance_weight) *
          mean_path_distance +
          static_cast<long double>(config_.speed_weight) * normalized_speed;
        // A zero-speed candidate is the final safe fallback, not a local
        // optimum.  Every forward candidate here has already passed the swept
        // footprint, path-corridor, lateral-acceleration, and complete-stop
        // checks.  Prefer one of those safe escape motions whenever available;
        // otherwise high clearance weight can keep a stopped vehicle parked
        // forever several metres before an avoidable obstacle.
        const bool makes_forward_progress =
          target_speed > 1.0e-6 && goal_progress > 1.0e-6L;
        const bool candidate_is_better =
          !found ||
          (makes_forward_progress && !found_forward_progress) ||
          (makes_forward_progress == found_forward_progress &&
          score > best_score);
        if (std::isfinite(score) && candidate_is_better) {
          found = true;
          found_forward_progress = makes_forward_progress;
          best_score = score;
          best_speed = target_speed;
          best_steering = target_steering;
          best_trajectory = trajectory;
        }
      }
    }
  }
  if (!found) {
    ControllerResult result;
    result.reason = "no safe DWA candidate";
    return result;
  }

  ControllerResult result =
    pid_.update(
    current_speed_mps, best_speed, config_.control_period_s,
    behavior_id, gear_id);
  result.command.steering_rad =
    std::clamp(
    move_toward(
      previous_steering_rad, best_steering,
      config_.maximum_steering_rate_radps *
      config_.control_period_s),
    -config_.max_steer_rad, config_.max_steer_rad);
  result.target = target;
  result.target_speed_mps = best_speed;
  result.local_trajectory = std::move(best_trajectory);
  result.reason = "ok";
  last_valid_ = result;
  return result;
}

const PidState & DwaController::pid_state() const noexcept
{
  return pid_.state();
}
const ControllerResult & DwaController::last_valid_result() const noexcept
{
  return last_valid_;
}
const std::vector<Trajectory> &
DwaController::candidate_trajectories() const noexcept
{
  return candidate_trajectories_;
}
} // namespace ad_planner
