#include "ad_planner/local_planning/common/road_corridor_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_planner/local_planning/common/occupancy_grid_reprojector.hpp"

namespace ad_planner
{
namespace
{

constexpr double kMinimumSegmentLengthSquaredM2 = 1.0e-18;
constexpr double kGeometryToleranceScale = 64.0;
constexpr double kSpatialBinSizeM = 20.0;
constexpr std::size_t kMaximumBinsPerSegment = 4096U;
constexpr std::size_t kMaximumBinsPerQuery = 16384U;

struct GridGeometry
{
  std::size_t width{0U};
  std::size_t height{0U};
  std::size_t cell_count{0U};
  double resolution_m{0.0};
  double origin_x_m{0.0};
  double origin_y_m{0.0};
  double cosine{1.0};
  double sine{0.0};
};

struct GridShape
{
  std::uint32_t width{0U};
  std::uint32_t height{0U};
  std::size_t cell_count{0U};
  float resolution_m{0.0F};
};

struct RouteProjection
{
  double s_m{0.0};
  double absolute_lateral_distance_m{0.0};
};

struct CorridorSegment
{
  double start_x_m{0.0};
  double start_y_m{0.0};
  double delta_x_m{0.0};
  double delta_y_m{0.0};
  double length_squared_m2{0.0};
  double length_m{0.0};
  double start_s_m{0.0};
  double delta_s_m{0.0};
  double start_left_width_m{0.0};
  double delta_left_width_m{0.0};
  double start_right_width_m{0.0};
  double delta_right_width_m{0.0};
  double minimum_x_m{0.0};
  double maximum_x_m{0.0};
  double minimum_y_m{0.0};
  double maximum_y_m{0.0};
};

struct CorridorGeometry
{
  std::vector<CorridorSegment> segments;
  std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>>
  segment_bins;
  std::vector<std::size_t> unindexed_segments;
};

struct CellRange
{
  std::size_t first_x{0U};
  std::size_t last_x{0U};
  std::size_t first_y{0U};
  std::size_t last_y{0U};
};

struct CellProjection
{
  bool valid{false};
  double s_m{0.0};
  double absolute_lateral_distance_m{
    std::numeric_limits<double>::infinity()};
};

struct AxisAlignedBounds
{
  double minimum_x_m{0.0};
  double maximum_x_m{0.0};
  double minimum_y_m{0.0};
  double maximum_y_m{0.0};
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

double tolerance_for(const std::initializer_list<double> values)
{
  double scale = 1.0;
  for (const double value : values) {
    scale = std::max(scale, std::abs(value));
  }
  return kGeometryToleranceScale * std::numeric_limits<double>::epsilon() * scale;
}

std::uint32_t checked_cell_dimension(
  const double minimum_m, const double maximum_m,
  const double resolution_m, const char * const axis)
{
  require_finite(minimum_m, "grid window minimum");
  require_finite(maximum_m, "grid window maximum");
  require_finite(resolution_m, "grid window resolution");
  if (!(maximum_m > minimum_m) || !(resolution_m > 0.0)) {
    throw std::invalid_argument(
            "grid window extents and resolution must be positive");
  }
  const double ratio = (maximum_m - minimum_m) / resolution_m;
  const double rounded = std::round(ratio);
  const double tolerance = tolerance_for({ratio, rounded});
  if (!finite(ratio) || !finite(rounded) ||
    std::abs(ratio - rounded) > tolerance || rounded < 1.0 ||
    rounded > static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
  {
    throw std::invalid_argument(
            std::string("grid window ") + axis +
            " extent is not an integral cell count");
  }
  return static_cast<std::uint32_t>(rounded);
}

GridShape checked_grid_shape(const RoadCorridorGridWindow & window)
{
  const float ros_resolution = static_cast<float>(window.resolution_m);
  if (!std::isfinite(ros_resolution) || !(ros_resolution > 0.0F)) {
    throw std::invalid_argument(
            "grid window resolution is not representable by OccupancyGrid");
  }
  const std::uint32_t width = checked_cell_dimension(
    window.minimum_x_m, window.maximum_x_m, window.resolution_m, "x");
  const std::uint32_t height = checked_cell_dimension(
    window.minimum_y_m, window.maximum_y_m, window.resolution_m, "y");
  const std::size_t width_size = static_cast<std::size_t>(width);
  const std::size_t height_size = static_cast<std::size_t>(height);
  if (width_size > std::numeric_limits<std::size_t>::max() / height_size) {
    throw std::invalid_argument("grid window cell count overflows");
  }
  const std::size_t cell_count = width_size * height_size;
  const std::vector<std::int8_t> allocation_probe;
  if (cell_count > allocation_probe.max_size()) {
    throw std::invalid_argument("grid window allocation is not representable");
  }
  return GridShape{width, height, cell_count, ros_resolution};
}

void validate_positive_stamp(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U ||
    (stamp.sec == 0 && stamp.nanosec == 0U))
  {
    throw std::invalid_argument("grid stamp must be positive and normalized");
  }
}

std::int64_t spatial_bin_index(const double coordinate_m)
{
  require_finite(coordinate_m, "spatial-index coordinate");
  const long double raw = std::floor(
    static_cast<long double>(coordinate_m) /
    static_cast<long double>(kSpatialBinSizeM));
  if (!std::isfinite(raw) ||
    raw < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
    raw > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument(
            "reference corridor spatial-index coordinate is not representable");
  }
  return static_cast<std::int64_t>(raw);
}

std::optional<std::size_t> inclusive_bin_count(
  const std::int64_t minimum, const std::int64_t maximum)
{
  if (maximum < minimum) {
    return std::nullopt;
  }
  const long double count =
    static_cast<long double>(maximum) -
    static_cast<long double>(minimum) + 1.0L;
  if (!std::isfinite(count) || count < 1.0L ||
    count > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
  {
    return std::nullopt;
  }
  return static_cast<std::size_t>(count);
}

void build_spatial_index(CorridorGeometry & geometry)
{
  for (std::size_t segment_index = 0U;
    segment_index < geometry.segments.size(); ++segment_index)
  {
    const auto & segment = geometry.segments[segment_index];
    const std::int64_t minimum_x =
      spatial_bin_index(segment.minimum_x_m);
    const std::int64_t maximum_x =
      spatial_bin_index(segment.maximum_x_m);
    const std::int64_t minimum_y =
      spatial_bin_index(segment.minimum_y_m);
    const std::int64_t maximum_y =
      spatial_bin_index(segment.maximum_y_m);
    const auto count_x = inclusive_bin_count(minimum_x, maximum_x);
    const auto count_y = inclusive_bin_count(minimum_y, maximum_y);
    if (!count_x || !count_y ||
      *count_x > kMaximumBinsPerSegment / *count_y)
    {
      geometry.unindexed_segments.push_back(segment_index);
      continue;
    }
    for (std::int64_t x = minimum_x;; ++x) {
      for (std::int64_t y = minimum_y;; ++y) {
        geometry.segment_bins[{x, y}].push_back(segment_index);
        if (y == maximum_y) {
          break;
        }
      }
      if (x == maximum_x) {
        break;
      }
    }
  }
}

CorridorGeometry validate_corridor(const ReferenceCorridor & corridor)
{
  if (corridor.frame_id.empty()) {
    throw std::invalid_argument("reference corridor frame must not be empty");
  }
  if (corridor.lanes.empty() ||
    corridor.primary_lane_index >= corridor.lanes.size())
  {
    throw std::invalid_argument("reference corridor has no valid primary lane");
  }

  CorridorGeometry geometry;
  for (const auto & lane : corridor.lanes) {
    if (lane.points.size() < 2U) {
      throw std::invalid_argument(
              "each reference corridor lane must contain at least two points");
    }
    const std::size_t additional_segments = lane.points.size() - 1U;
    if (additional_segments >
      geometry.segments.max_size() - geometry.segments.size())
    {
      throw std::invalid_argument("reference corridor segment count overflows");
    }
    geometry.segments.reserve(
      geometry.segments.size() + additional_segments);
  }

  for (std::size_t lane_index = 0U;
    lane_index < corridor.lanes.size(); ++lane_index)
  {
    const auto & lane = corridor.lanes[lane_index];
    if (lane.points.size() < 2U) {
      throw std::invalid_argument(
              "each reference corridor lane must contain at least two points");
    }
    for (std::size_t point_index = 0U;
      point_index < lane.points.size(); ++point_index)
    {
      const auto & route_point = lane.points[point_index];
      require_finite(route_point.pose.x, "reference x");
      require_finite(route_point.pose.y, "reference y");
      require_finite(route_point.pose.yaw_rad, "reference yaw");
      require_finite(route_point.route_s_m, "reference route progress");
      require_finite(route_point.curvature_inv_m, "reference curvature");
      require_finite(route_point.left_width_m, "reference left width");
      require_finite(route_point.right_width_m, "reference right width");
      require_finite(route_point.speed_limit_mps, "reference speed limit");
      if (!(route_point.left_width_m > 0.0) ||
        !(route_point.right_width_m > 0.0))
      {
        throw std::invalid_argument("reference corridor widths must be positive");
      }
      if (route_point.speed_limit_mps < 0.0) {
        throw std::invalid_argument(
                "reference corridor speed limits must be nonnegative");
      }
      if (point_index == 0U) {
        continue;
      }
      const auto & previous = lane.points[point_index - 1U];
      const double delta_s = route_point.route_s_m - previous.route_s_m;
      const double delta_x = route_point.pose.x - previous.pose.x;
      const double delta_y = route_point.pose.y - previous.pose.y;
      const double length_squared = delta_x * delta_x + delta_y * delta_y;
      if (!finite(delta_s) || !(delta_s > 0.0)) {
        throw std::invalid_argument(
                "reference corridor route progress must be strictly increasing");
      }
      if (!finite(delta_x) || !finite(delta_y) || !finite(length_squared) ||
        !(length_squared > kMinimumSegmentLengthSquaredM2))
      {
        throw std::invalid_argument(
                "reference corridor contains invalid geometry segments");
      }

      const double maximum_width = std::max(
        {previous.left_width_m, previous.right_width_m,
          route_point.left_width_m, route_point.right_width_m});
      const double minimum_x =
        std::min(previous.pose.x, route_point.pose.x) - maximum_width;
      const double maximum_x =
        std::max(previous.pose.x, route_point.pose.x) + maximum_width;
      const double minimum_y =
        std::min(previous.pose.y, route_point.pose.y) - maximum_width;
      const double maximum_y =
        std::max(previous.pose.y, route_point.pose.y) + maximum_width;
      const std::array<double, 4U> bounds{
        minimum_x, maximum_x, minimum_y, maximum_y};
      if (!std::all_of(
          bounds.begin(), bounds.end(),
          [](const double value) {return finite(value);}))
      {
        throw std::invalid_argument(
                "reference corridor width expansion overflows");
      }
      geometry.segments.push_back(
        CorridorSegment{
            previous.pose.x,
            previous.pose.y,
            delta_x,
            delta_y,
            length_squared,
            std::sqrt(length_squared),
            previous.route_s_m,
            delta_s,
            previous.left_width_m,
            route_point.left_width_m - previous.left_width_m,
            previous.right_width_m,
            route_point.right_width_m - previous.right_width_m,
            minimum_x,
            maximum_x,
            minimum_y,
            maximum_y});
    }

    std::vector<bool> adjacent_seen(corridor.lanes.size(), false);
    const auto validate_adjacency =
      [&](const std::vector<std::size_t> & adjacent_indices) {
        std::fill(adjacent_seen.begin(), adjacent_seen.end(), false);
        for (const std::size_t adjacent_index : adjacent_indices) {
          if (adjacent_index >= corridor.lanes.size() ||
            adjacent_index == lane_index || adjacent_seen[adjacent_index])
          {
            throw std::invalid_argument(
                    "reference corridor contains invalid lane adjacency");
          }
          adjacent_seen[adjacent_index] = true;
        }
      };
    validate_adjacency(lane.left_lane_indices);
    validate_adjacency(lane.right_lane_indices);
  }
  build_spatial_index(geometry);
  return geometry;
}

GridGeometry validate_grid_geometry(
  const std::string & corridor_frame,
  const nav_msgs::msg::OccupancyGrid & grid)
{
  if (grid.header.frame_id.empty() ||
    grid.header.frame_id != corridor_frame)
  {
    throw std::invalid_argument(
            "occupancy grid and reference corridor frames must match");
  }
  const double resolution = static_cast<double>(grid.info.resolution);
  if (!finite(resolution) || !(resolution > 0.0) ||
    grid.info.width == 0U || grid.info.height == 0U)
  {
    throw std::invalid_argument("occupancy grid dimensions are invalid");
  }
  const std::size_t width = static_cast<std::size_t>(grid.info.width);
  const std::size_t height = static_cast<std::size_t>(grid.info.height);
  if (width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::invalid_argument("occupancy grid dimensions overflow");
  }
  const std::size_t cell_count = width * height;
  const std::vector<std::int8_t> allocation_probe;
  if (cell_count > allocation_probe.max_size()) {
    throw std::invalid_argument("occupancy grid allocation is not representable");
  }

  const auto & origin = grid.info.origin;
  require_finite(origin.position.x, "occupancy grid origin x");
  require_finite(origin.position.y, "occupancy grid origin y");
  require_finite(origin.position.z, "occupancy grid origin z");
  const auto yaw = planar_yaw_from_quaternion(
    QuaternionComponents{
        origin.orientation.x, origin.orientation.y,
        origin.orientation.z, origin.orientation.w});
  if (!yaw) {
    throw std::invalid_argument(
            "occupancy grid origin quaternion must be finite, unit, and planar");
  }
  const double cosine = std::cos(*yaw);
  const double sine = std::sin(*yaw);
  const double span_x = static_cast<double>(width) * resolution;
  const double span_y = static_cast<double>(height) * resolution;
  if (!finite(span_x) || !finite(span_y) ||
    !finite(origin.position.x + cosine * span_x - sine * span_y) ||
    !finite(origin.position.y + sine * span_x + cosine * span_y))
  {
    throw std::invalid_argument("occupancy grid world geometry overflows");
  }

  return GridGeometry{
    width, height, cell_count, resolution,
    origin.position.x, origin.position.y, cosine, sine};
}

std::optional<RouteProjection> project_inside_segment(
  const CorridorSegment & segment, const double x_m, const double y_m)
{
  const double bounds_tolerance = tolerance_for(
    {x_m, y_m, segment.minimum_x_m, segment.maximum_x_m,
      segment.minimum_y_m, segment.maximum_y_m});
  if (x_m < segment.minimum_x_m - bounds_tolerance ||
    x_m > segment.maximum_x_m + bounds_tolerance ||
    y_m < segment.minimum_y_m - bounds_tolerance ||
    y_m > segment.maximum_y_m + bounds_tolerance)
  {
    return std::nullopt;
  }
  const double relative_x = x_m - segment.start_x_m;
  const double relative_y = y_m - segment.start_y_m;
  const double raw_ratio =
    (relative_x * segment.delta_x_m +
    relative_y * segment.delta_y_m) / segment.length_squared_m2;
  const double ratio_tolerance =
    tolerance_for(
    {raw_ratio, segment.start_s_m, segment.start_s_m + segment.delta_s_m});
  if (!finite(raw_ratio) ||
    raw_ratio < -ratio_tolerance || raw_ratio > 1.0 + ratio_tolerance)
  {
    return std::nullopt;
  }
  const double ratio = std::clamp(raw_ratio, 0.0, 1.0);
  const double lateral_m =
    (segment.delta_x_m * relative_y -
    segment.delta_y_m * relative_x) / segment.length_m;
  const double left_width =
    segment.start_left_width_m + ratio * segment.delta_left_width_m;
  const double right_width =
    segment.start_right_width_m + ratio * segment.delta_right_width_m;
  const double boundary_tolerance =
    tolerance_for({lateral_m, left_width, right_width});
  if (!finite(lateral_m) || !finite(left_width) || !finite(right_width) ||
    lateral_m > left_width + boundary_tolerance ||
    lateral_m < -right_width - boundary_tolerance)
  {
    return std::nullopt;
  }
  const double route_s =
    segment.start_s_m + ratio * segment.delta_s_m;
  const double absolute_lateral = std::abs(lateral_m);
  if (!finite(route_s) || !finite(absolute_lateral)) {
    return std::nullopt;
  }
  return RouteProjection{route_s, absolute_lateral};
}

std::array<double, 2U> world_point(
  const GridGeometry & geometry,
  const double local_x_m, const double local_y_m)
{
  const double world_x =
    geometry.origin_x_m +
    geometry.cosine * local_x_m - geometry.sine * local_y_m;
  const double world_y =
    geometry.origin_y_m +
    geometry.sine * local_x_m + geometry.cosine * local_y_m;
  if (!finite(world_x) || !finite(world_y)) {
    throw std::invalid_argument("occupancy grid cell geometry overflows");
  }
  return {world_x, world_y};
}

std::array<double, 2U> grid_local_point(
  const GridGeometry & geometry,
  const double world_x_m, const double world_y_m)
{
  const double delta_x = world_x_m - geometry.origin_x_m;
  const double delta_y = world_y_m - geometry.origin_y_m;
  const double local_x =
    geometry.cosine * delta_x + geometry.sine * delta_y;
  const double local_y =
    -geometry.sine * delta_x + geometry.cosine * delta_y;
  if (!finite(local_x) || !finite(local_y)) {
    throw std::invalid_argument("corridor-to-grid geometry overflows");
  }
  return {local_x, local_y};
}

AxisAlignedBounds grid_world_bounds(const GridGeometry & geometry)
{
  const double span_x =
    static_cast<double>(geometry.width) * geometry.resolution_m;
  const double span_y =
    static_cast<double>(geometry.height) * geometry.resolution_m;
  AxisAlignedBounds bounds{
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};
  for (const double local_x : {0.0, span_x}) {
    for (const double local_y : {0.0, span_y}) {
      const auto world = world_point(geometry, local_x, local_y);
      bounds.minimum_x_m = std::min(bounds.minimum_x_m, world[0]);
      bounds.maximum_x_m = std::max(bounds.maximum_x_m, world[0]);
      bounds.minimum_y_m = std::min(bounds.minimum_y_m, world[1]);
      bounds.maximum_y_m = std::max(bounds.maximum_y_m, world[1]);
    }
  }
  return bounds;
}

std::vector<std::size_t> all_segment_indices(
  const CorridorGeometry & corridor)
{
  std::vector<std::size_t> result(corridor.segments.size());
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = index;
  }
  return result;
}

std::vector<std::size_t> spatial_candidates(
  const CorridorGeometry & corridor, const GridGeometry & grid)
{
  const auto bounds = grid_world_bounds(grid);
  const std::int64_t minimum_x = spatial_bin_index(bounds.minimum_x_m);
  const std::int64_t maximum_x = spatial_bin_index(bounds.maximum_x_m);
  const std::int64_t minimum_y = spatial_bin_index(bounds.minimum_y_m);
  const std::int64_t maximum_y = spatial_bin_index(bounds.maximum_y_m);
  const auto count_x = inclusive_bin_count(minimum_x, maximum_x);
  const auto count_y = inclusive_bin_count(minimum_y, maximum_y);
  if (!count_x || !count_y ||
    *count_x > kMaximumBinsPerQuery / *count_y)
  {
    return all_segment_indices(corridor);
  }

  std::vector<std::size_t> result = corridor.unindexed_segments;
  for (std::int64_t x = minimum_x;; ++x) {
    for (std::int64_t y = minimum_y;; ++y) {
      const auto found = corridor.segment_bins.find({x, y});
      if (found != corridor.segment_bins.end()) {
        result.insert(
          result.end(), found->second.begin(), found->second.end());
      }
      if (y == maximum_y) {
        break;
      }
    }
    if (x == maximum_x) {
      break;
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

void record_cell_visits(
  RoadCorridorGridWork * const work, const CellRange & range)
{
  if (work == nullptr) {
    return;
  }
  const std::size_t width = range.last_x - range.first_x + 1U;
  const std::size_t height = range.last_y - range.first_y + 1U;
  if (width > std::numeric_limits<std::size_t>::max() / height ||
    width * height >
    std::numeric_limits<std::size_t>::max() -
    work->candidate_cell_visit_count)
  {
    work->candidate_cell_visit_count =
      std::numeric_limits<std::size_t>::max();
    return;
  }
  work->candidate_cell_visit_count += width * height;
}

std::optional<CellRange> segment_cell_range(
  const CorridorSegment & segment, const GridGeometry & geometry)
{
  double minimum_local_x = std::numeric_limits<double>::infinity();
  double maximum_local_x = -std::numeric_limits<double>::infinity();
  double minimum_local_y = std::numeric_limits<double>::infinity();
  double maximum_local_y = -std::numeric_limits<double>::infinity();
  for (const double world_x : {segment.minimum_x_m, segment.maximum_x_m}) {
    for (const double world_y : {segment.minimum_y_m, segment.maximum_y_m}) {
      const auto local =
        grid_local_point(geometry, world_x, world_y);
      minimum_local_x = std::min(minimum_local_x, local[0]);
      maximum_local_x = std::max(maximum_local_x, local[0]);
      minimum_local_y = std::min(minimum_local_y, local[1]);
      maximum_local_y = std::max(maximum_local_y, local[1]);
    }
  }
  const double grid_width_m =
    static_cast<double>(geometry.width) * geometry.resolution_m;
  const double grid_height_m =
    static_cast<double>(geometry.height) * geometry.resolution_m;
  if (maximum_local_x < 0.0 || maximum_local_y < 0.0 ||
    minimum_local_x > grid_width_m || minimum_local_y > grid_height_m)
  {
    return std::nullopt;
  }

  const auto clamped_index =
    [](const double local_m, const double resolution_m,
      const std::size_t cell_count) {
      const double raw_index = std::floor(local_m / resolution_m);
      if (raw_index <= 0.0) {
        return std::size_t{0U};
      }
      if (raw_index >= static_cast<double>(cell_count)) {
        return cell_count - 1U;
      }
      return static_cast<std::size_t>(raw_index);
    };
  std::size_t first_x =
    clamped_index(minimum_local_x, geometry.resolution_m, geometry.width);
  std::size_t last_x =
    clamped_index(maximum_local_x, geometry.resolution_m, geometry.width);
  std::size_t first_y =
    clamped_index(minimum_local_y, geometry.resolution_m, geometry.height);
  std::size_t last_y =
    clamped_index(maximum_local_y, geometry.resolution_m, geometry.height);
  // Include cells that only touch an AABB edge. This keeps endpoint and
  // adjacent-segment samples available to the union operation.
  if (first_x > 0U) {
    --first_x;
  }
  if (first_y > 0U) {
    --first_y;
  }
  if (last_x + 1U < geometry.width) {
    ++last_x;
  }
  if (last_y + 1U < geometry.height) {
    ++last_y;
  }
  return CellRange{first_x, last_x, first_y, last_y};
}

std::vector<std::uint16_t> rasterize_sample_coverage(
  const CorridorGeometry & corridor, const GridGeometry & geometry,
  const std::vector<std::size_t> & candidate_segments,
  RoadCorridorGridWork * const work)
{
  constexpr std::array<double, 3U> kFractions{0.0, 0.5, 1.0};
  std::vector<std::uint16_t> coverage(geometry.cell_count, 0U);
  for (const std::size_t segment_index : candidate_segments) {
    const auto & segment = corridor.segments[segment_index];
    const auto range = segment_cell_range(segment, geometry);
    if (!range) {
      continue;
    }
    record_cell_visits(work, *range);
    for (std::size_t y = range->first_y; y <= range->last_y; ++y) {
      for (std::size_t x = range->first_x; x <= range->last_x; ++x) {
        const std::size_t cell_index = y * geometry.width + x;
        std::uint16_t cell_coverage = coverage[cell_index];
        for (std::size_t sample_y = 0U;
          sample_y < kFractions.size(); ++sample_y)
        {
          for (std::size_t sample_x = 0U;
            sample_x < kFractions.size(); ++sample_x)
          {
            const std::size_t bit_index =
              sample_y * kFractions.size() + sample_x;
            const auto sample_bit =
              static_cast<std::uint16_t>(1U << bit_index);
            if ((cell_coverage & sample_bit) != 0U) {
              continue;
            }
            const auto world = world_point(
              geometry,
              (static_cast<double>(x) + kFractions[sample_x]) *
              geometry.resolution_m,
              (static_cast<double>(y) + kFractions[sample_y]) *
              geometry.resolution_m);
            if (project_inside_segment(segment, world[0], world[1])) {
              cell_coverage =
                static_cast<std::uint16_t>(cell_coverage | sample_bit);
            }
          }
        }
        coverage[cell_index] = cell_coverage;
      }
    }
  }
  return coverage;
}

std::vector<CellProjection> project_relevant_cell_centers(
  const CorridorGeometry & corridor, const GridGeometry & geometry,
  const nav_msgs::msg::OccupancyGrid & occupancy_grid,
  const std::int8_t occupied_threshold,
  const double near_s_m, const double far_s_m,
  const std::vector<std::size_t> & candidate_segments,
  RoadCorridorGridWork * const work)
{
  std::vector<CellProjection> projections(geometry.cell_count);
  for (const std::size_t segment_index : candidate_segments) {
    const auto & segment = corridor.segments[segment_index];
    const double segment_end_s_m =
      segment.start_s_m + segment.delta_s_m;
    const double interval_tolerance =
      tolerance_for(
      {near_s_m, far_s_m, segment.start_s_m, segment_end_s_m});
    if (segment_end_s_m < near_s_m - interval_tolerance ||
      segment.start_s_m > far_s_m + interval_tolerance)
    {
      continue;
    }
    const auto range = segment_cell_range(segment, geometry);
    if (!range) {
      continue;
    }
    record_cell_visits(work, *range);
    for (std::size_t y = range->first_y; y <= range->last_y; ++y) {
      for (std::size_t x = range->first_x; x <= range->last_x; ++x) {
        const std::size_t cell_index = y * geometry.width + x;
        const std::int8_t value = occupancy_grid.data[cell_index];
        if (value >= 0 && value < occupied_threshold) {
          continue;
        }
        const auto world = world_point(
          geometry,
          (static_cast<double>(x) + 0.5) * geometry.resolution_m,
          (static_cast<double>(y) + 0.5) * geometry.resolution_m);
        const auto projection =
          project_inside_segment(segment, world[0], world[1]);
        if (!projection) {
          continue;
        }
        const double interval_tolerance =
          tolerance_for({near_s_m, far_s_m, projection->s_m});
        if (projection->s_m < near_s_m - interval_tolerance ||
          projection->s_m > far_s_m + interval_tolerance ||
          (projections[cell_index].valid &&
          projections[cell_index].absolute_lateral_distance_m <=
          projection->absolute_lateral_distance_m))
        {
          continue;
        }
        projections[cell_index] = CellProjection{
          true, projection->s_m,
          projection->absolute_lateral_distance_m};
      }
    }
  }
  return projections;
}

void validate_grid_data(
  const nav_msgs::msg::OccupancyGrid & grid,
  const GridGeometry & geometry)
{
  if (grid.data.size() != geometry.cell_count) {
    throw std::invalid_argument("occupancy grid data size does not match geometry");
  }
  for (const std::int8_t value : grid.data) {
    if (value < -1 || value > 100) {
      throw std::invalid_argument("occupancy grid contains an invalid cell cost");
    }
  }
}

}  // namespace

struct PreparedRoadCorridor::Implementation
{
  std::string frame_id;
  CorridorGeometry geometry;
};

PreparedRoadCorridor::PreparedRoadCorridor(
  const ReferenceCorridor & corridor)
{
  auto implementation = std::make_shared<Implementation>();
  implementation->frame_id = corridor.frame_id;
  implementation->geometry = validate_corridor(corridor);
  implementation_ = std::move(implementation);
}

PreparedRoadCorridor::~PreparedRoadCorridor() = default;

const std::string & PreparedRoadCorridor::frame_id() const noexcept
{
  return implementation_->frame_id;
}

std::size_t PreparedRoadCorridor::segment_count() const noexcept
{
  return implementation_->geometry.segments.size();
}

nav_msgs::msg::OccupancyGrid make_route_aligned_grid_template(
  const RoadCorridorGridWindow & window,
  const std::string & route_frame,
  const builtin_interfaces::msg::Time & stamp,
  const Pose2 & route_from_base)
{
  if (route_frame.empty()) {
    throw std::invalid_argument("route frame must not be empty");
  }
  validate_positive_stamp(stamp);
  require_finite(route_from_base.x, "route from base x");
  require_finite(route_from_base.y, "route from base y");
  require_finite(route_from_base.yaw_rad, "route from base yaw");
  const GridShape shape = checked_grid_shape(window);
  const double cosine = std::cos(route_from_base.yaw_rad);
  const double sine = std::sin(route_from_base.yaw_rad);
  const double origin_x =
    route_from_base.x +
    cosine * window.minimum_x_m - sine * window.minimum_y_m;
  const double origin_y =
    route_from_base.y +
    sine * window.minimum_x_m + cosine * window.minimum_y_m;
  if (!finite(origin_x) || !finite(origin_y)) {
    throw std::invalid_argument("route-aligned grid origin overflows");
  }

  nav_msgs::msg::OccupancyGrid output;
  output.header.frame_id = route_frame;
  output.header.stamp = stamp;
  output.info.map_load_time = stamp;
  output.info.resolution = shape.resolution_m;
  output.info.width = shape.width;
  output.info.height = shape.height;
  output.info.origin.position.x = origin_x;
  output.info.origin.position.y = origin_y;
  output.info.origin.position.z = 0.0;
  const double half_yaw = 0.5 * route_from_base.yaw_rad;
  output.info.origin.orientation.z = std::sin(half_yaw);
  output.info.origin.orientation.w = std::cos(half_yaw);
  output.data.assign(shape.cell_count, static_cast<std::int8_t>(100));
  return output;
}

nav_msgs::msg::OccupancyGrid express_road_corridor_mask_in_base_frame(
  const nav_msgs::msg::OccupancyGrid & route_frame_mask,
  const RoadCorridorGridWindow & window,
  const std::string & base_frame)
{
  if (base_frame.empty()) {
    throw std::invalid_argument("base frame must not be empty");
  }
  validate_positive_stamp(route_frame_mask.header.stamp);
  const GridShape shape = checked_grid_shape(window);
  if (route_frame_mask.header.frame_id.empty() ||
    route_frame_mask.info.width != shape.width ||
    route_frame_mask.info.height != shape.height ||
    route_frame_mask.info.resolution != shape.resolution_m ||
    route_frame_mask.data.size() != shape.cell_count)
  {
    throw std::invalid_argument(
            "route-frame mask does not match the configured grid window");
  }
  for (const std::int8_t value : route_frame_mask.data) {
    if (value != 0 && value != 100) {
      throw std::invalid_argument(
              "route-frame mask must contain only drivable and blocked costs");
    }
  }

  nav_msgs::msg::OccupancyGrid output = route_frame_mask;
  output.header.frame_id = base_frame;
  output.info.map_load_time = output.header.stamp;
  output.info.origin.position.x = window.minimum_x_m;
  output.info.origin.position.y = window.minimum_y_m;
  output.info.origin.position.z = 0.0;
  output.info.origin.orientation.x = 0.0;
  output.info.origin.orientation.y = 0.0;
  output.info.origin.orientation.z = 0.0;
  output.info.origin.orientation.w = 1.0;
  return output;
}

nav_msgs::msg::OccupancyGrid rasterize_road_corridor(
  const ReferenceCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & grid_template)
{
  return rasterize_road_corridor(
    PreparedRoadCorridor(corridor), grid_template);
}

nav_msgs::msg::OccupancyGrid rasterize_road_corridor(
  const PreparedRoadCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & grid_template,
  RoadCorridorGridWork * const work)
{
  const auto & corridor_geometry = corridor.implementation_->geometry;
  const GridGeometry geometry =
    validate_grid_geometry(corridor.frame_id(), grid_template);
  const auto candidates = spatial_candidates(corridor_geometry, geometry);
  if (work != nullptr) {
    *work = RoadCorridorGridWork{
      corridor_geometry.segments.size(), candidates.size(), 0U};
  }
  const auto coverage =
    rasterize_sample_coverage(
    corridor_geometry, geometry, candidates, work);

  nav_msgs::msg::OccupancyGrid output = grid_template;
  output.data.assign(geometry.cell_count, static_cast<std::int8_t>(100));
  constexpr std::uint16_t kWholeCellCoverage = (1U << 9U) - 1U;
  for (std::size_t index = 0U; index < geometry.cell_count; ++index) {
    if (coverage[index] == kWholeCellCoverage) {
      output.data[index] = 0;
    }
  }
  return output;
}

RouteSliceOccupancy query_route_slice_occupancy(
  const ReferenceCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & occupancy_grid,
  const double near_s_m, const double far_s_m,
  const std::int8_t occupied_threshold)
{
  return query_route_slice_occupancy(
    PreparedRoadCorridor(corridor), occupancy_grid,
    near_s_m, far_s_m, occupied_threshold);
}

RouteSliceOccupancy query_route_slice_occupancy(
  const PreparedRoadCorridor & corridor,
  const nav_msgs::msg::OccupancyGrid & occupancy_grid,
  const double near_s_m, const double far_s_m,
  const std::int8_t occupied_threshold,
  RoadCorridorGridWork * const work)
{
  const auto & corridor_geometry = corridor.implementation_->geometry;
  const GridGeometry geometry =
    validate_grid_geometry(corridor.frame_id(), occupancy_grid);
  validate_grid_data(occupancy_grid, geometry);
  require_finite(near_s_m, "near route progress");
  require_finite(far_s_m, "far route progress");
  if (near_s_m > far_s_m) {
    throw std::invalid_argument(
            "near route progress must not exceed far route progress");
  }
  if (occupied_threshold < 0 || occupied_threshold > 100) {
    throw std::invalid_argument("occupied threshold must be in [0, 100]");
  }

  const auto candidates = spatial_candidates(corridor_geometry, geometry);
  if (work != nullptr) {
    *work = RoadCorridorGridWork{
      corridor_geometry.segments.size(), candidates.size(), 0U};
  }
  const auto projections = project_relevant_cell_centers(
    corridor_geometry, geometry, occupancy_grid, occupied_threshold,
    near_s_m, far_s_m, candidates, work);
  RouteSliceOccupancy result;
  for (std::size_t index = 0U; index < geometry.cell_count; ++index) {
    const auto & projection = projections[index];
    if (!projection.valid) {
      continue;
    }
    const std::int8_t value = occupancy_grid.data[index];
    if (value < 0) {
      ++result.unknown_cell_count;
      continue;
    }
    ++result.occupied_cell_count;
    if (!result.nearest_occupied_s_m ||
      projection.s_m < *result.nearest_occupied_s_m)
    {
      result.nearest_occupied_s_m = projection.s_m;
    }
  }
  return result;
}

}  // namespace ad_planner
