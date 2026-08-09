#include "ad_planner/visualization/route_markers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "ad_planner/local_planning/common/occupancy.hpp"

namespace ad_planner {
namespace {

constexpr double kMinimumSegmentLengthSquaredM2 = 1.0e-18;
constexpr double kPredictionSampleSpacingM = 0.25;
constexpr double kGeometryToleranceScale = 64.0;
constexpr std::size_t kMaximumPredictionSamplesPerLine = 65536U;

struct CorridorSegment {
  double start_x_m{0.0};
  double start_y_m{0.0};
  double delta_x_m{0.0};
  double delta_y_m{0.0};
  double length_squared_m2{0.0};
  double length_m{0.0};
  double start_left_width_m{0.0};
  double delta_left_width_m{0.0};
  double start_right_width_m{0.0};
  double delta_right_width_m{0.0};
  double minimum_x_m{0.0};
  double maximum_x_m{0.0};
  double minimum_y_m{0.0};
  double maximum_y_m{0.0};
};

struct CorridorGeometry {
  std::vector<CorridorSegment> segments;
  double minimum_x_m{std::numeric_limits<double>::infinity()};
  double maximum_x_m{-std::numeric_limits<double>::infinity()};
  double minimum_y_m{std::numeric_limits<double>::infinity()};
  double maximum_y_m{-std::numeric_limits<double>::infinity()};
};

bool finite(const double value) { return std::isfinite(value); }

void require_finite(const double value, const char *const name) {
  if (!finite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double tolerance_for(const std::initializer_list<double> values) {
  double scale = 1.0;
  for (const double value : values) {
    scale = std::max(scale, std::abs(value));
  }
  return kGeometryToleranceScale * std::numeric_limits<double>::epsilon() *
         scale;
}

std_msgs::msg::ColorRGBA color(const float red, const float green,
                               const float blue, const float alpha) {
  std_msgs::msg::ColorRGBA result;
  result.r = red;
  result.g = green;
  result.b = blue;
  result.a = alpha;
  return result;
}

visualization_msgs::msg::Marker clear_marker(const std::string &frame_id) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  return marker;
}

visualization_msgs::msg::Marker
cube_list_marker(const std::string &frame_id, const int id,
                 const Pose2 &grid_origin, const double resolution_m,
                 const std_msgs::msg::ColorRGBA &marker_color) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.ns = "ad_occupancy_corridor_relevance";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = grid_origin.x;
  marker.pose.position.y = grid_origin.y;
  marker.pose.orientation.z = std::sin(grid_origin.yaw_rad * 0.5);
  marker.pose.orientation.w = std::cos(grid_origin.yaw_rad * 0.5);
  marker.scale.x = resolution_m * 0.92;
  marker.scale.y = resolution_m * 0.92;
  marker.scale.z = std::max(0.08, resolution_m * 0.35);
  marker.color = marker_color;
  return marker;
}

visualization_msgs::msg::Marker
line_list_marker(const std::string &frame_id, const int id,
                 const std_msgs::msg::ColorRGBA &marker_color) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.ns = "ad_prediction_corridor_relevance";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.12;
  marker.color = marker_color;
  return marker;
}

void validate_adjacency(const std::vector<std::size_t> &adjacent_indices,
                        const std::size_t lane_index,
                        const std::size_t lane_count) {
  std::vector<bool> seen(lane_count, false);
  for (const std::size_t adjacent_index : adjacent_indices) {
    if (adjacent_index >= lane_count || adjacent_index == lane_index ||
        seen[adjacent_index]) {
      throw std::invalid_argument(
          "reference corridor contains invalid lane adjacency");
    }
    seen[adjacent_index] = true;
  }
}

CorridorGeometry validate_corridor(const std::string &frame_id,
                                   const ReferenceCorridor &corridor) {
  if (frame_id.empty() || corridor.frame_id.empty() ||
      frame_id != corridor.frame_id) {
    throw std::invalid_argument(
        "marker and reference corridor frames must match");
  }
  if (corridor.lanes.empty() ||
      corridor.primary_lane_index >= corridor.lanes.size()) {
    throw std::invalid_argument("reference corridor has no valid primary lane");
  }

  CorridorGeometry geometry;
  for (const auto &lane : corridor.lanes) {
    if (lane.points.size() < 2U) {
      throw std::invalid_argument(
          "each reference corridor lane must contain at least two points");
    }
    const std::size_t additional_segments = lane.points.size() - 1U;
    if (additional_segments >
        geometry.segments.max_size() - geometry.segments.size()) {
      throw std::invalid_argument("reference corridor segment count overflows");
    }
    geometry.segments.reserve(geometry.segments.size() + additional_segments);
  }

  for (std::size_t lane_index = 0U; lane_index < corridor.lanes.size();
       ++lane_index) {
    const auto &lane = corridor.lanes[lane_index];
    validate_adjacency(lane.left_lane_indices, lane_index,
                       corridor.lanes.size());
    validate_adjacency(lane.right_lane_indices, lane_index,
                       corridor.lanes.size());

    for (std::size_t point_index = 0U; point_index < lane.points.size();
         ++point_index) {
      const auto &point = lane.points[point_index];
      require_finite(point.pose.x, "reference x");
      require_finite(point.pose.y, "reference y");
      require_finite(point.pose.yaw_rad, "reference yaw");
      require_finite(point.route_s_m, "reference route progress");
      require_finite(point.curvature_inv_m, "reference curvature");
      require_finite(point.left_width_m, "reference left width");
      require_finite(point.right_width_m, "reference right width");
      require_finite(point.speed_limit_mps, "reference speed limit");
      if (!(point.left_width_m > 0.0) || !(point.right_width_m > 0.0)) {
        throw std::invalid_argument(
            "reference corridor widths must be positive");
      }
      if (point.speed_limit_mps < 0.0) {
        throw std::invalid_argument(
            "reference corridor speed limits must be nonnegative");
      }
      if (point_index == 0U) {
        continue;
      }

      const auto &previous = lane.points[point_index - 1U];
      const double delta_s = point.route_s_m - previous.route_s_m;
      const double delta_x = point.pose.x - previous.pose.x;
      const double delta_y = point.pose.y - previous.pose.y;
      const double length_squared = delta_x * delta_x + delta_y * delta_y;
      if (!finite(delta_s) || !(delta_s > 0.0)) {
        throw std::invalid_argument(
            "reference corridor route progress must be strictly increasing");
      }
      if (!finite(delta_x) || !finite(delta_y) || !finite(length_squared) ||
          !(length_squared > kMinimumSegmentLengthSquaredM2)) {
        throw std::invalid_argument(
            "reference corridor contains invalid geometry segments");
      }

      const double maximum_width =
          std::max({previous.left_width_m, previous.right_width_m,
                    point.left_width_m, point.right_width_m});
      const double minimum_x =
          std::min(previous.pose.x, point.pose.x) - maximum_width;
      const double maximum_x =
          std::max(previous.pose.x, point.pose.x) + maximum_width;
      const double minimum_y =
          std::min(previous.pose.y, point.pose.y) - maximum_width;
      const double maximum_y =
          std::max(previous.pose.y, point.pose.y) + maximum_width;
      if (!finite(minimum_x) || !finite(maximum_x) || !finite(minimum_y) ||
          !finite(maximum_y)) {
        throw std::invalid_argument(
            "reference corridor width expansion overflows");
      }

      geometry.segments.push_back(CorridorSegment{
          previous.pose.x, previous.pose.y, delta_x, delta_y, length_squared,
          std::sqrt(length_squared), previous.left_width_m,
          point.left_width_m - previous.left_width_m, previous.right_width_m,
          point.right_width_m - previous.right_width_m, minimum_x, maximum_x,
          minimum_y, maximum_y});
      geometry.minimum_x_m = std::min(geometry.minimum_x_m, minimum_x);
      geometry.maximum_x_m = std::max(geometry.maximum_x_m, maximum_x);
      geometry.minimum_y_m = std::min(geometry.minimum_y_m, minimum_y);
      geometry.maximum_y_m = std::max(geometry.maximum_y_m, maximum_y);
    }
  }
  return geometry;
}

bool point_is_inside_corridor(const CorridorGeometry &geometry,
                              const double x_m, const double y_m) {
  const double corridor_tolerance =
      tolerance_for({x_m, y_m, geometry.minimum_x_m, geometry.maximum_x_m,
                     geometry.minimum_y_m, geometry.maximum_y_m});
  if (x_m < geometry.minimum_x_m - corridor_tolerance ||
      x_m > geometry.maximum_x_m + corridor_tolerance ||
      y_m < geometry.minimum_y_m - corridor_tolerance ||
      y_m > geometry.maximum_y_m + corridor_tolerance) {
    return false;
  }

  for (const auto &segment : geometry.segments) {
    const double bounds_tolerance =
        tolerance_for({x_m, y_m, segment.minimum_x_m, segment.maximum_x_m,
                       segment.minimum_y_m, segment.maximum_y_m});
    if (x_m < segment.minimum_x_m - bounds_tolerance ||
        x_m > segment.maximum_x_m + bounds_tolerance ||
        y_m < segment.minimum_y_m - bounds_tolerance ||
        y_m > segment.maximum_y_m + bounds_tolerance) {
      continue;
    }

    const double relative_x = x_m - segment.start_x_m;
    const double relative_y = y_m - segment.start_y_m;
    const double raw_ratio =
        (relative_x * segment.delta_x_m + relative_y * segment.delta_y_m) /
        segment.length_squared_m2;
    const double ratio_tolerance = tolerance_for({raw_ratio});
    if (!finite(raw_ratio) || raw_ratio < -ratio_tolerance ||
        raw_ratio > 1.0 + ratio_tolerance) {
      continue;
    }
    const double ratio = std::clamp(raw_ratio, 0.0, 1.0);
    const double lateral_m =
        (segment.delta_x_m * relative_y - segment.delta_y_m * relative_x) /
        segment.length_m;
    const double left_width =
        segment.start_left_width_m + ratio * segment.delta_left_width_m;
    const double right_width =
        segment.start_right_width_m + ratio * segment.delta_right_width_m;
    const double width_tolerance =
        tolerance_for({lateral_m, left_width, right_width});
    if (finite(lateral_m) && finite(left_width) && finite(right_width) &&
        lateral_m <= left_width + width_tolerance &&
        lateral_m >= -right_width - width_tolerance) {
      return true;
    }
  }
  return false;
}

geometry_msgs::msg::Point marker_point(const double x_m, const double y_m,
                                       const double z_m) {
  if (!finite(x_m) || !finite(y_m) || !finite(z_m)) {
    throw std::invalid_argument("marker point must be finite");
  }
  geometry_msgs::msg::Point point;
  point.x = x_m;
  point.y = y_m;
  point.z = z_m;
  return point;
}

void validate_predicted_objects(const PredictedObjectSet &objects) {
  for (const auto &object : objects) {
    if (object.footprints.empty()) {
      throw std::invalid_argument(
          "each predicted object must contain at least one footprint");
    }
    double previous_time_s = -1.0;
    for (const auto &footprint : object.footprints) {
      require_finite(footprint.time_from_start_s, "prediction time");
      require_finite(footprint.pose.x, "prediction x");
      require_finite(footprint.pose.y, "prediction y");
      require_finite(footprint.pose.yaw_rad, "prediction yaw");
      require_finite(footprint.length_m, "prediction length");
      require_finite(footprint.width_m, "prediction width");
      require_finite(footprint.covariance_xx, "prediction covariance xx");
      require_finite(footprint.covariance_yy, "prediction covariance yy");
      require_finite(footprint.covariance_xy, "prediction covariance xy");
      const double covariance_midpoint =
          0.5 * footprint.covariance_xx + 0.5 * footprint.covariance_yy;
      const double covariance_radius = std::hypot(
          0.5 * footprint.covariance_xx - 0.5 * footprint.covariance_yy,
          footprint.covariance_xy);
      if (footprint.time_from_start_s < 0.0 ||
          !(footprint.time_from_start_s > previous_time_s) ||
          !(footprint.length_m > 0.0) || !(footprint.width_m > 0.0) ||
          footprint.covariance_xx < 0.0 || footprint.covariance_yy < 0.0 ||
          !std::isfinite(covariance_midpoint) ||
          !std::isfinite(covariance_radius) ||
          covariance_midpoint - covariance_radius < -1.0e-9) {
        throw std::invalid_argument("predicted footprint geometry is invalid");
      }
      previous_time_s = footprint.time_from_start_s;
    }
  }
}

std::array<geometry_msgs::msg::Point, 4U>
footprint_corners(const PredictedFootprint &footprint) {
  const double cosine = std::cos(footprint.pose.yaw_rad);
  const double sine = std::sin(footprint.pose.yaw_rad);
  const double half_length = footprint.length_m * 0.5;
  const double half_width = footprint.width_m * 0.5;
  const std::array<std::array<double, 2U>, 4U> local_corners{{
      {{half_length, half_width}},
      {{half_length, -half_width}},
      {{-half_length, -half_width}},
      {{-half_length, half_width}},
  }};
  std::array<geometry_msgs::msg::Point, 4U> result;
  for (std::size_t index = 0U; index < local_corners.size(); ++index) {
    const double x_m = footprint.pose.x + cosine * local_corners[index][0] -
                       sine * local_corners[index][1];
    const double y_m = footprint.pose.y + sine * local_corners[index][0] +
                       cosine * local_corners[index][1];
    result[index] = marker_point(x_m, y_m, 0.25);
  }
  return result;
}

std::size_t prediction_sample_count(const geometry_msgs::msg::Point &start,
                                    const geometry_msgs::msg::Point &end) {
  const double delta_x = end.x - start.x;
  const double delta_y = end.y - start.y;
  const double distance_m = std::hypot(delta_x, delta_y);
  const double samples = std::ceil(distance_m / kPredictionSampleSpacingM);
  if (!finite(delta_x) || !finite(delta_y) || !finite(distance_m) ||
      !finite(samples) ||
      samples > static_cast<double>(kMaximumPredictionSamplesPerLine)) {
    throw std::invalid_argument("prediction marker geometry is too large");
  }
  return std::max<std::size_t>(1U, static_cast<std::size_t>(samples));
}

void append_sampled_line(const geometry_msgs::msg::Point &start,
                         const geometry_msgs::msg::Point &end,
                         const CorridorGeometry &corridor,
                         const std::size_t maximum_segments,
                         visualization_msgs::msg::Marker &red,
                         visualization_msgs::msg::Marker &blue) {
  if (red.points.size() / 2U >= maximum_segments &&
      blue.points.size() / 2U >= maximum_segments) {
    return;
  }
  const std::size_t sample_count = prediction_sample_count(start, end);
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    const double start_fraction =
        static_cast<double>(sample) / static_cast<double>(sample_count);
    const double end_fraction =
        static_cast<double>(sample + 1U) / static_cast<double>(sample_count);
    const double middle_fraction = 0.5 * (start_fraction + end_fraction);
    const double middle_x = start.x + middle_fraction * (end.x - start.x);
    const double middle_y = start.y + middle_fraction * (end.y - start.y);
    auto &marker =
        point_is_inside_corridor(corridor, middle_x, middle_y) ? red : blue;
    if (marker.points.size() / 2U >= maximum_segments) {
      continue;
    }
    marker.points.push_back(
        marker_point(start.x + start_fraction * (end.x - start.x),
                     start.y + start_fraction * (end.y - start.y), start.z));
    marker.points.push_back(
        marker_point(start.x + end_fraction * (end.x - start.x),
                     start.y + end_fraction * (end.y - start.y), end.z));
    if (red.points.size() / 2U >= maximum_segments &&
        blue.points.size() / 2U >= maximum_segments) {
      break;
    }
  }
}

} // namespace

visualization_msgs::msg::MarkerArray build_occupancy_relevance_markers(
    const std::string &frame_id, const OccupancyGrid &grid,
    const ReferenceCorridor &corridor, const std::int8_t occupied_threshold,
    const RouteMarkerLimits &limits) {
  if (occupied_threshold < 0 || occupied_threshold > 100 ||
      limits.maximum_occupancy_points_per_class == 0U ||
      !validate_occupancy_grid(grid).valid) {
    throw std::invalid_argument("invalid occupancy relevance marker input");
  }
  if (!std::all_of(grid.cells.begin(), grid.cells.end(),
                   [](const std::int8_t value) {
                     return value >= -1 && value <= 100;
                   })) {
    throw std::invalid_argument("occupancy grid contains invalid cell values");
  }
  const CorridorGeometry corridor_geometry =
      validate_corridor(frame_id, corridor);

  visualization_msgs::msg::MarkerArray result;
  result.markers.push_back(clear_marker(frame_id));
  auto red = cube_list_marker(frame_id, 1, grid.origin, grid.resolution,
                              color(1.0F, 0.05F, 0.05F, 0.85F));
  auto blue = cube_list_marker(frame_id, 2, grid.origin, grid.resolution,
                               color(0.05F, 0.35F, 1.0F, 0.32F));
  red.points.reserve(limits.maximum_occupancy_points_per_class);
  blue.points.reserve(limits.maximum_occupancy_points_per_class);

  const double cosine = std::cos(grid.origin.yaw_rad);
  const double sine = std::sin(grid.origin.yaw_rad);
  bool complete = false;
  for (std::size_t row = 0U; row < grid.height && !complete; ++row) {
    for (std::size_t column = 0U; column < grid.width; ++column) {
      const std::int8_t value = grid.cells[row * grid.width + column];
      if (value < occupied_threshold) {
        continue;
      }
      const double local_x =
          (static_cast<double>(column) + 0.5) * grid.resolution;
      const double local_y = (static_cast<double>(row) + 0.5) * grid.resolution;
      const double x_m = grid.origin.x + cosine * local_x - sine * local_y;
      const double y_m = grid.origin.y + sine * local_x + cosine * local_y;
      auto &marker =
          point_is_inside_corridor(corridor_geometry, x_m, y_m) ? red : blue;
      if (marker.points.size() < limits.maximum_occupancy_points_per_class) {
        marker.points.push_back(marker_point(local_x, local_y, 0.12));
      }
      complete =
          red.points.size() >= limits.maximum_occupancy_points_per_class &&
          blue.points.size() >= limits.maximum_occupancy_points_per_class;
      if (complete) {
        break;
      }
    }
  }

  result.markers.push_back(std::move(red));
  result.markers.push_back(std::move(blue));
  return result;
}

visualization_msgs::msg::MarkerArray build_predicted_relevance_markers(
    const std::string &frame_id, const PredictedObjectSet &objects,
    const ReferenceCorridor &corridor, const RouteMarkerLimits &limits) {
  if (limits.maximum_prediction_segments_per_class == 0U) {
    throw std::invalid_argument("invalid predicted relevance marker limits");
  }
  const CorridorGeometry corridor_geometry =
      validate_corridor(frame_id, corridor);
  validate_predicted_objects(objects);

  visualization_msgs::msg::MarkerArray result;
  result.markers.push_back(clear_marker(frame_id));
  auto red = line_list_marker(frame_id, 1, color(1.0F, 0.05F, 0.05F, 0.95F));
  auto blue = line_list_marker(frame_id, 2, color(0.05F, 0.35F, 1.0F, 0.75F));

  for (const auto &object : objects) {
    for (const auto &footprint : object.footprints) {
      const auto corners = footprint_corners(footprint);
      for (std::size_t edge = 0U; edge < corners.size(); ++edge) {
        append_sampled_line(
            corners[edge], corners[(edge + 1U) % corners.size()],
            corridor_geometry, limits.maximum_prediction_segments_per_class,
            red, blue);
      }
    }
    for (std::size_t index = 0U; index + 1U < object.footprints.size();
         ++index) {
      const auto &start = object.footprints[index];
      const auto &end = object.footprints[index + 1U];
      append_sampled_line(
          marker_point(start.pose.x, start.pose.y, 0.25),
          marker_point(end.pose.x, end.pose.y, 0.25), corridor_geometry,
          limits.maximum_prediction_segments_per_class, red, blue);
    }
  }

  result.markers.push_back(std::move(red));
  result.markers.push_back(std::move(blue));
  return result;
}

} // namespace ad_planner
