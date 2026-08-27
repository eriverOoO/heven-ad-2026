#include "ad_viz/perception/object_marker_builder.hpp"

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_viz::perception
{
namespace
{

using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using geometry_msgs::msg::Point;
using std_msgs::msg::ColorRGBA;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;

void require_finite_positive(const double value, const char * name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}

void validate_config(const ObjectMarkerConfig & config)
{
  require_finite_positive(config.marker_lifetime_sec, "marker lifetime");
  require_finite_positive(config.box_line_width_m, "box line width");
  require_finite_positive(config.label_height_m, "label height");
  require_finite_positive(config.label_z_offset_m, "label z offset");
  require_finite_positive(config.velocity_scale_sec, "velocity scale");
  if (!std::isfinite(config.minimum_velocity_mps) || config.minimum_velocity_mps < 0.0) {
    throw std::invalid_argument("minimum velocity must be finite and nonnegative");
  }
}

builtin_interfaces::msg::Duration marker_lifetime(const double seconds)
{
  const auto nanoseconds = static_cast<std::int64_t>(
    std::llround(seconds * static_cast<double>(kNanosecondsPerSecond)));
  if (nanoseconds <= 0 ||
    nanoseconds / kNanosecondsPerSecond > std::numeric_limits<std::int32_t>::max())
  {
    throw std::invalid_argument("marker lifetime is not representable");
  }
  builtin_interfaces::msg::Duration result;
  result.sec = static_cast<std::int32_t>(nanoseconds / kNanosecondsPerSecond);
  result.nanosec = static_cast<std::uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return result;
}

void validate_header(const std_msgs::msg::Header & header)
{
  if (header.frame_id.empty() || header.frame_id.front() == '/') {
    throw std::invalid_argument("object frame must be nonempty and relative");
  }
}

void validate_pose(const geometry_msgs::msg::Pose & pose)
{
  const std::array<double, 7> values{
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
  if (!std::all_of(
      values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
      }))
  {
    throw std::invalid_argument("object pose must be finite");
  }
  const auto & q = pose.orientation;
  const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-5) {
    throw std::invalid_argument("object orientation must be a unit quaternion");
  }
}

void validate_shape(const Shape & shape)
{
  if (shape.type != Shape::BOUNDING_BOX) {
    throw std::invalid_argument("only BOUNDING_BOX objects can be visualized as boxes");
  }
  require_finite_positive(shape.dimensions.x, "object length");
  require_finite_positive(shape.dimensions.y, "object width");
  require_finite_positive(shape.dimensions.z, "object height");
}

const ObjectClassification & primary_classification(
  const std::vector<ObjectClassification> & classifications)
{
  if (classifications.empty()) {
    throw std::invalid_argument("object classification must not be empty");
  }
  const auto selected = std::max_element(
    classifications.begin(), classifications.end(),
    [](const auto & left, const auto & right) {
      return left.probability < right.probability;
    });
  if (!std::all_of(
      classifications.begin(), classifications.end(), [](const auto & item) {
        return item.label <= ObjectClassification::UNDER_DRIVABLE &&
        std::isfinite(item.probability) &&
        item.probability >= 0.0F && item.probability <= 1.0F;
      }))
  {
    throw std::invalid_argument("object classification is invalid");
  }
  return *selected;
}

std::string class_name(const std::uint8_t label)
{
  static constexpr std::array<const char *, 12> kNames{
    "UNKNOWN", "CAR", "TRUCK", "BUS", "TRAILER", "MOTORCYCLE",
    "BICYCLE", "PEDESTRIAN", "ANIMAL", "HAZARD", "OVER_DRIVABLE",
    "UNDER_DRIVABLE"};
  if (label >= kNames.size()) {
    throw std::invalid_argument("object classification label is out of range");
  }
  return kNames[label];
}

ColorRGBA color(const float red, const float green, const float blue, const float alpha)
{
  ColorRGBA result;
  result.r = red;
  result.g = green;
  result.b = blue;
  result.a = alpha;
  return result;
}

Marker base_marker(
  const std_msgs::msg::Header & header, const std::string & marker_namespace,
  const std::int32_t id, const ObjectMarkerConfig & config, const ColorRGBA & marker_color)
{
  Marker marker;
  marker.header = header;
  marker.ns = marker_namespace;
  marker.id = id;
  marker.action = Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.color = marker_color;
  marker.lifetime = marker_lifetime(config.marker_lifetime_sec);
  return marker;
}

Marker clear_marker(const std_msgs::msg::Header & header)
{
  Marker marker;
  marker.header = header;
  marker.action = Marker::DELETEALL;
  return marker;
}

std::array<Point, 8> box_corners(const geometry_msgs::msg::Vector3 & dimensions)
{
  const double x = dimensions.x * 0.5;
  const double y = dimensions.y * 0.5;
  const double z = dimensions.z * 0.5;
  std::array<Point, 8> result;
  const std::array<std::array<double, 3>, 8> coordinates{{
    {{-x, -y, -z}}, {{x, -y, -z}}, {{x, y, -z}}, {{-x, y, -z}},
    {{-x, -y, z}}, {{x, -y, z}}, {{x, y, z}}, {{-x, y, z}}}};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index].x = coordinates[index][0];
    result[index].y = coordinates[index][1];
    result[index].z = coordinates[index][2];
  }
  return result;
}

Marker box_marker(
  const std_msgs::msg::Header & header, const std::string & marker_namespace,
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Vector3 & dimensions,
  const ObjectMarkerConfig & config, const ColorRGBA & marker_color)
{
  Marker marker = base_marker(header, marker_namespace, 0, config, marker_color);
  marker.type = Marker::LINE_LIST;
  marker.pose = pose;
  marker.scale.x = config.box_line_width_m;
  const auto corners = box_corners(dimensions);
  static constexpr std::array<std::array<std::size_t, 2>, 12> kEdges{{
    {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
    {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
    {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}}};
  for (const auto & edge : kEdges) {
    marker.points.push_back(corners[edge[0]]);
    marker.points.push_back(corners[edge[1]]);
  }
  return marker;
}

Marker label_marker(
  const std_msgs::msg::Header & header, const std::string & marker_namespace,
  const geometry_msgs::msg::Pose & pose, const double height, const std::string & text,
  const ObjectMarkerConfig & config, const ColorRGBA & marker_color)
{
  Marker marker = base_marker(header, marker_namespace, 1, config, marker_color);
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose.position = pose.position;
  marker.pose.position.z += height * 0.5 + config.label_z_offset_m;
  marker.scale.z = config.label_height_m;
  marker.color.a = 1.0F;
  marker.text = text;
  return marker;
}

std::string probability_label(
  const ObjectClassification & classification, const double existence_probability)
{
  if (!std::isfinite(existence_probability) ||
    existence_probability < 0.0 || existence_probability > 1.0)
  {
    throw std::invalid_argument("existence probability must be in [0, 1]");
  }
  std::ostringstream stream;
  stream << class_name(classification.label) << ' ' << std::fixed <<
    std::setprecision(2) << existence_probability;
  return stream.str();
}

Marker history_marker(
  const std_msgs::msg::Header & header, const std::string & marker_namespace,
  const std::vector<Point> & points, const ObjectMarkerConfig & config,
  const ColorRGBA & marker_color)
{
  Marker marker = base_marker(header, marker_namespace, 3, config, marker_color);
  marker.type = Marker::LINE_STRIP;
  marker.scale.x = config.box_line_width_m * 0.5;
  marker.points = points;
  return marker;
}

Marker velocity_marker(
  const std_msgs::msg::Header & header, const std::string & marker_namespace,
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Twist & twist,
  const ObjectMarkerConfig & config, const ColorRGBA & marker_color)
{
  const std::array<double, 3> velocity{
    twist.linear.x, twist.linear.y, twist.linear.z};
  if (!std::all_of(
      velocity.begin(), velocity.end(), [](const double value) {
        return std::isfinite(value);
      }))
  {
    throw std::invalid_argument("tracked velocity must be finite");
  }
  Marker marker = base_marker(header, marker_namespace, 2, config, marker_color);
  marker.type = Marker::ARROW;
  marker.pose = pose;
  marker.scale.x = 0.10;
  marker.scale.y = 0.22;
  marker.scale.z = 0.30;
  Point start;
  Point end;
  end.x = twist.linear.x * config.velocity_scale_sec;
  end.y = twist.linear.y * config.velocity_scale_sec;
  end.z = twist.linear.z * config.velocity_scale_sec;
  marker.points = {start, end};
  return marker;
}

}  // namespace

std::string uuid_hex(const unique_identifier_msgs::msg::UUID & uuid)
{
  std::ostringstream stream;
  stream << std::hex << std::nouppercase << std::setfill('0');
  for (const std::uint8_t value : uuid.uuid) {
    stream << std::setw(2) << static_cast<unsigned int>(value);
  }
  return stream.str();
}

MarkerArray build_detection_markers(
  const autoware_perception_msgs::msg::DetectedObjects & input,
  const ObjectMarkerConfig & config)
{
  validate_header(input.header);
  validate_config(config);
  MarkerArray output;
  output.markers.push_back(clear_marker(input.header));
  const auto detection_color = color(1.0F, 0.55F, 0.10F, 0.90F);
  for (std::size_t index = 0; index < input.objects.size(); ++index) {
    const auto & object = input.objects[index];
    const auto & pose = object.kinematics.pose_with_covariance.pose;
    validate_pose(pose);
    validate_shape(object.shape);
    const auto & classification = primary_classification(object.classification);
    std::ostringstream marker_namespace;
    marker_namespace << "detected/" << std::setw(4) << std::setfill('0') << index;
    output.markers.push_back(
      box_marker(
        input.header, marker_namespace.str(), pose, object.shape.dimensions,
        config, detection_color));
    output.markers.push_back(
      label_marker(
        input.header, marker_namespace.str(), pose, object.shape.dimensions.z,
        probability_label(classification, object.existence_probability),
        config, detection_color));
  }
  return output;
}

MarkerArray build_tracked_markers(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const ObjectMarkerConfig & config)
{
  validate_header(input.header);
  validate_config(config);
  MarkerArray output;
  output.markers.push_back(clear_marker(input.header));
  const auto tracked_color = color(0.10F, 0.85F, 1.0F, 0.95F);
  for (const auto & object : input.objects) {
    const auto & pose = object.kinematics.pose_with_covariance.pose;
    validate_pose(pose);
    validate_shape(object.shape);
    const auto & classification = primary_classification(object.classification);
    const std::string uuid = uuid_hex(object.object_id);
    const std::string marker_namespace = "tracked/" + uuid;
    output.markers.push_back(
      box_marker(
        input.header, marker_namespace, pose, object.shape.dimensions,
        config, tracked_color));
    // Last 16 hex chars (last 8 bytes), not first: AB3DMOT's UUID encoding
    // (ab3dmot_ros.py::track_id_to_uuid) packs its small integer track id
    // into the *low* 8 bytes (bytes[8:16]), so the first 16 hex chars are
    // always "0000000000000000" for every AB3DMOT track and the id can
    // occupy any of the low 8 bytes depending on its magnitude -- only the
    // last 16 hex chars are guaranteed to show it in full. For Autoware
    // this remains a stable, effectively-unique suffix of its own UUID.
    const std::string id_suffix = uuid.substr(uuid.size() - 16U, 16U);
    output.markers.push_back(
      label_marker(
        input.header, marker_namespace, pose, object.shape.dimensions.z,
        probability_label(classification, object.existence_probability) +
        " " + config.id_prefix + id_suffix, config, tracked_color));

    const auto & twist = object.kinematics.twist_with_covariance.twist;
    const double speed = std::hypot(
      std::hypot(twist.linear.x, twist.linear.y), twist.linear.z);
    if (!object.kinematics.is_stationary && std::isfinite(speed) &&
      speed >= config.minimum_velocity_mps)
    {
      output.markers.push_back(
        velocity_marker(
          input.header, marker_namespace, pose, twist, config, tracked_color));
    }
  }
  return output;
}

MarkerArray build_trajectory_markers(
  const std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>> & histories,
  const std_msgs::msg::Header & header,
  const ObjectMarkerConfig & config)
{
  validate_header(header);
  validate_config(config);
  MarkerArray output;
  // Dimmer than the box/label/velocity color so the trajectory reads as a
  // trailing history, not another current-frame marker; not a DELETEALL
  // of its own -- callers append these into the same array/topic as
  // build_tracked_markers, which already emits one per frame.
  const auto trajectory_color = color(0.10F, 0.85F, 1.0F, 0.55F);
  for (const auto & [track_key, points] : histories) {
    if (points.size() < 2U) {
      continue;
    }
    output.markers.push_back(
      history_marker(
        header, "trajectory/" + config.id_prefix + track_key, points, config,
        trajectory_color));
  }
  return output;
}

}  // namespace ad_viz::perception
