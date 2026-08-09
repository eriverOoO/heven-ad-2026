#include "ad_viz/perception/marker_builder.hpp"

#include <ad_interfaces/msg/predicted_object.hpp>
#include <ad_interfaces/msg/predicted_state.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>
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
#include <unordered_set>

namespace ad_viz::perception
{
namespace
{

using ad_interfaces::msg::PredictedObject;
using ad_interfaces::msg::PredictedState;
using builtin_interfaces::msg::Duration;
using geometry_msgs::msg::PoseWithCovariance;
using geometry_msgs::msg::TwistWithCovariance;
using std_msgs::msg::ColorRGBA;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr std::int64_t kHalfSecondNs = 500000000LL;
constexpr std::int64_t kOneSecondNs = 1000000000LL;
constexpr double kQuaternionNormTolerance = 1.0e-6;
constexpr double kCovarianceRelativeTolerance = 1.0e-9;
constexpr double kMinimumMarkerLifetimeSec = 0.10;

bool finite_probability(const double value)
{
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

void require_finite_positive(const double value, const char * field_name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(
            std::string(field_name) + " must be finite and positive");
  }
}

void validate_frame_id(const std::string & frame_id)
{
  if (frame_id.empty() || frame_id.front() == '/') {
    throw std::invalid_argument(
            "predicted-object frame must be nonempty and relative");
  }
  if (std::any_of(
      frame_id.begin(), frame_id.end(),
      [](const unsigned char value) {
        return value <= 0x20U || value == 0x7FU;
      }))
  {
    throw std::invalid_argument(
            "predicted-object frame contains invalid whitespace or control bytes");
  }
}

Duration duration_from_seconds(const double seconds)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    throw std::invalid_argument("duration must be finite and positive");
  }
  const long double nanoseconds =
    static_cast<long double>(seconds) *
    static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(nanoseconds) ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument("duration is not representable");
  }
  const auto rounded = static_cast<std::int64_t>(std::llround(nanoseconds));
  if (rounded <= 0) {
    throw std::invalid_argument("duration rounds below one nanosecond");
  }
  if (rounded / kNanosecondsPerSecond >
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()))
  {
    throw std::invalid_argument(
            "duration seconds exceed builtin_interfaces/Duration range");
  }

  Duration result;
  result.sec = static_cast<std::int32_t>(rounded / kNanosecondsPerSecond);
  result.nanosec =
    static_cast<std::uint32_t>(rounded % kNanosecondsPerSecond);
  return result;
}

std::int64_t duration_ns(const Duration & duration)
{
  if (duration.sec < 0 ||
    duration.nanosec >= static_cast<std::uint32_t>(kNanosecondsPerSecond))
  {
    throw std::invalid_argument("prediction horizon is malformed");
  }
  return
    static_cast<std::int64_t>(duration.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(duration.nanosec);
}

void validate_configuration(const MarkerBuilderConfig & config)
{
  require_finite_positive(
    config.marker_lifetime_sec, "marker lifetime");
  require_finite_positive(config.stale_timeout_sec, "stale timeout");
  if (config.marker_lifetime_sec <= kMinimumMarkerLifetimeSec ||
    config.marker_lifetime_sec >= config.stale_timeout_sec)
  {
    throw std::invalid_argument(
            "marker lifetime must exceed one 10 Hz period and remain below stale timeout");
  }

  const std::array<double, 5> alphas{
    config.current_alpha, config.half_second_alpha,
    config.one_second_alpha, config.line_alpha, config.arrow_alpha};
  if (!std::all_of(
      alphas.begin(), alphas.end(),
      [](const double alpha) {
        return std::isfinite(alpha) && alpha >= 0.0 && alpha <= 1.0;
      }) ||
    !(config.current_alpha > config.half_second_alpha &&
    config.half_second_alpha > config.one_second_alpha))
  {
    throw std::invalid_argument(
            "marker alphas must be finite, bounded, and decrease over the prediction horizon");
  }

  require_finite_positive(config.line_width_m, "line width");
  require_finite_positive(config.box_line_width_m, "box line width");
  require_finite_positive(
    config.arrow_shaft_diameter_m, "arrow shaft diameter");
  require_finite_positive(
    config.arrow_head_diameter_m, "arrow head diameter");
  require_finite_positive(config.arrow_head_length_m, "arrow head length");
  require_finite_positive(config.label_height_m, "label height");
  require_finite_positive(config.label_z_offset_m, "label z offset");
}

void validate_covariance(
  const std::array<double, 36> & covariance, const char * field_name)
{
  if (!std::all_of(
      covariance.begin(), covariance.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument(
            std::string(field_name) + " must be finite");
  }

  double scale = 1.0;
  for (const double value : covariance) {
    scale = std::max(scale, std::abs(value));
  }
  const double tolerance = kCovarianceRelativeTolerance * scale;
  std::array<std::array<double, 6>, 6> matrix{};
  for (std::size_t row = 0; row < 6U; ++row) {
    for (std::size_t column = 0; column < 6U; ++column) {
      const double forward = covariance[row * 6U + column];
      const double reverse = covariance[column * 6U + row];
      if (std::abs(forward - reverse) > tolerance) {
        throw std::invalid_argument(
                std::string(field_name) + " must be symmetric");
      }
      matrix[row][column] = 0.5 * (forward + reverse);
    }
  }

  std::array<std::array<double, 6>, 6> lower{};
  std::array<double, 6> diagonal{};
  for (std::size_t pivot = 0; pivot < 6U; ++pivot) {
    double pivot_value = matrix[pivot][pivot];
    for (std::size_t k = 0; k < pivot; ++k) {
      pivot_value -=
        lower[pivot][k] * lower[pivot][k] * diagonal[k];
    }
    if (!std::isfinite(pivot_value) || pivot_value < -tolerance) {
      throw std::invalid_argument(
              std::string(field_name) + " must be positive semidefinite");
    }
    diagonal[pivot] =
      std::abs(pivot_value) <= tolerance ? 0.0 : pivot_value;
    lower[pivot][pivot] = 1.0;

    for (std::size_t row = pivot + 1U; row < 6U; ++row) {
      double residual = matrix[row][pivot];
      for (std::size_t k = 0; k < pivot; ++k) {
        residual -=
          lower[row][k] * lower[pivot][k] * diagonal[k];
      }
      if (diagonal[pivot] == 0.0) {
        if (std::abs(residual) > tolerance) {
          throw std::invalid_argument(
                  std::string(field_name) + " must be positive semidefinite");
        }
        lower[row][pivot] = 0.0;
      } else {
        lower[row][pivot] = residual / diagonal[pivot];
        if (!std::isfinite(lower[row][pivot])) {
          throw std::invalid_argument(
                  std::string(field_name) + " factorization became nonfinite");
        }
      }
    }
  }
}

void validate_pose(
  const PoseWithCovariance & pose, const char * field_name)
{
  const std::array<double, 7> geometry{
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
    pose.pose.orientation.x, pose.pose.orientation.y,
    pose.pose.orientation.z, pose.pose.orientation.w};
  if (!std::all_of(
      geometry.begin(), geometry.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument(
            std::string(field_name) + " geometry must be finite");
  }
  const auto & quaternion = pose.pose.orientation;
  const double norm_squared =
    quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w;
  if (!std::isfinite(norm_squared) ||
    std::abs(norm_squared - 1.0) > kQuaternionNormTolerance)
  {
    throw std::invalid_argument(
            std::string(field_name) + " quaternion must be unit length");
  }
  validate_covariance(pose.covariance, field_name);
}

void validate_twist(const TwistWithCovariance & twist)
{
  const std::array<double, 6> geometry{
    twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z,
    twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z};
  if (!std::all_of(
      geometry.begin(), geometry.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("initial twist geometry must be finite");
  }
  validate_covariance(twist.covariance, "initial twist covariance");
}

std::int64_t validate_state(const PredictedState & state)
{
  const std::int64_t horizon_ns = duration_ns(state.time_from_start);
  if (horizon_ns <= 0) {
    throw std::invalid_argument("prediction horizon must be positive");
  }
  validate_pose(state.pose, "predicted pose");
  return horizon_ns;
}

void validate_object(const PredictedObject & object)
{
  (void)uuid_namespace(object.object_id);
  if (!finite_probability(object.existence_probability) ||
    !finite_probability(object.classification_probability))
  {
    throw std::invalid_argument(
            "object probabilities must be finite and in [0, 1]");
  }
  require_finite_positive(object.dimensions.x, "object length");
  require_finite_positive(object.dimensions.y, "object width");
  require_finite_positive(object.dimensions.z, "object height");
  validate_pose(object.initial_pose, "initial pose");
  validate_twist(object.initial_twist);
  if (object.states.size() < 2U) {
    throw std::invalid_argument(
            "predicted object must contain at least two future states");
  }
  std::int64_t previous_horizon_ns = 0;
  for (std::size_t index = 0; index < object.states.size(); ++index) {
    const std::int64_t horizon_ns = validate_state(object.states[index]);
    if ((index == 0U && horizon_ns != kHalfSecondNs) ||
      (index == 1U && horizon_ns != kOneSecondNs))
    {
      throw std::invalid_argument(
              "first prediction horizons must be exactly 0.5 and 1.0 seconds");
    }
    if (horizon_ns <= previous_horizon_ns) {
      throw std::invalid_argument(
              "prediction horizons must be strictly increasing");
    }
    previous_horizon_ns = horizon_ns;
  }
}

ColorRGBA track_color(
  const unique_identifier_msgs::msg::UUID & object_id)
{
  std::uint64_t hash = 14695981039346656037ULL;
  for (const std::uint8_t value : object_id.uuid) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }

  const double hue =
    static_cast<double>(hash & 0xFFFFFFULL) / 16777216.0;
  const double saturation =
    0.65 + 0.25 * static_cast<double>((hash >> 24U) & 0xFFULL) / 255.0;
  const double value =
    0.85 + 0.15 * static_cast<double>((hash >> 32U) & 0xFFULL) / 255.0;
  const double scaled_hue = hue * 6.0;
  const auto sector = static_cast<std::uint32_t>(std::floor(scaled_hue));
  const double fraction = scaled_hue - static_cast<double>(sector);
  const double p = value * (1.0 - saturation);
  const double q = value * (1.0 - saturation * fraction);
  const double t = value * (1.0 - saturation * (1.0 - fraction));
  std::array<double, 3> rgb{};
  switch (sector % 6U) {
    case 0U: rgb = {value, t, p}; break;
    case 1U: rgb = {q, value, p}; break;
    case 2U: rgb = {p, value, t}; break;
    case 3U: rgb = {p, q, value}; break;
    case 4U: rgb = {t, p, value}; break;
    default: rgb = {value, p, q}; break;
  }
  ColorRGBA color;
  color.r = static_cast<float>(rgb[0]);
  color.g = static_cast<float>(rgb[1]);
  color.b = static_cast<float>(rgb[2]);
  color.a = 1.0F;
  return color;
}

Marker base_marker(
  const std_msgs::msg::Header & header,
  const std::string & object_namespace,
  const std::int32_t id,
  const MarkerBuilderConfig & config,
  const ColorRGBA & color)
{
  Marker marker;
  marker.header = header;
  marker.ns = object_namespace;
  marker.id = id;
  marker.action = Marker::ADD;
  marker.color = color;
  marker.lifetime = duration_from_seconds(config.marker_lifetime_sec);
  marker.pose.orientation.w = 1.0;
  return marker;
}

std::array<geometry_msgs::msg::Point, 8> box_corners(
  const geometry_msgs::msg::Vector3 & dimensions)
{
  const double half_x = dimensions.x * 0.5;
  const double half_y = dimensions.y * 0.5;
  const double half_z = dimensions.z * 0.5;
  std::array<geometry_msgs::msg::Point, 8> corners;
  const std::array<std::array<double, 3>, 8> coordinates{{
    {{-half_x, -half_y, -half_z}},
    {{half_x, -half_y, -half_z}},
    {{half_x, half_y, -half_z}},
    {{-half_x, half_y, -half_z}},
    {{-half_x, -half_y, half_z}},
    {{half_x, -half_y, half_z}},
    {{half_x, half_y, half_z}},
    {{-half_x, half_y, half_z}}
  }};
  for (std::size_t index = 0; index < corners.size(); ++index) {
    corners[index].x = coordinates[index][0];
    corners[index].y = coordinates[index][1];
    corners[index].z = coordinates[index][2];
  }
  return corners;
}

Marker box_marker(
  const std_msgs::msg::Header & header,
  const std::string & object_namespace,
  const std::int32_t id,
  const geometry_msgs::msg::Pose & pose,
  const geometry_msgs::msg::Vector3 & dimensions,
  const float alpha,
  const MarkerBuilderConfig & config,
  const ColorRGBA & base_color)
{
  Marker marker =
    base_marker(header, object_namespace, id, config, base_color);
  marker.type = Marker::LINE_LIST;
  marker.pose = pose;
  marker.scale.x = config.box_line_width_m;
  marker.color.a = alpha;
  const auto corners = box_corners(dimensions);
  static constexpr std::array<std::array<std::size_t, 2>, 12> kEdges{{
    {{0U, 1U}}, {{1U, 2U}}, {{2U, 3U}}, {{3U, 0U}},
    {{4U, 5U}}, {{5U, 6U}}, {{6U, 7U}}, {{7U, 4U}},
    {{0U, 4U}}, {{1U, 5U}}, {{2U, 6U}}, {{3U, 7U}}
  }};
  marker.points.reserve(kEdges.size() * 2U);
  for (const auto & edge : kEdges) {
    marker.points.push_back(corners[edge[0]]);
    marker.points.push_back(corners[edge[1]]);
  }
  return marker;
}

Marker label_marker(
  const std_msgs::msg::Header & header,
  const std::string & object_namespace,
  const PredictedObject & object,
  const MarkerBuilderConfig & config,
  const ColorRGBA & color)
{
  Marker marker = base_marker(
    header, object_namespace, kTrackLabelMarkerId, config, color);
  marker.type = Marker::TEXT_VIEW_FACING;
  marker.pose.position = object.initial_pose.pose.position;
  const double label_z = marker.pose.position.z +
    object.dimensions.z * 0.5 + config.label_z_offset_m;
  if (!std::isfinite(label_z)) {
    throw std::invalid_argument("track label position must remain finite");
  }
  marker.pose.position.z = label_z;
  marker.scale.z = config.label_height_m;
  marker.color.a = 1.0F;
  const double planar_speed = std::hypot(
    object.initial_twist.twist.linear.x,
    object.initial_twist.twist.linear.y);
  if (!std::isfinite(planar_speed)) {
    throw std::invalid_argument("track label speed must remain finite");
  }
  std::ostringstream stream;
  stream << object_namespace.substr(0U, 8U) << "  " << std::fixed <<
    std::setprecision(2) << planar_speed << " m/s";
  marker.text = stream.str();
  return marker;
}

}  // namespace

std::int64_t prediction_stamp_ns(const std_msgs::msg::Header & header)
{
  validate_frame_id(header.frame_id);
  if (header.stamp.sec < 0 ||
    header.stamp.nanosec >=
    static_cast<std::uint32_t>(kNanosecondsPerSecond))
  {
    throw std::invalid_argument("predicted-object stamp is malformed");
  }
  const std::int64_t result =
    static_cast<std::int64_t>(header.stamp.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(header.stamp.nanosec);
  if (result <= 0) {
    throw std::invalid_argument(
            "predicted-object stamp must be strictly positive");
  }
  return result;
}

std::string uuid_namespace(
  const unique_identifier_msgs::msg::UUID & object_id)
{
  if (std::all_of(
      object_id.uuid.begin(), object_id.uuid.end(),
      [](const std::uint8_t value) {return value == 0U;}))
  {
    throw std::invalid_argument("predicted-object UUID must not be nil");
  }
  std::ostringstream stream;
  stream << std::hex << std::nouppercase << std::setfill('0');
  for (const std::uint8_t value : object_id.uuid) {
    stream << std::setw(2) << static_cast<unsigned int>(value);
  }
  return stream.str();
}

MarkerArray build_delete_all(
  const std_msgs::msg::Header & header,
  const MarkerBuilderConfig & config)
{
  (void)prediction_stamp_ns(header);
  validate_configuration(config);
  Marker clear;
  clear.header = header;
  clear.action = Marker::DELETEALL;
  clear.lifetime = duration_from_seconds(config.marker_lifetime_sec);
  MarkerArray result;
  result.markers.push_back(std::move(clear));
  return result;
}

MarkerArray build_prediction_markers(
  const ad_interfaces::msg::PredictedObjectArray & input,
  const MarkerBuilderConfig & config)
{
  (void)prediction_stamp_ns(input.header);
  validate_configuration(config);

  std::unordered_set<std::string> object_namespaces;
  object_namespaces.reserve(input.objects.size());
  for (const auto & object : input.objects) {
    validate_object(object);
    const auto object_namespace = uuid_namespace(object.object_id);
    if (!object_namespaces.insert(object_namespace).second) {
      throw std::invalid_argument(
              "predicted-object UUIDs must be unique within an array");
    }
  }

  MarkerArray result = build_delete_all(input.header, config);
  if (input.objects.size() >
    (std::numeric_limits<std::size_t>::max() - 1U) / 6U)
  {
    throw std::invalid_argument("predicted-object count is not representable");
  }
  result.markers.reserve(1U + input.objects.size() * 6U);

  for (const auto & object : input.objects) {
    const std::string object_namespace = uuid_namespace(object.object_id);
    const ColorRGBA color = track_color(object.object_id);
    result.markers.push_back(
      box_marker(
        input.header, object_namespace, kCurrentBoxMarkerId,
        object.initial_pose.pose, object.dimensions,
        static_cast<float>(config.current_alpha), config, color));
    result.markers.push_back(
      box_marker(
        input.header, object_namespace, kHalfSecondBoxMarkerId,
        object.states[0].pose.pose, object.dimensions,
        static_cast<float>(config.half_second_alpha), config, color));
    result.markers.push_back(
      box_marker(
        input.header, object_namespace, kOneSecondBoxMarkerId,
        object.states[1].pose.pose, object.dimensions,
        static_cast<float>(config.one_second_alpha), config, color));

    Marker line =
      base_marker(
      input.header, object_namespace, kMotionLineMarkerId, config, color);
    line.type = Marker::LINE_STRIP;
    line.scale.x = config.line_width_m;
    line.color.a = static_cast<float>(config.line_alpha);
    line.points.reserve(1U + object.states.size());
    line.points.push_back(object.initial_pose.pose.position);
    for (const auto & state : object.states) {
      line.points.push_back(state.pose.pose.position);
    }
    result.markers.push_back(std::move(line));

    Marker arrow =
      base_marker(
      input.header, object_namespace, kVelocityArrowMarkerId, config, color);
    arrow.type = Marker::ARROW;
    arrow.scale.x = config.arrow_shaft_diameter_m;
    arrow.scale.y = config.arrow_head_diameter_m;
    arrow.scale.z = config.arrow_head_length_m;
    arrow.color.a = static_cast<float>(config.arrow_alpha);
    geometry_msgs::msg::Point arrow_end =
      object.initial_pose.pose.position;
    arrow_end.x += object.initial_twist.twist.linear.x;
    arrow_end.y += object.initial_twist.twist.linear.y;
    arrow_end.z += object.initial_twist.twist.linear.z;
    if (!std::isfinite(arrow_end.x) ||
      !std::isfinite(arrow_end.y) ||
      !std::isfinite(arrow_end.z))
    {
      throw std::invalid_argument(
              "velocity arrow endpoint must remain finite");
    }
    arrow.points = {object.initial_pose.pose.position, arrow_end};
    result.markers.push_back(std::move(arrow));
    result.markers.push_back(
      label_marker(
        input.header, object_namespace, object, config, color));
  }
  return result;
}

}  // namespace ad_viz::perception
