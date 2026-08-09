#include "autoware_prediction_node.hpp"

#include <ad_interfaces/msg/predicted_object.hpp>
#include <ad_interfaces/msg/predicted_state.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad_lidar_perception::tracking
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr double kQuaternionNormTolerance = 1.0e-6;

struct ConvertedObject
{
  TrackState2D track;
  geometry_msgs::msg::Quaternion normalized_orientation;
};

class PredictionInputError : public std::invalid_argument
{
public:
  PredictionInputError(
    const std::string & message,
    std::array<std::uint8_t, 16> culprit_uuid)
  : std::invalid_argument(message), culprit_uuid_(culprit_uuid) {}

  const std::array<std::uint8_t, 16> & culprit_uuid() const noexcept
  {
    return culprit_uuid_;
  }

private:
  std::array<std::uint8_t, 16> culprit_uuid_;
};

bool finite_probability(const double value)
{
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

std::int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    throw std::invalid_argument("tracked-object stamp is malformed");
  }
  const std::int64_t nanoseconds =
    static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(stamp.nanosec);
  if (nanoseconds <= 0) {
    throw std::invalid_argument(
            "tracked-object stamp must be strictly positive");
  }
  return nanoseconds;
}

std::int64_t maximum_age_ns(const double maximum_input_age_sec)
{
  if (!std::isfinite(maximum_input_age_sec) || maximum_input_age_sec <= 0.0) {
    throw std::invalid_argument(
            "maximum input age must be finite and strictly positive");
  }
  const long double nanoseconds =
    static_cast<long double>(maximum_input_age_sec) *
    static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(nanoseconds) ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument("maximum input age is not representable");
  }
  const long double rounded = std::round(nanoseconds);
  if (rounded < 1.0L) {
    throw std::invalid_argument("maximum input age is below one nanosecond");
  }
  return static_cast<std::int64_t>(rounded);
}

void validate_adapter_config(const AutowarePredictionAdapterConfig & config)
{
  if (config.expected_frame_id != "odom") {
    throw std::invalid_argument(
            "prediction input frame must be configured as odom");
  }
  (void)maximum_age_ns(config.maximum_input_age_sec);
  (void)predict_tracks({}, config.prediction);
}

geometry_msgs::msg::Quaternion normalized_orientation(
  const geometry_msgs::msg::Quaternion & orientation)
{
  const std::array<double, 4> components{orientation.x, orientation.y,
    orientation.z, orientation.w};
  if (!std::all_of(
      components.begin(), components.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("pose quaternion must be finite");
  }
  const double norm_squared =
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w;
  if (!std::isfinite(norm_squared) || norm_squared <= 0.0) {
    throw std::invalid_argument("pose quaternion has invalid norm");
  }
  const double norm = std::sqrt(norm_squared);
  if (std::abs(norm - 1.0) > kQuaternionNormTolerance) {
    throw std::invalid_argument("pose quaternion is not near unit length");
  }

  geometry_msgs::msg::Quaternion normalized;
  normalized.x = orientation.x / norm;
  normalized.y = orientation.y / norm;
  normalized.z = orientation.z / norm;
  normalized.w = orientation.w / norm;
  return normalized;
}

double
yaw_from_orientation(const geometry_msgs::msg::Quaternion & orientation)
{
  const double sine =
    2.0 * (orientation.w * orientation.z + orientation.x * orientation.y);
  const double cosine = 1.0 - 2.0 * (orientation.y * orientation.y +
    orientation.z * orientation.z);
  const double yaw = std::atan2(sine, cosine);
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("pose yaw is nonfinite");
  }
  return yaw;
}

geometry_msgs::msg::Quaternion orientation_from_yaw(const double yaw)
{
  geometry_msgs::msg::Quaternion orientation;
  orientation.z = std::sin(0.5 * yaw);
  orientation.w = std::cos(0.5 * yaw);
  return orientation;
}

std::pair<std::uint8_t, double> select_classification(
  const std::vector<autoware_perception_msgs::msg::ObjectClassification>
  & classifications)
{
  if (classifications.empty()) {
    throw std::invalid_argument("tracked object has no classification");
  }

  bool selected = false;
  std::uint8_t selected_label = 0U;
  double selected_probability = 0.0;
  for (const auto & classification : classifications) {
    if (classification.label >
      autoware_perception_msgs::msg::ObjectClassification::PEDESTRIAN ||
      !finite_probability(classification.probability))
    {
      throw std::invalid_argument("tracked-object classification is invalid");
    }
    if (!selected || classification.probability > selected_probability ||
      (classification.probability == selected_probability &&
      classification.label < selected_label))
    {
      selected = true;
      selected_label = classification.label;
      selected_probability = classification.probability;
    }
  }
  return {selected_label, selected_probability};
}

std::array<double, 4> covariance_xy(const std::array<double, 36> & covariance)
{
  return {covariance[0], covariance[1], covariance[6], covariance[7]};
}

double positive_variance(const double value, const double fallback)
{
  return std::isfinite(value) && value > 0.0 ? value : fallback;
}

ConvertedObject
convert_object(const autoware_perception_msgs::msg::TrackedObject & object)
{
  if (!finite_probability(object.existence_probability)) {
    throw std::invalid_argument("existence probability is invalid");
  }
  if (object.shape.type != autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    throw std::invalid_argument(
            "only bounding-box tracked objects are supported");
  }
  if (!std::isfinite(object.shape.dimensions.x) ||
    !std::isfinite(object.shape.dimensions.y) ||
    !std::isfinite(object.shape.dimensions.z) ||
    object.shape.dimensions.x <= 0.0 || object.shape.dimensions.y <= 0.0 ||
    object.shape.dimensions.z <= 0.0)
  {
    throw std::invalid_argument(
            "tracked-object dimensions must be finite and positive");
  }

  const auto classification = select_classification(object.classification);
  const auto orientation = normalized_orientation(
    object.kinematics.pose_with_covariance.pose.orientation);
  const double yaw = yaw_from_orientation(orientation);
  const auto & pose = object.kinematics.pose_with_covariance;
  if (!std::all_of(
      pose.covariance.begin(), pose.covariance.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("pose covariance must be finite");
  }
  const auto & local_twist = object.kinematics.twist_with_covariance;
  const WorldMotion2D world_motion = rotate_object_local_motion_to_world(
    yaw, local_twist.twist.linear.x, local_twist.twist.linear.y,
    covariance_xy(local_twist.covariance));

  ConvertedObject converted;
  converted.track.id = object.object_id.uuid;
  converted.track.classification = classification.first;
  converted.track.existence_probability = object.existence_probability;
  converted.track.classification_probability = classification.second;
  converted.track.x_m = pose.pose.position.x;
  converted.track.y_m = pose.pose.position.y;
  converted.track.z_m = pose.pose.position.z;
  converted.track.yaw_rad = yaw;
  converted.track.vx_world_mps = world_motion.vx_world_mps;
  converted.track.vy_world_mps = world_motion.vy_world_mps;
  converted.track.length_m = object.shape.dimensions.x;
  converted.track.width_m = object.shape.dimensions.y;
  converted.track.height_m = object.shape.dimensions.z;
  converted.track.position_covariance_xy = covariance_xy(pose.covariance);
  converted.track.velocity_covariance_xy = world_motion.velocity_covariance_xy;
  // The tracker legitimately carries ego roll/pitch into its odom-frame
  // object pose. Prediction is intentionally 2D, so retain the ZYX yaw and
  // publish a planar quaternion instead of rejecting the whole array.
  converted.normalized_orientation = orientation_from_yaw(yaw);
  return converted;
}

void set_covariance_xy(
  std::array<double, 36> & covariance,
  const std::array<double, 4> & xy)
{
  covariance[0] = xy[0];
  covariance[1] = xy[1];
  covariance[6] = xy[2];
  covariance[7] = xy[3];
}

void set_duration(
  builtin_interfaces::msg::Duration & duration,
  const std::int64_t nanoseconds)
{
  if (nanoseconds <= 0) {
    throw std::overflow_error("prediction duration must be positive");
  }
  const std::int64_t seconds = nanoseconds / kNanosecondsPerSecond;
  if (seconds > std::numeric_limits<std::int32_t>::max()) {
    throw std::overflow_error("prediction duration seconds overflow");
  }
  duration.sec = static_cast<std::int32_t>(seconds);
  duration.nanosec =
    static_cast<std::uint32_t>(nanoseconds % kNanosecondsPerSecond);
}

ad_interfaces::msg::PredictedObject
map_prediction(
  const autoware_perception_msgs::msg::TrackedObject & source,
  const ConvertedObject & converted,
  const PredictedTrack2D & prediction)
{
  ad_interfaces::msg::PredictedObject output;
  output.object_id = source.object_id;
  output.existence_probability =
    static_cast<float>(prediction.initial_state.existence_probability);
  output.classification = prediction.initial_state.classification;
  output.classification_probability =
    static_cast<float>(prediction.initial_state.classification_probability);
  output.initial_pose.pose = source.kinematics.pose_with_covariance.pose;
  output.initial_pose.pose.orientation = converted.normalized_orientation;
  set_covariance_xy(
    output.initial_pose.covariance,
    prediction.initial_state.position_covariance_xy);
  output.initial_pose.covariance[35] = positive_variance(
    source.kinematics.pose_with_covariance.covariance[35], 0.04);

  // PredictedObject default-constructs initial_twist with all fields zero.
  // Populate only the validated world-frame XY velocity and covariance.
  output.initial_twist.twist.linear.x = prediction.initial_state.vx_world_mps;
  output.initial_twist.twist.linear.y = prediction.initial_state.vy_world_mps;
  set_covariance_xy(
    output.initial_twist.covariance,
    prediction.initial_state.velocity_covariance_xy);

  output.dimensions.x = prediction.initial_state.length_m;
  output.dimensions.y = prediction.initial_state.width_m;
  output.dimensions.z = prediction.initial_state.height_m;
  output.states.reserve(prediction.states.size());
  for (const auto & predicted_state : prediction.states) {
    ad_interfaces::msg::PredictedState state;
    set_duration(state.time_from_start, predicted_state.time_from_start_ns);
    state.pose.pose.position.x = predicted_state.x_m;
    state.pose.pose.position.y = predicted_state.y_m;
    state.pose.pose.position.z = predicted_state.z_m;
    state.pose.pose.orientation = converted.normalized_orientation;
    set_covariance_xy(
      state.pose.covariance,
      predicted_state.position_covariance_xy);
    output.states.push_back(std::move(state));
  }
  return output;
}

TrackMeasurement2D
imm_measurement(
  const autoware_perception_msgs::msg::TrackedObject & source,
  const ConvertedObject & converted)
{
  const auto & pose_covariance =
    source.kinematics.pose_with_covariance.covariance;
  const auto & twist = source.kinematics.twist_with_covariance;
  TrackMeasurement2D measurement;
  measurement.x_m = converted.track.x_m;
  measurement.y_m = converted.track.y_m;
  measurement.yaw_rad = converted.track.yaw_rad;
  measurement.vx_world_mps = converted.track.vx_world_mps;
  measurement.vy_world_mps = converted.track.vy_world_mps;
  measurement.yaw_rate_radps = twist.twist.angular.z;
  measurement.position_variance_m2 =
    positive_variance(0.5 * (pose_covariance[0] + pose_covariance[7]), 0.25);
  measurement.velocity_variance_m2ps2 =
    positive_variance(
    0.5 * (converted.track.velocity_covariance_xy[0] +
    converted.track.velocity_covariance_xy[3]),
    1.0);
  measurement.yaw_variance_rad2 = positive_variance(pose_covariance[35], 0.04);
  measurement.yaw_rate_variance_rad2ps2 =
    positive_variance(twist.covariance[35], 0.04);
  return measurement;
}

void validate_imm_measurement(const TrackMeasurement2D & measurement)
{
  const std::array<double, 6> state{
    measurement.x_m,
    measurement.y_m,
    measurement.yaw_rad,
    measurement.vx_world_mps,
    measurement.vy_world_mps,
    measurement.yaw_rate_radps};
  if (!std::all_of(
      state.begin(), state.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument("IMM measurements must be finite");
  }
  const std::array<double, 4> variances{
    measurement.position_variance_m2,
    measurement.velocity_variance_m2ps2,
    measurement.yaw_variance_rad2,
    measurement.yaw_rate_variance_rad2ps2};
  if (!std::all_of(
      variances.begin(), variances.end(),
      [](const double value) {
        return std::isfinite(value) && value > 0.0;
      }))
  {
    throw std::invalid_argument(
            "IMM measurement variances must be finite and positive");
  }
}

ad_interfaces::msg::PredictedObject
map_imm_prediction(
  const autoware_perception_msgs::msg::TrackedObject & source,
  const ConvertedObject & converted,
  const ImmResult & prediction)
{
  ad_interfaces::msg::PredictedObject output;
  output.object_id = source.object_id;
  output.existence_probability = source.existence_probability;
  output.classification = converted.track.classification;
  output.classification_probability =
    static_cast<float>(converted.track.classification_probability);
  output.initial_pose.pose = source.kinematics.pose_with_covariance.pose;
  output.initial_pose.pose.orientation =
    orientation_from_yaw(prediction.fused_state.yaw_rad);
  set_covariance_xy(
    output.initial_pose.covariance,
    converted.track.position_covariance_xy);
  output.initial_pose.covariance[35] = positive_variance(
    source.kinematics.pose_with_covariance.covariance[35], 0.04);
  output.initial_twist.twist.linear.x = prediction.fused_state.vx_world_mps;
  output.initial_twist.twist.linear.y = prediction.fused_state.vy_world_mps;
  output.initial_twist.twist.angular.z = prediction.fused_state.yaw_rate_radps;
  set_covariance_xy(
    output.initial_twist.covariance,
    converted.track.velocity_covariance_xy);
  output.initial_twist.covariance[35] = positive_variance(
    source.kinematics.twist_with_covariance.covariance[35], 0.04);
  output.dimensions.x = converted.track.length_m;
  output.dimensions.y = converted.track.width_m;
  output.dimensions.z = converted.track.height_m;
  output.states.reserve(prediction.predicted_states.size());
  for (const auto & predicted_state : prediction.predicted_states) {
    ad_interfaces::msg::PredictedState state;
    const auto duration_ns = static_cast<std::int64_t>(std::llround(
        predicted_state.time_from_start_s * kNanosecondsPerSecond));
    set_duration(state.time_from_start, duration_ns);
    state.pose.pose.position.x = predicted_state.x_m;
    state.pose.pose.position.y = predicted_state.y_m;
    state.pose.pose.position.z = converted.track.z_m;
    state.pose.pose.orientation = orientation_from_yaw(predicted_state.yaw_rad);
    set_covariance_xy(
      state.pose.covariance,
      converted.track.position_covariance_xy);
    output.states.push_back(std::move(state));
  }
  return output;
}

std::string uuid_hex(const std::array<std::uint8_t, 16> & uuid)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(uuid.size() * 2U);
  for (const auto byte : uuid) {
    output.push_back(digits[(byte >> 4U) & 0x0fU]);
    output.push_back(digits[byte & 0x0fU]);
  }
  return output;
}

std::string probability_text(const double probability)
{
  std::ostringstream stream;
  stream.precision(std::numeric_limits<double>::max_digits10);
  stream << probability;
  return stream.str();
}

const char * model_name(const std::size_t model_index)
{
  switch (static_cast<MotionModel>(model_index)) {
    case MotionModel::kStationary:
      return "stationary";
    case MotionModel::kConstantVelocity:
      return "constant_velocity";
    case MotionModel::kCoordinatedTurn:
      return "coordinated_turn";
  }
  throw std::logic_error("IMM selected an unknown model");
}

diagnostic_msgs::msg::KeyValue key_value(
  const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue output;
  output.key = key;
  output.value = value;
  return output;
}

diagnostic_msgs::msg::DiagnosticStatus prediction_diagnostic(
  const std::array<std::uint8_t, 16> & uuid,
  const ImmResult & prediction,
  const ImmUpdateReason reason)
{
  const auto selected = static_cast<std::size_t>(std::distance(
      prediction.model_probabilities.begin(),
      std::max_element(
        prediction.model_probabilities.begin(),
        prediction.model_probabilities.end())));
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.name = uuid_hex(uuid);
  status.message = "IMM measurement accepted";
  status.hardware_id = "ad_autoware_prediction";
  status.values = {
    key_value(
      "stationary_probability",
      probability_text(prediction.model_probabilities[0])),
    key_value(
      "constant_velocity_probability",
      probability_text(prediction.model_probabilities[1])),
    key_value(
      "coordinated_turn_probability",
      probability_text(prediction.model_probabilities[2])),
    key_value("selected_mode", model_name(selected)),
    key_value(
      "reset_or_gating_reason",
      [reason]() -> const char * {
        switch (reason) {
          case ImmUpdateReason::kTrackInitialized:
            return "track_initialized";
          case ImmUpdateReason::kMeasurementAccepted:
            return "measurement_accepted";
          case ImmUpdateReason::kRetentionExpired:
            return "retention_expired";
          case ImmUpdateReason::kClockRollback:
            return "clock_rollback";
          case ImmUpdateReason::kUpdateIntervalClamped:
            return "update_interval_clamped";
        }
        return "unknown_update_reason";
      }())};
  return status;
}

std::string rejection_reason(const std::string & message)
{
  if (message == "tracked objects are not in the odom frame") {
    return "rejected_frame_gate";
  }
  if (message == "tracked-object stamp is not newer" ||
    message == "tracked-object stamp must be strictly positive" ||
    message == "tracked-object stamp is malformed")
  {
    return "rejected_monotonic_stamp_gate";
  }
  if (message == "tracked-object stamp is in the future") {
    return "rejected_future_stamp_gate";
  }
  if (message == "tracked-object array is stale") {
    return "rejected_stale_array_gate";
  }
  if (message.rfind("IMM ", 0U) == 0U) {
    return "rejected_imm_update";
  }
  return "rejected_object_validation";
}

diagnostic_msgs::msg::DiagnosticStatus rejection_status(
  const std::string & name,
  const std::string & message,
  const std::string & reason)
{
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  status.name = name;
  status.message = "Tracked-object prediction input rejected";
  status.hardware_id = "ad_autoware_prediction";
  status.values = {
    key_value("reset_or_gating_reason", reason),
    key_value("rejection_detail", message)};
  return status;
}

} // namespace

diagnostic_msgs::msg::DiagnosticArray rejected_prediction_diagnostics(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const std::string & rejection_message,
  const std::optional<std::array<std::uint8_t, 16>> culprit_uuid)
{
  diagnostic_msgs::msg::DiagnosticArray output;
  output.header = input.header;
  const auto reason = rejection_reason(rejection_message);
  output.status.reserve(culprit_uuid.has_value() ? 2U : 1U);
  output.status.push_back(
    rejection_status("tracked_object_array", rejection_message, reason));
  if (culprit_uuid.has_value()) {
    output.status.push_back(
      rejection_status(
        uuid_hex(*culprit_uuid), rejection_message, reason));
  }
  return output;
}

ad_interfaces::msg::PredictedObjectArray adapt_tracked_objects(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_successful_stamp_ns,
  const AutowarePredictionAdapterConfig & config)
{
  validate_adapter_config(config);
  if (input.header.frame_id != config.expected_frame_id) {
    throw std::invalid_argument("tracked objects are not in the odom frame");
  }
  const std::int64_t input_stamp_ns = stamp_to_ns(input.header.stamp);
  if (last_successful_stamp_ns.has_value() &&
    input_stamp_ns <= *last_successful_stamp_ns)
  {
    throw std::invalid_argument("tracked-object stamp is not newer");
  }
  if (input_stamp_ns > now_ns) {
    throw std::invalid_argument("tracked-object stamp is in the future");
  }
  if (now_ns - input_stamp_ns > maximum_age_ns(config.maximum_input_age_sec)) {
    throw std::invalid_argument("tracked-object array is stale");
  }

  std::vector<ConvertedObject> converted;
  std::vector<TrackState2D> tracks;
  converted.reserve(input.objects.size());
  tracks.reserve(input.objects.size());
  for (const auto & object : input.objects) {
    converted.push_back(convert_object(object));
    tracks.push_back(converted.back().track);
  }
  const auto predictions = predict_tracks(tracks, config.prediction);

  ad_interfaces::msg::PredictedObjectArray output;
  output.header = input.header;
  output.objects.reserve(predictions.size());
  for (std::size_t index = 0; index < predictions.size(); ++index) {
    output.objects.push_back(
      map_prediction(
        input.objects[index], converted[index], predictions[index]));
  }
  return output;
}

rclcpp::QoS prediction_output_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

class StatefulImmPredictionAdapter::Impl
{
public:
  explicit Impl(AutowarePredictionAdapterConfig config)
  : config_(std::move(config))
  {
    validate_adapter_config(config_);
    if (!std::isfinite(config_.imm_track_retention_sec) ||
      config_.imm_track_retention_sec <= 0.0)
    {
      throw std::invalid_argument(
              "IMM track retention must be finite and positive");
    }
    ImmPredictor config_validator(config_.imm_prediction);
    (void)config_validator;
  }

  PredictionAdaptation adapt_with_diagnostics(
    const autoware_perception_msgs::msg::TrackedObjects & input,
    const std::int64_t now_ns,
    const std::optional<std::int64_t> last_successful_stamp_ns)
  {
    if (input.header.frame_id != config_.expected_frame_id) {
      throw std::invalid_argument("tracked objects are not in the odom frame");
    }
    const std::int64_t input_stamp_ns = stamp_to_ns(input.header.stamp);
    if (last_successful_stamp_ns.has_value() &&
      input_stamp_ns <= *last_successful_stamp_ns)
    {
      throw std::invalid_argument("tracked-object stamp is not newer");
    }
    if (input_stamp_ns > now_ns) {
      throw std::invalid_argument("tracked-object stamp is in the future");
    }
    if (now_ns - input_stamp_ns >
      maximum_age_ns(config_.maximum_input_age_sec))
    {
      throw std::invalid_argument("tracked-object array is stale");
    }

    std::vector<ConvertedObject> converted;
    std::vector<TrackMeasurement2D> measurements;
    converted.reserve(input.objects.size());
    measurements.reserve(input.objects.size());
    std::set<std::array<std::uint8_t, 16>> input_uuids;
    for (const auto & object : input.objects) {
      const auto key = object.object_id.uuid;
      if (!input_uuids.insert(key).second) {
        throw PredictionInputError("duplicate tracked-object UUID", key);
      }
      try {
        converted.push_back(convert_object(object));
        measurements.push_back(imm_measurement(object, converted.back()));
        validate_imm_measurement(measurements.back());
      } catch (const std::invalid_argument & error) {
        throw PredictionInputError(error.what(), key);
      }
      const auto existing = tracks_.find(key);
      if (existing != tracks_.end() &&
        input_stamp_ns <= existing->second.last_seen_stamp_ns)
      {
        throw PredictionInputError(
                "IMM measurement time must increase monotonically", key);
      }
    }

    auto staged_tracks = tracks_;
    auto staged_expiry_tombstones = expiry_tombstones_;
    const std::int64_t retention_ns =
      maximum_age_ns(config_.imm_track_retention_sec);
    prune_expiry_tombstones(
      staged_expiry_tombstones, input_stamp_ns, retention_ns);
    std::set<std::array<std::uint8_t, 16>> expired_tracks;
    for (auto entry = staged_tracks.begin(); entry != staged_tracks.end(); ) {
      if (input_stamp_ns > entry->second.last_seen_stamp_ns &&
        input_stamp_ns - entry->second.last_seen_stamp_ns > retention_ns)
      {
        expired_tracks.insert(entry->first);
        staged_expiry_tombstones[entry->first] = input_stamp_ns;
        entry = staged_tracks.erase(entry);
      } else {
        ++entry;
      }
    }

    PredictionAdaptation output;
    output.predictions.header = input.header;
    output.predictions.objects.reserve(input.objects.size());
    output.diagnostics.header = input.header;
    output.diagnostics.status.reserve(input.objects.size());
    for (std::size_t index = 0; index < input.objects.size(); ++index) {
      const auto key = input.objects[index].object_id.uuid;
      auto existing = staged_tracks.find(key);
      if (existing != staged_tracks.end() &&
        input_stamp_ns > existing->second.last_seen_stamp_ns &&
        input_stamp_ns - existing->second.last_seen_stamp_ns > retention_ns)
      {
        expired_tracks.insert(key);
        staged_expiry_tombstones[key] = input_stamp_ns;
        staged_tracks.erase(existing);
      }
      auto [entry, inserted] =
        staged_tracks.try_emplace(key, config_.imm_prediction, input_stamp_ns);
      ImmUpdateReason reason = ImmUpdateReason::kMeasurementAccepted;
      if (inserted) {
        if (pending_reset_reason_.has_value() &&
          *pending_reset_reason_ == ImmUpdateReason::kClockRollback)
        {
          reason = ImmUpdateReason::kClockRollback;
        } else if (expired_tracks.count(key) != 0U ||
          staged_expiry_tombstones.count(key) != 0U)
        {
          reason = ImmUpdateReason::kRetentionExpired;
          staged_expiry_tombstones.erase(key);
        } else {
          reason = ImmUpdateReason::kTrackInitialized;
        }
      } else {
        const double update_interval_s =
          static_cast<double>(
          input_stamp_ns - entry->second.last_seen_stamp_ns) * 1.0e-9;
        if (update_interval_s >
          config_.imm_prediction.maximum_update_interval_s)
        {
          reason = ImmUpdateReason::kUpdateIntervalClamped;
        }
      }
      ImmResult prediction;
      try {
        prediction = entry->second.predictor.update(
          measurements[index], input_stamp_ns);
      } catch (const std::invalid_argument & error) {
        throw PredictionInputError(error.what(), key);
      }
      entry->second.last_seen_stamp_ns = input_stamp_ns;
      output.predictions.objects.push_back(
        map_imm_prediction(
          input.objects[index], converted[index], prediction));
      output.diagnostics.status.push_back(
        prediction_diagnostic(key, prediction, reason));
    }
    prune_expiry_tombstones(
      staged_expiry_tombstones, input_stamp_ns, retention_ns);
    tracks_ = std::move(staged_tracks);
    expiry_tombstones_ = std::move(staged_expiry_tombstones);
    if (!input.objects.empty()) {
      pending_reset_reason_.reset();
    }
    return output;
  }

  void reset(const ImmUpdateReason reason) noexcept
  {
    tracks_.clear();
    expiry_tombstones_.clear();
    pending_reset_reason_ = reason;
  }

private:
  static constexpr std::size_t kMaximumExpiryTombstones = 4096U;

  static void prune_expiry_tombstones(
    std::map<std::array<std::uint8_t, 16>, std::int64_t> & tombstones,
    const std::int64_t input_stamp_ns,
    const std::int64_t retention_ns)
  {
    constexpr std::int64_t minimum_lifetime_ns = 60 * kNanosecondsPerSecond;
    constexpr std::int64_t scale = 64;
    const std::int64_t scaled_lifetime_ns =
      retention_ns > std::numeric_limits<std::int64_t>::max() / scale ?
      std::numeric_limits<std::int64_t>::max() : retention_ns * scale;
    const std::int64_t lifetime_ns =
      std::max(minimum_lifetime_ns, scaled_lifetime_ns);
    for (auto entry = tombstones.begin(); entry != tombstones.end(); ) {
      if (input_stamp_ns > entry->second &&
        input_stamp_ns - entry->second > lifetime_ns)
      {
        entry = tombstones.erase(entry);
      } else {
        ++entry;
      }
    }
    while (tombstones.size() > kMaximumExpiryTombstones) {
      const auto oldest = std::min_element(
        tombstones.begin(), tombstones.end(),
        [](const auto & lhs, const auto & rhs) {
          if (lhs.second != rhs.second) {
            return lhs.second < rhs.second;
          }
          return lhs.first < rhs.first;
        });
      tombstones.erase(oldest);
    }
  }

  struct TrackHistory
  {
    TrackHistory(ImmConfig config, const std::int64_t stamp_ns)
    : predictor(std::move(config)), last_seen_stamp_ns(stamp_ns) {}

    ImmPredictor predictor;
    std::int64_t last_seen_stamp_ns;
  };

  AutowarePredictionAdapterConfig config_;
  std::map<std::array<std::uint8_t, 16>, TrackHistory> tracks_;
  std::map<std::array<std::uint8_t, 16>, std::int64_t> expiry_tombstones_;
  std::optional<ImmUpdateReason> pending_reset_reason_;
};

StatefulImmPredictionAdapter::StatefulImmPredictionAdapter(
  AutowarePredictionAdapterConfig config)
: impl_(std::make_unique<Impl>(std::move(config))) {}

StatefulImmPredictionAdapter::~StatefulImmPredictionAdapter() = default;
StatefulImmPredictionAdapter::StatefulImmPredictionAdapter(
  StatefulImmPredictionAdapter &&) noexcept = default;
StatefulImmPredictionAdapter & StatefulImmPredictionAdapter::operator=(
  StatefulImmPredictionAdapter &&) noexcept = default;

ad_interfaces::msg::PredictedObjectArray StatefulImmPredictionAdapter::adapt(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_successful_stamp_ns)
{
  return impl_->adapt_with_diagnostics(
    input, now_ns, last_successful_stamp_ns).predictions;
}

PredictionAdaptation StatefulImmPredictionAdapter::adapt_with_diagnostics(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_successful_stamp_ns)
{
  return impl_->adapt_with_diagnostics(
    input, now_ns, last_successful_stamp_ns);
}

void StatefulImmPredictionAdapter::reset(const ImmUpdateReason reason) noexcept
{
  impl_->reset(reason);
}

AutowarePredictionNode::AutowarePredictionNode(
  const rclcpp::NodeOptions & options)
: Node("ad_autoware_prediction", options)
{
  const std::string input_topic = declare_parameter<std::string>(
    "input_topic", "/ad/perception/objects/tracked");
  const std::string output_topic = declare_parameter<std::string>(
    "output_topic", "/ad/perception/objects/predicted");
  const std::string diagnostic_topic = declare_parameter<std::string>(
    "diagnostic_topic", "/ad/perception/objects/prediction_debug");
  config_.expected_frame_id =
    declare_parameter<std::string>("expected_frame_id", "odom");
  config_.maximum_input_age_sec =
    declare_parameter<double>("maximum_input_age_sec", 0.5);
  config_.prediction.horizons_s =
    declare_parameter<std::vector<double>>("horizons_s", {0.5, 1.0});
  config_.prediction.acceleration_noise_std_mps2 =
    declare_parameter<double>("acceleration_noise_std_mps2", 1.5);
  config_.imm_prediction.horizons_s = config_.prediction.horizons_s;
  config_.imm_prediction.initial_probabilities = {
    declare_parameter<double>("imm.initial_probability.stationary", 0.20),
    declare_parameter<double>(
      "imm.initial_probability.constant_velocity",
      0.60),
    declare_parameter<double>(
      "imm.initial_probability.coordinated_turn",
      0.20)};
  config_.imm_prediction.maximum_update_interval_s =
    declare_parameter<double>("imm.maximum_update_interval_s", 2.0);
  config_.imm_prediction.stationary_process_variance =
    declare_parameter<double>("imm.process_variance.stationary", 0.02);
  config_.imm_prediction.constant_velocity_process_variance =
    declare_parameter<double>("imm.process_variance.constant_velocity", 0.20);
  config_.imm_prediction.coordinated_turn_process_variance =
    declare_parameter<double>("imm.process_variance.coordinated_turn", 0.35);
  config_.imm_track_retention_sec =
    declare_parameter<double>("imm.track_retention_sec", 1.0);
  validate_adapter_config(config_);
  imm_adapter_ = std::make_unique<StatefulImmPredictionAdapter>(config_);

  if (get_node_base_interface()->resolve_topic_or_service_name(
      input_topic,
      false) ==
    get_node_base_interface()->resolve_topic_or_service_name(
      output_topic,
      false))
  {
    throw std::invalid_argument(
            "tracked-object input and predicted-object output topics collide");
  }
  const auto resolved_diagnostic =
    get_node_base_interface()->resolve_topic_or_service_name(
    diagnostic_topic, false);
  if (resolved_diagnostic ==
    get_node_base_interface()->resolve_topic_or_service_name(input_topic, false) ||
    resolved_diagnostic ==
    get_node_base_interface()->resolve_topic_or_service_name(output_topic, false))
  {
    throw std::invalid_argument(
            "prediction diagnostic topic must be distinct");
  }

  publisher_ = create_publisher<ad_interfaces::msg::PredictedObjectArray>(
    output_topic, prediction_output_qos());
  diagnostic_publisher_ =
    create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostic_topic,
    rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
  subscription_ =
    create_subscription<autoware_perception_msgs::msg::TrackedObjects>(
    input_topic,
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile(),
    [this](const autoware_perception_msgs::msg::TrackedObjects::
    ConstSharedPtr input) {on_tracked_objects(input);});
}

void AutowarePredictionNode::on_tracked_objects(
  const autoware_perception_msgs::msg::TrackedObjects::ConstSharedPtr input)
{
  try {
    const std::int64_t input_stamp_ns = stamp_to_ns(input->header.stamp);
    if (last_successful_stamp_ns_.has_value() &&
      input_stamp_ns < *last_successful_stamp_ns_)
    {
      // MORAI resets simulated time together with object tracks. Clear the
      // per-object IMM history so the new run can start cleanly.
      imm_adapter_->reset(ImmUpdateReason::kClockRollback);
      last_successful_stamp_ns_.reset();
    }
    auto output = imm_adapter_->adapt_with_diagnostics(
      *input, now().nanoseconds(),
      last_successful_stamp_ns_);
    publisher_->publish(output.predictions);
    diagnostic_publisher_->publish(output.diagnostics);
    last_successful_stamp_ns_ = stamp_to_ns(input->header.stamp);
  } catch (const std::exception & error) {
    std::optional<std::array<std::uint8_t, 16>> culprit_uuid;
    if (const auto * input_error =
      dynamic_cast<const PredictionInputError *>(&error))
    {
      culprit_uuid = input_error->culprit_uuid();
    }
    diagnostic_publisher_->publish(
      rejected_prediction_diagnostics(*input, error.what(), culprit_uuid));
    RCLCPP_WARN(
      get_logger(), "Rejected tracked-object array: %s",
      error.what());
  }
}

} // namespace ad_lidar_perception::tracking
