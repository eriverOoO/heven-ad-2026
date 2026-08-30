#include "dynamic_object_risk_node.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_lidar_perception::planning
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

std::int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    throw std::invalid_argument("stamp is malformed");
  }
  const std::int64_t nanoseconds =
    static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(stamp.nanosec);
  if (nanoseconds <= 0) {
    throw std::invalid_argument("stamp must be strictly positive");
  }
  return nanoseconds;
}

std::int64_t age_limit_ns(const double seconds)
{
  const long double nanoseconds =
    static_cast<long double>(seconds) * static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(static_cast<double>(nanoseconds)) || nanoseconds <= 0.0L) {
    return 0;
  }
  return static_cast<std::int64_t>(nanoseconds);
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

RiskFrameResult rejected(std::string reason)
{
  RiskFrameResult result;
  result.published = false;
  result.reason = std::move(reason);
  return result;
}

PredictedObjectInput unpack_object(const ad_interfaces::msg::PredictedObject & source)
{
  PredictedObjectInput object;
  std::copy(
    source.object_id.uuid.begin(), source.object_id.uuid.end(),
    object.object_id.begin());
  object.classification = source.classification;
  object.classification_probability = source.classification_probability;
  object.existence_probability = source.existence_probability;
  object.x_m = source.initial_pose.pose.position.x;
  object.y_m = source.initial_pose.pose.position.y;
  object.length_m = source.dimensions.x;
  object.width_m = source.dimensions.y;
  object.vx_world_mps = source.initial_twist.twist.linear.x;
  object.vy_world_mps = source.initial_twist.twist.linear.y;
  object.position_covariance_xy = {
    source.initial_pose.covariance[0], source.initial_pose.covariance[1],
    source.initial_pose.covariance[6], source.initial_pose.covariance[7]};
  object.predicted_points.reserve(source.states.size());
  for (const auto & state : source.states) {
    PredictedPoint point;
    point.time_s = static_cast<double>(state.time_from_start.sec) +
      static_cast<double>(state.time_from_start.nanosec) * 1.0e-9;
    point.x_m = state.pose.pose.position.x;
    point.y_m = state.pose.pose.position.y;
    object.predicted_points.push_back(point);
  }
  return object;
}

ad_interfaces::msg::DynamicObjectRisk serialize(const DynamicObjectRiskResult & source)
{
  ad_interfaces::msg::DynamicObjectRisk output;
  std::copy(
    source.object_id.begin(), source.object_id.end(),
    output.object_id.uuid.begin());
  output.classification = source.classification;
  output.classification_probability = source.classification_probability;
  output.existence_probability = source.existence_probability;
  output.x_rel_m = static_cast<float>(source.x_rel_m);
  output.y_rel_m = static_cast<float>(source.y_rel_m);
  output.distance_m = static_cast<float>(source.distance_m);
  output.vx_rel_mps = static_cast<float>(source.vx_rel_mps);
  output.vy_rel_mps = static_cast<float>(source.vy_rel_mps);
  output.relative_speed_mps = static_cast<float>(source.relative_speed_mps);
  output.range_rate_mps = static_cast<float>(source.range_rate_mps);
  output.longitudinal_closing_mps = static_cast<float>(source.longitudinal_closing_mps);
  output.ttc_valid = source.ttc_valid;
  output.ttc_s = static_cast<float>(source.ttc_valid ? source.ttc_s : 0.0);
  output.cpa_valid = source.cpa_valid;
  output.cpa_time_s = static_cast<float>(source.cpa_valid ? source.cpa_time_s : 0.0);
  output.cpa_distance_m = static_cast<float>(source.cpa_valid ? source.cpa_distance_m : 0.0);
  output.predicted_min_separation_valid = source.predicted_min_separation_valid;
  output.predicted_min_separation_m = static_cast<float>(
    source.predicted_min_separation_valid ? source.predicted_min_separation_m : 0.0);
  output.predicted_min_separation_time_s = static_cast<float>(
    source.predicted_min_separation_valid ? source.predicted_min_separation_time_s : 0.0);
  output.predicted_states.reserve(source.predicted_states.size());
  for (const auto & source_state : source.predicted_states) {
    ad_interfaces::msg::DynamicObjectRiskState state;
    const double whole_seconds = std::floor(source_state.time_s);
    state.time_from_start.sec = static_cast<std::int32_t>(whole_seconds);
    state.time_from_start.nanosec = static_cast<std::uint32_t>(std::llround(
      (source_state.time_s - whole_seconds) * 1.0e9));
    if (state.time_from_start.nanosec == 1'000'000'000U) {
      ++state.time_from_start.sec;
      state.time_from_start.nanosec = 0U;
    }
    state.x_rel_m = static_cast<float>(source_state.x_rel_m);
    state.y_rel_m = static_cast<float>(source_state.y_rel_m);
    output.predicted_states.push_back(std::move(state));
  }
  output.position_uncertainty_m = static_cast<float>(source.position_uncertainty_m);
  return output;
}

diagnostic_msgs::msg::KeyValue key_value(const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue entry;
  entry.key = key;
  entry.value = value;
  return entry;
}

}  // namespace

RiskFrameResult build_risk_frame(
  const ad_interfaces::msg::PredictedObjectArray & prediction,
  const std::optional<EgoSample> & ego,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_prediction_stamp_ns,
  const RiskNodeConfig & config)
{
  std::int64_t prediction_stamp_ns = 0;
  try {
    prediction_stamp_ns = stamp_to_ns(prediction.header.stamp);
  } catch (const std::exception & error) {
    return rejected(std::string("prediction ") + error.what());
  }

  if (prediction.header.frame_id != config.prediction_frame_id) {
    return rejected("prediction frame_id is not '" + config.prediction_frame_id + "'");
  }
  if (last_prediction_stamp_ns.has_value() &&
    prediction_stamp_ns <= *last_prediction_stamp_ns)
  {
    return rejected("prediction is a duplicate or backward frame");
  }

  const std::int64_t future_skew_ns = age_limit_ns(config.maximum_future_skew_s);
  if (prediction_stamp_ns > now_ns + future_skew_ns) {
    return rejected("prediction stamp is in the future");
  }
  const std::int64_t prediction_age_limit_ns = age_limit_ns(config.maximum_prediction_age_s);
  if (prediction_age_limit_ns > 0 && now_ns - prediction_stamp_ns > prediction_age_limit_ns) {
    return rejected("prediction is stale");
  }

  if (!ego.has_value()) {
    return rejected("ego state is unavailable");
  }
  const std::int64_t ego_age_limit_ns = age_limit_ns(config.maximum_ego_age_s);
  const std::int64_t ego_skew_ns = std::llabs(prediction_stamp_ns - ego->stamp_ns);
  if (ego_age_limit_ns > 0 && ego_skew_ns > ego_age_limit_ns) {
    return rejected("ego state is stale relative to the prediction");
  }

  std::vector<PredictedObjectInput> objects;
  objects.reserve(prediction.objects.size());
  for (const auto & source : prediction.objects) {
    objects.push_back(unpack_object(source));
  }

  DynamicObjectRiskComputation computation;
  try {
    computation = compute_dynamic_object_risks(ego->state, objects, config.risk);
  } catch (const std::exception & error) {
    return rejected(std::string("ego state is non-finite: ") + error.what());
  }

  RiskFrameResult result;
  result.published = true;
  result.objects_in = prediction.objects.size();
  result.rejected_non_finite_state = computation.rejected_non_finite_state;
  result.rejected_non_finite_dimensions = computation.rejected_non_finite_dimensions;
  result.rejected_over_budget = computation.rejected_over_budget;
  result.risks.header = prediction.header;
  result.risks.header.frame_id = config.output_frame_id;
  result.risks.objects.reserve(computation.objects.size());
  for (const auto & risk : computation.objects) {
    if (risk.ttc_valid) {
      ++result.ttc_valid_count;
    }
    if (risk.cpa_valid) {
      ++result.cpa_valid_count;
    }
    result.risks.objects.push_back(serialize(risk));
  }
  result.objects_out = result.risks.objects.size();
  return result;
}

DynamicObjectRiskNode::DynamicObjectRiskNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("ad_dynamic_object_risk", options)
{
  const std::string prediction_topic = declare_parameter<std::string>(
    "prediction_topic", "/ad/perception/objects/predicted");
  const std::string odometry_topic = declare_parameter<std::string>(
    "odometry_topic", "/ad/localization/odometry");
  output_topic_ = declare_parameter<std::string>(
    "output_topic", "/ad/planning/dynamic_object_risks");
  const std::string diagnostic_topic = declare_parameter<std::string>(
    "diagnostic_topic", "/ad/planning/dynamic_object_risks/diagnostics");

  config_.prediction_frame_id =
    declare_parameter<std::string>("prediction_frame_id", "odom");
  config_.output_frame_id =
    declare_parameter<std::string>("output_frame_id", "base_link");
  config_.maximum_prediction_age_s =
    declare_parameter<double>("maximum_prediction_age_s", 0.5);
  config_.maximum_ego_age_s = declare_parameter<double>("maximum_ego_age_s", 0.5);
  config_.maximum_future_skew_s =
    declare_parameter<double>("maximum_future_skew_s", 0.10);
  config_.risk.closing_speed_epsilon_mps =
    declare_parameter<double>("closing_speed_epsilon_mps", 0.05);
  config_.risk.cpa_horizon_s = declare_parameter<double>("cpa_horizon_s", 6.0);
  config_.risk.ttc_horizon_s = declare_parameter<double>("ttc_horizon_s", 6.0);
  config_.risk.ego_half_length_m =
    declare_parameter<double>("ego_half_length_m", 2.5);
  config_.risk.ego_half_width_m =
    declare_parameter<double>("ego_half_width_m", 1.1);
  config_.risk.minimum_object_radius_m =
    declare_parameter<double>("minimum_object_radius_m", 0.30);
  config_.risk.maximum_objects = static_cast<std::size_t>(
    declare_parameter<int>("maximum_objects", 256));
  runtime_summary_interval_frames_ = static_cast<std::size_t>(
    declare_parameter<int>("runtime_summary_interval_frames", 180));

  config_.risk = config_.risk.validated();

  if (config_.maximum_prediction_age_s <= 0.0 || config_.maximum_ego_age_s <= 0.0 ||
    config_.maximum_future_skew_s < 0.0)
  {
    throw std::invalid_argument(
            "dynamic object risk age / skew limits must be positive (skew >= 0)");
  }
  if (output_topic_ == prediction_topic || output_topic_ == odometry_topic ||
    diagnostic_topic == output_topic_)
  {
    throw std::invalid_argument("dynamic object risk topics must be distinct");
  }

  risk_publisher_ = create_publisher<ad_interfaces::msg::DynamicObjectRiskArray>(
    output_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile());
  diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostic_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());

  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
    [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {on_odometry(message);});
  prediction_subscription_ = create_subscription<ad_interfaces::msg::PredictedObjectArray>(
    prediction_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile(),
    [this](ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr message) {
      on_prediction(message);
    });

  RCLCPP_INFO(
    get_logger(),
    "ad_dynamic_object_risk: %s + %s -> %s (frame %s)",
    prediction_topic.c_str(), odometry_topic.c_str(), output_topic_.c_str(),
    config_.output_frame_id.c_str());
}

void DynamicObjectRiskNode::on_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr message)
{
  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(message->header.stamp);
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Ignoring odometry: %s", error.what());
    return;
  }

  EgoSample sample;
  sample.stamp_ns = stamp_ns;
  sample.state.x_m = message->pose.pose.position.x;
  sample.state.y_m = message->pose.pose.position.y;
  sample.state.yaw_rad = yaw_from_quaternion(message->pose.pose.orientation);
  // Canonical localization publishes a body-frame longitudinal speed; lateral
  // and yaw-rate are structurally unobserved and treated as zero.
  sample.state.longitudinal_speed_mps = message->twist.twist.linear.x;

  const bool finite_sample = std::isfinite(sample.state.x_m) &&
    std::isfinite(sample.state.y_m) && std::isfinite(sample.state.yaw_rad) &&
    std::isfinite(sample.state.longitudinal_speed_mps);
  if (!finite_sample) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Ignoring non-finite odometry sample");
    return;
  }

  if (last_ego_position_.has_value()) {
    const double moved = std::hypot(
      sample.state.x_m - last_ego_position_->first,
      sample.state.y_m - last_ego_position_->second);
    if (moved > 0.5 && std::abs(sample.state.longitudinal_speed_mps) < 0.05) {
      ++ego_twist_contradiction_frames_;
    }
  }
  last_ego_position_ = std::make_pair(sample.state.x_m, sample.state.y_m);
  latest_ego_ = sample;
}

void DynamicObjectRiskNode::on_prediction(
  const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr message)
{
  const auto started = std::chrono::steady_clock::now();

  std::optional<std::int64_t> effective_last = last_prediction_stamp_ns_;
  try {
    const std::int64_t stamp_ns = stamp_to_ns(message->header.stamp);
    if (last_prediction_stamp_ns_.has_value() && stamp_ns < *last_prediction_stamp_ns_ &&
      *last_prediction_stamp_ns_ - stamp_ns > config_.clock_rollback_threshold_ns)
    {
      // Simulated-time reset: forget the previous frame so the new run starts.
      last_prediction_stamp_ns_.reset();
      last_ego_position_.reset();
      effective_last.reset();
    }
  } catch (const std::exception &) {
    // build_risk_frame reports the malformed stamp below.
  }

  const std::int64_t now_ns = now().nanoseconds();
  const RiskFrameResult result =
    build_risk_frame(*message, latest_ego_, now_ns, effective_last, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();

  if (result.published) {
    risk_publisher_->publish(result.risks);
    try {
      last_prediction_stamp_ns_ = stamp_to_ns(message->header.stamp);
    } catch (const std::exception &) {
    }
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Rejected prediction frame: %s",
      result.reason.c_str());
  }

  publish_diagnostics(message->header, result, latency_ms);
  record_runtime_metrics(result, latency_ms);
}

void DynamicObjectRiskNode::publish_diagnostics(
  const std_msgs::msg::Header & header, const RiskFrameResult & result,
  const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "dynamic_object_risk";
  status.hardware_id = output_topic_;
  status.level = result.published ? diagnostic_msgs::msg::DiagnosticStatus::OK
    : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.published ? "ok" : result.reason;
  status.values.push_back(key_value("published", result.published ? "true" : "false"));
  status.values.push_back(key_value("reject_reason", result.reason));
  status.values.push_back(key_value("objects_in", std::to_string(result.objects_in)));
  status.values.push_back(key_value("objects_out", std::to_string(result.objects_out)));
  status.values.push_back(
    key_value("rejected_non_finite_state", std::to_string(result.rejected_non_finite_state)));
  status.values.push_back(
    key_value(
      "rejected_non_finite_dimensions",
      std::to_string(result.rejected_non_finite_dimensions)));
  status.values.push_back(
    key_value("rejected_over_budget", std::to_string(result.rejected_over_budget)));
  status.values.push_back(key_value("ttc_valid", std::to_string(result.ttc_valid_count)));
  status.values.push_back(key_value("cpa_valid", std::to_string(result.cpa_valid_count)));
  status.values.push_back(
    key_value(
      "ego_twist_contradiction_frames",
      std::to_string(ego_twist_contradiction_frames_)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostic_publisher_->publish(array);
}

void DynamicObjectRiskNode::record_runtime_metrics(
  const RiskFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++risk_frames_;
    objects_in_total_ += result.objects_in;
    objects_out_total_ += result.objects_out;
    ttc_valid_total_ += result.ttc_valid_count;
    cpa_valid_total_ += result.cpa_valid_count;
  } else {
    ++rejected_frames_;
  }
  rejected_objects_total_ += result.rejected_non_finite_state +
    result.rejected_non_finite_dimensions + result.rejected_over_budget;
  ++predicted_frames_;
  latency_ms_.push_back(latency_ms);

  const auto interval = runtime_summary_interval_frames_;
  if (interval == 0U || latency_ms_.size() % interval != 0U) {
    return;
  }
  auto ordered = latency_ms_;
  std::sort(ordered.begin(), ordered.end());
  const auto quantile = [&ordered](const double q) {
      if (ordered.empty()) {
        return 0.0;
      }
      const auto index = static_cast<std::size_t>(
        std::clamp(q, 0.0, 1.0) * static_cast<double>(ordered.size() - 1));
      return ordered[index];
    };
  RCLCPP_INFO(
    get_logger(),
    "DYNAMIC_OBJECT_RISK_RUNTIME_SUMMARY frames=%zu published=%zu rejected=%zu "
    "objects_in=%zu objects_out=%zu rejected_objects=%zu ttc_valid=%zu "
    "cpa_valid=%zu ego_twist_contradiction_frames=%zu "
    "latency_ms_median=%.4f latency_ms_p95=%.4f latency_ms_max=%.4f",
    predicted_frames_, risk_frames_, rejected_frames_, objects_in_total_,
    objects_out_total_, rejected_objects_total_, ttc_valid_total_,
    cpa_valid_total_, ego_twist_contradiction_frames_, quantile(0.5),
    quantile(0.95), ordered.back());
}

}  // namespace ad_lidar_perception::planning
