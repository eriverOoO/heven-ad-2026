#include "roundabout_gap_response_node.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_planner
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

std::int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    throw std::invalid_argument("stamp is malformed");
  }
  const auto value = static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(stamp.nanosec);
  if (value <= 0) {
    throw std::invalid_argument("stamp must be strictly positive");
  }
  return value;
}

std::int64_t seconds_to_ns(const double seconds)
{
  const long double value = static_cast<long double>(seconds) * kNanosecondsPerSecond;
  if (!std::isfinite(value) || value < 0.0L ||
    value > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument("duration is not representable");
  }
  return static_cast<std::int64_t>(value);
}

std::size_t positive_size(const int value, const char * const name)
{
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

std::size_t nonnegative_size(const int value, const char * const name)
{
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be nonnegative");
  }
  return static_cast<std::size_t>(value);
}

diagnostic_msgs::msg::KeyValue key_value(const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue output;
  output.key = key;
  output.value = value;
  return output;
}

RoundaboutGapResponseFrameResult rejected(std::string reason)
{
  RoundaboutGapResponseFrameResult output;
  output.reject_reason = std::move(reason);
  return output;
}

bool all_finite(const ad_interfaces::msg::RoundaboutGapRisk & risk)
{
  return std::isfinite(risk.classification_probability) &&
         std::isfinite(risk.existence_probability) &&
         std::isfinite(risk.object_map_distance_to_conflict_m) &&
         std::isfinite(risk.object_entry_time_s) && std::isfinite(risk.object_exit_time_s) &&
         std::isfinite(risk.arrival_delta_s) && std::isfinite(risk.temporal_gap_s) &&
         std::isfinite(risk.minimum_temporal_gap_s) &&
         std::isfinite(risk.prediction_horizon_s) && std::isfinite(risk.ttc_s) &&
         std::isfinite(risk.cpa_time_s) && std::isfinite(risk.cpa_distance_m) &&
         std::isfinite(risk.predicted_min_separation_m) &&
         std::isfinite(risk.predicted_min_separation_time_s);
}

bool valid_nonnegative(const bool valid, const float value)
{
  return !valid || value >= 0.0F;
}

std::uint8_t action_constant(const RoundaboutGapResponseAction action)
{
  switch (action) {
    case RoundaboutGapResponseAction::kYield:
      return ad_interfaces::msg::RoundaboutGapResponse::ACTION_YIELD;
    case RoundaboutGapResponseAction::kHold:
      return ad_interfaces::msg::RoundaboutGapResponse::ACTION_HOLD;
    case RoundaboutGapResponseAction::kRelease:
    default:
      return ad_interfaces::msg::RoundaboutGapResponse::ACTION_RELEASE;
  }
}

std::uint8_t reason_constant(const RoundaboutGapResponseReason reason)
{
  return static_cast<std::uint8_t>(reason);
}

double percentile(std::vector<double> values, const double quantile)
{
  if (values.empty()) {return 0.0;}
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(std::ceil(
      quantile * static_cast<double>(values.size()))) - 1U;
  return values[std::min(index, values.size() - 1U)];
}

}  // namespace

RoundaboutGapResponseFrameResult build_roundabout_gap_response_frame(
  const ad_interfaces::msg::RoundaboutGapRiskArray & risks,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_risk_stamp_ns,
  const RoundaboutGapResponseFrameConfig & config)
{
  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(risks.header.stamp);
    if (!(config.maximum_input_age_s > 0.0)) {
      return rejected("response frame configuration is invalid");
    }
    if (risks.header.frame_id != config.expected_frame_id) {
      return rejected("roundabout risk frame_id is unexpected");
    }
    if (risks.conflict_zone_id.empty() ||
      risks.conflict_zone_id != config.expected_conflict_zone_id)
    {
      return rejected("roundabout conflict zone is invalid");
    }
    if (last_risk_stamp_ns.has_value() && stamp_ns <= *last_risk_stamp_ns) {
      return rejected("roundabout risk array is a duplicate or backward frame");
    }
    if (stamp_ns > now_ns) {
      return rejected("roundabout risk array stamp is in the future");
    }
    if (now_ns - stamp_ns > seconds_to_ns(config.maximum_input_age_s)) {
      return rejected("roundabout risk array is stale");
    }
  } catch (const std::exception & error) {
    return rejected(error.what());
  }

  if (!std::isfinite(risks.ego_entry_time_s) || !std::isfinite(risks.ego_exit_time_s) ||
    !std::isfinite(risks.ego_route_distance_to_entry_m) ||
    !std::isfinite(risks.ego_route_distance_to_exit_m) ||
    !std::isfinite(risks.ego_speed_mps) || risks.ego_speed_mps < 0.0F ||
    !valid_nonnegative(risks.ego_entry_valid, risks.ego_entry_time_s) ||
    !valid_nonnegative(risks.ego_exit_valid, risks.ego_exit_time_s) ||
    (risks.ego_entry_valid && risks.ego_exit_valid &&
    risks.ego_exit_time_s < risks.ego_entry_time_s))
  {
    return rejected("roundabout ego facts are non-finite");
  }
  const auto actual_relevant = static_cast<std::size_t>(std::count_if(
      risks.objects.begin(), risks.objects.end(),
      [](const auto & risk) {return risk.relevant_to_conflict;}));
  if (actual_relevant != risks.relevant_object_count) {
    return rejected("relevant_object_count is inconsistent with objects");
  }
  if (actual_relevant > config.policy.maximum_relevant_objects) {
    return rejected("relevant object count exceeds policy budget");
  }

  RoundaboutGapResponseInput input;
  input.ego_in_conflict_now = risks.ego_in_conflict_now;
  input.ego_entry_valid = risks.ego_entry_valid;
  input.ego_entry_time_s = risks.ego_entry_time_s;
  input.ego_exit_valid = risks.ego_exit_valid;
  input.ego_exit_time_s = risks.ego_exit_time_s;
  input.ego_route_distance_to_entry_m = risks.ego_route_distance_to_entry_m;
  input.ego_route_distance_to_exit_m = risks.ego_route_distance_to_exit_m;
  input.ego_speed_mps = risks.ego_speed_mps;
  input.relevant_objects.reserve(actual_relevant);

  for (const auto & risk : risks.objects) {
    const bool interval_summary_expected = risks.ego_entry_valid && risks.ego_exit_valid &&
      risk.predicted_conflict_interval_count > 0U;
    if (!all_finite(risk) || risk.classification_probability < 0.0F ||
      risk.classification_probability > 1.0F || risk.existence_probability < 0.0F ||
      risk.existence_probability > 1.0F || risk.object_map_distance_to_conflict_m < 0.0F ||
      !valid_nonnegative(risk.object_entry_valid, risk.object_entry_time_s) ||
      !valid_nonnegative(risk.object_exit_valid, risk.object_exit_time_s) ||
      !valid_nonnegative(risk.temporal_gap_valid, risk.temporal_gap_s) ||
      !valid_nonnegative(risk.minimum_temporal_gap_valid, risk.minimum_temporal_gap_s) ||
      risk.prediction_horizon_s < 0.0F ||
      (risk.later_reentry_detected != (risk.predicted_conflict_interval_count > 1U)) ||
      (risk.minimum_temporal_gap_valid != interval_summary_expected) ||
      (risk.any_occupancy_overlap &&
      (!risk.minimum_temporal_gap_valid || risk.minimum_temporal_gap_s != 0.0F)) ||
      (risk.any_occupancy_overlap && !interval_summary_expected) ||
      (risk.prediction_covers_ego_exit &&
      (!risks.ego_exit_valid ||
      risk.prediction_horizon_s + 1.0e-3F < risks.ego_exit_time_s)))
    {
      return rejected("roundabout object facts are malformed or inconsistent");
    }
    if (!risk.relevant_to_conflict) {
      continue;
    }
    if (risk.predicted_conflict_interval_count == 0U ||
      (risks.ego_entry_valid && risks.ego_exit_valid &&
      !risk.minimum_temporal_gap_valid))
    {
      return rejected("relevant object interval summary is inconsistent");
    }
    RoundaboutGapResponseObjectInput object;
    std::copy(risk.object_id.uuid.begin(), risk.object_id.uuid.end(), object.object_id.begin());
    object.any_occupancy_overlap = risk.any_occupancy_overlap;
    object.minimum_temporal_gap_valid = risk.minimum_temporal_gap_valid;
    object.minimum_temporal_gap_s = risk.minimum_temporal_gap_s;
    object.prediction_covers_ego_exit = risk.prediction_covers_ego_exit;
    input.relevant_objects.push_back(object);
  }

  RoundaboutGapResponseResult policy;
  try {
    policy = compute_roundabout_gap_response(input, config.policy);
  } catch (const std::exception & error) {
    return rejected(std::string("roundabout response policy rejected facts: ") + error.what());
  }

  RoundaboutGapResponseFrameResult result;
  result.published = true;
  result.action = policy.action;
  auto & output = result.output;
  output.header = risks.header;
  output.action = action_constant(policy.action);
  output.active = policy.active;
  output.reason = reason_constant(policy.reason);
  output.conflict_zone_id = risks.conflict_zone_id;
  output.source_object_valid = policy.has_source;
  if (policy.has_source) {
    std::copy(
      policy.source_object_id.begin(), policy.source_object_id.end(),
      output.source_object_id.uuid.begin());
  }
  output.relevant_object_count = static_cast<std::uint16_t>(policy.relevant_object_count);
  output.ego_speed_mps = static_cast<float>(policy.ego_speed_mps);
  output.ego_route_distance_to_entry_m =
    static_cast<float>(policy.ego_route_distance_to_entry_m);
  output.available_distance_m = static_cast<float>(policy.available_distance_m);
  output.comfortable_stop_distance_m = static_cast<float>(policy.comfortable_stop_distance_m);
  output.limiting_gap_valid = policy.limiting_gap_valid;
  output.limiting_gap_s = static_cast<float>(
    policy.limiting_gap_valid ? policy.limiting_gap_s : 0.0);
  output.conflict_overlap_present = policy.conflict_overlap_present;
  output.complete_prediction_coverage = policy.complete_prediction_coverage;
  return result;
}

RoundaboutGapResponseNode::RoundaboutGapResponseNode(const rclcpp::NodeOptions & options)
: Node("ad_roundabout_gap_response", options)
{
  const auto input_topic = declare_parameter<std::string>(
    "topics.roundabout_gap_risks", "/ad/planning/roundabout_gap_risks");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/roundabout_gap_response");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/roundabout_gap_response/diagnostics");
  config_.expected_frame_id = declare_parameter<std::string>("expected_frame_id", "map");
  config_.expected_conflict_zone_id = declare_parameter<std::string>(
    "conflict_zone_id", "kcity_roundabout");
  config_.maximum_input_age_s = declare_parameter<double>("maximum_input_age_s", 0.5);
  config_.policy.minimum_release_gap_s =
    declare_parameter<double>("minimum_release_gap_s", 2.0);
  config_.policy.entry_standoff_m = declare_parameter<double>("entry_standoff_m", 6.0);
  config_.policy.comfortable_deceleration_mps2 =
    declare_parameter<double>("comfortable_deceleration_mps2", 1.8);
  config_.policy.stopped_speed_threshold_mps =
    declare_parameter<double>("stopped_speed_threshold_mps", 0.5);
  config_.policy.maximum_relevant_objects = positive_size(
    declare_parameter<int>("maximum_relevant_objects", 256), "maximum_relevant_objects");
  config_.policy = config_.policy.validated();
  runtime_summary_interval_frames_ = nonnegative_size(
    declare_parameter<int>("runtime_summary_interval_frames", 200),
    "runtime_summary_interval_frames");
  if (input_topic.empty() || output_topic_.empty() || diagnostics_topic_.empty() ||
    config_.expected_frame_id.empty() || config_.expected_conflict_zone_id.empty())
  {
    throw std::invalid_argument("roundabout response topics/context must be nonempty");
  }

  publisher_ = create_publisher<ad_interfaces::msg::RoundaboutGapResponse>(
    output_topic_, rclcpp::QoS(1).reliable().durability_volatile());
  diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
  subscription_ = create_subscription<ad_interfaces::msg::RoundaboutGapRiskArray>(
    input_topic, rclcpp::QoS(1).reliable().durability_volatile(),
    [this](ad_interfaces::msg::RoundaboutGapRiskArray::ConstSharedPtr message) {
      on_risks(message);
    });
  RCLCPP_INFO(
    get_logger(), "ad_roundabout_gap_response: %s -> %s",
    input_topic.c_str(), output_topic_.c_str());
}

void RoundaboutGapResponseNode::on_risks(
  const ad_interfaces::msg::RoundaboutGapRiskArray::ConstSharedPtr message)
{
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  ++received_;
  const auto result = build_roundabout_gap_response_frame(
    *message, now().nanoseconds(), last_risk_stamp_ns_, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  if (result.published) {
    publisher_->publish(result.output);
    last_risk_stamp_ns_ = stamp_to_ns(message->header.stamp);
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Roundabout response frame rejected: %s", result.reject_reason.c_str());
  }
  publish_diagnostics(message->header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void RoundaboutGapResponseNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const RoundaboutGapResponseFrameResult & result,
  const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "roundabout_gap_response";
  status.hardware_id = output_topic_;
  status.level = result.published ? diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.published ? "ok" : result.reject_reason;
  status.values.push_back(key_value("published", result.published ? "true" : "false"));
  status.values.push_back(key_value("reject_reason", result.reject_reason));
  status.values.push_back(key_value("action", std::to_string(result.output.action)));
  status.values.push_back(key_value("reason", std::to_string(result.output.reason)));
  status.values.push_back(key_value("active", result.output.active ? "true" : "false"));
  status.values.push_back(
    key_value(
      "relevant_object_count", std::to_string(result.output.relevant_object_count)));
  status.values.push_back(
    key_value(
      "complete_prediction_coverage",
      result.output.complete_prediction_coverage ? "true" : "false"));
  status.values.push_back(
    key_value(
      "conflict_overlap_present", result.output.conflict_overlap_present ? "true" : "false"));
  status.values.push_back(
    key_value(
      "limiting_gap_s",
      std::to_string(result.output.limiting_gap_s)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void RoundaboutGapResponseNode::record_runtime(
  const RoundaboutGapResponseFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++published_;
    latency_ms_.push_back(latency_ms);
    if (result.action == RoundaboutGapResponseAction::kRelease) {
      ++release_;
    } else if (result.action == RoundaboutGapResponseAction::kYield) {++yield_;} else {++hold_;}
    if (result.output.reason < reasons_.size()) {++reasons_[result.output.reason];}
  } else {
    ++rejected_;
  }
  if (runtime_summary_interval_frames_ > 0U && received_ % runtime_summary_interval_frames_ == 0U) {
    const double max_latency = latency_ms_.empty() ? 0.0 :
      *std::max_element(latency_ms_.begin(), latency_ms_.end());
    RCLCPP_INFO(
      get_logger(),
      "ROUNDABOUT_GAP_RESPONSE_RUNTIME_SUMMARY received=%zu published=%zu release=%zu "
      "yield=%zu hold=%zu rejected=%zu reasons=%zu,%zu,%zu,%zu,%zu,%zu "
      "latency_ms_median=%.6f latency_ms_p95=%.6f latency_ms_max=%.6f",
      received_, published_, release_, yield_, hold_, rejected_, reasons_[0], reasons_[1],
      reasons_[2], reasons_[3], reasons_[4], reasons_[5], percentile(latency_ms_, 0.5),
      percentile(latency_ms_, 0.95), max_latency);
  }
}

}  // namespace ad_planner
