#include "highway_merge_gap_response_node.hpp"

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
constexpr double kStationToleranceM = 1.0e-3;

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

HighwayMergeGapResponseFrameResult rejected(std::string reason)
{
  HighwayMergeGapResponseFrameResult output;
  output.reject_reason = std::move(reason);
  return output;
}

bool object_finite(const ad_interfaces::msg::HighwayMergeGapRisk & risk)
{
  return std::isfinite(risk.classification_probability) &&
         std::isfinite(risk.existence_probability) &&
         std::isfinite(risk.object_route_s_m) &&
         std::isfinite(risk.object_lateral_offset_m) &&
         std::isfinite(risk.predicted_corridor_entry_time_s) &&
         std::isfinite(risk.delta_s_now_m) &&
         std::isfinite(risk.object_longitudinal_speed_mps) &&
         std::isfinite(risk.relative_longitudinal_speed_mps) &&
         std::isfinite(risk.delta_s_at_merge_m) &&
         std::isfinite(risk.longitudinal_closing_speed_mps) &&
         std::isfinite(risk.time_to_route_coincidence_s) &&
         std::isfinite(risk.predicted_min_route_gap_m) &&
         std::isfinite(risk.predicted_min_route_gap_time_s) &&
         std::isfinite(risk.prediction_horizon_s) && std::isfinite(risk.ttc_s) &&
         std::isfinite(risk.cpa_time_s) && std::isfinite(risk.cpa_distance_m) &&
         std::isfinite(risk.predicted_min_separation_m) &&
         std::isfinite(risk.predicted_min_separation_time_s);
}

std::uint8_t action_constant(const HighwayMergeGapResponseAction action)
{
  switch (action) {
    case HighwayMergeGapResponseAction::kWait:
      return ad_interfaces::msg::HighwayMergeGapResponse::ACTION_WAIT;
    case HighwayMergeGapResponseAction::kHold:
      return ad_interfaces::msg::HighwayMergeGapResponse::ACTION_HOLD;
    case HighwayMergeGapResponseAction::kMergeReady:
    default:
      return ad_interfaces::msg::HighwayMergeGapResponse::ACTION_MERGE_READY;
  }
}

std::uint8_t reason_constant(const HighwayMergeGapResponseReason reason)
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

void copy_uuid(
  const std::array<std::uint8_t, 16U> & source,
  unique_identifier_msgs::msg::UUID & destination)
{
  std::copy(source.begin(), source.end(), destination.uuid.begin());
}

}  // namespace

HighwayMergeGapResponseFrameResult build_highway_merge_gap_response_frame(
  const ad_interfaces::msg::HighwayMergeGapRiskArray & risks,
  const std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const HighwayMergeGapResponseFrameConfig & config)
{
  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(risks.header.stamp);
    if (!(config.maximum_input_age_s > 0.0)) {
      return rejected("response frame configuration is invalid");
    }
    if (risks.header.frame_id != config.expected_frame_id) {
      return rejected("highway merge risk frame_id is unexpected");
    }
    if (risks.merge_zone_id.empty() ||
      risks.merge_zone_id != config.expected_merge_zone_id)
    {
      return rejected("highway merge zone is invalid");
    }
    if (last_risk_stamp_ns.has_value() &&
      *last_risk_stamp_ns - stamp_ns > config.clock_rollback_threshold_ns)
    {
      // Large backward jump: a sim-time reset, not a duplicate. Drop the latch.
      last_risk_stamp_ns.reset();
    }
    if (last_risk_stamp_ns.has_value() && stamp_ns <= *last_risk_stamp_ns) {
      return rejected("highway merge risk array is a duplicate or backward frame");
    }
    if (stamp_ns > now_ns) {
      return rejected("highway merge risk array stamp is in the future");
    }
    if (now_ns - stamp_ns > seconds_to_ns(config.maximum_input_age_s)) {
      return rejected("highway merge risk array is stale");
    }
  } catch (const std::exception & error) {
    return rejected(error.what());
  }

  if (!std::isfinite(risks.merge_zone_entry_route_s_m) ||
    !std::isfinite(risks.merge_reference_route_s_m) ||
    !std::isfinite(risks.ego_route_s_m) ||
    !std::isfinite(risks.ego_longitudinal_speed_mps) ||
    !std::isfinite(risks.ego_route_distance_to_zone_entry_m) ||
    !std::isfinite(risks.ego_route_distance_to_merge_m) ||
    !std::isfinite(risks.ego_merge_time_s) ||
    risks.ego_longitudinal_speed_mps < 0.0F ||
    (risks.ego_merge_timing_valid && risks.ego_merge_time_s < 0.0F))
  {
    return rejected("highway merge ego facts are non-finite or inconsistent");
  }

  const auto actual_relevant = static_cast<std::size_t>(std::count_if(
      risks.objects.begin(), risks.objects.end(),
      [](const auto & risk) {return risk.relevant_to_merge;}));
  if (actual_relevant != risks.relevant_object_count) {
    return rejected("relevant_object_count is inconsistent with objects");
  }

  HighwayMergeGapResponseInput input;
  input.ego_merge_timing_valid = risks.ego_merge_timing_valid;
  input.ego_merge_time_s = risks.ego_merge_time_s;
  input.ego_speed_mps = risks.ego_longitudinal_speed_mps;
  input.ego_route_distance_to_merge_m = risks.ego_route_distance_to_merge_m;
  input.relevant_objects.reserve(actual_relevant);

  for (const auto & risk : risks.objects) {
    if (!object_finite(risk) || risk.classification_probability < 0.0F ||
      risk.classification_probability > 1.0F || risk.existence_probability < 0.0F ||
      risk.existence_probability > 1.0F || risk.prediction_horizon_s < 0.0F ||
      (static_cast<int>(risk.is_ahead_at_merge) + static_cast<int>(risk.is_behind_at_merge) +
      static_cast<int>(risk.is_alongside_at_merge) > 1) ||
      ((risk.is_ahead_at_merge || risk.is_behind_at_merge || risk.is_alongside_at_merge) &&
      !risk.delta_s_at_merge_valid) ||
      (risk.delta_s_at_merge_valid && !risks.ego_merge_timing_valid) ||
      (risk.prediction_covers_merge_time && !risks.ego_merge_timing_valid) ||
      (risk.prediction_covers_merge_time &&
      risk.prediction_horizon_s + 1.0e-3F < risks.ego_merge_time_s) ||
      (risk.longitudinal_gap_closing && !(risk.longitudinal_closing_speed_mps > 0.0F)) ||
      (risk.time_to_route_coincidence_valid && !risk.longitudinal_gap_closing) ||
      (risk.predicted_min_route_gap_valid && risk.predicted_min_route_gap_m < 0.0F))
    {
      return rejected("highway merge object facts are malformed or inconsistent");
    }
    if (!risk.relevant_to_merge) {
      continue;
    }
    HighwayMergeGapResponseObjectInput object;
    std::copy(risk.object_id.uuid.begin(), risk.object_id.uuid.end(), object.object_id.begin());
    object.delta_s_now_m = risk.delta_s_now_m;
    object.object_longitudinal_speed_mps = risk.object_longitudinal_speed_mps;
    object.delta_s_at_merge_valid = risk.delta_s_at_merge_valid;
    object.delta_s_at_merge_m = risk.delta_s_at_merge_m;
    object.is_ahead_at_merge = risk.is_ahead_at_merge;
    object.is_behind_at_merge = risk.is_behind_at_merge;
    object.is_alongside_at_merge = risk.is_alongside_at_merge;
    object.longitudinal_gap_closing = risk.longitudinal_gap_closing;
    object.longitudinal_closing_speed_mps = risk.longitudinal_closing_speed_mps;
    object.time_to_route_coincidence_valid = risk.time_to_route_coincidence_valid;
    object.time_to_route_coincidence_s = risk.time_to_route_coincidence_s;
    object.predicted_min_route_gap_valid = risk.predicted_min_route_gap_valid;
    object.predicted_min_route_gap_m = risk.predicted_min_route_gap_m;
    object.prediction_covers_merge_time = risk.prediction_covers_merge_time;
    if (input.relevant_objects.size() < config.policy.maximum_relevant_objects) {
      input.relevant_objects.push_back(object);
    } else {
      ++input.relevant_objects_omitted;
    }
  }

  // Applicability: the ego is in the merge approach / zone for this traversal.
  // A far ego or an ego past the merge reference station is inactive.
  const bool past_merge =
    risks.ego_route_distance_to_merge_m < -kStationToleranceM;
  const bool far_before =
    risks.ego_route_distance_to_zone_entry_m > config.maximum_approach_distance_m;
  input.applicable = !past_merge && !far_before;

  HighwayMergeGapResponseResult policy;
  try {
    policy = compute_highway_merge_gap_response(input, config.policy);
  } catch (const std::exception & error) {
    return rejected(
      std::string("highway merge response policy rejected facts: ") + error.what());
  }

  HighwayMergeGapResponseFrameResult result;
  result.published = true;
  result.action = policy.action;
  auto & output = result.output;
  output.header = risks.header;
  output.action = action_constant(policy.action);
  output.active = policy.active;
  output.reason = reason_constant(policy.reason);
  output.merge_zone_id = risks.merge_zone_id;
  output.source_object_valid = policy.has_source;
  if (policy.has_source) {
    copy_uuid(policy.source_object_id, output.source_object_id);
  }
  output.relevant_object_count = static_cast<std::uint16_t>(policy.relevant_object_count);
  output.ego_speed_mps = static_cast<float>(policy.ego_speed_mps);
  output.ego_route_distance_to_merge_m =
    static_cast<float>(policy.ego_route_distance_to_merge_m);
  output.ego_merge_timing_valid = policy.ego_merge_timing_valid;
  output.ego_merge_time_s = static_cast<float>(policy.ego_merge_time_s);
  output.available_distance_m = static_cast<float>(policy.available_distance_m);
  output.comfortable_stop_distance_m = static_cast<float>(policy.comfortable_stop_distance_m);
  output.complete_prediction_coverage = policy.complete_prediction_coverage;
  output.front_object_valid = policy.front_object_valid;
  if (policy.front_object_valid) {
    copy_uuid(policy.front_object_id, output.front_object_id);
  }
  output.front_gap_m = static_cast<float>(policy.front_gap_m);
  output.front_time_headway_valid = policy.front_time_headway_valid;
  output.front_time_headway_s = static_cast<float>(policy.front_time_headway_s);
  output.rear_object_valid = policy.rear_object_valid;
  if (policy.rear_object_valid) {
    copy_uuid(policy.rear_object_id, output.rear_object_id);
  }
  output.rear_gap_m = static_cast<float>(policy.rear_gap_m);
  output.rear_time_headway_valid = policy.rear_time_headway_valid;
  output.rear_time_headway_s = static_cast<float>(policy.rear_time_headway_s);
  output.rear_closing_object_valid = policy.rear_closing_object_valid;
  if (policy.rear_closing_object_valid) {
    copy_uuid(policy.rear_closing_object_id, output.rear_closing_object_id);
  }
  output.rear_closing_time_valid = policy.rear_closing_time_valid;
  output.rear_closing_time_s = static_cast<float>(policy.rear_closing_time_s);
  output.predicted_route_clearance_valid = policy.predicted_route_clearance_valid;
  output.minimum_predicted_route_gap_m =
    static_cast<float>(policy.minimum_predicted_route_gap_m);
  return result;
}

HighwayMergeGapResponseNode::HighwayMergeGapResponseNode(const rclcpp::NodeOptions & options)
: Node("ad_highway_merge_gap_response", options)
{
  const auto input_topic = declare_parameter<std::string>(
    "topics.highway_merge_gap_risks", "/ad/planning/highway_merge_gap_risks");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/highway_merge_gap_response");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/highway_merge_gap_response/diagnostics");
  config_.expected_frame_id = declare_parameter<std::string>("expected_frame_id", "map");
  config_.expected_merge_zone_id = declare_parameter<std::string>(
    "merge_zone_id", "kcity_highway_onramp");
  config_.maximum_input_age_s = declare_parameter<double>("maximum_input_age_s", 0.5);
  config_.maximum_approach_distance_m =
    declare_parameter<double>("maximum_approach_distance_m", 400.0);
  config_.policy.minimum_front_time_headway_s =
    declare_parameter<double>("minimum_front_time_headway_s", 1.5);
  config_.policy.minimum_rear_time_headway_s =
    declare_parameter<double>("minimum_rear_time_headway_s", 2.0);
  config_.policy.minimum_rear_closing_time_s =
    declare_parameter<double>("minimum_rear_closing_time_s", 3.0);
  config_.policy.minimum_predicted_route_gap_m =
    declare_parameter<double>("minimum_predicted_route_gap_m", 6.0);
  config_.policy.merge_standoff_m = declare_parameter<double>("merge_standoff_m", 6.0);
  config_.policy.comfortable_deceleration_mps2 =
    declare_parameter<double>("comfortable_deceleration_mps2", 1.8);
  config_.policy.stopped_speed_threshold_mps =
    declare_parameter<double>("stopped_speed_threshold_mps", 0.5);
  config_.policy.speed_epsilon_mps = declare_parameter<double>("speed_epsilon_mps", 0.5);
  config_.policy.maximum_relevant_objects = positive_size(
    declare_parameter<int>("maximum_relevant_objects", 256), "maximum_relevant_objects");
  config_.policy = config_.policy.validated();
  if (!(config_.maximum_approach_distance_m > 0.0) ||
    !std::isfinite(config_.maximum_approach_distance_m))
  {
    throw std::invalid_argument("maximum_approach_distance_m must be finite and positive");
  }
  runtime_summary_interval_frames_ = nonnegative_size(
    declare_parameter<int>("runtime_summary_interval_frames", 200),
    "runtime_summary_interval_frames");
  if (input_topic.empty() || output_topic_.empty() || diagnostics_topic_.empty() ||
    config_.expected_frame_id.empty() || config_.expected_merge_zone_id.empty())
  {
    throw std::invalid_argument("highway merge response topics/context must be nonempty");
  }

  publisher_ = create_publisher<ad_interfaces::msg::HighwayMergeGapResponse>(
    output_topic_, rclcpp::QoS(1).reliable().durability_volatile());
  diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
  subscription_ = create_subscription<ad_interfaces::msg::HighwayMergeGapRiskArray>(
    input_topic, rclcpp::QoS(1).reliable().durability_volatile(),
    [this](ad_interfaces::msg::HighwayMergeGapRiskArray::ConstSharedPtr message) {
      on_risks(message);
    });
  RCLCPP_INFO(
    get_logger(), "ad_highway_merge_gap_response: %s -> %s",
    input_topic.c_str(), output_topic_.c_str());
}

void HighwayMergeGapResponseNode::on_risks(
  const ad_interfaces::msg::HighwayMergeGapRiskArray::ConstSharedPtr message)
{
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  ++received_;
  const auto result = build_highway_merge_gap_response_frame(
    *message, now().nanoseconds(), last_risk_stamp_ns_, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  if (result.published) {
    publisher_->publish(result.output);
    try {
      last_risk_stamp_ns_ = stamp_to_ns(message->header.stamp);
    } catch (const std::exception &) {
      last_risk_stamp_ns_.reset();
    }
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Highway merge response frame rejected: %s", result.reject_reason.c_str());
  }
  publish_diagnostics(message->header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void HighwayMergeGapResponseNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const HighwayMergeGapResponseFrameResult & result,
  const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "highway_merge_gap_response";
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
    key_value("relevant_object_count", std::to_string(result.output.relevant_object_count)));
  status.values.push_back(
    key_value(
      "complete_prediction_coverage",
      result.output.complete_prediction_coverage ? "true" : "false"));
  status.values.push_back(
    key_value("front_gap_m", std::to_string(result.output.front_gap_m)));
  status.values.push_back(
    key_value("rear_gap_m", std::to_string(result.output.rear_gap_m)));
  status.values.push_back(
    key_value("rear_closing_time_s", std::to_string(result.output.rear_closing_time_s)));
  status.values.push_back(
    key_value(
      "minimum_predicted_route_gap_m",
      std::to_string(result.output.minimum_predicted_route_gap_m)));
  status.values.push_back(
    key_value("available_distance_m", std::to_string(result.output.available_distance_m)));
  status.values.push_back(
    key_value(
      "comfortable_stop_distance_m",
      std::to_string(result.output.comfortable_stop_distance_m)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void HighwayMergeGapResponseNode::record_runtime(
  const HighwayMergeGapResponseFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++published_;
    latency_ms_.push_back(latency_ms);
    if (!result.output.active) {
      ++inactive_;
    } else if (result.action == HighwayMergeGapResponseAction::kMergeReady) {
      ++merge_ready_;
    } else if (result.action == HighwayMergeGapResponseAction::kWait) {
      ++wait_;
    } else {
      ++hold_;
    }
    if (result.output.reason < reasons_.size()) {++reasons_[result.output.reason];}
  } else {
    ++rejected_;
  }
  if (runtime_summary_interval_frames_ > 0U &&
    received_ % runtime_summary_interval_frames_ == 0U)
  {
    const double max_latency = latency_ms_.empty() ? 0.0 :
      *std::max_element(latency_ms_.begin(), latency_ms_.end());
    RCLCPP_INFO(
      get_logger(),
      "HIGHWAY_MERGE_GAP_RESPONSE_RUNTIME_SUMMARY received=%zu published=%zu "
      "merge_ready=%zu wait=%zu hold=%zu inactive=%zu rejected=%zu "
      "reasons=%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu "
      "latency_ms_median=%.6f latency_ms_p95=%.6f latency_ms_max=%.6f",
      received_, published_, merge_ready_, wait_, hold_, inactive_, rejected_,
      reasons_[0], reasons_[1], reasons_[2], reasons_[3], reasons_[4], reasons_[5],
      reasons_[6], reasons_[7], reasons_[8], percentile(latency_ms_, 0.5),
      percentile(latency_ms_, 0.95), max_latency);
  }
}

}  // namespace ad_planner
