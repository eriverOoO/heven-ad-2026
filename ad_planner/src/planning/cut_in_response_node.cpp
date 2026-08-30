#include "cut_in_response_node.hpp"

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

std::int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    throw std::invalid_argument("stamp is malformed");
  }
  const std::int64_t value =
    static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
    static_cast<std::int64_t>(stamp.nanosec);
  if (value <= 0) {
    throw std::invalid_argument("stamp must be strictly positive");
  }
  return value;
}

std::int64_t seconds_to_ns(const double seconds)
{
  const long double value = static_cast<long double>(seconds) *
    static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(value) || value < 0.0L ||
    value > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument("cut-in response duration is not representable");
  }
  return static_cast<std::int64_t>(value);
}

diagnostic_msgs::msg::KeyValue key_value(
  const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = value;
  return result;
}

bool finite_fact(const bool valid, const double value)
{
  return std::isfinite(value) && (!valid || value >= 0.0);
}

CutInResponseFrameResult rejected(std::string reason)
{
  CutInResponseFrameResult result;
  result.reason = std::move(reason);
  return result;
}

std::uint8_t action_constant(const CutInResponseAction action)
{
  switch (action) {
    case CutInResponseAction::kSlowdown:
      return ad_interfaces::msg::CutInResponse::ACTION_SLOWDOWN;
    case CutInResponseAction::kHold:
      return ad_interfaces::msg::CutInResponse::ACTION_HOLD;
    case CutInResponseAction::kNone:
    default:
      return ad_interfaces::msg::CutInResponse::ACTION_NONE;
  }
}

std::uint8_t reason_constant(const CutInResponseReason reason)
{
  switch (reason) {
    case CutInResponseReason::kApproachingEntry:
      return ad_interfaces::msg::CutInResponse::REASON_APPROACHING_ENTRY;
    case CutInResponseReason::kCollisionConflict:
      return ad_interfaces::msg::CutInResponse::REASON_COLLISION_CONFLICT;
    case CutInResponseReason::kSmallPredictedClearance:
      return ad_interfaces::msg::CutInResponse::REASON_SMALL_PREDICTED_CLEARANCE;
    case CutInResponseReason::kNone:
    default:
      return ad_interfaces::msg::CutInResponse::REASON_NONE;
  }
}

}  // namespace

CutInResponseFrameResult build_cut_in_response_frame(
  const ad_interfaces::msg::CutInRiskArray & risks,
  const std::optional<EgoSpeedSample> & ego,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_risk_stamp_ns,
  const CutInResponseFrameConfig & config)
{
  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(risks.header.stamp);
  } catch (const std::exception & error) {
    return rejected(error.what());
  }
  if (risks.header.frame_id.empty()) {
    return rejected("cut-in risk array has an empty frame_id");
  }
  if (last_risk_stamp_ns.has_value() && stamp_ns <= *last_risk_stamp_ns) {
    return rejected("cut-in risk array is a duplicate or backward frame");
  }
  try {
    if (stamp_ns > now_ns + seconds_to_ns(config.maximum_future_skew_s)) {
      return rejected("cut-in risk array stamp is in the future");
    }
    if (now_ns - stamp_ns > seconds_to_ns(config.maximum_input_age_s)) {
      return rejected("cut-in risk array is stale");
    }
    if (!ego.has_value()) {
      return rejected("odometry is unavailable");
    }
    if (std::llabs(stamp_ns - ego->stamp_ns) >
      seconds_to_ns(config.maximum_odometry_skew_s))
    {
      return rejected("odometry is stale relative to cut-in risk array");
    }
  } catch (const std::exception & error) {
    return rejected(error.what());
  }

  CutInResponseFrameResult result;
  std::vector<CutInResponseObjectInput> candidates;
  candidates.reserve(risks.risks.size());
  for (const auto & risk : risks.risks) {
    if (!risk.cut_in_candidate) {
      continue;
    }
    ++result.candidates_in;
    if (!std::isfinite(risk.route_s_rel_m) ||
      !std::isfinite(risk.lateral_velocity_toward_corridor_mps) ||
      !finite_fact(risk.predicted_entry_valid, risk.predicted_entry_time_s) ||
      !std::isfinite(risk.predicted_entry_route_s_rel_m) ||
      !finite_fact(risk.ttc_valid, risk.ttc_s) ||
      !finite_fact(risk.cpa_valid, risk.cpa_time_s) ||
      !finite_fact(risk.cpa_valid, risk.cpa_distance_m) ||
      !finite_fact(risk.predicted_min_separation_valid, risk.predicted_min_separation_m) ||
      !finite_fact(risk.predicted_min_separation_valid, risk.predicted_min_separation_time_s))
    {
      ++result.rejected_malformed;
      continue;
    }
    CutInResponseObjectInput object;
    std::copy(
      risk.object_id.uuid.begin(), risk.object_id.uuid.end(), object.object_id.begin());
    object.side = risk.side;
    object.route_s_rel_m = risk.route_s_rel_m;
    object.lateral_velocity_toward_corridor_mps = risk.lateral_velocity_toward_corridor_mps;
    object.predicted_entry_valid = risk.predicted_entry_valid;
    object.predicted_entry_time_s = risk.predicted_entry_time_s;
    object.predicted_entry_route_s_rel_m = risk.predicted_entry_route_s_rel_m;
    object.ttc_valid = risk.ttc_valid;
    object.ttc_s = risk.ttc_s;
    object.cpa_valid = risk.cpa_valid;
    object.cpa_time_s = risk.cpa_time_s;
    object.cpa_distance_m = risk.cpa_distance_m;
    object.predicted_min_separation_valid = risk.predicted_min_separation_valid;
    object.predicted_min_separation_m = risk.predicted_min_separation_m;
    object.predicted_min_separation_time_s = risk.predicted_min_separation_time_s;
    candidates.push_back(object);
  }

  CutInResponseResult response;
  try {
    response = compute_cut_in_response(ego->speed_mps, candidates, config.policy);
  } catch (const std::exception & error) {
    return rejected(std::string("cut-in response policy is invalid: ") + error.what());
  }

  result.published = true;
  result.action = response.action;
  result.rejected_over_budget = response.rejected_over_budget;

  ad_interfaces::msg::CutInResponse & output = result.output;
  output.header = risks.header;
  output.action = action_constant(response.action);
  output.reason = reason_constant(response.reason);
  output.active = response.active;
  output.candidate_count = static_cast<std::uint16_t>(
    std::min<std::size_t>(response.candidate_count, std::numeric_limits<std::uint16_t>::max()));
  output.requested_max_speed_valid = response.requested_max_speed_valid;
  output.requested_max_speed_mps = static_cast<float>(
    response.requested_max_speed_valid ? response.requested_max_speed_mps : 0.0);
  output.ego_speed_mps = static_cast<float>(response.ego_speed_mps);
  if (response.has_source) {
    const auto & source = response.source;
    std::copy(
      source.input.object_id.begin(), source.input.object_id.end(),
      output.source_object_id.uuid.begin());
    output.source_side = source.input.side;
    output.required_deceleration_mps2 =
      static_cast<float>(source.required_deceleration_mps2);
    output.route_s_rel_m = static_cast<float>(source.input.route_s_rel_m);
    output.predicted_entry_valid = source.input.predicted_entry_valid;
    output.predicted_entry_time_s = static_cast<float>(
      source.input.predicted_entry_valid ? source.input.predicted_entry_time_s : 0.0);
    output.predicted_entry_route_s_rel_m = static_cast<float>(
      source.input.predicted_entry_valid ? source.input.predicted_entry_route_s_rel_m : 0.0);
    output.lateral_velocity_toward_corridor_mps =
      static_cast<float>(source.input.lateral_velocity_toward_corridor_mps);
    output.ttc_valid = source.input.ttc_valid;
    output.ttc_s = static_cast<float>(source.input.ttc_valid ? source.input.ttc_s : 0.0);
    output.cpa_valid = source.input.cpa_valid;
    output.cpa_time_s = static_cast<float>(
      source.input.cpa_valid ? source.input.cpa_time_s : 0.0);
    output.cpa_distance_m = static_cast<float>(
      source.input.cpa_valid ? source.input.cpa_distance_m : 0.0);
    output.predicted_min_separation_valid = source.input.predicted_min_separation_valid;
    output.predicted_min_separation_m = static_cast<float>(
      source.input.predicted_min_separation_valid ?
      source.input.predicted_min_separation_m : 0.0);
  }
  return result;
}

CutInResponseNode::CutInResponseNode(const rclcpp::NodeOptions & options)
: Node("ad_cut_in_response", options)
{
  const std::string risk_topic = declare_parameter<std::string>(
    "topics.cut_in_risks", "/ad/planning/cut_in_risks");
  const std::string odometry_topic = declare_parameter<std::string>(
    "topics.odometry", "/ad/localization/odometry");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/cut_in_response");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/cut_in_response/diagnostics");

  config_.maximum_input_age_s = declare_parameter<double>("maximum_input_age_s", 0.5);
  config_.maximum_odometry_skew_s =
    declare_parameter<double>("maximum_odometry_skew_s", 0.5);
  config_.maximum_future_skew_s = declare_parameter<double>("maximum_future_skew_s", 0.10);
  config_.policy.longitudinal_standoff_m =
    declare_parameter<double>("longitudinal_standoff_m", 6.0);
  config_.policy.comfortable_deceleration_mps2 =
    declare_parameter<double>("comfortable_deceleration_mps2", 1.8);
  config_.policy.maximum_deceleration_mps2 =
    declare_parameter<double>("maximum_deceleration_mps2", 3.0);
  config_.policy.minimum_response_speed_mps =
    declare_parameter<double>("minimum_response_speed_mps", 1.0);
  config_.policy.maximum_risks = positive_size(
    declare_parameter<int>("maximum_risks", 256), "maximum_risks");
  config_.policy = config_.policy.validated();
  runtime_summary_interval_frames_ = nonnegative_size(
    declare_parameter<int>("runtime_summary_interval_frames", 200),
    "runtime_summary_interval_frames");

  if (!(config_.maximum_input_age_s > 0.0) ||
    !(config_.maximum_odometry_skew_s > 0.0) ||
    !std::isfinite(config_.maximum_future_skew_s) ||
    config_.maximum_future_skew_s < 0.0 ||
    risk_topic.empty() || odometry_topic.empty() ||
    output_topic_.empty() || diagnostics_topic_.empty())
  {
    throw std::invalid_argument("cut-in response node parameters are invalid");
  }

  publisher_ = create_publisher<ad_interfaces::msg::CutInResponse>(
    output_topic_, rclcpp::QoS(1).reliable().durability_volatile());
  diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(10).reliable(),
    [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
      on_odometry(message);
    });
  risk_subscription_ = create_subscription<ad_interfaces::msg::CutInRiskArray>(
    risk_topic, rclcpp::QoS(1).reliable().durability_volatile(),
    [this](ad_interfaces::msg::CutInRiskArray::ConstSharedPtr message) {
      on_risks(message);
    });
  RCLCPP_INFO(
    get_logger(), "ad_cut_in_response: %s + %s -> %s",
    risk_topic.c_str(), odometry_topic.c_str(), output_topic_.c_str());
}

void CutInResponseNode::on_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr message)
{
  try {
    static_cast<void>(stamp_to_ns(message->header.stamp));
    if (message->header.frame_id.empty() ||
      !std::isfinite(message->twist.twist.linear.x))
    {
      throw std::invalid_argument("odometry metadata or speed is invalid");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odometry_ = *message;
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Ignoring odometry: %s", error.what());
  }
}

void CutInResponseNode::on_risks(
  const ad_interfaces::msg::CutInRiskArray::ConstSharedPtr message)
{
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  ++risk_messages_received_;

  std::optional<EgoSpeedSample> ego;
  if (latest_odometry_.has_value()) {
    try {
      ego = EgoSpeedSample{
        latest_odometry_->twist.twist.linear.x,
        stamp_to_ns(latest_odometry_->header.stamp)};
    } catch (const std::exception &) {
      ego.reset();
    }
  }

  std::int64_t stamp_ns = 0;
  bool stamp_ok = true;
  try {
    stamp_ns = stamp_to_ns(message->header.stamp);
  } catch (const std::exception &) {
    stamp_ok = false;
  }
  std::optional<std::int64_t> effective_last = last_risk_stamp_ns_;
  if (stamp_ok && last_risk_stamp_ns_.has_value() &&
    stamp_ns < *last_risk_stamp_ns_ &&
    *last_risk_stamp_ns_ - stamp_ns > config_.clock_rollback_threshold_ns)
  {
    last_risk_stamp_ns_.reset();
    effective_last.reset();
  }

  const CutInResponseFrameResult result = build_cut_in_response_frame(
    *message, ego, now().nanoseconds(), effective_last, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();

  if (result.published) {
    // The policy is stateless: exactly one response per accepted frame, never
    // latched. A consumer applies its own freshness timeout and treats a
    // missing response as "no cut-in longitudinal constraint".
    publisher_->publish(result.output);
    last_risk_stamp_ns_ = stamp_ns;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Cut-in response frame rejected: %s",
      result.reason.c_str());
  }
  publish_diagnostics(message->header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void CutInResponseNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const CutInResponseFrameResult & result,
  const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "cut_in_response";
  status.hardware_id = output_topic_;
  status.level = result.published ? diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.published ? "ok" : result.reason;
  status.values.push_back(key_value("published", result.published ? "true" : "false"));
  status.values.push_back(key_value("reject_reason", result.reason));
  status.values.push_back(
    key_value("action", std::to_string(static_cast<int>(result.output.action))));
  status.values.push_back(
    key_value("reason", std::to_string(static_cast<int>(result.output.reason))));
  status.values.push_back(
    key_value("active", result.output.active ? "true" : "false"));
  status.values.push_back(
    key_value("candidate_count", std::to_string(result.output.candidate_count)));
  status.values.push_back(
    key_value("requested_max_speed_mps",
    std::to_string(result.output.requested_max_speed_mps)));
  status.values.push_back(
    key_value("required_deceleration_mps2",
    std::to_string(result.output.required_deceleration_mps2)));
  status.values.push_back(
    key_value("ego_speed_mps", std::to_string(result.output.ego_speed_mps)));
  status.values.push_back(
    key_value("rejected_malformed", std::to_string(result.rejected_malformed)));
  status.values.push_back(
    key_value("rejected_over_budget", std::to_string(result.rejected_over_budget)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void CutInResponseNode::record_runtime(
  const CutInResponseFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++response_messages_published_;
    if (result.action == CutInResponseAction::kSlowdown) {
      ++slowdown_frames_;
    } else if (result.action == CutInResponseAction::kHold) {
      ++hold_frames_;
    }
  } else {
    ++rejected_frames_;
  }
  latency_ms_.push_back(latency_ms);
  if (runtime_summary_interval_frames_ == 0U ||
    latency_ms_.size() % runtime_summary_interval_frames_ != 0U)
  {
    return;
  }
  auto ordered = latency_ms_;
  std::sort(ordered.begin(), ordered.end());
  const auto quantile = [&ordered](const double q) {
      const auto index = static_cast<std::size_t>(
        q * static_cast<double>(ordered.size() - 1U));
      return ordered[index];
    };
  RCLCPP_INFO(
    get_logger(),
    "CUT_IN_RESPONSE_RUNTIME_SUMMARY risk_messages=%zu published=%zu rejected=%zu "
    "slowdown_frames=%zu hold_frames=%zu latency_ms_median=%.4f latency_ms_p95=%.4f "
    "latency_ms_max=%.4f",
    risk_messages_received_, response_messages_published_, rejected_frames_,
    slowdown_frames_, hold_frames_, quantile(0.5), quantile(0.95), ordered.back());
}

}  // namespace ad_planner
