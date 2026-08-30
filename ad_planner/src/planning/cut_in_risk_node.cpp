#include "cut_in_risk_node.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <tf2/exceptions.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ad_planner/io/route_corridor_loader.hpp"
#include "ad_planner/local_planning/common/local_motion_frame.hpp"

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
    throw std::invalid_argument("cut-in duration is not representable");
  }
  return static_cast<std::int64_t>(value);
}

double duration_seconds(const builtin_interfaces::msg::Duration & duration)
{
  if (duration.sec < 0 || duration.nanosec >= kNanosecondsPerSecond) {
    throw std::invalid_argument("predicted-state duration is malformed");
  }
  return static_cast<double>(duration.sec) +
         static_cast<double>(duration.nanosec) * 1.0e-9;
}

std::string resolve_data_dir(const std::string & parameter)
{
  if (!parameter.empty()) {
    return parameter;
  }
  const char * const environment = std::getenv("AD_DATA_DIR");
  if (environment && *environment != '\0') {
    return environment;
  }
  throw std::invalid_argument("set data_dir or AD_DATA_DIR");
}

CutInFrameResult rejected(std::string reason)
{
  CutInFrameResult result;
  result.reason = std::move(reason);
  return result;
}

bool valid_route_mask(const nav_msgs::msg::OccupancyGrid & mask)
{
  const auto & origin = mask.info.origin;
  if (mask.info.width != 1040U || mask.info.height != 200U ||
    mask.info.resolution != 0.1F ||
    origin.position.x != -4.0 || origin.position.y != -10.0 ||
    origin.position.z != 0.0 || origin.orientation.x != 0.0 ||
    origin.orientation.y != 0.0 || origin.orientation.z != 0.0 ||
    origin.orientation.w != 1.0 ||
    mask.data.size() != static_cast<std::size_t>(mask.info.width) * mask.info.height)
  {
    return false;
  }
  return std::all_of(
    mask.data.begin(), mask.data.end(),
    [](const std::int8_t value) {return value == 0 || value == 100;});
}

CutInObjectInput unpack(const ad_interfaces::msg::DynamicObjectRisk & source)
{
  CutInObjectInput object;
  std::copy(
    source.object_id.uuid.begin(), source.object_id.uuid.end(),
    object.object_id.begin());
  object.classification = source.classification;
  object.classification_probability = source.classification_probability;
  object.existence_probability = source.existence_probability;
  object.x_rel_m = source.x_rel_m;
  object.y_rel_m = source.y_rel_m;
  object.vx_rel_mps = source.vx_rel_mps;
  object.vy_rel_mps = source.vy_rel_mps;
  object.predicted_states.reserve(source.predicted_states.size());
  for (const auto & source_state : source.predicted_states) {
    object.predicted_states.push_back(CutInPredictedStateInput{
      duration_seconds(source_state.time_from_start),
      source_state.x_rel_m, source_state.y_rel_m});
  }
  object.ttc_valid = source.ttc_valid;
  object.ttc_s = source.ttc_s;
  object.cpa_valid = source.cpa_valid;
  object.cpa_time_s = source.cpa_time_s;
  object.cpa_distance_m = source.cpa_distance_m;
  object.predicted_min_separation_valid =
    source.predicted_min_separation_valid;
  object.predicted_min_separation_m = source.predicted_min_separation_m;
  object.predicted_min_separation_time_s =
    source.predicted_min_separation_time_s;
  return object;
}

ad_interfaces::msg::CutInRisk serialize(const CutInRiskResult & source)
{
  ad_interfaces::msg::CutInRisk output;
  std::copy(
    source.object_id.begin(), source.object_id.end(),
    output.object_id.uuid.begin());
  output.classification = source.classification;
  output.classification_probability = source.classification_probability;
  output.existence_probability = source.existence_probability;
  output.route_s_rel_m = static_cast<float>(source.route_s_rel_m);
  output.lateral_offset_m = static_cast<float>(source.lateral_offset_m);
  output.corridor_left_width_m = static_cast<float>(source.corridor_left_width_m);
  output.corridor_right_width_m = static_cast<float>(source.corridor_right_width_m);
  output.inside_corridor = source.inside_corridor;
  output.near_boundary = source.near_boundary;
  output.adjacent_region = source.adjacent_region;
  output.side = static_cast<std::uint8_t>(source.side);
  output.lateral_velocity_mps = static_cast<float>(source.lateral_velocity_mps);
  output.lateral_velocity_toward_corridor_mps = static_cast<float>(
    source.lateral_velocity_toward_corridor_mps);
  output.approaching_corridor = source.approaching_corridor;
  output.longitudinally_relevant = source.longitudinally_relevant;
  output.predicted_entry_valid = source.predicted_entry_valid;
  output.predicted_entry_time_s = static_cast<float>(
    source.predicted_entry_valid ? source.predicted_entry_time_s : 0.0);
  output.predicted_entry_route_s_rel_m = static_cast<float>(
    source.predicted_entry_valid ? source.predicted_entry_route_s_rel_m : 0.0);
  output.predicted_entry_lateral_offset_m = static_cast<float>(
    source.predicted_entry_valid ? source.predicted_entry_lateral_offset_m : 0.0);
  output.predicted_entry_sustained = source.predicted_entry_sustained;
  output.ttc_valid = source.ttc_valid;
  output.ttc_s = static_cast<float>(source.ttc_valid ? source.ttc_s : 0.0);
  output.cpa_valid = source.cpa_valid;
  output.cpa_time_s = static_cast<float>(source.cpa_valid ? source.cpa_time_s : 0.0);
  output.cpa_distance_m = static_cast<float>(
    source.cpa_valid ? source.cpa_distance_m : 0.0);
  output.predicted_min_separation_valid =
    source.predicted_min_separation_valid;
  output.predicted_min_separation_m = static_cast<float>(
    source.predicted_min_separation_valid ?
    source.predicted_min_separation_m : 0.0);
  output.predicted_min_separation_time_s = static_cast<float>(
    source.predicted_min_separation_valid ?
    source.predicted_min_separation_time_s : 0.0);
  output.cut_in_candidate = source.cut_in_candidate;
  return output;
}

diagnostic_msgs::msg::KeyValue key_value(
  const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = key;
  result.value = value;
  return result;
}

std::string uuid_text(const std::array<std::uint8_t, 16U> & uuid)
{
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(32U);
  for (const auto byte : uuid) {
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0fU]);
  }
  return result;
}

}  // namespace

CutInFrameResult build_cut_in_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const nav_msgs::msg::OccupancyGrid & route_mask,
  const std::optional<RouteEgoSample> & ego,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_risk_stamp_ns,
  const ReferenceCorridor & corridor,
  const CutInFrameConfig & config)
{
  std::int64_t stamp_ns = 0;
  std::int64_t mask_stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(risks.header.stamp);
    mask_stamp_ns = stamp_to_ns(route_mask.header.stamp);
  } catch (const std::exception & error) {
    return rejected(error.what());
  }
  if (risks.header.frame_id != config.risk_frame_id) {
    return rejected("risk array frame is not '" + config.risk_frame_id + "'");
  }
  if (route_mask.header.frame_id != config.route_mask_frame_id) {
    return rejected("route mask frame is not '" + config.route_mask_frame_id + "'");
  }
  if (stamp_ns != mask_stamp_ns) {
    return rejected("risk array and route mask stamps differ");
  }
  if (last_risk_stamp_ns.has_value() && stamp_ns <= *last_risk_stamp_ns) {
    return rejected("risk array is a duplicate or backward frame");
  }
  try {
    if (stamp_ns > now_ns + seconds_to_ns(config.maximum_future_skew_s)) {
      return rejected("risk array stamp is in the future");
    }
    if (now_ns - stamp_ns > seconds_to_ns(config.maximum_input_age_s)) {
      return rejected("risk array and route context are stale");
    }
    if (!ego.has_value()) {
      return rejected("odometry is unavailable");
    }
    if (std::llabs(stamp_ns - ego->stamp_ns) >
      seconds_to_ns(config.maximum_odometry_skew_s))
    {
      return rejected("odometry is stale relative to risk array");
    }
  } catch (const std::exception & error) {
    return rejected(error.what());
  }
  if (!valid_route_mask(route_mask)) {
    return rejected("active route mask geometry or cells are invalid");
  }
  if (corridor.frame_id.empty() || corridor.lanes.empty() ||
    corridor.primary_lane_index >= corridor.lanes.size())
  {
    return rejected("active reference corridor is invalid");
  }

  std::vector<CutInObjectInput> objects;
  objects.reserve(risks.objects.size());
  std::size_t rejected_during_unpack = 0U;
  for (const auto & risk : risks.objects) {
    try {
      objects.push_back(unpack(risk));
    } catch (const std::exception &) {
      ++rejected_during_unpack;
    }
  }

  CutInComputation computation;
  try {
    computation = compute_cut_in_risks(
      corridor.lanes[corridor.primary_lane_index], ego->state,
      objects, config.cut_in);
  } catch (const std::exception & error) {
    return rejected(std::string("cut-in context is invalid: ") + error.what());
  }

  CutInFrameResult result;
  result.published = true;
  result.objects_in = risks.objects.size();
  result.rejected_malformed_objects =
    rejected_during_unpack + computation.rejected_malformed_objects;
  result.rejected_over_budget = computation.rejected_over_budget;
  result.output.header = risks.header;
  result.output.header.frame_id = corridor.frame_id;
  result.output.risks.reserve(computation.risks.size());
  for (const auto & risk : computation.risks) {
    result.candidate_count += risk.cut_in_candidate ? 1U : 0U;
    result.predicted_entry_valid_count += risk.predicted_entry_valid ? 1U : 0U;
    result.output.risks.push_back(serialize(risk));
  }
  result.objects_out = result.output.risks.size();
  return result;
}

CutInRiskNode::CutInRiskNode(const rclcpp::NodeOptions & options)
: Node("ad_cut_in_risk", options), pairer_(positive_size(
    declare_parameter<int>("maximum_pending_frames", 16),
    "maximum_pending_frames")),
  transform_timeout_(0, 0)
{
  const std::string data_dir = resolve_data_dir(
    declare_parameter<std::string>("data_dir", ""));
  std::filesystem::path corridor_path = declare_parameter<std::string>(
    "route_corridor_file", "map/route_corridor.json");
  if (corridor_path.is_relative()) {
    corridor_path = std::filesystem::path(data_dir) / corridor_path;
  }
  const auto loaded = load_route_corridor(
    corridor_path,
    {{"global_path", declare_parameter<std::string>(
      "route_corridor.expected_global_path_sha256", "")}});
  corridor_ = loaded.corridor;

  const std::string risk_topic = declare_parameter<std::string>(
    "topics.dynamic_object_risks", "/ad/planning/dynamic_object_risks");
  const std::string route_mask_topic = declare_parameter<std::string>(
    "topics.route_mask", "/ad/planning/drivable_mask");
  const std::string odometry_topic = declare_parameter<std::string>(
    "topics.odometry", "/ad/localization/odometry");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/cut_in_risks");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/cut_in_risks/diagnostics");
  config_.risk_frame_id = declare_parameter<std::string>(
    "risk_frame_id", "base_link");
  config_.route_mask_frame_id = declare_parameter<std::string>(
    "route_mask_frame_id", "base_link");
  config_.maximum_input_age_s = declare_parameter<double>(
    "maximum_input_age_s", 0.5);
  config_.maximum_odometry_skew_s = declare_parameter<double>(
    "maximum_odometry_skew_s", 0.5);
  config_.maximum_future_skew_s = declare_parameter<double>(
    "maximum_future_skew_s", 0.10);
  config_.cut_in.forward_analysis_distance_m = declare_parameter<double>(
    "forward_analysis_distance_m", 80.0);
  config_.cut_in.rear_analysis_distance_m = declare_parameter<double>(
    "rear_analysis_distance_m", 5.0);
  config_.cut_in.minimum_lateral_approach_speed_mps = declare_parameter<double>(
    "minimum_lateral_approach_speed_mps", 0.20);
  config_.cut_in.boundary_margin_m = declare_parameter<double>(
    "boundary_margin_m", 0.25);
  config_.cut_in.maximum_lateral_gap_m = declare_parameter<double>(
    "maximum_lateral_gap_m", 5.0);
  config_.cut_in.minimum_sustained_entry_time_s = declare_parameter<double>(
    "minimum_sustained_entry_time_s", 0.50);
  config_.cut_in.maximum_objects = positive_size(
    declare_parameter<int>("maximum_objects", 256), "maximum_objects");
  config_.cut_in = config_.cut_in.validated();
  runtime_summary_interval_frames_ = nonnegative_size(
    declare_parameter<int>("runtime_summary_interval_frames", 200),
    "runtime_summary_interval_frames");
  const double transform_timeout_s = declare_parameter<double>(
    "transform_timeout_s", 0.05);
  transform_timeout_ = rclcpp::Duration::from_seconds(transform_timeout_s);
  if (!(config_.maximum_input_age_s > 0.0) ||
    !(config_.maximum_odometry_skew_s > 0.0) ||
    !std::isfinite(config_.maximum_future_skew_s) ||
    config_.maximum_future_skew_s < 0.0 || !(transform_timeout_s > 0.0) ||
    risk_topic.empty() || route_mask_topic.empty() || odometry_topic.empty() ||
    output_topic_.empty() || diagnostics_topic_.empty())
  {
    throw std::invalid_argument("cut-in node parameters are invalid");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  publisher_ = create_publisher<ad_interfaces::msg::CutInRiskArray>(
    output_topic_, rclcpp::QoS(1).reliable().durability_volatile());
  diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
  odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(10).reliable(),
    [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
      on_odometry(message);
    });
  risk_subscription_ =
    create_subscription<ad_interfaces::msg::DynamicObjectRiskArray>(
    risk_topic, rclcpp::QoS(1).reliable().durability_volatile(),
    [this](ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message) {
      on_risks(message);
    });
  mask_subscription_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
    route_mask_topic, rclcpp::QoS(8).reliable().durability_volatile(),
    [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
      on_route_mask(message);
    });
  RCLCPP_INFO(
    get_logger(), "ad_cut_in_risk: %s + %s -> %s using %s",
    risk_topic.c_str(), route_mask_topic.c_str(), output_topic_.c_str(),
    corridor_path.c_str());
}

void CutInRiskNode::on_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr message)
{
  try {
    static_cast<void>(stamp_to_ns(message->header.stamp));
    const double yaw = tf2::getYaw(message->pose.pose.orientation);
    if (message->header.frame_id.empty() ||
      !std::isfinite(message->pose.pose.position.x) ||
      !std::isfinite(message->pose.pose.position.y) ||
      !std::isfinite(yaw) || !std::isfinite(message->twist.twist.linear.x))
    {
      throw std::invalid_argument("odometry metadata or state is invalid");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odometry_ = *message;
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Ignoring odometry: %s", error.what());
  }
}

void CutInRiskNode::on_risks(
  const ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  ++risk_messages_received_;
  try {
    const auto pair = pairer_.add_left(stamp_to_ns(message->header.stamp), *message);
    if (pair.has_value()) {
      process_pair(std::move(*pair));
    }
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Rejected cut-in risk input: %s", error.what());
  }
}

void CutInRiskNode::on_route_mask(
  const nav_msgs::msg::OccupancyGrid::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    const auto pair = pairer_.add_right(stamp_to_ns(message->header.stamp), *message);
    if (pair.has_value()) {
      process_pair(std::move(*pair));
    }
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Rejected cut-in route input: %s", error.what());
  }
}

void CutInRiskNode::process_pair(Pairer::Pair pair)
{
  const auto started = std::chrono::steady_clock::now();
  std::optional<RouteEgoSample> route_ego;
  if (latest_odometry_.has_value()) {
    try {
      const auto & odometry = *latest_odometry_;
      const rclcpp::Time stamp(pair.stamp_ns, RCL_ROS_TIME);
      FrameTransform2 route_from_odometry;
      if (corridor_.frame_id != odometry.header.frame_id) {
        const auto transform = tf_buffer_->lookupTransform(
          corridor_.frame_id, odometry.header.frame_id, stamp,
          transform_timeout_);
        route_from_odometry = FrameTransform2{
          transform.transform.translation.x,
          transform.transform.translation.y,
          tf2::getYaw(transform.transform.rotation)};
      }
      const Pose2 odometry_pose{
        odometry.pose.pose.position.x, odometry.pose.pose.position.y,
        tf2::getYaw(odometry.pose.pose.orientation)};
      route_ego = RouteEgoSample{
        CutInEgoState{
          transform_pose(route_from_odometry, odometry_pose),
          odometry.twist.twist.linear.x},
        stamp_to_ns(odometry.header.stamp)};
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cut-in route transform unavailable: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Cut-in odometry transform rejected: %s", error.what());
    }
  }

  std::optional<std::int64_t> effective_last = last_risk_stamp_ns_;
  if (last_risk_stamp_ns_.has_value() &&
    pair.stamp_ns < *last_risk_stamp_ns_ &&
    *last_risk_stamp_ns_ - pair.stamp_ns > config_.clock_rollback_threshold_ns)
  {
    last_risk_stamp_ns_.reset();
    effective_last.reset();
  }
  const CutInFrameResult result = build_cut_in_frame(
    pair.left, pair.right, route_ego, now().nanoseconds(), effective_last,
    corridor_, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  if (result.published) {
    publisher_->publish(result.output);
    last_risk_stamp_ns_ = pair.stamp_ns;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Cut-in frame rejected: %s",
      result.reason.c_str());
  }
  publish_diagnostics(pair.left.header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void CutInRiskNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const CutInFrameResult & result,
  const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "cut_in_risk";
  status.hardware_id = output_topic_;
  status.level = result.published ? diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.published ? "ok" : result.reason;
  status.values.push_back(key_value("published", result.published ? "true" : "false"));
  status.values.push_back(key_value("reject_reason", result.reason));
  status.values.push_back(key_value("objects_in", std::to_string(result.objects_in)));
  status.values.push_back(key_value("objects_out", std::to_string(result.objects_out)));
  status.values.push_back(key_value("candidates", std::to_string(result.candidate_count)));
  status.values.push_back(key_value(
    "predicted_entry_valid", std::to_string(result.predicted_entry_valid_count)));
  status.values.push_back(key_value(
    "rejected_malformed_objects", std::to_string(result.rejected_malformed_objects)));
  status.values.push_back(key_value(
    "rejected_over_budget", std::to_string(result.rejected_over_budget)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void CutInRiskNode::record_runtime(
  const CutInFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++cut_in_messages_published_;
    total_risk_objects_ += result.objects_in;
    candidate_object_frames_ += result.candidate_count;
    predicted_entry_valid_total_ += result.predicted_entry_valid_count;
    for (const auto & risk : result.output.risks) {
      if (risk.cut_in_candidate) {
        std::array<std::uint8_t, 16U> uuid{};
        std::copy(risk.object_id.uuid.begin(), risk.object_id.uuid.end(), uuid.begin());
        candidate_uuids_.insert(uuid);
        RCLCPP_DEBUG(
          get_logger(),
          "CUT_IN_CANDIDATE uuid=%s side=%u s=%.3f d=%.3f toward=%.3f "
          "entry_t=%.3f ttc_valid=%d ttc=%.3f cpa_valid=%d cpa_t=%.3f "
          "cpa_d=%.3f min_sep_valid=%d min_sep=%.3f",
          uuid_text(uuid).c_str(), risk.side, risk.route_s_rel_m,
          risk.lateral_offset_m, risk.lateral_velocity_toward_corridor_mps,
          risk.predicted_entry_time_s, risk.ttc_valid, risk.ttc_s,
          risk.cpa_valid, risk.cpa_time_s, risk.cpa_distance_m,
          risk.predicted_min_separation_valid,
          risk.predicted_min_separation_m);
      }
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
    "CUT_IN_RISK_RUNTIME_SUMMARY risk_messages=%zu published=%zu rejected=%zu "
    "risk_objects=%zu candidate_object_frames=%zu unique_candidate_uuids=%zu "
    "predicted_entry_valid=%zu latency_ms_median=%.4f latency_ms_p95=%.4f "
    "latency_ms_max=%.4f",
    risk_messages_received_, cut_in_messages_published_, rejected_frames_,
    total_risk_objects_, candidate_object_frames_, candidate_uuids_.size(),
    predicted_entry_valid_total_, quantile(0.5), quantile(0.95), ordered.back());
}

}  // namespace ad_planner
