#include "roundabout_gap_risk_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
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

#include "ad_planner/io/roundabout_conflict_loader.hpp"
#include "ad_planner/io/route_corridor_loader.hpp"
#include "ad_planner/local_planning/common/local_motion_frame.hpp"
#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

namespace ad_planner
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

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
    throw std::invalid_argument("duration is not representable");
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

RoundaboutFrameResult rejected(std::string reason)
{
  RoundaboutFrameResult result;
  result.reason = std::move(reason);
  return result;
}

RoundaboutObjectInput unpack(const ad_interfaces::msg::DynamicObjectRisk & source)
{
  RoundaboutObjectInput object;
  std::copy(
    source.object_id.uuid.begin(), source.object_id.uuid.end(),
    object.object_id.begin());
  object.classification = source.classification;
  object.classification_probability = source.classification_probability;
  object.existence_probability = source.existence_probability;
  object.x_rel_m = source.x_rel_m;
  object.y_rel_m = source.y_rel_m;
  object.predicted_states.reserve(source.predicted_states.size());
  for (const auto & state : source.predicted_states) {
    object.predicted_states.push_back(RoundaboutPredictedStateInput{
      duration_seconds(state.time_from_start), state.x_rel_m, state.y_rel_m});
  }
  object.ttc_valid = source.ttc_valid;
  object.ttc_s = source.ttc_s;
  object.cpa_valid = source.cpa_valid;
  object.cpa_time_s = source.cpa_time_s;
  object.cpa_distance_m = source.cpa_distance_m;
  object.predicted_min_separation_valid = source.predicted_min_separation_valid;
  object.predicted_min_separation_m = source.predicted_min_separation_m;
  object.predicted_min_separation_time_s = source.predicted_min_separation_time_s;
  return object;
}

ad_interfaces::msg::RoundaboutGapRisk serialize(const RoundaboutGapRiskResult & source)
{
  ad_interfaces::msg::RoundaboutGapRisk output;
  std::copy(
    source.object_id.begin(), source.object_id.end(),
    output.object_id.uuid.begin());
  output.classification = source.classification;
  output.classification_probability = source.classification_probability;
  output.existence_probability = source.existence_probability;
  output.relevant_to_conflict = source.relevant_to_conflict;
  output.object_map_distance_to_conflict_m =
    static_cast<float>(source.object_map_distance_to_conflict_m);
  output.object_in_conflict_now = source.object_in_conflict_now;
  output.object_entry_valid = source.object_entry_valid;
  output.object_entry_time_s = static_cast<float>(
    source.object_entry_valid ? source.object_entry_time_s : 0.0);
  output.object_exit_valid = source.object_exit_valid;
  output.object_exit_time_s = static_cast<float>(
    source.object_exit_valid ? source.object_exit_time_s : 0.0);
  output.arrival_delta_valid = source.arrival_delta_valid;
  output.arrival_delta_s = static_cast<float>(
    source.arrival_delta_valid ? source.arrival_delta_s : 0.0);
  output.temporal_gap_valid = source.temporal_gap_valid;
  output.temporal_gap_s = static_cast<float>(
    source.temporal_gap_valid ? source.temporal_gap_s : 0.0);
  output.occupancy_overlap = source.occupancy_overlap;
  output.ttc_valid = source.ttc_valid;
  output.ttc_s = static_cast<float>(source.ttc_valid ? source.ttc_s : 0.0);
  output.cpa_valid = source.cpa_valid;
  output.cpa_time_s = static_cast<float>(source.cpa_valid ? source.cpa_time_s : 0.0);
  output.cpa_distance_m = static_cast<float>(
    source.cpa_valid ? source.cpa_distance_m : 0.0);
  output.predicted_min_separation_valid = source.predicted_min_separation_valid;
  output.predicted_min_separation_m = static_cast<float>(
    source.predicted_min_separation_valid ? source.predicted_min_separation_m : 0.0);
  output.predicted_min_separation_time_s = static_cast<float>(
    source.predicted_min_separation_valid ?
    source.predicted_min_separation_time_s : 0.0);
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

RoundaboutFrameResult build_roundabout_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const std::optional<RoundaboutEgoSample> & ego,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_risk_stamp_ns,
  const ReferenceCorridor & corridor,
  const RoundaboutConflictZone & zone,
  const RoundaboutFrameConfig & config)
{
  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(risks.header.stamp);
  } catch (const std::exception & error) {
    return rejected(error.what());
  }
  if (risks.header.frame_id != config.risk_frame_id) {
    return rejected("risk array frame is not '" + config.risk_frame_id + "'");
  }
  if (last_risk_stamp_ns.has_value() && stamp_ns <= *last_risk_stamp_ns) {
    return rejected("risk array is a duplicate or backward frame");
  }
  try {
    if (stamp_ns > now_ns + seconds_to_ns(config.maximum_future_skew_s)) {
      return rejected("risk array stamp is in the future");
    }
    if (now_ns - stamp_ns > seconds_to_ns(config.maximum_input_age_s)) {
      return rejected("risk array is stale");
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
  if (corridor.frame_id.empty() || corridor.lanes.empty() ||
    corridor.primary_lane_index >= corridor.lanes.size())
  {
    return rejected("active reference corridor is invalid");
  }

  const ReferenceLane & primary =
    corridor.lanes[corridor.primary_lane_index];
  double ego_route_s_m = 0.0;
  try {
    const FrenetState ego_frenet = project_to_frenet(
      primary, EgoState{ego->map_pose, ego->longitudinal_speed_mps, 0.0});
    ego_route_s_m = ego_frenet.s_m;
  } catch (const std::exception & error) {
    return rejected(std::string("ego route projection failed: ") + error.what());
  }

  std::vector<RoundaboutObjectInput> objects;
  objects.reserve(risks.objects.size());
  std::size_t rejected_during_unpack = 0U;
  for (const auto & risk : risks.objects) {
    try {
      objects.push_back(unpack(risk));
    } catch (const std::exception &) {
      ++rejected_during_unpack;
    }
  }

  RoundaboutGapComputation computation;
  try {
    computation = compute_roundabout_gap_risks(
      zone,
      RoundaboutEgoState{
        ego->map_pose, ego->longitudinal_speed_mps, ego_route_s_m},
      objects, config.gap);
  } catch (const std::exception & error) {
    return rejected(std::string("roundabout context is invalid: ") + error.what());
  }

  RoundaboutFrameResult result;
  result.published = true;
  result.objects_in = risks.objects.size();
  result.rejected_malformed_objects =
    rejected_during_unpack + computation.rejected_malformed_objects;
  result.rejected_over_budget = computation.rejected_over_budget;
  result.relevant_object_count = computation.relevant_object_count;
  result.ego_entry_valid = computation.ego.entry_valid;

  auto & output = result.output;
  output.header = risks.header;
  output.header.frame_id = corridor.frame_id;
  output.conflict_zone_id = zone.id;
  output.ego_in_conflict_now = computation.ego.in_conflict_now;
  output.ego_entry_valid = computation.ego.entry_valid;
  output.ego_entry_time_s = static_cast<float>(
    computation.ego.entry_valid ? computation.ego.entry_time_s : 0.0);
  output.ego_exit_valid = computation.ego.exit_valid;
  output.ego_exit_time_s = static_cast<float>(
    computation.ego.exit_valid ? computation.ego.exit_time_s : 0.0);
  output.ego_route_distance_to_entry_m =
    static_cast<float>(computation.ego.route_distance_to_entry_m);
  output.ego_route_distance_to_exit_m =
    static_cast<float>(computation.ego.route_distance_to_exit_m);
  output.ego_speed_mps = static_cast<float>(ego->longitudinal_speed_mps);
  output.relevant_object_count =
    static_cast<std::uint16_t>(std::min<std::size_t>(
      computation.relevant_object_count,
      std::numeric_limits<std::uint16_t>::max()));

  output.objects.reserve(computation.objects.size());
  for (const auto & risk : computation.objects) {
    result.object_entry_valid_count += risk.object_entry_valid ? 1U : 0U;
    result.object_exit_valid_count += risk.object_exit_valid ? 1U : 0U;
    result.temporal_gap_valid_count += risk.temporal_gap_valid ? 1U : 0U;
    result.occupancy_overlap_count += risk.occupancy_overlap ? 1U : 0U;
    output.objects.push_back(serialize(risk));
  }
  result.objects_out = output.objects.size();
  return result;
}

RoundaboutGapRiskNode::RoundaboutGapRiskNode(const rclcpp::NodeOptions & options)
: Node("ad_roundabout_gap_risk", options), transform_timeout_(0, 0)
{
  const std::string data_dir = resolve_data_dir(
    declare_parameter<std::string>("data_dir", ""));
  std::filesystem::path corridor_path = declare_parameter<std::string>(
    "route_corridor_file", "map/route_corridor.json");
  if (corridor_path.is_relative()) {
    corridor_path = std::filesystem::path(data_dir) / corridor_path;
  }
  const auto loaded_corridor = load_route_corridor(
    corridor_path,
    {{"global_path", declare_parameter<std::string>(
      "route_corridor.expected_global_path_sha256", "")}});
  corridor_ = loaded_corridor.corridor;
  route_frame_id_ = corridor_.frame_id;

  std::filesystem::path conflict_path = declare_parameter<std::string>(
    "conflict_geometry_file", "");
  if (conflict_path.empty()) {
    conflict_path = std::filesystem::path(
      ament_index_cpp::get_package_share_directory("ad_planner")) /
      "config" / "roundabout_conflicts.json";
  }
  const std::string zone_id = declare_parameter<std::string>(
    "conflict_zone_id", "kcity_roundabout");
  const auto loaded_conflict = load_roundabout_conflict(conflict_path, zone_id);
  zone_ = loaded_conflict.zone;

  // Cross-check that the conflict polygon belongs to the loaded route: the
  // primary route centerline must lie inside the polygon at the recorded entry,
  // exit, and midpoint stations. This ties the two configs together without a
  // file hash.
  if (corridor_.lanes.empty() ||
    corridor_.primary_lane_index >= corridor_.lanes.size())
  {
    throw std::runtime_error("roundabout gap risk: reference corridor has no primary lane");
  }
  const ReferenceLane & primary = corridor_.lanes[corridor_.primary_lane_index];
  const double consistency_margin_m = declare_parameter<double>(
    "polygon_consistency_margin_m", 2.0);
  if (!(consistency_margin_m >= 0.0) || !std::isfinite(consistency_margin_m)) {
    throw std::invalid_argument("polygon_consistency_margin_m must be finite and nonnegative");
  }
  for (const double station :
    {zone_.route_s_enter_m, zone_.route_s_exit_m,
      0.5 * (zone_.route_s_enter_m + zone_.route_s_exit_m)})
  {
    FrenetState centerline;
    centerline.s_m = station;
    const auto sample = frenet_to_cartesian(primary, centerline, 0.0);
    const double x_m = sample.pose.x;
    const double y_m = sample.pose.y;
    if (!point_in_conflict_polygon(zone_.polygon_m, x_m, y_m) &&
      distance_to_conflict_polygon(zone_.polygon_m, x_m, y_m) >
      consistency_margin_m)
    {
      throw std::runtime_error(
              "roundabout gap risk: conflict polygon is not consistent with the "
              "loaded route corridor at station " + std::to_string(station));
    }
  }

  const std::string risk_topic = declare_parameter<std::string>(
    "topics.dynamic_object_risks", "/ad/planning/dynamic_object_risks");
  const std::string odometry_topic = declare_parameter<std::string>(
    "topics.odometry", "/ad/localization/odometry");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/roundabout_gap_risks");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/roundabout_gap_risks/diagnostics");
  config_.risk_frame_id = declare_parameter<std::string>(
    "risk_frame_id", "base_link");
  config_.maximum_input_age_s = declare_parameter<double>(
    "maximum_input_age_s", 0.5);
  config_.maximum_odometry_skew_s = declare_parameter<double>(
    "maximum_odometry_skew_s", 0.5);
  config_.maximum_future_skew_s = declare_parameter<double>(
    "maximum_future_skew_s", 0.10);
  config_.gap.ego_speed_epsilon_mps = declare_parameter<double>(
    "ego_speed_epsilon_mps", 0.5);
  config_.gap.maximum_ego_approach_distance_m = declare_parameter<double>(
    "maximum_ego_approach_distance_m", 400.0);
  config_.gap.maximum_objects = nonnegative_size(
    declare_parameter<int>("maximum_objects", 256), "maximum_objects");
  config_.gap = config_.gap.validated();
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
    risk_topic.empty() || odometry_topic.empty() ||
    output_topic_.empty() || diagnostics_topic_.empty())
  {
    throw std::invalid_argument("roundabout gap risk node parameters are invalid");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  publisher_ = create_publisher<ad_interfaces::msg::RoundaboutGapRiskArray>(
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
  RCLCPP_INFO(
    get_logger(),
    "ad_roundabout_gap_risk: %s -> %s (zone '%s', %zu polygon vertices, route s "
    "[%.2f, %.2f]) using %s",
    risk_topic.c_str(), output_topic_.c_str(), zone_.id.c_str(),
    zone_.polygon_m.size(), zone_.route_s_enter_m, zone_.route_s_exit_m,
    conflict_path.c_str());
}

void RoundaboutGapRiskNode::on_odometry(
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

std::optional<RoundaboutEgoSample> RoundaboutGapRiskNode::transform_ego(
  const std::int64_t stamp_ns)
{
  if (!latest_odometry_.has_value()) {
    return std::nullopt;
  }
  try {
    const auto & odometry = *latest_odometry_;
    const rclcpp::Time stamp(stamp_ns, RCL_ROS_TIME);
    FrameTransform2 route_from_odometry;
    if (route_frame_id_ != odometry.header.frame_id) {
      const auto transform = tf_buffer_->lookupTransform(
        route_frame_id_, odometry.header.frame_id, stamp, transform_timeout_);
      route_from_odometry = FrameTransform2{
        transform.transform.translation.x,
        transform.transform.translation.y,
        tf2::getYaw(transform.transform.rotation)};
    }
    const Pose2 odometry_pose{
      odometry.pose.pose.position.x, odometry.pose.pose.position.y,
      tf2::getYaw(odometry.pose.pose.orientation)};
    return RoundaboutEgoSample{
      transform_pose(route_from_odometry, odometry_pose),
      odometry.twist.twist.linear.x,
      stamp_to_ns(odometry.header.stamp)};
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Roundabout route transform unavailable: %s", error.what());
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Roundabout odometry transform rejected: %s", error.what());
  }
  return std::nullopt;
}

void RoundaboutGapRiskNode::on_risks(
  const ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  ++risk_messages_received_;
  const auto started = std::chrono::steady_clock::now();

  std::int64_t stamp_ns = 0;
  try {
    stamp_ns = stamp_to_ns(message->header.stamp);
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Rejected roundabout risk input: %s", error.what());
    ++rejected_frames_;
    return;
  }

  std::optional<std::int64_t> effective_last = last_risk_stamp_ns_;
  if (last_risk_stamp_ns_.has_value() && stamp_ns < *last_risk_stamp_ns_ &&
    *last_risk_stamp_ns_ - stamp_ns > config_.clock_rollback_threshold_ns)
  {
    last_risk_stamp_ns_.reset();
    effective_last.reset();
  }

  const auto ego = transform_ego(stamp_ns);
  const RoundaboutFrameResult result = build_roundabout_frame(
    *message, ego, now().nanoseconds(), effective_last, corridor_, zone_,
    config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();

  if (result.published) {
    publisher_->publish(result.output);
    last_risk_stamp_ns_ = stamp_ns;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Roundabout frame rejected: %s",
      result.reason.c_str());
  }
  publish_diagnostics(message->header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void RoundaboutGapRiskNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const RoundaboutFrameResult & result, const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "roundabout_gap_risk";
  status.hardware_id = output_topic_;
  status.level = result.published ? diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.published ? "ok" : result.reason;
  status.values.push_back(key_value("published", result.published ? "true" : "false"));
  status.values.push_back(key_value("reject_reason", result.reason));
  status.values.push_back(key_value("objects_in", std::to_string(result.objects_in)));
  status.values.push_back(key_value("objects_out", std::to_string(result.objects_out)));
  status.values.push_back(
    key_value("relevant_objects", std::to_string(result.relevant_object_count)));
  status.values.push_back(
    key_value("ego_entry_valid", result.ego_entry_valid ? "true" : "false"));
  status.values.push_back(key_value(
    "object_entry_valid", std::to_string(result.object_entry_valid_count)));
  status.values.push_back(key_value(
    "temporal_gap_valid", std::to_string(result.temporal_gap_valid_count)));
  status.values.push_back(key_value(
    "occupancy_overlap", std::to_string(result.occupancy_overlap_count)));
  status.values.push_back(key_value(
    "rejected_malformed_objects", std::to_string(result.rejected_malformed_objects)));
  status.values.push_back(key_value(
    "rejected_over_budget", std::to_string(result.rejected_over_budget)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void RoundaboutGapRiskNode::record_runtime(
  const RoundaboutFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++gap_messages_published_;
    total_risk_objects_ += result.objects_in;
    relevant_object_frames_ += result.relevant_object_count;
    object_entry_valid_total_ += result.object_entry_valid_count;
    temporal_gap_valid_total_ += result.temporal_gap_valid_count;
    occupancy_overlap_total_ += result.occupancy_overlap_count;
    ego_entry_valid_frames_ += result.ego_entry_valid ? 1U : 0U;
    for (const auto & risk : result.output.objects) {
      if (!risk.relevant_to_conflict) {
        continue;
      }
      std::array<std::uint8_t, 16U> uuid{};
      std::copy(risk.object_id.uuid.begin(), risk.object_id.uuid.end(), uuid.begin());
      relevant_uuids_.insert(uuid);
      RCLCPP_DEBUG(
        get_logger(),
        "ROUNDABOUT_GAP uuid=%s in_now=%d obj_entry=%.3f obj_exit_valid=%d "
        "obj_exit=%.3f arrival_delta_valid=%d arrival_delta=%.3f gap_valid=%d "
        "gap=%.3f overlap=%d",
        uuid_text(uuid).c_str(), risk.object_in_conflict_now,
        risk.object_entry_time_s, risk.object_exit_valid, risk.object_exit_time_s,
        risk.arrival_delta_valid, risk.arrival_delta_s, risk.temporal_gap_valid,
        risk.temporal_gap_s, risk.occupancy_overlap);
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
    "ROUNDABOUT_GAP_RISK_RUNTIME_SUMMARY risk_messages=%zu published=%zu "
    "rejected=%zu risk_objects=%zu relevant_object_frames=%zu "
    "unique_relevant_uuids=%zu object_entry_valid=%zu temporal_gap_valid=%zu "
    "occupancy_overlap=%zu ego_entry_valid_frames=%zu latency_ms_median=%.4f "
    "latency_ms_p95=%.4f latency_ms_max=%.4f",
    risk_messages_received_, gap_messages_published_, rejected_frames_,
    total_risk_objects_, relevant_object_frames_, relevant_uuids_.size(),
    object_entry_valid_total_, temporal_gap_valid_total_,
    occupancy_overlap_total_, ego_entry_valid_frames_, quantile(0.5),
    quantile(0.95), ordered.back());
}

}  // namespace ad_planner
