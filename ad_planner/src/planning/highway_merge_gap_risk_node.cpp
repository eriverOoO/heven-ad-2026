#include "highway_merge_gap_risk_node.hpp"

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

#include "ad_planner/io/merge_geometry_loader.hpp"
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

MergeFrameResult rejected(std::string reason)
{
  MergeFrameResult result;
  result.reason = std::move(reason);
  return result;
}

MergeObjectInput unpack(const ad_interfaces::msg::DynamicObjectRisk & source)
{
  MergeObjectInput object;
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
  for (const auto & state : source.predicted_states) {
    object.predicted_states.push_back(MergePredictedStateInput{
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

ad_interfaces::msg::HighwayMergeGapRisk serialize(
  const HighwayMergeGapRiskResult & source)
{
  ad_interfaces::msg::HighwayMergeGapRisk output;
  std::copy(
    source.object_id.begin(), source.object_id.end(),
    output.object_id.uuid.begin());
  output.classification = source.classification;
  output.classification_probability = source.classification_probability;
  output.existence_probability = source.existence_probability;
  output.relevant_to_merge = source.relevant_to_merge;
  output.object_route_s_m = static_cast<float>(source.object_route_s_m);
  output.object_lateral_offset_m = static_cast<float>(source.object_lateral_offset_m);
  output.object_in_target_corridor_now = source.object_in_target_corridor_now;
  output.predicted_to_enter_target_corridor =
    source.predicted_to_enter_target_corridor;
  output.predicted_corridor_entry_valid = source.predicted_corridor_entry_valid;
  output.predicted_corridor_entry_time_s = static_cast<float>(
    source.predicted_corridor_entry_valid ?
    source.predicted_corridor_entry_time_s : 0.0);
  output.delta_s_now_m = static_cast<float>(source.delta_s_now_m);
  output.object_longitudinal_speed_mps =
    static_cast<float>(source.object_longitudinal_speed_mps);
  output.relative_longitudinal_speed_mps =
    static_cast<float>(source.relative_longitudinal_speed_mps);
  output.delta_s_at_merge_valid = source.delta_s_at_merge_valid;
  output.delta_s_at_merge_m = static_cast<float>(
    source.delta_s_at_merge_valid ? source.delta_s_at_merge_m : 0.0);
  output.is_ahead_at_merge = source.is_ahead_at_merge;
  output.is_behind_at_merge = source.is_behind_at_merge;
  output.is_alongside_at_merge = source.is_alongside_at_merge;
  output.longitudinal_gap_closing = source.longitudinal_gap_closing;
  output.longitudinal_closing_speed_mps =
    static_cast<float>(source.longitudinal_closing_speed_mps);
  output.time_to_route_coincidence_valid = source.time_to_route_coincidence_valid;
  output.time_to_route_coincidence_s = static_cast<float>(
    source.time_to_route_coincidence_valid ?
    source.time_to_route_coincidence_s : 0.0);
  output.predicted_min_route_gap_valid = source.predicted_min_route_gap_valid;
  output.predicted_min_route_gap_m = static_cast<float>(
    source.predicted_min_route_gap_valid ? source.predicted_min_route_gap_m : 0.0);
  output.predicted_min_route_gap_time_s = static_cast<float>(
    source.predicted_min_route_gap_valid ?
    source.predicted_min_route_gap_time_s : 0.0);
  output.prediction_horizon_s = static_cast<float>(source.prediction_horizon_s);
  output.prediction_covers_merge_time = source.prediction_covers_merge_time;
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

MergeFrameResult build_highway_merge_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const std::optional<MergeEgoSample> & ego,
  const std::int64_t now_ns,
  const std::optional<std::int64_t> last_risk_stamp_ns,
  const std::string & output_frame_id,
  const ReferenceLane & ego_lane,
  const ReferenceLane & target_lane_window,
  const MergeZone & zone,
  const MergeFrameConfig & config)
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
  if (output_frame_id.empty() || ego_lane.points.size() < 2U ||
    target_lane_window.points.size() < 2U)
  {
    return rejected("active merge corridor is invalid");
  }

  double ego_route_s_m = 0.0;
  double ego_route_speed_mps = 0.0;
  try {
    const FrenetState ego_frenet = project_to_frenet(
      ego_lane, EgoState{ego->map_pose, ego->longitudinal_speed_mps, 0.0});
    ego_route_s_m = ego_frenet.s_m;
    ego_route_speed_mps = ego_frenet.s_dot_mps;
  } catch (const std::exception & error) {
    return rejected(std::string("ego route projection failed: ") + error.what());
  }

  std::vector<MergeObjectInput> objects;
  objects.reserve(risks.objects.size());
  std::size_t rejected_during_unpack = 0U;
  for (const auto & risk : risks.objects) {
    try {
      objects.push_back(unpack(risk));
    } catch (const std::exception &) {
      ++rejected_during_unpack;
    }
  }

  const MergeEgoState ego_state{
    ego->map_pose, ego->longitudinal_speed_mps, ego_route_s_m,
    ego_route_speed_mps};

  HighwayMergeGapComputation computation;
  try {
    computation = compute_highway_merge_gap_risks(
      zone, target_lane_window, ego_state, objects, config.merge);
  } catch (const std::exception & error) {
    return rejected(std::string("merge context is invalid: ") + error.what());
  }

  MergeFrameResult result;
  result.published = true;
  result.objects_in = risks.objects.size();
  result.rejected_malformed_objects =
    rejected_during_unpack + computation.rejected_malformed_objects;
  result.rejected_over_budget = computation.rejected_over_budget;
  result.relevant_object_count = computation.relevant_object_count;
  result.ego_merge_timing_valid = computation.ego.timing_valid;
  result.ego_in_merge_zone_now = computation.ego.in_merge_zone_now;
  result.merge_gap_valid = computation.merge_gap_valid;

  auto & output = result.output;
  output.header = risks.header;
  output.header.frame_id = output_frame_id;
  output.merge_zone_id = zone.id;
  output.target_lane_sequence_id = ego_lane.lane_sequence_id;
  output.source_lane_sequence_id = zone.source_lane_id;
  output.merge_zone_entry_route_s_m =
    static_cast<float>(zone.route_s_zone_entry_m);
  output.merge_reference_route_s_m =
    static_cast<float>(computation.merge_reference_route_s_m);
  output.ego_route_s_m = static_cast<float>(ego_route_s_m);
  output.ego_longitudinal_speed_mps = static_cast<float>(ego_route_speed_mps);
  output.ego_route_distance_to_zone_entry_m =
    static_cast<float>(computation.ego.route_distance_to_zone_entry_m);
  output.ego_route_distance_to_merge_m =
    static_cast<float>(computation.ego.route_distance_to_merge_m);
  output.ego_in_merge_zone_now = computation.ego.in_merge_zone_now;
  output.ego_merge_timing_valid = computation.ego.timing_valid;
  output.ego_merge_time_s = static_cast<float>(
    computation.ego.timing_valid ? computation.ego.merge_time_s : 0.0);
  output.relevant_object_count = static_cast<std::uint16_t>(
    std::min<std::size_t>(
      computation.relevant_object_count,
      std::numeric_limits<std::uint16_t>::max()));

  output.nearest_leading_valid = computation.nearest_leading_valid;
  std::copy(
    computation.nearest_leading_object_id.begin(),
    computation.nearest_leading_object_id.end(),
    output.nearest_leading_object_id.uuid.begin());
  output.nearest_leading_delta_s_at_merge_m = static_cast<float>(
    computation.nearest_leading_valid ?
    computation.nearest_leading_delta_s_at_merge_m : 0.0);
  output.nearest_trailing_valid = computation.nearest_trailing_valid;
  std::copy(
    computation.nearest_trailing_object_id.begin(),
    computation.nearest_trailing_object_id.end(),
    output.nearest_trailing_object_id.uuid.begin());
  output.nearest_trailing_delta_s_at_merge_m = static_cast<float>(
    computation.nearest_trailing_valid ?
    computation.nearest_trailing_delta_s_at_merge_m : 0.0);
  output.merge_gap_valid = computation.merge_gap_valid;
  output.merge_gap_m = static_cast<float>(
    computation.merge_gap_valid ? computation.merge_gap_m : 0.0);

  output.objects.reserve(computation.objects.size());
  for (const auto & risk : computation.objects) {
    result.in_corridor_now_count += risk.object_in_target_corridor_now ? 1U : 0U;
    result.predicted_to_enter_count +=
      risk.predicted_to_enter_target_corridor ? 1U : 0U;
    result.delta_s_at_merge_valid_count += risk.delta_s_at_merge_valid ? 1U : 0U;
    result.ahead_count += risk.is_ahead_at_merge ? 1U : 0U;
    result.behind_count += risk.is_behind_at_merge ? 1U : 0U;
    result.alongside_count += risk.is_alongside_at_merge ? 1U : 0U;
    result.gap_closing_count += risk.longitudinal_gap_closing ? 1U : 0U;
    result.coincidence_valid_count +=
      risk.time_to_route_coincidence_valid ? 1U : 0U;
    result.predicted_min_route_gap_valid_count +=
      risk.predicted_min_route_gap_valid ? 1U : 0U;
    result.prediction_covers_merge_time_count +=
      risk.prediction_covers_merge_time ? 1U : 0U;
    output.objects.push_back(serialize(risk));
  }
  result.objects_out = output.objects.size();
  return result;
}

HighwayMergeGapRiskNode::HighwayMergeGapRiskNode(const rclcpp::NodeOptions & options)
: Node("ad_highway_merge_gap_risk", options), transform_timeout_(0, 0)
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
  const ReferenceCorridor & corridor = loaded_corridor.corridor;
  route_frame_id_ = corridor.frame_id;
  output_frame_id_ = corridor.frame_id;

  std::filesystem::path merge_path = declare_parameter<std::string>(
    "merge_geometry_file", "");
  if (merge_path.empty()) {
    merge_path = std::filesystem::path(
      ament_index_cpp::get_package_share_directory("ad_planner")) /
      "config" / "highway_merge.json";
  }
  const std::string zone_id = declare_parameter<std::string>(
    "merge_zone_id", "kcity_highway_onramp");
  const auto loaded_merge = load_merge_geometry(merge_path, zone_id);
  zone_ = loaded_merge.zone;

  // Resolve the target and source lanes in the checksum-verified corridor.
  const auto lane_index = [&](const std::string & id) -> std::size_t {
      for (std::size_t index = 0U; index < corridor.lanes.size(); ++index) {
        if (corridor.lanes[index].lane_sequence_id == id) {
          return index;
        }
      }
      throw std::runtime_error(
              "highway merge gap risk: lane '" + id +
              "' is not in the loaded route corridor");
    };
  const std::size_t target_lane_index = lane_index(zone_.target_lane_id);
  const std::size_t source_lane_index = lane_index(zone_.source_lane_id);

  const double consistency_margin_m = declare_parameter<double>(
    "merge_geometry_consistency_margin_m", 2.0);
  if (!(consistency_margin_m >= 0.0) || !std::isfinite(consistency_margin_m)) {
    throw std::invalid_argument(
            "merge_geometry_consistency_margin_m must be finite and nonnegative");
  }

  // The merge-zone stations must match the source acceleration lane's own
  // route_s_m span, and the target corridor must cover the whole zone.
  const ReferenceLane & source_lane = corridor.lanes[source_lane_index];
  const double source_first_s = source_lane.points.front().route_s_m;
  const double source_last_s = source_lane.points.back().route_s_m;
  if (std::abs(source_first_s - zone_.route_s_zone_entry_m) > consistency_margin_m ||
    std::abs(source_last_s - zone_.route_s_merge_complete_m) > consistency_margin_m)
  {
    throw std::runtime_error(
            "highway merge gap risk: merge zone stations are not consistent with "
            "source lane '" + zone_.source_lane_id + "' in the route corridor");
  }
  const ReferenceLane & target_lane = corridor.lanes[target_lane_index];
  if (target_lane.points.front().route_s_m > zone_.route_s_zone_entry_m ||
    target_lane.points.back().route_s_m < zone_.route_s_merge_complete_m)
  {
    throw std::runtime_error(
            "highway merge gap risk: target lane '" + zone_.target_lane_id +
            "' does not span the merge zone");
  }
  target_lane_full_ = target_lane;

  const std::string risk_topic = declare_parameter<std::string>(
    "topics.dynamic_object_risks", "/ad/planning/dynamic_object_risks");
  const std::string odometry_topic = declare_parameter<std::string>(
    "topics.odometry", "/ad/localization/odometry");
  output_topic_ = declare_parameter<std::string>(
    "topics.output", "/ad/planning/highway_merge_gap_risks");
  diagnostics_topic_ = declare_parameter<std::string>(
    "topics.diagnostics", "/ad/planning/highway_merge_gap_risks/diagnostics");
  config_.risk_frame_id = declare_parameter<std::string>(
    "risk_frame_id", "base_link");
  config_.maximum_input_age_s = declare_parameter<double>(
    "maximum_input_age_s", 0.5);
  config_.maximum_odometry_skew_s = declare_parameter<double>(
    "maximum_odometry_skew_s", 0.5);
  config_.maximum_future_skew_s = declare_parameter<double>(
    "maximum_future_skew_s", 0.10);
  config_.merge.ego_speed_epsilon_mps = declare_parameter<double>(
    "ego_speed_epsilon_mps", 0.5);
  config_.merge.maximum_ego_approach_distance_m = declare_parameter<double>(
    "maximum_ego_approach_distance_m", 400.0);
  config_.merge.relevant_rear_window_m = declare_parameter<double>(
    "relevant_rear_window_m", 150.0);
  config_.merge.relevant_front_window_m = declare_parameter<double>(
    "relevant_front_window_m", 120.0);
  config_.merge.alongside_longitudinal_band_m = declare_parameter<double>(
    "alongside_longitudinal_band_m", 5.0);
  config_.merge.target_corridor_lateral_margin_m = declare_parameter<double>(
    "target_corridor_lateral_margin_m", 0.5);
  config_.merge.closing_speed_epsilon_mps = declare_parameter<double>(
    "closing_speed_epsilon_mps", 0.1);
  config_.merge.maximum_objects = nonnegative_size(
    declare_parameter<int>("maximum_objects", 256), "maximum_objects");
  config_.merge = config_.merge.validated();

  // Project ego and objects onto a station-bounded window of the target lane
  // around the merge zone: it fully contains the ego approach bound and the
  // object relevance window, and keeps per-frame O(objects * samples * points)
  // projection cost bounded regardless of the full route length. An object
  // outside the window clamps to a window end (large lateral offset) and is
  // correctly reported not relevant_to_merge.
  const double window_pad_m = declare_parameter<double>(
    "merge_corridor_window_pad_m", 30.0);
  if (!(window_pad_m >= 0.0) || !std::isfinite(window_pad_m)) {
    throw std::invalid_argument("merge_corridor_window_pad_m must be finite and nonnegative");
  }
  {
    const double back_reach = std::max(
      config_.merge.maximum_ego_approach_distance_m,
      config_.merge.relevant_rear_window_m) + window_pad_m;
    const double window_lo = zone_.route_s_zone_entry_m - back_reach;
    const double window_hi = zone_.route_s_merge_complete_m +
      config_.merge.relevant_front_window_m + window_pad_m;
    target_lane_window_.lane_sequence_id = target_lane.lane_sequence_id;
    target_lane_window_.source_link_ids = target_lane.source_link_ids;
    for (const auto & point : target_lane.points) {
      if (point.route_s_m >= window_lo && point.route_s_m <= window_hi) {
        target_lane_window_.points.push_back(point);
      }
    }
    if (target_lane_window_.points.size() < 2U) {
      throw std::runtime_error(
              "highway merge gap risk: target lane window is degenerate");
    }
    if (target_lane_window_.points.front().route_s_m > zone_.route_s_zone_entry_m ||
      target_lane_window_.points.back().route_s_m < zone_.route_s_merge_complete_m)
    {
      throw std::runtime_error(
              "highway merge gap risk: target lane window does not span the merge zone");
    }
  }

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
    throw std::invalid_argument("highway merge gap risk node parameters are invalid");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  publisher_ = create_publisher<ad_interfaces::msg::HighwayMergeGapRiskArray>(
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
    "ad_highway_merge_gap_risk: %s -> %s (zone '%s', target lane '%s', source "
    "lane '%s', route s [%.2f, %.2f]) using %s",
    risk_topic.c_str(), output_topic_.c_str(), zone_.id.c_str(),
    zone_.target_lane_id.c_str(), zone_.source_lane_id.c_str(),
    zone_.route_s_zone_entry_m, zone_.route_s_merge_complete_m,
    merge_path.c_str());
}

void HighwayMergeGapRiskNode::on_odometry(
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

std::optional<MergeEgoSample> HighwayMergeGapRiskNode::transform_ego(
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
    return MergeEgoSample{
      transform_pose(route_from_odometry, odometry_pose),
      odometry.twist.twist.linear.x,
      stamp_to_ns(odometry.header.stamp)};
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Highway merge route transform unavailable: %s", error.what());
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Highway merge odometry transform rejected: %s", error.what());
  }
  return std::nullopt;
}

void HighwayMergeGapRiskNode::on_risks(
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
      "Rejected highway merge risk input: %s", error.what());
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
  const MergeFrameResult result = build_highway_merge_frame(
    *message, ego, now().nanoseconds(), effective_last, output_frame_id_,
    target_lane_full_, target_lane_window_, zone_, config_);
  const double latency_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();

  if (result.published) {
    publisher_->publish(result.output);
    last_risk_stamp_ns_ = stamp_ns;
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Highway merge frame rejected: %s",
      result.reason.c_str());
  }
  publish_diagnostics(message->header, result, latency_ms);
  record_runtime(result, latency_ms);
}

void HighwayMergeGapRiskNode::publish_diagnostics(
  const std_msgs::msg::Header & header,
  const MergeFrameResult & result, const double latency_ms)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header = header;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "highway_merge_gap_risk";
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
  status.values.push_back(key_value(
    "ego_merge_timing_valid", result.ego_merge_timing_valid ? "true" : "false"));
  status.values.push_back(key_value(
    "ego_in_merge_zone_now", result.ego_in_merge_zone_now ? "true" : "false"));
  status.values.push_back(key_value(
    "in_corridor_now", std::to_string(result.in_corridor_now_count)));
  status.values.push_back(key_value(
    "predicted_to_enter", std::to_string(result.predicted_to_enter_count)));
  status.values.push_back(key_value(
    "delta_s_at_merge_valid", std::to_string(result.delta_s_at_merge_valid_count)));
  status.values.push_back(key_value("ahead", std::to_string(result.ahead_count)));
  status.values.push_back(key_value("behind", std::to_string(result.behind_count)));
  status.values.push_back(
    key_value("alongside", std::to_string(result.alongside_count)));
  status.values.push_back(
    key_value("gap_closing", std::to_string(result.gap_closing_count)));
  status.values.push_back(key_value(
    "coincidence_valid", std::to_string(result.coincidence_valid_count)));
  status.values.push_back(key_value(
    "predicted_min_route_gap_valid",
    std::to_string(result.predicted_min_route_gap_valid_count)));
  status.values.push_back(key_value(
    "prediction_covers_merge_time",
    std::to_string(result.prediction_covers_merge_time_count)));
  status.values.push_back(
    key_value("merge_gap_valid", result.merge_gap_valid ? "true" : "false"));
  status.values.push_back(key_value(
    "rejected_malformed_objects", std::to_string(result.rejected_malformed_objects)));
  status.values.push_back(key_value(
    "rejected_over_budget", std::to_string(result.rejected_over_budget)));
  status.values.push_back(key_value("latency_ms", std::to_string(latency_ms)));
  array.status.push_back(std::move(status));
  diagnostics_publisher_->publish(array);
}

void HighwayMergeGapRiskNode::record_runtime(
  const MergeFrameResult & result, const double latency_ms)
{
  if (result.published) {
    ++gap_messages_published_;
    total_risk_objects_ += result.objects_in;
    relevant_object_frames_ += result.relevant_object_count;
    in_corridor_now_total_ += result.in_corridor_now_count;
    predicted_to_enter_total_ += result.predicted_to_enter_count;
    delta_s_at_merge_valid_total_ += result.delta_s_at_merge_valid_count;
    ahead_total_ += result.ahead_count;
    behind_total_ += result.behind_count;
    alongside_total_ += result.alongside_count;
    gap_closing_total_ += result.gap_closing_count;
    coincidence_valid_total_ += result.coincidence_valid_count;
    predicted_min_route_gap_valid_total_ +=
      result.predicted_min_route_gap_valid_count;
    prediction_covers_merge_time_total_ +=
      result.prediction_covers_merge_time_count;
    ego_merge_timing_valid_frames_ += result.ego_merge_timing_valid ? 1U : 0U;
    merge_gap_valid_frames_ += result.merge_gap_valid ? 1U : 0U;
    for (const auto & risk : result.output.objects) {
      if (!risk.relevant_to_merge) {
        continue;
      }
      std::array<std::uint8_t, 16U> uuid{};
      std::copy(risk.object_id.uuid.begin(), risk.object_id.uuid.end(), uuid.begin());
      relevant_uuids_.insert(uuid);
      RCLCPP_DEBUG(
        get_logger(),
        "HIGHWAY_MERGE uuid=%s s=%.2f d=%.2f in_now=%d pred_enter=%d "
        "delta_s_now=%.2f obj_long_v=%.2f rel_long_v=%.2f dsm_valid=%d "
        "delta_s_at_merge=%.2f ahead=%d behind=%d alongside=%d closing=%d "
        "close_v=%.2f coincide=%.2f min_gap_valid=%d min_gap=%.2f "
        "horizon=%.2f covers=%d",
        uuid_text(uuid).c_str(), risk.object_route_s_m, risk.object_lateral_offset_m,
        risk.object_in_target_corridor_now, risk.predicted_to_enter_target_corridor,
        risk.delta_s_now_m, risk.object_longitudinal_speed_mps,
        risk.relative_longitudinal_speed_mps, risk.delta_s_at_merge_valid,
        risk.delta_s_at_merge_m, risk.is_ahead_at_merge, risk.is_behind_at_merge,
        risk.is_alongside_at_merge, risk.longitudinal_gap_closing,
        risk.longitudinal_closing_speed_mps, risk.time_to_route_coincidence_s,
        risk.predicted_min_route_gap_valid, risk.predicted_min_route_gap_m,
        risk.prediction_horizon_s, risk.prediction_covers_merge_time);
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
    "HIGHWAY_MERGE_GAP_RISK_RUNTIME_SUMMARY risk_messages=%zu published=%zu "
    "rejected=%zu risk_objects=%zu relevant_object_frames=%zu "
    "unique_relevant_uuids=%zu in_corridor_now=%zu predicted_to_enter=%zu "
    "delta_s_at_merge_valid=%zu ahead=%zu behind=%zu alongside=%zu "
    "gap_closing=%zu coincidence_valid=%zu predicted_min_route_gap_valid=%zu "
    "prediction_covers_merge_time=%zu ego_merge_timing_valid_frames=%zu "
    "merge_gap_valid_frames=%zu latency_ms_median=%.4f latency_ms_p95=%.4f "
    "latency_ms_max=%.4f",
    risk_messages_received_, gap_messages_published_, rejected_frames_,
    total_risk_objects_, relevant_object_frames_, relevant_uuids_.size(),
    in_corridor_now_total_, predicted_to_enter_total_,
    delta_s_at_merge_valid_total_, ahead_total_, behind_total_, alongside_total_,
    gap_closing_total_, coincidence_valid_total_,
    predicted_min_route_gap_valid_total_, prediction_covers_merge_time_total_,
    ego_merge_timing_valid_frames_, merge_gap_valid_frames_, quantile(0.5),
    quantile(0.95), ordered.back());
}

}  // namespace ad_planner
