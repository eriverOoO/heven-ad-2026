#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_NODE_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_NODE_HPP_

#include <ad_interfaces/msg/dynamic_object_risk_array.hpp>
#include <ad_interfaces/msg/highway_merge_gap_risk_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"
#include "ad_planner/planning/highway_merge_gap_risk.hpp"

namespace ad_planner
{

struct MergeEgoSample
{
  Pose2 map_pose;
  double longitudinal_speed_mps{0.0};
  std::int64_t stamp_ns{0};
};

struct MergeFrameConfig
{
  std::string risk_frame_id{"base_link"};
  double maximum_input_age_s{0.5};
  double maximum_odometry_skew_s{0.5};
  double maximum_future_skew_s{0.10};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  HighwayMergeParameters merge;
};

struct MergeFrameResult
{
  ad_interfaces::msg::HighwayMergeGapRiskArray output;
  bool published{false};
  std::string reason;
  std::size_t objects_in{0U};
  std::size_t objects_out{0U};
  std::size_t relevant_object_count{0U};
  std::size_t in_corridor_now_count{0U};
  std::size_t predicted_to_enter_count{0U};
  std::size_t delta_s_at_merge_valid_count{0U};
  std::size_t ahead_count{0U};
  std::size_t behind_count{0U};
  std::size_t alongside_count{0U};
  std::size_t gap_closing_count{0U};
  std::size_t coincidence_valid_count{0U};
  std::size_t predicted_min_route_gap_valid_count{0U};
  std::size_t prediction_covers_merge_time_count{0U};
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};
  bool ego_merge_timing_valid{false};
  bool ego_in_merge_zone_now{false};
  bool merge_gap_valid{false};
};

// ego_lane is the full target lane (the ego drives the whole route, so it must
// project against the full centerline). target_lane_window is a station-bounded
// window of that lane around the merge zone -- objects project against it to
// keep per-frame cost independent of the route length; an object outside the
// window clamps to a window end and is reported not relevant_to_merge.
// output_frame_id is the corridor frame ("map").
MergeFrameResult build_highway_merge_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const std::optional<MergeEgoSample> & ego,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const std::string & output_frame_id,
  const ReferenceLane & ego_lane,
  const ReferenceLane & target_lane_window,
  const MergeZone & zone,
  const MergeFrameConfig & config);

class HighwayMergeGapRiskNode final : public rclcpp::Node
{
public:
  explicit HighwayMergeGapRiskNode(const rclcpp::NodeOptions & options);

private:
  void on_risks(ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  std::optional<MergeEgoSample> transform_ego(std::int64_t stamp_ns);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const MergeFrameResult & result, double latency_ms);
  void record_runtime(const MergeFrameResult & result, double latency_ms);

  ReferenceLane target_lane_full_;
  ReferenceLane target_lane_window_;
  std::string output_frame_id_;
  MergeZone zone_;
  MergeFrameConfig config_;
  std::string output_topic_;
  std::string diagnostics_topic_;
  std::string route_frame_id_;
  rclcpp::Duration transform_timeout_;

  std::optional<nav_msgs::msg::Odometry> latest_odometry_;
  std::optional<std::int64_t> last_risk_stamp_ns_;
  std::mutex mutex_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<ad_interfaces::msg::DynamicObjectRiskArray>::SharedPtr
    risk_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Publisher<ad_interfaces::msg::HighwayMergeGapRiskArray>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;

  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t risk_messages_received_{0U};
  std::size_t gap_messages_published_{0U};
  std::size_t rejected_frames_{0U};
  std::size_t total_risk_objects_{0U};
  std::size_t relevant_object_frames_{0U};
  std::size_t in_corridor_now_total_{0U};
  std::size_t predicted_to_enter_total_{0U};
  std::size_t delta_s_at_merge_valid_total_{0U};
  std::size_t ahead_total_{0U};
  std::size_t behind_total_{0U};
  std::size_t alongside_total_{0U};
  std::size_t gap_closing_total_{0U};
  std::size_t coincidence_valid_total_{0U};
  std::size_t predicted_min_route_gap_valid_total_{0U};
  std::size_t prediction_covers_merge_time_total_{0U};
  std::size_t ego_merge_timing_valid_frames_{0U};
  std::size_t merge_gap_valid_frames_{0U};
  std::set<std::array<std::uint8_t, 16U>> relevant_uuids_;
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_NODE_HPP_
