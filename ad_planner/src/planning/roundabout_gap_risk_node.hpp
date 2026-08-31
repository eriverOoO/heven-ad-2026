#ifndef AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_NODE_HPP_
#define AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_NODE_HPP_

#include <ad_interfaces/msg/dynamic_object_risk_array.hpp>
#include <ad_interfaces/msg/roundabout_gap_risk_array.hpp>
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
#include "ad_planner/planning/roundabout_gap_risk.hpp"

namespace ad_planner
{

struct RoundaboutEgoSample
{
  Pose2 map_pose;
  double longitudinal_speed_mps{0.0};
  std::int64_t stamp_ns{0};
};

struct RoundaboutFrameConfig
{
  std::string risk_frame_id{"base_link"};
  double maximum_input_age_s{0.5};
  double maximum_odometry_skew_s{0.5};
  double maximum_future_skew_s{0.10};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  RoundaboutGapParameters gap;
};

struct RoundaboutFrameResult
{
  ad_interfaces::msg::RoundaboutGapRiskArray output;
  bool published{false};
  std::string reason;
  std::size_t objects_in{0U};
  std::size_t objects_out{0U};
  std::size_t relevant_object_count{0U};
  std::size_t object_entry_valid_count{0U};
  std::size_t object_exit_valid_count{0U};
  std::size_t temporal_gap_valid_count{0U};
  std::size_t occupancy_overlap_count{0U};
  std::size_t multi_interval_object_count{0U};
  std::size_t later_reentry_count{0U};
  std::size_t any_occupancy_overlap_count{0U};
  std::size_t minimum_temporal_gap_valid_count{0U};
  std::size_t prediction_covers_ego_exit_count{0U};
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};
  bool ego_entry_valid{false};
};

RoundaboutFrameResult build_roundabout_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const std::optional<RoundaboutEgoSample> & ego,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const ReferenceCorridor & corridor,
  const RoundaboutConflictZone & zone,
  const RoundaboutFrameConfig & config);

class RoundaboutGapRiskNode final : public rclcpp::Node
{
public:
  explicit RoundaboutGapRiskNode(const rclcpp::NodeOptions & options);

private:
  void on_risks(ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  std::optional<RoundaboutEgoSample> transform_ego(std::int64_t stamp_ns);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const RoundaboutFrameResult & result, double latency_ms);
  void record_runtime(const RoundaboutFrameResult & result, double latency_ms);

  ReferenceCorridor corridor_;
  RoundaboutConflictZone zone_;
  RoundaboutFrameConfig config_;
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
  rclcpp::Publisher<ad_interfaces::msg::RoundaboutGapRiskArray>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;

  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t risk_messages_received_{0U};
  std::size_t gap_messages_published_{0U};
  std::size_t rejected_frames_{0U};
  std::size_t total_risk_objects_{0U};
  std::size_t relevant_object_frames_{0U};
  std::size_t object_entry_valid_total_{0U};
  std::size_t temporal_gap_valid_total_{0U};
  std::size_t occupancy_overlap_total_{0U};
  std::size_t multi_interval_object_total_{0U};
  std::size_t later_reentry_total_{0U};
  std::size_t any_occupancy_overlap_total_{0U};
  std::size_t minimum_temporal_gap_valid_total_{0U};
  std::size_t prediction_covers_ego_exit_total_{0U};
  std::size_t ego_entry_valid_frames_{0U};
  std::set<std::array<std::uint8_t, 16U>> relevant_uuids_;
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_NODE_HPP_
