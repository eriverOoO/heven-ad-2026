#ifndef AD_PLANNER__PLANNING__CUT_IN_RISK_NODE_HPP_
#define AD_PLANNER__PLANNING__CUT_IN_RISK_NODE_HPP_

#include <ad_interfaces/msg/cut_in_risk_array.hpp>
#include <ad_interfaces/msg/dynamic_object_risk_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
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
#include <utility>
#include <vector>

#include "ad_planner/common/exact_stamp_pairer.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"
#include "ad_planner/planning/cut_in_risk.hpp"

namespace ad_planner
{

struct RouteEgoSample
{
  CutInEgoState state;
  std::int64_t stamp_ns{0};
};

struct CutInFrameConfig
{
  std::string risk_frame_id{"base_link"};
  std::string route_mask_frame_id{"base_link"};
  double maximum_input_age_s{0.5};
  double maximum_odometry_skew_s{0.5};
  double maximum_future_skew_s{0.10};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  CutInParameters cut_in;
};

struct CutInFrameResult
{
  ad_interfaces::msg::CutInRiskArray output;
  bool published{false};
  std::string reason;
  std::size_t objects_in{0U};
  std::size_t objects_out{0U};
  std::size_t candidate_count{0U};
  std::size_t predicted_entry_valid_count{0U};
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};
};

CutInFrameResult build_cut_in_frame(
  const ad_interfaces::msg::DynamicObjectRiskArray & risks,
  const nav_msgs::msg::OccupancyGrid & route_mask,
  const std::optional<RouteEgoSample> & ego,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const ReferenceCorridor & corridor,
  const CutInFrameConfig & config);

class CutInRiskNode final : public rclcpp::Node
{
public:
  explicit CutInRiskNode(const rclcpp::NodeOptions & options);

private:
  using Pairer = ExactStampPairer<
    ad_interfaces::msg::DynamicObjectRiskArray,
    nav_msgs::msg::OccupancyGrid>;

  void on_risks(ad_interfaces::msg::DynamicObjectRiskArray::ConstSharedPtr message);
  void on_route_mask(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  void process_pair(Pairer::Pair pair);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const CutInFrameResult & result,
    double latency_ms);
  void record_runtime(
    const CutInFrameResult & result, double latency_ms);

  ReferenceCorridor corridor_;
  CutInFrameConfig config_;
  std::string output_topic_;
  std::string diagnostics_topic_;
  std::optional<nav_msgs::msg::Odometry> latest_odometry_;
  std::optional<std::int64_t> last_risk_stamp_ns_;
  Pairer pairer_;
  std::mutex mutex_;
  rclcpp::Duration transform_timeout_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<ad_interfaces::msg::DynamicObjectRiskArray>::SharedPtr
    risk_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr mask_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Publisher<ad_interfaces::msg::CutInRiskArray>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;

  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t risk_messages_received_{0U};
  std::size_t cut_in_messages_published_{0U};
  std::size_t rejected_frames_{0U};
  std::size_t total_risk_objects_{0U};
  std::size_t candidate_object_frames_{0U};
  std::size_t predicted_entry_valid_total_{0U};
  std::set<std::array<std::uint8_t, 16U>> candidate_uuids_;
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__CUT_IN_RISK_NODE_HPP_
