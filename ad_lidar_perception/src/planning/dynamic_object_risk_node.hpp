#ifndef AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_NODE_HPP_
#define AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_NODE_HPP_

#include "ad_lidar_perception/planning/dynamic_object_risk.hpp"

#include <ad_interfaces/msg/dynamic_object_risk_array.hpp>
#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ad_lidar_perception::planning
{

// Pure transform used by both the node and its tests: canonical prediction
// array + ego state -> planner-facing risk array. Never throws; a rejected
// frame is reported through `reason` with an empty `objects` list.
struct RiskFrameResult
{
  ad_interfaces::msg::DynamicObjectRiskArray risks;
  bool published{false};
  std::string reason;  // empty on success
  std::size_t objects_in{0};
  std::size_t objects_out{0};
  std::size_t rejected_non_finite_state{0};
  std::size_t rejected_non_finite_dimensions{0};
  std::size_t rejected_over_budget{0};
  std::size_t ttc_valid_count{0};
  std::size_t cpa_valid_count{0};
};

struct EgoSample
{
  EgoState state;
  std::int64_t stamp_ns{0};
};

struct RiskNodeConfig
{
  std::string prediction_frame_id{"odom"};
  std::string output_frame_id{"base_link"};
  double maximum_prediction_age_s{0.5};
  double maximum_ego_age_s{0.5};
  double maximum_future_skew_s{0.10};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  RiskParameters risk;
};

// `now_ns` is the wall/ROS clock; `ego` is the latest accepted odometry sample
// (nullopt when none has arrived); `last_prediction_stamp_ns` is the stamp of
// the previous accepted prediction frame.
RiskFrameResult build_risk_frame(
  const ad_interfaces::msg::PredictedObjectArray & prediction,
  const std::optional<EgoSample> & ego,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_prediction_stamp_ns,
  const RiskNodeConfig & config);

class DynamicObjectRiskNode final : public rclcpp::Node
{
public:
  explicit DynamicObjectRiskNode(const rclcpp::NodeOptions & options);

private:
  void on_prediction(ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr message);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  void publish_diagnostics(
    const std_msgs::msg::Header & header, const RiskFrameResult & result,
    double latency_ms);
  void record_runtime_metrics(const RiskFrameResult & result, double latency_ms);

  RiskNodeConfig config_;
  std::string output_topic_;

  rclcpp::Subscription<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
    prediction_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Publisher<ad_interfaces::msg::DynamicObjectRiskArray>::SharedPtr
    risk_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;

  std::optional<EgoSample> latest_ego_;
  std::optional<std::pair<double, double>> last_ego_position_;
  std::optional<std::int64_t> last_prediction_stamp_ns_;

  std::size_t runtime_summary_interval_frames_{180};
  std::size_t predicted_frames_{0};
  std::size_t risk_frames_{0};
  std::size_t rejected_frames_{0};
  std::size_t objects_in_total_{0};
  std::size_t objects_out_total_{0};
  std::size_t rejected_objects_total_{0};
  std::size_t ttc_valid_total_{0};
  std::size_t cpa_valid_total_{0};
  std::size_t ego_twist_contradiction_frames_{0};
  std::vector<double> latency_ms_;
};

}  // namespace ad_lidar_perception::planning

#endif  // AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_NODE_HPP_
