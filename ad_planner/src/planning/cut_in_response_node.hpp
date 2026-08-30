#ifndef AD_PLANNER__PLANNING__CUT_IN_RESPONSE_NODE_HPP_
#define AD_PLANNER__PLANNING__CUT_IN_RESPONSE_NODE_HPP_

#include <ad_interfaces/msg/cut_in_response.hpp>
#include <ad_interfaces/msg/cut_in_risk_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ad_planner/planning/cut_in_response.hpp"

namespace ad_planner
{

struct EgoSpeedSample
{
  double speed_mps{0.0};
  std::int64_t stamp_ns{0};
};

struct CutInResponseFrameConfig
{
  double maximum_input_age_s{0.5};
  double maximum_odometry_skew_s{0.5};
  double maximum_future_skew_s{0.10};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  CutInResponseParameters policy;
};

struct CutInResponseFrameResult
{
  ad_interfaces::msg::CutInResponse output;
  bool published{false};
  std::string reason;
  std::size_t candidates_in{0U};
  std::size_t rejected_malformed{0U};
  std::size_t rejected_over_budget{0U};
  CutInResponseAction action{CutInResponseAction::kNone};
};

CutInResponseFrameResult build_cut_in_response_frame(
  const ad_interfaces::msg::CutInRiskArray & risks,
  const std::optional<EgoSpeedSample> & ego,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const CutInResponseFrameConfig & config);

class CutInResponseNode final : public rclcpp::Node
{
public:
  explicit CutInResponseNode(const rclcpp::NodeOptions & options);

private:
  void on_risks(ad_interfaces::msg::CutInRiskArray::ConstSharedPtr message);
  void on_odometry(nav_msgs::msg::Odometry::ConstSharedPtr message);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const CutInResponseFrameResult & result,
    double latency_ms);
  void record_runtime(const CutInResponseFrameResult & result, double latency_ms);

  CutInResponseFrameConfig config_;
  std::string output_topic_;
  std::string diagnostics_topic_;

  std::mutex mutex_;
  std::optional<nav_msgs::msg::Odometry> latest_odometry_;
  std::optional<std::int64_t> last_risk_stamp_ns_;

  rclcpp::Subscription<ad_interfaces::msg::CutInRiskArray>::SharedPtr risk_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
  rclcpp::Publisher<ad_interfaces::msg::CutInResponse>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;

  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t risk_messages_received_{0U};
  std::size_t response_messages_published_{0U};
  std::size_t rejected_frames_{0U};
  std::size_t slowdown_frames_{0U};
  std::size_t hold_frames_{0U};
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__CUT_IN_RESPONSE_NODE_HPP_
