#ifndef AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_NODE_HPP_
#define AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_NODE_HPP_

#include <ad_interfaces/msg/roundabout_gap_response.hpp>
#include <ad_interfaces/msg/roundabout_gap_risk_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ad_planner/planning/roundabout_gap_response.hpp"

namespace ad_planner
{

struct RoundaboutGapResponseFrameConfig
{
  double maximum_input_age_s{0.5};
  std::string expected_frame_id{"map"};
  std::string expected_conflict_zone_id{"kcity_roundabout"};
  RoundaboutGapResponseParameters policy;
};

struct RoundaboutGapResponseFrameResult
{
  ad_interfaces::msg::RoundaboutGapResponse output;
  bool published{false};
  std::string reject_reason;
  RoundaboutGapResponseAction action{RoundaboutGapResponseAction::kRelease};
};

RoundaboutGapResponseFrameResult build_roundabout_gap_response_frame(
  const ad_interfaces::msg::RoundaboutGapRiskArray & risks,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const RoundaboutGapResponseFrameConfig & config);

class RoundaboutGapResponseNode final : public rclcpp::Node
{
public:
  explicit RoundaboutGapResponseNode(const rclcpp::NodeOptions & options);

private:
  void on_risks(ad_interfaces::msg::RoundaboutGapRiskArray::ConstSharedPtr message);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const RoundaboutGapResponseFrameResult & result,
    double latency_ms);
  void record_runtime(const RoundaboutGapResponseFrameResult & result, double latency_ms);

  RoundaboutGapResponseFrameConfig config_;
  std::string output_topic_;
  std::string diagnostics_topic_;
  std::mutex mutex_;
  std::optional<std::int64_t> last_risk_stamp_ns_;
  rclcpp::Subscription<ad_interfaces::msg::RoundaboutGapRiskArray>::SharedPtr subscription_;
  rclcpp::Publisher<ad_interfaces::msg::RoundaboutGapResponse>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t received_{0U};
  std::size_t published_{0U};
  std::size_t rejected_{0U};
  std::size_t release_{0U};
  std::size_t yield_{0U};
  std::size_t hold_{0U};
  std::array<std::size_t, 6U> reasons_{};
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_NODE_HPP_
