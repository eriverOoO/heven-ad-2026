#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_NODE_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_NODE_HPP_

#include <ad_interfaces/msg/highway_merge_gap_response.hpp>
#include <ad_interfaces/msg/highway_merge_gap_risk_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ad_planner/planning/highway_merge_gap_response.hpp"

namespace ad_planner
{

struct HighwayMergeGapResponseFrameConfig
{
  double maximum_input_age_s{0.5};
  std::int64_t clock_rollback_threshold_ns{500'000'000};
  std::string expected_frame_id{"map"};
  std::string expected_merge_zone_id{"kcity_highway_onramp"};
  // Must be <= the risk node's maximum_ego_approach_distance_m so a far ego is
  // never reported active by this node before the risk node stops evaluating.
  double maximum_approach_distance_m{400.0};
  HighwayMergeGapResponseParameters policy;
};

struct HighwayMergeGapResponseFrameResult
{
  ad_interfaces::msg::HighwayMergeGapResponse output;
  bool published{false};
  std::string reject_reason;
  HighwayMergeGapResponseAction action{HighwayMergeGapResponseAction::kMergeReady};
};

HighwayMergeGapResponseFrameResult build_highway_merge_gap_response_frame(
  const ad_interfaces::msg::HighwayMergeGapRiskArray & risks,
  std::int64_t now_ns,
  std::optional<std::int64_t> last_risk_stamp_ns,
  const HighwayMergeGapResponseFrameConfig & config);

class HighwayMergeGapResponseNode final : public rclcpp::Node
{
public:
  explicit HighwayMergeGapResponseNode(const rclcpp::NodeOptions & options);

private:
  void on_risks(ad_interfaces::msg::HighwayMergeGapRiskArray::ConstSharedPtr message);
  void publish_diagnostics(
    const std_msgs::msg::Header & header,
    const HighwayMergeGapResponseFrameResult & result,
    double latency_ms);
  void record_runtime(
    const HighwayMergeGapResponseFrameResult & result, double latency_ms);

  HighwayMergeGapResponseFrameConfig config_;
  std::string output_topic_;
  std::string diagnostics_topic_;
  std::mutex mutex_;
  std::optional<std::int64_t> last_risk_stamp_ns_;
  rclcpp::Subscription<ad_interfaces::msg::HighwayMergeGapRiskArray>::SharedPtr subscription_;
  rclcpp::Publisher<ad_interfaces::msg::HighwayMergeGapResponse>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
  std::size_t runtime_summary_interval_frames_{200U};
  std::size_t received_{0U};
  std::size_t published_{0U};
  std::size_t rejected_{0U};
  std::size_t merge_ready_{0U};
  std::size_t wait_{0U};
  std::size_t hold_{0U};
  std::size_t inactive_{0U};
  std::array<std::size_t, 9U> reasons_{};
  std::vector<double> latency_ms_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_NODE_HPP_
