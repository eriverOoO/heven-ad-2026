#pragma once

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "ad_viz/perception/marker_builder.hpp"

namespace ad_viz::perception
{

rclcpp::QoS predicted_object_input_qos();
rclcpp::QoS perception_marker_output_qos();

class MarkerPublicationState
{
public:
  using Clock = std::chrono::steady_clock;

  explicit MarkerPublicationState(std::chrono::nanoseconds stale_timeout);

  bool accepts(std::int64_t stamp_ns) const noexcept;
  void record_clock_rollback() noexcept;
  bool clock_rollback_reset_due(std::int64_t stamp_ns) const noexcept;
  void reset_for_clock_rollback() noexcept;
  void record_successful_publication(
    std::int64_t stamp_ns, Clock::time_point receipt_time);
  bool stale_clear_due(Clock::time_point now) const noexcept;
  void record_stale_clear_publication();

private:
  std::chrono::nanoseconds stale_timeout_;
  std::optional<std::int64_t> last_successful_stamp_ns_;
  std::optional<Clock::time_point> last_successful_receipt_time_;
  bool stale_clear_published_{false};
  bool clock_rollback_pending_{false};
};

class PerceptionMarkerNode final : public rclcpp::Node
{
public:
  explicit PerceptionMarkerNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_predictions(
    const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input);
  void on_stale_timer();

  MarkerBuilderConfig builder_config_;
  MarkerPublicationState publication_state_;
  std::chrono::nanoseconds stale_check_period_;
  std::atomic<bool> clock_rollback_observed_{false};
  std::mutex state_mutex_;
  std::optional<std_msgs::msg::Header> last_successful_header_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
  rclcpp::Subscription<
    ad_interfaces::msg::PredictedObjectArray>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr stale_timer_;
  rclcpp::JumpHandler::SharedPtr clock_jump_handler_;
};

}  // namespace ad_viz::perception
