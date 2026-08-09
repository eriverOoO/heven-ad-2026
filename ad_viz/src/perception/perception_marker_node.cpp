#include "ad_viz/perception/perception_marker_node.hpp"

#include <rclcpp/create_timer.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_viz::perception
{
namespace
{

std::chrono::nanoseconds positive_duration(
  const double seconds, const char * field_name)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    throw std::invalid_argument(
            std::string(field_name) + " must be finite and positive");
  }
  const long double nanoseconds =
    static_cast<long double>(seconds) * 1000000000.0L;
  if (!std::isfinite(nanoseconds) ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument(
            std::string(field_name) + " is not representable");
  }
  const auto rounded = static_cast<std::int64_t>(std::llround(nanoseconds));
  if (rounded <= 0) {
    throw std::invalid_argument(
            std::string(field_name) + " rounds below one nanosecond");
  }
  return std::chrono::nanoseconds(rounded);
}

}  // namespace

rclcpp::QoS predicted_object_input_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1))
         .reliable()
         .durability_volatile();
}

rclcpp::QoS perception_marker_output_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1))
         .reliable()
         .durability_volatile();
}

MarkerPublicationState::MarkerPublicationState(
  const std::chrono::nanoseconds stale_timeout)
: stale_timeout_(stale_timeout)
{
  if (stale_timeout_ <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("stale timeout must be positive");
  }
}

void MarkerPublicationState::record_clock_rollback() noexcept
{
  clock_rollback_pending_ = true;
}

bool MarkerPublicationState::clock_rollback_reset_due(
  const std::int64_t stamp_ns) const noexcept
{
  return clock_rollback_pending_ && stamp_ns > 0 &&
         last_successful_stamp_ns_.has_value() &&
         stamp_ns < *last_successful_stamp_ns_;
}

void MarkerPublicationState::reset_for_clock_rollback() noexcept
{
  last_successful_stamp_ns_.reset();
  last_successful_receipt_time_.reset();
  stale_clear_published_ = false;
  clock_rollback_pending_ = false;
}

bool MarkerPublicationState::accepts(const std::int64_t stamp_ns) const noexcept
{
  return stamp_ns > 0 &&
         (!last_successful_stamp_ns_.has_value() ||
         stamp_ns > *last_successful_stamp_ns_);
}

void MarkerPublicationState::record_successful_publication(
  const std::int64_t stamp_ns, const Clock::time_point receipt_time)
{
  if (!accepts(stamp_ns)) {
    throw std::logic_error(
            "successful marker publication stamp must be strictly increasing");
  }
  last_successful_stamp_ns_ = stamp_ns;
  last_successful_receipt_time_ = receipt_time;
  stale_clear_published_ = false;
}

bool MarkerPublicationState::stale_clear_due(
  const Clock::time_point now) const noexcept
{
  return last_successful_receipt_time_.has_value() &&
         !stale_clear_published_ &&
         now > *last_successful_receipt_time_ &&
         now - *last_successful_receipt_time_ > stale_timeout_;
}

void MarkerPublicationState::record_stale_clear_publication()
{
  if (!last_successful_receipt_time_.has_value()) {
    throw std::logic_error(
            "cannot record a stale clear before a successful publication");
  }
  stale_clear_published_ = true;
}

PerceptionMarkerNode::PerceptionMarkerNode(
  const rclcpp::NodeOptions & options)
: Node("ad_perception_marker", options),
  builder_config_{
    declare_parameter<double>("marker_lifetime_sec", 0.5),
    declare_parameter<double>("stale_timeout_sec", 1.0)},
  publication_state_(
    positive_duration(builder_config_.stale_timeout_sec, "stale timeout")),
  stale_check_period_(
    positive_duration(
      declare_parameter<double>("stale_check_period_sec", 0.1),
      "stale check period"))
{
  const auto clock_rollback_reset_threshold = positive_duration(
    declare_parameter<double>("clock_rollback_reset_sec", 5.0),
    "clock rollback reset threshold");
  rcl_jump_threshold_t jump_threshold{};
  jump_threshold.on_clock_change = false;
  jump_threshold.min_forward.nanoseconds = 0;
  jump_threshold.min_backward.nanoseconds =
    -clock_rollback_reset_threshold.count();
  clock_jump_handler_ = get_clock()->create_jump_callback(
    []() noexcept {},
    [this](const rcl_time_jump_t & jump) noexcept {
      if (jump.delta.nanoseconds < 0) {
        clock_rollback_observed_.store(true, std::memory_order_release);
      }
    },
    jump_threshold);

  // Validate all marker-style parameters before creating graph endpoints.
  std_msgs::msg::Header validation_header;
  validation_header.frame_id = "odom";
  validation_header.stamp.nanosec = 1U;
  (void)build_delete_all(validation_header, builder_config_);

  const std::string input_topic = declare_parameter<std::string>(
    "input_topic", "/ad/perception/objects/predicted");
  const std::string output_topic = declare_parameter<std::string>(
    "output_topic", "/ad/viz/perception/objects");
  if (input_topic.empty() || output_topic.empty() ||
    get_node_topics_interface()->resolve_topic_name(input_topic) ==
    get_node_topics_interface()->resolve_topic_name(output_topic))
  {
    throw std::invalid_argument(
            "predicted-object input and marker output topics must be nonempty and distinct");
  }

  publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    output_topic, perception_marker_output_qos());
  subscription_ =
    create_subscription<ad_interfaces::msg::PredictedObjectArray>(
    input_topic, predicted_object_input_qos(),
    [this](
      const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input)
    {
      on_predictions(input);
    });
  stale_timer_ = create_wall_timer(
    stale_check_period_, [this]() {on_stale_timer();});
}

void PerceptionMarkerNode::on_predictions(
  const ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr input)
{
  const auto receipt_time = MarkerPublicationState::Clock::now();
  try {
    const std::int64_t stamp_ns = prediction_stamp_ns(input->header);
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (clock_rollback_observed_.exchange(
        false, std::memory_order_acq_rel))
    {
      publication_state_.record_clock_rollback();
    }
    const bool reset_for_rollback =
      publication_state_.clock_rollback_reset_due(stamp_ns);
    if (!publication_state_.accepts(stamp_ns) && !reset_for_rollback) {
      RCLCPP_WARN(
        get_logger(),
        "Rejected predicted-object array with non-increasing successful stamp");
      return;
    }

    // The pure builder validates the whole array before a single marker can
    // escape. The watermark is advanced only after publish() returns.
    auto output = build_prediction_markers(*input, builder_config_);
    if (reset_for_rollback) {
      RCLCPP_WARN(
        get_logger(),
        "Reset marker timestamp watermark after a large ROS clock rollback");
    }
    publisher_->publish(output);
    if (reset_for_rollback) {
      publication_state_.reset_for_clock_rollback();
    }
    publication_state_.record_successful_publication(stamp_ns, receipt_time);
    last_successful_header_ = input->header;
  } catch (const std::exception & error) {
    RCLCPP_WARN(
      get_logger(), "Rejected predicted-object array: %s", error.what());
  }
}

void PerceptionMarkerNode::on_stale_timer()
{
  try {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!last_successful_header_.has_value() ||
      !publication_state_.stale_clear_due(
        MarkerPublicationState::Clock::now()))
    {
      return;
    }
    auto clear = build_delete_all(*last_successful_header_, builder_config_);
    publisher_->publish(clear);
    publication_state_.record_stale_clear_publication();
  } catch (const std::exception & error) {
    RCLCPP_WARN(
      get_logger(), "Failed to clear stale perception markers: %s", error.what());
  }
}

}  // namespace ad_viz::perception
