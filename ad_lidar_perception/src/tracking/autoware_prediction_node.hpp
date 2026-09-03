#ifndef AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_
#define AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_

#include <ad_lidar_perception/tracking/cv_predictor.hpp>
#include <ad_lidar_perception/tracking/imm_predictor.hpp>

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ad_lidar_perception::tracking
{

// Source of the yaw-rate measurement fed into the IMM coordinated-turn model.
//   kTracker        - use tracked_objects.twist.twist.angular.z verbatim
//                     (reproduces the pre-Curve-Aware-Prediction behaviour;
//                     AB3DMOT has no angular-velocity state so this is 0).
//   kMotionHistory  - derive a robust turn rate from recent world-frame
//                     velocity history; fall back to the tracker value when
//                     the conservative validity gates reject the window.
enum class YawRateSource
{
  kTracker,
  kMotionHistory,
};

// Curve-Aware Prediction v1: bounded per-track velocity-history turn-rate
// estimator. Parameter surface deliberately minimal. Defaults chosen from the
// morai_cam4_20260813_163222 prediction audit
// (docs/perception/curve_aware_prediction_v1.md).
struct MotionHistoryYawRateConfig
{
  // Consecutive world-velocity samples required before an estimate is produced.
  // history_samples - 1 adjacent heading slopes are then reduced by a median.
  // Must be >= 3 (a 2-slope median is not outlier-robust).
  std::size_t history_samples{4U};
  // Samples slower than this carry pure KF velocity noise in atan2(vy, vx);
  // any sub-threshold sample in the window rejects the estimate (-> CV).
  double min_speed_mps{2.0};
  // Hard clamp on |omega_est|. Safety ceiling, not a tuning target.
  double max_yaw_rate_radps{1.5};
  // Fixed variance reported for the derived rate. NOT a measured angular
  // velocity: a finite difference over ~history_samples/tracking_rate seconds of
  // noisy KF velocity. Intentionally larger than the 0.04 tracker-twist
  // fallback.
  double yaw_rate_variance_rad2ps2{0.10};
};

struct AutowarePredictionAdapterConfig
{
  std::string expected_frame_id{"odom"};
  double maximum_input_age_sec{0.5};
  CvPredictionConfig prediction;
  ImmConfig imm_prediction;
  double imm_track_retention_sec{1.0};
  YawRateSource yaw_rate_source{YawRateSource::kTracker};
  MotionHistoryYawRateConfig motion_history;
};

// Parses the yaw_rate_source parameter string ("tracker" | "motion_history").
// Throws std::invalid_argument on an unrecognised value.
YawRateSource parse_yaw_rate_source(const std::string & value);

ad_interfaces::msg::PredictedObjectArray adapt_tracked_objects(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  std::int64_t now_ns, std::optional<std::int64_t> last_successful_stamp_ns,
  const AutowarePredictionAdapterConfig & config);

rclcpp::QoS prediction_output_qos();

struct PredictionAdaptation
{
  ad_interfaces::msg::PredictedObjectArray predictions;
  diagnostic_msgs::msg::DiagnosticArray diagnostics;
};

enum class ImmUpdateReason
{
  kTrackInitialized,
  kMeasurementAccepted,
  kRetentionExpired,
  kClockRollback,
  kUpdateIntervalClamped,
};

diagnostic_msgs::msg::DiagnosticArray rejected_prediction_diagnostics(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const std::string & rejection_message,
  std::optional<std::array<std::uint8_t, 16>> culprit_uuid = std::nullopt);

class StatefulImmPredictionAdapter
{
public:
  explicit StatefulImmPredictionAdapter(AutowarePredictionAdapterConfig config);
  ~StatefulImmPredictionAdapter();

  StatefulImmPredictionAdapter(StatefulImmPredictionAdapter &&) noexcept;
  StatefulImmPredictionAdapter &
  operator=(StatefulImmPredictionAdapter &&) noexcept;
  StatefulImmPredictionAdapter(const StatefulImmPredictionAdapter &) = delete;
  StatefulImmPredictionAdapter &
  operator=(const StatefulImmPredictionAdapter &) = delete;

  ad_interfaces::msg::PredictedObjectArray
  adapt(
    const autoware_perception_msgs::msg::TrackedObjects & input,
    std::int64_t now_ns,
    std::optional<std::int64_t> last_successful_stamp_ns);
  PredictionAdaptation adapt_with_diagnostics(
    const autoware_perception_msgs::msg::TrackedObjects & input,
    std::int64_t now_ns,
    std::optional<std::int64_t> last_successful_stamp_ns);
  void reset(
    ImmUpdateReason reason = ImmUpdateReason::kTrackInitialized) noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class AutowarePredictionNode final : public rclcpp::Node
{
public:
  explicit AutowarePredictionNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_tracked_objects(
    const autoware_perception_msgs::msg::TrackedObjects::ConstSharedPtr
    input);
  void record_runtime_metrics(
    std::size_t input_objects, std::size_t output_objects,
    std::size_t unavailable_orientation_objects, double step_latency_ms);

  AutowarePredictionAdapterConfig config_;
  std::unique_ptr<StatefulImmPredictionAdapter> imm_adapter_;
  std::optional<std::int64_t> last_successful_stamp_ns_;
  std::size_t runtime_summary_interval_frames_{0U};
  std::size_t rejected_arrays_{0U};
  std::size_t input_objects_{0U};
  std::size_t output_objects_{0U};
  std::size_t unavailable_orientation_objects_{0U};
  std::vector<double> step_latency_ms_;
  rclcpp::Publisher<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
    publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;
  rclcpp::Subscription<autoware_perception_msgs::msg::TrackedObjects>::SharedPtr
    subscription_;
};

} // namespace ad_lidar_perception::tracking

#endif // AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_
