#ifndef AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_
#define AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_

#include <ad_lidar_perception/tracking/cv_predictor.hpp>
#include <ad_lidar_perception/tracking/imm_predictor.hpp>

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ad_lidar_perception::tracking
{

struct AutowarePredictionAdapterConfig
{
  std::string expected_frame_id{"odom"};
  double maximum_input_age_sec{0.5};
  CvPredictionConfig prediction;
  ImmConfig imm_prediction;
  double imm_track_retention_sec{1.0};
};

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

  AutowarePredictionAdapterConfig config_;
  std::unique_ptr<StatefulImmPredictionAdapter> imm_adapter_;
  std::optional<std::int64_t> last_successful_stamp_ns_;
  rclcpp::Publisher<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
    publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;
  rclcpp::Subscription<autoware_perception_msgs::msg::TrackedObjects>::SharedPtr
    subscription_;
};

} // namespace ad_lidar_perception::tracking

#endif // AD_LIDAR_PERCEPTION__TRACKING__AUTOWARE_PREDICTION_NODE_HPP_
