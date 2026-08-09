#include "ad_lidar_perception/preprocessing/pointcloud_densifier.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cctype>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_lidar_perception::preprocessing
{
namespace
{

struct PreviousObservation
{
  builtin_interfaces::msg::Time stamp;
  std::string frame;
};

bool valid_relative_frame(const std::string & frame)
{
  if (frame.empty() || frame.front() == '/') {
    return false;
  }
  for (const auto character : frame) {
    if (std::isspace(static_cast<unsigned char>(character)) != 0) {
      return false;
    }
  }
  return true;
}

DensifierTransform from_message(
  const geometry_msgs::msg::TransformStamped & message)
{
  return {
    {
      message.transform.translation.x,
      message.transform.translation.y,
      message.transform.translation.z,
    },
    {
      message.transform.rotation.x,
      message.transform.rotation.y,
      message.transform.rotation.z,
      message.transform.rotation.w,
    },
  };
}

const char * status_name(const DensifierStatus status)
{
  switch (status) {
    case DensifierStatus::kFirstFrame:
      return "first_frame";
    case DensifierStatus::kFused:
      return "fused";
    case DensifierStatus::kNoEligibleHistory:
      return "no_eligible_history";
    case DensifierStatus::kNonIncreasingStamp:
      return "non_increasing_stamp";
    case DensifierStatus::kStaleHistory:
      return "stale_history";
    case DensifierStatus::kSchemaMismatch:
      return "schema_mismatch";
    case DensifierStatus::kFrameMismatch:
      return "frame_mismatch";
    case DensifierStatus::kTransformUnavailable:
      return "transform_unavailable";
    case DensifierStatus::kTransformTranslationJump:
      return "transform_translation_jump";
    case DensifierStatus::kTransformRotationJump:
      return "transform_rotation_jump";
    case DensifierStatus::kMalformedCurrent:
      return "malformed_current";
    case DensifierStatus::kMalformedHistory:
      return "malformed_history";
    case DensifierStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

}  // namespace

class PointcloudDensifierNode final : public rclcpp::Node
{
public:
  PointcloudDensifierNode()
  : Node("ad_pointcloud_densifier"),
    input_topic_(declare_parameter<std::string>(
        "topics.input", "/ad/perception/lidar/nonground_finite")),
    output_topic_(declare_parameter<std::string>(
        "topics.output", "/ad/perception/lidar/nonground_densified")),
    fixed_frame_(declare_parameter<std::string>("fixed_frame", "odom")),
    transform_timeout_sec_(
      declare_parameter<double>("transform_timeout_sec", 0.05)),
    densifier_(load_config()),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    if (input_topic_.empty() || output_topic_.empty()) {
      throw std::invalid_argument("densifier topics must be nonempty");
    }
    if (
      get_node_topics_interface()->resolve_topic_name(input_topic_) ==
      get_node_topics_interface()->resolve_topic_name(output_topic_))
    {
      throw std::invalid_argument("densifier input and output topics must differ");
    }
    if (!valid_relative_frame(fixed_frame_)) {
      throw std::invalid_argument(
              "densifier fixed_frame must be a valid relative frame");
    }
    if (!std::isfinite(transform_timeout_sec_) || transform_timeout_sec_ < 0.0) {
      throw std::invalid_argument(
              "densifier transform_timeout_sec must be finite and nonnegative");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr input) {
        on_cloud(*input);
      });
  }

private:
  DensifierConfig load_config()
  {
    DensifierConfig config;
    config.voxel_size_m = declare_parameter<double>("voxel_size_m", 0.30);
    config.roi_min_x_m = declare_parameter<double>("roi.min_x_m", 20.0);
    config.roi_max_x_m = declare_parameter<double>("roi.max_x_m", 100.0);
    config.roi_min_y_m = declare_parameter<double>("roi.min_y_m", -12.0);
    config.roi_max_y_m = declare_parameter<double>("roi.max_y_m", 12.0);
    config.maximum_history_age_sec = declare_parameter<double>(
      "maximum_history_age_sec", 0.25);
    config.maximum_translation_jump_m = declare_parameter<double>(
      "maximum_translation_jump_m", 5.0);
    config.maximum_rotation_jump_rad = declare_parameter<double>(
      "maximum_rotation_jump_rad", 0.35);
    return config;
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2 & input)
  {
    std::optional<DensifierTransform> current_from_previous;
    if (densifier_.has_history() && previous_observation_) {
      try {
        current_from_previous = from_message(tf_buffer_.lookupTransform(
            input.header.frame_id, rclcpp::Time(input.header.stamp),
            previous_observation_->frame,
            rclcpp::Time(previous_observation_->stamp), fixed_frame_,
            rclcpp::Duration::from_seconds(transform_timeout_sec_)));
      } catch (const tf2::TransformException & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "densifier TF unavailable; publishing current-only: %s", error.what());
        current_from_previous = std::nullopt;
      } catch (const std::exception & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "densifier TF request invalid; publishing current-only: %s", error.what());
        current_from_previous = std::nullopt;
      }
    }

    auto result = densifier_.process(input, current_from_previous);
    if (densifier_.has_history()) {
      previous_observation_ = PreviousObservation{
        input.header.stamp, input.header.frame_id};
    } else {
      previous_observation_.reset();
    }
    if (
      result.status != DensifierStatus::kFirstFrame &&
      result.status != DensifierStatus::kFused &&
      result.status != DensifierStatus::kNoEligibleHistory)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "densifier current-only fallback: %s", status_name(result.status));
    }
    publisher_->publish(std::move(result.cloud));
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string fixed_frame_;
  double transform_timeout_sec_;
  PointcloudDensifier densifier_;
  std::optional<PreviousObservation> previous_observation_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

std::shared_ptr<rclcpp::Node> make_pointcloud_densifier_node()
{
  return std::make_shared<PointcloudDensifierNode>();
}

}  // namespace ad_lidar_perception::preprocessing
