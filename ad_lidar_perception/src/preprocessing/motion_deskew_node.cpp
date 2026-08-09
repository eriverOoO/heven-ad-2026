#include "ad_lidar_perception/preprocessing/motion_deskewer.hpp"
#include "ad_lidar_perception/preprocessing/motion_history.hpp"
#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace ad_lidar_perception::preprocessing
{
namespace
{

constexpr std::size_t kMaximumHistorySamples = 4096U;
constexpr auto kDiagnosticThrottle = std::chrono::seconds(2);

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

}  // namespace

class MotionDeskewNode : public rclcpp::Node
{
public:
  MotionDeskewNode()
  : Node("ad_motion_deskew"),
    input_topic_(declare_parameter<std::string>(
        "topics.input", "/ad/sensors/lidar/points")),
    output_topic_(declare_parameter<std::string>(
        "topics.output", "/ad/perception/lidar/deskewed")),
    imu_topic_(declare_parameter<std::string>(
        "topics.imu", "/ad/sensors/imu/data")),
    wheel_topic_(declare_parameter<std::string>(
        "topics.wheel", "/ad/localization/input/wheel_speed")),
    base_frame_(declare_parameter<std::string>("base_frame", "base_link")),
    history_age_sec_(declare_parameter<double>("limits.history_age_sec", 1.0)),
    pending_timeout_sec_(
      declare_parameter<double>("limits.pending_timeout_sec", 0.15)),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    const auto mode = declare_parameter<std::string>("deskew_mode", "3d");
    if (mode == "3d") {
      options_.mode = DeskewMode::kThreeDimensional;
    } else if (mode == "2d") {
      options_.mode = DeskewMode::kTwoDimensional;
    } else {
      throw std::invalid_argument("deskew_mode must be '2d' or '3d'");
    }
    options_.maximum_scan_duration_sec = declare_parameter<double>(
      "limits.maximum_scan_duration_sec", 0.20);
    options_.maximum_imu_gap_sec = declare_parameter<double>(
      "limits.maximum_imu_gap_sec", 0.12);
    options_.maximum_wheel_gap_sec = declare_parameter<double>(
      "limits.maximum_wheel_gap_sec", 0.20);
    options_.maximum_integration_step_sec = declare_parameter<double>(
      "limits.integration_substep_sec", 0.005);
    const auto maximum_point_count = declare_parameter<std::int64_t>(
      "limits.maximum_point_count", 300000);
    const auto pending_depth = declare_parameter<std::int64_t>(
      "limits.pending_depth", 4);
    if (maximum_point_count <= 0 || pending_depth <= 0) {
      throw std::invalid_argument("point and pending limits must be positive");
    }
    options_.maximum_point_count = static_cast<std::size_t>(maximum_point_count);
    pending_depth_ = static_cast<std::size_t>(pending_depth);
    if (
      !std::isfinite(history_age_sec_) || history_age_sec_ <= 0.0 ||
      !std::isfinite(pending_timeout_sec_) || pending_timeout_sec_ <= 0.0 ||
      !std::isfinite(options_.maximum_scan_duration_sec) ||
      options_.maximum_scan_duration_sec <= 0.0 ||
      !std::isfinite(options_.maximum_imu_gap_sec) ||
      options_.maximum_imu_gap_sec <= 0.0 ||
      !std::isfinite(options_.maximum_wheel_gap_sec) ||
      options_.maximum_wheel_gap_sec <= 0.0 ||
      !std::isfinite(options_.maximum_integration_step_sec) ||
      options_.maximum_integration_step_sec <= 0.0)
    {
      throw std::invalid_argument("deskew timing limits must be finite and positive");
    }
    if (!valid_relative_frame(base_frame_)) {
      throw std::invalid_argument("base_frame must be a valid relative frame");
    }
    if (
      input_topic_.empty() || output_topic_.empty() || imu_topic_.empty() ||
      wheel_topic_.empty())
    {
      throw std::invalid_argument("deskew topics must be nonempty");
    }
    const auto resolved_input =
      get_node_topics_interface()->resolve_topic_name(input_topic_);
    const auto resolved_output =
      get_node_topics_interface()->resolve_topic_name(output_topic_);
    if (resolved_input == resolved_output) {
      throw std::invalid_argument("deskew input and output topics must differ");
    }

    history_ = std::make_unique<MotionHistory>(
      history_age_sec_, kMaximumHistorySamples);
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        on_cloud(*message);
      });
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {
        on_imu(*message);
      });
    wheel_subscription_ =
      create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      wheel_topic_, rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr message) {
        on_wheel(*message);
      });
    pending_timer_ = create_wall_timer(
      std::chrono::milliseconds(20), [this]() {process_pending();});
  }

private:
  struct PendingScan
  {
    sensor_msgs::msg::PointCloud2 cloud;
    std::chrono::steady_clock::time_point received;
    std::string last_reason;
  };

  void warn_throttled(const std::string & key, const std::string & message)
  {
    const auto now = std::chrono::steady_clock::now();
    const auto found = last_diagnostic_.find(key);
    if (found == last_diagnostic_.end() || now - found->second >= kDiagnosticThrottle) {
      RCLCPP_WARN(get_logger(), "%s", message.c_str());
      last_diagnostic_[key] = now;
    }
  }

  void reset_state(const std::string & source)
  {
    history_->clear();
    pending_.clear();
    last_cloud_stamp_.reset();
    last_imu_stamp_.reset();
    last_wheel_stamp_.reset();
    warn_throttled(
      "epoch_reset", "deskew state reset after backward " + source + " timestamp");
  }

  void observe_stamp(
    std::optional<std::int64_t> & previous, const std::int64_t current,
    const std::string & source)
  {
    if (previous.has_value() && current < previous.value()) {
      reset_state(source);
    }
    previous = current;
  }

  RigidTransform3d lookup_base_from_frame(
    const std::string & source_frame,
    const builtin_interfaces::msg::Time & stamp)
  {
    if (!valid_relative_frame(source_frame)) {
      throw std::invalid_argument("source frame must be a valid relative frame");
    }
    if (source_frame == base_frame_) {
      return RigidTransform3d{};
    }
    const auto transform = tf_buffer_.lookupTransform(
      base_frame_, source_frame, stamp, rclcpp::Duration::from_seconds(0.0));
    return rigid_transform_from_quaternion(
      {
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
      },
      {
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
      });
  }

  void on_imu(const sensor_msgs::msg::Imu & message)
  {
    try {
      const auto stamp_ns = ros_stamp_nanoseconds(message.header.stamp);
      observe_stamp(last_imu_stamp_, stamp_ns, "IMU");
      if (!valid_relative_frame(message.header.frame_id)) {
        throw std::invalid_argument("IMU frame must be a valid relative frame");
      }
      const std::array<double, 3> source_rate{
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
      };
      if (!std::isfinite(source_rate[0]) || !std::isfinite(source_rate[1]) ||
        !std::isfinite(source_rate[2]))
      {
        throw std::invalid_argument("IMU angular velocity must be finite");
      }
      const auto base_from_imu = lookup_base_from_frame(
        message.header.frame_id, message.header.stamp);
      const auto & rotation = base_from_imu.rotation;
      const std::array<double, 3> base_rate{
        rotation[0] * source_rate[0] + rotation[1] * source_rate[1] +
        rotation[2] * source_rate[2],
        rotation[3] * source_rate[0] + rotation[4] * source_rate[1] +
        rotation[5] * source_rate[2],
        rotation[6] * source_rate[0] + rotation[7] * source_rate[1] +
        rotation[8] * source_rate[2],
      };
      const auto update = history_->add_imu_sample(
        ros_stamp_seconds(message.header.stamp), base_rate);
      if (update == MotionHistoryUpdate::kEpochReset) {
        reset_state("IMU history");
        history_->add_imu_sample(ros_stamp_seconds(message.header.stamp), base_rate);
      }
      process_pending();
    } catch (const std::exception & error) {
      warn_throttled("imu", "deskew rejected IMU sample: " + std::string(error.what()));
    }
  }

  void on_wheel(const geometry_msgs::msg::TwistWithCovarianceStamped & message)
  {
    try {
      const auto stamp_ns = ros_stamp_nanoseconds(message.header.stamp);
      observe_stamp(last_wheel_stamp_, stamp_ns, "wheel");
      if (message.header.frame_id != base_frame_) {
        throw std::invalid_argument("wheel frame must equal configured base_frame");
      }
      const auto velocity = message.twist.twist.linear.x;
      if (!std::isfinite(velocity)) {
        throw std::invalid_argument("wheel longitudinal velocity must be finite");
      }
      const auto update = history_->add_wheel_sample(
        ros_stamp_seconds(message.header.stamp), velocity);
      if (update == MotionHistoryUpdate::kEpochReset) {
        reset_state("wheel history");
        history_->add_wheel_sample(ros_stamp_seconds(message.header.stamp), velocity);
      }
      process_pending();
    } catch (const std::exception & error) {
      warn_throttled(
        "wheel", "deskew rejected wheel sample: " + std::string(error.what()));
    }
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2 & message)
  {
    try {
      const auto stamp_ns = ros_stamp_nanoseconds(message.header.stamp);
      observe_stamp(last_cloud_stamp_, stamp_ns, "cloud");
      if (!valid_relative_frame(message.header.frame_id)) {
        throw std::invalid_argument("cloud frame must be a valid relative frame");
      }
      static_cast<void>(XyzirtCloudView(message));
      if (pending_.size() >= pending_depth_) {
        pending_.pop_front();
        warn_throttled(
          "pending_depth", "deskew dropped oldest scan because pending queue is full");
      }
      pending_.push_back({message, std::chrono::steady_clock::now(), "motion not ready"});
      process_pending();
    } catch (const std::exception & error) {
      warn_throttled(
        "cloud", "deskew rejected input cloud: " + std::string(error.what()));
    }
  }

  void process_pending()
  {
    while (!pending_.empty()) {
      auto & pending = pending_.front();
      const auto now = std::chrono::steady_clock::now();
      if (
        std::chrono::duration<double>(now - pending.received).count() >=
        pending_timeout_sec_)
      {
        warn_throttled(
          "pending_timeout",
          "deskew timed out without output: " + pending.last_reason);
        pending_.pop_front();
        continue;
      }
      try {
        const auto base_from_lidar = lookup_base_from_frame(
          pending.cloud.header.frame_id, pending.cloud.header.stamp);
        auto result = deskew_xyzirt_cloud(
          pending.cloud, *history_, base_from_lidar, options_);
        const auto action = pending_deskew_action(result);
        if (action == PendingDeskewAction::kRetry) {
          pending.last_reason = result.error;
          return;
        }
        if (action == PendingDeskewAction::kDrop) {
          warn_throttled(
            "permanent_scan", "deskew dropped invalid scan: " + result.error);
          pending_.pop_front();
          continue;
        }
        publisher_->publish(std::move(result.cloud.value()));
        pending_.pop_front();
      } catch (const tf2::TransformException & error) {
        pending.last_reason = "cloud transform unavailable: " + std::string(error.what());
        return;
      } catch (const std::exception & error) {
        warn_throttled(
          "permanent_scan", "deskew dropped invalid scan: " + std::string(error.what()));
        pending_.pop_front();
        continue;
      }
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string imu_topic_;
  std::string wheel_topic_;
  std::string base_frame_;
  double history_age_sec_;
  double pending_timeout_sec_;
  std::size_t pending_depth_{4U};
  MotionDeskewOptions options_;
  std::unique_ptr<MotionHistory> history_;
  std::deque<PendingScan> pending_;
  std::optional<std::int64_t> last_cloud_stamp_;
  std::optional<std::int64_t> last_imu_stamp_;
  std::optional<std::int64_t> last_wheel_stamp_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
  last_diagnostic_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
  wheel_subscription_;
  rclcpp::TimerBase::SharedPtr pending_timer_;
};

std::shared_ptr<rclcpp::Node> make_motion_deskew_node()
{
  return std::make_shared<MotionDeskewNode>();
}

}  // namespace ad_lidar_perception::preprocessing
