#include "ad_lidar_perception/preprocessing/gravity_leveler.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_lidar_perception::preprocessing
{
namespace
{

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

GravityLevelingTransform from_message(
  const geometry_msgs::msg::TransformStamped & message)
{
  return GravityLevelingTransform{
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

}  // namespace

class GravityLevelerNode : public rclcpp::Node
{
public:
  GravityLevelerNode()
  : Node("ad_gravity_leveler"),
    input_topic_(declare_parameter<std::string>(
        "topics.input", "/ad/perception/lidar/cropped")),
    output_topic_(declare_parameter<std::string>(
        "topics.output", "/ad/perception/lidar/leveled")),
    odom_frame_(declare_parameter<std::string>("odom_frame", "odom")),
    base_frame_(declare_parameter<std::string>("base_frame", "base_link")),
    expected_input_frame_(declare_parameter<std::string>("expected_input_frame", "")),
    output_frame_(declare_parameter<std::string>("output_frame", "")),
    transform_timeout_sec_(declare_parameter<double>("transform_timeout_sec", 0.1)),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_),
    tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this))
  {
    if (input_topic_.empty() || output_topic_.empty()) {
      throw std::invalid_argument("gravity leveler input and output topics must be nonempty");
    }
    if (
      get_node_topics_interface()->resolve_topic_name(input_topic_) ==
      get_node_topics_interface()->resolve_topic_name(output_topic_))
    {
      throw std::invalid_argument("gravity leveler input and output topics must differ");
    }
    if (!valid_relative_frame(odom_frame_) || !valid_relative_frame(base_frame_)) {
      throw std::invalid_argument(
              "gravity leveler odom_frame and base_frame must be valid relative frames");
    }
    if (
      (!expected_input_frame_.empty() && !valid_relative_frame(expected_input_frame_)) ||
      (!output_frame_.empty() && !valid_relative_frame(output_frame_)))
    {
      throw std::invalid_argument(
              "gravity leveler expected/output frames must be empty or valid relative frames");
    }
    if (
      !expected_input_frame_.empty() && !output_frame_.empty() &&
      output_frame_ != derive_leveled_frame(expected_input_frame_))
    {
      throw std::invalid_argument(
              "gravity leveler configured output frame does not match expected input frame");
    }
    if (!std::isfinite(transform_timeout_sec_) || transform_timeout_sec_ < 0.0) {
      throw std::invalid_argument(
              "gravity leveler transform_timeout_sec must be finite and nonnegative");
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
  void on_cloud(const sensor_msgs::msg::PointCloud2 & input)
  {
    try {
      if (!valid_relative_frame(input.header.frame_id)) {
        throw std::invalid_argument(
                "gravity leveler input frame must be a valid relative frame");
      }
      if (
        !expected_input_frame_.empty() &&
        input.header.frame_id != expected_input_frame_)
      {
        throw std::invalid_argument(
                "gravity leveler input frame does not match selected sensor profile");
      }
      const auto derived_output_frame = derive_leveled_frame(input.header.frame_id);
      const auto actual_output_frame =
        output_frame_.empty() ? derived_output_frame : output_frame_;
      if (actual_output_frame != derived_output_frame) {
        throw std::invalid_argument(
                "gravity leveler output frame does not match actual input frame");
      }

      const auto odom_from_base_message = tf_buffer_.lookupTransform(
        odom_frame_, base_frame_, rclcpp::Time(input.header.stamp),
        rclcpp::Duration::from_seconds(transform_timeout_sec_));
      const auto base_from_lidar_message = tf_buffer_.lookupTransform(
        base_frame_, input.header.frame_id, rclcpp::Time(input.header.stamp),
        rclcpp::Duration::from_seconds(transform_timeout_sec_));
      auto result = level_xyzirt_cloud(
        input, from_message(odom_from_base_message),
        from_message(base_from_lidar_message), actual_output_frame);

      geometry_msgs::msg::TransformStamped output_transform;
      output_transform.header.stamp = input.header.stamp;
      output_transform.header.frame_id = base_frame_;
      output_transform.child_frame_id = actual_output_frame;
      output_transform.transform.translation.x = result.base_from_level.translation[0];
      output_transform.transform.translation.y = result.base_from_level.translation[1];
      output_transform.transform.translation.z = result.base_from_level.translation[2];
      output_transform.transform.rotation.x = result.base_from_level.quaternion_xyzw[0];
      output_transform.transform.rotation.y = result.base_from_level.quaternion_xyzw[1];
      output_transform.transform.rotation.z = result.base_from_level.quaternion_xyzw[2];
      output_transform.transform.rotation.w = result.base_from_level.quaternion_xyzw[3];
      tf_broadcaster_->sendTransform(output_transform);
      publisher_->publish(std::move(result.cloud));
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "gravity leveler has no complete timestamped TF; cloud dropped: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "gravity leveler rejected cloud; cloud dropped: %s", error.what());
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string expected_input_frame_;
  std::string output_frame_;
  double transform_timeout_sec_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

std::shared_ptr<rclcpp::Node> make_gravity_leveler_node()
{
  return std::make_shared<GravityLevelerNode>();
}

}  // namespace ad_lidar_perception::preprocessing
