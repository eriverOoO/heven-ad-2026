#include "ad_lidar_perception/preprocessing/self_crop_filter.hpp"

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

class SelfCropFilterNode : public rclcpp::Node
{
public:
  SelfCropFilterNode()
  : Node("ad_self_crop_filter"),
    input_topic_(declare_parameter<std::string>(
        "topics.input", "/ad/perception/lidar/deskewed")),
    output_topic_(declare_parameter<std::string>(
        "topics.output", "/ad/perception/lidar/cropped")),
    base_frame_(declare_parameter<std::string>("base_frame", "base_link")),
    transform_timeout_sec_(declare_parameter<double>("transform_timeout_sec", 0.1)),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    bounds_.min_x_m = declare_parameter<double>("bounds.min_x_m", -0.990);
    bounds_.max_x_m = declare_parameter<double>("bounds.max_x_m", 4.045);
    bounds_.min_y_m = declare_parameter<double>("bounds.min_y_m", -1.145);
    bounds_.max_y_m = declare_parameter<double>("bounds.max_y_m", 1.145);
    bounds_.min_z_m = declare_parameter<double>("bounds.min_z_m", -0.200);
    bounds_.max_z_m = declare_parameter<double>("bounds.max_z_m", 1.805);
    if (input_topic_.empty() || output_topic_.empty()) {
      throw std::invalid_argument("self crop input and output topics must be nonempty");
    }
    if (
      get_node_topics_interface()->resolve_topic_name(input_topic_) ==
      get_node_topics_interface()->resolve_topic_name(output_topic_))
    {
      throw std::invalid_argument("self crop input and output topics must differ");
    }
    if (!valid_relative_frame(base_frame_)) {
      throw std::invalid_argument("self crop base_frame must be a valid relative frame");
    }
    if (!std::isfinite(transform_timeout_sec_) || transform_timeout_sec_ < 0.0) {
      throw std::invalid_argument(
              "self crop transform_timeout_sec must be finite and nonnegative");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    auto input_qos = rclcpp::SensorDataQoS();
    if (declare_parameter<bool>("input_reliable", false)) {
      input_qos.reliable();
    }
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, input_qos,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr input) {
        on_cloud(*input);
      });
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2 & input)
  {
    try {
      if (!valid_relative_frame(input.header.frame_id)) {
        throw std::invalid_argument("self crop input frame must be a valid relative frame");
      }
      std::optional<RigidTransform3> base_from_input;
      if (input.header.frame_id == base_frame_) {
        base_from_input = RigidTransform3{};
      } else {
        const auto transform = tf_buffer_.lookupTransform(
          base_frame_, input.header.frame_id, rclcpp::Time(input.header.stamp),
          rclcpp::Duration::from_seconds(transform_timeout_sec_));
        base_from_input = RigidTransform3{
          transform.transform.translation.x,
          transform.transform.translation.y,
          transform.transform.translation.z,
          transform.transform.rotation.x,
          transform.transform.rotation.y,
          transform.transform.rotation.z,
          transform.transform.rotation.w,
        };
      }
      auto result = crop_self_points(input, bounds_, base_from_input);
      publisher_->publish(std::move(result.cloud));
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "self crop has no timestamped base transform; cloud dropped: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "self crop rejected cloud without output: %s", error.what());
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string base_frame_;
  double transform_timeout_sec_;
  SelfCropBounds bounds_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

std::shared_ptr<rclcpp::Node> make_self_crop_filter_node()
{
  return std::make_shared<SelfCropFilterNode>();
}

}  // namespace ad_lidar_perception::preprocessing
