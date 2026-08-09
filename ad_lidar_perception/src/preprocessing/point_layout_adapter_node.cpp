#include "ad_lidar_perception/preprocessing/point_layout_adapter_node.hpp"

#include "ad_lidar_perception/preprocessing/point_layout_converter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ad_lidar_perception::preprocessing
{
namespace
{

std::uint8_t byte_parameter(
  rclcpp::Node & node, const std::string & name, const std::int64_t default_value)
{
  const auto value = node.declare_parameter<std::int64_t>(name, default_value);
  if (value < 0 || value > 255) {
    throw std::invalid_argument(name + " must be an integer in [0, 255]");
  }
  return static_cast<std::uint8_t>(value);
}

}  // namespace

class PointLayoutAdapterNode : public rclcpp::Node
{
public:
  PointLayoutAdapterNode()
  : Node("ad_point_layout_adapter"),
    input_topic_(
      declare_parameter<std::string>(
        "topics.input", "/ad/perception/lidar/cropped")),
    output_topic_(
      declare_parameter<std::string>(
        "topics.output", "/ad/perception/lidar/points_xyzirc"))
  {
    config_.intensity_scale = declare_parameter<double>("intensity.scale", 1.0);
    config_.intensity_offset = declare_parameter<double>("intensity.offset", 0.0);
    config_.nonfinite_intensity =
      byte_parameter(*this, "intensity.nonfinite_value", 0);
    config_.return_type = byte_parameter(*this, "return_type", 0);
    if (input_topic_.empty() || output_topic_.empty()) {
      throw std::invalid_argument("input and output topics must be nonempty");
    }
    const auto resolved_input_topic =
      get_node_topics_interface()->resolve_topic_name(input_topic_);
    const auto resolved_output_topic =
      get_node_topics_interface()->resolve_topic_name(output_topic_);
    if (resolved_input_topic == resolved_output_topic) {
      throw std::invalid_argument(
              "input and output topics must differ after ROS name resolution");
    }
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr input) {
        on_points(*input);
      });
  }

private:
  void on_points(const sensor_msgs::msg::PointCloud2 & input)
  {
    try {
      auto result = convert_morai_xyzirt_to_point_xyzirc(input, config_);
      publisher_->publish(std::move(result.cloud));
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cannot convert MORAI XYZIRT cloud: %s", error.what());
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  ConverterConfig config_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

std::shared_ptr<rclcpp::Node> make_point_layout_adapter_node()
{
  return std::make_shared<PointLayoutAdapterNode>();
}

}  // namespace ad_lidar_perception::preprocessing
