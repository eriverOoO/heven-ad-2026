#include "ad_lidar_perception/preprocessing/finite_point_filter.hpp"
#include "ad_lidar_perception/preprocessing/finite_point_filter_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace ad_lidar_perception::preprocessing
{

class FinitePointFilterNode final : public rclcpp::Node
{
public:
  FinitePointFilterNode()
  : Node("ad_finite_point_filter")
  {
    const auto input_topic = declare_parameter<std::string>(
      "topics.input", "/ad/perception/lidar/nonground");
    const auto output_topic = declare_parameter<std::string>(
      "topics.output", "/ad/perception/lidar/nonground_finite");
    if (input_topic.empty() || output_topic.empty()) {
      throw std::invalid_argument("finite-point-filter topics must be nonempty");
    }
    if (
      get_node_topics_interface()->resolve_topic_name(input_topic) ==
      get_node_topics_interface()->resolve_topic_name(output_topic))
    {
      throw std::invalid_argument(
              "finite-point-filter input and output topics must differ");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr input) {
        try {
          auto result = filter_finite_xyz(*input);
          if (result.stats.removed_nonfinite > 0U) {
            RCLCPP_DEBUG(
              get_logger(), "removed %zu nonfinite points from %zu",
              result.stats.removed_nonfinite, result.stats.input_points);
          }
          publisher_->publish(std::move(result.cloud));
        } catch (const std::exception & error) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "cannot sanitize PointCloud2: %s", error.what());
        }
      });
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

std::shared_ptr<rclcpp::Node> make_finite_point_filter_node()
{
  return std::make_shared<FinitePointFilterNode>();
}

}  // namespace ad_lidar_perception::preprocessing
