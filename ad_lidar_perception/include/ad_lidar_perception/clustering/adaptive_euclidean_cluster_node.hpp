#pragma once

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <memory>
#include <string>

namespace ad_lidar_perception::clustering
{

class AdaptiveEuclideanClusterNode : public rclcpp::Node
{
public:
  explicit AdaptiveEuclideanClusterNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~AdaptiveEuclideanClusterNode() override;

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr message);

  std::string input_topic_;
  std::string objects_topic_;
  std::string clusters_topic_;
  double min_x_m_{-4.0};
  double max_x_m_{100.0};
  double min_y_m_{-25.0};
  double max_y_m_{25.0};
  double min_z_m_{-1.0};
  double max_z_m_{3.0};
  double maximum_dynamic_object_diagonal_m_{12.0};

  class Impl;
  std::unique_ptr<Impl> impl_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr
    objects_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    clusters_publisher_;
};

} // namespace ad_lidar_perception::clustering
