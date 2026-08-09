#include "ad_lidar_perception/preprocessing/point_layout_adapter_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    ad_lidar_perception::preprocessing::make_point_layout_adapter_node());
  rclcpp::shutdown();
  return 0;
}
