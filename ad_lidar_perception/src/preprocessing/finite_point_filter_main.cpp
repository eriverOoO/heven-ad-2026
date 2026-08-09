#include "ad_lidar_perception/preprocessing/finite_point_filter_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node =
    ad_lidar_perception::preprocessing::make_finite_point_filter_node();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
