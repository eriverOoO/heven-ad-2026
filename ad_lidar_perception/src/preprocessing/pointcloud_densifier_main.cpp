#include <rclcpp/rclcpp.hpp>

#include <memory>

namespace ad_lidar_perception::preprocessing
{
std::shared_ptr<rclcpp::Node> make_pointcloud_densifier_node();
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = ad_lidar_perception::preprocessing::make_pointcloud_densifier_node();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
