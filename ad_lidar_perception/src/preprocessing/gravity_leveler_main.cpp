#include <rclcpp/rclcpp.hpp>

#include <memory>

namespace ad_lidar_perception::preprocessing
{
std::shared_ptr<rclcpp::Node> make_gravity_leveler_node();
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = ad_lidar_perception::preprocessing::make_gravity_leveler_node();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
