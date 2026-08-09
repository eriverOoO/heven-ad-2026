#include <rclcpp/rclcpp.hpp>

#include <memory>

namespace ad_lidar_perception::preprocessing
{
std::shared_ptr<rclcpp::Node> make_motion_deskew_node();
}  // namespace ad_lidar_perception::preprocessing

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(ad_lidar_perception::preprocessing::make_motion_deskew_node());
  rclcpp::shutdown();
  return 0;
}
