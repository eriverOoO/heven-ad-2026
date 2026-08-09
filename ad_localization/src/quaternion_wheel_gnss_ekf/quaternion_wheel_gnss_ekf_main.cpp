#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(ad_localization::make_quaternion_wheel_gnss_ekf_node());
  rclcpp::shutdown();
  return 0;
}
