#include "ad_localization/imu_quaternion_encoder/imu_quaternion_encoder_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(ad_localization::make_imu_quaternion_encoder_node());
  rclcpp::shutdown();
  return 0;
}
