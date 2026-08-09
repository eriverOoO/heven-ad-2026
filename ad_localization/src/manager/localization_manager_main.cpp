#include "ad_localization/manager/localization_manager_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(ad_localization::make_localization_manager_node());
  rclcpp::shutdown();
  return 0;
}
