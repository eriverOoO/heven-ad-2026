// SPDX-License-Identifier: GPL-2.0-or-later
#include <rclcpp/rclcpp.hpp>

#include <fast_lio/fastlio_node.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(fast_lio::make_fastlio_node());
  rclcpp::shutdown();
  return 0;
}
