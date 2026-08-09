#include "ad_lidar_perception/occupancy_grid/occupancy_grid_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(ad_lidar_perception::occupancy_grid::make_occupancy_grid_node());
  rclcpp::shutdown();
  return 0;
}
