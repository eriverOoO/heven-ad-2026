#include "highway_merge_gap_response_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<ad_planner::HighwayMergeGapResponseNode>(rclcpp::NodeOptions{}));
  rclcpp::shutdown();
  return 0;
}
