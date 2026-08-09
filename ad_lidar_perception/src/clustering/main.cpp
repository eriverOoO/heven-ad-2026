#include <ad_lidar_perception/clustering/adaptive_euclidean_cluster_node.hpp>

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<
      ad_lidar_perception::clustering::AdaptiveEuclideanClusterNode>());
  rclcpp::shutdown();
  return 0;
}
