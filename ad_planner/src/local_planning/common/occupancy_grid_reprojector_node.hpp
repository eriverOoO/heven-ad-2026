#ifndef AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_NODE_HPP_
#define AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_NODE_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "ad_planner/local_planning/common/occupancy_grid_reprojector.hpp"

namespace ad_planner
{

class OccupancyGridReprojectorNode final : public rclcpp::Node
{
public:
  explicit OccupancyGridReprojectorNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_grid(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr message);
  void warn_drop(const std::string & reason);

  GridReprojectionConfig config_;
  std::string target_frame_;
  std::string ego_frame_;
  rclcpp::Duration transform_timeout_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::mutex callback_mutex_;
  // Session-scoped: a looping bag or backward ROS clock jump requires restart.
  std::optional<std::int64_t> last_successfully_published_stamp_ns_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__OCCUPANCY_GRID_REPROJECTOR_NODE_HPP_
