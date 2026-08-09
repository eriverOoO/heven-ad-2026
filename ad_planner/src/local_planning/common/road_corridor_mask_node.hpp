#ifndef AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_MASK_NODE_HPP_
#define AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_MASK_NODE_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "ad_planner/io/route_corridor_loader.hpp"
#include "ad_planner/local_planning/common/road_corridor_grid.hpp"

namespace ad_planner
{

class RoadCorridorMaskNode final : public rclcpp::Node
{
public:
  explicit RoadCorridorMaskNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void on_trigger(const std_msgs::msg::Header & header);
  void warn_drop(const std::string & reason);
  bool already_published(std::int64_t stamp_ns) const;
  void remember_published(std::int64_t stamp_ns);

  LoadedRouteCorridor route_corridor_;
  std::optional<PreparedRoadCorridor> prepared_route_corridor_;
  RoadCorridorGridWindow window_;
  std::string base_frame_;
  rclcpp::Duration transform_timeout_;
  std::size_t deduplication_cache_size_{0U};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    lidar_subscription_;
  rclcpp::Subscription<ad_interfaces::msg::PredictedObjectArray>::SharedPtr
    predicted_objects_subscription_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::deque<std::int64_t> published_stamps_ns_;
  std::optional<std::int64_t> latest_published_stamp_ns_;
  std::mutex callback_mutex_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__ROAD_CORRIDOR_MASK_NODE_HPP_
