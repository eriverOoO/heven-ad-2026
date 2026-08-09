#include "ad_lidar_perception/occupancy_grid/combined_occupancy_grid_node.hpp"

#include "ad_lidar_perception/occupancy_grid/exact_stamp_pairer.hpp"
#include "ad_lidar_perception/occupancy_grid/grid_combiner.hpp"

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_lidar_perception::occupancy_grid
{
namespace
{

std::int64_t positive_stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.nanosec >= 1'000'000'000U ||
    stamp.sec < 0 ||
    (stamp.sec == 0 && stamp.nanosec == 0U))
  {
    throw std::invalid_argument("occupancy-grid stamp must be positive");
  }
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

GridLayerMetadata validate_grid(const nav_msgs::msg::OccupancyGrid & grid)
{
  if (grid.header.frame_id.empty()) {
    throw std::invalid_argument("occupancy-grid frame must be nonempty");
  }
  if (!std::isfinite(grid.info.resolution) || grid.info.resolution <= 0.0F ||
    grid.info.width == 0U || grid.info.height == 0U ||
    grid.info.width >
    std::numeric_limits<std::size_t>::max() / grid.info.height)
  {
    throw std::invalid_argument("occupancy-grid dimensions are invalid");
  }
  const std::size_t cell_count =
    static_cast<std::size_t>(grid.info.width) * grid.info.height;
  if (grid.data.size() != cell_count) {
    throw std::invalid_argument("occupancy-grid data size is invalid");
  }
  if (!std::isfinite(grid.info.origin.position.x) ||
    !std::isfinite(grid.info.origin.position.y) ||
    !std::isfinite(grid.info.origin.position.z) ||
    !std::isfinite(grid.info.origin.orientation.x) ||
    !std::isfinite(grid.info.origin.orientation.y) ||
    !std::isfinite(grid.info.origin.orientation.z) ||
    !std::isfinite(grid.info.origin.orientation.w) ||
    grid.info.origin.position.z != 0.0 ||
    grid.info.origin.orientation.x != 0.0 ||
    grid.info.origin.orientation.y != 0.0 ||
    grid.info.origin.orientation.z != 0.0 ||
    grid.info.origin.orientation.w != 1.0)
  {
    throw std::invalid_argument(
            "occupancy-grid origin must have zero z and identity orientation");
  }
  for (const auto cell : grid.data) {
    if (cell < -1 || cell > 100) {
      throw std::invalid_argument("occupancy-grid cost is outside [-1, 100]");
    }
  }
  return GridLayerMetadata{
    GridGeometry{
      grid.info.origin.position.x,
      grid.info.origin.position.y,
      static_cast<double>(grid.info.resolution),
      grid.info.width,
      grid.info.height},
    grid.header.frame_id,
    positive_stamp_ns(grid.header.stamp)};
}

struct ValidatedGrid
{
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr message;
  GridLayerMetadata metadata;
};

using GridPairer = ExactStampPairer<ValidatedGrid, ValidatedGrid>;

}  // namespace

class CombinedOccupancyGridNode : public rclcpp::Node
{
public:
  CombinedOccupancyGridNode()
  : Node("ad_combined_occupancy_grid")
  {
    const auto maximum_pending_messages =
      declare_parameter<int>("maximum_pending_messages", 8);
    if (maximum_pending_messages <= 0) {
      throw std::invalid_argument(
              "maximum_pending_messages must be positive");
    }
    pairer_ = std::make_unique<GridPairer>(
      static_cast<std::size_t>(maximum_pending_messages));
    const auto static_topic = declare_parameter<std::string>(
      "topics.static_grid", "/ad/perception/occupancy/static");
    const auto dynamic_topic = declare_parameter<std::string>(
      "topics.dynamic_grid", "/ad/perception/occupancy/dynamic");
    const auto combined_topic = declare_parameter<std::string>(
      "topics.combined_grid", "/ad/perception/occupancy/combined");
    const auto compatibility_topic = declare_parameter<std::string>(
      "topics.compatibility_grid", "/ad/perception/occupancy_grid");
    if (static_topic.empty() || dynamic_topic.empty() ||
      combined_topic.empty() || compatibility_topic.empty())
    {
      throw std::invalid_argument("occupancy-layer topics must be nonempty");
    }
    const auto resolve = [this](const std::string & topic) {
        return get_node_topics_interface()->resolve_topic_name(topic);
      };
    if (resolve(static_topic) == resolve(dynamic_topic) ||
      resolve(combined_topic) == resolve(compatibility_topic) ||
      resolve(static_topic) == resolve(combined_topic) ||
      resolve(static_topic) == resolve(compatibility_topic) ||
      resolve(dynamic_topic) == resolve(combined_topic) ||
      resolve(dynamic_topic) == resolve(compatibility_topic))
    {
      throw std::invalid_argument(
              "occupancy-layer topics must resolve to distinct endpoints");
    }

    combined_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      combined_topic, rclcpp::SensorDataQoS());
    compatibility_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      compatibility_topic, rclcpp::SensorDataQoS());
    static_subscription_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      static_topic, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr input) {
        on_static_grid(std::move(input));
      });
    dynamic_subscription_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      dynamic_topic, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr input) {
        on_dynamic_grid(std::move(input));
      });
  }

private:
  void publish_pair(std::optional<GridPairer::Pair> pair)
  {
    if (!pair.has_value()) {
      return;
    }
    if (!layers_are_compatible(
        pair->left.metadata, pair->right.metadata))
    {
      throw std::invalid_argument(
              "exact-stamp occupancy layers have incompatible geometry "
              "or frame");
    }
    nav_msgs::msg::OccupancyGrid result = *pair->left.message;
    result.data = combine_cost_layers(
      pair->left.message->data, pair->right.message->data);
    combined_publisher_->publish(result);
    compatibility_publisher_->publish(result);
  }

  void on_dynamic_grid(nav_msgs::msg::OccupancyGrid::ConstSharedPtr input)
  {
    try {
      auto metadata = validate_grid(*input);
      const auto stamp_ns = metadata.stamp_ns;
      std::lock_guard<std::mutex> lock(callback_mutex_);
      publish_pair(
        pairer_->add_right(
          stamp_ns, ValidatedGrid{std::move(input), std::move(metadata)}));
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "not publishing invalid dynamic occupancy pair: %s", error.what());
    }
  }

  void on_static_grid(nav_msgs::msg::OccupancyGrid::ConstSharedPtr input)
  {
    try {
      auto metadata = validate_grid(*input);
      const auto stamp_ns = metadata.stamp_ns;
      std::lock_guard<std::mutex> lock(callback_mutex_);
      publish_pair(
        pairer_->add_left(
          stamp_ns, ValidatedGrid{std::move(input), std::move(metadata)}));
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "not publishing invalid static occupancy pair: %s", error.what());
    }
  }

  std::mutex callback_mutex_;
  std::unique_ptr<GridPairer> pairer_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr combined_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
    compatibility_publisher_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
    static_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
    dynamic_subscription_;
};

std::shared_ptr<rclcpp::Node> make_combined_occupancy_grid_node()
{
  return std::make_shared<CombinedOccupancyGridNode>();
}

}  // namespace ad_lidar_perception::occupancy_grid
