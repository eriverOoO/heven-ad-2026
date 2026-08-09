#include "ad_lidar_perception/occupancy_grid/grid_builder.hpp"
#include "ad_lidar_perception/occupancy_grid/exact_stamp_pairer.hpp"
#include "ad_lidar_perception/occupancy_grid/occupancy_grid_node.hpp"

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad_lidar_perception::occupancy_grid
{
namespace
{

bool positive_stamp(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.nanosec < 1'000'000'000U &&
         (stamp.sec > 0 || (stamp.sec == 0 && stamp.nanosec > 0U));
}

std::int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  if (!positive_stamp(stamp)) {
    throw std::invalid_argument("message stamp must be positive");
  }
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

void validate_cloud_structure(const sensor_msgs::msg::PointCloud2 & cloud)
{
  if (cloud.height == 0U || cloud.width == 0U ||
    cloud.point_step == 0U || cloud.data.empty() ||
    cloud.is_bigendian ||
    cloud.width > std::numeric_limits<std::size_t>::max() / cloud.point_step)
  {
    throw std::invalid_argument(
            "unsupported empty, big-endian, or malformed PointCloud2");
  }
  const std::size_t minimum_row_size =
    static_cast<std::size_t>(cloud.width) * cloud.point_step;
  if (cloud.row_step != minimum_row_size) {
    throw std::invalid_argument(
            "PointCloud2 row padding is unsupported");
  }
  if (cloud.height >
    std::numeric_limits<std::size_t>::max() / cloud.row_step ||
    cloud.data.size() != static_cast<std::size_t>(cloud.height) * cloud.row_step)
  {
    throw std::invalid_argument("malformed PointCloud2 row or data size");
  }

  std::array<bool, 3U> found{false, false, false};
  constexpr std::array<const char *, 3U> names{"x", "y", "z"};
  for (const auto & field : cloud.fields) {
    for (std::size_t index = 0U; index < names.size(); ++index) {
      if (field.name != names[index]) {
        continue;
      }
      if (found[index] ||
        field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        field.count != 1U ||
        field.offset > cloud.point_step ||
        cloud.point_step - field.offset < sizeof(float))
      {
        throw std::invalid_argument("malformed PointCloud2 XYZ field");
      }
      found[index] = true;
    }
  }
  if (!found[0] || !found[1] || !found[2]) {
    throw std::invalid_argument("PointCloud2 is missing an XYZ field");
  }
}

std::int64_t duration_nanoseconds(double duration_sec)
{
  if (!std::isfinite(duration_sec) || duration_sec < 0.0 ||
    duration_sec >
    static_cast<double>(std::numeric_limits<std::int64_t>::max()) / 1.0e9)
  {
    throw std::invalid_argument(
            "persistence.duration_sec must be finite and nonnegative");
  }
  return static_cast<std::int64_t>(std::llround(duration_sec * 1.0e9));
}

void append_points(
  const sensor_msgs::msg::PointCloud2 & cloud,
  std::vector<Point3> & points)
{
  validate_cloud_structure(cloud);
  const std::size_t count =
    static_cast<std::size_t>(cloud.width) * cloud.height;
  if (count > points.max_size() - points.size()) {
    throw std::length_error("temporal point-cloud history is too large");
  }
  points.reserve(points.size() + count);
  sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> z(cloud, "z");
  for (; x != x.end(); ++x, ++y, ++z) {
    points.push_back(Point3{*x, *y, *z});
  }
}

bool inside_ego_clearance(const Point3 & point, const GridConfig & config)
{
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         point.x >= config.ego_clear_x_min &&
         point.x <= config.ego_clear_x_max &&
         point.y >= config.ego_clear_y_min &&
         point.y <= config.ego_clear_y_max;
}

void append_transformed_points(
  const std::vector<Point3> & input,
  const geometry_msgs::msg::TransformStamped & transform,
  std::vector<Point3> & output)
{
  if (input.size() > output.max_size() - output.size()) {
    throw std::length_error("temporal point-cloud history is too large");
  }
  output.reserve(output.size() + input.size());
  tf2::Transform target_from_source;
  tf2::fromMsg(transform.transform, target_from_source);
  for (const auto & point : input) {
    const auto transformed =
      target_from_source * tf2::Vector3(point.x, point.y, point.z);
    output.push_back(
      Point3{transformed.x(), transformed.y(), transformed.z()});
  }
}

}  // namespace

class AdLidarPerceptionNode : public rclcpp::Node
{
public:
  AdLidarPerceptionNode()
  : Node("ad_lidar_perception"),
    target_frame_(declare_parameter<std::string>("target_frame", "base_link")),
    transform_timeout_sec_(
      declare_parameter<double>("transform_timeout_sec", 0.05)),
    persistence_duration_ns_(duration_nanoseconds(
        declare_parameter<double>("persistence.duration_sec", 0.0))),
    persistence_fixed_frame_(
      declare_parameter<std::string>("persistence.fixed_frame", "odom")),
    persistence_maximum_clouds_(static_cast<std::size_t>(
        declare_parameter<int>("persistence.maximum_clouds", 8))),
    road_gate_enabled_(
      declare_parameter<bool>("road_gate.enabled", false)),
    road_gate_maximum_pending_messages_(
      declare_parameter<int>("road_gate.maximum_pending_messages", 8)),
    builder_(GridConfig{
      declare_parameter<double>("x_min", -4.0),
      declare_parameter<double>("x_max", 100.0),
      declare_parameter<double>("y_min", -10.0),
      declare_parameter<double>("y_max", 10.0),
      declare_parameter<double>("z_min", 0.1),
      declare_parameter<double>("z_max", 2.0),
      declare_parameter<double>("resolution", 0.1),
      declare_parameter<double>("inflation_radius_m", 1.8),
      declare_parameter<double>("inflation_cost_scaling_factor", 2.0),
      declare_parameter<double>("ego_clear_x_min", -1.0),
      declare_parameter<double>("ego_clear_x_max", 4.05),
      declare_parameter<double>("ego_clear_y_min", -1.15),
      declare_parameter<double>("ego_clear_y_max", 1.15)}),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    const auto points_topic = declare_parameter<std::string>(
      "topics.points", "/ad/sensors/lidar/points");
    const auto drivable_mask_topic = declare_parameter<std::string>(
      "topics.drivable_mask", "/ad/planning/drivable_mask");
    const auto grid_topic = declare_parameter<std::string>(
      "topics.occupancy_grid", "/ad/perception/occupancy/static");
    const auto static_ungated_topic = declare_parameter<std::string>(
      "visualization.topics.static_ungated",
      "/ad/viz/perception/occupancy/static_ungated");
    if (target_frame_.empty()) {
      throw std::invalid_argument("target_frame must be nonempty");
    }
    if (!std::isfinite(transform_timeout_sec_) ||
      transform_timeout_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "transform_timeout_sec must be finite and positive");
    }
    if (persistence_duration_ns_ > 0 &&
      (persistence_fixed_frame_.empty() ||
      persistence_maximum_clouds_ == 0U ||
      persistence_maximum_clouds_ >
      static_cast<std::size_t>(std::numeric_limits<int>::max())))
    {
      throw std::invalid_argument(
              "enabled persistence requires a fixed frame and positive "
              "maximum_clouds");
    }
    if (road_gate_maximum_pending_messages_ <= 0 ||
      road_gate_maximum_pending_messages_ > 1024)
    {
      throw std::invalid_argument(
              "road_gate.maximum_pending_messages must be in [1, 1024]");
    }
    if (road_gate_enabled_ && drivable_mask_topic.empty()) {
      throw std::invalid_argument(
              "enabled road gate requires topics.drivable_mask");
    }
    if (static_ungated_topic.empty()) {
      throw std::invalid_argument(
              "visualization.topics.static_ungated must be nonempty");
    }

    publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      grid_topic, rclcpp::SensorDataQoS());
    static_ungated_publisher_ =
      create_publisher<nav_msgs::msg::OccupancyGrid>(
      static_ungated_topic, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        on_points(std::move(message));
      });
    if (road_gate_enabled_) {
      road_gate_pairer_ = std::make_unique<RoadGatePairer>(
        static_cast<std::size_t>(road_gate_maximum_pending_messages_));
      drivable_mask_subscription_ =
        create_subscription<nav_msgs::msg::OccupancyGrid>(
        drivable_mask_topic, rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
          on_drivable_mask(std::move(message));
        });
    }
  }

private:
  using RoadGatePairer = ExactStampPairer<
    sensor_msgs::msg::PointCloud2::ConstSharedPtr, DrivableMask>;

  struct StampedCloud
  {
    std::int64_t stamp_ns{0};
    std::vector<Point3> fixed_points;
  };

  sensor_msgs::msg::PointCloud2 transform_cloud(
    const sensor_msgs::msg::PointCloud2 & input,
    const std::string & target_frame,
    const rclcpp::Time & stamp)
  {
    if (input.header.frame_id == target_frame) {
      return input;
    }
    const auto transform = tf_buffer_.lookupTransform(
      target_frame, input.header.frame_id, stamp,
      rclcpp::Duration::from_seconds(transform_timeout_sec_));
    sensor_msgs::msg::PointCloud2 output;
    tf2::doTransform(input, output, transform);
    return output;
  }

  std::vector<Point3> current_points(
    const sensor_msgs::msg::PointCloud2 & input)
  {
    const auto stamp = rclcpp::Time(input.header.stamp);
    if (persistence_duration_ns_ == 0) {
      std::vector<Point3> points;
      append_points(transform_cloud(input, target_frame_, stamp), points);
      return points;
    }

    const std::int64_t stamp_ns = stamp.nanoseconds();
    if (!history_.empty() && stamp_ns <= history_.back().stamp_ns) {
      // MORAI time may jump backwards during a reset. Never mix observations
      // from different simulator timelines.
      history_.clear();
    }
    std::vector<Point3> current;
    append_points(transform_cloud(input, target_frame_, stamp), current);
    const auto & grid_config = builder_.config();
    current.erase(
      std::remove_if(
        current.begin(), current.end(),
        [&grid_config](const Point3 & point) {
          return inside_ego_clearance(point, grid_config);
        }),
      current.end());
    std::vector<Point3> fixed;
    append_transformed_points(
      current,
      tf_buffer_.lookupTransform(
        persistence_fixed_frame_, target_frame_, stamp,
        rclcpp::Duration::from_seconds(transform_timeout_sec_)),
      fixed);
    history_.push_back(StampedCloud{stamp_ns, std::move(fixed)});
    while (!history_.empty() &&
      (stamp_ns - history_.front().stamp_ns > persistence_duration_ns_ ||
      history_.size() > persistence_maximum_clouds_))
    {
      history_.pop_front();
    }

    const auto fixed_to_target = tf_buffer_.lookupTransform(
      target_frame_, persistence_fixed_frame_, stamp,
      rclcpp::Duration::from_seconds(transform_timeout_sec_));
    std::vector<Point3> points;
    for (const auto & stored : history_) {
      append_transformed_points(stored.fixed_points, fixed_to_target, points);
    }
    return points;
  }

  DrivableMask convert_drivable_mask(
    const nav_msgs::msg::OccupancyGrid & input) const
  {
    (void)stamp_nanoseconds(input.header.stamp);
    if (input.header.frame_id != target_frame_) {
      throw std::invalid_argument(
              "drivable-mask frame must match target_frame");
    }
    const auto & origin = input.info.origin;
    if (!std::isfinite(origin.position.x) ||
      !std::isfinite(origin.position.y) ||
      !std::isfinite(origin.position.z) ||
      !std::isfinite(origin.orientation.x) ||
      !std::isfinite(origin.orientation.y) ||
      !std::isfinite(origin.orientation.z) ||
      !std::isfinite(origin.orientation.w) ||
      origin.position.x != builder_.config().x_min ||
      origin.position.y != builder_.config().y_min ||
      origin.position.z != 0.0 ||
      origin.orientation.x != 0.0 ||
      origin.orientation.y != 0.0 ||
      origin.orientation.z != 0.0 ||
      origin.orientation.w != 1.0 ||
      static_cast<double>(input.info.resolution) !=
      builder_.config().resolution ||
      static_cast<std::size_t>(input.info.width) != builder_.width() ||
      static_cast<std::size_t>(input.info.height) != builder_.height() ||
      input.data.size() != builder_.width() * builder_.height())
    {
      throw std::invalid_argument(
              "drivable-mask geometry must exactly match occupancy grid");
    }
    if (std::any_of(
        input.data.begin(), input.data.end(),
        [](const std::int8_t value) {
          return value < -1 || value > 100;
        }))
    {
      throw std::invalid_argument("drivable mask has an invalid cell value");
    }
    return DrivableMask{
      origin.position.x,
      origin.position.y,
      static_cast<double>(input.info.resolution),
      static_cast<std::size_t>(input.info.width),
      static_cast<std::size_t>(input.info.height),
      input.data};
  }

  void process_pair(std::optional<RoadGatePairer::Pair> pair)
  {
    if (!pair.has_value()) {
      return;
    }
    const auto points = current_points(*pair->left);
    const auto & drivable_mask = pair->right;
    // Both layers are derived from the same transformed point set and stamp.
    // The ungated layer is visualization-only; the safety/planning layer is
    // still published only after an exact-stamp drivable mask is available.
    auto ungated = builder_.build(points);
    auto gated = builder_.build(points, drivable_mask);
    publish_grid(
      pair->left->header.stamp, std::move(ungated),
      *static_ungated_publisher_);
    publish_grid(
      pair->left->header.stamp, std::move(gated), *publisher_);
  }

  void on_points(sensor_msgs::msg::PointCloud2::ConstSharedPtr input)
  {
    try {
      if (input->header.frame_id.empty()) {
        throw std::invalid_argument("point cloud frame must be nonempty");
      }
      const auto stamp_ns = stamp_nanoseconds(input->header.stamp);
      validate_cloud_structure(*input);
      if (road_gate_enabled_) {
        process_pair(road_gate_pairer_->add_left(stamp_ns, std::move(input)));
        return;
      }
      const auto points = current_points(*input);
      auto ungated = builder_.build(points);
      auto planning = ungated;
      publish_grid(
        input->header.stamp, std::move(ungated),
        *static_ungated_publisher_);
      publish_grid(
        input->header.stamp, std::move(planning), *publisher_);
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "cannot build occupancy grid: %s", error.what());
    }
  }

  void on_drivable_mask(
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr input)
  {
    try {
      const auto stamp_ns = stamp_nanoseconds(input->header.stamp);
      auto mask = convert_drivable_mask(*input);
      process_pair(
        road_gate_pairer_->add_right(stamp_ns, std::move(mask)));
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cannot apply drivable mask: %s", error.what());
    }
  }

  void publish_grid(
    const builtin_interfaces::msg::Time & stamp,
    std::vector<std::int8_t> data,
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid> & publisher)
  {
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = stamp;
    grid.header.frame_id = target_frame_;
    grid.info.resolution = static_cast<float>(builder_.config().resolution);
    grid.info.width = static_cast<std::uint32_t>(builder_.width());
    grid.info.height = static_cast<std::uint32_t>(builder_.height());
    grid.info.origin.position.x = builder_.config().x_min;
    grid.info.origin.position.y = builder_.config().y_min;
    grid.info.origin.orientation.w = 1.0;
    grid.data = std::move(data);
    publisher.publish(grid);
  }

  std::string target_frame_;
  double transform_timeout_sec_;
  std::int64_t persistence_duration_ns_;
  std::string persistence_fixed_frame_;
  std::size_t persistence_maximum_clouds_;
  bool road_gate_enabled_;
  int road_gate_maximum_pending_messages_;
  std::deque<StampedCloud> history_;
  GridBuilder builder_;
  std::unique_ptr<RoadGatePairer> road_gate_pairer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
    static_ungated_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
    drivable_mask_subscription_;
};

std::shared_ptr<rclcpp::Node> make_occupancy_grid_node()
{
  return std::make_shared<AdLidarPerceptionNode>();
}

}  // namespace ad_lidar_perception::occupancy_grid
