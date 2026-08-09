#include "road_corridor_mask_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>

#include "ad_planner/local_planning/common/occupancy_grid_reprojector.hpp"

namespace ad_planner
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr RoadCorridorGridWindow kRequiredWindow{
  -4.0, 100.0, -10.0, 10.0, 0.1};

std::int64_t checked_stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
    (stamp.sec == 0 && stamp.nanosec == 0U))
  {
    throw std::invalid_argument(
            "trigger header stamp must be positive and normalized");
  }
  const std::int64_t seconds = static_cast<std::int64_t>(stamp.sec);
  if (seconds >
    (std::numeric_limits<std::int64_t>::max() -
    static_cast<std::int64_t>(stamp.nanosec)) / kNanosecondsPerSecond)
  {
    throw std::invalid_argument("trigger header stamp overflows nanoseconds");
  }
  return seconds * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

rclcpp::Duration checked_timeout(const double timeout_sec)
{
  if (!std::isfinite(timeout_sec) || !(timeout_sec > 0.0)) {
    throw std::invalid_argument(
            "transform_timeout_sec must be finite and positive");
  }
  const long double nanoseconds =
    static_cast<long double>(timeout_sec) *
    static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(nanoseconds) || nanoseconds < 1.0L ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument(
            "transform_timeout_sec is not representable");
  }
  return rclcpp::Duration::from_nanoseconds(
    static_cast<std::int64_t>(nanoseconds));
}

std::string resolve_data_dir(const std::string & parameter)
{
  if (!parameter.empty()) {
    return parameter;
  }
  const char * const environment = std::getenv("AD_DATA_DIR");
  if (environment && *environment != '\0') {
    return environment;
  }
  throw std::invalid_argument("set data_dir or AD_DATA_DIR");
}

bool finite_translation(const geometry_msgs::msg::Vector3 & translation)
{
  return std::isfinite(translation.x) &&
         std::isfinite(translation.y) &&
         std::isfinite(translation.z);
}

QuaternionComponents components(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  return QuaternionComponents{
    quaternion.x, quaternion.y, quaternion.z, quaternion.w};
}

void require_required_window(const RoadCorridorGridWindow & window)
{
  if (window.minimum_x_m != kRequiredWindow.minimum_x_m ||
    window.maximum_x_m != kRequiredWindow.maximum_x_m ||
    window.minimum_y_m != kRequiredWindow.minimum_y_m ||
    window.maximum_y_m != kRequiredWindow.maximum_y_m ||
    window.resolution_m != kRequiredWindow.resolution_m)
  {
    throw std::invalid_argument(
            "grid parameters must remain x=[-4,100], y=[-10,10], "
            "resolution=0.1");
  }
}

}  // namespace

RoadCorridorMaskNode::RoadCorridorMaskNode(
  const rclcpp::NodeOptions & options)
: Node("ad_road_corridor_mask", options),
  transform_timeout_(0, 0)
{
  const std::string data_dir = resolve_data_dir(
    declare_parameter<std::string>("data_dir", ""));
  std::filesystem::path corridor_path = declare_parameter<std::string>(
    "route_corridor_file", "map/route_corridor.json");
  if (corridor_path.empty()) {
    throw std::invalid_argument("route_corridor_file must not be empty");
  }
  if (corridor_path.is_relative()) {
    corridor_path = std::filesystem::path(data_dir) / corridor_path;
  }
  const std::string expected_global_path_sha256 =
    declare_parameter<std::string>(
    "route_corridor.expected_global_path_sha256", "");
  route_corridor_ = load_route_corridor(
    corridor_path, {{"global_path", expected_global_path_sha256}});
  prepared_route_corridor_.emplace(route_corridor_.corridor);

  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  transform_timeout_ = checked_timeout(
    declare_parameter<double>("transform_timeout_sec", 0.05));
  const int deduplication_cache_size = declare_parameter<int>(
    "deduplication_cache_size", 64);
  if (deduplication_cache_size <= 0 || deduplication_cache_size > 4096) {
    throw std::invalid_argument(
            "deduplication_cache_size must be in [1, 4096]");
  }
  deduplication_cache_size_ =
    static_cast<std::size_t>(deduplication_cache_size);
  window_ = RoadCorridorGridWindow{
    declare_parameter<double>("grid.minimum_x_m", -4.0),
    declare_parameter<double>("grid.maximum_x_m", 100.0),
    declare_parameter<double>("grid.minimum_y_m", -10.0),
    declare_parameter<double>("grid.maximum_y_m", 10.0),
    declare_parameter<double>("grid.resolution_m", 0.1)};
  require_required_window(window_);

  const std::string lidar_topic = declare_parameter<std::string>(
    "topics.lidar_points", "/ad/sensors/lidar/points");
  const std::string predicted_objects_topic =
    declare_parameter<std::string>(
    "topics.predicted_objects", "/ad/perception/objects/predicted");
  const std::string mask_topic = declare_parameter<std::string>(
    "topics.drivable_mask", "/ad/planning/drivable_mask");
  if (route_corridor_.corridor.frame_id.empty() || base_frame_.empty() ||
    lidar_topic.empty() || predicted_objects_topic.empty() ||
    mask_topic.empty())
  {
    throw std::invalid_argument(
            "route/base frames and mask trigger topics must not be empty");
  }
  const auto resolve_topic = [this](const std::string & topic) {
      return get_node_topics_interface()->resolve_topic_name(topic);
    };
  const std::string resolved_lidar = resolve_topic(lidar_topic);
  const std::string resolved_predictions = resolve_topic(
    predicted_objects_topic);
  const std::string resolved_mask = resolve_topic(mask_topic);
  if (resolved_lidar == resolved_predictions ||
    resolved_lidar == resolved_mask ||
    resolved_predictions == resolved_mask)
  {
    throw std::invalid_argument(
            "lidar, predicted-object, and mask topics must be distinct");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
    mask_topic,
    rclcpp::QoS(rclcpp::KeepLast(8)).reliable().durability_volatile());
  lidar_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    lidar_topic, rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
      on_trigger(message->header);
    });
  predicted_objects_subscription_ =
    create_subscription<ad_interfaces::msg::PredictedObjectArray>(
    predicted_objects_topic,
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile(),
    [this](
      ad_interfaces::msg::PredictedObjectArray::ConstSharedPtr message)
    {
      on_trigger(message->header);
    });
}

bool RoadCorridorMaskNode::already_published(
  const std::int64_t stamp_ns) const
{
  return std::find(
    published_stamps_ns_.begin(), published_stamps_ns_.end(),
    stamp_ns) != published_stamps_ns_.end();
}

void RoadCorridorMaskNode::remember_published(const std::int64_t stamp_ns)
{
  published_stamps_ns_.push_back(stamp_ns);
  while (published_stamps_ns_.size() > deduplication_cache_size_) {
    published_stamps_ns_.pop_front();
  }
  latest_published_stamp_ns_ = stamp_ns;
}

void RoadCorridorMaskNode::warn_drop(const std::string & reason)
{
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "road corridor mask trigger dropped: %s", reason.c_str());
}

void RoadCorridorMaskNode::on_trigger(const std_msgs::msg::Header & header)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  try {
    if (header.frame_id.empty()) {
      throw std::invalid_argument("trigger header frame must not be empty");
    }
    const std::int64_t stamp_ns = checked_stamp_ns(header.stamp);
    if (latest_published_stamp_ns_.has_value() &&
      stamp_ns < *latest_published_stamp_ns_ &&
      *latest_published_stamp_ns_ - stamp_ns > kNanosecondsPerSecond)
    {
      published_stamps_ns_.clear();
      latest_published_stamp_ns_.reset();
      RCLCPP_INFO(
        get_logger(),
        "ROS clock epoch rolled back; cleared mask stamp deduplication");
    }
    if (already_published(stamp_ns)) {
      return;
    }
    const rclcpp::Time exact_stamp(header.stamp, RCL_ROS_TIME);
    const auto route_from_base_message = tf_buffer_->lookupTransform(
      route_corridor_.corridor.frame_id, base_frame_,
      exact_stamp, transform_timeout_);
    if (!finite_translation(route_from_base_message.transform.translation)) {
      throw std::invalid_argument(
              "route from base translation must be finite");
    }
    const auto route_from_base_yaw = planar_yaw_from_quaternion(
      components(route_from_base_message.transform.rotation));
    if (!route_from_base_yaw) {
      throw std::invalid_argument(
              "route from base rotation must be finite, unit, and planar");
    }
    const Pose2 route_from_base{
      route_from_base_message.transform.translation.x,
      route_from_base_message.transform.translation.y,
      *route_from_base_yaw};
    const auto grid_template = make_route_aligned_grid_template(
      window_, route_corridor_.corridor.frame_id,
      header.stamp, route_from_base);
    const auto route_mask = rasterize_road_corridor(
      *prepared_route_corridor_, grid_template);
    publisher_->publish(
      express_road_corridor_mask_in_base_frame(
        route_mask, window_, base_frame_));
    remember_published(stamp_ns);
  } catch (const tf2::TransformException & error) {
    warn_drop(std::string("exact-stamp transform unavailable: ") + error.what());
  } catch (const std::exception & error) {
    warn_drop(error.what());
  } catch (...) {
    warn_drop("unknown processing error");
  }
}

}  // namespace ad_planner
