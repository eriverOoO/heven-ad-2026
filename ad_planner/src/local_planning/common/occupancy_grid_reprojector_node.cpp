#include "occupancy_grid_reprojector_node.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>

namespace ad_planner
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;

std::optional<std::int64_t> valid_stamp_ns(
  const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
    (stamp.sec == 0 && stamp.nanosec == 0))
  {
    return std::nullopt;
  }
  const std::int64_t seconds = static_cast<std::int64_t>(stamp.sec);
  if (seconds >
    (std::numeric_limits<std::int64_t>::max() -
    static_cast<std::int64_t>(stamp.nanosec)) / kNanosecondsPerSecond)
  {
    return std::nullopt;
  }
  return seconds * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

QuaternionComponents components(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  return QuaternionComponents{
    quaternion.x, quaternion.y, quaternion.z, quaternion.w};
}

bool finite_translation(
  const geometry_msgs::msg::Vector3 & translation)
{
  return std::isfinite(translation.x) &&
         std::isfinite(translation.y) &&
         std::isfinite(translation.z);
}

rclcpp::Duration checked_timeout(const double timeout_sec)
{
  if (!std::isfinite(timeout_sec) || timeout_sec < 0.0) {
    throw std::invalid_argument(
            "transform_timeout_sec must be finite and nonnegative");
  }
  const long double nanoseconds =
    static_cast<long double>(timeout_sec) *
    static_cast<long double>(kNanosecondsPerSecond);
  if (!std::isfinite(nanoseconds) ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()) ||
    (timeout_sec > 0.0 && nanoseconds < 1.0L))
  {
    throw std::invalid_argument(
            "transform_timeout_sec is not representable as an rclcpp duration");
  }
  return rclcpp::Duration::from_nanoseconds(
    static_cast<std::int64_t>(nanoseconds));
}

}  // namespace

OccupancyGridReprojectorNode::OccupancyGridReprojectorNode(
  const rclcpp::NodeOptions & options)
: Node("ad_occupancy_grid_reprojector", options),
  transform_timeout_(0, 0)
{
  const std::string input_topic = declare_parameter<std::string>(
    "input_topic", "/ad/perception/occupancy_grid");
  const std::string output_topic = declare_parameter<std::string>(
    "output_topic", "/ad/planner/mppi/occupancy_grid_odom");
  target_frame_ = declare_parameter<std::string>("target_frame", "odom");
  ego_frame_ = declare_parameter<std::string>("ego_frame", "base_link");
  config_.width_m = declare_parameter<double>("width_m", 54.0);
  config_.height_m = declare_parameter<double>("height_m", 54.0);
  config_.resolution_m = declare_parameter<double>("resolution_m", 0.1);
  const std::int64_t outside_value = declare_parameter<std::int64_t>(
    "outside_value", -1);
  if (outside_value != -1) {
    throw std::invalid_argument("outside_value must be exactly -1");
  }
  config_.outside_value = static_cast<std::int8_t>(outside_value);
  transform_timeout_ = checked_timeout(
    declare_parameter<double>("transform_timeout_sec", 0.05));
  if (input_topic.empty() || output_topic.empty() ||
    target_frame_.empty() || ego_frame_.empty())
  {
    throw std::invalid_argument("topics and frame names must not be empty");
  }
  if (!std::isfinite(config_.width_m) || config_.width_m <= 0.0 ||
    !std::isfinite(config_.height_m) || config_.height_m <= 0.0 ||
    !std::isfinite(config_.resolution_m) || config_.resolution_m <= 0.0)
  {
    throw std::invalid_argument(
            "grid width, height, and resolution must be finite and positive");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
    output_topic,
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile());
  subscription_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
    input_topic, rclcpp::SensorDataQoS(),
    std::bind(
      &OccupancyGridReprojectorNode::on_grid, this, std::placeholders::_1));
}

void OccupancyGridReprojectorNode::warn_drop(const std::string & reason)
{
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "dropping occupancy grid: %s", reason.c_str());
}

void OccupancyGridReprojectorNode::on_grid(
  const nav_msgs::msg::OccupancyGrid::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  try {
    const auto stamp_ns = valid_stamp_ns(message->header.stamp);
    if (!stamp_ns) {
      warn_drop("stamp must be strictly positive and representable");
      return;
    }
    if (last_successfully_published_stamp_ns_ &&
      *stamp_ns <= *last_successfully_published_stamp_ns_)
    {
      return;
    }
    if (message->header.frame_id.empty()) {
      warn_drop("input frame is empty");
      return;
    }
    if (!std::isfinite(message->info.origin.position.x) ||
      !std::isfinite(message->info.origin.position.y) ||
      !std::isfinite(message->info.origin.position.z))
    {
      warn_drop("source origin translation is not finite");
      return;
    }
    const auto source_origin_yaw = planar_yaw_from_quaternion(
      components(message->info.origin.orientation));
    if (!source_origin_yaw) {
      warn_drop("source origin quaternion is not finite, near-unit, and planar");
      return;
    }

    const rclcpp::Time exact_stamp(message->header.stamp, RCL_ROS_TIME);
    const auto odom_from_source_message = tf_buffer_->lookupTransform(
      target_frame_, message->header.frame_id, exact_stamp, transform_timeout_);
    const auto odom_from_ego_message = tf_buffer_->lookupTransform(
      target_frame_, ego_frame_, exact_stamp, transform_timeout_);
    if (!finite_translation(odom_from_source_message.transform.translation)) {
      warn_drop("odom from source translation is not finite");
      return;
    }
    const auto transform_yaw = planar_yaw_from_quaternion(
      components(odom_from_source_message.transform.rotation));
    if (!transform_yaw) {
      warn_drop("odom from source quaternion is not finite, near-unit, and planar");
      return;
    }
    if (!finite_translation(odom_from_ego_message.transform.translation)) {
      warn_drop("odom from ego translation is not finite");
      return;
    }

    OccupancyGrid source;
    source.origin = Pose2{
      message->info.origin.position.x,
      message->info.origin.position.y,
      *source_origin_yaw};
    source.resolution = static_cast<double>(message->info.resolution);
    source.width = static_cast<std::size_t>(message->info.width);
    source.height = static_cast<std::size_t>(message->info.height);
    source.cells = message->data;
    source.valid = true;
    source.fresh = true;
    const FrameTransform2 odom_from_source{
      odom_from_source_message.transform.translation.x,
      odom_from_source_message.transform.translation.y,
      *transform_yaw};
    const Point2 ego_in_odom{
      odom_from_ego_message.transform.translation.x,
      odom_from_ego_message.transform.translation.y};
    const auto reprojected = reproject_occupancy_grid(
      source, odom_from_source, ego_in_odom, config_);
    if (!reprojected) {
      warn_drop("source grid or reprojection geometry is invalid");
      return;
    }

    nav_msgs::msg::OccupancyGrid output;
    output.header.stamp = message->header.stamp;
    output.header.frame_id = target_frame_;
    output.info.map_load_time = message->header.stamp;
    output.info.resolution = static_cast<float>(reprojected->resolution);
    output.info.width = static_cast<std::uint32_t>(reprojected->width);
    output.info.height = static_cast<std::uint32_t>(reprojected->height);
    output.info.origin.position.x = reprojected->origin.x;
    output.info.origin.position.y = reprojected->origin.y;
    output.info.origin.position.z = 0.0;
    output.info.origin.orientation.x = 0.0;
    output.info.origin.orientation.y = 0.0;
    output.info.origin.orientation.z = 0.0;
    output.info.origin.orientation.w = 1.0;
    output.data = reprojected->cells;
    publisher_->publish(output);
    last_successfully_published_stamp_ns_ = *stamp_ns;
  } catch (const tf2::TransformException & error) {
    warn_drop(std::string("exact-stamp transform unavailable: ") + error.what());
  } catch (const std::exception & error) {
    warn_drop(std::string("processing failed: ") + error.what());
  } catch (...) {
    warn_drop("processing failed with an unknown exception");
  }
}

}  // namespace ad_planner
