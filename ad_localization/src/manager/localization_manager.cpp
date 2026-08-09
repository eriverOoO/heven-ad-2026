#include "ad_localization/manager/localization_manager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ad_localization
{
namespace
{

bool valid_frame(const std::string & frame)
{
  return !frame.empty() && frame.front() != '/' && frame.find(' ') == std::string::npos;
}

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

std::optional<std::int64_t> valid_stamp_ns(
  const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

bool finite_pose_and_twist(const nav_msgs::msg::Odometry & candidate)
{
  const auto & position = candidate.pose.pose.position;
  const auto & orientation = candidate.pose.pose.orientation;
  const auto & linear = candidate.twist.twist.linear;
  const auto & angular = candidate.twist.twist.angular;
  const std::array<double, 13> values{
    position.x, position.y, position.z,
    orientation.x, orientation.y, orientation.z, orientation.w,
    linear.x, linear.y, linear.z, angular.x, angular.y, angular.z};
  if (!std::all_of(
      values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      }) ||
    !std::all_of(
      candidate.pose.covariance.begin(), candidate.pose.covariance.end(),
      [](double value) {return std::isfinite(value);}) ||
    !std::all_of(
      candidate.twist.covariance.begin(), candidate.twist.covariance.end(),
      [](double value) {return std::isfinite(value);}))
  {
    return false;
  }
  const double norm_squared =
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w;
  return std::abs(norm_squared - 1.0) <= 1.0e-3;
}

}  // namespace

LocalizationManager::LocalizationManager(LocalizationManagerConfig config)
: config_(std::move(config))
{
  if (!valid_frame(config_.map_frame) || !valid_frame(config_.odom_frame) ||
    !valid_frame(config_.base_frame))
  {
    throw std::invalid_argument("localization manager frames must be valid relative TF frames");
  }
  if (config_.map_frame == config_.odom_frame ||
    config_.map_frame == config_.base_frame ||
    config_.odom_frame == config_.base_frame)
  {
    throw std::invalid_argument("localization manager frames must be distinct");
  }
}

std::optional<nav_msgs::msg::Odometry> LocalizationManager::accept(
  const nav_msgs::msg::Odometry & candidate)
{
  if (candidate.header.frame_id != config_.odom_frame ||
    candidate.child_frame_id != config_.base_frame ||
    !finite_pose_and_twist(candidate))
  {
    return std::nullopt;
  }
  const auto candidate_stamp_ns = valid_stamp_ns(candidate.header.stamp);
  if (!candidate_stamp_ns ||
    (last_stamp_ns_ && *candidate_stamp_ns <= *last_stamp_ns_))
  {
    return std::nullopt;
  }
  last_stamp_ns_ = *candidate_stamp_ns;
  return candidate;
}

void LocalizationManager::reset() noexcept
{
  last_stamp_ns_.reset();
}

geometry_msgs::msg::TransformStamped odometry_transform(
  const nav_msgs::msg::Odometry & odometry)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header = odometry.header;
  transform.child_frame_id = odometry.child_frame_id;
  transform.transform.translation.x = odometry.pose.pose.position.x;
  transform.transform.translation.y = odometry.pose.pose.position.y;
  transform.transform.translation.z = odometry.pose.pose.position.z;
  transform.transform.rotation = odometry.pose.pose.orientation;
  return transform;
}

geometry_msgs::msg::TransformStamped map_to_odom_transform(
  const LocalizationManagerConfig & config,
  const builtin_interfaces::msg::Time & stamp)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = config.map_frame;
  transform.child_frame_id = config.odom_frame;
  transform.transform.rotation.w = 1.0;
  return transform;
}

}  // namespace ad_localization
