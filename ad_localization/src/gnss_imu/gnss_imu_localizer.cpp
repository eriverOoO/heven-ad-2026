#include "ad_localization/gnss_imu/gnss_imu_localizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ad_localization
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;

std::optional<std::int64_t> valid_stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
    (stamp.sec == 0 && stamp.nanosec == 0))
  {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond + stamp.nanosec;
}

bool valid_frame(const std::string & frame)
{
  return !frame.empty() && frame.front() != '/' && frame.find("//") == std::string::npos &&
         frame.find_first_of(" \t\r\n") == std::string::npos;
}

std::optional<geometry_msgs::msg::Quaternion> normalized_quaternion(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  if (!std::isfinite(quaternion.x) || !std::isfinite(quaternion.y) ||
    !std::isfinite(quaternion.z) || !std::isfinite(quaternion.w))
  {
    return std::nullopt;
  }
  const double norm_squared = quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w;
  if (!std::isfinite(norm_squared) || norm_squared < 1.0e-12) {
    return std::nullopt;
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  geometry_msgs::msg::Quaternion result;
  result.x = quaternion.x * inverse_norm;
  result.y = quaternion.y * inverse_norm;
  result.z = quaternion.z * inverse_norm;
  result.w = quaternion.w * inverse_norm;
  return result;
}

geometry_msgs::msg::Quaternion inverse(const geometry_msgs::msg::Quaternion & quaternion)
{
  geometry_msgs::msg::Quaternion result;
  result.x = -quaternion.x;
  result.y = -quaternion.y;
  result.z = -quaternion.z;
  result.w = quaternion.w;
  return result;
}

geometry_msgs::msg::Quaternion multiply(
  const geometry_msgs::msg::Quaternion & lhs,
  const geometry_msgs::msg::Quaternion & rhs)
{
  geometry_msgs::msg::Quaternion result;
  result.x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y;
  result.y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x;
  result.z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w;
  result.w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z;
  return result;
}

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion result;
  result.z = std::sin(yaw * 0.5);
  result.w = std::cos(yaw * 0.5);
  return result;
}

std::array<double, 3> rotate_vector(
  const geometry_msgs::msg::Quaternion & quaternion,
  const std::array<double, 3> & vector)
{
  const double tx = 2.0 * (quaternion.y * vector[2] - quaternion.z * vector[1]);
  const double ty = 2.0 * (quaternion.z * vector[0] - quaternion.x * vector[2]);
  const double tz = 2.0 * (quaternion.x * vector[1] - quaternion.y * vector[0]);
  return {
    vector[0] + quaternion.w * tx + (quaternion.y * tz - quaternion.z * ty),
    vector[1] + quaternion.w * ty + (quaternion.z * tx - quaternion.x * tz),
    vector[2] + quaternion.w * tz + (quaternion.x * ty - quaternion.y * tx),
  };
}

bool finite_position(const geometry_msgs::msg::Point & position)
{
  return std::isfinite(position.x) && std::isfinite(position.y) &&
         std::isfinite(position.z);
}

bool synchronized(std::int64_t lhs, std::int64_t rhs, double tolerance_sec)
{
  const double difference_sec =
    std::abs(static_cast<double>(lhs - rhs)) / static_cast<double>(kNanosecondsPerSecond);
  return std::isfinite(difference_sec) && difference_sec <= tolerance_sec;
}

void validate_config(GnssImuLocalizerConfig & config)
{
  if (!valid_frame(config.reference_frame) || !valid_frame(config.base_frame) ||
    !valid_frame(config.imu_frame) || config.reference_frame == config.base_frame)
  {
    throw std::invalid_argument("localizer frames must be valid and reference/base must differ");
  }
  if (!std::isfinite(config.synchronization_tolerance_sec) ||
    config.synchronization_tolerance_sec < 0.0 ||
    !std::isfinite(config.gnss_xy_variance_m2) || config.gnss_xy_variance_m2 <= 0.0 ||
    !std::isfinite(config.gnss_z_variance_m2) || config.gnss_z_variance_m2 <= 0.0 ||
    !std::isfinite(config.imu_orientation_variance_rad2) ||
    config.imu_orientation_variance_rad2 <= 0.0 ||
    !std::isfinite(config.unobserved_twist_variance) ||
    config.unobserved_twist_variance <= 0.0 ||
    !std::isfinite(config.world_yaw_offset_rad) ||
    !std::all_of(
      config.gnss_lever_arm_m.begin(), config.gnss_lever_arm_m.end(),
      [](double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument(
            "localizer synchronization, covariance, world-yaw offset, and lever-arm "
            "values must be finite and valid");
  }
  const auto orientation = normalized_quaternion(config.base_to_imu_orientation);
  if (!orientation) {
    throw std::invalid_argument("IMU mount orientation must be a valid quaternion");
  }
  config.base_to_imu_orientation = *orientation;
}

}  // namespace

GnssImuLocalizer::GnssImuLocalizer(GnssImuLocalizerConfig config)
: config_(std::move(config))
{
  validate_config(config_);
}

void GnssImuLocalizer::reset()
{
  latest_imu_.reset();
  latest_wheel_speed_.reset();
  last_imu_stamp_ns_.reset();
  last_wheel_stamp_ns_.reset();
  last_output_stamp_ns_.reset();
}

bool GnssImuLocalizer::observe_imu(const sensor_msgs::msg::Imu & imu)
{
  const auto stamp_ns = valid_stamp_ns(imu.header.stamp);
  if (!stamp_ns || imu.header.frame_id != config_.imu_frame ||
    !normalized_quaternion(imu.orientation) ||
    (last_imu_stamp_ns_ && *stamp_ns <= *last_imu_stamp_ns_))
  {
    return false;
  }
  latest_imu_ = imu;
  last_imu_stamp_ns_ = stamp_ns;
  return true;
}

bool GnssImuLocalizer::observe_wheel_speed(
  const geometry_msgs::msg::TwistWithCovarianceStamped & wheel_speed)
{
  const auto stamp_ns = valid_stamp_ns(wheel_speed.header.stamp);
  if (!stamp_ns || wheel_speed.header.frame_id != config_.base_frame ||
    !std::isfinite(wheel_speed.twist.twist.linear.x) ||
    !std::isfinite(wheel_speed.twist.covariance[0]) ||
    wheel_speed.twist.covariance[0] <= 0.0 ||
    (last_wheel_stamp_ns_ && *stamp_ns <= *last_wheel_stamp_ns_))
  {
    return false;
  }
  latest_wheel_speed_ = wheel_speed;
  last_wheel_stamp_ns_ = stamp_ns;
  return true;
}

std::optional<nav_msgs::msg::Odometry> GnssImuLocalizer::observe_gnss(
  const geometry_msgs::msg::PoseStamped & antenna_pose)
{
  const auto gnss_stamp_ns = valid_stamp_ns(antenna_pose.header.stamp);
  if (!gnss_stamp_ns || antenna_pose.header.frame_id != config_.reference_frame ||
    !finite_position(antenna_pose.pose.position) ||
    (last_output_stamp_ns_ && *gnss_stamp_ns <= *last_output_stamp_ns_) ||
    !latest_imu_)
  {
    return std::nullopt;
  }

  const auto imu_stamp_ns = valid_stamp_ns(latest_imu_->header.stamp);
  const auto world_imu = normalized_quaternion(latest_imu_->orientation);
  if (!imu_stamp_ns || !world_imu ||
    !synchronized(*gnss_stamp_ns, *imu_stamp_ns, config_.synchronization_tolerance_sec))
  {
    return std::nullopt;
  }

  const auto world_base = normalized_quaternion(multiply(
      yaw_quaternion(config_.world_yaw_offset_rad),
      multiply(*world_imu, inverse(config_.base_to_imu_orientation))));
  if (!world_base) {
    return std::nullopt;
  }
  const auto world_lever_arm = rotate_vector(*world_base, config_.gnss_lever_arm_m);

  nav_msgs::msg::Odometry odometry;
  odometry.header = antenna_pose.header;
  odometry.child_frame_id = config_.base_frame;
  odometry.pose.pose.position.x = antenna_pose.pose.position.x - world_lever_arm[0];
  odometry.pose.pose.position.y = antenna_pose.pose.position.y - world_lever_arm[1];
  odometry.pose.pose.position.z = antenna_pose.pose.position.z - world_lever_arm[2];
  odometry.pose.pose.orientation = *world_base;
  const std::array<double, 6> pose_variances{
    config_.gnss_xy_variance_m2,
    config_.gnss_xy_variance_m2,
    config_.gnss_z_variance_m2,
    config_.imu_orientation_variance_rad2,
    config_.imu_orientation_variance_rad2,
    config_.imu_orientation_variance_rad2};
  for (std::size_t axis = 0; axis < pose_variances.size(); ++axis) {
    odometry.pose.covariance[axis * 6 + axis] = pose_variances[axis];
    odometry.twist.covariance[axis * 6 + axis] =
      config_.unobserved_twist_variance;
  }
  if (latest_wheel_speed_) {
    const auto wheel_stamp_ns = valid_stamp_ns(latest_wheel_speed_->header.stamp);
    if (wheel_stamp_ns && synchronized(
        *gnss_stamp_ns, *wheel_stamp_ns, config_.synchronization_tolerance_sec))
    {
      odometry.twist.twist.linear.x = latest_wheel_speed_->twist.twist.linear.x;
      odometry.twist.covariance[0] = latest_wheel_speed_->twist.covariance[0];
    }
  }
  last_output_stamp_ns_ = gnss_stamp_ns;
  return odometry;
}

}  // namespace ad_localization
