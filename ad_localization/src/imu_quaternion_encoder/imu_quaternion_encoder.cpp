#include "ad_localization/imu_quaternion_encoder/imu_quaternion_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ad_localization
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::size_t kMaximumImuHistorySamples = 512U;

std::optional<std::int64_t> valid_stamp_ns(
  const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

bool finite(double value)
{
  return std::isfinite(value);
}

geometry_msgs::msg::Quaternion quaternion_from_rpy(
  double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  geometry_msgs::msg::Quaternion result;
  result.x = sr * cp * cy - cr * sp * sy;
  result.y = cr * sp * cy + sr * cp * sy;
  result.z = cr * cp * sy - sr * sp * cy;
  result.w = cr * cp * cy + sr * sp * sy;
  return result;
}

bool normalize_quaternion(geometry_msgs::msg::Quaternion & quaternion)
{
  if (!finite(quaternion.x) || !finite(quaternion.y) || !finite(quaternion.z) ||
    !finite(quaternion.w))
  {
    return false;
  }
  const double norm = std::sqrt(
    quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w);
  if (!finite(norm) || norm < 1.0e-6) {
    return false;
  }
  quaternion.x /= norm;
  quaternion.y /= norm;
  quaternion.z /= norm;
  quaternion.w /= norm;
  return true;
}

geometry_msgs::msg::Quaternion inverse(
  const geometry_msgs::msg::Quaternion & quaternion)
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

std::array<double, 3> rotate(
  const geometry_msgs::msg::Quaternion & q,
  const std::array<double, 3> & vector)
{
  const std::array<double, 3> u{q.x, q.y, q.z};
  const double s = q.w;
  const double dot = u[0] * vector[0] + u[1] * vector[1] + u[2] * vector[2];
  const std::array<double, 3> cross{
    u[1] * vector[2] - u[2] * vector[1],
    u[2] * vector[0] - u[0] * vector[2],
    u[0] * vector[1] - u[1] * vector[0]};
  const double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  return {
    2.0 * dot * u[0] + (s * s - uu) * vector[0] + 2.0 * s * cross[0],
    2.0 * dot * u[1] + (s * s - uu) * vector[1] + 2.0 * s * cross[1],
    2.0 * dot * u[2] + (s * s - uu) * vector[2] + 2.0 * s * cross[2]};
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

void fill_covariance(
  nav_msgs::msg::Odometry & output,
  const ImuQuaternionEncoderConfig & config)
{
  output.pose.covariance.fill(0.0);
  output.twist.covariance.fill(0.0);
  output.pose.covariance[0] = config.position_variance_m2;
  output.pose.covariance[7] = config.position_variance_m2;
  output.pose.covariance[14] = config.position_variance_m2;
  output.pose.covariance[21] = config.orientation_variance_rad2;
  output.pose.covariance[28] = config.orientation_variance_rad2;
  output.pose.covariance[35] = config.orientation_variance_rad2;
  output.twist.covariance[0] = config.speed_variance_m2ps2;
  output.twist.covariance[7] = config.speed_variance_m2ps2;
  output.twist.covariance[14] = config.speed_variance_m2ps2;
}

}  // namespace

ImuQuaternionEncoderMode parse_imu_quaternion_encoder_mode(
  const std::string & value)
{
  if (value == "status_pose") {
    return ImuQuaternionEncoderMode::kStatusPose;
  }
  if (value == "dead_reckoning") {
    return ImuQuaternionEncoderMode::kDeadReckoning;
  }
  throw std::invalid_argument(
          "imu_quaternion_encoder mode must be status_pose or dead_reckoning");
}

ImuQuaternionEncoder::ImuQuaternionEncoder(ImuQuaternionEncoderConfig config)
: config_(std::move(config))
{
  const std::array<double, 15> values{
    config_.status_origin_to_base_m[0], config_.status_origin_to_base_m[1],
    config_.status_origin_to_base_m[2], config_.gnss_lever_arm_m[0],
    config_.gnss_lever_arm_m[1], config_.gnss_lever_arm_m[2],
    config_.world_yaw_offset_rad, config_.maximum_imu_age_sec,
    config_.maximum_integration_dt_sec, config_.automatic_reseed_distance_m,
    config_.automatic_reseed_candidate_radius_m,
    config_.automatic_reseed_max_interval_sec, config_.position_variance_m2,
    config_.orientation_variance_rad2, config_.speed_variance_m2ps2};
  if (!std::all_of(values.begin(), values.end(), finite) ||
    config_.maximum_imu_age_sec < 0.0 ||
    config_.maximum_integration_dt_sec <= 0.0 ||
    config_.initial_seed_sample_count <= 0 ||
    config_.automatic_reseed_distance_m <= 0.0 ||
    config_.automatic_reseed_confirmation_samples <= 0 ||
    config_.automatic_reseed_candidate_radius_m < 0.0 ||
    config_.automatic_reseed_max_interval_sec <= 0.0 ||
    config_.position_variance_m2 < 0.0 ||
    config_.orientation_variance_rad2 < 0.0 ||
    config_.speed_variance_m2ps2 < 0.0)
  {
    throw std::invalid_argument(
            "imu_quaternion_encoder numeric configuration is invalid");
  }
  if (!normalize_quaternion(config_.base_to_imu_orientation)) {
    throw std::invalid_argument(
            "imu_quaternion_encoder IMU mount orientation is invalid");
  }
}

void ImuQuaternionEncoder::reset() noexcept
{
  imu_history_.clear();
  latest_imu_stamp_ns_.reset();
  latest_seed_stamp_ns_.reset();
  last_status_stamp_ns_.reset();
  integration_stamp_ns_.reset();
  pending_seed_.reset();
  dead_reckoning_position_.reset();
  clear_initial_seed_accumulator();
  clear_automatic_reseed_candidate();
}

std::optional<sensor_msgs::msg::Imu> ImuQuaternionEncoder::causal_imu_at(
  std::int64_t target_stamp_ns) const
{
  for (auto iterator = imu_history_.rbegin(); iterator != imu_history_.rend(); ++iterator) {
    const auto imu_stamp_ns = valid_stamp_ns(iterator->header.stamp);
    if (!imu_stamp_ns || *imu_stamp_ns > target_stamp_ns) {
      continue;
    }
    const double age_sec = static_cast<double>(target_stamp_ns - *imu_stamp_ns) * 1.0e-9;
    if (age_sec <= config_.maximum_imu_age_sec) {
      return *iterator;
    }
    break;
  }
  return std::nullopt;
}

std::optional<geometry_msgs::msg::Quaternion>
ImuQuaternionEncoder::world_base_orientation(
  const sensor_msgs::msg::Imu & imu) const
{
  auto orientation = multiply(
    yaw_quaternion(config_.world_yaw_offset_rad),
    multiply(imu.orientation, inverse(config_.base_to_imu_orientation)));
  if (!normalize_quaternion(orientation)) {
    return std::nullopt;
  }
  return orientation;
}

void ImuQuaternionEncoder::clear_initial_seed_accumulator() noexcept
{
  initial_seed_sum_ = {0.0, 0.0, 0.0};
  initial_seed_samples_ = 0U;
}

bool ImuQuaternionEncoder::accumulate_initial_seed(
  const geometry_msgs::msg::PoseStamped & seed)
{
  initial_seed_sum_[0] += seed.pose.position.x;
  initial_seed_sum_[1] += seed.pose.position.y;
  initial_seed_sum_[2] += seed.pose.position.z;
  ++initial_seed_samples_;
  if (initial_seed_samples_ < static_cast<std::size_t>(
      config_.initial_seed_sample_count))
  {
    return false;
  }

  auto averaged_seed = seed;
  const double sample_count = static_cast<double>(initial_seed_samples_);
  averaged_seed.pose.position.x = initial_seed_sum_[0] / sample_count;
  averaged_seed.pose.position.y = initial_seed_sum_[1] / sample_count;
  averaged_seed.pose.position.z = initial_seed_sum_[2] / sample_count;
  pending_seed_ = std::move(averaged_seed);
  clear_initial_seed_accumulator();
  try_initialize_pending_seed();
  return true;
}

void ImuQuaternionEncoder::clear_automatic_reseed_candidate() noexcept
{
  automatic_reseed_candidate_.reset();
  automatic_reseed_candidate_sum_ = {0.0, 0.0, 0.0};
  automatic_reseed_candidate_samples_ = 0U;
}

bool ImuQuaternionEncoder::consider_automatic_reseed(
  const geometry_msgs::msg::PoseStamped & seed,
  std::int64_t seed_stamp_ns)
{
  if (!config_.automatic_reseed_enabled || !dead_reckoning_position_) {
    clear_automatic_reseed_candidate();
    return false;
  }

  const auto causal_imu = causal_imu_at(seed_stamp_ns);
  if (!causal_imu) {
    return false;
  }
  const auto orientation = world_base_orientation(*causal_imu);
  if (!orientation) {
    return false;
  }
  const auto world_lever_arm = rotate(*orientation, config_.gnss_lever_arm_m);
  const double predicted_antenna_x =
    (*dead_reckoning_position_)[0] + world_lever_arm[0];
  const double predicted_antenna_y =
    (*dead_reckoning_position_)[1] + world_lever_arm[1];
  const double distance = std::hypot(
    seed.pose.position.x - predicted_antenna_x,
    seed.pose.position.y - predicted_antenna_y);
  if (distance < config_.automatic_reseed_distance_m) {
    clear_automatic_reseed_candidate();
    return false;
  }

  bool continues_candidate = false;
  if (automatic_reseed_candidate_) {
    const auto candidate_stamp_ns = valid_stamp_ns(
      automatic_reseed_candidate_->header.stamp);
    if (candidate_stamp_ns) {
      const double interval_sec = static_cast<double>(
        seed_stamp_ns - *candidate_stamp_ns) * 1.0e-9;
      const double candidate_distance = std::hypot(
        seed.pose.position.x - automatic_reseed_candidate_->pose.position.x,
        seed.pose.position.y - automatic_reseed_candidate_->pose.position.y);
      continues_candidate = interval_sec >= 0.0 &&
        interval_sec <= config_.automatic_reseed_max_interval_sec &&
        candidate_distance <= config_.automatic_reseed_candidate_radius_m;
    }
  }

  if (continues_candidate) {
    ++automatic_reseed_candidate_samples_;
    automatic_reseed_candidate_sum_[0] += seed.pose.position.x;
    automatic_reseed_candidate_sum_[1] += seed.pose.position.y;
    automatic_reseed_candidate_sum_[2] += seed.pose.position.z;
  } else {
    automatic_reseed_candidate_samples_ = 1U;
    automatic_reseed_candidate_sum_ = {
      seed.pose.position.x,
      seed.pose.position.y,
      seed.pose.position.z};
  }
  automatic_reseed_candidate_ = seed;
  if (automatic_reseed_candidate_samples_ < static_cast<std::size_t>(
      config_.automatic_reseed_confirmation_samples))
  {
    return false;
  }

  auto averaged_seed = seed;
  const double sample_count = static_cast<double>(
    automatic_reseed_candidate_samples_);
  averaged_seed.pose.position.x =
    automatic_reseed_candidate_sum_[0] / sample_count;
  averaged_seed.pose.position.y =
    automatic_reseed_candidate_sum_[1] / sample_count;
  averaged_seed.pose.position.z =
    automatic_reseed_candidate_sum_[2] / sample_count;
  pending_seed_ = std::move(averaged_seed);
  clear_automatic_reseed_candidate();
  try_initialize_pending_seed();
  return true;
}

void ImuQuaternionEncoder::try_initialize_pending_seed()
{
  if (!pending_seed_) {
    return;
  }
  const auto seed_stamp_ns = valid_stamp_ns(pending_seed_->header.stamp);
  if (!seed_stamp_ns ||
    (last_status_stamp_ns_ && *seed_stamp_ns <= *last_status_stamp_ns_))
  {
    pending_seed_.reset();
    return;
  }
  const auto causal_imu = causal_imu_at(*seed_stamp_ns);
  if (!causal_imu) {
    return;
  }
  const auto orientation = world_base_orientation(*causal_imu);
  if (!orientation) {
    return;
  }
  const auto world_lever_arm = rotate(*orientation, config_.gnss_lever_arm_m);
  dead_reckoning_position_ = std::array<double, 3>{
    pending_seed_->pose.position.x - world_lever_arm[0],
    pending_seed_->pose.position.y - world_lever_arm[1],
    pending_seed_->pose.position.z - world_lever_arm[2]};
  integration_stamp_ns_ = *seed_stamp_ns;
  pending_seed_.reset();
}

bool ImuQuaternionEncoder::observe_imu(const sensor_msgs::msg::Imu & imu)
{
  if (imu.header.frame_id != config_.imu_frame) {
    return false;
  }
  const auto sample_stamp_ns = valid_stamp_ns(imu.header.stamp);
  auto normalized = imu;
  if (!sample_stamp_ns ||
    (latest_imu_stamp_ns_ && *sample_stamp_ns <= *latest_imu_stamp_ns_) ||
    !normalize_quaternion(normalized.orientation))
  {
    return false;
  }
  imu_history_.push_back(std::move(normalized));
  if (imu_history_.size() > kMaximumImuHistorySamples) {
    imu_history_.pop_front();
  }
  latest_imu_stamp_ns_ = *sample_stamp_ns;
  try_initialize_pending_seed();
  return true;
}

bool ImuQuaternionEncoder::observe_gnss_seed(
  const geometry_msgs::msg::PoseStamped & seed)
{
  const auto sample_stamp_ns = valid_stamp_ns(seed.header.stamp);
  if (config_.mode != ImuQuaternionEncoderMode::kDeadReckoning ||
    seed.header.frame_id != config_.reference_frame || !sample_stamp_ns ||
    (latest_seed_stamp_ns_ && *sample_stamp_ns <= *latest_seed_stamp_ns_) ||
    (last_status_stamp_ns_ && *sample_stamp_ns <= *last_status_stamp_ns_) ||
    !finite(seed.pose.position.x) || !finite(seed.pose.position.y) ||
    !finite(seed.pose.position.z))
  {
    return false;
  }
  latest_seed_stamp_ns_ = *sample_stamp_ns;
  if (!dead_reckoning_position_ || !integration_stamp_ns_) {
    clear_automatic_reseed_candidate();
    if (pending_seed_) {
      try_initialize_pending_seed();
      return false;
    }
    return accumulate_initial_seed(seed);
  }
  return consider_automatic_reseed(seed, *sample_stamp_ns);
}

std::optional<nav_msgs::msg::Odometry> ImuQuaternionEncoder::observe_status(
  const ad_morai_interfaces::msg::EgoVehicleStatus & status)
{
  const auto sample_stamp_ns = valid_stamp_ns(status.header.stamp);
  if (status.header.frame_id != config_.status_frame || !sample_stamp_ns ||
    (last_status_stamp_ns_ && *sample_stamp_ns <= *last_status_stamp_ns_) ||
    !finite(status.signed_velocity))
  {
    return std::nullopt;
  }

  nav_msgs::msg::Odometry output;
  output.header.stamp = status.header.stamp;
  output.header.frame_id = config_.reference_frame;
  output.child_frame_id = config_.base_frame;
  output.twist.twist.linear.x = status.signed_velocity;
  fill_covariance(output, config_);

  if (config_.mode == ImuQuaternionEncoderMode::kStatusPose) {
    if (!finite(status.position.x) || !finite(status.position.y) ||
      !finite(status.position.z) || !finite(status.rpy.x) ||
      !finite(status.rpy.y) || !finite(status.rpy.z) ||
      (config_.reject_zero_status_position && status.position.x == 0.0 &&
      status.position.y == 0.0 && status.position.z == 0.0))
    {
      return std::nullopt;
    }
    output.pose.pose.orientation = quaternion_from_rpy(
      status.rpy.x, status.rpy.y, status.rpy.z);
    const auto offset = rotate(
      output.pose.pose.orientation, config_.status_origin_to_base_m);
    output.pose.pose.position.x = status.position.x + offset[0];
    output.pose.pose.position.y = status.position.y + offset[1];
    output.pose.pose.position.z = status.position.z + offset[2];
    output.twist.twist.angular = status.angular_velocity;
    last_status_stamp_ns_ = *sample_stamp_ns;
    return output;
  }

  try_initialize_pending_seed();
  if (!dead_reckoning_position_ || !integration_stamp_ns_) {
    return std::nullopt;
  }
  if (*sample_stamp_ns < *integration_stamp_ns_) {
    return std::nullopt;
  }
  const double dt = static_cast<double>(
    *sample_stamp_ns - *integration_stamp_ns_) * 1.0e-9;
  if (dt > config_.maximum_integration_dt_sec) {
    dead_reckoning_position_.reset();
    integration_stamp_ns_.reset();
    pending_seed_.reset();
    clear_initial_seed_accumulator();
    clear_automatic_reseed_candidate();
    last_status_stamp_ns_ = *sample_stamp_ns;
    try_initialize_pending_seed();
    return std::nullopt;
  }
  const auto causal_imu = causal_imu_at(*sample_stamp_ns);
  if (!causal_imu) {
    return std::nullopt;
  }
  const auto orientation = world_base_orientation(*causal_imu);
  if (!orientation) {
    return std::nullopt;
  }
  if (dt > 0.0) {
    const double yaw = yaw_from_quaternion(*orientation);
    (*dead_reckoning_position_)[0] += status.signed_velocity * std::cos(yaw) * dt;
    (*dead_reckoning_position_)[1] += status.signed_velocity * std::sin(yaw) * dt;
  }
  output.pose.pose.position.x = (*dead_reckoning_position_)[0];
  output.pose.pose.position.y = (*dead_reckoning_position_)[1];
  output.pose.pose.position.z = (*dead_reckoning_position_)[2];
  output.pose.pose.orientation = *orientation;
  output.twist.twist.angular = causal_imu->angular_velocity;
  integration_stamp_ns_ = *sample_stamp_ns;
  last_status_stamp_ns_ = *sample_stamp_ns;
  try_initialize_pending_seed();
  return output;
}

}  // namespace ad_localization
